""" Implementation of the Anthropic models.

Feature table:
    - Async chat:       YES (1)
    - Return JSON:      YES (2)
    - Structured types: YES (Claude 4.x only)
    - Token count:      YES
    - Image support:    YES
    - Tool use:         YES

Models:
Claude 3 Opus:   claude-3-opus-20240229
Claude 3 Sonnet: claude-3-5-sonnet-20240620
Claude 3 Haiku:  claude-3-haiku-20240307
Claude 4 Sonnet: claude-sonnet-4-5-*
Claude 4 Opus:   claude-opus-4-*
Claude 4 Haiku:  claude-haiku-4-*

Supported parameters:
max_tokens: int (default 800)
temperature: float (default 0.8)

(1) In contrast to Model.chat, Model.chat_async cannot return json and does not return input and output token counts
(2) Claude 4.x uses structured outputs beta for guaranteed JSON. Claude 3.x uses legacy parsing with deprecation warning.

"""

import json
import logging
import os
import re
from typing import Any, AsyncGenerator

import httpx
from anthropic import Anthropic, AsyncAnthropic, APIConnectionError, AuthenticationError, PermissionDeniedError, \
    APITimeoutError, RateLimitError, BadRequestError, InternalServerError
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Models that support structured outputs (beta)
# See: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
STRUCTURED_OUTPUT_MODELS = re.compile(r'claude-(sonnet-4|opus-4|haiku-4)')
STRUCTURED_OUTPUTS_BETA = 'structured-outputs-2025-11-13'


from justai.model.message import Message
from justai.models.basemodel import (
    BaseModel,
    DEFAULT_TIMEOUT,
    identify_image_format_from_base64,
    ConnectionException,
    AuthorizationException,
    ModelOverloadException,
    RatelimitException,
    BadRequestException,
    GeneralException,
    ImageInput,
)
from justai.tools.display import ERROR_COLOR, color_print
from justai.tools.images import to_base64_image


class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by Anthropic."
        super().__init__(model_name, params, system_message)
        self.cached_prompt = None

        # Authentication
        if "ANTHROPIC_API_KEY" in params:
            api_key = params["ANTHROPIC_API_KEY"]
            del params["ANTHROPIC_API_KEY"]
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY") or dotenv_values()["ANTHROPIC_API_KEY"]
        if not api_key:
            color_print(
                "No Anthropic API key found. Create one at https://console.anthropic.com/settings/keys and "
                + "set it in the .env file like ANTHROPIC_API_KEY=here_comes_your_key.",
                color=ERROR_COLOR,
            )

        # Client
        timeout = httpx.Timeout(params.get('timeout', DEFAULT_TIMEOUT))
        if params.get('async'):
            http_client = httpx.AsyncClient(timeout=timeout)
            self.client = AsyncAnthropic(api_key=api_key, http_client=http_client)
        else:
            http_client = httpx.Client(timeout=timeout)
            self.client = Anthropic(api_key=api_key, http_client=http_client)

        # Required model parameters
        if "max_tokens" not in params:
            params["max_tokens"] = 800

        self.supports_cached_prompts = True
        self.messages = []

    def prompt(self, prompt: str, images: ImageInput = None, tools: list = None, return_json: bool = False, response_format=None) \
            -> tuple[Any, int|None, int|None, dict|None]:
        # Reset messages
        self.messages = []
        return self.chat(prompt, images, tools, return_json, response_format)


    def chat(self, prompt: str, images: ImageInput = None, tools: list = None, return_json: bool = False, response_format=None) \
            -> tuple[Any, int|None, int|None, dict|None]:
        # Check if we should use structured outputs (beta) for newer models
        use_structured_outputs = return_json and self._supports_structured_outputs()

        if use_structured_outputs:
            try:
                message = self._completion_with_structured_output(prompt, images, tools, response_format)
            except TypeError as e:
                # Fall back to legacy path if SDK doesn't support structured outputs
                # This happens with older anthropic SDK versions
                logger.warning(
                    f'Structured outputs not available (SDK may be outdated): {e}. '
                    'Falling back to legacy JSON parsing. '
                    'Upgrade anthropic SDK with: pip install --upgrade anthropic'
                )
                use_structured_outputs = False
                message = self.completion(prompt, images, tools, return_json, response_format)
        else:
            message = self.completion(prompt, images, tools, return_json, response_format)

        # Text content
        response_str = message.content[0].text
        if return_json:
            if use_structured_outputs:
                # Structured outputs guarantees valid JSON
                response = json.loads(response_str)
            else:
                # Legacy JSON parsing for older models or fallback
                response = self._parse_json_legacy(response_str)
        else:
            response = response_str

        # Token count
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        if self.cached_prompt:
            self.cache_creation_input_tokens = message.usage.cache_creation_input_tokens
            self.cache_read_input_tokens = message.usage.cache_read_input_tokens
        else:
            self.cache_creation_input_tokens = self.cache_read_input_tokens = 0

        return response, input_tokens, output_tokens

    def _supports_structured_outputs(self) -> bool:
        """Check if the model supports structured outputs (Claude 4.x models)."""
        return bool(STRUCTURED_OUTPUT_MODELS.search(self.model_name))

    def _parse_json_legacy(self, response_str: str) -> dict | list:
        """
        Legacy JSON parsing for older models that don't support structured outputs.
        TODO: Remove when Claude 3.x models are deprecated.
        """
        response_str = response_str.split('</json>')[0]
        if response_str.startswith('```json'):
            response_str = response_str[7:-3]
        try:
            return json.loads(response_str, strict=False)
        except json.decoder.JSONDecodeError:
            logger.error(f'Error decoding JSON: {response_str[:200]}')
            try:
                return extract_dict(response_str)
            except json.decoder.JSONDecodeError as e:
                raise ValueError(f'Expected JSON but got: {response_str[:200]}') from e

    def _completion_with_structured_output(self, prompt: str, images: ImageInput = None,
                                           tools: list = None, response_format=None):
        """Use the structured outputs beta API for guaranteed JSON responses."""
        system_message = self.cached_system_message() if self.cached_prompt else self.system_message

        # Add user message to conversation history
        if prompt or images:
            user_message = create_anthropic_message('user', prompt, images)
            self.messages.append(user_message)

        # Build output_format schema
        if response_format and hasattr(response_format, 'model_json_schema'):
            # Pydantic model - extract JSON schema
            schema = response_format.model_json_schema()
        elif response_format and isinstance(response_format, dict):
            # Already a JSON schema dict
            schema = response_format
        else:
            # Default: accept any JSON object
            schema = {'type': 'object'}

        # Ensure additionalProperties is set to false (required by Anthropic)
        if schema.get('type') == 'object' and 'additionalProperties' not in schema:
            schema['additionalProperties'] = False

        output_format_param = {
            'type': 'json_schema',
            'schema': schema
        }

        antr_tools = transform_tools(tools or []) if tools is not None else None

        try:
            api_params = {
                'model': self.model_name,
                'messages': self.messages,
                'system': system_message,
                **self.model_params
            }

            if antr_tools:
                api_params['tools'] = antr_tools

            # Use parse() for Pydantic models (recommended by Anthropic SDK)
            if response_format and hasattr(response_format, 'model_json_schema'):
                return self.client.beta.messages.parse(
                    betas=[STRUCTURED_OUTPUTS_BETA],
                    output_format=response_format,
                    **api_params
                )
            else:
                # Use create() with output_format for raw JSON schemas
                return self.client.beta.messages.create(
                    betas=[STRUCTURED_OUTPUTS_BETA],
                    output_format=output_format_param,
                    **api_params
                )

        except TypeError:
            # Re-raise TypeError directly for fallback handling in chat()
            # This happens when SDK doesn't support output_format parameter
            raise
        except APIConnectionError as e:
            logger.error(f'LLM call failed (APIConnectionError): {e!r}')
            raise ConnectionException(e)
        except (AuthenticationError, PermissionDeniedError) as e:
            logger.error(f'LLM call failed (Auth): {e!r}')
            raise AuthorizationException(e)
        except InternalServerError as e:
            logger.error(f'LLM call failed (500): {e!r}')
            raise ModelOverloadException(e)
        except RateLimitError as e:
            logger.error(f'LLM call failed (RateLimit): {e!r}')
            raise RatelimitException(e)
        except BadRequestError as e:
            logger.error(f'LLM call failed (BadRequest): {e!r}')
            raise BadRequestException(e)
        except Exception as e:
            logger.error(f'LLM call failed (Unexpected): {e!r}')
            raise GeneralException(e)

    async def prompt_async(self, prompt: str, images: ImageInput = None) -> AsyncGenerator[tuple[str, str], None]:
        async for content, reasoning in self.chat_async(prompt, images):
            if content or reasoning:
                yield content, reasoning

    async def chat_async(self, prompt: str, images: ImageInput = None) -> AsyncGenerator[tuple[str, str], None]:
        stream = self.completion(prompt, images, stream=True)
        for event in stream:
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                yield event.delta.text, None  # 2nd parameter is reasoning_content. Not available yet for Anthropic

    def completion(self, prompt: str, images: ImageInput = None, tools: list = None, return_json: bool = False, response_format=None, stream=False):
        system_message = self.cached_system_message() if self.cached_prompt else self.system_message
        
        # Add user message to conversation history if it's a new message
        if prompt or images:
            user_message = create_anthropic_message('user', prompt, images)
            self.messages.append(user_message)
        
        antr_tools = transform_tools(tools or []) if tools is not None else None

        for _ in range(3):
            try:
                if stream:
                    if tools:
                        raise NotImplementedError('Anthropic model does not support streaming and tools at the same time')
                    return self.client.messages.create(
                        model=self.model_name,
                        system=system_message,
                        messages=self.messages,
                        stream=True,
                        **self.model_params
                    )
                
                # Prepare messages for the API call
                api_messages = []
                tool_use_context = {}
                
                for msg in self.messages:
                    if msg['role'] == 'user':
                        # For user messages, include them as-is
                        api_messages.append(msg)
                    elif msg['role'] == 'assistant':
                        # For assistant messages, include tool_use blocks and text content
                        content = []
                        for item in msg.get('content', []):
                            if isinstance(item, dict):
                                if item.get('type') == 'tool_use':
                                    # Store tool use context for later reference
                                    tool_use_context[item['id']] = item
                                    content.append(item)
                                elif item.get('type') == 'text':
                                    content.append(item)
                            elif isinstance(item, str):
                                content.append({"type": "text", "text": item})
                        
                        if content:
                            api_messages.append({"role": "assistant", "content": content})
                    
                    # Tool results are handled as part of the next user message
                    # So we don't need to add them to api_messages
                
                # Prepare the API call parameters
                api_params = {
                    'model': self.model_name,
                    'messages': api_messages,
                    **self.model_params
                }
                
                # Only add system message if not using cached prompt
                api_params['system'] = system_message
                    
                # Only add tools if we have any
                if antr_tools:
                    api_params['tools'] = antr_tools
                
                # Make the API call
                result = self.client.messages.create(**api_params)
            except APIConnectionError as e:
                print("LLM call failed (APIConnectionError):", repr(e))
                raise ConnectionException(e)
            except (AuthenticationError, PermissionDeniedError) as e:
                print("LLM call failed (Auth):", repr(e))
                raise AuthorizationException(e)
            except InternalServerError as e:
                print("LLM call failed (500):", repr(e))
                raise ModelOverloadException(e)
            except RateLimitError as e:
                print("LLM call failed (RateLimit):", repr(e))
                raise RatelimitException(e)
            except BadRequestError as e:
                print("LLM call failed (BadRequest):", repr(e))
                raise BadRequestException(e)
            except Exception as e:
                print("LLM call failed (Unexpected):", repr(e))
                raise GeneralException(e)

            # Check for tool use in the response
            tool_use_blocks = [block for block in result.content if hasattr(block, 'type') and block.type == "tool_use"]
            
            if tool_use_blocks:
                # Create a new assistant message with the tool use blocks
                assistant_content = []
                tool_results = []
                
                for block in result.content:
                    if hasattr(block, 'type') and block.type == 'tool_use':
                        # Add tool use to the assistant's message
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                        
                        # Execute the function
                        function = self.encapsulating_model.functions.get(block.name)
                        if not function:
                            raise ValueError(f"Function {block.name} not found in model's functions")
                        
                        function_result = function(**block.input)
                        
                        # Prepare tool result for the next user message
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(function_result) if not isinstance(function_result, (str, int, float, bool)) else str(function_result)
                        })
                    elif hasattr(block, 'text'):
                        assistant_content.append({"type": "text", "text": block.text})
                
                # Add the assistant's message with tool use to history
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Add tool results as a new user message if there are any
                if tool_results:
                    self.messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                
                # Continue the conversation with the tool results
                continue
            
            # If we get here, we have a final response with no tool use
            # Add the assistant's response to conversation history
            assistant_content = []
            for block in result.content:
                if hasattr(block, 'text'):
                    assistant_content.append({"type": "text", "text": block.text})
                elif hasattr(block, 'tool_use'):
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.tool_use.id,
                        "name": block.tool_use.name,
                        "input": block.tool_use.input
                    })
            
            if assistant_content:
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            return result

    def cached_system_message(self) -> list[dict]:
        return [
                  {
                    "type": "text",
                    "text": self.system_message,
                  },
                  {
                    "type": "text",
                    "text": self.cached_prompt,
                    "cache_control": {"type": "ephemeral"}
                  }
                ]

    def token_count(self, text: str) -> int:
        message = create_anthropic_message('user', text)
        response = self.client.messages.count_tokens(model=self.model_name, messages=[message])
        return response.input_tokens


def transform_tools(tools: list[dict]) -> list[dict]:
    """
    Transforms tools into the format expected by Anthropic's API.
    """
    if not tools:
        return None
        
    type_mapping = {
        int: "integer",
        str: "string",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    
    transformed_tools = []
    
    for tool in tools:
        if "function" in tool:
            # This is a function tool
            tool_def = {
                "name": tool["function"].__name__,
                "description": tool.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Add parameters to input_schema
            for param_name, param_type in tool.get("parameters", {}).items():
                tool_def["input_schema"]["properties"][param_name] = {
                    "type": type_mapping.get(param_type, "string"),
                    "description": param_name
                }
                
            # Add required parameters
            if "required_parameters" in tool:
                tool_def["input_schema"]["required"] = tool["required_parameters"]
                
            transformed_tools.append(tool_def)
    
    return transformed_tools if transformed_tools else None


def create_anthropic_message(role: str, prompt: str, images: ImageInput = None):

    content = []

    if images:
        for img in images:
            base64img = to_base64_image(img)
            mime_type = identify_image_format_from_base64(base64img)
            content += [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64img,
                    },
                }
            ]

    if prompt:
        content += [{"type": "text", "text": prompt}]

    return {"role": role, "content": content}


def extract_dict(text: str) -> dict:
    """ Extracts the first valid JSON dictionary from a string, even if it contains nested objects. """
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError('No JSON object found in text', text, 0)

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise json.JSONDecodeError('No valid JSON object found', text, start)
