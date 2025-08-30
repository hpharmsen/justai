""" Implementation of the OpenAI models. 

Feature table:
    - Async chat:       YES
    - Return JSON:      YES
    - Structured types: YES, via Pydantic  TODO: Add support for native Python types
    - Token count:      YES
    - Image support:    YES
    - Tool use:         YES

Supported parameters:    
    # The maximum number of tokens to generate in the completion.
    # Defaults to 16
    # The token count of your prompt plus max_tokens cannot exceed the model's context length.
    # Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
    self.model_params['max_tokens'] = params.get('max_tokens', 800)

    # What sampling temperature to use, between 0 and 2.
    # Higher values like 0.8 will make the output more random, while lower values like 0.2
    # will make it more focused and deterministic.
    # We generally recommend altering this or top_p but not both
    # Defaults to 1
    self.model_params['temperature'] = params.get('temperature', 0.5)

    # An alternative to sampling with temperature, called nucleus sampling,
    # where the model considers the results of the tokens with top_p probability mass.
    # So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    # We generally recommend altering this or temperature but not both.
    # Defaults to 1
    self.model_params['top_p'] = params.get('top_p', 1)

    # How many completions to generate for each prompt.
    # Because this parameter generates many completions, it can quickly consume your token quota.
    # Use carefully and ensure that you have reasonable settings for max_tokens.
    self.model_params['n'] = params.get('n', 1)

    # Number between -2.0 and 2.0.
    # Positive values penalize new tokens based on whether they appear in the text so far,
    # increasing the model's likelihood to talk about new topics.
    # Defaults to 0
    self.model_params['presence_penalty'] = params.get('presence_penalty', 0)

    # Number between -2.0 and 2.0.
    # Positive values penalize new tokens based on their existing frequency in the text so far,
    # decreasing the model's likelihood to repeat the same line verbatim.
    # Defaults to 0
    self.model_params['frequency_penalty'] = params.get('frequency_penalty', 0)
"""

import asyncio
import json
import os
from typing import Any, AsyncGenerator

import instructor
import tiktoken
from dotenv import dotenv_values
from openai import OpenAI, NOT_GIVEN, APIConnectionError, \
    RateLimitError, APITimeoutError, AuthenticationError, PermissionDeniedError, BadRequestError

from justai.model.message import Message, ToolUseMessage
from justai.tools.display import color_print, ERROR_COLOR, DEBUG_COLOR2
from justai.models.basemodel import (
    BaseModel,
    ConnectionException,
    AuthorizationException,
    ModelOverloadException,
    RatelimitException,
    BadRequestException,
    GeneralException,
    ImageInput,
)
from justai.tools.images import to_base64_image


class OpenAICompletionsModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by OpenAI."
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = params.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or dotenv_values()["OPENAI_API_KEY"]
        if not api_key:
            color_print("No OpenAI API key found. Create one at https://platform.openai.com/account/api-keys and " +
                        "set it in the .env file like OPENAI_API_KEY=here_comes_your_key.", color=ERROR_COLOR)

        # instructor.patch makes the OpenAI client compatible with structured output via response_model=”
        # Works only for OpenAI models
        self.client = instructor.patch(OpenAI(api_key=api_key))
        self.messages = [{"role": "system", "content": self.system_message}]

    def chat(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None, dict|None]:

        raise NotImplementedError("Justai with the Open AI Completion API does not support chat anymore, use prompt or another model")

    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) -> tuple[Any, int|None, int|None]:

        if not tools: # Models like deepseek-chat don't like tools to be an empty list
            tools = NOT_GIVEN

        self.messages = [{"role": "system", "content": self.system_message}]  # Reset messages to system prompt

        completion = self.completion(prompt, images, tools, return_json, response_format)

        # if response_format:
        #     # Intended behavior bij OpenAI. When response_format is specified, the raw response is alreay
        #     # deserialized into the requested format.
        #     # Disadvantage: the raw response is not available so no token count or tool use
        #     return completion, None, None

        message = completion.choices[0].message
        message_text = message.content
        input_token_count = completion.usage.prompt_tokens
        output_token_count = completion.usage.completion_tokens

        if message_text and message_text.startswith("```json"):
            print("Unexpected JSON response found in completion")
            message_text = message_text[7:-3]
        result = json.loads(message_text) if return_json and self.supports_return_json else message_text

        if self.debug:
            color_print(f"{message_text}", color=DEBUG_COLOR2)

        return result, input_token_count, output_token_count

    async def prompt_async(self, prompt: str) -> AsyncGenerator[tuple[str, str], None]:

        async for content, reasoning in self.chat_async(prompt):
            yield content, reasoning

    async def chat_async(self, prompt: str) -> AsyncGenerator[tuple[str, str], None]:
        # Get the streaming response
        stream = self.completion(
            messages=self.messages,
            tools=None,
            return_json=False,
            response_model=None,
            stream=True
        )

        # Process the streaming response
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            content = getattr(delta, 'content', None)
            reasoning = getattr(delta, 'reasoning_content', None)

            if content or reasoning:
                yield (content or ''), (reasoning or '')
                # Small sleep to prevent overwhelming the event loop
                await asyncio.sleep(0.01)

    def completion(self, prompt: str, images: ImageInput, tools=NOT_GIVEN, return_json: bool = False,
                   response_format = NOT_GIVEN, stream: bool = False):

        if tools and not self.supports_function_calling:
            raise NotImplementedError(f"{self.model_name} does not support function calling")
        if images and not self.supports_image_input:
            raise NotImplementedError(f"{self.model_name} does not support image input")

        content = [{"type": "text", "text": prompt or ""}]
        if images:
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{to_base64_image(image)}"
                        },
                    }
                )
        self.messages += [{'role':'user', 'content':content}]
        if response_format:
            if "openai.com" not in str(self.client.base_url):
                raise NotImplementedError("response_model is only supported with OpenAI models")
            if stream:
                raise NotImplementedError("streaming is not supported with response_model")
        else:
            if return_json and not stream and self.supports_return_json:
                response_format = {"type": "json_object"}

            if self.model_name.startswith("gpt-5"):
                self.model_params["temperature"] = 1  # Only the default of 1 is supported in GPT-5

        # Create the completion with streaming
        tool_spec = NOT_GIVEN if tools is NOT_GIVEN else self.create_tool_spec(tools)

        for _ in range(3):
            try:
                if response_format:
                    result = self.client.chat.completions.parse(
                        model=self.model_name,
                        messages=self.messages,
                        tools=tool_spec,
                        response_format=response_format,
                        **self.model_params,
                    )
                else:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        tools=tool_spec,
                        stream=stream,
                        **self.model_params,
                    )
            except APITimeoutError as e:
                raise ModelOverloadException(e)
            except APIConnectionError as e:
                raise ConnectionException(e)
            except (AuthenticationError, PermissionDeniedError) as e:
                raise AuthorizationException(e)
            except RateLimitError as e:
                raise RatelimitException(e)
            except BadRequestError as e:
                raise BadRequestException(e)
            except NotImplementedError:
                raise
            except Exception as e:
                raise GeneralException(e)

            # Inspect the model’s response
            response_msg = result.choices[0].message

            if not response_msg.tool_calls:
                if "response_format" in self.model_params:
                    del self.model_params["response_format"]
                return result

            # Tool call was triggered
            for tool_call in response_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                for tool in tools:
                    if tool['function'].__name__ == fn_name:
                        break
                else:
                    raise ValueError(f"Function {fn_name} not found")

                function = tool['function']
                result = function(**fn_args)

                # Send tool result back to the model
                self.messages.append(response_msg.model_dump())  # include model’s function call message
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )


    @staticmethod
    def transform_messages(messages: list[Message]) -> list[dict]:
        transformed_messages = []

        for message in messages:
            msg = {"role": message.role}

            # Handle tool messages (function calls and their results)
            # Todo: tool use is geen onderdeel meer van message maar staat in de Model class.
            if isinstance(message, ToolUseMessage):
                if message.role == 'assistant' and 'function_to_call' in message.tool_use:
                    # This is a function call from the assistant
                    msg["content"] = None
                    msg["tool_calls"] = [{
                        "id": message.tool_use.get('call_id', 'call_' + str(hash(str(message.tool_use)))),
                        "type": "function",
                        "function": {
                            "name": message.tool_use['function_to_call'],
                            "arguments": json.dumps(message.tool_use['function_parameters'])
                        }
                    }]
                elif message.role == 'tool':
                    # This is a function result
                    function_result = message.tool_use.get('function_result', '')
                    if not isinstance(function_result, str):
                        function_result = json.dumps(function_result)
                    msg["content"] = function_result
                    msg["tool_call_id"] = message.tool_use.get('call_id', '')
                    msg["name"] = message.tool_use.get('function_to_call', '')
            # Handle regular messages
            else:
                if message.images:
                    content = [{"type": "text", "text": message.content or ""}]
                    for image in message.images:
                        content.append({
                            "type": "image_url",
                            "image_url": {'url': f"data:image/jpeg;base64,{to_base64_image(image)}"}
                        })
                    msg["content"] = content
                else:
                    msg["content"] = message.content or ""

            transformed_messages.append(msg)

        return transformed_messages

    # @staticmethod
    # def tool_use_message(tool_use) -> Message:
    #     """ Creates a message with the result of a tool use. """
    #     return ToolUseMessage(tool_use=tool_use)

    def token_count(self, text: str) -> int:
        """ Returns the number of tokens in a string. """
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fall back to cl100k_base encoding for newer models not yet in tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    @staticmethod
    def create_tool_spec(tools: list[dict]) -> list[dict]:
        if not tools:
            return []

        type_mapping = {
            int: "integer",
            str: "string",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        tool_spec = []
        for tool in tools:
            if 'function' in tool and callable(tool['function']):
                function_name = tool['function'].__name__
            elif 'function' in tool and isinstance(tool['function'], str):
                function_name = tool['function']
            else:
                # If function is not provided, use a default name
                function_name = tool.get('name', 'unknown_function')
                
            tool_spec.append({
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": tool.get('description', ''),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param_name: {
                                "type": type_mapping.get(_type, "string"),
                                "description": param_name,
                            }
                            for param_name, _type in tool.get('parameters', {}).items()
                        },
                        "required": tool.get('required_parameters', []),
                    }
                }
            })
        return tool_spec