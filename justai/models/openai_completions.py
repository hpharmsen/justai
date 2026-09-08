"""Implementation of the OpenAI models.

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
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import tiktoken
from openai import (
    NOT_GIVEN,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from justai.models.basemodel import (
    JSON_TYPE_MAP,
    AuthorizationException,
    BadRequestException,
    BaseModel,
    ConnectionException,
    GeneralException,
    ImageInput,
    ModelOverloadException,
    RatelimitException,
    StreamChunk,
    ToolCallRequest,
    client_retry_kwargs,
    client_timeout,
    get_api_key,
    stream_timeout,
)
from justai.tools.display import DEBUG_COLOR2, color_print
from justai.tools.images import to_base64_image


def map_openai_error(e: Exception) -> Exception:
    """Translate an OpenAI SDK exception to the matching justai exception.

    Order matters: APITimeoutError subclasses APIConnectionError, so it must come first.
    """
    if isinstance(e, APITimeoutError):
        return ModelOverloadException(e)
    if isinstance(e, APIConnectionError):
        return ConnectionException(e)
    if isinstance(e, httpx.TransportError):
        # Mid-stream read failures (ReadError, ReadTimeout, RemoteProtocolError)
        # surface as raw httpx errors: the SDK only wraps the request it issues
        # itself, not the bytes we pull off an already-established stream.
        return ConnectionException(e)
    if isinstance(e, (AuthenticationError, PermissionDeniedError)):
        return AuthorizationException(e)
    if isinstance(e, RateLimitError):
        return RatelimitException(e)
    if isinstance(e, BadRequestError):
        return BadRequestException(e)
    return GeneralException(e)


def tiktoken_token_count(model_name: str, text: str) -> int:
    """Count tokens with tiktoken, falling back to cl100k_base for models it doesn't know."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(text))


class OpenAICompletionsModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name}, a large language model trained by OpenAI.'
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = get_api_key(params, 'OPENAI_API_KEY', 'OpenAI', 'https://platform.openai.com/account/api-keys')

        self.client = OpenAI(api_key=api_key, timeout=client_timeout(params), **client_retry_kwargs(params))
        self.supports_function_calling = True
        # Provider-specific hook for extra kwargs (e.g. OpenRouter passes reasoning via extra_body).

        # Only include system message if not empty (some providers reject empty system messages)
        assert self.system_message is None or isinstance(self.system_message, str), (
            f'system_message must be a string, got {type(self.system_message)}'
        )
        if self.system_message and self.system_message.strip():
            self.messages = [{'role': 'system', 'content': self.system_message}]
        else:
            self.messages = []

    def _extra_api_kwargs(self) -> dict:
        """Provider hook: extra kwargs merged into every chat.completions call. Override in subclass."""
        return {}

    def chat(
        self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format
    ) -> tuple[Any, int | None, int | None, dict | None]:

        raise NotImplementedError(
            'Justai with the Open AI Completion API does not support chat anymore, use prompt or another model'
        )

    def prompt(
        self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format
    ) -> tuple[Any, int | None, int | None]:

        if not tools:  # Models like deepseek-chat don't like tools to be an empty list
            tools = NOT_GIVEN

        # Reset messages - only include system message if not empty
        if self.system_message and self.system_message.strip():
            self.messages = [{'role': 'system', 'content': self.system_message}]
        else:
            self.messages = []

        completion = self.completion(prompt, images, tools, return_json, response_format)

        message = completion.choices[0].message
        message_text = message.content
        input_token_count = completion.usage.prompt_tokens
        output_token_count = completion.usage.completion_tokens
        # Record before parsing: an empty or unparseable answer was billed all the same.
        self.record_usage(input_token_count, output_token_count)

        if message_text and message_text.startswith('```json'):
            message_text = message_text[7:-3]
        if return_json and self.supports_return_json:
            if not message_text:
                raise ValueError(f'Expected JSON response but got empty response from {self.model_name}')
            result = json.loads(message_text)
        else:
            result = message_text

        if self.debug:
            color_print(f'{message_text}', color=DEBUG_COLOR2)

        return result, input_token_count, output_token_count

    async def prompt_async(self, prompt: str, images: ImageInput = None) -> AsyncGenerator[tuple[str, str], None]:
        async for content, reasoning in self.chat_async(prompt, images):
            yield content, reasoning

    async def chat_async(self, prompt: str, images: ImageInput = None) -> AsyncGenerator[tuple[str, str], None]:
        # Get the streaming response
        stream = self.completion(
            prompt=prompt, images=images, tools=NOT_GIVEN, return_json=False, response_format=NOT_GIVEN, stream=True
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

    def completion(
        self,
        prompt: str,
        images: ImageInput,
        tools=NOT_GIVEN,
        return_json: bool = False,
        response_format=NOT_GIVEN,
        stream: bool = False,
    ):

        if tools and not self.supports_function_calling:
            raise NotImplementedError(f'{self.model_name} does not support function calling')
        if images and not self.supports_image_input:
            raise NotImplementedError(f'{self.model_name} does not support image input')

        content = [{'type': 'text', 'text': prompt or ''}]
        if images:
            for image in images:
                content.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{to_base64_image(image)}'},
                    }
                )
        self.messages += [{'role': 'user', 'content': content}]
        if response_format:
            if 'openai.com' not in str(self.client.base_url):
                raise NotImplementedError('response_model is only supported with OpenAI models')
            if stream:
                raise NotImplementedError('streaming is not supported with response_model')
        else:
            if return_json and not stream and self.supports_return_json:
                response_format = {'type': 'json_object'}

            if self.model_name.startswith('gpt-5'):
                self.model_params['temperature'] = 1  # Only the default of 1 is supported in GPT-5

        # Create the completion with streaming
        tool_spec = NOT_GIVEN if tools is NOT_GIVEN else self.create_tool_spec(tools)

        for _ in range(3):  # Max 3 function calls to prevent infinite loop
            try:
                if response_format:
                    # Structured output requires enough tokens to complete the JSON.
                    # Truncated JSON is always useless, so enforce a reasonable minimum.
                    params = {**self.api_params}
                    max_allowed = getattr(self, 'max_output_tokens', 16384)
                    min_structured = min(16384, max_allowed)
                    if params.get('max_tokens', 0) < min_structured:
                        params['max_tokens'] = min_structured
                    result = self.client.chat.completions.parse(
                        model=self.model_name,
                        messages=self.messages,
                        tools=tool_spec,
                        response_format=response_format,
                        **params,
                        **self._extra_api_kwargs(),
                    )
                else:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        tools=tool_spec,
                        stream=stream,
                        **self.api_params,
                        **self._extra_api_kwargs(),
                    )
            except NotImplementedError:
                raise  # Raised deliberately above; not an API failure
            except Exception as e:
                raise map_openai_error(e)

            # For streaming, return the stream directly
            if stream:
                return result

            # Inspect the model's response
            response_msg = result.choices[0].message

            if not response_msg.tool_calls:
                if 'response_format' in self.model_params:
                    del self.model_params['response_format']
                return result

            # Tool call was triggered
            for tool_call in response_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                for tool in tools:
                    if tool['function'].__name__ == fn_name:
                        break
                else:
                    raise ValueError(f'Function {fn_name} not found')

                function = tool['function']
                result = function(**fn_args)

                # Send tool result back to the model
                self.messages.append(response_msg.model_dump())  # include model’s function call message
                self.messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': json.dumps(result),
                    }
                )

    def token_count(self, text: str) -> int:
        """Returns the number of tokens in a string."""
        return tiktoken_token_count(self.model_name, text)

    @staticmethod
    def create_tool_spec(tools: list[dict]) -> list[dict]:
        if not tools:
            return []

        tool_spec = []
        for tool in tools:
            if 'function' in tool and callable(tool['function']):
                function_name = tool['function'].__name__
            elif 'function' in tool and isinstance(tool['function'], str):
                function_name = tool['function']
            else:
                # If function is not provided, use a default name
                function_name = tool.get('name', 'unknown_function')

            tool_spec.append(
                {
                    'type': 'function',
                    'function': {
                        'name': function_name,
                        'description': tool.get('description', ''),
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                param_name: {
                                    'type': JSON_TYPE_MAP.get(_type, 'string'),
                                    'description': param_name,
                                }
                                for param_name, _type in tool.get('parameters', {}).items()
                            },
                            'required': tool.get('required_parameters', []),
                        },
                    },
                }
            )
        return tool_spec

    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator[StreamChunk, None]:
        """Stateless streaming call for the Chat Completions API."""
        # Build tool spec from provider-agnostic format
        tool_spec = NOT_GIVEN
        if tools:
            tool_spec = []
            for t in tools:
                params = t.get('parameters') or t.get('input_schema', {})
                tool_spec.append(
                    {
                        'type': 'function',
                        'function': {
                            'name': t['name'],
                            'description': t.get('description', ''),
                            'parameters': params,
                        },
                    }
                )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tool_spec,
                stream=True,
                timeout=stream_timeout(self.model_params),
                **self.api_params,
                **self._extra_api_kwargs(),
            )
        except Exception as e:
            raise map_openai_error(e)

        # Accumulate tool calls from streaming deltas
        pending_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
        input_tokens = 0
        output_tokens = 0

        # The stream can also fail while it is being read, not just while it is
        # being opened. Those failures need the same mapping, otherwise raw httpx
        # exceptions escape past every `except ConnectionException` the caller has.
        try:
            for chunk in response:
                if not chunk.choices:
                    # Usage chunk (some providers send usage in a separate chunk)
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens or 0
                        output_tokens = chunk.usage.completion_tokens or 0
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                # Text content
                if delta.content:
                    yield StreamChunk(type='text', content=delta.content)

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                'id': tc_delta.id or '',
                                'name': tc_delta.function.name if tc_delta.function and tc_delta.function.name else '',
                                'arguments': '',
                            }
                        else:
                            if tc_delta.id:
                                pending_tool_calls[idx]['id'] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                pending_tool_calls[idx]['name'] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            pending_tool_calls[idx]['arguments'] += tc_delta.function.arguments

                # Usage from chunk
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
        except Exception as e:
            raise map_openai_error(e)

        # Emit accumulated tool calls
        if pending_tool_calls:
            tool_calls = []
            for idx in sorted(pending_tool_calls):
                tc = pending_tool_calls[idx]
                tool_calls.append(
                    ToolCallRequest(
                        id=tc['id'],
                        name=tc['name'],
                        arguments=json.loads(tc['arguments']) if tc['arguments'] else {},
                    )
                )
            yield StreamChunk(type='tool_calls', tool_calls=tool_calls)

        yield StreamChunk(type='done', input_tokens=input_tokens, output_tokens=output_tokens)

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool result message for the Chat Completions API."""
        return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': result}

    def format_assistant_message(self, text: str, tool_calls: list[ToolCallRequest] | None = None) -> list[dict]:
        """Format an assistant message for the Chat Completions API."""
        msg = {'role': 'assistant', 'content': text or ''}
        if tool_calls:
            msg['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]
        return [msg]
