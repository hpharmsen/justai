"""Implementation of the OpenAI Responses APImodels.

Feature table:
    - Async chat:       YES
    - Return JSON:      NO
    - Structured types: YES, via Pydantic  TODO: Add support for native Python types
    - Token count:      YES
    - Image support:    YES
    - Tool use:         YES
    - File input:       NOT YET IMPLEMENTED
    - Web search:       NOT YET IMPLEMENTED
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import AsyncGenerator
from io import BytesIO
from typing import Any

import httpx
import pydantic
from jsonschema import Draft202012Validator, exceptions, validators
from openai import OpenAI
from PIL import Image

from justai.models.basemodel import (
    JSON_TYPE_MAP,
    BaseModel,
    ImageInput,
    StreamChunk,
    ToolCallRequest,
    client_retry_kwargs,
    client_timeout,
    get_api_key,
    parse_tool_arguments,
)
from justai.models.openai_completions import map_openai_error, tiktoken_token_count
from justai.tools.images import extract_images, get_image_type, to_base64_data_uri, to_base64_image

# Models that support the reasoning parameter (GPT-5.6 and GPT-6, per OpenAI docs, September 2026).
EFFORT_MODELS = re.compile(r'gpt-(5\.6|6)')
# GPT-6 Astra answers reasoning effort 'none' with HTTP 400.
NO_NONE_EFFORT_MODELS = re.compile(r'gpt-6-astra')


def _validate_json_schema(schema: Any) -> None:
    """Raise ValueError if schema is not a valid JSON Schema. Honours $schema when present."""
    validator_cls = validators.validator_for(schema, default=Draft202012Validator)
    try:
        validator_cls.check_schema(schema)
    except exceptions.SchemaError as e:
        path = '/' + '/'.join(map(str, e.path)) if e.path else '(root)'
        raise ValueError(f'response_format is not a valid JSON Schema [at {path}]: {e.message}') from e


class OpenAIResponsesModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name}, a large language model trained by OpenAI.'
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = get_api_key(params, 'OPENAI_API_KEY', 'OpenAI', 'https://platform.openai.com/account/api-keys')

        self.client = OpenAI(api_key=api_key, timeout=client_timeout(params), **client_retry_kwargs(params))

        # Diversions from the features that are supported or not supported by default
        self.supports_function_calling = True
        self.supports_image_generation = True

        self.last_response_id = None

    def _validate_effort(self, value: Any) -> None:
        # 'none' is a valid pass-through for effort models that can turn reasoning off.
        if (
            value == 'none'
            and EFFORT_MODELS.search(self.model_name)
            and not NO_NONE_EFFORT_MODELS.search(self.model_name)
        ):
            return
        super()._validate_effort(value)

    def resolve_effort(self) -> tuple[str | None, str | None]:
        level = self.model_params.get('effort')
        if level is None:
            return (None, None)
        if EFFORT_MODELS.search(self.model_name):
            # Every justai level is native. 'max' goes through although the SDK's ReasoningEffort
            # Literal stops at 'xhigh': the API accepts it (OpenAI docs, September 2026).
            return (level, None)
        return (None, f'effort is not supported by {self.model_name}, ignoring')

    def _reasoning_extra(self) -> dict:
        """Return {'reasoning': {'effort': X}} when applicable, else {}. Emits warnings."""
        native, warn = self.resolve_effort()
        if warn:
            self._emit_effort_warning(warn)
        return {'reasoning': {'effort': native}} if native is not None else {}

    def _responses_create(self, **kwargs):
        return self.client.responses.create(**kwargs, **self._reasoning_extra())

    def prompt(
        self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format, _chat=False
    ) -> tuple[Any, int | None, int | None]:

        content = self.create_content(prompt, images)
        input_list = [{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': content}]
        tool_spec = self.create_tool_spec(tools)

        last_response_id = self.last_response_id if _chat else None

        is_pydantic = (
            bool(response_format)
            and isinstance(response_format, type)
            and issubclass(response_format, pydantic.BaseModel)
        )

        for run in range(3):  # Max 3 function calls to prevent infinite loop
            try:
                if is_pydantic:
                    # Pydantic model: the strict format responses.parse() would send, but the
                    # raw text comes back. Model validates it; return_json is ignored here. A private
                    # SDK module, imported here so an SDK that moves it breaks structured output only.
                    from openai.lib._parsing._responses import type_to_text_format_param

                    response = self._responses_create(
                        model=self.model_name,
                        input=input_list,
                        tools=tool_spec,
                        text={'format': type_to_text_format_param(response_format)},
                        previous_response_id=last_response_id,
                    )
                elif response_format:
                    _validate_json_schema(response_format)
                    response = self._responses_create(
                        model=self.model_name,
                        input=input_list,
                        tools=tool_spec,
                        text={
                            'format': {
                                'type': 'json_schema',
                                'name': 'response_format',
                                'strict': True,
                                'schema': response_format,
                            }
                        },
                        previous_response_id=last_response_id,
                    )
                elif return_json:
                    response = self._responses_create(
                        model=self.model_name,
                        input=input_list,
                        tools=tool_spec,
                        text={'format': {'type': 'json_object'}},
                        previous_response_id=last_response_id,
                    )
                else:
                    response = self._responses_create(
                        model=self.model_name, input=input_list, tools=tool_spec, previous_response_id=last_response_id
                    )
            except Exception as e:
                raise map_openai_error(e)

            # Save the response id for subsequent requests
            self.last_response_id = response.id if _chat else None

            # Save function call outputs for subsequent requests
            function_call = None
            function_call_arguments = None
            input_list += response.output

            for item in response.output:
                if item.type == 'function_call':
                    function_call = item
                    function_call_arguments = json.loads(item.arguments)

            if not function_call or run == 2:
                # Record before parsing: output that turns out not to be JSON was billed too.
                # OpenAI cachet server-side en vanzelf; alleen de read-teller vult zich.
                # Anders dan bij Anthropic zijn deze tokens een deelverzameling van
                # input_tokens, niet iets dat er los naast staat.
                details = getattr(response.usage, 'input_tokens_details', None)
                self.record_usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    cache_read_tokens=getattr(details, 'cached_tokens', 0),
                )
                if return_json and not is_pydantic:
                    output = json.loads(response.output_text)
                else:
                    output = response.output_text
                return output, response.usage.input_tokens, response.usage.output_tokens

            # The rest of the function is to process the function call output
            # Numbering is from OpenAI's docs: https://platform.openai.com/docs/guides/function-calling#function-tool-example

            for tool in tools:
                if tool['function'].__name__ == function_call.name:
                    function = tool['function']
                    break
            else:
                raise ValueError(f'Function {function_call.name} not found')

            # 3. Execute the function logic for get_horoscope
            result = {function_call.name: function(*function_call_arguments.values())}

            # 4. Provide function call results to the model
            input_list.append(
                {
                    'type': 'function_call_output',
                    'call_id': function_call.call_id,
                    'output': json.dumps(result),
                }
            )

    async def prompt_async(
        self, prompt: str, images: list[ImageInput] = None, _chat=False
    ) -> AsyncGenerator[tuple[str, str], None]:

        content = self.create_content(prompt, images)
        input_ = [{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': content}]

        last_response_id = self.last_response_id if _chat else None

        response = self._responses_create(
            model=self.model_name, input=input_, stream=True, previous_response_id=last_response_id
        )

        for event in response:
            # A Stream has no id; it arrives in the first event. Save it for subsequent requests
            if event.type == 'response.created':
                self.last_response_id = event.response.id if _chat else None
            elif hasattr(event, 'delta'):
                yield event.delta, ''  # Second value is reasoning (not available for OpenAI)

    def chat(
        self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format
    ) -> tuple[Any, int | None, int | None]:
        return self.prompt(prompt, images, tools, return_json, response_format, _chat=True)

    async def chat_async(self, prompt: str, images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        async for chunk in self.prompt_async(prompt, images, _chat=True):
            yield chunk

    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator[StreamChunk, None]:
        """Stateless streaming call. Caller manages conversation history."""
        # Build input list from messages
        input_list = []
        for msg in messages:
            if 'role' not in msg:
                # Tool result or other provider-specific format — pass through as-is
                input_list.append(msg)
            elif msg['role'] == 'system':
                input_list.append({'role': 'developer', 'content': msg['content']})
            else:
                input_list.append(msg)

        # Build tool spec from tool dicts, accepting both 'parameters' and 'input_schema' keys
        tool_spec = []
        if tools:
            for t in tools:
                params = t.get('parameters') or t.get('input_schema', {})
                tool_spec.append(
                    {
                        'type': 'function',
                        'name': t['name'],
                        'description': t.get('description', ''),
                        'parameters': params,
                    }
                )

        try:
            response = self._responses_create(
                model=self.model_name,
                input=input_list,
                tools=tool_spec or [],
                stream=True,
            )
        except Exception as e:
            raise map_openai_error(e)

        # Track function calls: {output_index: {call_id, name, arguments_str}}
        pending_calls = {}
        tool_calls = []
        input_tokens = 0
        output_tokens = 0

        for event in response:
            if event.type == 'response.output_text.delta':
                yield StreamChunk(type='text', content=event.delta)

            elif event.type == 'response.output_item.added':
                if hasattr(event, 'item') and event.item.type == 'function_call':
                    pending_calls[event.output_index] = {
                        'call_id': event.item.call_id,
                        'name': event.item.name,
                        'arguments': '',
                    }

            elif event.type == 'response.function_call_arguments.delta':
                if event.output_index in pending_calls:
                    pending_calls[event.output_index]['arguments'] += event.delta

            elif event.type == 'response.function_call_arguments.done':
                if event.output_index in pending_calls:
                    call = pending_calls[event.output_index]
                    arguments, raw = parse_tool_arguments(call['arguments'])
                    tool_calls.append(
                        ToolCallRequest(
                            id=call['call_id'],
                            name=call['name'],
                            arguments=arguments,
                            raw_arguments=raw,
                        )
                    )

            elif event.type == 'response.completed':
                usage = event.response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens

        if tool_calls:
            yield StreamChunk(type='tool_calls', tool_calls=tool_calls)

        yield StreamChunk(type='done', input_tokens=input_tokens, output_tokens=output_tokens)

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool result message for the OpenAI Responses API."""
        return {'type': 'function_call_output', 'call_id': tool_call_id, 'output': result}

    def format_assistant_message(self, text: str, tool_calls: list[ToolCallRequest] | None = None) -> list[dict]:
        """Format an assistant message for OpenAI Responses API."""
        items = []
        if text:
            items.append(
                {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': text}],
                }
            )
        for tc in tool_calls or []:
            items.append(
                {
                    'type': 'function_call',
                    'call_id': tc.id,
                    'name': tc.name,
                    'arguments': json.dumps(tc.arguments),
                }
            )
        return items

    @staticmethod
    def create_content(prompt, images: list[ImageInput]) -> list[dict]:
        content = [{'type': 'input_text', 'text': prompt}]

        if images:
            for image in images:
                # Always convert to base64 data URI to avoid URL download issues
                # Some servers (like Wikipedia) block OpenAI's download attempts
                image_url = to_base64_data_uri(image)
                content += [{'type': 'input_image', 'image_url': image_url}]
        return content

    @staticmethod
    def create_tool_spec(tools: list[dict]) -> list[dict]:
        if not tools:
            return []

        tool_spec = []
        for tool in tools:
            tool_spec += [
                {
                    'type': 'function',
                    'name': tool['function'].__name__,
                    'description': tool['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            param_name: {
                                'type': JSON_TYPE_MAP.get(_type, 'string'),  # Default to string if type not in map
                                'description': param_name,  # Or a more descriptive text if available
                            }
                            for param_name, _type in tool['parameters'].items()
                        },
                        'required': tool['required_parameters'] or [],
                    },
                }
            ]
        return tool_spec

    def token_count(self, text: str) -> int:
        """Returns the number of tokens in a string."""
        return tiktoken_token_count(self.model_name, text)

    def generate_image(self, prompt, images: ImageInput, size: tuple[int, int] | None = None, options: dict = None):
        if 'gpt-image' in self.model_name:
            return self._generate_image_api(prompt, images, size, options)
        return self._generate_image_responses(prompt, images)

    def _pick_image_api_size(self, size: tuple[int, int] | None) -> str:
        from math import sqrt

        if not size:
            raise ValueError(f'{self.model_name} requires an explicit size=(width, height) for image generation')

        def fit_size(size: tuple[int, int]) -> tuple[int, int]:
            """OpenAI Size constraints:
            - Maximum edge length must be less than or equal to 3840px
            - Both edges must be multiples of 16px
            - Long edge to short edge ratio must not exceed 3:1
            - Total pixels must be at least 655,360 and no more than 8,294,400"""
            min_pixels = 655_360
            max_pixels = 8_294_400
            max_edge = 3_840
            max_ratio = 3

            w, h = size
            ratio = w / h
            if ratio > max_ratio:
                w = h * max_ratio
            elif ratio < 1 / max_ratio:
                h = w * max_ratio

            pixels = w * h
            if pixels > max_pixels:
                scale = sqrt(max_pixels / pixels)
                w *= scale
                h *= scale
            elif pixels < min_pixels:
                scale = sqrt(min_pixels / pixels)
                w *= scale
                h *= scale

            if max(w, h) > max_edge:
                scale = max_edge / max(w, h)
                w *= scale
                h *= scale

            w = max(16, round(w / 16) * 16)
            h = max(16, round(h / 16) * 16)

            while w * h > max_pixels or max(w, h) > max_edge or max(w, h) / min(w, h) > max_ratio:
                if w >= h:
                    w -= 16
                else:
                    h -= 16

            while w * h < min_pixels:
                if w <= h:
                    w += 16
                else:
                    h += 16

            return w, h

        w, h = fit_size(size)
        return f'{w}x{h}'

    def _generate_image_api(
        self, prompt, images: ImageInput, size: tuple[int, int] | None = None, options: dict = None
    ):
        """Generate image via OpenAI Images API (gpt-image-2)."""
        api_size = self._pick_image_api_size(size)
        quality = (options or {}).get('quality', 'medium')

        if images:
            # images.edit: convert reference images to PNG file-like objects with proper mime type
            image_files = []
            for i, image in enumerate(images):
                img_type = get_image_type(image)
                if img_type == 'image_url':
                    img_data = httpx.get(image, headers={'User-Agent': 'JustAI/1.0'}).content
                    # Re-encode as PNG to ensure correct format
                    pil = Image.open(BytesIO(img_data))
                    buf = BytesIO()
                    pil.save(buf, format='PNG')
                    img_data = buf.getvalue()
                elif img_type == 'pil_image':
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    img_data = buf.getvalue()
                else:
                    img_data = image
                image_files.append((f'ref_{i}.png', img_data, 'image/png'))

            resp = self.client.images.edit(
                model=self.model_name,
                image=image_files,
                prompt=prompt,
                n=1,
                size=api_size,
                quality=quality,
            )
        else:
            resp = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                n=1,
                size=api_size,
                quality=quality,
            )

        img_data = resp.data[0]
        if hasattr(img_data, 'b64_json') and img_data.b64_json:
            raw = base64.b64decode(img_data.b64_json)
        elif hasattr(img_data, 'url') and img_data.url:
            raw = httpx.get(img_data.url).content
        else:
            raise RuntimeError('No image data in Images API response')
        return Image.open(BytesIO(raw))

    def _generate_image_responses(self, prompt, images: ImageInput):
        """Generate image via Responses API with image_generation tool."""
        client = OpenAI()
        input_structure = [
            {
                'role': 'user',
                'content': [{'type': 'input_text', 'text': prompt}],
            }
        ]
        if images:
            for image in images:
                if get_image_type(image) == 'image_url':
                    image_dict = {'type': 'input_image', 'image_url': image}
                else:
                    image_dict = {'type': 'input_image', 'image_data': to_base64_image(image)}
                input_structure[0]['content'] += [image_dict]

        resp = client.responses.create(
            model=self.model_name,
            input=input_structure,
            tools=[{'type': 'image_generation'}],
        )

        images_b64 = extract_images(resp)

        # Decode base64 -> PIL image
        raw = base64.b64decode(images_b64[0])
        img = Image.open(BytesIO(raw))
        return img
