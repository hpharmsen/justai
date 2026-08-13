""" Implementation of the Google models.
https://ai.google.dev/gemini-api/docs/migrate

Feature table:
    - Async chat:       YES (1)
    - Return JSON:      YES
    - Structured types: YES, via Python type definition
    - Token counter:    YES
    - Image support:    YES 
    - Tool use:         YES (via stream/agent)

Supported parameters:
    max_output_tokens= 400,
    top_k= 2,
    top_p= 0.5,
    temperature= 0.5,
    response_mime_type= 'application/json',
    stop_sequences= ['\n'],
    seed=42,

(1) In contrast to Model.chat, Model.chat_async cannot return json and does not return input and output token counts

"""
import json
import logging
import re
from io import BytesIO
from typing import Any, AsyncGenerator

from PIL import Image
from google import genai
from google.genai.errors import APIError

from justai.model.model import ImageInput
from justai.models.anthropic_models import extract_json
from justai.models.basemodel import (
    get_api_key, BaseModel, DEFAULT_TIMEOUT, StreamChunk, ToolCallRequest,
    AuthorizationException, BadRequestException, ConnectionException, GeneralException,
    ModelOverloadException, RatelimitException, TimeoutException, TruncatedResponseException,
)
from justai.tools.images import to_pil_image

logger = logging.getLogger(__name__)


# Only Gemini 3.x uses thinking_level; older Gemini 2.x uses thinking_budget (out of scope).
EFFORT_MODELS_GEMINI3 = re.compile(r'gemini-3')

_EFFORT_MAP_GEMINI3 = {
    'low': ('LOW', None),
    'medium': ('MEDIUM', None),
    'high': ('HIGH', None),
    'xhigh': ('HIGH', 'xhigh -> HIGH (Gemini has no higher tier)'),
    'max': ('HIGH', 'max -> HIGH (Gemini has no higher tier)'),
}


def _map_google_error(e: APIError) -> Exception:
    """Translate a google-genai APIError to the matching justai exception, logging at WARNING.

    WARNING, not ERROR: this fires on every failed attempt and the caller may still
    retry successfully. Only whoever gives up knows the failure is final.

    `code` and `details` stay reachable on the result: Gemini puts its retryDelay in
    details, and a caller that backs off needs it.
    """
    code = getattr(e, 'code', None) or 0
    if code in (401, 403):
        exc, label = AuthorizationException(e), 'Auth'
    elif code == 408:
        exc, label = TimeoutException(e), 'Timeout'
    elif code == 429:
        exc, label = RatelimitException(e), 'RateLimit'
    elif code == 503:
        exc, label = ModelOverloadException(e), 'Overloaded'
    elif 400 <= code < 500:
        exc, label = BadRequestException(e), f'BadRequest {code}'
    elif 500 <= code < 600:
        exc, label = ConnectionException(e), f'ServerError {code}'
    else:
        exc, label = GeneralException(e), f'APIError {code or "?"}'
    exc.code = code or None
    exc.details = getattr(e, 'details', None)
    logger.warning(f'LLM call failed ({label}): {e!r}')
    return exc


class GoogleModel(BaseModel):

    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by Google."
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = get_api_key(params, ('GEMINI_API_KEY', 'GOOGLE_API_KEY'), 'Google',
                              'https://aistudio.google.com/app/apikey')

        # Client (Google uses milliseconds for timeout)
        timeout_ms = int(params.get('timeout', DEFAULT_TIMEOUT) * 1000)
        http_options = genai.types.HttpOptions(timeout=timeout_ms)
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.chat_session = None  # Google uses this to keep track of the chat

        # Diversions from the features that are supported or not supported by default
        self.supports_function_calling = True
        self.supports_automatic_function_calling = True
        self.supports_image_generation = True

        # GenerateContentConfig forbids extras, so an unsupported kwarg either crashes deep
        # inside pydantic on the first call or, worse, is never noticed. Say so right here.
        unknown = sorted(set(self.api_params) - set(genai.types.GenerateContentConfig.model_fields))
        if unknown:
            raise ValueError(f'{model_name} does not accept parameter(s): {", ".join(unknown)}')

    def resolve_effort(self) -> tuple[str | None, str | None]:
        level = self.model_params.get('effort')
        if level is None:
            return (None, None)
        if EFFORT_MODELS_GEMINI3.search(self.model_name):
            native, warn_key = _EFFORT_MAP_GEMINI3[level]
            warn = None if warn_key is None else f'effort={level!r} not natively supported by {self.model_name}; {warn_key}'
            return (native, warn)
        return (None, f'effort is not supported by {self.model_name}, ignoring')

    def _thinking_config_extra(self) -> dict:
        """Return {'thinking_config': ThinkingConfig(...)} when applicable, else {}."""
        native, warn = self.resolve_effort()
        if warn:
            self._emit_effort_warning(warn)
        if native is None:
            return {}
        return {'thinking_config': genai.types.ThinkingConfig(thinking_level=native)}

    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) -> str | object:
        if isinstance(images, str):
            images = [images]
        opened_images = [to_pil_image(img) for img in images] if images else []
        if opened_images:
            prompt = [prompt] + opened_images
        if tools and isinstance(tools[0], dict):
            tools = [tool['function'] for tool in tools]
        params = {**self.api_params}
        if response_format or return_json:
            # Structured output requires enough tokens to complete the JSON, so raise a
            # ceiling that is too low to finish. Only a ceiling the caller set themselves:
            # without one the API default is the model maximum, which beats any number
            # invented here.
            MIN_STRUCTURED_TOKENS = 16384
            ceiling = params.get('max_output_tokens') or 0
            if 0 < ceiling < MIN_STRUCTURED_TOKENS:
                params['max_output_tokens'] = MIN_STRUCTURED_TOKENS
        config = genai.types.GenerateContentConfig(system_instruction=self.system_message, tools=tools,
                                                   **self._thinking_config_extra(), **params)
        if return_json:
            config.response_mime_type = "application/json"
        if response_format:
            config.response_mime_type = "application/json"
            config.response_schema = response_format
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt,
                                                           config=config)
        except APIError as e:
            raise _map_google_error(e) from e
        return convert_to_justai_response(response, return_json or response_format)

    def chat(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None]:

        if return_json:
            raise NotImplementedError('google_model.chat does not support return_json. Use prompt() instead')
        if response_format:
            raise NotImplementedError('google_model.chat does not support response_format. Use prompt() instead')
        if images:
            raise NotImplementedError('google_model.chat does not support images. Use prompt() instead')

        if not self.chat_session:
            self.chat_session = self.client.chats.create(model=self.model_name)
        try:
            response = self.chat_session.send_message(message=prompt)
        except APIError as e:
            raise _map_google_error(e) from e
        return convert_to_justai_response(response, return_json)

    async def prompt_async(self, prompt: str, images: list[ImageInput] = None) -> AsyncGenerator[tuple[str, str], None]:
        if images:
            raise NotImplementedError('google_model. ..._async does not support images. Use prompt() instead')

        config = genai.types.GenerateContentConfig(
            system_instruction=self.system_message,
            **self._thinking_config_extra(),
        )
        async for chunk in self._stream_chunks(contents=prompt, config=config):
            if chunk.text:
                yield chunk.text, ''

    async def chat_async(self, prompt: str, images: list[ImageInput] = None) -> AsyncGenerator[tuple[str, str], None]:
        async for chunk in self.prompt_async(prompt, images):
            yield chunk

    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator[StreamChunk, None]:
        """Stateless streaming call with tool support for the Agent."""
        # Extract system message and convert messages to Google Content objects
        system_instruction = ''
        contents = []
        pending_tool_parts = []

        def flush_tool_parts():
            """Merge consecutive tool results into a single Content."""
            if pending_tool_parts:
                contents.append(genai.types.Content(role='user', parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for msg in messages:
            if msg['role'] == 'tool':
                # Collect tool results; they'll be merged into one Content
                for r in msg['results']:
                    pending_tool_parts.append(
                        genai.types.Part.from_function_response(name=r['name'], response={'result': r['result']})
                    )
                continue

            flush_tool_parts()

            if msg['role'] == 'system':
                system_instruction = msg['content']
            elif msg['role'] == 'user':
                contents.append(genai.types.Content(
                    role='user',
                    parts=[genai.types.Part.from_text(text=msg['content'])],
                ))
            elif msg['role'] == 'model':
                parts = []
                for part in msg.get('parts', []):
                    if 'text' in part:
                        parts.append(genai.types.Part.from_text(text=part['text']))
                    elif 'function_call' in part:
                        fc = part['function_call']
                        parts.append(genai.types.Part.from_function_call(name=fc['name'], args=fc['args']))
                if not parts and 'content' in msg:
                    parts.append(genai.types.Part.from_text(text=msg['content']))
                contents.append(genai.types.Content(role='model', parts=parts))

        flush_tool_parts()

        # Convert tool specs to Google FunctionDeclarations
        google_tools = None
        if tools:
            declarations = []
            for t in tools:
                params = t.get('input_schema', {})
                declarations.append(genai.types.FunctionDeclaration(
                    name=t['name'],
                    description=t.get('description', ''),
                    parameters_json_schema=params if params.get('properties') else None,
                ))
            google_tools = [genai.types.Tool(function_declarations=declarations)]

        config = genai.types.GenerateContentConfig(
            system_instruction=system_instruction or self.system_message,
            tools=google_tools,
            **self._thinking_config_extra(),
            **self.api_params,
        )

        response_stream = self._stream_chunks(contents=contents, config=config)

        input_tokens = 0
        output_tokens = 0
        tool_calls = []

        async for chunk in response_stream:
            # Only yield text when there are no function calls to avoid SDK warning
            if not chunk.function_calls and chunk.text:
                yield StreamChunk(type='text', content=chunk.text)
            if chunk.function_calls:
                for fc in chunk.function_calls:
                    tool_calls.append(ToolCallRequest(
                        id=fc.id or fc.name,
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))
            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count or input_tokens
                output_tokens = (chunk.usage_metadata.candidates_token_count or 0) + \
                                (chunk.usage_metadata.thoughts_token_count or 0)

        if tool_calls:
            yield StreamChunk(type='tool_calls', tool_calls=tool_calls)
        yield StreamChunk(type='done', input_tokens=input_tokens, output_tokens=output_tokens)

    async def _stream_chunks(self, contents, config) -> AsyncGenerator[Any, None]:
        """Yield stream chunks, translating provider errors on both setup and mid-stream."""
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config)
            async for chunk in stream:
                yield chunk
        except APIError as e:
            raise _map_google_error(e) from e

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool result message for Google."""
        return {
            'role': 'tool',
            'results': [{'name': tool_name, 'result': result}],
        }

    def format_assistant_message(self, text: str, tool_calls: list[ToolCallRequest] | None = None) -> list[dict]:
        """Format an assistant message for Google."""
        parts = []
        if text:
            parts.append({'text': text})
        for tc in (tool_calls or []):
            parts.append({'function_call': {'name': tc.name, 'args': tc.arguments}})
        return [{'role': 'model', 'parts': parts}]

    def token_count(self, text: str) -> int:
        response = self.client.models.count_tokens(model=self.model_name, contents=text)
        return response.total_tokens

    def generate_image(self, prompt, images: ImageInput, size: tuple[int, int] | None = None, options: dict = None):
        images = [to_pil_image(img) for img in images] if images else []

        # Build config from options if provided
        config = None
        if options:
            config_params = {}
            if 'aspect_ratio' in options:
                config_params['aspect_ratio'] = options['aspect_ratio']
            if 'number_of_images' in options:
                config_params['number_of_images'] = options['number_of_images']
            if config_params:
                config = genai.types.GenerateContentConfig(**config_params)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=images + [prompt],
            config=config,
        )

        parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else None
        if not parts:
            return None
        for part in parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                return image


def convert_to_justai_response(response, return_json):
    input_token_count = response.usage_metadata.prompt_token_count
    output_token_count = (response.usage_metadata.candidates_token_count or 0) + \
                         (response.usage_metadata.thoughts_token_count or 0)
    if not return_json:
        result = response.text
    elif response.parsed:
        result = response.parsed
    elif _hit_output_limit(response):
        raise TruncatedResponseException(
            f'Gemini hit its output limit after {output_token_count} tokens, JSON is incomplete')
    else:
        result = _parse_gemini_json(response.text)
    return result, input_token_count, output_token_count


def _hit_output_limit(response) -> bool:
    """True when generation stopped on max_output_tokens instead of finishing its answer."""
    candidate = next(iter(response.candidates or []), None)
    reason = getattr(candidate, 'finish_reason', None)
    return getattr(reason, 'name', reason) == 'MAX_TOKENS'


def _parse_gemini_json(text: str) -> dict | list:
    """Parse JSON from Gemini response, tolerating markdown fences and trailing text.

    Raises GeneralException when the response contains no parseable JSON, so callers
    can treat it as a model contract violation rather than a low-level decode crash.
    """
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*\n?', '', stripped)
        stripped = re.sub(r'\n?```\s*$', '', stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return extract_json(stripped)
    except json.JSONDecodeError as e:
        raise GeneralException(f'Gemini returned non-JSON response: {stripped[:200]}') from e
