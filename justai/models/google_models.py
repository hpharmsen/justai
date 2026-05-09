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
import os
from io import BytesIO
from typing import Any, AsyncGenerator

from PIL import Image
from dotenv import dotenv_values
from google import genai

from justai.model.message import Message
from justai.model.model import ImageInput
from justai.models.basemodel import BaseModel, DEFAULT_TIMEOUT, StreamChunk, ToolCallRequest
from justai.tools.display import ERROR_COLOR, color_print
from justai.tools.images import to_pil_image

class GoogleModel(BaseModel):

    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by Google."
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = params.get("GEMINI_API_KEY") or params.get("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or \
                  os.getenv("GOOGLE_API_KEY") or dotenv_values()["GEMINI_API_KEY"] or dotenv_values()["GOOGLE_API_KEY"]
        if not api_key:
            color_print("No Google API key found. Create one at https://aistudio.google.com/app/apikey and " +
                        "set it in the .env file like GOOGLE_API_KEY=here_comes_your_key.", color=ERROR_COLOR)

        # Client (Google uses milliseconds for timeout)
        timeout_ms = int(params.get('timeout', DEFAULT_TIMEOUT) * 1000)
        http_options = genai.types.HttpOptions(timeout=timeout_ms)
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.chat_session = None  # Google uses this to keep track of the chat

        # Diversions from the features that are supported or not supported by default
        self.supports_function_calling = True
        self.supports_automatic_function_calling = True
        self.supports_image_generation = True

    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) -> str | object:
        if isinstance(images, str):
            images = [images]
        opened_images = [to_pil_image(img) for img in images] if images else []
        if opened_images:
            prompt = [prompt] + opened_images
        if tools and isinstance(tools[0], dict):
            tools = [tool['function'] for tool in tools]
        params = {**self.model_params}
        if response_format or return_json:
            # Structured output requires enough tokens to complete the JSON.
            # Truncated JSON is always useless, so enforce a reasonable minimum.
            MIN_STRUCTURED_TOKENS = 16384
            if params.get('max_output_tokens', 0) < MIN_STRUCTURED_TOKENS:
                params['max_output_tokens'] = MIN_STRUCTURED_TOKENS
        config = genai.types.GenerateContentConfig(system_instruction=self.system_message, tools=tools,
                                                   **params)
        if return_json:
            config.response_mime_type = "application/json"
        if response_format:
            config.response_mime_type = "application/json"
            config.response_schema = response_format
        response = self.client.models.generate_content(model=self.model_name, contents=prompt,
                                                       config=config)
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
        response = self.chat_session.send_message(message=prompt)
        return convert_to_justai_response(response, return_json)

    async def prompt_async(self, prompt: str, images: list[ImageInput] = None) -> AsyncGenerator[tuple[str, str], None]:
        if images:
            raise NotImplementedError('google_model. ..._async does not support images. Use prompt() instead')

        config = genai.types.GenerateContentConfig(system_instruction=self.system_message)
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        async for chunk in stream:
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
            **self.model_params,
        )

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        )

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

    def generate_image(self, prompt, images: ImageInput, options: dict = None):
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
    result = response.text if not return_json else response.parsed if response.parsed else json.loads(response.text)
    return result, input_token_count, output_token_count
