"""Implementation of local GGUF models via llama-cpp-python.

Feature table:
    - Async chat:       YES (streaming via sync iterator)
    - Return JSON:      YES (via response_format)
    - Structured types: NO
    - Token count:      YES (via Llama tokenizer)
    - Image support:    YES (requires clip_model_path parameter)
    - Tool use:         NO

Supported parameters:
n_ctx: int (default 0, uses model's training context) - context window size (constructor only)
n_gpu_layers: int (default -1) - GPU layers, -1 for all (constructor only)
n_threads: int (default 4) - CPU threads (constructor only)
n_batch: int (default 512) - batch size (constructor only)
clip_model_path: str - path to mmproj GGUF file for vision models (constructor only)
chat_handler: str - chat handler class name, e.g. 'Llava15ChatHandler' (constructor only)
max_tokens: int (default 800) - max output tokens (per-call)
temperature: float (default 0.8) - sampling temperature (per-call)
"""

import base64
import io
import os
import sys
from contextlib import contextmanager
from typing import Any, AsyncGenerator

from PIL.Image import Image

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None
    Llava15ChatHandler = None

from justai.models.basemodel import BaseModel, ImageInput


@contextmanager
def _suppress_stderr():
    """Suppress C-level stderr output from llama.cpp."""
    stderr_fd = sys.stderr.fileno()
    old_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, stderr_fd)
        os.close(old_fd)


class GgufModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name.split("/")[-1].replace(".gguf", "")}, a helpful assistant.'
        super().__init__(model_name, params, system_message)

        if Llama is None:
            raise ImportError('To use local GGUF models, install llama-cpp-python: pip install justai[llama]')

        # Constructor params (immutable after init)
        clip_model_path = params.pop('clip_model_path', None)
        chat_handler_name = params.pop('chat_handler', None)
        n_ctx = params.get('n_ctx', 0)
        n_gpu_layers = params.get('n_gpu_layers', -1)
        n_threads = params.get('n_threads', 4)
        n_batch = params.get('n_batch', 512)

        # Capability flags
        self.supports_image_input = clip_model_path is not None
        self.supports_tool_use = False
        self.supports_function_calling = False
        self.supports_return_json = True

        # Per-call params (mutable via model_params)
        self.model_params['max_tokens'] = params.get('max_tokens', 800)
        if 'temperature' not in self.model_params:
            self.model_params['temperature'] = 0.8

        # Set up vision chat handler if clip model is provided
        chat_handler = None
        if clip_model_path:
            handler_cls = Llava15ChatHandler
            if chat_handler_name:
                import llama_cpp.llama_chat_format as fmt
                handler_cls = getattr(fmt, chat_handler_name)
            with _suppress_stderr():
                chat_handler = handler_cls(clip_model_path=clip_model_path)

        # Create client after params are set (suppress C-level stderr noise)
        with _suppress_stderr():
            self.client = Llama(
                model_path=model_name,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                chat_handler=chat_handler,
                verbose=False,
            )

        # Conversation state
        self.messages = []

    @staticmethod
    def _image_to_data_uri(image) -> str:
        """Convert a single image (PIL, bytes, or URL string) to a base64 data URI."""
        if isinstance(image, Image):
            buf = io.BytesIO()
            image.save(buf, format=image.format or 'PNG')
            data = base64.b64encode(buf.getvalue()).decode()
            return f'data:image/png;base64,{data}'
        if isinstance(image, bytes):
            data = base64.b64encode(image).decode()
            return f'data:image/png;base64,{data}'
        if isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
            return image
        if isinstance(image, str):
            return image  # Assume already a data URI or file path
        raise ValueError(f'Unsupported image type: {type(image)}')

    def _build_user_content(self, prompt: str, images: ImageInput = None) -> str | list[dict]:
        """Build user message content, with image blocks if images are provided."""
        if not images:
            return prompt
        if not isinstance(images, list):
            images = [images]
        content = [{'type': 'image_url', 'image_url': {'url': self._image_to_data_uri(img)}} for img in images]
        content.append({'type': 'text', 'text': prompt})
        return content

    def prompt(self, prompt: str, images: ImageInput = None, tools: list = None,
               return_json: bool = False, response_format=None) -> tuple[Any, int | None, int | None]:
        """Stateless prompt - resets conversation history."""
        self.messages = []
        return self.chat(prompt, images, tools, return_json, response_format)

    def chat(self, prompt: str, images: ImageInput = None, tools: list = None,
             return_json: bool = False, response_format=None) -> tuple[Any, int | None, int | None]:
        """Stateful chat - maintains conversation history."""
        content = self._build_user_content(prompt, images)
        self.messages.append({'role': 'user', 'content': content})
        messages = [{'role': 'system', 'content': self.system_message}] + self.messages

        kwargs = {
            'messages': messages,
            'temperature': self.model_params.get('temperature', 0.8),
            'max_tokens': self.model_params.get('max_tokens', 800),
        }
        if return_json or response_format:
            kwargs['response_format'] = {'type': 'json_object'}

        with _suppress_stderr():
            output = self.client.create_chat_completion(**kwargs)

        result = output['choices'][0]['message']['content']
        self.messages.append({'role': 'assistant', 'content': result})

        return result, output['usage']['prompt_tokens'], output['usage']['completion_tokens']

    async def prompt_async(self, prompt: str, images: ImageInput = None) -> AsyncGenerator[tuple[str, str], None]:
        """Stateless streaming prompt."""
        self.messages = []
        async for content, reasoning in self.chat_async(prompt, images):
            yield content, reasoning

    async def chat_async(self, prompt: str, images: ImageInput = None) -> AsyncGenerator[tuple[str, str], None]:
        """Stateful streaming chat via sync iterator."""
        content = self._build_user_content(prompt, images)
        self.messages.append({'role': 'user', 'content': content})
        messages = [{'role': 'system', 'content': self.system_message}] + self.messages

        with _suppress_stderr():
            stream = self.client.create_chat_completion(
                messages=messages,
                temperature=self.model_params.get('temperature', 0.8),
                max_tokens=self.model_params.get('max_tokens', 800),
                stream=True,
            )

        full_response = ''
        for chunk in stream:
            delta = chunk['choices'][0].get('delta', {})
            content = delta.get('content', '')
            if content:
                full_response += content
                yield content, ''

        self.messages.append({'role': 'assistant', 'content': full_response})

    def token_count(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(self.client.tokenize(text.encode('utf-8')))
