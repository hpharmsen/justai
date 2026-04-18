"""Implementation of local GGUF models via llama-cpp-python.

Feature table:
    - Async chat:       YES (streaming via sync iterator)
    - Return JSON:      YES (via response_format)
    - Structured types: NO
    - Token count:      YES (via Llama tokenizer)
    - Image support:    NO
    - Tool use:         NO

Supported parameters:
n_ctx: int (default 8192) - context window size (constructor only)
n_gpu_layers: int (default -1) - GPU layers, -1 for all (constructor only)
n_threads: int (default 4) - CPU threads (constructor only)
n_batch: int (default 512) - batch size (constructor only)
max_tokens: int (default 800) - max output tokens (per-call)
temperature: float (default 0.8) - sampling temperature (per-call)
"""

from typing import Any, AsyncGenerator

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from justai.models.basemodel import BaseModel, ImageInput


class GgufModel(BaseModel):
    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f'You are {model_name.split("/")[-1].replace(".gguf", "")}, a helpful assistant.'
        super().__init__(model_name, params, system_message)

        if Llama is None:
            raise ImportError('To use local GGUF models, install llama-cpp-python: pip install justai[llama]')

        # Capability flags
        self.supports_image_input = False
        self.supports_tool_use = False
        self.supports_function_calling = False
        self.supports_return_json = True

        # Constructor params (immutable after init)
        n_ctx = params.get('n_ctx', 8192)
        n_gpu_layers = params.get('n_gpu_layers', -1)
        n_threads = params.get('n_threads', 4)
        n_batch = params.get('n_batch', 512)

        # Per-call params (mutable via model_params)
        self.model_params['max_tokens'] = params.get('max_tokens', 800)
        if 'temperature' not in self.model_params:
            self.model_params['temperature'] = 0.8

        # Create client after params are set
        self.client = Llama(
            model_path=model_name,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # Conversation state
        self.messages = []

    def prompt(self, prompt: str, images: ImageInput = None, tools: list = None,
               return_json: bool = False, response_format=None) -> tuple[Any, int | None, int | None]:
        """Stateless prompt - resets conversation history."""
        self.messages = []
        return self.chat(prompt, images, tools, return_json, response_format)

    def chat(self, prompt: str, images: ImageInput = None, tools: list = None,
             return_json: bool = False, response_format=None) -> tuple[Any, int | None, int | None]:
        """Stateful chat - maintains conversation history."""
        self.messages.append({'role': 'user', 'content': prompt})
        messages = [{'role': 'system', 'content': self.system_message}] + self.messages

        kwargs = {
            'messages': messages,
            'temperature': self.model_params.get('temperature', 0.8),
            'max_tokens': self.model_params.get('max_tokens', 800),
        }
        if return_json or response_format:
            kwargs['response_format'] = {'type': 'json_object'}

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
        self.messages.append({'role': 'user', 'content': prompt})
        messages = [{'role': 'system', 'content': self.system_message}] + self.messages

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
