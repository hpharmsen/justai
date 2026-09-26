"""Handles the GPT API and the conversation state."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from PIL.Image import Image

from justai.models.basemodel import ImageInput
from justai.models.modelfactory import ModelFactory
from justai.models.systemone import restore_level_keys
from justai.tools.cache import cache_save, cached_response
from justai.tools.images import crop_to_fit


def _cache_encode(result) -> str:
    """Serialise a structured result so sqlite can bind it; pydantic objects dump themselves."""
    dump = getattr(result, 'model_dump_json', None)
    return dump() if dump else json.dumps(result)


def _to_pydantic(result, response_format):
    """Convert result to a Pydantic model instance when response_format is a Pydantic class."""
    if not response_format:
        return result
    try:
        from pydantic import BaseModel as PydanticModel
    except ImportError:
        return result
    if not (isinstance(response_format, type) and issubclass(response_format, PydanticModel)):
        return result
    if isinstance(result, response_format):
        return result
    if isinstance(result, dict):
        return response_format.model_validate(result)
    if isinstance(result, str):
        return response_format.model_validate_json(result)
    return result


class Model:
    def __init__(self, model_name: str, **kwargs):

        # Model parameters
        self.model = ModelFactory.create(model_name, **kwargs)
        self.model.encapsulating_model = self

        # Parameters to save the current conversation
        self.save_dir = Path(__file__).resolve().parent / 'saves'
        self.message_memory = 20  # Number of messages to remember. Limits token usage.
        self.messages = []  # List of Message objects
        self.tools = []  # List of tools to use / functions to call
        self.functions = {}  # The actual functions to call with key the name of the function and as value the function

        self.input_token_count = 0
        self.output_token_count = 0
        self.last_response_time = 0

        self.logger = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes any open connections in the underlying model."""
        self.model.close()

    def __setattr__(self, name, value):
        if name not in self.__dict__ and hasattr(self, 'model') and name in self.model.model_params:
            # Not an existing property on Model but a model_params property. Validate then set in model_params.
            validator = self.model._VALIDATORS.get(name)
            if validator:
                validator(value)  # Raises before write on invalid value
            self.model.model_params[name] = value
            if name == 'effort' and value is not None:
                # Emit downmap/ignore warning immediately on assignment.
                _, warn = self.model.resolve_effort()
                if warn:
                    self.model._emit_effort_warning(warn)
        else:
            # Update the property as intended
            super().__setattr__(name, value)

    def set_api_key(self, key: str):
        """Used when using Aigent from a browser where the user has to specify a key"""
        self.model.set('api_key', key)

    @property
    def model_name(self):
        return self.model.model_name

    @property
    def system(self):  # This function can be overwritten by child classes to make the system message dynamic
        return self.model.system_message

    @system.setter
    def system(self, value):
        self.model.system_message = value

    @property
    def cached_prompt(self):
        if hasattr(self.model, 'cached_prompt'):
            return self.model.cached_prompt
        raise AttributeError('Model does not support cached_prompt')

    @cached_prompt.setter
    def cached_prompt(self, value):
        if hasattr(self.model, 'cached_prompt'):
            self.model.cached_prompt = value
        else:
            raise AttributeError('Model does not support cached_prompt')

    @property
    def cache_creation_input_tokens(self):
        """Tokens naar de cache geschreven in de laatste call. Nul als niets gemeten is."""
        return self.model.cache_creation_input_tokens

    @property
    def cache_read_input_tokens(self):
        """Tokens uit de cache gelezen in de laatste call. Nul als niets gemeten is."""
        return self.model.cache_read_input_tokens

    def reset(self):
        self.messages = []

    def add_tool(
        self,
        function: Callable,
        description: str | None = None,
        parameters: dict = None,
        required_parameters: list = None,
    ):
        if description is None and not self.model.supports_function_calling:
            raise NotImplementedError(f'{self.model.model_name} does not support function calling')
        if not description and not self.model.supports_automatic_function_calling:
            raise NotImplementedError(f'{self.model.model_name} does not support automatic function calling')
        tool = {
            'type': 'function',
            'function': function,
            'description': description,
            'parameters': parameters,
            'required_parameters': required_parameters,
        }
        self.tools.append(tool)
        self.functions[function.__name__] = function

    def last_token_count(self):
        return self.input_token_count, self.output_token_count, self.input_token_count + self.output_token_count

    def _call(self, call):
        """Run a provider call and keep the token counters describing that call, also when it
        raises. A response that arrived and then failed to parse was billed, and reporting the
        previous call's numbers instead would be worse than reporting none.
        """
        self.input_token_count = self.output_token_count = 0
        self.model.last_usage = None
        try:
            return call()
        except Exception:
            if self.model.last_usage:
                self.input_token_count, self.output_token_count = self.model.last_usage
            raise

    def prompt(self, prompt: str, *, images: ImageInput = None, return_json=False, response_format=None, cached=True):
        self.raise_for_unsupported(images, return_json)

        start_time = time.time()
        if images and not isinstance(images, list):
            images = [images]

        # sqlite binds strings, not dicts, so a structured result is stored serialised and
        # parsed back on a hit. Without this every write failed and cached=True was a no-op.
        structured = bool(return_json or response_format)
        response = None

        if cached:
            response = cached_response(
                self.model.model_name,
                self.model.model_params,
                self.model.system_message,
                prompt,
                images,
                self.tools,
                return_json,
                response_format,
            )

        if response:
            result, _, _ = response
            if structured:
                result = json.loads(result)
            self.input_token_count = self.output_token_count = 0
        else:
            response = self._call(
                lambda: self.model.prompt(
                    prompt, images=images, tools=self.tools, return_json=return_json, response_format=response_format
                )
            )
            if cached:
                stored = (_cache_encode(response[0]), *response[1:]) if structured else response
                cache_save(
                    stored,
                    self.model.model_name,
                    self.model.model_params,
                    self.model.system_message,
                    prompt,
                    images,
                    self.tools,
                    return_json,
                    response_format,
                )

            result, self.input_token_count, self.output_token_count = response

        self.last_response_time = time.time() - start_time
        return _to_pydantic(result, response_format)

    def classify(
        self,
        state: str | dict | list,
        options: dict | list | None = None,
        *,
        instructions: str | None = None,
        questions: dict | None = None,
        cached=True,
    ) -> dict:
        """Ask a System One model a typed question about `state` and get a dict back.

        A dict of options gives a choice, an ordered list gives a score, no options gives a
        yes/no. Pass `questions` instead to ask several at once; the result is then keyed by
        question name. Models that are not System One raise NotImplementedError.
        """
        start_time = time.time()
        # 'classify' keeps these entries away from prompt entries, which hash an argument list
        # of the same length.
        key = (self.model.model_name, self.model.model_params, 'classify', state, options, instructions, questions)

        response = cached_response(*key) if cached else None
        if response:
            # sqlite binds strings, not dicts, so the answer went in serialised.
            result = restore_level_keys(json.loads(response[0]), single=questions is None)
            self.input_token_count = self.output_token_count = 0
        else:
            response = self._call(
                lambda: self.model.classify(
                    state, options, instructions=instructions, questions=questions
                )
            )
            if cached:
                cache_save((json.dumps(response[0]), *response[1:]), *key)
            result, self.input_token_count, self.output_token_count = response

        self.last_response_time = time.time() - start_time
        return result

    def chat(self, prompt: str, *, images: ImageInput = None, return_json=False, response_format=None, cached=False):
        self.raise_for_unsupported(images, return_json)
        if cached:
            raise NotImplementedError('Model.chat does not support cached=True. Use prompt instead.')

        start_time = time.time()
        if images and not isinstance(images, list):
            images = [images]

        response = self._call(
            lambda: self.model.chat(
                prompt, images=images, tools=self.tools, return_json=return_json, response_format=response_format
            )
        )

        result, self.input_token_count, self.output_token_count = response
        self.last_response_time = time.time() - start_time
        return _to_pydantic(result, response_format)

    async def prompt_async(self, prompt, *, images: ImageInput = None):
        # Using 'async for' to properly yield from the chat_async generator
        if images and not isinstance(images, list):
            images = [images]
        async for content, reasoning in self.model.prompt_async(prompt=prompt, images=images):
            yield content, reasoning

    async def chat_async(self, prompt, *, images: ImageInput = None):
        if images and not isinstance(images, list):
            images = [images]
        async for word in self.model.chat_async(prompt=prompt, images=images):
            if word:
                yield word

    async def prompt_async_reasoning(self, prompt, *, images: ImageInput = None):
        self.reset()
        # Using 'async for' to properly yield from the chat_async_reasoning generator
        async for word, reasoning_content in self.chat_async_reasoning(prompt=prompt, images=images):
            yield word, reasoning_content

    async def chat_async_reasoning(self, prompt, *, images: ImageInput = None):
        """Same as chat_async but returns the reasoning content as well"""
        if images and not isinstance(images, list):
            images = [images]
        async for word, reasoning_content in self.model.chat_async(prompt=prompt, images=images):
            if word or reasoning_content:
                yield word, reasoning_content

    async def stream(self, messages: list[dict], tools: list[dict] | None = None):
        """Stateless streaming call. Forwards to provider's stream()."""
        async for chunk in self.model.stream(messages, tools):
            yield chunk

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool result message for the current provider."""
        return self.model.format_tool_result(tool_call_id, tool_name, result)

    def format_assistant_message(self, text: str, tool_calls=None) -> list[dict]:
        """Format an assistant message for the current provider."""
        return self.model.format_assistant_message(text, tool_calls)

    def raise_for_unsupported(self, images: ImageInput = None, return_json=False):
        if return_json and not self.model.supports_return_json:
            raise NotImplementedError(f'{self.model.model_name} does not support return_json')
        if images and not self.model.supports_image_input:
            raise NotImplementedError(f'{self.model.model_name} does not support image input')

    def token_count(self, text: str):
        return self.model.token_count(text)

    def generate_image(
        self, prompt: str, images: ImageInput = None, size: tuple[int, int] | None = None, options: dict = None
    ) -> Image:
        if not self.model.supports_image_generation:
            raise NotImplementedError(f'{self.model.model_name} does not support image generation')
        if images and not isinstance(images, list):
            images = [images]
        image = self.model.generate_image(prompt, images, size=size, options=options)
        if size:
            image = crop_to_fit(image, size[0], size[1])
        return image
