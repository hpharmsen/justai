""" Handles the GPT API and the conversation state. """
import json
import time
from pathlib import Path
from typing import Optional, Union, List, Callable

from PIL.Image import Image

from justai.models.basemodel import ImageInput
from justai.tools.cache import cached_response, cache_save
from justai.model.message import Message
from justai.models.modelfactory import ModelFactory
from justai.tools.images import crop_to_fit


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
        """ Closes any open connections in the underlying model. """
        if hasattr(self.model, 'close'):
            self.model.close()

    def __setattr__(self, name, value):
        if name not in self.__dict__ and hasattr(self, 'model') and name in self.model.model_params:
            # Not an existing property model but a model_params property. Set it in model_params
            self.model.model_params[name] = value
        else:
            # Update the property as intended
            super().__setattr__(name, value)

    def set_api_key(self, key: str):
        """ Used when using Aigent from a browser where the user has to specify a key """
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
        raise AttributeError("Model does not support cached_prompt")

    @cached_prompt.setter
    def cached_prompt(self, value):
        if hasattr(self.model, 'cached_prompt'):
            self.model.cached_prompt = value
        else:
            raise AttributeError("Model does not support cached_prompt")

    @property
    def cache_creation_input_tokens(self):
        if hasattr(self.model, 'cache_creation_input_tokens'):
            return self.model.cache_creation_input_tokens
        raise AttributeError("Model does not support cache_creation_input_tokens")
    
    @property
    def cache_read_input_tokens(self):
        if hasattr(self.model, 'cache_read_input_tokens'):
            return self.model.cache_read_input_tokens
        raise AttributeError("Model does not support cache_read_input_tokens")
        
    def reset(self):
        self.messages = []

    def add_tool(self, function: Callable, description: str|None = None, parameters: dict = None, required_parameters: list=None):
        if description is None and not self.model.supports_function_calling:
            raise NotImplementedError(f"{self.model.model_name} does not support function calling")
        if not description and not self.model.supports_automatic_function_calling:
            raise NotImplementedError(f"{self.model.model_name} does not support automatic function calling")
        tool = {'type': 'function', 'function': function, 'description': description, 'parameters': parameters,
                'required_parameters': required_parameters}
        self.tools.append(tool)
        self.functions[function.__name__] = function

    def last_token_count(self):
        return self.input_token_count, self.output_token_count, self.input_token_count + self.output_token_count

    def prompt(self, prompt: str, *, images: ImageInput = None, return_json=False, response_format=None, cached=True):
        self.raise_for_unsupported()

        start_time = time.time()
        if images and not isinstance(images, list):
            images = [images]

        response = None

        if cached:
            response = cached_response(self.model, prompt, images, self.tools, return_json, response_format)

        if response:
            result, _, _, tool_use = response
            self.input_token_count = self.output_token_count = 0
        else:
            response = self.model.prompt(prompt, images=images, tools=self.tools, return_json=return_json,
                                         response_format=response_format)
            if cached:
                cache_save(response, self.model, prompt, images, self.tools, return_json, response_format)

            result, self.input_token_count, self.output_token_count = response

        self.last_response_time = time.time() - start_time
        return result

    def chat(self, prompt: str, *, images: ImageInput = None, return_json=False, response_format=None, cached=False):
        self.raise_for_unsupported()
        if cached:
            raise NotImplementedError("Model.chat does not support cached=True. Use prompt instead.")

        start_time = time.time()
        if images and not isinstance(images, list):
            images = [images]

        response = self.model.chat(prompt, images=images, tools=self.tools, return_json=return_json,
                                   response_format=response_format)

        result, self.input_token_count, self.output_token_count = response
        self.last_response_time = time.time() - start_time
        return result
    
    async def prompt_async(self, prompt, *, images: ImageInput = None):
        # Using 'async for' to properly yield from the chat_async generator
        async for content, reasoning in self.model.prompt_async(prompt=prompt):
            yield content, reasoning

    async def chat_async(self, prompt, *, images: ImageInput = None):
        if images and not isinstance(images, list):
            images = [images]
        for word in self.model.chat_async(prompt=prompt, images=images):
            if word:
                yield word

    async def prompt_async_reasoning(self, prompt, *, images: ImageInput = None):
        self.reset()
        # Using 'async for' to properly yield from the chat_async_reasoning generator
        async for word, reasoning_content in self.chat_async_reasoning(prompt=prompt, images=images):
            yield word, reasoning_content

    async def chat_async_reasoning(self, prompt, *, images: ImageInput = None):
        """ Same as chat_async but returns the reasoning content as well
        """
        if images and not isinstance(images, list):
            images = [images]
        async for word, reasoning_content in self.model.chat_async(prompt=prompt, images=images):
            if word or reasoning_content:
                yield word, reasoning_content

    def raise_for_unsupported(self, images: ImageInput = None, return_json=False):
        if return_json and not self.model.supports_return_json:
            raise NotImplementedError(f"{self.model.model_name} does not support return_json")
        if images and not self.model.supports_image_input:
            raise NotImplementedError(f"{self.model.model_name} does not support image input")

    def token_count(self, text: str):
        return self.model.token_count(text)

    def generate_image(self, prompt: str, images: ImageInput = None, size: tuple[int, int]|None = None) -> Image:
       if not self.model.supports_image_generation:
           raise NotImplementedError(f"{self.model.model_name} does not support image generation")
       image = self.model.generate_image(prompt, images)
       if size:
           image = crop_to_fit(image, size[0], size[1])
       return image

