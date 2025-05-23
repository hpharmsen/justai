""" Handles the GPT API and the conversation state. """
import json
import re
import time
from pathlib import Path
from PIL.Image import Image

from justai.tools.cache import cached_llm_response
from justai.model.message import Message
from justai.models.modelfactory import ModelFactory


class Model:
    def __init__(self, model_name: str, **kwargs):
        
        # Model parameters
        self.model = ModelFactory.create(model_name, **kwargs)

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
        
    def __setattr__(self, name, value):
        if name not in self.__dict__ and hasattr(self, 'model') and name in self.model.model_params:
            # Not an existing property model but a model_params property. Set it in model_params
            self.model.model_params[name] = value
        else:
            # Update the property as intended
            super().__setattr__(name, value)

    @classmethod
    def from_json(cls, model_name, model_data, **kwargs):
        """ Creates an model from a json string. Usefull in stateless environments like a web page """
        model = cls(model_name, **kwargs)
        dictionary = json.loads(model_data)
        for key, value in dictionary.items():
            match key:
                case 'messages':
                    model.messages = [Message.from_dict(m) for m in value]
                case _:
                    model.__setattr__(key, value)
        return model

    def set_api_key(self, key: str):
        """ Used when using Aigent from a browser where the user has to specify a key """
        self.model.set('api_key', key)

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

    def append_messages(self, prompt: str,
                        images: [list[str] | list[bytes] | list[Image] | str | bytes | Image | None] = None):
        if images:
            if not isinstance(images, list):
                images = [images]
        else:
            images = []
        self.messages.append(Message('user', prompt, images))
        return self.messages

    def add_function(self, func: callable, description: str, parameters: dict, required_parameters=None):
        """ Adds a function to the model.
        name: name of the function
        description: description of the function
        parameters: dictionary with parameter description as key and parameter type as value """
        tool = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        re.sub(r"\W+", "", description).lower(): {
                            "type": _type.__name__ if isinstance(_type, type) else str(_type),
                            "description": description,
                        }
                        for description, _type in parameters.items()
                    },
                    "required": required_parameters or [],
                    "additionalProperties": False
                }
            }
        }
        self.tools.append(tool)
        self.functions[func.__name__] = func

    def get_messages(self) -> list[Message]:
        return self.messages[-self.message_memory:]

    def last_token_count(self):
        return self.input_token_count, self.output_token_count, self.input_token_count + self.output_token_count

    def chat(self, prompt, *, images: [list[str] | list[bytes] | list[Image] | str | bytes | Image | None] = None,
             return_json=False, response_format=None, cached=True):
        start_time = time.time()
        if images and not isinstance(images, list):
            images = [images]
        self.append_messages(prompt, images)

        model_response = cached_llm_response(self.model, self.get_messages(), self.tools, return_json=return_json,
                                             response_format=response_format, use_cache=cached)
        result, self.input_token_count, self.output_token_count, tool_use = model_response
        calls = 0  # Safety to prevent infinite loop
        while tool_use and calls < 3:
            calls += 1
            function = self.functions.get(tool_use["function_to_call"])
            if not function:
                raise ValueError(f"Function {tool_use['function_to_call']} not found")

            # Run the tool/function
            tool_use['function_result'] = function(*tool_use['function_parameters'].values())

            # Add a model specific message with the tool use result to the conversation
            self.messages.append(self.model.tool_use_message(tool_use))

            model_response = cached_llm_response(
                self.model,
                self.get_messages(),
                self.tools,
                return_json=return_json,
                response_format=response_format,
                use_cache=cached,
            )
            result, self.input_token_count, self.output_token_count, tool_use = model_response

        # Add the result to the messages
        self.messages.append(Message('assistant', str(result)))
        self.last_response_time = time.time() - start_time
        return result
    
    async def chat_async(self, prompt, *,
                         images: [list[str] | list[bytes] | list[Image] | str | bytes | Image | None] = None):
        if images and not isinstance(images, list):
            images = [images]
        self.append_messages(prompt, images)
        for word, _ in self.model.chat_async(messages=self.get_messages()):
            if word:
                yield word

    async def chat_async_reasoning(self, prompt, *,
                         images: [list[str] | list[bytes] | list[Image] | str | bytes | Image | None] = None):
        """ Same as chat_async but returns the reasoning content as well
        """
        if images and not isinstance(images, list):
            images = [images]
        self.append_messages(prompt, images)
        for word, reasoning_content in self.model.chat_async(messages=self.get_messages()):
            if word or reasoning_content:
                yield word, reasoning_content

    def after_response(self):
        # content is in messages[-1]['completion']['choices'][0]['message']['content']
        return  # Can be overridden

    def token_count(self, text: str):
        return self.model.token_count(text)
