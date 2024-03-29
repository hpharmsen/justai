""" Handles the GPT API and the conversation state. """
import json
import os
import re
from pathlib import Path

import openai
import tiktoken
from openai import APIConnectionError, APIError, RateLimitError, OpenAI
from openai._types import NOT_GIVEN
from dotenv import load_dotenv

from justai.tools.display import print_message, color_print, SYSTEM_COLOR, ERROR_COLOR, DEBUG_COLOR2, DEBUG_COLOR1
from justai.agent.message import Message

BASE_SYSTEM = "You are ChatGPT, a large language model trained by OpenAI."


class Agent:
    def __init__(self, model: str, **kwargs):
        # Authentication
        if not os.getenv("OPENAI_API_KEY"):
            load_dotenv()  # Load the .env file into the environment
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            color_print("No OpenAI API key found. Create one at https://platform.openai.com/account/api-keys and " +
                        "set it in the .env file like OPENAI_API_KEY=here_comes_your_key.", color=ERROR_COLOR)
        self.client = OpenAI()
        self.functions = {}  # Callable GPT functions
        self.return_type = None  # Structure of the data of the function called by the model. Default is text.

        self.system_message = BASE_SYSTEM

        # Model parameters
        self.model = model

        # The maximum number of tokens to generate in the completion.
        # Defaults to 16
        # The token count of your prompt plus max_tokens cannot exceed the model's context length.
        # Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
        self.max_tokens = kwargs.get('max_tokens', 800)

        # What sampling temperature to use, between 0 and 2.
        # Higher values like 0.8 will make the output more random, while lower values like 0.2
        # will make it more focused and deterministic.
        # We generally recommend altering this or top_p but not both
        # Defaults to 1
        self.temperature = kwargs.get('temperature', 0.5)

        # An alternative to sampling with temperature, called nucleus sampling,
        # where the model considers the results of the tokens with top_p probability mass.
        # So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        # We generally recommend altering this or temperature but not both.
        # Defaults to 1
        self.top_p = kwargs.get('top_p', 1)

        # How many completions to generate for each prompt.
        # Because this parameter generates many completions, it can quickly consume your token quota.
        # Use carefully and ensure that you have reasonable settings for max_tokens and stop.
        self.n = kwargs.get('n', 1)

        # Up to 4 sequences where the API will stop generating further tokens.
        # The returned text will not contain the stop sequence.
        # Example: [" Human:", " AI:"]
        self.stop = kwargs.get('stop')

        # Number between -2.0 and 2.0.
        # Positive values penalize new tokens based on whether they appear in the text so far,
        # increasing the model's likelihood to talk about new topics.
        # Defaults to 0
        self.presence_penalty = kwargs.get('presence_penalty', 0)

        # Number between -2.0 and 2.0.
        # Positive values penalize new tokens based on their existing frequency in the text so far,
        # decreasing the model's likelihood to repeat the same line verbatim.
        # Defaults to 0
        self.frequency_penalty = kwargs.get('frequency_penalty', 0)

        # Parameters to save the current conversation
        self.name = ''  # Name of the current conversation
        self.save_dir = Path(__file__).resolve().parent / 'saves'
        self.save_dir.mkdir(exist_ok=True)
        self.message_memory = 20  # Number of messages to remember. Limits token usage.
        self.messages = []

        self._last_token_count = 0
        self.debug = False

    @classmethod
    def from_json(cls, s, *args, **kwargs):
        agent = cls(*args, **kwargs)
        dictionary = json.loads(s)
        for key, value in dictionary.items():
            match key:
                case 'save_dir':
                    agent.save_dir = Path(value)
                case 'messages':
                    agent.messages = [Message.from_dict(m) for m in value]
                case _:
                    setattr(agent, key, value)
        return agent

    def set_api_key(self, key: str):
        """ Used when using Aigent from a browser where the user has to specify a key """
        openai.api_key = key

    def system(self):  # This function can be overwritten by child classes to make the system message dynamic
        return self.system_message

    def reset(self):
        self.name = ''
        self.messages = []

    def get_messages(self):
        result = [{'role': 'system', 'content': self.system()}]
        for m in self.messages[-self.message_memory:]:
            message = {'role': m.role, 'content': m.text}
            result.append(message)
        return result

    def last_token_count(self):
        return self._last_token_count

    def set_return_type(self, model):
        self.return_type = [{
                    "name": model.__name__,
                    "description": model.__doc__,
                    "parameters": model.json_schema()}]

    def chat(self, prompt, add_to_messages=True, return_json=False):

        def chat_completion_request(function_call="auto"):
            if self.debug:
                color_print(f"\nRunning completion with these messages", color=DEBUG_COLOR1)
                for message in self.messages:
                    if hasattr(message, 'text'):
                        color_print(f"{message}", color=DEBUG_COLOR1)
                if self.functions:
                    color_print(f"And these functions", color=DEBUG_COLOR1)
                    for function in self.functions.values():
                        color_print(f"{function.function_name}({function.parameters})", color=DEBUG_COLOR1)
                print()

            messages = self.get_messages()
            last_error = None
            for _ in range(3):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        n=self.n,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop,
                        response_format={"type": "json_object"} if return_json else NOT_GIVEN,
                    )
                    if self.debug and hasattr(completion.choices[0], 'text'):
                        color_print(f"{completion.choices[0].text}", color=DEBUG_COLOR2)
                    self._last_token_count = completion.usage.total_tokens
                    return completion
                except APIConnectionError as e:
                    color_print("Connection error.", color=SYSTEM_COLOR)
                    last_error = e
                except APIError as e:
                    color_print("API error", color=SYSTEM_COLOR)
                    last_error = e
                except RateLimitError as e:
                    color_print(f"{self.model} is overloaded", color=SYSTEM_COLOR)
                    last_error = e
            print('Too many errors. Aborting.')
            raise last_error

        if self.messages and not self.name:
            self.name = re.sub(r'\W+', '', self.messages[0].text).replace(' ', '_')[:20]
        self.messages += [Message('user', prompt)]

        completion = chat_completion_request()
        message_text = completion.choices[0].message.content
        if message_text.startswith('```json'):
            message_text = message_text[7:-3]
            message = Message('assistant', json.loads(message_text))
        else:
            message = Message('assistant', message_text)
        if add_to_messages:
            self.messages += [message]

        return json.loads(message.text) if return_json else message.text

    def after_response(self):
        # content is in messages[-1]['completion']['choices'][0]['message']['content']
        return  # Can be overridden

    def save(self, name=None):
        if name:
            self.name = name

        assert self.name
        with open((self.save_dir / self.name).with_suffix('.txt'), "w") as f:
            f.write(f"system: {self.system()}\n")
            for message in self.messages:
                f.write(f"{message.role}: {message.text}\n")

    def load(self, name):
        def save_message(msg):
            if msg.role == 'system':
                self.system_message = msg.text
            else:
                self.messages += [msg]

        self.messages = []
        self.name = name
        if not name.endswith('.txt'):
            name += '.txt'
        filename = self.save_dir / name
        if not os.path.isfile(filename):
            color_print(f"New conversation:  {filename}", color=SYSTEM_COLOR)
            return
        with open(filename, "r") as f:
            message = Message()
            assert not message
            for line in f.readlines():
                line = line[:-1]
                try:
                    role, text = line.split(': ', 1)
                except ValueError:
                    message.text += '\n' + line
                    continue
                if role in ['system', 'user', 'assistant', 'function']:
                    if message:
                        save_message(message)
                    message = Message(role=role, text_or_json=text)
                else:
                    message.text += '\n' + line
            if message:
                save_message(message)
        print_message(Message('system', self.system()), 'system')
        for message in self.messages:
            print_message(message.text, message.role)

    def file_input(self, filename):
        with open(filename, "r") as f:
            prompt = f.read()
        self.chat(prompt)

    def dumps(self) -> str:
        data = {}
        for key, value in self.__dict__.items():
            if value is not None:
                try:
                    json.dumps(value)
                    data[key] = value
                except (TypeError, ValueError):
                    match key:
                        case 'save_dir':
                            data[key] = str(value)
                        case 'messages':
                            data[key] = [message.to_dict() for message in value]
        return json.dumps(data, indent=2)

    def token_count(self, text: str):
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))
