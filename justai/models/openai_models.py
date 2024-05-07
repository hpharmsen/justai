import json
import os

import openai
import tiktoken
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, RateLimitError, OpenAI
from openai._types import NOT_GIVEN

from justai.tools.display import color_print, ERROR_COLOR, DEBUG_COLOR1, DEBUG_COLOR2, SYSTEM_COLOR
from justai.models.model import Model


class OpenAIModel(Model):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model trained by OpenAI."
        super().__init__(model_name, params, system_message)

        # Authentication
        if "OPENAI_API_KEY" in params:
            openai.api_key = params["OPENAI_API_KEY"]
        else:
            if not os.getenv("OPENAI_API_KEY"):
                load_dotenv()  # Load the .env file into the environment
            openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            color_print("No OpenAI API key found. Create one at https://platform.openai.com/account/api-keys and " +
                        "set it in the .env file like OPENAI_API_KEY=here_comes_your_key.", color=ERROR_COLOR)
        self.client = OpenAI()

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
        # Use carefully and ensure that you have reasonable settings for max_tokens and stop.
        self.model_params['n'] = params.get('n', 1)

        # Up to 4 sequences where the API will stop generating further tokens.
        # The returned text will not contain the stop sequence.
        # Example: [" Human:", " AI:"]
        self.model_params['stop'] = params.get('stop')

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

    def chat(self, messages: list[dict], return_json: bool, use_cache: bool = False, max_retries=None):
        if max_retries is None:
            max_retries = 3

        if self.debug:
            color_print("\nRunning completion with these messages", color=DEBUG_COLOR1)
            [color_print(m, color=DEBUG_COLOR1) for m in messages if hasattr(m, 'text')]
            print()

        last_error = None
        for _ in range(max_retries):
            try:
                completion = self.completion(messages, return_json)
                message_text = completion.choices[0].message.content
                input_token_count = completion.usage.prompt_tokens
                output_token_count = completion.usage.completion_tokens
                break
            except APIConnectionError as e:
                color_print("Connection error.", color=SYSTEM_COLOR)
                last_error = e
            except APIError as e:
                color_print("API error", color=SYSTEM_COLOR)
                last_error = e
            except RateLimitError as e:
                color_print(f"{self.model_name} is overloaded", color=SYSTEM_COLOR)
                last_error = e
        else:
            print('Too many errors. Aborting.')
            raise last_error

        if message_text.startswith('```json'):
            message_text = message_text[7:-3]
        if self.debug:
            color_print(f"{message_text}", color=DEBUG_COLOR2)

        text = json.loads(message_text) if return_json else message_text
        return text, input_token_count, output_token_count
    
    def chat_async(self, messages: list[dict]):
        for item in self.completion(messages, stream=True):
            if hasattr(item.choices[0].delta, "content"):
                yield item.choices[0].delta.content
               
    def completion(self, messages: list[dict], return_json: bool = False, stream: bool=False):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.model_params['temperature'],
            max_tokens=self.model_params['max_tokens'],
            n=self.model_params['n'],
            top_p=self.model_params['top_p'],
            frequency_penalty=self.model_params['frequency_penalty'],
            presence_penalty=self.model_params['presence_penalty'],
            stop=self.model_params['stop'],
            response_format={"type": "json_object"} if return_json else NOT_GIVEN,
            stream = stream
        )

    def token_count(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))
