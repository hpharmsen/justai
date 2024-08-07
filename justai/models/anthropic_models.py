import base64
import json
import os

import anthropic
import httpx
from dotenv import load_dotenv

from justai.agent.message import Message
from justai.models.model import Model, OverloadedException
from justai.tools.display import ERROR_COLOR, color_print


# Models:
# Claude 3 Opus	    claude-3-opus-20240229
# Claude 3 Sonnet	claude-3-5-sonnet-20240620
# Claude 3 Haiku	claude-3-haiku-20240307
#
# Supported parameters:
# max_tokens: int (default 800)
# temperature: float (default 0.8)
#
# Asynchronous use:
# Call Agent.chat_async(messages: list[dict]) -> str
# In contrast to Agent.chat, Agent.chat_async cannot return json and does not return input and output token counts

class AnthropicModel(Model):

    def __init__(self, model_name: str, params: dict):
        """ Model implemention should create attributes for all supported parameters """
        system_message = f"You are {model_name}, a large language model trained by Anthropic."
        super().__init__(model_name, params, system_message)

        # Authentication
        if "ANTHROPIC_API_KEY" in params:
            api_key = params["ANTHROPIC_API_KEY"]
        else:
            if not os.getenv("ANTHROPIC_API_KEY"):
                load_dotenv()  # Load the .env file into the environment
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            color_print("No Anthropic API key found. Create one at https://console.anthropic.com/settings/keys and " +
                        "set it in the .env file like ANTHROPIC_API_KEY=here_comes_your_key.", color=ERROR_COLOR)

        # Client
        if params.get('async'):
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

        # Model specific parameters
        self.model_params['max_tokens'] = params.get('max_tokens', 800)
        self.model_params['temperature'] = params.get('temperature', 0.8)

    def chat(self, messages: list[Message], return_json: bool, response_format, max_retries=None, log=None) \
             -> tuple[[str | object], int, int]:
        if response_format:
            raise NotImplementedError("Anthropic does not support response_format")

        antr_messages = self.transform_messages(messages, return_json)
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.model_params['max_tokens'],
                temperature=self.model_params['temperature'],
                system=self.system_message,
                messages=antr_messages
            )
        except anthropic.InternalServerError as e:
            raise OverloadedException(e)

        response_str = message.content[0].text
        if return_json:
            response_str = response_str.split('</json>')[0]  #!!
            try:
                response = json.loads(response_str, strict=False)
            except json.decoder.JSONDecodeError:
                print('ERROR DECODING JSON, RESPONSE:', response_str)
        else:
            response = response_str
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        return response, input_tokens, output_tokens

    def chat_async(self, messages: list[Message]) -> str:

        messages = self.transform_messages(messages, return_json=False)

        try:
            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.model_params['max_tokens'],
                temperature=self.model_params['temperature'],
                system=self.system_message,
                messages=messages,
                stream=True
            )
        except anthropic.InternalServerError as e:
            raise OverloadedException(e)

        for event in stream:
            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                yield event.delta.text

    def token_count(self, text: str) -> int:
        return -1  # Not implemented

    def transform_messages(self, messages: list[Message], return_json: bool) -> list[dict]:
        # Anthropic does not allow messages to start with an assistant message
        msgs = [msg for msg in messages if msg.role != 'system']
        if msgs and msgs[0].role == 'assistant':
            msgs = msgs[1:]
        if msgs and return_json:
            msgs += [Message('assistant', '<json>')]
        result = [create_anthropic_message(msg) for msg in msgs]
        return result


def create_anthropic_message(message: Message):
    content = [{"type": "text", "text": message.content}]
    if message.image:
        if message.type == 'image_url':
            img = httpx.get(message.image).content
        else:
            img = message.image
        img_base64 = base64.b64encode(img).decode("utf-8")
        content = [
                      {
                          "type": "image",
                          "source": {
                              "type": "base64",
                              "media_type": "image/jpeg",
                              "data": img_base64,
                          }
                      }
                  ] + content
    return {"role": message.role, "content": content}
