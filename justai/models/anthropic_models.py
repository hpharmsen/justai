""" Implementation of the Anthropic models. 

Feature table:
    - Async chat:       YES (1)
    - Return JSON:      YES
    - Structured types: NO
    - Token count:      YES
    - Image support:    YES 
    
Models:
Claude 3 Opus:	 claude-3-opus-20240229
Claude 3 Sonnet: claude-3-5-sonnet-20240620
Claude 3 Haiku:  claude-3-haiku-20240307

Supported parameters:
max_tokens: int (default 800)
temperature: float (default 0.8)

(1) In contrast to Agent.chat, Agent.chat_async cannot return json and does not return input and output token counts

"""

import json
import os

import anthropic
from dotenv import dotenv_values

from justai.agent.message import Message
from justai.models.model import Model, OverloadedException, identify_image_format_from_base64
from justai.tools.display import ERROR_COLOR, color_print


class AnthropicModel(Model):
    def __init__(self, model_name: str, params: dict):
        system_message = f"You are {model_name}, a large language model trained by Anthropic."
        super().__init__(model_name, params, system_message)

        # Authentication
        if "ANTHROPIC_API_KEY" in params:
            api_key = params["ANTHROPIC_API_KEY"]
            del params["ANTHROPIC_API_KEY"]
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY") or dotenv_values()["ANTHROPIC_API_KEY"]
        if not api_key:
            color_print(
                "No Anthropic API key found. Create one at https://console.anthropic.com/settings/keys and "
                + "set it in the .env file like ANTHROPIC_API_KEY=here_comes_your_key.",
                color=ERROR_COLOR,
            )

        # Client
        if params.get("async"):
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

        # Required model parameters
        if "max_tokens" not in params:
            params["max_tokens"] = 800

    def chat(
        self, messages: list[Message], return_json: bool, response_format, max_retries=None, log=None
    ) -> tuple[[str | object], int, int]:
        if response_format:
            raise NotImplementedError("Anthropic does not support response_format")

        antr_messages = transform_messages(messages, return_json)
        try:
            message = self.client.messages.create(
                model=self.model_name, system=self.system_message, messages=antr_messages, **self.model_params
            )
        except anthropic.InternalServerError as e:
            raise OverloadedException(e)

        response_str = message.content[0].text
        if return_json:
            response_str = response_str.split("</json>")[0]  # !!
            try:
                response = json.loads(response_str, strict=False)
            except json.decoder.JSONDecodeError:
                print("ERROR DECODING JSON, RESPONSE:", response_str)
                response = response_str
        else:
            response = response_str
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        return response, input_tokens, output_tokens

    def chat_async(self, messages: list[Message]) -> str:
        try:
            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.model_params["max_tokens"],
                temperature=self.model_params["temperature"],
                system=self.system_message,
                messages=transform_messages(messages, return_json=False),
                stream=True,
            )
        except anthropic.InternalServerError as e:
            raise OverloadedException(e)

        for event in stream:
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                yield event.delta.text

    def token_count(self, text: str) -> int:
        tokenizer = self.client.get_tokenizer()
        encoded_text = tokenizer.encode(text)
        return len(encoded_text.ids)

def transform_messages(messages: list[Message], return_json: bool) -> list[dict]:
    # Anthropic does not allow messages to start with an assistant message
    msgs = messages[next(i for i, message in enumerate(messages) if message.role != "system") :]

    if msgs and return_json:
        msgs += [Message("assistant", "<json>")]
    result = [create_anthropic_message(msg) for msg in msgs]
    return result


def create_anthropic_message(message: Message):
    content = []
    for img in message.images:
        base64img = Message.to_base64_image(img)
        mime_type= identify_image_format_from_base64(base64img)
        content += [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64img,
                },
            }
        ]
    content += [{"type": "text", "text": message.content}]
    return {"role": message.role, "content": content}
