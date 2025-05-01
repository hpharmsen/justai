""" Implementation of the Anthropic models. 
Uses the Google AI library. TODO: Add support for Google Vertex library one day

Feature table:
    - Async chat:       YES (1)
    - Return JSON:      YES
    - Structured types: YES, via Python type definition
    - Token counter:    YES
    - Image support:    YES 
    - Tool use:         NO (not yet)

Models:
gemini-1.5-flash -> gemini-1.5-flash-001
gemini-1.5-pro -> gemini-1.5-pro-001

Supported parameters: 
max_tokens: int (default: None)
temperature: float (default:None)
stop_sequences: list[str] (default: None)
candidate_count: int (default: None)
topP: float (default: None)
topK: int (default: None)

(1) In contrast to Model.chat, Model.chat_async cannot return json and does not return input and output token counts

"""
import json
import os
import absl.logging
from contextlib import contextmanager
import time

from dotenv import dotenv_values
import google
import google.generativeai as genai

from justai.model.message import Message
from justai.models.basemodel import BaseModel
from justai.tools.display import ERROR_COLOR, color_print


class GoogleModel(BaseModel):

    def __init__(self, model_name: str, params: dict):
        system_message = f"You are {model_name}, a large language model trained by Google."
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = params.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or dotenv_values()["GOOGLE_API_KEY"]
        if api_key:
            genai.configure(api_key=api_key)
        else:
            color_print("No Google API key found. Create one at https://aistudio.google.com/app/apikey and " +
                        "set it in the .env file like GOOGLE_API_KEY=here_comes_your_key.", color=ERROR_COLOR)

        # Client
        self.client = genai.GenerativeModel(model_name, system_instruction=system_message)

    def chat(self, messages: list[Message], tools, return_json: bool, response_format, max_retries=None, log=None) \
            -> tuple[[str | object], int, int]:
        if tools:
            raise NotImplementedError("tools are not supported (yet)")

        google_messages = transform_messages(messages, return_json)
        chat = self.client.start_chat(history=google_messages[:-1])

        with temporary_verbosity(absl.logging.WARNING):
            while True:
                try:
                    response = chat.send_message(
                        content=google_messages[-1]['parts'], generation_config=self.generation_config(return_json, response_format)
                    )
                    break
                except google.api_core.exceptions.ResourceExhausted:
                    print('Gemini exhausted, retrying...')
                    time.sleep(1)

        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.total_token_count - input_tokens
        if return_json or response_format:
            response = json.loads(response.text)
        else:
            response = response.text
        return response, input_tokens, output_tokens

    def chat_async(self, messages: list[Message]) -> [str, str]:

        google_messages = transform_messages(messages, False) # Was: messages[:-1]
        try:
            # Initialize the streaming response
            response = self.client.generate_content(google_messages, stream=True,
                                                    generation_config=self.generation_config(False))

            # Collect the streamed parts
            full_response = ''
            for chunk in response:
                full_response += chunk.text
                yield chunk.text, None   # 2nd parameter is reasoning_content. Not available yet for Gemini yet

            # Ensure the iteration completes
            response.resolve()

        except google.generativeai.types.generation_types.IncompleteIterationError:
            print("Please let the response complete iteration before accessing the final accumulated attributes.")

    def token_count(self, text: str) -> int:
        return self.client.count_tokens(text).total_tokens

    def generation_config(self, return_json=False, response_format=None) -> genai.types.GenerationConfig:
        return genai.types.GenerationConfig(
            response_schema=response_format,
            # specifies format of the JSON requested if response_mime_type is `application/json`
            response_mime_type='application/json' if return_json or response_format else 'text/plain',
            **self.model_params
        )


def transform_messages(messages: list[Message], return_json: bool) -> list[dict]:
    return [google_message(msg, return_json) for msg in messages]


def google_message(msg: Message, return_json) -> dict:
    return {
        'role': 'model' if msg.role == 'assistant' else 'user',
        'parts': [Message.to_pil_image(img) for img in msg.images] + [msg.content]
    }


@contextmanager
def temporary_verbosity(level):
    # Sla het huidige logniveau op
    original_level = absl.logging.get_verbosity()
    try:
        # Wijzig naar het nieuwe logniveau
        absl.logging.set_verbosity(level)
        yield
    finally:
        # Herstel het oorspronkelijke logniveau
        absl.logging.set_verbosity(original_level)