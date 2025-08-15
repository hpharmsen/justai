""" Implementation of the Google models.
https://ai.google.dev/gemini-api/docs/migrate

Feature table:
    - Async chat:       YES (1)
    - Return JSON:      YES
    - Structured types: YES, via Python type definition
    - Token counter:    YES
    - Image support:    YES 
    - Tool use:         NO (not yet)

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
from typing import Any, AsyncGenerator

import absl.logging
from contextlib import contextmanager
import time

from dotenv import dotenv_values
#import google
from google import genai

from justai.model.message import Message
from justai.model.model import ImageInput
from justai.models.basemodel import BaseModel
from justai.tools.cache import recursive_hash, CacheDB
from justai.tools.display import ERROR_COLOR, color_print


class GoogleModel(BaseModel):

    def __init__(self, model_name: str, params: dict):
        system_message = f"You are {model_name}, a large language model trained by Google."
        super().__init__(model_name, params, system_message)

        # Authentication
        api_key = params.get("GEMINI_API_KEY") or params.get("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or \
                  os.getenv("GOOGLE_API_KEY") or dotenv_values()["GEMINI_API_KEY"] or dotenv_values()["GOOGLE_API_KEY"]
        if not api_key:
            color_print("No Google API key found. Create one at https://aistudio.google.com/app/apikey and " +
                        "set it in the .env file like GOOGLE_API_KEY=here_comes_your_key.", color=ERROR_COLOR)

        # Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.chat_session = None  # Google uses this to keep track of the chat


    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) -> str | object:
        if isinstance(images, str):
            images = [images]
        opened_images = [Message.to_pil_image(img) for img in images] if images else []
        if opened_images:
            prompt = [prompt] + opened_images
        config = {}
        if return_json:
            config["response_mime_type"] = "application/json"
        if response_format:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_format
        response = self.client.models.generate_content(model=self.model_name, contents=prompt,
                                                       config=config)
        return convert_to_justai_response(response, return_json or response_format)

    def chat(self, messages: list[Message], tools: list, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None, dict|None]:
        if not self.chat_session:
            self.chat_session = self.client.chats.create(model="gemini-2.0-flash")

        assert len(messages) > 0, 'google_model.chat requires at least one message'
        assert not return_json, 'google_model.chat does not support return_json. Use prompt() instead'
        assert not response_format, 'google_model.chat does not support response_format. Use prompt() instead'
        assert not hasattr(messages[-1], 'images'), 'google_model.chat does not support image messages. Use prompt() instead'
        response = self.chat_session.send_message(message=messages[-1].content)
        return convert_to_justai_response(response, return_json)

    async def prompt_async(self, prompt: str) -> AsyncGenerator[tuple[str, str], None]:
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text, ''

    def chat_async(self, messages: list[Message]) -> AsyncGenerator[tuple[str, str], None]:
        print('google_model.chat_async is not supported. Use prompt_async() instead.')
        raise NotImplementedError

    def token_count(self, text: str) -> int:
        response = self.client.models.count_tokens(model=self.model_name, contents=text)
        return response.total_tokens


def transform_messages(messages: list[Message], return_json: bool) -> list[dict]:
    return [google_message(msg, return_json) for msg in messages]


def google_message(msg: Message, return_json) -> dict:
    return {
        'role': 'model' if msg.role == 'assistant' else 'user',
        'parts': [Message.to_pil_image(img) for img in msg.images] + [msg.content]
    }


def convert_to_justai_response(response, return_json):
    input_token_count = response.usage_metadata.prompt_token_count
    output_token_count = (response.usage_metadata.candidates_token_count or 0) + \
                         (response.usage_metadata.thoughts_token_count or 0)
    tool_use = None  # TODO: implement
    #result = json.loads(response.text) if return_json else response.text
    result = response.text if not return_json else response.parsed if response.parsed else json.loads(response.text)
    return result, input_token_count, output_token_count, tool_use


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