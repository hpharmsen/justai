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
    max_output_tokens= 400,
    top_k= 2,
    top_p= 0.5,
    temperature= 0.5,
    response_mime_type= 'application/json',
    stop_sequences= ['\n'],
    seed=42,

(1) In contrast to Model.chat, Model.chat_async cannot return json and does not return input and output token counts

"""
import json
import os
from io import BytesIO
from typing import Any, AsyncGenerator

from PIL import Image
from dotenv import dotenv_values
from google import genai

from justai.model.message import Message
from justai.model.model import ImageInput
from justai.models.basemodel import BaseModel
from justai.tools.display import ERROR_COLOR, color_print
from justai.tools.images import to_pil_image


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
        self.chat_session = None  # Google uses this to keep track of the chat

        # Diversions from the features that are supported or not supported by default
        self.supports_function_calling = True
        self.supports_automatic_function_calling = True
        self.supports_image_generation = True

    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) -> str | object:
        if isinstance(images, str):
            images = [images]
        opened_images = [Message.to_pil_image(img) for img in images] if images else []
        if opened_images:
            prompt = [prompt] + opened_images
        if tools and isinstance(tools[0], dict):
            tools = [tool['function'] for tool in tools]
        config = genai.types.GenerateContentConfig(system_instruction=self.system_message, tools=tools,
                                                   **self.model_params)
        if return_json:
            config.response_mime_type = "application/json"
        if response_format:
            config.response_mime_type = "application/json"
            config.response_schema = response_format
        response = self.client.models.generate_content(model=self.model_name, contents=prompt,
                                                       config=config)
        return convert_to_justai_response(response, return_json or response_format)

    def chat(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None]:

        if return_json:
            raise NotImplementedError('google_model.chat does not support return_json. Use prompt() instead')
        if response_format:
            raise NotImplementedError('google_model.chat does not support response_format. Use prompt() instead')
        if images:
            raise NotImplementedError('google_model.chat does not support images. Use prompt() instead')

        if not self.chat_session:
            self.chat_session = self.client.chats.create(model=self.model_name)
        response = self.chat_session.send_message(message=prompt)
        return convert_to_justai_response(response, return_json)

    async def prompt_async(self, prompt: str,  images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        if images:
            raise NotImplementedError('google_model. ..._async does not support images. Use prompt() instead')

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text, ''

    async def chat_async(self, prompt: str,  images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        async for chunk in self.prompt_async(prompt, images):
            yield chunk

    def token_count(self, text: str) -> int:
        response = self.client.models.count_tokens(model=self.model_name, contents=text)
        return response.total_tokens

    def generate_image(self, prompt, images: ImageInput):
        client = genai.Client()

        if not isinstance(images, list):
            images = [images]
        images = [to_pil_image(img) for img in images]

        response = client.models.generate_content(
            model=self.model_name,
            contents=images + [prompt],
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                return image


def convert_to_justai_response(response, return_json):
    input_token_count = response.usage_metadata.prompt_token_count
    output_token_count = (response.usage_metadata.candidates_token_count or 0) + \
                         (response.usage_metadata.thoughts_token_count or 0)
    result = response.text if not return_json else response.parsed if response.parsed else json.loads(response.text)
    return result, input_token_count, output_token_count
