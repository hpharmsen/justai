"""Implementation of the Reve image models.

Feature table:
    - Async chat:       NO
    - Return JSON:      YES
    - Structured types: NO
    - Token counter:    YES
    - Image support:    YES
    - Tool use:         NO

Supported parameters:
    options['aspect_ratio'] = One of '16:9', '9:16', '3:2', '2:3', '4:3', '3:4', or '1:1'. # Default: '3:2'
"""

import base64
import json
import os
import sys
from io import BytesIO

import httpx
from PIL import Image

from justai.model.model import ImageInput
from justai.models.basemodel import BaseModel, DEFAULT_TIMEOUT, GeneralException, BadRequestException
from justai.tools.display import ERROR_COLOR, color_print
from justai.tools.images import to_base64_image


class ReveModel(BaseModel):

    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        system_message = f"You are {model_name}, a large language model trained by Google."
        super().__init__(model_name, params, system_message)

        # Authentication
        self.api_key = params.get("REVE_API_KEY") or os.getenv("REVE_API_KEY")
        if not self.api_key:
            color_print("No REVE_API_KEY found. Create one at https://api.reve.com and " +
                        "set it in the .env file like REVE_API_KEY=here_comes_your_key.", color=ERROR_COLOR)

        # Diversions from the features that are supported or not supported by default
        self.supports_image_generation = True

    def generate_image(self, prompt, images: ImageInput, options: dict = None):
        # options['aspect_ratio'] = One of '16:9', '9:16', '3:2', '2:3', '4:3', '3:4', or '1:1'.
        # Default: '3:2'

        endpoint = "https://api.reve.com/v1/image/remix" if images else "https://api.reve.com/v1/image/create"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = {"prompt": prompt, "version": "latest"}
        if images:
            payload["reference_images"] = [to_base64_image(im) for im in images]
        if options and options.get("aspect_ratio"):
            payload["aspect_ratio"] = options["aspect_ratio"]

        timeout = self.model_params.get('timeout', DEFAULT_TIMEOUT)
        try:
            response = httpx.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Parse the response
            result = response.json()
            print(f"Request ID: {result['request_id']}")
            print(f"Credits used: {result['credits_used']}")
            print(f"Credits remaining: {result['credits_remaining']}")

            if result.get("content_violation"):
                raise BadRequestException("content_violation")
            else:
                print("Image generated successfully!")
                # The base64 image data is in result['image']
                image_data = base64.b64decode(result['image'])
                image = Image.open(BytesIO(image_data))
                return image

        except httpx.RequestError as e:
            raise GeneralException(e)
        except json.JSONDecodeError as e:
            raise GeneralException(e)

    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format):
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}")

    def chat(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format):
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}")

    async def prompt_async(self, prompt: str, images: list[ImageInput]):
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}")

    async def chat_async(self, prompt: str, images: list[ImageInput]):
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}")

    def token_count(self, text: str) -> int:
        raise NotImplementedError(f"{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}")
