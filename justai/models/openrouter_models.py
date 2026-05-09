import base64
import os
from io import BytesIO

from dotenv import dotenv_values
from openai import OpenAI
from PIL import Image

from justai.models.basemodel import BaseModel, ImageInput, DEFAULT_TIMEOUT
from justai.models.openai_completions import OpenAICompletionsModel
from justai.tools.display import color_print, ERROR_COLOR
from justai.tools.images import to_base64_data_uri


class OpenRouterModel(OpenAICompletionsModel):
    def __init__(self, model_name: str, params: dict = None):
        system_message = f"You are {model_name}, a large language model."
        BaseModel.__init__(self, model_name, params, system_message)

        # Authentication
        keyname = "OPENROUTER_API_KEY"
        api_key = params.get(keyname) or os.getenv(keyname) or dotenv_values()[keyname]
        if not api_key:
            color_print("No OpenRouter API key found. Create one at https://openrouter.ai/settings/keys and " +
                        f"set it in the .env file like {keyname}=here_comes_your_key.", color=ERROR_COLOR)
        timeout = params.get('timeout', DEFAULT_TIMEOUT)
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=timeout)

        self.messages = [{"role": "system", "content": self.system_message}]
        self.supports_image_generation = True

    def generate_image(self, prompt: str, images: ImageInput = None, size: tuple[int, int] | None = None, options: dict = None) -> Image:
        """Generate an image via OpenRouter chat completions with modalities."""
        content = [{"type": "text", "text": prompt}]
        if images:
            for image in images:
                data_uri = image if isinstance(image, str) and image.startswith('http') else to_base64_data_uri(image)
                content.append({"type": "image_url", "image_url": {"url": data_uri}})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            modalities=["image", "text"],
            timeout=300,
        )

        message = resp.choices[0].message

        # Extract base64 image data from response
        b64 = None

        # OpenRouter returns images in message.images[]
        if hasattr(message, 'images') and message.images:
            img_entry = message.images[0]
            url = img_entry['image_url']['url'] if isinstance(img_entry, dict) else img_entry.image_url.url
            if ',' in url:
                b64 = url.split(',', 1)[1]

        # Fallback: check content blocks
        if not b64 and hasattr(message, 'content') and message.content:
            for block in (message.content if isinstance(message.content, list) else [message.content]):
                if isinstance(block, str) and block.startswith('data:image'):
                    b64 = block.split(',', 1)[1]
                elif hasattr(block, 'type') and block.type == 'image_url':
                    url = block.image_url.url if hasattr(block, 'image_url') else None
                    if url and ',' in url:
                        b64 = url.split(',', 1)[1]
                if b64:
                    break

        if b64:
            return Image.open(BytesIO(base64.b64decode(b64)))
        raise ValueError(f'No image returned by {self.model_name}')
