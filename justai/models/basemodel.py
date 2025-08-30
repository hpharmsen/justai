import base64
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, Optional, Union

from PIL.Image import Image

ImageInput = Optional[Union[
    list[str],
    list[bytes],
    list[Image],
    str,
    bytes,
    Image
]]


from justai.model.message import Message


class ConnectionException(Exception):
    pass

class AuthorizationException(Exception):
    pass

class ModelOverloadException(Exception):
    pass

class RatelimitException(Exception):
    pass

class BadRequestException(Exception):
    pass

class TimeoutException(Exception):
    pass

class GeneralException(Exception):
    pass


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, model_name: str, params: dict, system_message: str):
        """ Model implemention should create attributes for all supported parameters """
        self.model_name = model_name
        self.model_params = params  # Specific parameters for specific models like temperature
        self.system_message = system_message
        self.debug = params.get('debug', False)

        # Fields to indicate of certain capabilities are supported.
        # Can (and will be) overridden by specific models that do not support it
        self.supports_return_json = True
        self.supports_image_input = True
        self.supports_tool_use = True

        self.supports_function_calling = False
        self.supports_automatic_function_calling = False
        self.supports_cached_prompts = False
        self.supports_image_generation = False

        # The Model class that wraps this model so this model can set attributes there like token count
        # This value will be set by the Model class itself after instantiation
        self.encapsulating_model = None

    def set(self, key: str, value):
        if not hasattr(self, key):
            raise (AttributeError(f"Model has no attribute {key}"))
        setattr(self, key, value)

    @abstractmethod
    def prompt(self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None]:
        ...

    @abstractmethod
    def chat(self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None]:
        ...

    @abstractmethod
    def prompt_async(self, prompt: str,  images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        ...

    @abstractmethod
    def chat_async(self, prompt: str,  images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        ...

    @abstractmethod
    def token_count(self, text: str) -> int:
        ...


def identify_image_format_from_base64(encoded_data: str) -> str:
    raw_data = base64.b64decode(encoded_data)[:8]

    # Magic numbers and corresponding mime types for each image format
    formats = {
        b'\xff\xd8\xff': 'image/jpeg',  # JPEG files
        b'\x89PNG\r\n\x1a\n': 'image/png',  # PNG files
        b'GIF87a': 'image/gif',  # GIF files (version 87a)
        b'GIF89a': 'image/gif',  # GIF files (version 89a)
        b'BM': 'image/bmp',  # BMP files
        b'II*\x00': 'image/tiff',  # TIFF files (little endian)
        b'MM\x00*': 'image/tiff',  # TIFF files (big endian)
        b'RIFF': 'image/webp',  # WebP files (part of RIFF)
        b'<svg': 'image/svg+xml',  # SVG files (this is a rough check; SVG is a text-based format)
        b'<!DOCT': 'image/svg+xml',  # SVG files with DOCTYPE declaration
    }

    # Check the raw data against known magic numbers
    for magic, mime_type in formats.items():
        if raw_data.startswith(magic):
            return mime_type

    return 'unknown/unknown'
