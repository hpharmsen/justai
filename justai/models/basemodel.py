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

# Default timeout in seconds for all API calls
DEFAULT_TIMEOUT = 120.0


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

    def generate_image(self, *args, **kwargs):
        """ Overwrite in subclasses that DO support image generation."""
        raise NotImplementedError(
            f"generate_image() is not supported by {self.__class__.__name__}"
        )

    @abstractmethod
    def token_count(self, text: str) -> int:
        ...


def identify_image_format_from_base64(encoded_data: str) -> str:
    """Identify image format from base64 data. Returns MIME type supported by LLM APIs."""
    raw_data = base64.b64decode(encoded_data)[:12]  # Need 12 bytes for WebP detection

    # Magic numbers and corresponding mime types for each image format
    # Ordered by specificity (longer magic bytes first)
    formats = [
        (b'\x89PNG\r\n\x1a\n', 'image/png'),   # PNG files
        (b'GIF87a', 'image/gif'),              # GIF files (version 87a)
        (b'GIF89a', 'image/gif'),              # GIF files (version 89a)
        (b'\xff\xd8\xff', 'image/jpeg'),       # JPEG files
    ]

    # Check the raw data against known magic numbers
    for magic, mime_type in formats:
        if raw_data.startswith(magic):
            return mime_type

    # WebP files: RIFF....WEBP (bytes 0-3: RIFF, bytes 8-11: WEBP)
    if raw_data[:4] == b'RIFF' and len(raw_data) >= 12 and raw_data[8:12] == b'WEBP':
        return 'image/webp'

    # Default to JPEG for unknown formats (most widely supported)
    return 'image/jpeg'
