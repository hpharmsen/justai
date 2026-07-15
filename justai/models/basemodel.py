import base64
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Union

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

class RefusalException(Exception):
    """Raised when a model refuses to answer (e.g. Anthropic safety classifier)."""
    def __init__(self, category: str = 'unknown', message: str = ''):
        self.category = category
        super().__init__(f'Model refused response: {category}. {message}'.strip())


class EffortDownmapWarning(UserWarning):
    """Emitted when the requested effort level is downmapped to a supported one."""


UNIVERSAL_EFFORT_LEVELS: frozenset[str] = frozenset({'low', 'medium', 'high', 'xhigh', 'max'})


@dataclass
class ToolCallRequest:
    """A tool/function call requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class StreamChunk:
    """A chunk from a streaming LLM response."""
    type: str  # 'text' | 'tool_calls' | 'done'
    content: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


class BaseModel(ABC):

    # Keys that live in model_params but must not be forwarded to provider APIs.
    # Subclasses extend by overriding with a broader frozenset.
    _NON_API_PARAMS: frozenset[str] = frozenset({'timeout', 'async', 'debug', 'effort'})

    # Levels the user may pass as `effort=...`. Providers may extend (e.g. OpenAI adds 'none').
    EFFORT_VALID: frozenset[str] = UNIVERSAL_EFFORT_LEVELS

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

        # Effort scaffolding. Seed default so `Model.__setattr__` routes `model.effort = ...` writes here.
        self.model_params.setdefault('effort', None)
        self._effort_warned: set[str] = set()
        self._VALIDATORS: dict[str, Callable[[Any], None]] = {'effort': self._validate_effort}
        # Track which keys were user-supplied so auto-raise heuristics don't overwrite explicit values.
        self._user_supplied: set[str] = {k for k in params.keys() if k not in {'effort'}}
        if params.get('effort') is not None:
            self._validate_effort(params['effort'])
            # Emit ignore/downmap warning immediately for feedback at construction time.
            _, warn = self.resolve_effort()
            if warn:
                self._emit_effort_warning(warn)

    @property
    def api_params(self) -> dict:
        """model_params filtered down to keys safe to forward to the provider API."""
        return {k: v for k, v in self.model_params.items() if k not in self._NON_API_PARAMS}

    def set(self, key: str, value):
        if not hasattr(self, key):
            raise (AttributeError(f"Model has no attribute {key}"))
        setattr(self, key, value)

    def _validate_effort(self, value: Any) -> None:
        """Raise ValueError if value is not a legal effort level for this provider."""
        if value is None:
            return
        if value not in self.EFFORT_VALID:
            valid = sorted(v for v in self.EFFORT_VALID)
            raise ValueError(
                f'effort must be one of {valid} or None; got {value!r}'
            )

    def resolve_effort(self) -> tuple[Any, str | None]:
        """Translate `self.model_params['effort']` to the provider-native wire value.

        Returns (native_value, warning_message).
        - native_value=None means: send nothing to the provider.
        - warning_message=None means: no downmap warning to emit.
        Base implementation: no effort support at all (native=None, warn if user set effort).
        """
        effort = self.model_params.get('effort')
        if effort is None:
            return (None, None)
        return (None, f'effort={effort!r} is not supported by {self.__class__.__name__}, ignoring')

    def _emit_effort_warning(self, message: str) -> None:
        """Emit an EffortDownmapWarning at most once per (model, message) combination."""
        if message in self._effort_warned:
            return
        self._effort_warned.add(message)
        warnings.warn(message, EffortDownmapWarning, stacklevel=3)

    @abstractmethod
    def prompt(self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None, dict|None]:
        ...

    @abstractmethod
    def chat(self, prompt: str, images: list[ImageInput], tools, return_json: bool, response_format) \
            -> tuple[Any, int|None, int|None, dict|None]:
        ...

    @abstractmethod
    def prompt_async(self, prompt: str,  images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        ...

    @abstractmethod
    def chat_async(self, prompt: str,  images: list[ImageInput]) -> AsyncGenerator[tuple[str, str], None]:
        ...

    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator['StreamChunk', None]:
        """Stateless streaming call. Caller manages conversation history."""
        raise NotImplementedError(f'stream() is not supported by {self.__class__.__name__}')
        yield  # Make it a generator

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> dict:
        """Format a tool result message for this provider."""
        raise NotImplementedError(f'format_tool_result() is not supported by {self.__class__.__name__}')

    def format_assistant_message(self, text: str, tool_calls: list['ToolCallRequest'] | None = None) -> list[dict]:
        """Format an assistant message with optional tool calls. Returns list of message dicts."""
        raise NotImplementedError(f'format_assistant_message() is not supported by {self.__class__.__name__}')

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
