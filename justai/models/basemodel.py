import base64
import inspect
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Union

import httpx
from dotenv import dotenv_values
from PIL.Image import Image

from justai.tools.display import ERROR_COLOR, color_print

ImageInput = Optional[Union[
    list[str],
    list[bytes],
    list[Image],
    str,
    bytes,
    Image
]]


# Default timeout in seconds for all API calls
DEFAULT_TIMEOUT = 120.0

# Retries performed by the provider SDK itself. None means "leave the SDK default"
# (2 for both the Anthropic and OpenAI clients). Pass max_retries=0 when the call
# runs under a hard deadline, e.g. inside a web request: an LLM call takes 5-20s,
# so a single blind retry already blows a 30s budget.
DEFAULT_MAX_RETRIES = None

# Retries the agent loop performs itself, when max_retries is not given anywhere.
DEFAULT_AGENT_RETRIES = 3

# Connecting, writing and pool-waiting never legitimately take long; only reading
# can. One flat timeout for all four means a dead connection hangs for the full
# read budget before anyone notices.
CONNECT_TIMEOUT = 10.0

# Silence *between* two chunks of a stream. A healthy stream emits every few
# hundred ms, so 30s of nothing means the connection is gone. This does not cap
# the total length of a response, only the gaps in it.
STREAM_READ_TIMEOUT = 30.0


def client_retry_kwargs(params: dict) -> dict:
    """max_retries kwarg for a provider SDK client; empty dict keeps the SDK default."""
    max_retries = params.get('max_retries', DEFAULT_MAX_RETRIES)
    return {} if max_retries is None else {'max_retries': max_retries}


def client_retry_attempts(params: dict) -> int | None:
    """max_retries as an attempt count (initial call included); None keeps the SDK default."""
    max_retries = params.get('max_retries', DEFAULT_MAX_RETRIES)
    return None if max_retries is None else max_retries + 1


def client_timeout(params: dict) -> httpx.Timeout:
    """Timeout for the SDK client. Read stays generous: a non-streaming call waits
    for the entire answer in one read, so tightening it would break long generations."""
    return httpx.Timeout(
        params.get('timeout', DEFAULT_TIMEOUT),
        connect=CONNECT_TIMEOUT, write=CONNECT_TIMEOUT, pool=CONNECT_TIMEOUT,
    )


def stream_timeout(params: dict) -> httpx.Timeout:
    """Timeout for one streaming request. Read applies per chunk here, so it can be
    far tighter than the client default without capping the total response."""
    read = min(params.get('timeout', DEFAULT_TIMEOUT), STREAM_READ_TIMEOUT)
    return httpx.Timeout(
        read, connect=CONNECT_TIMEOUT, write=CONNECT_TIMEOUT, pool=CONNECT_TIMEOUT,
    )


# Python type -> JSON Schema type, shared by every provider's tool-spec builder.
JSON_TYPE_MAP: dict[type, str] = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
}


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

class TruncatedResponseException(GeneralException):
    """Raised when the model stopped on its output limit, so the answer is incomplete.

    Subclasses GeneralException so existing handlers keep catching it. A separate type
    lets the caller split on size (retry smaller, raise the ceiling) instead of guessing
    whether an unparseable answer was cut off or simply not JSON.
    """

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
    _NON_API_PARAMS: frozenset[str] = frozenset({'timeout', 'max_retries', 'async', 'debug', 'effort'})

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
        self._user_supplied: set[str] = {k for k in params if k not in {'effort'}}
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

    def close(self) -> None:
        """Close the provider's HTTP client. No-op for providers that don't hold one.

        Async-only clients (AsyncAnthropic) are skipped: closing them needs a running
        event loop, which a synchronous __exit__ cannot provide.
        """
        close = getattr(getattr(self, 'client', None), 'close', None)
        if close and not inspect.iscoroutinefunction(close):
            close()

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


def get_api_key(params: dict, keynames: str | tuple[str, ...], provider: str, url: str) -> str:
    """Resolve an API key from params, the environment or .env, in that order.

    Pops the key out of `params` so it never leaks into the provider API call.
    Raises AuthorizationException when no key is found, so a missing key surfaces at
    construction time rather than as an opaque 401 on the first call.
    """
    names = (keynames,) if isinstance(keynames, str) else keynames
    from_params = [params.pop(name, None) for name in names]  # Pop all, so none leak into api_params
    env = dotenv_values()
    for name, supplied in zip(names, from_params):
        key = supplied or os.getenv(name) or env.get(name)
        if key:
            return key
    keyname = names[0]
    color_print(f'No {provider} API key found. Create one at {url} and '
                f'set it in the .env file like {keyname}=here_comes_your_key.', color=ERROR_COLOR)
    raise AuthorizationException(f'No {keyname} found')


def identify_image_format_from_base64(encoded_data: str) -> str:
    """Identify image format from base64 data. Returns MIME type supported by LLM APIs."""
    from justai.tools.images import detect_mime_type  # Local import: images.py imports nothing from here

    mime_type = detect_mime_type(base64.b64decode(encoded_data)[:12])  # Need 12 bytes for WebP detection
    # detect_mime_type also recognises BMP, which no LLM API accepts. Fall back to its own default.
    return mime_type if mime_type != 'image/bmp' else 'image/jpeg'
