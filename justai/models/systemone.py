"""System One models: typed, calibrated decisions instead of generated text.

One wire format, three entrances: Jev at TypeSafe, Jev through OpenRouter, and any
self-hosted model serving POST /v1/systemone. This module holds the format itself;
the provider classes live at the bottom.
"""

import json
import os
import sys
import time

import httpx

from justai.models.basemodel import (
    AuthorizationException,
    BadRequestException,
    BaseModel,
    ConnectionException,
    CONNECT_TIMEOUT,
    GeneralException,
    ImageInput,
    ModelOverloadException,
    RatelimitException,
    TimeoutException,
    get_api_key,
)

# "a maximum of 255 options per Choice"
MAX_CHOICE_OPTIONS = 255
# "A Score should have at least two levels; the API accepts up to 10"
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Name for the one question of the single-question form. Never visible in the API.
SINGLE = 'answer'

# These models answer in 70 to 500 ms, so the 120s DEFAULT_TIMEOUT is useless here: a call
# still hanging after two minutes is dead, not slow.
CLASSIFY_TIMEOUT = 30.0

# Retries justai performs itself. The provider docs prescribe them for 429 and 529, and a
# classifier is called in volume, so this path gets hit for real. Matches the 2 that the
# Anthropic and OpenAI SDK clients default to; DEFAULT_MAX_RETRIES is None ("keep the SDK
# default"), which is not a number this hand-rolled loop can use.
CLASSIFY_MAX_RETRIES = 2

# Statuses worth waiting out. Everything else fails on the first try: a 422 does not
# improve with patience.
RETRY_STATUSES = frozenset({429, 529})

# Answer fields that a Score keys by level. The vendor SDKs expose these as ints, so
# justai does too; over the wire JSON can only give us strings.
LEVEL_KEYED = ('legend', 'probabilities')


def build_question(options: dict | list | None, instructions: str) -> dict:
    """Turn options into a question spec. The shape of options picks the type.

    `instructions` is what the model is actually asked; the API rejects a question without
    it with a 400, for all three types.
    """
    assert instructions, 'classify() needs instructions: the question to ask about the state'
    if isinstance(options, dict):
        assert len(options) <= MAX_CHOICE_OPTIONS, f'A choice takes at most {MAX_CHOICE_OPTIONS} options'
        question = {'type': 'choice', 'criteria': options}
    elif isinstance(options, list):
        assert MIN_SCORE_LEVELS <= len(options) <= MAX_SCORE_LEVELS, (
            f'A score takes {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {len(options)}'
        )
        question = {'type': 'score', 'criteria': options}
    elif options is None:
        question = {'type': 'noul'}
    else:
        raise TypeError(f'options should be a dict, a list or None, got {type(options).__name__}')
    question['instructions'] = instructions
    return question


def build_payload(
    wire_model_name: str,
    state: str | dict | list,
    options: dict | list | None = None,
    *,
    instructions: str | None = None,
    questions: dict | None = None,
) -> dict:
    """Build the request body. Identical at every provider."""
    assert not (options is not None and questions), 'Pass either options or questions, not both'
    assert not (instructions and questions), 'With questions=, the instructions live in each question'
    assert isinstance(state, (str, dict, list)), f'state should be a str, dict or list, got {type(state).__name__}'
    if questions is None:
        questions = {SINGLE: build_question(options, instructions)}
    return {'model': wire_model_name, 'state': state, 'questions': questions}


def with_int_levels(answer: dict) -> dict:
    """Key a score's legend and probabilities by level instead of by string.

    Only scores: a choice keys those maps by option name, which must stay untouched.
    Idempotent, because the result goes through the cache's json round trip and comes
    back restringified.
    """
    if answer.get('type') != 'score':
        return answer
    return answer | {
        field: {int(level): value for level, value in answer[field].items()} for field in LEVEL_KEYED if field in answer
    }


def unpack(body: dict, single: bool) -> dict:
    """Pull the answers out of a response. Drops nothing and renames nothing."""
    return restore_level_keys(body['answers'][SINGLE] if single else body['answers'], single)


def restore_level_keys(result: dict, single: bool) -> dict:
    """Re-apply the int level keys to an unpacked result. Needed after the cache too, which
    stores the dict as JSON and hands it back with those keys restringified."""
    return with_int_levels(result) if single else {name: with_int_levels(a) for name, a in result.items()}


def usage_of(body: dict) -> tuple[int, int]:
    """Read the token counts. `cost` is reported too but justai does not track it yet."""
    usage = body.get('usage') or {}
    return usage.get('input_tokens', 0), usage.get('output_tokens', 0)


def map_http_error(e: Exception) -> Exception:
    """Translate an httpx or json error to the matching justai exception.

    Order matters twice: httpx.TimeoutException subclasses TransportError, so it comes
    first, and 529 is tested before the generic 5xx branch or an overload would surface
    as a connection problem.
    """
    if isinstance(e, httpx.TimeoutException):
        return TimeoutException(e)
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code in (401, 403):
            return AuthorizationException(e)
        if code in (400, 422):
            return BadRequestException(e)
        if code == 429:
            return RatelimitException(e)
        if code == 529:
            return ModelOverloadException(e)
        if code >= 500:
            return ConnectionException(e)
        return GeneralException(e)
    if isinstance(e, httpx.TransportError):
        return ConnectionException(e)
    if isinstance(e, json.JSONDecodeError):
        return GeneralException(e)
    return GeneralException(e)


class SystemOneMixin:
    """The HTTP half of classification. Subclasses supply the URL, headers and model name."""

    def classify_url(self) -> str:
        raise NotImplementedError

    def classify_headers(self) -> dict:
        raise NotImplementedError

    def wire_model_name(self) -> str:
        raise NotImplementedError

    @property
    def systemone_client(self) -> httpx.Client:
        """One keep-alive client per model. A TLS handshake costs 50-100 ms and these models
        answer in 70-500 ms, so a fresh connection per call gives away most of the speed
        that is the whole point of this model category."""
        client = getattr(self, '_systemone_client', None)
        if client is None or client.is_closed:
            client = self._systemone_client = httpx.Client()
        return client

    def classify(
        self,
        state: str | dict | list,
        options: dict | list | None = None,
        *,
        instructions: str | None = None,
        questions: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[dict, int, int]:
        payload = build_payload(
            self.wire_model_name(), state, options, instructions=instructions, questions=questions
        )
        limits = httpx.Timeout(
            timeout or self.model_params.get('timeout', CLASSIFY_TIMEOUT),
            connect=CONNECT_TIMEOUT,
            write=CONNECT_TIMEOUT,
            pool=CONNECT_TIMEOUT,
        )
        max_retries = self.model_params.get('max_retries')
        if max_retries is None:
            max_retries = CLASSIFY_MAX_RETRIES

        for attempt in range(max_retries + 1):
            try:
                response = self.systemone_client.post(
                    self.classify_url(), json=payload, headers=self.classify_headers(), timeout=limits
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in RETRY_STATUSES and attempt < max_retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise map_http_error(e) from e
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                # Deliberately not `except Exception`: that would swallow bugs in the builder.
                raise map_http_error(e) from e

            # Before unpack, which may trip over a missing key. Those tokens are billed either way.
            input_tokens, output_tokens = usage_of(body)
            self.record_usage(input_tokens, output_tokens)
            return unpack(body, single=questions is None), input_tokens, output_tokens

    def close(self) -> None:
        """Close our own client first. BaseModel.close only knows about self.client, which at
        OpenRouterModel is already the OpenAI client."""
        client = getattr(self, '_systemone_client', None)
        if client is not None:
            client.close()
        super().close()


class SystemOneModel(SystemOneMixin, BaseModel):
    """Jev at TypeSafe (`jev*`) and any self-hosted System One server (`systemone/<name>`)."""

    _NON_API_PARAMS = BaseModel._NON_API_PARAMS | {'base_url'}

    def __init__(self, model_name: str, params: dict = None):
        params = params or {}
        super().__init__(model_name, params, system_message='')

        if model_name.startswith('systemone/'):
            base_url = params.get('base_url')
            assert base_url, "A self-hosted System One model needs a base_url, e.g. Model('systemone/kev-3b', base_url='http://localhost:8000')"
            self.base_url = base_url.rstrip('/')
            # Self-hosted servers usually want no key at all, so a missing one is not an error.
            self.api_key = params.pop('SYSTEMONE_API_KEY', None) or os.getenv('SYSTEMONE_API_KEY')
        else:
            self.base_url = 'https://api.typesafe.ai'
            self.api_key = get_api_key(params, 'TYPESAFE_API_KEY', 'TypeSafe', 'https://typesafe.ai/settings/keys')

        # No text comes out of these models, so none of the text features apply.
        self.supports_return_json = False
        self.supports_image_input = False
        self.supports_tool_use = False

    def classify_url(self) -> str:
        return f'{self.base_url}/v1/systemone'

    def classify_headers(self) -> dict:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def wire_model_name(self) -> str:
        return self.model_name.removeprefix('systemone/')

    def prompt(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}')

    def chat(self, prompt: str, images: ImageInput, tools: list, return_json: bool, response_format):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}')

    async def prompt_async(self, prompt: str, images: list[ImageInput]):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}')

    async def chat_async(self, prompt: str, images: list[ImageInput]):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}')

    def token_count(self, text: str) -> int:
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} is not supported by {self.__class__.__name__}')
