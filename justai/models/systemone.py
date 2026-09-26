"""System One models: typed, calibrated decisions instead of generated text.

One wire format, three entrances: Jev at TypeSafe, Jev through OpenRouter, and any
self-hosted model serving POST /v1/systemone. This module holds the format itself;
the provider classes live at the bottom.
"""

import json

import httpx

from justai.models.basemodel import (
    AuthorizationException,
    BadRequestException,
    ConnectionException,
    GeneralException,
    ModelOverloadException,
    RatelimitException,
    TimeoutException,
)

# "a maximum of 255 options per Choice"
MAX_CHOICE_OPTIONS = 255
# "A Score should have at least two levels; the API accepts up to 10"
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Name for the one question of the single-question form. Never visible in the API.
SINGLE = 'answer'

# Answer fields that a Score keys by level. The vendor SDKs expose these as ints, so
# justai does too; over the wire JSON can only give us strings.
LEVEL_KEYED = ('legend', 'probabilities')


def build_question(options: dict | list | None, instructions: str | None) -> dict:
    """Turn options into a question spec. The shape of options picks the type."""
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
    if instructions:
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
    assert isinstance(state, (str, dict, list)), f'state should be a str, dict or list, got {type(state).__name__}'
    if questions is None:
        questions = {SINGLE: build_question(options, instructions)}
    return {'model': wire_model_name, 'state': state, 'questions': questions}


def _with_int_levels(answer: dict) -> dict:
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
    answers = body['answers']
    if single:
        return _with_int_levels(answers[SINGLE])
    return {name: _with_int_levels(answer) for name, answer in answers.items()}


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
