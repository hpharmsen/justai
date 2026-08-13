"""Tests for GoogleModel error translation, output ceiling and kwarg validation.

Runs without a Google API key: the genai client is mocked everywhere.

Usage:
    uv run pytest tests/test_google_errors.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.genai.errors import APIError, ClientError, ServerError

from justai import (AuthorizationException, BadRequestException, ConnectionException, GeneralException,
                    ModelOverloadException, Model, RatelimitException, TimeoutException,
                    TruncatedResponseException)
from justai.models.google_models import _map_google_error, convert_to_justai_response
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


class _Usage:
    prompt_token_count = 3
    candidates_token_count = 4
    thoughts_token_count = 0


class _Candidate:
    def __init__(self, finish_reason: str):
        self.finish_reason = finish_reason


class _FakeResponse:
    """Stand-in for a genai GenerateContentResponse. A MagicMock will not do: its
    truthy `.parsed` would short-circuit every JSON branch under test."""

    def __init__(self, text: str, parsed=None, finish_reason: str = 'STOP'):
        self.text = text
        self.parsed = parsed
        self.usage_metadata = _Usage()
        self.candidates = [_Candidate(finish_reason)]


def _mock_model(response: _FakeResponse = None, error: Exception = None, **kwargs):
    """A gemini Model with a mocked genai client. Returns (model, client)."""
    m = Model('gemini-2.5-flash', GEMINI_API_KEY='k', **kwargs)
    client = MagicMock()
    if error is not None:
        client.models.generate_content.side_effect = error
    else:
        client.models.generate_content.return_value = response or _FakeResponse('hi')
    m.model.client = client
    return m, client


# ---------------------------------------------------------------------------
# 1. No output ceiling is invented for a caller who passed none
# ---------------------------------------------------------------------------

def test_json_call_without_kwarg_sets_no_output_ceiling():
    m, client = _mock_model(_FakeResponse('{"a": 1}'))
    m.prompt('hi', return_json=True, cached=False)
    config = client.models.generate_content.call_args.kwargs['config']
    assert config.max_output_tokens is None


def test_json_call_with_too_low_ceiling_is_raised_to_the_minimum():
    m, client = _mock_model(_FakeResponse('{"a": 1}'), max_output_tokens=1000)
    m.prompt('hi', return_json=True, cached=False)
    config = client.models.generate_content.call_args.kwargs['config']
    assert config.max_output_tokens == 16384


def test_json_call_with_high_ceiling_is_left_alone():
    m, client = _mock_model(_FakeResponse('{"a": 1}'), max_output_tokens=65535)
    m.prompt('hi', return_json=True, cached=False)
    config = client.models.generate_content.call_args.kwargs['config']
    assert config.max_output_tokens == 65535


# ---------------------------------------------------------------------------
# 2. APIError translation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('code, expected', [
    (401, AuthorizationException),
    (403, AuthorizationException),
    (408, TimeoutException),
    (429, RatelimitException),
    (400, BadRequestException),
    (404, BadRequestException),
    (422, BadRequestException),
    (503, ModelOverloadException),
    (500, ConnectionException),
    (502, ConnectionException),
])
def test_api_error_maps_to_justai_exception(code, expected):
    mapped = _map_google_error(APIError(code, {'error': {'message': 'boom', 'status': 'X'}}))
    assert isinstance(mapped, expected)


def test_mapped_exception_keeps_code_and_details():
    payload = {'error': {'message': 'quota', 'status': 'RESOURCE_EXHAUSTED',
                         'details': [{'retryDelay': '31s'}]}}
    mapped = _map_google_error(ClientError(429, payload))
    assert mapped.code == 429
    assert mapped.details['error']['details'][0]['retryDelay'] == '31s'


def test_prompt_translates_client_error():
    m, _ = _mock_model(error=ClientError(429, {'error': {'message': 'quota'}}))
    with pytest.raises(RatelimitException):
        m.prompt('hi', cached=False)


def test_prompt_translates_server_error():
    m, _ = _mock_model(error=ServerError(503, {'error': {'message': 'overloaded'}}))
    with pytest.raises(ModelOverloadException):
        m.prompt('hi', cached=False)


# ---------------------------------------------------------------------------
# 3. Truncation gets its own type, unparseable JSON keeps the old one
# ---------------------------------------------------------------------------

def test_truncated_json_raises_truncated_response_exception():
    response = _FakeResponse('{"names": ["Antoine Amedin Houns', finish_reason='MAX_TOKENS')
    with pytest.raises(TruncatedResponseException):
        convert_to_justai_response(response, True)


def test_unparseable_json_still_raises_general_exception():
    response = _FakeResponse('no json here at all', finish_reason='STOP')
    with pytest.raises(GeneralException) as excinfo:
        convert_to_justai_response(response, True)
    assert not isinstance(excinfo.value, TruncatedResponseException)


def test_truncated_response_exception_is_a_general_exception():
    """Existing `except GeneralException` callers must keep catching truncation."""
    assert issubclass(TruncatedResponseException, GeneralException)


def test_truncated_non_json_response_is_returned_as_is():
    """Plain text callers keep whatever came back; only the JSON contract is strict."""
    result, _, _ = convert_to_justai_response(_FakeResponse('half a senten', finish_reason='MAX_TOKENS'), False)
    assert result == 'half a senten'


# ---------------------------------------------------------------------------
# 4. Unknown kwargs are rejected at construction instead of silently dropped
# ---------------------------------------------------------------------------

def test_unknown_kwarg_raises_at_construction():
    with pytest.raises(ValueError, match='verbositty'):
        Model('gemini-2.5-flash', GEMINI_API_KEY='k', verbositty=2)


def test_supported_api_kwargs_are_accepted():
    m = Model('gemini-2.5-flash', GEMINI_API_KEY='k', temperature=0.5, max_output_tokens=100)
    assert m.model.api_params == {'temperature': 0.5, 'max_output_tokens': 100}


def test_non_api_kwargs_are_accepted():
    """timeout/max_retries/debug/effort are consumed elsewhere, not by the Google API."""
    m = Model('gemini-2.5-flash', GEMINI_API_KEY='k', timeout=5, max_retries=3, debug=True)
    assert m.model.api_params == {}
