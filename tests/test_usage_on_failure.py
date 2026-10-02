"""Tests that a call which fails after the provider reported usage still reports its tokens.

A response that arrives and then fails to yield a usable answer is billed all the same: the
reasoning tokens were generated. Discarding the usage on the error path makes that spend
invisible to anyone tallying what a run cost.

Usage:
    uv run pytest tests/test_usage_on_failure.py
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from justai import BadRequestException, Model, TruncatedResponseException
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


# ---------------------------------------------------------------------------
# Anthropic: all of max_tokens spent on thinking, no text block in the response
# ---------------------------------------------------------------------------


def _thinking_only_response(input_tokens: int, output_tokens: int) -> MagicMock:
    thinking = MagicMock()
    thinking.type = 'thinking'
    resp = MagicMock()
    resp.content = [thinking]
    resp.stop_reason = 'max_tokens'
    resp.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return resp


def test_anthropic_no_text_block_still_reports_usage():
    m = Model('claude-fable-5-1', ANTHROPIC_API_KEY='k')
    client = MagicMock()
    client.messages.create.return_value = _thinking_only_response(645, 32768)
    m.model.client = client

    with pytest.raises(BadRequestException):
        m.prompt('hi', cached=False)

    assert m.last_token_count() == (645, 32768, 33413)


def test_anthropic_successful_call_still_reports_usage():
    """The happy path must keep reporting the same numbers as before."""
    m = Model('claude-fable-5-1', ANTHROPIC_API_KEY='k')
    text = MagicMock()
    text.type = 'text'
    text.text = 'the answer is 42'
    resp = _thinking_only_response(10, 20)
    resp.content = [text]
    client = MagicMock()
    client.messages.create.return_value = resp
    m.model.client = client

    assert m.prompt('hi', cached=False) == 'the answer is 42'
    assert m.last_token_count() == (10, 20, 30)


def test_failure_before_any_response_reports_no_usage():
    """Nothing came back, so nothing was billed. Do not invent a number."""
    m = Model('claude-fable-5-1', ANTHROPIC_API_KEY='k')
    client = MagicMock()
    client.messages.create.side_effect = ConnectionError('no route to host')
    m.model.client = client

    with pytest.raises(Exception):
        m.prompt('hi', cached=False)

    assert m.last_token_count() == (0, 0, 0)


def test_stale_usage_is_not_carried_into_the_next_failure():
    """A later failure must not report the tokens of an earlier successful call."""
    m = Model('claude-fable-5-1', ANTHROPIC_API_KEY='k')
    text = MagicMock()
    text.type = 'text'
    text.text = 'first answer'
    ok = _thinking_only_response(10, 20)
    ok.content = [text]
    client = MagicMock()
    client.messages.create.return_value = ok
    m.model.client = client
    m.prompt('hi', cached=False)

    client.messages.create.side_effect = ConnectionError('no route to host')
    with pytest.raises(Exception):
        m.prompt('hi again', cached=False)

    assert m.last_token_count() == (0, 0, 0)


# ---------------------------------------------------------------------------
# Gemini: output ceiling hit halfway through the JSON
# ---------------------------------------------------------------------------


class _Usage:
    prompt_token_count = 12
    candidates_token_count = 4000
    thoughts_token_count = 1000


class _Candidate:
    finish_reason = 'MAX_TOKENS'


class _TruncatedResponse:
    text = '{"a": 1'
    parsed = None
    usage_metadata = _Usage()
    candidates = [_Candidate()]


def test_gemini_truncated_json_still_reports_usage():
    m = Model('gemini-2.5-flash', GEMINI_API_KEY='k')
    client = MagicMock()
    client.models.generate_content.return_value = _TruncatedResponse()
    m.model.client = client

    with pytest.raises(TruncatedResponseException):
        m.prompt('hi', return_json=True, cached=False)

    assert m.last_token_count() == (12, 5000, 5012)


# ---------------------------------------------------------------------------
# OpenAI: text that was asked to be JSON and is not
# ---------------------------------------------------------------------------


def test_openai_responses_unparseable_json_still_reports_usage():
    """The responses API returns text that turns out not to be JSON. It was billed anyway."""
    m = Model('gpt-5.6-luna', OPENAI_API_KEY='k')
    response = MagicMock()
    response.output = []
    response.output_text = 'Sorry, I cannot do that.'
    response.usage = MagicMock(input_tokens=15, output_tokens=2500)
    client = MagicMock()
    client.responses.create.return_value = response
    m.model.client = client

    with pytest.raises(json.JSONDecodeError):
        m.prompt('hi', return_json=True, cached=False)

    assert m.last_token_count() == (15, 2500, 2515)


def test_openai_completions_empty_json_still_reports_usage():
    m = Model('MiniMax-M3', MINIMAX_API_KEY='k')
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='', tool_calls=None))]
    completion.usage = MagicMock(prompt_tokens=7, completion_tokens=900)
    client = MagicMock()
    # A JSON request goes through parse(), not create().
    client.chat.completions.create.return_value = completion
    client.chat.completions.parse.return_value = completion
    m.model.client = client

    with pytest.raises(ValueError):
        m.prompt('hi', return_json=True, cached=False)

    assert m.last_token_count() == (7, 900, 907)


def test_openai_completions_truncated_json_raises_truncated():
    """finish_reason 'length' means the JSON was cut off; say so instead of a bare JSONDecodeError."""
    m = Model('openrouter/google/gemini-3-flash-preview', OPENROUTER_API_KEY='k')
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='{\n  "oordeel": ', tool_calls=None), finish_reason='length')]
    completion.usage = MagicMock(prompt_tokens=7, completion_tokens=16384)
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    client.chat.completions.parse.return_value = completion
    m.model.client = client

    with pytest.raises(TruncatedResponseException):
        m.prompt('hi', return_json=True, cached=False)

    assert m.last_token_count() == (7, 16384, 16391)
