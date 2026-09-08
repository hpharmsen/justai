"""Tests voor de cachetellers.

Zonder deze cijfers kun je niet zien of prompt caching werkt. De API geeft geen fout
als een cache mist, alleen andere getallen, dus de teller is de enige meting.

Let op het verschil tussen providers, dat de tellers zelf niet uitdrukken: bij
Anthropic staan de gecachte tokens los van `input_tokens`, bij OpenAI en Gemini zijn
ze er een deelverzameling van.

Draaien zonder API-key.

Usage:
    uv run pytest tests/test_cache_stats.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from justai import Model
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


# ---------------------------------------------------------------------------
# 1. BaseModel-basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('model_name', 'key'),
    [
        ('claude-sonnet-4-6', 'ANTHROPIC_API_KEY'),
        ('gpt-5-mini', 'OPENAI_API_KEY'),
        ('gemini-2.5-flash', 'GOOGLE_API_KEY'),
        ('deepseek-chat', 'DEEPSEEK_API_KEY'),
    ],
)
def test_every_provider_exposes_cache_counters(model_name, key):
    # Nul betekent hier "niets gemeten", niet "geen cache". Maar een AttributeError
    # dwingt elke aanroeper tot een hasattr-dans, en dat is geen antwoord.
    m = Model(model_name, **{key: 'k'})
    assert m.cache_creation_input_tokens == 0
    assert m.cache_read_input_tokens == 0


def test_record_usage_sets_cache_counters():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    m.model.record_usage(10, 5, cache_creation_tokens=100, cache_read_tokens=900)
    assert m.cache_creation_input_tokens == 100
    assert m.cache_read_input_tokens == 900


def test_record_usage_without_cache_args_resets_to_zero():
    # Anders lekken de cijfers van de vorige call door in een call die niets cachete.
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    m.model.record_usage(10, 5, cache_read_tokens=900)
    m.model.record_usage(10, 5)
    assert m.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# 2. Anthropic: ook zonder cached_prompt
# ---------------------------------------------------------------------------


def _mock_anthropic_response(cache_creation=0, cache_read=0):
    resp = MagicMock()
    block = MagicMock()
    block.text = 'ok'
    block.type = 'text'
    resp.content = [block]
    resp.stop_reason = 'end_turn'
    resp.usage = MagicMock(
        input_tokens=7,
        output_tokens=3,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    return resp


def test_anthropic_reports_cache_tokens_without_cached_prompt():
    # Regressie: de tellers zaten achter een `if self.cached_prompt`, terwijl de
    # breakpoints nu automatisch gezet worden en er dus altijd iets te melden is.
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    m.model.client = MagicMock()
    m.model.client.messages.create.return_value = _mock_anthropic_response(cache_creation=0, cache_read=1234)
    m.model.chat('hallo')
    assert m.cache_read_input_tokens == 1234


def test_anthropic_reports_cache_creation():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    m.model.client = MagicMock()
    m.model.client.messages.create.return_value = _mock_anthropic_response(cache_creation=555)
    m.model.chat('hallo')
    assert m.cache_creation_input_tokens == 555


# ---------------------------------------------------------------------------
# 3. OpenAI en Gemini
# ---------------------------------------------------------------------------


def test_openai_responses_reports_cached_tokens():
    m = Model('gpt-5-mini', OPENAI_API_KEY='k')
    resp = MagicMock()
    resp.output = []
    resp.output_text = 'ok'
    resp.id = 'resp_test'
    resp.usage = MagicMock(
        input_tokens=100, output_tokens=5, input_tokens_details=MagicMock(cached_tokens=64)
    )
    m.model.client = MagicMock()
    m.model.client.responses.create.return_value = resp
    m.model.chat('hallo')
    assert m.cache_read_input_tokens == 64


def test_openai_completions_reports_cached_tokens():
    m = Model('deepseek-chat', DEEPSEEK_API_KEY='k')
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = 'ok'
    choice.message.tool_calls = None
    choice.message.reasoning_content = None
    completion.choices = [choice]
    completion.usage = MagicMock(
        prompt_tokens=100, completion_tokens=5, prompt_tokens_details=MagicMock(cached_tokens=32)
    )
    m.model.client = MagicMock()
    m.model.client.chat.completions.create.return_value = completion
    m.model.chat('hallo')
    assert m.cache_read_input_tokens == 32


def test_missing_usage_details_do_not_raise():
    # Oudere SDK-responses hebben het veld niet. Een AttributeError midden in een
    # geslaagde call zou een gefactureerd antwoord weggooien om een telling.
    m = Model('gpt-5-mini', OPENAI_API_KEY='k')
    resp = MagicMock()
    resp.output = []
    resp.output_text = 'ok'
    resp.id = 'resp_test'
    resp.usage = MagicMock(input_tokens=100, output_tokens=5)
    del resp.usage.input_tokens_details
    m.model.client = MagicMock()
    m.model.client.responses.create.return_value = resp
    m.model.chat('hallo')
    assert m.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# 4. De agent-loop: per iteratie afleesbaar
# ---------------------------------------------------------------------------


def test_stream_done_chunk_carries_cache_tokens():
    # Zonder dit kun je van een agent-run niet zien of de breakpoints iets deden.
    from justai.models.basemodel import StreamChunk

    chunk = StreamChunk(type='done', input_tokens=1, output_tokens=1)
    assert chunk.cache_creation_input_tokens is None
    assert chunk.cache_read_input_tokens is None
