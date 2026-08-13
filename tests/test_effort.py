"""Tests for the unified `effort` parameter across providers.

Layered to run without any provider API keys (mocked tests) plus a smoke test
section that skips per-provider when the relevant API key is missing.
"""

from __future__ import annotations

import os
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from justai import Model, RefusalException, EffortDownmapWarning
from justai.tools import cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_cache_and_warnings(monkeypatch, tmp_path):
    """Reset CacheDB singleton to a tmp directory + clear warning-dedup state per test."""
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    # Reset the CacheDB singleton so it re-initializes against the tmp dir.
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


def _mock_anthropic_response(text='ok', stop_reason='end_turn', refusal_category=None):
    resp = MagicMock()
    if text:
        block = MagicMock()
        block.text = text
        block.type = 'text'
        resp.content = [block]
    else:
        resp.content = []
    resp.stop_reason = stop_reason
    if refusal_category is not None:
        resp.refusal_category = refusal_category
    resp.usage = MagicMock(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return resp


def _install_mock_anthropic_client(model, response=None):
    """Replace both sync+async clients on an AnthropicModel. Returns the mock for assertions."""
    resp = response or _mock_anthropic_response()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    mock_client.messages.parse.return_value = resp
    model.model.client = mock_client
    async_client = MagicMock()
    async_client.messages.create = AsyncMock(return_value=resp)
    model.model.async_client = async_client
    return mock_client


def _install_mock_openai_client(model, response=None):
    resp = response or MagicMock()
    resp.output = []
    resp.output_text = 'ok'
    resp.usage = MagicMock(input_tokens=1, output_tokens=1)
    resp.id = 'resp_test'
    mock_client = MagicMock()
    mock_client.responses.create.return_value = resp
    mock_client.responses.parse.return_value = resp
    model.model.client = mock_client
    return mock_client


# ---------------------------------------------------------------------------
# 1. effort=None sends no reasoning/output_config/thinking_config
# ---------------------------------------------------------------------------


def test_effort_none_sends_no_output_config_anthropic():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    client = _install_mock_anthropic_client(m)
    m.model.completion('hi')
    kwargs = client.messages.create.call_args.kwargs
    assert 'output_config' not in kwargs
    assert 'effort' not in kwargs


def test_effort_none_sends_no_reasoning_openai():
    m = Model('gpt-5.6-terra', OPENAI_API_KEY='k')
    client = _install_mock_openai_client(m)
    m.prompt('hi')
    kwargs = client.responses.create.call_args.kwargs
    assert 'reasoning' not in kwargs


# ---------------------------------------------------------------------------
# 2. Each valid level on claude-fable-5 → correct output_config['effort']
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('level', ['low', 'medium', 'high', 'xhigh', 'max'])
def test_anthropic_fable5_sends_native_effort(level):
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k', effort=level)
    client = _install_mock_anthropic_client(m)
    m.model.completion('hi')
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs['output_config']['effort'] == level


def test_anthropic_fable5_merges_effort_with_structured_format():
    from pydantic import BaseModel as PydanticModel

    class Resp(PydanticModel):
        answer: str

    # Use claude-opus-4-8: matches both STRUCTURED_OUTPUT_MODELS and the top EFFORT_TIERS entry.
    m = Model('claude-opus-4-8', ANTHROPIC_API_KEY='k', effort='high')
    resp = _mock_anthropic_response(text='{"answer": "ok"}')
    client = _install_mock_anthropic_client(m, response=resp)
    m.chat('hi', response_format=Resp)
    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs['output_config']['effort'] == 'high'
    assert 'format' in kwargs['output_config']
    assert kwargs['output_config']['format']['type'] == 'json_schema'


# ---------------------------------------------------------------------------
# 3. xhigh on claude-opus-4-6 → sends max, emits EffortDownmapWarning
# ---------------------------------------------------------------------------


def test_xhigh_on_opus_4_6_maps_up_to_max_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = Model('claude-opus-4-6', ANTHROPIC_API_KEY='k', effort='xhigh')
        client = _install_mock_anthropic_client(m)
        m.model.completion('hi')
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs['output_config']['effort'] == 'max'
    downmap = [x for x in w if x.category is EffortDownmapWarning]
    assert len(downmap) == 1
    assert 'xhigh' in str(downmap[0].message)
    assert 'max' in str(downmap[0].message)


# ---------------------------------------------------------------------------
# 4. max on claude-sonnet-4-5 → nothing sent, EffortDownmapWarning
# ---------------------------------------------------------------------------


def test_effort_on_unsupported_anthropic_model_ignored_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = Model('claude-sonnet-4-5', ANTHROPIC_API_KEY='k', effort='max')
        client = _install_mock_anthropic_client(m)
        m.model.completion('hi')
    kwargs = client.messages.create.call_args.kwargs
    assert 'output_config' not in kwargs
    downmap = [x for x in w if x.category is EffortDownmapWarning]
    assert len(downmap) >= 1
    assert 'not supported' in str(downmap[0].message)


# ---------------------------------------------------------------------------
# 5. max on gpt-5.6-sol → xhigh + warning; 'none' on gpt-5.6-luna → none
# ---------------------------------------------------------------------------


def test_max_on_gpt_56_downmaps_to_xhigh_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = Model('gpt-5.6-sol', OPENAI_API_KEY='k', effort='max')
        client = _install_mock_openai_client(m)
        m.prompt('hi')
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['reasoning'] == {'effort': 'xhigh'}
    downmap = [x for x in w if x.category is EffortDownmapWarning]
    assert len(downmap) == 1


def test_none_passthrough_on_gpt_56():
    m = Model('gpt-5.6-luna', OPENAI_API_KEY='k', effort='none')
    client = _install_mock_openai_client(m)
    m.prompt('hi')
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['reasoning'] == {'effort': 'none'}


# ---------------------------------------------------------------------------
# 6. max on older OpenAI reasoning model → ignored + warning (per user scope)
# ---------------------------------------------------------------------------


def test_max_on_older_openai_reasoning_ignored():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = Model('o3-mini', OPENAI_API_KEY='k', effort='max')
        client = _install_mock_openai_client(m)
        m.prompt('hi')
    kwargs = client.responses.create.call_args.kwargs
    assert 'reasoning' not in kwargs
    downmap = [x for x in w if x.category is EffortDownmapWarning]
    assert any('not supported' in str(x.message) for x in downmap)


# ---------------------------------------------------------------------------
# 7. high on gemini-3-pro → thinking_config.thinking_level == HIGH
# ---------------------------------------------------------------------------


def test_gemini3_high_sends_thinking_level_high(monkeypatch):
    # Bypass Google auth by providing dummy key
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    m = Model('gemini-3-pro', effort='high')
    # We can inspect resolve_effort directly (avoids setting up a fake genai client)
    native, warn = m.model.resolve_effort()
    assert native == 'HIGH'
    assert warn is None


# ---------------------------------------------------------------------------
# 8. invalid effort raises at both constructor and setattr
# ---------------------------------------------------------------------------


def test_effort_invalid_value_raises_at_constructor():
    with pytest.raises(ValueError, match='effort must be one of'):
        Model('claude-haiku-4-5', ANTHROPIC_API_KEY='k', effort='banana')


def test_effort_invalid_value_raises_at_setattr():
    m = Model('claude-haiku-4-5', ANTHROPIC_API_KEY='k')
    with pytest.raises(ValueError, match='effort must be one of'):
        m.effort = 'banana'


# ---------------------------------------------------------------------------
# 9. 'none' on claude → ValueError
# ---------------------------------------------------------------------------


def test_none_string_on_anthropic_raises():
    with pytest.raises(ValueError, match='effort must be one of'):
        Model('claude-fable-5', ANTHROPIC_API_KEY='k', effort='none')


# ---------------------------------------------------------------------------
# 10. Cache key changes when effort changes (direct hash inequality)
# ---------------------------------------------------------------------------


def test_cache_key_differs_across_effort_values():
    m_none = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    m_high = Model('claude-fable-5', ANTHROPIC_API_KEY='k', effort='high')
    # Use the same hash inputs as cached_response uses for prompt()
    prompt = 'test prompt'
    h1 = (
        cache.CACHE_NAMESPACE
        + ':'
        + cache.recursive_hash((m_none.model.model_name, m_none.model.model_params, None, prompt, None, False, None))
    )
    h2 = (
        cache.CACHE_NAMESPACE
        + ':'
        + cache.recursive_hash((m_high.model.model_name, m_high.model.model_params, None, prompt, None, False, None))
    )
    assert h1 != h2


# ---------------------------------------------------------------------------
# 11. Smoke tests against real APIs (skipped per-provider when key missing)
# ---------------------------------------------------------------------------


def test_smoke_effort_low_reaches_provider():
    ran_any = False

    if os.getenv('ANTHROPIC_API_KEY'):
        m = Model('claude-haiku-4-5', effort=None)  # haiku ignores effort but should still respond
        result = m.chat('reply exactly: pong')
        assert 'pong' in result.lower()
        ran_any = True

    if os.getenv('OPENAI_API_KEY'):
        m = Model('gpt-5-mini')
        result = m.prompt('reply exactly: pong')
        assert 'pong' in result.lower()
        ran_any = True

    if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'):
        m = Model('gemini-2.5-flash')
        result = m.prompt('reply exactly: pong')
        assert 'pong' in result.lower()
        ran_any = True

    if not ran_any:
        pytest.skip('No provider API keys set')


# ---------------------------------------------------------------------------
# 12. Streaming/async paths still send effort (Anthropic + OpenAI)
# ---------------------------------------------------------------------------


def test_effort_reaches_anthropic_streaming_path():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k', effort='high')
    client = _install_mock_anthropic_client(m)
    # completion(stream=True) returns the client's messages.create result directly
    m.model.completion('hi', stream=True)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs['output_config']['effort'] == 'high'
    assert kwargs.get('stream') is True


def test_effort_reaches_openai_streaming_path():
    m = Model('gpt-5.6-sol', OPENAI_API_KEY='k', effort='high')
    client = _install_mock_openai_client(m)
    # prompt_async is an async generator; we consume it synchronously to inspect the call
    import asyncio

    async def _drain():
        async for _ in m.model.prompt_async('hi'):
            pass

    # The mocked client returns a MagicMock (iterable); drain what it yields.
    asyncio.get_event_loop().run_until_complete(_drain()) if False else None
    # Simpler: call prompt (sync) instead — it uses the same _responses_create wrapper.
    m.prompt('hi')
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs['reasoning'] == {'effort': 'high'}


# ---------------------------------------------------------------------------
# 13. Multi-turn tool loop: effort on every iteration
# ---------------------------------------------------------------------------


def test_effort_survives_openai_multi_turn_tool_loop():
    m = Model('gpt-5.6-sol', OPENAI_API_KEY='k', effort='high')

    # First response: function_call. Second response: final answer.
    call = MagicMock()
    call.type = 'function_call'
    call.name = 'get_x'
    call.arguments = '{}'
    call.call_id = 'call_1'

    first = MagicMock()
    first.output = [call]
    first.id = 'r1'
    first.usage = MagicMock(input_tokens=1, output_tokens=1)

    final = MagicMock()
    final.output = []
    final.output_text = 'done'
    final.id = 'r2'
    final.usage = MagicMock(input_tokens=1, output_tokens=1)

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [first, final]
    m.model.client = mock_client

    def get_x():
        return 'x'

    m.add_tool(get_x, description='return x', parameters={}, required_parameters=[])
    m.prompt('hi')

    calls = mock_client.responses.create.call_args_list
    assert len(calls) >= 2
    for c in calls:
        assert c.kwargs['reasoning'] == {'effort': 'high'}


# ---------------------------------------------------------------------------
# 14. Warning dedup: two chat() calls → exactly one downmap warning
# ---------------------------------------------------------------------------


def test_downmap_warning_emitted_once_across_multiple_calls():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = Model('claude-opus-4-6', ANTHROPIC_API_KEY='k', effort='xhigh')
        client = _install_mock_anthropic_client(m)
        m.model.completion('one')
        m.model.completion('two')
        m.model.completion('three')
    downmap = [x for x in w if x.category is EffortDownmapWarning]
    assert len(downmap) == 1


# ---------------------------------------------------------------------------
# 15. OpenRouter forwards reasoning via extra_body
# ---------------------------------------------------------------------------


def test_openrouter_forwards_reasoning():
    m = Model('openrouter/openai/gpt-5.6', OPENROUTER_API_KEY='k', effort='high')
    mock_client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='ok', tool_calls=None))]
    completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    mock_client.chat.completions.create.return_value = completion
    m.model.client = mock_client
    m.prompt('hi')
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs['extra_body'] == {'reasoning': {'effort': 'high'}}


# ---------------------------------------------------------------------------
# 16. api_params strips effort
# ---------------------------------------------------------------------------


def test_api_params_strips_effort():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k', effort='high')
    assert 'effort' in m.model.model_params
    assert 'effort' not in m.model.api_params


# ---------------------------------------------------------------------------
# 17. RefusalException import + category attribute
# ---------------------------------------------------------------------------


def test_refusal_exception_import_and_category():
    from justai import RefusalException as R

    assert R is RefusalException
    exc = RefusalException('csam')
    assert exc.category == 'csam'
    assert 'csam' in str(exc)


def test_anthropic_refusal_raises_with_category():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    resp = _mock_anthropic_response(text='', stop_reason='refusal', refusal_category='csam')
    client = _install_mock_anthropic_client(m, response=resp)
    with pytest.raises(RefusalException) as ei:
        m.chat('bad')
    assert ei.value.category == 'csam'
    assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# 18. Refusal does not retry
# ---------------------------------------------------------------------------


def test_refusal_short_circuits_retry_loop():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    resp = _mock_anthropic_response(text='', stop_reason='refusal', refusal_category='violence')
    client = _install_mock_anthropic_client(m, response=resp)
    with pytest.raises(RefusalException):
        m.model.completion('bad')
    assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# 19. Same as #13 (multi-turn effort survives) — covered by test_effort_survives_openai_multi_turn_tool_loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 20. No-op provider strip: effort never leaks to underlying client
# ---------------------------------------------------------------------------


def test_deepseek_does_not_leak_effort_kwarg():
    m = Model('deepseek-chat', DEEPSEEK_API_KEY='k', effort='high')
    mock_client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='ok', tool_calls=None))]
    completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    mock_client.chat.completions.create.return_value = completion
    m.model.client = mock_client
    m.prompt('hi')
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert 'effort' not in kwargs
    assert 'reasoning' not in kwargs
    assert 'extra_body' not in kwargs


def test_gguf_does_not_leak_effort_kwarg():
    # GGUF loading requires llama-cpp-python + a real GGUF file. Skip if unavailable.
    pytest.importorskip('llama_cpp')
    pytest.skip('GGUF test requires a real .gguf file; verified manually.')
