"""Tests voor de Anthropic cache-breakpoints.

Prompt caching is een prefix-match: één byte verschil ergens in de prefix maakt alles
erna ongeldig. Deze tests bewaken de twee dingen die daarbij stil kunnen misgaan, zonder
foutmelding en zonder dat je het aan de response ziet:

  1. dat het breakpoint op de juiste plek staat, en
  2. dat we de messages van de aanroeper niet muteren.

Draaien zonder API-key.

Usage:
    uv run pytest tests/test_prompt_caching.py
"""

from __future__ import annotations

import asyncio
import copy
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from justai import Model
from justai.models.anthropic_models import AnthropicModel, apply_cache_control
from justai.models.basemodel import GeneralException
from justai.tools import cache

BREAKPOINT = {'type': 'ephemeral'}


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    """Reset de CacheDB-singleton naar een tmp-map, zodat tests elkaar niet vervuilen."""
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


def _blocks(text):
    return [{'type': 'text', 'text': text}]


def _count_breakpoints(system, messages) -> int:
    found = 0
    if isinstance(system, list):
        found += sum('cache_control' in b for b in system)
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            found += sum(isinstance(b, dict) and 'cache_control' in b for b in content)
    return found


# ---------------------------------------------------------------------------
# 1. Systeem-breakpoint
# ---------------------------------------------------------------------------


def test_string_system_becomes_block_with_breakpoint():
    system, _ = apply_cache_control('een systeemprompt', [{'role': 'user', 'content': 'hi'}])
    assert system == [{'type': 'text', 'text': 'een systeemprompt', 'cache_control': BREAKPOINT}]


def test_breakpoint_lands_on_last_system_block():
    # De rendervolgorde is tools, system, messages. Een breakpoint op het laatste
    # systeemblok cachet dus tools en systeemprompt samen.
    system, _ = apply_cache_control(_blocks('een') + _blocks('twee'), [{'role': 'user', 'content': 'hi'}])
    assert 'cache_control' not in system[0]
    assert system[1]['cache_control'] == BREAKPOINT


def test_existing_system_breakpoint_is_not_duplicated():
    # cached_system_message() zet er zelf al een op het laatste blok.
    model = AnthropicModel('claude-sonnet-4-6', {'ANTHROPIC_API_KEY': 'k'})
    model.cached_prompt = 'een lange vaste tekst'
    system, _ = apply_cache_control(model.cached_system_message(), [{'role': 'user', 'content': 'hi'}])
    assert sum('cache_control' in b for b in system) == 1


def test_empty_system_is_left_alone():
    # Een leeg tekstblok meesturen levert een 400 op.
    system, _ = apply_cache_control('', [{'role': 'user', 'content': 'hi'}])
    assert system == ''


# ---------------------------------------------------------------------------
# 2. Geschiedenis-breakpoint
# ---------------------------------------------------------------------------


def test_no_history_breakpoint_on_single_message():
    # Een cache write kost 1,25x. Bij één message is er niets om te hergebruiken,
    # dus zou het breakpoint puur verlies zijn.
    _, messages = apply_cache_control('sys', [{'role': 'user', 'content': 'hi'}])
    assert _count_breakpoints(None, messages) == 0


def test_history_breakpoint_from_second_message():
    msgs = [
        {'role': 'user', 'content': 'een'},
        {'role': 'assistant', 'content': _blocks('twee')},
        {'role': 'user', 'content': _blocks('drie')},
    ]
    _, out = apply_cache_control('sys', msgs)
    assert out[-1]['content'][-1]['cache_control'] == BREAKPOINT
    assert _count_breakpoints(None, out) == 1


def test_string_content_becomes_block_with_breakpoint():
    msgs = [{'role': 'user', 'content': 'een'}, {'role': 'assistant', 'content': 'twee'}]
    _, out = apply_cache_control('sys', msgs)
    assert out[-1]['content'] == [{'type': 'text', 'text': 'twee', 'cache_control': BREAKPOINT}]


def test_breakpoint_on_tool_result_block():
    # De agent-loop eindigt elke iteratie op een user-message met tool_result-blokken.
    msgs = [
        {'role': 'user', 'content': 'een'},
        {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'f', 'input': {}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'ok'}]},
    ]
    _, out = apply_cache_control('sys', msgs)
    assert out[-1]['content'][-1]['cache_control'] == BREAKPOINT


def test_empty_content_list_is_left_alone():
    msgs = [{'role': 'user', 'content': 'een'}, {'role': 'assistant', 'content': []}]
    _, out = apply_cache_control('sys', msgs)
    assert out[-1]['content'] == []


# ---------------------------------------------------------------------------
# 3. Copy-on-write
# ---------------------------------------------------------------------------


def test_input_messages_are_not_mutated():
    # De agent hergebruikt zijn messages-lijst tussen iteraties. Muteren verandert de
    # bytes van de prefix bij de volgende request, waarna de cache stil mist.
    original = [
        {'role': 'user', 'content': 'een'},
        {'role': 'assistant', 'content': _blocks('twee')},
    ]
    snapshot = copy.deepcopy(original)
    apply_cache_control('sys', original)
    assert original == snapshot


def test_input_system_list_is_not_mutated():
    original = _blocks('een')
    snapshot = copy.deepcopy(original)
    apply_cache_control(original, [{'role': 'user', 'content': 'hi'}])
    assert original == snapshot


def test_untouched_messages_are_shared_not_copied():
    # Alleen de laatste message wordt gekopieerd; de rest mag dezelfde objecten blijven.
    msgs = [{'role': 'user', 'content': 'een'}, {'role': 'assistant', 'content': _blocks('twee')}]
    _, out = apply_cache_control('sys', msgs)
    assert out[0] is msgs[0]
    assert out[-1] is not msgs[-1]


# ---------------------------------------------------------------------------
# 4. TTL en uitschakelen
# ---------------------------------------------------------------------------


def test_ttl_1h_is_sent():
    system, _ = apply_cache_control('sys', [{'role': 'user', 'content': 'hi'}], ttl='1h')
    assert system[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}


def test_default_ttl_is_not_sent():
    # 5m is de default van de API; meesturen voegt niets toe.
    system, _ = apply_cache_control('sys', [{'role': 'user', 'content': 'hi'}], ttl='5m')
    assert system[-1]['cache_control'] == BREAKPOINT


def test_disabled_returns_input_unchanged():
    msgs = [{'role': 'user', 'content': 'een'}, {'role': 'user', 'content': 'twee'}]
    system, out = apply_cache_control('sys', msgs, enabled=False)
    assert system == 'sys'
    assert out is msgs


def test_never_more_than_four_breakpoints():
    # De API staat er maximaal vier toe. Wij zetten er twee.
    msgs = [{'role': 'user', 'content': f'beurt {i}'} for i in range(10)]
    system, out = apply_cache_control(_blocks('een') + _blocks('twee'), msgs)
    assert _count_breakpoints(system, out) <= 4


# ---------------------------------------------------------------------------
# 5. Komt het ook echt op de wire terecht?
# ---------------------------------------------------------------------------


def _mock_response():
    resp = MagicMock()
    block = MagicMock()
    block.text = 'ok'
    block.type = 'text'
    resp.content = [block]
    resp.stop_reason = 'end_turn'
    resp.usage = MagicMock(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return resp


def _install_mock_client(model, resp=None):
    resp = resp or _mock_response()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = resp
    model.model.client = mock_client
    return mock_client


def test_completion_sends_system_breakpoint():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    client = _install_mock_client(m)
    m.model.completion('hallo')
    system = client.messages.create.call_args.kwargs['system']
    assert isinstance(system, list)
    assert system[-1]['cache_control'] == BREAKPOINT


# ---------------------------------------------------------------------------
# 6. De twee schakelaars
# ---------------------------------------------------------------------------


def test_cache_ttl_param_reaches_the_request():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k', cache_ttl='1h')
    client = _install_mock_client(m)
    m.model.completion('hallo')
    system = client.messages.create.call_args.kwargs['system']
    assert system[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}


def test_prompt_cache_false_sends_no_breakpoints():
    # Ontsnappingsluik voor wie alleen one-shot calls doet en de 1,25x write-premie
    # niet wil betalen op een cache die nooit gelezen wordt.
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k', prompt_cache=False)
    client = _install_mock_client(m)
    m.model.completion('beurt een')
    m.model.completion('beurt twee')
    kwargs = client.messages.create.call_args.kwargs
    assert 'cache_control' not in str(kwargs['system'])
    assert 'cache_control' not in str(kwargs['messages'])


@pytest.mark.parametrize('param', ['cache_ttl', 'prompt_cache'])
def test_switches_do_not_leak_into_the_api_call(param):
    # model_params gaan standaard mee de provider-API in; deze twee horen daar niet.
    value = '1h' if param == 'cache_ttl' else False
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k', **{param: value})
    client = _install_mock_client(m)
    m.model.completion('hallo')
    assert param not in client.messages.create.call_args.kwargs


def test_completion_sends_history_breakpoint_on_second_turn():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    client = _install_mock_client(m)
    m.model.completion('beurt een')
    m.model.completion('beurt twee')
    messages = client.messages.create.call_args.kwargs['messages']
    assert len(messages) > 1
    assert messages[-1]['content'][-1]['cache_control'] == BREAKPOINT


def test_completion_does_not_corrupt_stored_history():
    # completion() bewaart self.messages tussen beurten. Als het breakpoint daarin
    # blijft hangen, verschuift het bij elke beurt mee en groeit het aantal.
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    _install_mock_client(m)
    m.model.completion('beurt een')
    m.model.completion('beurt twee')
    assert 'cache_control' not in str(m.model.messages)


class _StopStream(Exception):
    """Sentinel om de stream af te breken zodra de request-params vastliggen."""


def _capture_stream_params(model, messages):
    """Draai stream() tot de request-params vastliggen en geef ze terug.

    De suite gebruikt geen pytest-asyncio; conventie is asyncio.run in een gewone test.
    """
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        raise _StopStream

    model.model.async_client = MagicMock()
    model.model.async_client.messages.create = AsyncMock(side_effect=fake_create)

    async def drain():
        async for _ in model.model.stream(messages):
            pass

    # De foutmapper van de provider verpakt _StopStream als GeneralException.
    with pytest.raises(GeneralException):
        asyncio.run(drain())
    return captured


def test_stream_sends_breakpoints():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    captured = _capture_stream_params(
        m,
        [
            {'role': 'system', 'content': 'de systeemprompt'},
            {'role': 'user', 'content': 'een'},
            {'role': 'assistant', 'content': _blocks('twee')},
            {'role': 'user', 'content': _blocks('drie')},
        ],
    )
    assert isinstance(captured['system'], list)
    assert captured['system'][-1]['cache_control'] == BREAKPOINT
    assert captured['messages'][-1]['content'][-1]['cache_control'] == BREAKPOINT


def test_stream_does_not_mutate_callers_messages():
    # De Agent hergebruikt deze lijst elke iteratie; muteren breekt de volgende prefix-match.
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    messages = [
        {'role': 'system', 'content': 'de systeemprompt'},
        {'role': 'user', 'content': 'een'},
        {'role': 'assistant', 'content': _blocks('twee')},
    ]
    snapshot = copy.deepcopy(messages)
    _capture_stream_params(m, messages)
    assert messages == snapshot


# ---------------------------------------------------------------------------
# 7. Tegen de echte API (opt-in)
# ---------------------------------------------------------------------------


LONG_TEXT = 'Achtergronddocument.\n' + '\n'.join(
    f'Regel {i}: kade {i % 7} verwerkte {i * 13} containers, ploegleider '
    f'{"Anna" if i % 2 else "Bram"}, wachttijd {i % 45} minuten.'
    for i in range(220)
)


@pytest.mark.skipif(
    not os.getenv('RUN_LIVE_TESTS'),
    reason='kost echte tokens; zet RUN_LIVE_TESTS=1 om te draaien (sleutel komt uit .env)',
)
def test_live_second_turn_reads_the_cache():
    # De enige manier om te bewijzen dat de breakpoints op de goede plek staan is
    # het de API laten zeggen. Onder de drempel (1024 tokens voor sonnet-4-6)
    # cachet die stil niets, vandaar de lengte van LONG_TEXT.
    m = Model('claude-sonnet-4-6', max_tokens=20)
    m.system_message = 'Antwoord met maximaal drie woorden.'
    m.cached_prompt = LONG_TEXT

    m.chat('Wie was ploegleider in regel 4?', cached=False)  # cached=False omzeilt de lokale cache
    first_total = m.input_token_count + m.cache_read_input_tokens + m.cache_creation_input_tokens

    m.chat('En in regel 5?', cached=False)
    assert m.cache_read_input_tokens > 1000
    # De tweede beurt betaalt bijna niets tegen vol tarief.
    assert m.input_token_count < first_total / 10
