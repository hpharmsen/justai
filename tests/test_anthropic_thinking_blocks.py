"""Regression tests for AnthropicModel.chat() handling of non-text content blocks.

Extended-thinking models (e.g. claude-sonnet-5) return content lists like
[ThinkingBlock, TextBlock]. The chat() method must find the text block instead
of blindly indexing content[0].
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from justai import Model
from justai.models.basemodel import BadRequestException
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


def _block(block_type: str, **attrs) -> MagicMock:
    b = MagicMock()
    b.type = block_type
    for k, v in attrs.items():
        setattr(b, k, v)
    return b


def _mock_response(content_blocks: list, stop_reason: str = 'end_turn') -> MagicMock:
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return resp


def _install_client(model, response) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response
    client.messages.parse.return_value = response
    model.model.client = client
    return client


def test_thinking_block_before_text_returns_text():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    thinking = _block('thinking', thinking='let me reason...')
    text = _block('text', text='the answer is 42')
    _install_client(m, _mock_response([thinking, text]))

    response, _, _ = m.model.prompt('hi')

    assert response == 'the answer is 42'


def test_thinking_block_before_json_text_returns_parsed_json():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    thinking = _block('thinking', thinking='reasoning about the shape...')
    text = _block('text', text='{"answer": 42}')
    _install_client(m, _mock_response([thinking, text]))

    response, _, _ = m.model.prompt('hi', return_json=True)

    assert response == {'answer': 42}


def test_multiple_text_blocks_are_joined():
    """Citations and mid-stream fallbacks split one answer over several text blocks."""
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    blocks = [
        _block('thinking', thinking='reasoning...'),
        _block('text', text='the answer '),
        _block('text', text='is 42'),
    ]
    _install_client(m, _mock_response(blocks))

    response, _, _ = m.model.prompt('hi')

    assert response == 'the answer is 42'


def test_no_text_block_raises_bad_request():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    only_thinking = _block('thinking', thinking='never got to a text block')
    _install_client(m, _mock_response([only_thinking]))

    with pytest.raises(BadRequestException):
        m.model.prompt('hi')


def test_non_json_response_raises_bad_request_not_value_error():
    """Regression: previously raised raw ValueError, crashing concurrent callers."""
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    text = _block('text', text='[Analyseer plattegrond]\n\nHet dakterras...')
    _install_client(m, _mock_response([text]))

    with pytest.raises(BadRequestException):
        m.model.prompt('hi', return_json=True)


class _Obj:
    """Attribute bag. Not a MagicMock: hasattr() must be able to return False."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _collect(async_gen):
    async def run():
        return [chunk async for chunk in async_gen]

    return asyncio.run(run())


def test_chat_async_skips_thinking_deltas():
    """Streaming yields text deltas only; thinking deltas carry .thinking, not .text."""
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    events = [
        _Obj(delta=_Obj(type='thinking_delta', thinking='let me reason...')),
        _Obj(delta=_Obj(type='text_delta', text='the answer is 42')),
    ]
    client = MagicMock()
    client.messages.create.return_value = events
    m.model.client = client

    assert _collect(m.model.chat_async('hi')) == [('the answer is 42', None)]


def test_stateless_stream_skips_thinking_deltas():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    events = [
        _Obj(type='message_start', message=_Obj(usage=_Obj(input_tokens=7))),
        _Obj(type='content_block_delta', delta=_Obj(type='thinking_delta', thinking='hmm')),
        _Obj(type='content_block_delta', delta=_Obj(type='text_delta', text='the answer is 42')),
        _Obj(type='message_delta', usage=_Obj(output_tokens=3)),
    ]
    m.model.async_client = MagicMock()
    m.model.async_client.messages.create = AsyncMock(return_value=_FakeAsyncStream(events))

    chunks = _collect(m.model.stream([{'role': 'user', 'content': 'hi'}]))

    assert [(c.type, c.content) for c in chunks if c.type == 'text'] == [('text', 'the answer is 42')]
    assert chunks[-1].type == 'done' and chunks[-1].input_tokens == 7
