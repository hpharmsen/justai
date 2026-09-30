"""Tests for streaming chat on OpenAI: conversation chaining, text-only deltas, usage and errors.

With stream=True the SDK returns a Stream, which has no .id. The response id arrives inside
the stream, in the response.created event.

Usage:
    uv run pytest tests/test_openai_stream_chat.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from justai import ConnectionException, Model, RatelimitException
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


class FakeStream:
    """Iterable of events without an id, like openai.Stream."""

    def __init__(self, response_id: str, words: list[str]):
        self.events = [
            SimpleNamespace(type='response.created', response=SimpleNamespace(id=response_id)),
            *[SimpleNamespace(type='response.output_text.delta', delta=w) for w in words],
        ]

    def __iter__(self):
        return iter(self.events)


async def _collect(model: Model, prompt: str) -> list[tuple[str, str]]:
    return [chunk async for chunk in model.chat_async_reasoning(prompt)]


def test_chat_async_reasoning_streams_and_chains_response_id():
    m = Model('gpt-5-mini')
    client = MagicMock()
    client.responses.create.side_effect = [FakeStream('resp_1', ['Hel', 'lo']), FakeStream('resp_2', ['Hi'])]
    m.model.client = client

    assert asyncio.run(_collect(m, 'hi')) == [('Hel', ''), ('lo', '')]
    asyncio.run(_collect(m, 'again'))

    assert client.responses.create.call_args_list[1].kwargs['previous_response_id'] == 'resp_1'


def _stream_model(*streams) -> tuple[Model, MagicMock]:
    m = Model('gpt-5-mini')
    client = MagicMock()
    client.responses.create.side_effect = list(streams)
    m.model.client = client
    return m, client


def test_function_call_argument_deltas_are_not_yielded_as_text():
    stream = FakeStream('resp_1', ['Hi'])
    stream.events.append(SimpleNamespace(type='response.function_call_arguments.delta', delta='{"x": 1}'))
    m, _ = _stream_model(stream)

    assert asyncio.run(_collect(m, 'hi')) == [('Hi', '')]


def test_streaming_records_token_usage():
    stream = FakeStream('resp_1', ['Hi'])
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, input_tokens_details=SimpleNamespace(cached_tokens=4))
    stream.events.append(SimpleNamespace(type='response.completed', response=SimpleNamespace(usage=usage)))
    m, _ = _stream_model(stream)

    asyncio.run(_collect(m, 'hi'))

    assert m.last_token_count() == (10, 5, 15)
    assert m.cache_read_input_tokens == 4


def test_api_error_on_create_is_mapped():
    request = httpx.Request('POST', 'https://api.openai.com/v1/responses')
    error = openai.RateLimitError('slow down', response=httpx.Response(429, request=request), body=None)
    m, _ = _stream_model(error)

    with pytest.raises(RatelimitException):
        asyncio.run(_collect(m, 'hi'))


def test_mid_stream_transport_error_is_mapped():
    class BrokenStream(FakeStream):
        def __iter__(self):
            yield from self.events
            raise httpx.ReadError('connection dropped')

    m, _ = _stream_model(BrokenStream('resp_1', ['Hi']))

    with pytest.raises(ConnectionException):
        asyncio.run(_collect(m, 'hi'))
