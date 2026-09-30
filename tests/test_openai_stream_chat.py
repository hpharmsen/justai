"""Tests that streaming chat on OpenAI works and keeps the conversation going.

With stream=True the SDK returns a Stream, which has no .id. The response id arrives inside
the stream, in the response.created event.

Usage:
    uv run pytest tests/test_openai_stream_chat.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from justai import Model
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
