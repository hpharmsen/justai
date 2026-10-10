"""Tests that sampling params reach Anthropic through extra_body, on every create path.

anthropic 1.13 dropped temperature, top_p and top_k from Messages.create. The API still
accepts them, so justai sends them in extra_body. The mock clients here are autospecced on
the real SDK classes: a kwarg the SDK no longer accepts fails the test instead of vanishing
into a MagicMock.

Usage:
    uv run pytest tests/test_anthropic_sampling.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, create_autospec

import pydantic
import pytest
from anthropic.resources.messages import AsyncMessages, Messages

from justai import Model
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


class Answer(pydantic.BaseModel):
    answer: str


def _response(text: str = 'ok') -> MagicMock:
    block = MagicMock(type='text', text=text)
    resp = MagicMock(content=[block], stop_reason='end_turn')
    resp.usage = MagicMock(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return resp


class _EmptyAsyncStream:
    """Async context manager over an empty event stream, like AsyncStream."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _install_clients(m: Model, text: str = 'ok'):
    """Sync and async clients whose messages.create checks kwargs against the real SDK signature."""
    sync_messages = create_autospec(Messages, instance=True)
    sync_messages.create.return_value = _response(text)
    m.model.client = MagicMock(messages=sync_messages)

    async_spec = create_autospec(AsyncMessages, instance=True)

    async def async_create(**kwargs):
        async_spec.create(**kwargs)  # raises TypeError on a kwarg the SDK does not accept
        return _EmptyAsyncStream()

    m.model.async_client = MagicMock()
    m.model.async_client.messages.create = async_create
    return sync_messages.create, async_spec.create


def _run_path(m: Model, path: str) -> dict:
    """Drive one create path; return the kwargs the SDK received."""
    sync_create, async_create = _install_clients(m, text='{"answer": "ok"}' if path == 'structured' else 'ok')
    if path == 'prompt':
        m.prompt('x', cached=False)
    elif path == 'structured':
        m.prompt('x', response_format=Answer, cached=False)
    elif path == 'stream':
        m.model.completion('x', stream=True)
    elif path == 'agent_stream':

        async def run():
            return [c async for c in m.stream([{'role': 'user', 'content': 'x'}])]

        asyncio.run(run())
    create = async_create if path == 'agent_stream' else sync_create
    assert create.call_count == 1
    return create.call_args.kwargs


PATHS = ['prompt', 'structured', 'stream', 'agent_stream']


@pytest.mark.parametrize('path', PATHS)
def test_temperature_goes_in_extra_body(path):
    kwargs = _run_path(Model('claude-haiku-4-5', temperature=0), path)
    assert 'temperature' not in kwargs
    assert kwargs['extra_body']['temperature'] == 0


@pytest.mark.parametrize('path', PATHS)
def test_top_p_and_top_k_go_in_extra_body(path):
    kwargs = _run_path(Model('claude-haiku-4-5', top_p=0.9, top_k=40), path)
    assert kwargs['extra_body'] == {'top_p': 0.9, 'top_k': 40}


@pytest.mark.parametrize('path', PATHS)
def test_restricted_model_sends_no_temperature_anywhere(path):
    with pytest.warns(UserWarning, match='temperature'):
        m = Model('claude-sonnet-5', temperature=0.2)
    kwargs = _run_path(m, path)
    assert 'temperature' not in kwargs
    assert 'temperature' not in (kwargs.get('extra_body') or {})


@pytest.mark.parametrize('path', PATHS)
def test_caller_extra_body_is_kept(path):
    kwargs = _run_path(Model('claude-haiku-4-5', temperature=0, extra_body={'foo': 1}), path)
    assert kwargs['extra_body'] == {'foo': 1, 'temperature': 0}


@pytest.mark.parametrize('path', PATHS)
def test_plain_model_sends_no_extra_body(path):
    kwargs = _run_path(Model('claude-haiku-4-5'), path)
    assert 'extra_body' not in kwargs
