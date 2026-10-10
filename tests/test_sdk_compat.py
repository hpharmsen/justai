"""Tests that justai works with anthropic 1.x and openai 3.x.

Both SDKs moved from httpx to httpx2. anthropic 1.x rejects an httpx.Client as http_client
and an httpx.Timeout as timeout, at construction and per request. The Responses API takes
the output cap as max_output_tokens; max_tokens from the model params has to be forwarded
under that name.

Usage:
    uv run pytest tests/test_sdk_compat.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import anthropic
import httpx2
import openai
import pytest
import tomllib

from justai import ConnectionException, Model
from justai.models.anthropic_models import _map_anthropic_error
from justai.models.openai_completions import map_openai_error
from justai.tools import cache

PYPROJECT = Path(__file__).parent.parent / 'pyproject.toml'

ANTHROPIC_SSE = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],'
    b'"model":"claude-sonnet-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b'event: message_stop\n'
    b'data: {"type":"message_stop"}\n\n'
)

OPENAI_RESPONSE = {
    'id': 'resp_1',
    'object': 'response',
    'created_at': 0,
    'model': 'gpt-6-luna',
    'status': 'completed',
    'parallel_tool_calls': False,
    'tool_choice': 'auto',
    'tools': [],
    'output': [
        {
            'type': 'message',
            'id': 'msg_1',
            'role': 'assistant',
            'status': 'completed',
            'content': [{'type': 'output_text', 'text': 'hoi', 'annotations': []}],
        }
    ],
    'usage': {
        'input_tokens': 1,
        'output_tokens': 1,
        'total_tokens': 2,
        'input_tokens_details': {'cached_tokens': 0},
        'output_tokens_details': {'reasoning_tokens': 0},
    },
}


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


def _read_timeout(client) -> float:
    """Read timeout the SDK client applies to every request."""
    t = client.timeout
    return t if isinstance(t, (int, float)) else t.read


# ---------------------------------------------------------------------------
# Anthropic: client construction and timeouts
# ---------------------------------------------------------------------------


def test_anthropic_sync_model_constructs_with_timeout():
    m = Model('claude-sonnet-5', timeout=20)
    assert _read_timeout(m.model.client) == 20
    assert _read_timeout(m.model.async_client) == 20


def test_anthropic_async_model_constructs_with_timeout():
    m = Model('claude-sonnet-5', timeout=20, **{'async': True})
    assert isinstance(m.model.client, anthropic.AsyncAnthropic)
    assert _read_timeout(m.model.async_client) == 20


def test_anthropic_sync_stream_sends_sdk_compatible_timeout():
    seen = []

    def handler(request):
        seen.append(request.extensions['timeout'])
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=ANTHROPIC_SSE)

    m = Model('claude-sonnet-5', timeout=20)
    m.model.client = anthropic.Anthropic(
        api_key='test', http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )

    list(m.model.completion('x', stream=True))

    assert seen and seen[0]['read'] == 20


def test_anthropic_async_stream_sends_sdk_compatible_timeout():
    seen = []

    def handler(request):
        seen.append(request.extensions['timeout'])
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=ANTHROPIC_SSE)

    m = Model('claude-sonnet-5', timeout=20)
    m.model.async_client = anthropic.AsyncAnthropic(
        api_key='test', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )

    async def run():
        return [chunk async for chunk in m.stream([{'role': 'user', 'content': 'x'}])]

    asyncio.run(run())

    assert seen and seen[0]['read'] == 20


# ---------------------------------------------------------------------------
# OpenAI Responses: max_tokens forwarded as max_output_tokens, no temperature
# ---------------------------------------------------------------------------


def _responses_body(m: Model) -> dict:
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=OPENAI_RESPONSE)

    m.model.client = openai.OpenAI(api_key='test', http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert m.prompt('x', cached=False) == 'hoi'
    assert len(bodies) == 1
    return bodies[0]


def test_responses_forwards_max_tokens_as_max_output_tokens():
    body = _responses_body(Model('gpt-6-luna', max_tokens=8192))
    assert body.get('max_output_tokens') == 8192
    assert 'max_tokens' not in body
    assert 'temperature' not in body


def test_responses_without_max_tokens_sends_no_max_output_tokens():
    body = _responses_body(Model('gpt-6-luna'))
    assert 'max_output_tokens' not in body
    assert 'temperature' not in body


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_pyproject_dependencies():
    deps = tomllib.loads(PYPROJECT.read_text())['project']['dependencies']
    names = [d.split(';')[0].strip() for d in deps]
    assert not any(n.startswith('instructor') for n in names)
    assert 'anthropic>=1' in names
    assert 'openai>=3' in names


def test_import_without_future_or_deprecation_warnings():
    result = subprocess.run(
        [sys.executable, '-W', 'error::FutureWarning', '-W', 'error::DeprecationWarning', '-c', 'import justai'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Mid-stream transport errors: both SDKs now raise httpx2 errors, not httpx errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('error', [httpx2.ReadError('dropped'), httpx2.RemoteProtocolError('closed')])
@pytest.mark.parametrize('mapper', [_map_anthropic_error, map_openai_error])
def test_httpx2_transport_error_maps_to_connection_exception(mapper, error):
    assert isinstance(mapper(error), ConnectionException)
