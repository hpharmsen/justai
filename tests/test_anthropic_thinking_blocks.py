"""Regression tests for AnthropicModel.chat() handling of non-text content blocks.

Extended-thinking models (e.g. claude-sonnet-5) return content lists like
[ThinkingBlock, TextBlock]. The chat() method must find the text block instead
of blindly indexing content[0].
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
