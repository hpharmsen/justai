"""Unit tests for the MiniMax provider.

Runs without a real MINIMAX_API_KEY: we pass a dummy key through kwargs
so ModelFactory + MiniMaxModel init resolve without touching the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from justai import Model
from justai.tools import cache


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    """Reset CacheDB singleton to a tmp directory per test."""
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


@pytest.mark.parametrize('model_name', ['MiniMax-M3', 'MiniMax-M2.7-highspeed', 'minimax-m3'])
def test_factory_routes_minimax_prefix_case_insensitively(model_name):
    from justai.models.modelfactory import ModelFactory
    from justai.models.minimax_models import MiniMaxModel

    model = ModelFactory.create(model_name, MINIMAX_API_KEY='test')
    assert isinstance(model, MiniMaxModel)


def test_minimax_client_base_url():
    from justai.models.minimax_models import MiniMaxModel

    model = MiniMaxModel('MiniMax-M3', params={'MINIMAX_API_KEY': 'test'})
    assert 'api.minimax.io' in str(model.client.base_url)


def test_m3_supports_images_and_m2_does_not():
    from justai.models.minimax_models import MiniMaxModel

    assert MiniMaxModel('MiniMax-M3', params={'MINIMAX_API_KEY': 'test'}).supports_image_input is True
    assert MiniMaxModel('MiniMax-M2.7-highspeed', params={'MINIMAX_API_KEY': 'test'}).supports_image_input is False


def test_image_input_on_text_only_model_raises():
    m = Model('MiniMax-M2.7-highspeed', MINIMAX_API_KEY='test')
    with pytest.raises(NotImplementedError):
        m.prompt('Wat zie je?', images=[b'not-a-real-image'], cached=False)


def test_prompt_sends_reasoning_split():
    """Without reasoning_split MiniMax inlines <think>...</think> into message.content."""
    m = Model('MiniMax-M3', MINIMAX_API_KEY='test')

    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='hoi', tool_calls=None))]
    completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = completion
    m.model.client = mock_client

    assert m.prompt('Hallo', cached=False) == 'hoi'
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == 'MiniMax-M3'
    assert kwargs['extra_body'] == {'reasoning_split': True}


def test_fenced_json_is_parsed():
    """MiniMax ignores response_format and always wraps JSON in a ```json fence."""
    m = Model('MiniMax-M3', MINIMAX_API_KEY='test')

    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='```json\n{"aantal": 3}\n```', tool_calls=None))]
    completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

    mock_client = MagicMock()
    # return_json routes through .parse, not .create
    mock_client.chat.completions.parse.return_value = completion
    m.model.client = mock_client

    assert m.prompt('Hoeveel?', return_json=True, cached=False) == {'aantal': 3}
    assert mock_client.chat.completions.parse.call_args.kwargs['extra_body'] == {'reasoning_split': True}
