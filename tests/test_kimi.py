"""Unit tests for the Kimi (Moonshot) provider.

Runs without a real MOONSHOT_API_KEY: we pass a dummy key through kwargs
so ModelFactory + KimiModel init resolve without touching the network.
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


def test_factory_routes_kimi_prefix():
    from justai.models.modelfactory import ModelFactory
    from justai.models.kimi_models import KimiModel
    model = ModelFactory.create('kimi-k2.6', MOONSHOT_API_KEY='test')
    assert isinstance(model, KimiModel)


def test_factory_routes_moonshot_prefix():
    from justai.models.modelfactory import ModelFactory
    from justai.models.kimi_models import KimiModel
    model = ModelFactory.create('moonshot-v1-8k', MOONSHOT_API_KEY='test')
    assert isinstance(model, KimiModel)


def test_kimi_client_base_url():
    from justai.models.kimi_models import KimiModel
    model = KimiModel('kimi-k2.6', params={'MOONSHOT_API_KEY': 'test'})
    assert 'api.moonshot.ai' in str(model.client.base_url)


def test_kimi_supports_function_calling_and_images():
    from justai.models.kimi_models import KimiModel
    model = KimiModel('kimi-k2.6', params={'MOONSHOT_API_KEY': 'test'})
    assert model.supports_function_calling is True
    assert model.supports_image_input is True


def test_kimi_temperature_is_clamped_to_one():
    from justai.models.kimi_models import KimiModel
    model = KimiModel('kimi-k2.6', params={'MOONSHOT_API_KEY': 'test', 'temperature': 1.7})
    assert model.api_params['temperature'] == 1.0


def test_kimi_temperature_under_one_is_preserved():
    from justai.models.kimi_models import KimiModel
    model = KimiModel('kimi-k2.6', params={'MOONSHOT_API_KEY': 'test', 'temperature': 0.4})
    assert model.api_params['temperature'] == 0.4


def test_kimi_prompt_hits_completions_create():
    """Full round-trip through Model → KimiModel with a mocked OpenAI client."""
    m = Model('kimi-k2.6', MOONSHOT_API_KEY='test')

    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content='hoi', tool_calls=None))]
    completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = completion
    m.model.client = mock_client

    result = m.prompt('Hallo')
    assert result == 'hoi'
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs['model'] == 'kimi-k2.6'
