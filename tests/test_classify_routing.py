"""Provider routing, transport and retries for System One classification.

httpx is mocked throughout, so nothing leaves the machine, except for the two smoke
tests at the bottom which skip without a key.

Usage:
    venv/bin/pytest tests/test_classify_routing.py
"""

import os

import httpx
import pytest
from dotenv import load_dotenv

# The smoke tests below skip on a missing key, so the key has to be visible at collection
# time. Without this they silently skip even when .env has one.
load_dotenv()

import justai.tools.cache as cache
from justai import Model, BadRequestException, ModelOverloadException, RatelimitException
from justai.models.openrouter_models import OpenRouterModel
from justai.models.systemone import SystemOneModel

CHOICE_BODY = {
    'id': 'dec_1',
    'model': 'jev-latest',
    'answers': {'answer': {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': 1.0}, 'confidence': 1.0}},
    'usage': {'input_tokens': 476, 'output_tokens': 70, 'cost': 0.000019992},
}
OPTIONS = {'billing': 'about money', 'technical': 'about bugs'}
ASK = 'What is this about?'


@pytest.fixture(autouse=True)
def isolate_cache_and_warnings(monkeypatch, tmp_path):
    """Reset CacheDB singleton to a tmp directory + clear warning-dedup state per test."""
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Record backoff durations instead of serving them."""
    slept = []
    monkeypatch.setattr('justai.models.systemone.time.sleep', slept.append)
    return slept


class Transport:
    """Queues responses and records every request the mixin posts."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({'url': url, 'json': json, 'headers': headers or {}, 'timeout': timeout})
        status, body = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        request = httpx.Request('POST', url)
        return httpx.Response(status, json=body, request=request)


@pytest.fixture
def transport(monkeypatch):
    def install(*responses):
        stub = Transport(*(responses or [(200, CHOICE_BODY)]))
        monkeypatch.setattr(httpx.Client, 'post', lambda self, url, **kw: stub(url, **kw))
        return stub

    return install


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    for name in ('OPENROUTER_API_KEY', 'TYPESAFE_API_KEY', 'SYSTEMONE_API_KEY'):
        monkeypatch.setenv(name, f'test-{name.lower()}')
    monkeypatch.setattr('justai.models.basemodel.dotenv_values', dict)


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


def test_factory_routes_systemone_prefix():
    assert isinstance(Model('systemone/kev-3b', base_url='http://localhost:8000').model, SystemOneModel)


def test_factory_routes_jev_prefix():
    assert isinstance(Model('jev-1.13.0').model, SystemOneModel)


def test_factory_keeps_openrouter_first():
    assert isinstance(Model('openrouter/typesafe/jev-1.13').model, OpenRouterModel)


def test_systemone_requires_base_url():
    with pytest.raises(AssertionError):
        Model('systemone/kev-3b')


def test_systemone_works_without_api_key(monkeypatch):
    monkeypatch.delenv('SYSTEMONE_API_KEY', raising=False)
    model = Model('systemone/kev-3b', base_url='http://localhost:8000')
    assert model.model.api_key is None


# ---------------------------------------------------------------------------
# URLs, headers and body
# ---------------------------------------------------------------------------


def test_systemone_posts_to_base_url(transport):
    stub = transport()
    Model('systemone/kev-3b', base_url='http://localhost:8000').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['url'] == 'http://localhost:8000/v1/systemone'


def test_base_url_trailing_slash_is_handled(transport):
    stub = transport()
    Model('systemone/kev-3b', base_url='http://localhost:8000/').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['url'] == 'http://localhost:8000/v1/systemone'


def test_jev_posts_to_typesafe(transport):
    stub = transport()
    Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['url'] == 'https://api.typesafe.ai/v1/systemone'


def test_openrouter_posts_to_systemone(transport):
    stub = transport()
    Model('openrouter/typesafe/jev-1.13').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['url'] == 'https://openrouter.ai/api/v1/systemone'


def test_openrouter_sends_bearer_key(transport):
    stub = transport()
    Model('openrouter/typesafe/jev-1.13').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['headers']['Authorization'] == 'Bearer test-openrouter_api_key'


def test_openrouter_sends_full_slug(transport):
    stub = transport()
    Model('openrouter/typesafe/jev-1.13').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['json']['model'] == 'typesafe/jev-1.13'


def test_systemone_sends_bare_model_name(transport):
    stub = transport()
    Model('systemone/kev-3b', base_url='http://localhost:8000').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['json']['model'] == 'kev-3b'


def test_base_url_and_key_stay_out_of_body(transport):
    stub = transport()
    Model('systemone/kev-3b', base_url='http://localhost:8000').classify('hi', OPTIONS, instructions=ASK, cached=False)
    body = str(stub.calls[0]['json'])
    assert 'base_url' not in body and 'localhost' not in body and 'test-systemone_api_key' not in body


def test_classify_uses_a_short_default_timeout(transport):
    stub = transport()
    Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert stub.calls[0]['timeout'].read == 30.0


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_classify_records_usage(transport):
    transport()
    model = Model('jev-latest')
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert model.last_token_count()[:2] == (476, 70)


def test_classify_records_usage_before_unpack(transport):
    """Tokens are billed even when the body turns out to be unusable."""
    transport((200, {'usage': {'input_tokens': 12, 'output_tokens': 3}}))
    model = Model('jev-latest')
    with pytest.raises(Exception):
        model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert model.last_token_count()[:2] == (12, 3)


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def test_retries_on_429_then_succeeds(transport):
    stub = transport((429, {}), (429, {}), (200, CHOICE_BODY))
    result = Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert result['choice'] == 'billing'
    assert len(stub.calls) == 3


def test_retries_on_529(transport):
    stub = transport((529, {}), (200, CHOICE_BODY))
    Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert len(stub.calls) == 2


def test_retries_give_up_as_ratelimit(transport):
    stub = transport((429, {}))
    with pytest.raises(RatelimitException):
        Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert len(stub.calls) == 3


def test_retries_give_up_as_overload(transport):
    stub = transport((529, {}))
    with pytest.raises(ModelOverloadException):
        Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert len(stub.calls) == 3


def test_no_retry_on_422(transport):
    stub = transport((422, {}))
    with pytest.raises(BadRequestException):
        Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert len(stub.calls) == 1


def test_max_retries_zero_does_one_call(transport):
    stub = transport((429, {}))
    with pytest.raises(RatelimitException):
        Model('jev-latest', max_retries=0).classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert len(stub.calls) == 1


def test_backoff_is_exponential(transport, no_sleep):
    transport((429, {}))
    with pytest.raises(RatelimitException):
        Model('jev-latest').classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert no_sleep == [2, 4]


# ---------------------------------------------------------------------------
# Capabilities and lifecycle
# ---------------------------------------------------------------------------


def test_systemone_stubs_prompt():
    with pytest.raises(NotImplementedError):
        Model('jev-latest').prompt('hello')


def test_systemone_reports_no_text_capabilities():
    model = Model('jev-latest').model
    assert not model.supports_return_json
    assert not model.supports_image_input
    assert not model.supports_tool_use


def test_close_closes_http_client(transport):
    transport()
    model = Model('jev-latest')
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    model.close()
    assert model.model._systemone_client.is_closed


def test_openrouter_close_still_closes_openai_client(transport):
    """The mixin overrides close(), so the OpenAI client must not be left open."""
    transport()
    model = Model('openrouter/typesafe/jev-1.13')
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    model.close()
    assert model.model._systemone_client.is_closed


# ---------------------------------------------------------------------------
# Smoke tests against the real endpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.getenv('TYPESAFE_API_KEY'), reason='needs TYPESAFE_API_KEY')
def test_typesafe_smoke(monkeypatch):
    monkeypatch.undo()
    answer = Model('jev-latest').classify('I was charged twice for my subscription.', OPTIONS, instructions=ASK, cached=False)
    assert answer['choice'] in OPTIONS


@pytest.mark.skipif(not os.getenv('OPENROUTER_API_KEY'), reason='needs OPENROUTER_API_KEY')
def test_openrouter_smoke_string_state(monkeypatch):
    """Settles whether the OpenRouter route accepts a plain string as state."""
    monkeypatch.undo()
    answer = Model('openrouter/typesafe/jev-1.13').classify(
        'I was charged twice for my subscription.', OPTIONS, instructions=ASK, cached=False
    )
    assert answer['choice'] in OPTIONS


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
