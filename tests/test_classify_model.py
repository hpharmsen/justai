"""Model.classify: caching, counters, timing and the unsupported-model path.

The provider layer is stubbed out, so these tests say nothing about the wire format;
that is tests/test_classify.py.

Usage:
    venv/bin/pytest tests/test_classify_model.py
"""

import pytest

import justai.tools.cache as cache
from justai import Model
from justai.models.basemodel import BaseModel

OPTIONS = {'billing': 'about money', 'technical': 'about bugs'}
ASK = 'What is this about?'
ANSWER = {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': 0.9, 'technical': 0.1}, 'confidence': 0.9}
SCORE_ANSWER = {
    'type': 'score',
    'score': 1.04,
    'legend': {0: 'calm', 1: 'annoyed', 2: 'furious'},
    'probabilities': {0: 0.01, 1: 0.95, 2: 0.04},
    'confidence': 0.93,
}


@pytest.fixture(autouse=True)
def isolate_cache_and_warnings(monkeypatch, tmp_path):
    """Reset CacheDB singleton to a tmp directory + clear warning-dedup state per test."""
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setenv('TYPESAFE_API_KEY', 'test-key')
    monkeypatch.setattr('justai.models.basemodel.dotenv_values', dict)


class Provider:
    """Stands in for SystemOneModel.classify and counts how often it is reached."""

    def __init__(self, answer=ANSWER, tokens=(476, 70), fail=False):
        self.answer, self.tokens, self.fail = answer, tokens, fail
        self.calls = 0

    def __call__(self, model):
        def classify(state, options=None, *, instructions=None, questions=None):
            self.calls += 1
            model.model.record_usage(*self.tokens)
            if self.fail:
                raise KeyError('answers')
            return self.answer, *self.tokens

        model.model.classify = classify
        return model


@pytest.fixture
def model():
    def build(**kwargs):
        return Provider(**kwargs), None

    return build


def jev(provider):
    return provider(Model('jev-latest'))


# ---------------------------------------------------------------------------
# Unsupported models
# ---------------------------------------------------------------------------


def test_basemodel_classify_raises_with_class_name():
    class Dummy(BaseModel):
        def __init__(self):
            pass

        def prompt(self, *a, **kw): ...
        def chat(self, *a, **kw): ...
        async def prompt_async(self, *a, **kw): ...
        async def chat_async(self, *a, **kw): ...

        def token_count(self, text):
            return 0

    with pytest.raises(NotImplementedError) as excinfo:
        Dummy().classify('x', OPTIONS)
    assert 'classify() is not supported by Dummy' in str(excinfo.value)


def test_claude_classify_raises(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    with pytest.raises(NotImplementedError) as excinfo:
        Model('claude-sonnet-5').classify('x', OPTIONS, instructions=ASK)
    assert 'classify() is not supported by AnthropicModel' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Result, counters and timing
# ---------------------------------------------------------------------------


def test_classify_returns_dict():
    provider = Provider()
    result = jev(provider).classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert result == ANSWER


def test_classify_fills_token_counts():
    model = jev(Provider())
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert model.last_token_count() == (476, 70, 546)


def test_classify_sets_response_time():
    model = jev(Provider())
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert model.last_response_time > 0


def test_classify_keeps_tokens_on_failure():
    """A provider that fails after record_usage still spent those tokens."""
    model = jev(Provider(fail=True))
    with pytest.raises(KeyError):
        model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert model.last_token_count()[:2] == (476, 70)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_classify_caches_by_default():
    provider = Provider()
    model = jev(provider)
    assert model.classify('hi', OPTIONS, instructions=ASK) == ANSWER
    assert model.classify('hi', OPTIONS, instructions=ASK) == ANSWER
    assert provider.calls == 1


def test_classify_cached_false_skips_cache():
    provider = Provider()
    model = jev(provider)
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    model.classify('hi', OPTIONS, instructions=ASK, cached=False)
    assert provider.calls == 2


def test_classify_cache_survives_dict_result():
    """sqlite binds strings, so the answer goes in serialised and must come back a dict."""
    model = jev(Provider())
    model.classify('hi', OPTIONS, instructions=ASK)
    assert model.classify('hi', OPTIONS, instructions=ASK) == ANSWER


def test_classify_cache_keeps_int_score_keys():
    """json.loads restringifies the level keys, so the hit path has to coerce them back."""
    model = jev(Provider(answer=SCORE_ANSWER))
    model.classify('hi', ['calm', 'annoyed', 'furious'], instructions=ASK)
    hit = model.classify('hi', ['calm', 'annoyed', 'furious'], instructions=ASK)
    assert hit == SCORE_ANSWER
    assert hit['legend'][2] == 'furious'
    assert hit['probabilities'][1] == 0.95


def test_classify_cache_hit_zeroes_counters():
    model = jev(Provider())
    model.classify('hi', OPTIONS, instructions=ASK)
    model.classify('hi', OPTIONS, instructions=ASK)
    assert model.last_token_count() == (0, 0, 0)


def test_different_state_misses_the_cache():
    provider = Provider()
    model = jev(provider)
    model.classify('hi', OPTIONS, instructions=ASK)
    model.classify('other', OPTIONS, instructions=ASK)
    assert provider.calls == 2


def test_different_options_miss_the_cache():
    provider = Provider()
    model = jev(provider)
    model.classify('hi', OPTIONS, instructions=ASK)
    model.classify('hi', {'sales': 'about buying'}, instructions=ASK)
    assert provider.calls == 2


def test_different_instructions_miss_the_cache():
    provider = Provider()
    model = jev(provider)
    model.classify('hi', OPTIONS, instructions=ASK)
    model.classify('hi', OPTIONS, instructions='Which team owns this?')
    assert provider.calls == 2


def test_different_questions_miss_the_cache():
    provider = Provider()
    model = jev(provider)
    model.classify('hi', questions={'a': {'type': 'noul', 'instructions': 'Urgent?'}})
    model.classify('hi', questions={'a': {'type': 'noul', 'instructions': 'Angry?'}})
    assert provider.calls == 2


def test_classify_does_not_collide_with_prompt_cache():
    """classify and prompt hash argument lists of the same length."""
    provider = Provider()
    model = jev(provider)
    model.classify('hi', OPTIONS, instructions=ASK)
    assert (
        cache.cached_response(model.model.model_name, model.model.model_params, '', 'hi', None, [], False, None) is None
    )


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
