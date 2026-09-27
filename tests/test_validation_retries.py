"""Tests for validation_retries on Model.chat() and Model.prompt(). No network, no API keys.

Usage:
    uv run pytest tests/test_validation_retries.py
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError, field_validator

import justai.model.model as model_mod
from justai import Model
from justai.tools import cache
from justai.tools.validation import ValidationRetryError


class Person(BaseModel):
    name: str
    age: int

    @field_validator('name')
    @classmethod
    def full_name(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('Must contain first and last name')
        return v


class Customer(BaseModel):
    email: str


class Line(BaseModel):
    quantity: int


class Order(BaseModel):
    customer: Customer
    items: list[Line]


VALID = '{"name": "Ada Lovelace", "age": 36}'
INVALID = '{"name": "Ada", "age": 36}'


class FakeProvider:
    """Walks a list of answers and records every call's arguments."""

    def __init__(self, answers: list, in_tokens: int = 10, out_tokens: int = 5):
        self.answers, self.in_tokens, self.out_tokens = list(answers), in_tokens, out_tokens
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({'prompt': prompt, **kwargs})
        return self.answers.pop(0), self.in_tokens, self.out_tokens


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


def _model(method: str, answers: list, **kwargs) -> tuple[Model, FakeProvider]:
    model = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    fake = FakeProvider(answers, **kwargs)
    setattr(model.model, method, fake)
    return model, fake


def test_valid_first_time_one_call():
    model, fake = _model('chat', [VALID])
    result = model.chat('who?', response_format=Person, validation_retries=2)
    assert result == Person(name='Ada Lovelace', age=36)
    assert len(fake.calls) == 1


def test_invalid_then_valid():
    model, fake = _model('chat', [INVALID, VALID])
    result = model.chat('who?', response_format=Person, validation_retries=2)
    assert isinstance(result, Person) and result.name == 'Ada Lovelace'
    assert len(fake.calls) == 2


def test_repair_prompt_carries_feedback():
    model, fake = _model('chat', [INVALID, VALID])
    model.chat('who?', response_format=Person, validation_retries=1)
    repair = fake.calls[1]['prompt']
    assert repair.startswith('The previous response failed validation.')
    assert '- name: Value error, Must contain first and last name' in repair
    assert 'who?' not in repair
    assert fake.calls[1]['response_format'] is Person


def test_multiple_errors_all_in_feedback():
    model, fake = _model('chat', ['{"name": "Ada", "age": "old"}', VALID])
    model.chat('who?', response_format=Person, validation_retries=1)
    repair = fake.calls[1]['prompt']
    assert '- name: Value error, Must contain first and last name' in repair
    assert '- age: Input should be a valid integer' in repair


def test_exhausted_raises_retry_error():
    model, fake = _model('chat', [INVALID, INVALID, INVALID, VALID])
    with pytest.raises(ValidationRetryError) as exc:
        model.chat('who?', response_format=Person, validation_retries=2)
    assert len(fake.calls) == 3
    assert exc.value.attempts == 3
    assert exc.value.raw == INVALID
    assert isinstance(exc.value.last_error, ValidationError)


def test_zero_retries_raises_validation_error():
    model, fake = _model('chat', [INVALID, VALID])
    with pytest.raises(ValidationError) as exc:
        model.chat('who?', response_format=Person, validation_retries=0)
    assert not isinstance(exc.value, ValidationRetryError)
    assert len(fake.calls) == 1


def test_default_is_zero():
    model, fake = _model('chat', [INVALID, VALID])
    with pytest.raises(ValidationError):
        model.chat('who?', response_format=Person)
    assert len(fake.calls) == 1


def test_nested_model_path():
    model, fake = _model('chat', ['{"customer": {}, "items": []}', '{"customer": {"email": "a@b.c"}, "items": []}'])
    model.chat('order?', response_format=Order, validation_retries=1)
    assert '- customer.email: Field required' in fake.calls[1]['prompt']


def test_list_field_path():
    bad = {'customer': {'email': 'a@b.c'}, 'items': [{'quantity': 1}, {'quantity': 2}, {'quantity': 'x'}]}
    good = {'customer': {'email': 'a@b.c'}, 'items': [{'quantity': 1}]}
    model, fake = _model('chat', [bad, good])
    result = model.chat('order?', response_format=Order, validation_retries=1)
    assert '- items.2.quantity:' in fake.calls[1]['prompt']
    assert result.items[0].quantity == 1


def test_malformed_json_is_repairable():
    model, fake = _model('chat', ['{"name": ', VALID])
    result = model.chat('who?', response_format=Person, validation_retries=1)
    assert len(fake.calls) == 2
    assert 'Invalid JSON' in fake.calls[1]['prompt']
    assert result.name == 'Ada Lovelace'


def test_prompt_repair_includes_original_and_answer():
    model, fake = _model('prompt', [INVALID, VALID])
    result = model.prompt('who is it?', response_format=Person, validation_retries=1, cached=False)
    repair = fake.calls[1]['prompt']
    assert 'who is it?' in repair
    assert INVALID in repair
    assert 'Must contain first and last name' in repair
    assert result.name == 'Ada Lovelace'


def test_prompt_repair_dumps_dict_answer():
    bad = {'name': 'Ada', 'age': 36}
    model, fake = _model('prompt', [bad, VALID])
    model.prompt('who is it?', response_format=Person, validation_retries=1, cached=False)
    assert json.dumps(bad) in fake.calls[1]['prompt']


def test_images_only_on_first_call():
    model, fake = _model('chat', [INVALID, VALID])
    model.chat('who?', images='https://x/y.png', response_format=Person, validation_retries=1)
    assert fake.calls[0]['images'] == ['https://x/y.png']
    assert fake.calls[1]['images'] is None

    model, fake = _model('prompt', [INVALID, VALID])
    model.prompt('who?', images='https://x/y.png', response_format=Person, validation_retries=1, cached=False)
    assert fake.calls[0]['images'] == ['https://x/y.png']
    assert fake.calls[1]['images'] is None


def test_prompt_cache_stores_only_final(monkeypatch):
    saved = []
    monkeypatch.setattr(model_mod, 'cached_response', lambda *a: None)
    monkeypatch.setattr(model_mod, 'cache_save', lambda response, *key: saved.append((response, key)))
    model, _ = _model('prompt', [INVALID, VALID])
    model.prompt('who is it?', response_format=Person, validation_retries=1)
    assert len(saved) == 1
    response, key = saved[0]
    assert Person.model_validate_json(response[0]) == Person(name='Ada Lovelace', age=36)
    assert 'who is it?' in key  # stored under the original prompt, not the repair prompt
    assert not any('previous response' in str(k) for k in key)


def test_prompt_cache_never_stores_invalid(monkeypatch):
    saved = []
    monkeypatch.setattr(model_mod, 'cached_response', lambda *a: None)
    monkeypatch.setattr(model_mod, 'cache_save', lambda response, *key: saved.append(response))
    model, _ = _model('prompt', [INVALID])
    with pytest.raises(ValidationError):
        model.prompt('who is it?', response_format=Person)
    assert saved == []


def test_valid_cache_hit_makes_no_call(monkeypatch):
    monkeypatch.setattr(model_mod, 'cached_response', lambda *a: (VALID, 3, 4))
    model, fake = _model('prompt', [])
    result = model.prompt('who is it?', response_format=Person, validation_retries=2)
    assert result.name == 'Ada Lovelace'
    assert fake.calls == []
    assert model.last_token_count() == (0, 0, 0)


def test_token_counts_summed():
    model, _ = _model('chat', [INVALID, VALID], in_tokens=10, out_tokens=5)
    model.chat('who?', response_format=Person, validation_retries=2)
    assert (model.input_token_count, model.output_token_count) == (20, 10)

    model, _ = _model('chat', [INVALID, INVALID, INVALID], in_tokens=10, out_tokens=5)
    with pytest.raises(ValidationRetryError):
        model.chat('who?', response_format=Person, validation_retries=2)
    assert (model.input_token_count, model.output_token_count) == (30, 15)


def test_token_counts_kept_when_provider_fails_mid_loop():
    model, fake = _model('chat', [INVALID], in_tokens=10, out_tokens=5)
    with pytest.raises(IndexError):  # the fake runs out of answers on the repair call
        model.chat('who?', response_format=Person, validation_retries=2)
    assert len(fake.calls) == 2
    assert (model.input_token_count, model.output_token_count) == (10, 5)


@pytest.mark.parametrize('fmt', [{'type': 'object', 'properties': {}}, list[Person], None])
def test_retries_require_pydantic_format(fmt):
    model, fake = _model('chat', [VALID])
    with pytest.raises(ValueError):
        model.chat('who?', response_format=fmt, validation_retries=1)
    with pytest.raises(ValueError):
        model.prompt('who?', response_format=fmt, validation_retries=1, cached=False)
    assert fake.calls == []


def test_exported_from_package():
    from justai import ValidationRetryError as exported

    assert exported is ValidationRetryError
