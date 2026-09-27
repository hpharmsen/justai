"""Tests that providers return the raw JSON text for a Pydantic response_format.

Validation against the Pydantic class lives in `Model` (`_to_pydantic`), not in the
provider. The schema that goes over the wire must stay what the SDK's parse() sent.

Usage:
    uv run pytest tests/test_structured_passthrough.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pydantic
import pytest
from pydantic import BaseModel, field_validator

from justai import BadRequestException, Model
from justai.tools import cache


class Person(BaseModel):
    name: str
    age: int

    @field_validator('age')
    @classmethod
    def adult(cls, v: int) -> int:
        assert v >= 18, 'must be an adult'
        return v


VALID = '{"name": "Ada", "age": 36}'
MINOR = '{"name": "Tim", "age": 7}'


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('CACHE_DIR', str(tmp_path))
    cache.CacheDB._instance = None
    yield
    cache.CacheDB._instance = None


def _anthropic_response(text: str) -> MagicMock:
    block = MagicMock()
    block.type = 'text'
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = 'end_turn'
    resp.usage = MagicMock(input_tokens=11, output_tokens=22, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return resp


def _anthropic(text: str, model_name: str = 'claude-sonnet-4-6') -> tuple[Model, MagicMock]:
    m = Model(model_name, ANTHROPIC_API_KEY='k')
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(text)
    m.model.client = client
    return m, client


def _openai(text: str) -> tuple[Model, MagicMock]:
    m = Model('gpt-5.6-luna', OPENAI_API_KEY='k')
    resp = MagicMock()
    resp.output = []
    resp.output_text = text
    resp.id = 'resp_123'
    resp.usage = MagicMock(input_tokens=5, output_tokens=6)
    client = MagicMock()
    client.responses.create.return_value = resp
    m.model.client = client
    return m, client


def test_private_sdk_helpers_exist():
    """Both helpers are private SDK modules; an upgrade that moves them must fail here."""
    from anthropic.lib._parse._transform import transform_schema
    from openai.lib._parsing._responses import type_to_text_format_param

    assert transform_schema(Person.model_json_schema())['additionalProperties'] is False
    assert type_to_text_format_param(Person)['strict'] is True


# ---------------------------------------------------------------------------
# Anthropic structured path
# ---------------------------------------------------------------------------


def test_anthropic_structured_uses_create_not_parse():
    from anthropic.lib._parse._transform import transform_schema
    from pydantic import TypeAdapter

    m, client = _anthropic(VALID)
    m.model.chat('who?', response_format=Person)

    client.messages.parse.assert_not_called()
    fmt = client.messages.create.call_args.kwargs['output_config']['format']
    assert fmt == {'type': 'json_schema', 'schema': transform_schema(Person.model_json_schema())}
    # Exactly what messages.parse() sent before: it builds the schema from a TypeAdapter.
    assert fmt['schema'] == transform_schema(TypeAdapter(Person).json_schema())


def test_anthropic_structured_returns_raw_text():
    m, _ = _anthropic(MINOR)
    response, _, _ = m.model.chat('who?', response_format=Person)
    assert response == MINOR
    assert m.model.last_usage == (11, 22)


def test_anthropic_structured_appends_assistant():
    m, _ = _anthropic(VALID)
    m.model.chat('who?', response_format=Person)
    assert m.model.messages[-1] == {'role': 'assistant', 'content': [{'type': 'text', 'text': VALID}]}


def test_anthropic_dict_schema_still_returns_dict():
    schema = {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']}
    m, _ = _anthropic('{"a": 1}')
    response, _, _ = m.model.chat('a?', response_format=schema)
    assert response == {'a': 1}


# ---------------------------------------------------------------------------
# Anthropic legacy path (models without structured outputs)
# ---------------------------------------------------------------------------


def test_anthropic_legacy_returns_raw_on_bad_json():
    m, _ = _anthropic('no json here', model_name='claude-fable-5')
    response, _, _ = m.model.chat('who?', response_format=Person)
    assert response == 'no json here'


def test_anthropic_legacy_parses_fenced_json():
    m, _ = _anthropic('```json' + VALID + '```', model_name='claude-fable-5')
    response, _, _ = m.model.chat('who?', response_format=Person)
    assert response == {'name': 'Ada', 'age': 36}


def test_anthropic_legacy_return_json_still_raises():
    m, _ = _anthropic('no json here', model_name='claude-fable-5')
    with pytest.raises(BadRequestException):
        m.model.chat('who?', return_json=True)


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def test_openai_pydantic_uses_create_with_strict_format():
    from openai.lib._parsing._responses import type_to_text_format_param

    m, client = _openai(VALID)
    m.model.prompt('who?', [], None, False, Person)

    client.responses.parse.assert_not_called()
    assert not hasattr(m.model, '_responses_parse')
    assert client.responses.create.call_args.kwargs['text'] == {'format': type_to_text_format_param(Person)}


def test_openai_pydantic_returns_output_text():
    m, _ = _openai(MINOR)
    response, _, _ = m.model.chat('who?', [], None, False, Person)
    assert response == MINOR
    assert m.model.last_response_id == 'resp_123'


# ---------------------------------------------------------------------------
# End to end through Model: validation moved there, the public contract is unchanged
# ---------------------------------------------------------------------------


def test_model_chat_returns_pydantic_instance():
    m, _ = _anthropic(VALID)
    assert m.chat('who?', response_format=Person, cached=False) == Person(name='Ada', age=36)


def test_model_chat_raises_validation_error_on_validator_failure():
    m, _ = _anthropic(MINOR)
    with pytest.raises(pydantic.ValidationError):
        m.chat('who?', response_format=Person, cached=False)
