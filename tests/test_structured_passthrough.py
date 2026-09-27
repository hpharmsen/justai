"""Tests that providers return the raw JSON text for a Pydantic response_format.

Validation against the Pydantic class lives in `Model` (`_to_pydantic`), not in the
provider. The schema that goes over the wire must stay what the SDK's parse() sent.

Usage:
    uv run pytest tests/test_structured_passthrough.py
"""

from __future__ import annotations

import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Validation-retry repair loop through a real provider's history bookkeeping
#
# tests/test_validation_retries.py drives the same loop against a FakeProvider
# stub. These characterize it against the real Anthropic/OpenAI code paths, so
# a change to their history/cache/previous_response_id bookkeeping that breaks
# the repair cycle shows up here, not only in production.
# ---------------------------------------------------------------------------


def test_anthropic_validation_repair_uses_provider_history_and_cache_breakpoint():
    m = Model('claude-sonnet-4-6', ANTHROPIC_API_KEY='k')
    client = MagicMock()
    invalid_response = _anthropic_response(MINOR)
    invalid_response.usage = MagicMock(
        input_tokens=11, output_tokens=22, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    valid_response = _anthropic_response(VALID)
    valid_response.usage = MagicMock(
        input_tokens=13, output_tokens=27, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    client.messages.create.side_effect = [invalid_response, valid_response]
    m.model.client = client

    result = m.chat('Who?', response_format=Person, validation_retries=1)

    assert result == Person(name='Ada', age=36)
    assert client.messages.create.call_count == 2

    # Second call's messages end with, in order: the original question, the
    # invalid answer (appended by chat() itself), and the repair feedback.
    second_messages = client.messages.create.call_args_list[1].kwargs['messages']
    assert len(second_messages) == 3
    original_user, assistant_invalid, feedback_user = second_messages
    assert original_user['role'] == 'user'
    assert original_user['content'][-1]['text'] == 'Who?'
    assert assistant_invalid == {'role': 'assistant', 'content': [{'type': 'text', 'text': MINOR}]}
    assert feedback_user['role'] == 'user'
    feedback_text = feedback_user['content'][-1]['text']
    assert 'Validation errors:' in feedback_text

    # Cache breakpoint (docs/solutions/2026-09-08-prompt-caching-faalt-stil.md) sits on
    # the last message only, not left behind on the assistant turn in between.
    assert feedback_user['content'][-1]['cache_control'] == {'type': 'ephemeral'}
    assert 'cache_control' not in assistant_invalid['content'][-1]
    assert 'cache_control' not in original_user['content'][-1]

    # self.messages (uncached) holds the full turn history the repair was built from.
    assert len(m.model.messages) == 4
    assert m.model.messages[0] == original_user
    assert m.model.messages[1] == assistant_invalid
    assert m.model.messages[2]['role'] == 'user'
    assert m.model.messages[2]['content'][-1]['text'] == feedback_text
    assert 'cache_control' not in m.model.messages[2]['content'][-1]
    assert m.model.messages[3] == {'role': 'assistant', 'content': [{'type': 'text', 'text': VALID}]}

    assert (m.input_token_count, m.output_token_count) == (11 + 13, 22 + 27)


def test_openai_validation_repair_uses_previous_response_id():
    m = Model('gpt-5.6-luna', OPENAI_API_KEY='k')
    client = MagicMock()

    resp_invalid = MagicMock()
    resp_invalid.output = []
    resp_invalid.output_text = MINOR
    resp_invalid.id = 'resp_invalid'
    resp_invalid.usage = MagicMock(input_tokens=5, output_tokens=6)

    resp_valid = MagicMock()
    resp_valid.output = []
    resp_valid.output_text = VALID
    resp_valid.id = 'resp_valid'
    resp_valid.usage = MagicMock(input_tokens=7, output_tokens=8)

    client.responses.create.side_effect = [resp_invalid, resp_valid]
    m.model.client = client

    result = m.chat('Who?', response_format=Person, validation_retries=1)

    assert result == Person(name='Ada', age=36)
    assert client.responses.create.call_count == 2

    second_kwargs = client.responses.create.call_args_list[1].kwargs
    assert second_kwargs['previous_response_id'] == 'resp_invalid'
    feedback_messages = [msg for msg in second_kwargs['input'] if msg.get('role') == 'user']
    assert len(feedback_messages) == 1
    assert 'Validation errors:' in feedback_messages[0]['content'][-1]['text']

    assert (m.input_token_count, m.output_token_count) == (5 + 7, 6 + 8)


def test_providers_import_without_private_sdk_helpers():
    """An SDK that moves its private parse helpers breaks structured output only, not the provider import."""
    code = (
        'import sys\n'
        'class Block:\n'
        '    def find_spec(self, name, path=None, target=None):\n'
        "        if name in ('anthropic.lib._parse._transform', 'openai.lib._parsing._responses'):\n"
        '            raise ImportError(name)\n'
        'sys.meta_path.insert(0, Block())\n'
        'import justai.models.anthropic_models, justai.models.openai_responses\n'
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
