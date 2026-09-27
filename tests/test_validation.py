import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

from justai import Model
from justai.models.basemodel import parse_tool_arguments
from justai.tools.validation import (
    STRUCTURED_FEEDBACK,
    TOOL_FEEDBACK,
    RepairBudget,
    ValidationRetryError,
    format_validation_errors,
)


class Person(BaseModel):
    name: str

    @field_validator('name')
    @classmethod
    def full_name(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('Must contain first and last name')
        return v


class Customer(BaseModel):
    email: int


class Order(BaseModel):
    customer: Customer


class Item(BaseModel):
    quantity: int = Field(gt=0)


class Cart(BaseModel):
    items: list[Item]


def errors_of(validate, raw) -> ValidationError:
    with pytest.raises(ValidationError) as exc:
        validate(raw)
    return exc.value


def test_format_single_error():
    err = errors_of(Person.model_validate, {'name': 'Piet'})
    assert format_validation_errors(err) == '- name: Value error, Must contain first and last name'


def test_format_nested_path():
    err = errors_of(Order.model_validate, {'customer': {'email': 'x'}})
    assert format_validation_errors(err).startswith('- customer.email: ')


def test_format_list_path():
    err = errors_of(Cart.model_validate, {'items': [{'quantity': 1}, {'quantity': 1}, {'quantity': 0}]})
    assert format_validation_errors(err) == '- items.2.quantity: Input should be greater than 0'


def test_format_root_error():
    err = errors_of(TypeAdapter(int).validate_python, 'abc')
    assert format_validation_errors(err).startswith('- (root): ')


def test_format_omits_input():
    err = errors_of(Person.model_validate, {'name': 'SecretSingleName'})
    out = format_validation_errors(err)
    assert 'SecretSingleName' not in out
    assert 'http' not in out


def test_format_caps_at_20():
    err = errors_of(TypeAdapter(list[int]).validate_python, ['x'] * 25)
    lines = format_validation_errors(err).splitlines()
    assert len(lines) == 21
    assert lines[-1] == '- ... and 5 more'


def test_check_valid_returns_value():
    budget = RepairBudget(Person.model_validate, 2, 'Person')
    value, feedback = budget.check({'name': 'Jan Jansen'})
    assert value == Person(name='Jan Jansen')
    assert feedback is None


def test_check_invalid_returns_feedback():
    budget = RepairBudget(Cart.model_validate, 2, 'Cart')
    value, feedback = budget.check({'items': [{'quantity': 0}, {'quantity': 'x'}]})
    assert value is None
    assert feedback.startswith('The previous response failed validation.')
    assert '- items.0.quantity: ' in feedback
    assert '- items.1.quantity: ' in feedback
    assert feedback.endswith('Do not explain the correction.')


def test_check_tool_template_uses_label():
    budget = RepairBudget(Person.model_validate, 1, 'create_invoice', TOOL_FEEDBACK)
    _, feedback = budget.check({'name': 'Piet'})
    assert feedback.startswith('Tool call `create_invoice` could not be executed')
    assert feedback.endswith('Correct the tool arguments and call the tool again.')


def test_check_exhausted_raises():
    budget = RepairBudget(Person.model_validate, 2, 'Person')
    assert budget.check({'name': 'a'})[1]
    assert budget.check({'name': 'b'})[1]
    with pytest.raises(ValidationRetryError) as exc:
        budget.check({'name': 'c'})
    assert exc.value.attempts == 3
    assert isinstance(exc.value.last_error, ValidationError)
    assert exc.value.__cause__ is exc.value.last_error
    assert exc.value.raw == {'name': 'c'}


def test_check_zero_retries_raises_at_once():
    budget = RepairBudget(Person.model_validate, 0, 'Person')
    with pytest.raises(ValidationRetryError) as exc:
        budget.check({'name': 'a'})
    assert exc.value.attempts == 1


def test_check_other_exception_passes_through():
    def validate(raw):
        raise KeyError('boom')

    budget = RepairBudget(validate, 0, 'x')
    with pytest.raises(KeyError):
        budget.check({})
    assert budget.failures == 0


def test_check_resets_counter_after_valid():
    budget = RepairBudget(Person.model_validate, 1, 'Person')
    assert budget.check({'name': 'a'})[1]
    assert budget.check({'name': 'Jan Jansen'})[1] is None
    assert budget.failures == 0
    assert budget.check({'name': 'b'})[1]


def test_retry_error_is_value_error():
    budget = RepairBudget(Person.model_validate, 0, 'Person')
    with pytest.raises(ValueError):
        budget.check({'name': 'a'})


def test_debug_log_has_no_payload(caplog):
    budget = RepairBudget(Person.model_validate, 2, 'Person')
    with caplog.at_level(logging.DEBUG, logger='justai.tools.validation'):
        budget.check({'name': 'SecretSingleName'})
    assert 'Person validation failed, retry 1/2' in caplog.text
    assert 'SecretSingleName' not in caplog.text


def test_templates_have_placeholders():
    assert '{errors}' in STRUCTURED_FEEDBACK
    assert '{name}' in TOOL_FEEDBACK and '{errors}' in TOOL_FEEDBACK


# Task 04: malformed JSON in tool arguments must not crash a stream.


def test_parse_tool_arguments_valid():
    assert parse_tool_arguments('{"a": 1}') == ({'a': 1}, None)


def test_parse_tool_arguments_empty():
    assert parse_tool_arguments('') == ({}, None)


def test_parse_tool_arguments_none():
    assert parse_tool_arguments(None) == ({}, None)


def test_parse_tool_arguments_malformed():
    assert parse_tool_arguments('{"a": ') == ({}, '{"a": ')


def test_parse_tool_arguments_not_an_object():
    assert parse_tool_arguments('[1]') == ({}, '[1]')


class _Obj:
    """Attribute bag. Not a MagicMock: hasattr() must be able to return False."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _tool_calls(model) -> list:
    async def run():
        return [chunk async for chunk in model.stream([{'role': 'user', 'content': 'hi'}])]

    chunks = asyncio.run(run())
    assert chunks[-1].type == 'done'
    return [tc for c in chunks if c.type == 'tool_calls' for tc in c.tool_calls]


def _assert_malformed(calls: list) -> None:
    assert len(calls) == 1
    assert calls[0].arguments == {}
    assert calls[0].raw_arguments == '{"a": '


def test_anthropic_stream_malformed_tool_arguments():
    m = Model('claude-fable-5', ANTHROPIC_API_KEY='k')
    events = [
        _Obj(type='content_block_start', content_block=_Obj(type='tool_use', id='t1', name='f')),
        _Obj(type='content_block_delta', delta=_Obj(type='input_json_delta', partial_json='{"a": ')),
        _Obj(type='content_block_stop'),
    ]
    m.model.async_client = MagicMock()
    m.model.async_client.messages.create = AsyncMock(return_value=_FakeAsyncStream(events))
    _assert_malformed(_tool_calls(m.model))


def test_openai_responses_stream_malformed_tool_arguments():
    m = Model('gpt-5.6-terra', OPENAI_API_KEY='k')
    events = [
        _Obj(
            type='response.output_item.added', output_index=0, item=_Obj(type='function_call', call_id='c1', name='f')
        ),
        _Obj(type='response.function_call_arguments.delta', output_index=0, delta='{"a": '),
        _Obj(type='response.function_call_arguments.done', output_index=0),
    ]
    m.model.client = MagicMock()
    m.model.client.responses.create.return_value = events
    _assert_malformed(_tool_calls(m.model))


def test_openai_completions_stream_malformed_tool_arguments():
    m = Model('MiniMax-M3', MINIMAX_API_KEY='k')
    fn = _Obj(name='f', arguments='{"a": ')
    delta = _Obj(content=None, tool_calls=[_Obj(index=0, id='c1', function=fn)])
    chunk = _Obj(choices=[_Obj(delta=delta)], usage=None)
    m.model.client = MagicMock()
    m.model.client.chat.completions.create.return_value = [chunk]
    _assert_malformed(_tool_calls(m.model))
