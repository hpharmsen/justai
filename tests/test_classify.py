"""Wire-format tests for System One classification. No network, no API keys.

Usage:
    venv/bin/pytest tests/test_classify.py
"""

import json

import httpx
import pytest

from justai import (
    AuthorizationException,
    BadRequestException,
    ConnectionException,
    GeneralException,
    ModelOverloadException,
    RatelimitException,
    TimeoutException,
)
from justai.models.systemone import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    SINGLE,
    build_payload,
    build_question,
    map_http_error,
    unpack,
    usage_of,
)


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request('POST', 'https://example.test/v1/systemone')
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f'{code}', request=request, response=response)


# ---------------------------------------------------------------------------
# build_question
# ---------------------------------------------------------------------------


ASK = 'What is this about?'


def test_dict_options_give_choice():
    question = build_question({'billing': 'about money', 'technical': 'about bugs'}, ASK)
    assert question['type'] == 'choice'
    assert question['criteria'] == {'billing': 'about money', 'technical': 'about bugs'}


def test_list_options_give_score():
    question = build_question(['calm', 'annoyed', 'furious'], ASK)
    assert question['type'] == 'score'
    assert question['criteria'] == ['calm', 'annoyed', 'furious']


def test_no_options_give_noul():
    question = build_question(None, ASK)
    assert question['type'] == 'noul'
    assert 'criteria' not in question


def test_instructions_land_in_question():
    assert build_question(None, 'Is this urgent?')['instructions'] == 'Is this urgent?'


def test_instructions_are_required():
    """The API answers 400 on a question without instructions, for all three types."""
    for options in ({'a': 'b'}, ['low', 'high'], None):
        with pytest.raises(AssertionError):
            build_question(options, '')


def test_choice_at_255_passes():
    options = {f'o{i}': f'option {i}' for i in range(MAX_CHOICE_OPTIONS)}
    assert len(build_question(options, ASK)['criteria']) == 255


def test_choice_over_255_raises():
    options = {f'o{i}': f'option {i}' for i in range(MAX_CHOICE_OPTIONS + 1)}
    with pytest.raises(AssertionError):
        build_question(options, ASK)


def test_score_at_2_and_10_pass():
    assert len(build_question(['a'] * MIN_SCORE_LEVELS, ASK)['criteria']) == 2
    assert len(build_question(['a'] * MAX_SCORE_LEVELS, ASK)['criteria']) == 10


def test_score_at_1_and_11_raise():
    with pytest.raises(AssertionError):
        build_question(['a'], ASK)
    with pytest.raises(AssertionError):
        build_question(['a'] * 11, ASK)


def test_options_rejects_other_types():
    with pytest.raises(TypeError) as excinfo:
        build_question('billing', ASK)
    assert 'str' in str(excinfo.value)


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_state_accepts_str_dict_list():
    for state in ('plain text', {'subject': 'refund'}, ['line one', 'line two']):
        assert build_payload('jev-latest', state, None, instructions=ASK)['state'] == state


def test_state_rejects_other_types():
    with pytest.raises(AssertionError):
        build_payload('jev-latest', 42, None, instructions=ASK)


def test_options_and_questions_are_exclusive():
    with pytest.raises(AssertionError):
        build_payload('jev-latest', 'x', {'a': 'b'}, questions={'q': {'type': 'noul'}})


def test_instructions_and_questions_are_exclusive():
    with pytest.raises(AssertionError):
        build_payload('jev-latest', 'x', instructions=ASK, questions={'q': {'type': 'noul'}})


def test_questions_pass_through_unchanged():
    questions = {'urgency': {'type': 'noul', 'instructions': 'Urgent?'}}
    assert build_payload('jev-latest', 'x', questions=questions)['questions'] == questions


def test_single_question_lands_under_single_key():
    payload = build_payload('jev-latest', 'x', None, instructions='Urgent?')
    assert list(payload['questions']) == [SINGLE]
    assert payload['questions'][SINGLE]['instructions'] == 'Urgent?'


def test_payload_carries_wire_model_name():
    assert build_payload('kev-3b', 'x', None, instructions=ASK)['model'] == 'kev-3b'


# ---------------------------------------------------------------------------
# unpack
# ---------------------------------------------------------------------------


def test_unpack_single_flattens():
    body = {'answers': {SINGLE: {'type': 'noul', 'noul': 0.95}}}
    assert unpack(body, single=True) == {'type': 'noul', 'noul': 0.95}


def test_unpack_batch_keeps_map():
    body = {'answers': {'urgency': {'type': 'noul', 'noul': 0.1}, 'topic': {'type': 'noul', 'noul': 0.2}}}
    assert unpack(body, single=False) == body['answers']


def test_unpack_keeps_unknown_fields():
    body = {'answers': {SINGLE: {'type': 'noul', 'noul': 0.5, 'rationale_id': 'abc'}}}
    assert unpack(body, single=True)['rationale_id'] == 'abc'


def test_unpack_score_keys_are_ints():
    """Score legend and probabilities key by level, like the vendor SDKs do."""
    body = {
        'answers': {
            SINGLE: {
                'type': 'score',
                'score': 2,
                'legend': {'0': 'Calm', '1': 'Frustrated', '2': 'Very angry'},
                'probabilities': {'0': 0.1, '1': 0.2, '2': 0.7},
                'confidence': 0.7,
            }
        }
    }
    answer = unpack(body, single=True)
    assert answer['legend'][2] == 'Very angry'
    assert answer['probabilities'][2] == 0.7
    assert set(answer['legend']) == {0, 1, 2}


def test_unpack_choice_keys_stay_strings():
    """Choice options are named, not numbered, so their keys must not be touched."""
    body = {
        'answers': {
            SINGLE: {
                'type': 'choice',
                'choice': 'billing',
                'probabilities': {'billing': 0.8, 'technical': 0.2},
                'confidence': 0.8,
            }
        }
    }
    assert unpack(body, single=True)['probabilities'] == {'billing': 0.8, 'technical': 0.2}


def test_unpack_score_ints_survive_a_json_round_trip():
    """The cache stores the dict as JSON, which restringifies int keys."""
    body = {
        'answers': {
            SINGLE: {'type': 'score', 'score': 1, 'legend': {'0': 'low', '1': 'high'}, 'probabilities': {'0': 0.3, '1': 0.7}}
        }
    }
    answer = unpack(body, single=True)
    assert unpack({'answers': {SINGLE: json.loads(json.dumps(answer))}}, single=True) == answer


# ---------------------------------------------------------------------------
# usage_of
# ---------------------------------------------------------------------------


def test_usage_of_reads_tokens():
    assert usage_of({'usage': {'input_tokens': 476, 'output_tokens': 70}}) == (476, 70)


def test_usage_of_ignores_cost():
    assert usage_of({'usage': {'input_tokens': 476, 'output_tokens': 70, 'cost': 0.000019992}}) == (476, 70)


def test_usage_of_survives_missing_usage():
    assert usage_of({}) == (0, 0)


# ---------------------------------------------------------------------------
# map_http_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'code, expected',
    [
        (401, AuthorizationException),
        (403, AuthorizationException),
        (400, BadRequestException),
        (422, BadRequestException),
        (429, RatelimitException),
        (529, ModelOverloadException),
        (500, ConnectionException),
        (502, ConnectionException),
        (418, GeneralException),
    ],
)
def test_map_http_error_per_status(code, expected):
    assert isinstance(map_http_error(_status_error(code)), expected)


def test_map_http_error_529_is_not_connection():
    assert not isinstance(map_http_error(_status_error(529)), ConnectionException)


def test_map_http_error_timeout_before_transport():
    mapped = map_http_error(httpx.ReadTimeout('too slow'))
    assert isinstance(mapped, TimeoutException)
    assert not isinstance(mapped, ConnectionException)


def test_map_http_error_transport_is_connection():
    assert isinstance(map_http_error(httpx.ConnectError('no route')), ConnectionException)


def test_map_http_error_json_decode():
    error = json.JSONDecodeError('Expecting value', '<not json>', 0)
    assert isinstance(map_http_error(error), GeneralException)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
