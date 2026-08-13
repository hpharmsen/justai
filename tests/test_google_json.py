"""Tests for justai.models.google_models._parse_gemini_json.

Reproduces bug: Gemini returned JSON followed by extra data, causing
json.decoder.JSONDecodeError: Extra data at line 5 column 1.
"""

import pytest

from justai.models.basemodel import GeneralException
from justai.models.google_models import _parse_gemini_json


def test_plain_json_object():
    assert _parse_gemini_json('{"a": 1}') == {'a': 1}


def test_plain_json_array():
    assert _parse_gemini_json('[1, 2, 3]') == [1, 2, 3]


def test_json_with_markdown_fence():
    text = '```json\n{"answer": 42}\n```'
    assert _parse_gemini_json(text) == {'answer': 42}


def test_json_with_bare_fence():
    text = '```\n{"answer": 42}\n```'
    assert _parse_gemini_json(text) == {'answer': 42}


def test_json_with_trailing_text():
    """This is the original crash case: valid JSON followed by extra content."""
    text = '{"answer": 42}\n\nExtra explanation from the model.'
    assert _parse_gemini_json(text) == {'answer': 42}


def test_json_with_leading_text():
    text = 'Here is the answer:\n{"answer": 42}'
    assert _parse_gemini_json(text) == {'answer': 42}


def test_invalid_json_raises_general_exception():
    """When Gemini ignores the JSON contract and returns pure prose, callers get GeneralException."""
    with pytest.raises(GeneralException, match='non-JSON'):
        _parse_gemini_json('no json here at all')
