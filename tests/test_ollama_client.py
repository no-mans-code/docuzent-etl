import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuzent_etl.ollama_client import extract_json_block, parse_json_response  # noqa: E402


def test_extract_json_block_from_clean_json():
    assert extract_json_block('{"a": 1}') == '{"a": 1}'


def test_extract_json_block_strips_a_markdown_fence():
    assert extract_json_block('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_block_finds_json_amid_prose():
    """Real behavior seen from real models: explanatory text before or
    after the JSON object, despite being told to respond with JSON only."""
    text = 'Sure, here is the schema:\n{"a": 1}\nLet me know if you need anything else!'
    assert extract_json_block(text) == '{"a": 1}'


def test_extract_json_block_handles_nested_braces():
    text = '{"a": {"b": 1}, "c": [1, 2, {"d": 3}]}'
    assert extract_json_block(text) == text


def test_parse_json_response_succeeds_on_valid_json():
    parsed, err = parse_json_response('{"a": 1}')
    assert err is None
    assert parsed == {"a": 1}


def test_parse_json_response_never_raises_on_garbage():
    parsed, err = parse_json_response("this is not json at all")
    assert parsed is None
    assert err is not None
