import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuzent_etl.codegen import _strip_code_fence, _validate_rows  # noqa: E402
from docuzent_etl.schema_detect import DetectedSchema, SchemaField  # noqa: E402


def test_strip_code_fence_removes_a_clean_fence():
    text = "```python\nprint('hi')\n```"
    assert _strip_code_fence(text) == "print('hi')"


def test_strip_code_fence_handles_trailing_prose_after_the_closing_fence():
    """Real regression, found while testing against real EDGAR filings:
    qwen2.5:3b routinely adds an explanatory paragraph *after* the
    closing fence ("This script assumes..."), which a naive
    "is the last line a fence" check leaves glued onto the code,
    producing a syntax error at execution time."""
    text = "```python\nprint('hi')\n```\n\nThis script assumes the input file exists and is valid JSON."
    assert _strip_code_fence(text) == "print('hi')"


def test_strip_code_fence_leaves_unfenced_text_alone():
    text = "print('hi')"
    assert _strip_code_fence(text) == "print('hi')"


def test_strip_code_fence_handles_a_fence_with_no_closer():
    text = "```python\nprint('hi')"
    assert _strip_code_fence(text) == "print('hi')"


def _schema():
    return DetectedSchema(table_name="t", description="", fields=[SchemaField("a", "string"), SchemaField("b", "number")])


def test_validate_rows_accepts_rows_with_every_required_field():
    rows, err = _validate_rows([{"a": "x", "b": 1}], _schema())
    assert err is None
    assert rows == [{"a": "x", "b": 1}]


def test_validate_rows_rejects_a_non_list():
    rows, err = _validate_rows({"a": "x"}, _schema())
    assert rows is None
    assert "array" in err


def test_validate_rows_rejects_a_row_missing_a_required_field():
    rows, err = _validate_rows([{"a": "x"}], _schema())
    assert rows is None
    assert "b" in err


def test_validate_rows_accepts_an_empty_list():
    """An empty list is *structurally* valid - real testing found this
    matters: it must not be confused with "successfully extracted real
    data" by anything reading this result (see the pipeline's separate
    `rows_extracted` count, not just `codegen_succeeded`)."""
    rows, err = _validate_rows([], _schema())
    assert err is None
    assert rows == []
