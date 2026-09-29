import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuzent_etl.codegen import _find_anchor_table, _strip_code_fence, _validate_rows  # noqa: E402
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


def _make_docling_doc_file(tmp_path: Path, tables: list[list[list[str]]]) -> Path:
    doc_tables = []
    for rows in tables:
        grid = [[{"text": cell} for cell in row] for row in rows]
        doc_tables.append({"data": {"grid": grid, "num_rows": len(rows), "num_cols": len(rows[0]) if rows else 0}})
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"tables": doc_tables}), encoding="utf-8")
    return path


def test_find_anchor_table_finds_a_real_matching_table(tmp_path):
    path = _make_docling_doc_file(tmp_path, [[["Region", "Value"], ["Central", "5"], ["Eastern", "3"]]])
    table = _find_anchor_table(path, "Region")
    assert table is not None
    assert len(table.rows) == 3


def test_find_anchor_table_returns_none_when_nothing_matches(tmp_path):
    """The real property this exists for: a document that genuinely
    doesn't contain the target table must not be mistaken for a bug -
    see the module's own comment on why blindly retrying every 0-row
    result would be wrong (it would pressure the model to fabricate
    data for a document that has none)."""
    path = _make_docling_doc_file(tmp_path, [[["Something else entirely", "Value"], ["x", "1"]]])
    table = _find_anchor_table(path, "Region")
    assert table is None


def test_find_anchor_table_is_case_insensitive(tmp_path):
    path = _make_docling_doc_file(tmp_path, [[["REGION", "Value"], ["Central", "5"]]])
    table = _find_anchor_table(path, "region")
    assert table is not None
