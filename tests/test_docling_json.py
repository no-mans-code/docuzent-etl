import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuzent_etl.docling_json import ExtractedTable, load_tables  # noqa: E402


def test_is_trivial_for_a_1x1_table():
    t = ExtractedTable(table_index=0, num_rows=1, num_cols=1, rows=[["x"]])
    assert t.is_trivial()


def test_is_trivial_for_an_all_empty_table():
    t = ExtractedTable(table_index=0, num_rows=2, num_cols=2, rows=[["", ""], ["", ""]])
    assert t.is_trivial()


def test_is_not_trivial_for_a_real_table():
    t = ExtractedTable(table_index=0, num_rows=2, num_cols=2, rows=[["Name", "Amount"], ["AAPL", "100"]])
    assert not t.is_trivial()


def test_to_markdown_caps_long_tables():
    rows = [[f"row{i}"] for i in range(50)]
    t = ExtractedTable(table_index=0, num_rows=50, num_cols=1, rows=rows)
    md = t.to_markdown(max_rows=5)
    assert "row0" in md
    assert "row4" in md
    assert "row5" not in md
    assert "45 more rows" in md


def _make_docling_doc(tables: list[list[list[str]]]) -> dict:
    """A minimal Docling-JSON-shaped fixture - just the `tables` field
    with the real `data.grid` shape this module actually reads, not the
    full real document schema (layout/provenance metadata this module
    never looks at)."""
    doc_tables = []
    for rows in tables:
        grid = [[{"text": cell} for cell in row] for row in rows]
        doc_tables.append({"data": {"grid": grid, "num_rows": len(rows), "num_cols": len(rows[0]) if rows else 0}})
    return {"tables": doc_tables}


def test_load_tables_filters_out_trivial_tables_but_keeps_real_ones(tmp_path):
    doc = _make_docling_doc(
        [
            [[""]],  # trivial - a layout artifact
            [["Name", "Amount"], ["AAPL", "100"]],  # real data
        ]
    )
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    tables = load_tables(path)
    assert len(tables) == 1
    assert tables[0].rows == [["Name", "Amount"], ["AAPL", "100"]]
    # The real table's Docling index (1, not 0) is preserved, so a later
    # step can point back at exactly which table in the source document
    # this came from.
    assert tables[0].table_index == 1
