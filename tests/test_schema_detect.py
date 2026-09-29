import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docuzent_etl.schema_detect import _validate  # noqa: E402

ANCHOR = "Table I - Non-Derivative Securities"


def test_validate_accepts_a_well_formed_schema():
    schema, err = _validate({"table_name": "t", "table_anchor_text": ANCHOR, "fields": [{"name": "a", "type": "string"}]})
    assert err is None
    assert schema.table_name == "t"
    assert schema.fields[0].name == "a"
    assert schema.table_anchor_text == ANCHOR


def test_validate_rejects_a_non_object():
    schema, err = _validate([1, 2, 3])
    assert schema is None
    assert "object" in err


def test_validate_rejects_missing_required_keys():
    schema, err = _validate({"table_name": "t"})
    assert schema is None
    assert "fields" in err


def test_validate_rejects_a_missing_table_anchor_text():
    """Real testing against EDGAR filings showed codegen otherwise has no
    way to reliably pick the right table out of a dozen-plus in a real
    document - this field isn't optional."""
    schema, err = _validate({"table_name": "t", "fields": [{"name": "a", "type": "string"}]})
    assert schema is None
    assert "table_anchor_text" in err


def test_validate_rejects_an_empty_fields_array():
    schema, err = _validate({"table_name": "t", "table_anchor_text": ANCHOR, "fields": []})
    assert schema is None
    assert "non-empty" in err


def test_validate_rejects_an_unsupported_field_type():
    schema, err = _validate({"table_name": "t", "table_anchor_text": ANCHOR, "fields": [{"name": "a", "type": "date"}]})
    assert schema is None
    assert "date" in err


def test_validate_rejects_a_field_missing_its_type():
    schema, err = _validate({"table_name": "t", "table_anchor_text": ANCHOR, "fields": [{"name": "a"}]})
    assert schema is None
    assert "name and type" in err
