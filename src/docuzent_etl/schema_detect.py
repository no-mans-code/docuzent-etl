"""LLM-driven schema detection across a document group - issue #1.

Given a group of Docling-parsed documents, finds the tables that share
a genuinely consistent shape across the whole group and proposes one
JSON schema for them - not per-document ad-hoc shapes, which would make
downstream analysis across the group impossible. Validates the model's
own proposal and retries (feeding the real parse/validation error back)
rather than silently accepting malformed output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import ollama_client
from .docling_json import ExtractedTable, load_group_tables

MAX_ATTEMPTS = 3


@dataclass
class SchemaField:
    name: str
    type: str


@dataclass
class DetectedSchema:
    table_name: str
    description: str
    fields: list[SchemaField]
    # A short, literal snippet of text from the target table's own
    # header/title cell (e.g. "Table I - Non-Derivative Securities...") -
    # added after real testing against EDGAR filings showed codegen
    # otherwise has to blindly guess which of a dozen-plus tables in a
    # real document is the right one, with no anchor to search for. A
    # text anchor generalizes across documents better than a raw table
    # index, since boilerplate tables can shift a target table's index
    # between documents in the same group even when its own title text
    # is identical and stable (it's a fixed SEC form label here).
    table_anchor_text: str = ""

    @staticmethod
    def from_dict(d: dict) -> "DetectedSchema":
        fields = [SchemaField(name=f["name"], type=f["type"]) for f in d["fields"]]
        return DetectedSchema(table_name=d["table_name"], description=d.get("description", ""), fields=fields, table_anchor_text=d.get("table_anchor_text", ""))

    def as_dict(self) -> dict:
        return {
            "table_name": self.table_name,
            "description": self.description,
            "fields": [{"name": f.name, "type": f.type} for f in self.fields],
            "table_anchor_text": self.table_anchor_text,
        }


@dataclass
class SchemaDetectionAttempt:
    attempt_number: int
    prompt: str
    raw_response: str
    error: str | None
    schema: DetectedSchema | None


@dataclass
class SchemaDetectionResult:
    attempts: list[SchemaDetectionAttempt]
    schema: DetectedSchema | None

    @property
    def succeeded(self) -> bool:
        return self.schema is not None


def _build_prompt(group_tables: dict[str, list[ExtractedTable]], previous_error: str | None = None) -> str:
    parts = [
        "You are analyzing a GROUP of related documents to find valuable, structured tabular data.",
        "Below are the real tables extracted from each document (Docling table index in brackets).",
        "Some tables are boilerplate/layout artifacts (headers, signature blocks, cover pages) - ignore those.",
        "Find the ONE kind of table that appears, in a genuinely consistent shape, across the documents,",
        "and propose a single JSON schema for it that would fit every document's version of that table.",
        "",
    ]
    for doc_name, tables in group_tables.items():
        parts.append(f"=== {Path(doc_name).name} ===")
        for t in tables:
            parts.append(f"[table {t.table_index}] ({t.num_rows}x{t.num_cols})")
            parts.append(t.to_markdown())
        parts.append("")

    parts.append(
        "Respond with ONLY a JSON object, no prose, no markdown fences, matching exactly this shape:\n"
        '{"table_name": "...", "description": "...", "table_anchor_text": "...", '
        '"fields": [{"name": "...", "type": "string|number|boolean"}]}\n'
        "table_anchor_text must be a short, EXACT, literal snippet of text copied from the target table's own "
        "title/header cell as shown above (not paraphrased) - something that reliably identifies which table this "
        "is in ANY document in the group, even though its position among all the tables in the document may differ."
    )
    if previous_error:
        parts.append(f"\nYour previous attempt failed: {previous_error}\nFix it and respond with valid JSON only.")
    return "\n".join(parts)


def _validate(parsed: object) -> tuple[DetectedSchema | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "top-level JSON must be an object"
    if "table_name" not in parsed or "fields" not in parsed:
        return None, "missing required keys: table_name, fields"
    anchor = parsed.get("table_anchor_text", "")
    if not isinstance(anchor, str) or not anchor.strip():
        return None, "table_anchor_text is missing or empty - it must be a literal snippet from the target table's own header/title cell"
    if not isinstance(parsed["fields"], list) or not parsed["fields"]:
        return None, "fields must be a non-empty array"
    for f in parsed["fields"]:
        if not isinstance(f, dict) or "name" not in f or "type" not in f:
            return None, "each field must be an object with name and type"
        if f["type"] not in ("string", "number", "boolean"):
            return None, f"unsupported field type: {f['type']!r} (expected string/number/boolean)"
    try:
        return DetectedSchema.from_dict(parsed), None
    except Exception as e:  # defensive - the checks above should already catch real issues
        return None, f"schema did not match the expected shape: {e}"


def detect_schema(host: str, model: str, docling_json_paths: list[Path], max_attempts: int = MAX_ATTEMPTS) -> SchemaDetectionResult:
    """Runs the detect -> validate -> retry loop for real - every attempt
    is recorded (the model-evaluation issue needs this log), not just
    the final result."""
    group_tables = load_group_tables(docling_json_paths)
    attempts: list[SchemaDetectionAttempt] = []
    previous_error: str | None = None

    for attempt_number in range(1, max_attempts + 1):
        prompt = _build_prompt(group_tables, previous_error)
        result = ollama_client.generate(host, model, prompt)
        parsed, parse_error = ollama_client.parse_json_response(result.response)
        if parse_error:
            attempts.append(SchemaDetectionAttempt(attempt_number, prompt, result.response, parse_error, None))
            previous_error = parse_error
            continue
        schema, validation_error = _validate(parsed)
        attempts.append(SchemaDetectionAttempt(attempt_number, prompt, result.response, validation_error, schema))
        if schema is not None:
            return SchemaDetectionResult(attempts=attempts, schema=schema)
        previous_error = validation_error

    return SchemaDetectionResult(attempts=attempts, schema=None)
