"""LLM-generated Python ETL scripts, executed and validated with retry -
issue #2.

Deliberately minimal sandboxing for now: a subprocess, a working
directory, a timeout - no container/VM isolation, no seccomp, no network
namespace. Real sandboxing is tracked separately and matters much more
once raw (non-Docling) documents are in scope; for Docling-JSON-only
input this is a real, but smaller, risk - not zero, and not silently
treated as zero. See the repo README's sandboxing note.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import docling_json, ollama_client
from .schema_detect import DetectedSchema

MAX_ATTEMPTS = 3
EXECUTION_TIMEOUT_S = 30


@dataclass
class CodegenAttempt:
    attempt_number: int
    prompt: str
    generated_code: str
    stdout: str
    stderr: str
    error: str | None
    extracted_rows: list[dict] | None


@dataclass
class CodegenResult:
    attempts: list[CodegenAttempt]
    code: str | None
    rows: list[dict] | None

    @property
    def succeeded(self) -> bool:
        return self.code is not None


def _build_prompt(schema: DetectedSchema, previous_error: str | None = None) -> str:
    schema_json = ", ".join(f'"{f.name}" ({f.type})' for f in schema.fields)
    parts = [
        "Write a complete, standalone Python 3 script that:",
        "1. Reads a Docling JSON document from the file path given as sys.argv[1].",
        "2. Parses doc['tables'] - each table has data.grid, a 2D array of ROWS, each row a list of cell dicts with a 'text' field.",
        "   grid[i] is one ROW (a list of cells), NOT a single cell - grid[i][j]['text'] is the text of one specific cell.",
        "   Concrete correct example: row = table['data']['grid'][2]  ->  a list of cell dicts (one whole row).",
        "                             cell_text = row[0]['text']       ->  a single cell's text (a string).",
        "   A common WRONG mistake, seen repeatedly in testing, in more than one form - all of these are WRONG:",
        "     grid[0]['text']          <- WRONG: grid[0] is a row (a list), indexing it with a string key fails",
        "     grid[0].get('text')      <- WRONG: same mistake via .get() - a list has no .get() method at all",
        "     table['data']['grid'][0]['text']  <- WRONG: same mistake, just with more of the path spelled out",
        "   grid[0] is ALWAYS a list of cells, never a cell itself, no matter how the access expression is written.",
        "   To search every cell in a table for a text match, iterate: for row in grid: for cell in row: cell['text'] ...",
        f"3. Finds the ONE table whose grid contains a cell whose text contains this exact snippet (case-insensitive, substring match): {schema.table_anchor_text!r}",
        "   That snippet identifies the target table's title/header - it is NOT itself a data value to extract.",
        "4. Real tables like this routinely have MULTIPLE header-like rows before real data starts: a merged title row",
        "   spanning all columns, then a row of numbered column labels, then a row of abbreviated column labels.",
        "   Skip every row that looks like a header/label (repeats the title, or looks like 'N. Some Label' rather",
        "   than an actual value) - real data rows contain concrete values (dates, numbers, codes, names), not labels.",
        "5. For each real data row found, build a dict with exactly these fields, mapped by column position:",
        f"   {schema_json}",
        "   For any field of type 'number': real cell text is routinely formatted, not a bare number - '100%', '$317.23',",
        "   '1,438', or with surrounding whitespace. Strip non-numeric characters (%, $, commas, whitespace) BEFORE calling",
        "   float()/int() - do not let an unstripped '%' or ',' cause a ValueError and silently produce None/null for every",
        "   row (a real bug seen in testing: null in every row is usually this, not genuinely missing data).",
        "   For any field of type 'string': copy the cell's text AS-IS, in full - do NOT extract only a number from inside",
        "   it. E.g. a cell reading 'Winkler (9.1 mm)' for a 'string' field must be stored as the full text 'Winkler (9.1 mm)',",
        "   never truncated down to just 9.1 - that numeric-only extraction is a real bug seen in testing, caused by",
        "   over-applying the 'number' field's stripping rule above to a field the schema explicitly says is 'string'.",
        "6. Prints ONLY the resulting JSON list to stdout via print(json.dumps(rows)) - no other output, no prose.",
        "7. If no matching table or no real data rows are found, print an empty JSON list: []",
        "",
        "The script must not make any network calls, must not read/write any file other than sys.argv[1],",
        "and must run to completion in under 10 seconds. Respond with ONLY the Python code - no markdown fences, no explanation.",
    ]
    if previous_error:
        parts.append(f"\nYour previous attempt failed: {previous_error}\nFix it and respond with only the corrected Python code.")
    return "\n".join(parts)


def _strip_code_fence(text: str) -> str:
    """Real models routinely add prose *after* the closing fence too
    ("This script assumes...") - a naive "is the last line a fence"
    check misses that and leaves trailing prose glued onto the code,
    which is a syntax error at execution time. This finds the first
    fenced block specifically, wherever the fence closes, and ignores
    everything outside it."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    # Drop the opening fence line (may have a language tag, e.g. ```python).
    lines = lines[1:]
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            return "\n".join(lines[:i]).strip()
    # No closing fence found at all - better to return everything after
    # the opener than silently produce an empty script.
    return "\n".join(lines).strip()


def _validate_rows(rows: object, schema: DetectedSchema) -> tuple[list[dict] | None, str | None]:
    if not isinstance(rows, list):
        return None, "output must be a JSON array"
    field_names = {f.name for f in schema.fields}
    for row in rows:
        if not isinstance(row, dict):
            return None, "every row must be a JSON object"
        missing = field_names - set(row.keys())
        if missing:
            return None, f"row is missing required field(s): {sorted(missing)}"
    return rows, None


def _find_anchor_table(docling_json_path: Path, anchor_text: str) -> docling_json.ExtractedTable | None:
    """Ground truth, computed independently of any generated code: does a
    table matching `anchor_text` actually exist in this document?

    This exists to answer a real question raised during testing: a
    generated script can run cleanly and print `[]` - a *structurally*
    valid, empty result - for two very different reasons that look
    identical from the output alone: (a) the script has a real bug (a
    wrong table-lookup condition, a broken header heuristic, wrong
    number parsing - all three seen in real testing), or (b) the
    document genuinely doesn't contain this table, and `[]` is the
    honest, correct answer. Blindly retrying on any 0-row result would
    wrongly pressure the model to fabricate data in case (b). This
    function is the deciding signal: if it finds a real table with real
    data, (a) is what happened; if it finds nothing, (b) is - accept the
    empty result rather than manufacture a reason to retry."""
    tables = docling_json.load_tables(docling_json_path)
    anchor_lower = anchor_text.lower()
    for t in tables:
        if any(anchor_lower in cell.lower() for row in t.rows for cell in row):
            return t
    return None


def _run_generated_script(code: str, docling_json_path: Path) -> tuple[str, str, str | None]:
    """Executes the generated script in its own subprocess against the
    real input file - returns (stdout, stderr, execution_error)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path, str(docling_json_path)],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return proc.stdout, proc.stderr, f"script exited with code {proc.returncode}"
        return proc.stdout, proc.stderr, None
    except subprocess.TimeoutExpired:
        return "", "", f"script did not finish within {EXECUTION_TIMEOUT_S}s"
    finally:
        Path(script_path).unlink(missing_ok=True)


def generate_and_validate(host: str, model: str, schema: DetectedSchema, docling_json_path: Path, max_attempts: int = MAX_ATTEMPTS) -> CodegenResult:
    attempts: list[CodegenAttempt] = []
    previous_error: str | None = None

    for attempt_number in range(1, max_attempts + 1):
        prompt = _build_prompt(schema, previous_error)
        result = ollama_client.generate(host, model, prompt)
        code = _strip_code_fence(result.response)

        stdout, stderr, exec_error = _run_generated_script(code, docling_json_path)
        if exec_error:
            attempts.append(CodegenAttempt(attempt_number, prompt, code, stdout, stderr, exec_error, None))
            previous_error = f"{exec_error}\nstderr: {stderr[:500]}"
            continue

        parsed, parse_error = ollama_client.parse_json_response(stdout)
        if parse_error:
            attempts.append(CodegenAttempt(attempt_number, prompt, code, stdout, stderr, parse_error, None))
            previous_error = f"script ran but its output wasn't valid JSON: {parse_error}\nstdout was: {stdout[:500]}"
            continue

        rows, validation_error = _validate_rows(parsed, schema)

        # A structurally-valid empty result needs one more check before
        # it's accepted as a real success - see `_find_anchor_table`'s
        # doc comment for why "the script printed []" is ambiguous on
        # its own.
        if rows == []:
            ground_truth = _find_anchor_table(docling_json_path, schema.table_anchor_text)
            if ground_truth is not None and len(ground_truth.rows) > 1:
                validation_error = (
                    f"Your script ran successfully but extracted 0 rows. A table matching your anchor text "
                    f"DOES exist in this document with {len(ground_truth.rows)} real rows - your table-matching "
                    f"or row-extraction logic has a bug, this document is not simply missing the data. "
                    f"Here is the real table content to check your logic against:\n{ground_truth.to_markdown()}"
                )
                rows = None  # not accepted - a genuine bug, not a correct empty result

        attempts.append(CodegenAttempt(attempt_number, prompt, code, stdout, stderr, validation_error, rows))
        if rows is not None:
            return CodegenResult(attempts=attempts, code=code, rows=rows)
        previous_error = validation_error

    return CodegenResult(attempts=attempts, code=None, rows=None)
