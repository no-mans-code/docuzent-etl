"""Orchestrates the full pipeline: schema detection across a document
group (issue #1), then codegen+validation (issue #2). Logs every attempt
to a JSONL file when `log_path` is given - this is exactly the data the
model-evaluation issue (#4) needs, so it's built in from the start
rather than bolted on later.

Two modes, and real testing against Manitoba crop reports (see the repo
README) showed why both matter, not just one:

- `run_pipeline` (per-document codegen): generates an *independent*
  script for every document. Useful for seeing how reliably codegen
  succeeds at all, but each success is really "codegen worked once,"
  not evidence the approach generalizes across the group.
- `run_pipeline_shared_script` (generate once, apply to the group): the
  real test the sibling issues actually ask for - does ONE generated
  script, from ONE example document, correctly extract from every other
  document in the group. This is the honest measure of "did we build
  something that generalizes," not per-document luck.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import codegen, schema_detect


def _write_log(log_path: Path | None, log_entry: dict) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def run_pipeline(host: str, model: str, docling_json_paths: list[Path], log_path: Path | None = None) -> dict:
    """Per-document codegen - see module docstring for why this is a
    different (weaker) signal than `run_pipeline_shared_script`."""
    t0 = time.time()
    schema_result = schema_detect.detect_schema(host, model, docling_json_paths)

    per_document = []
    schema_dict = schema_result.schema.as_dict() if schema_result.succeeded else None

    if schema_result.succeeded:
        for path in docling_json_paths:
            codegen_result = codegen.generate_and_validate(host, model, schema_result.schema, path)
            per_document.append(
                {
                    "document": str(path),
                    "codegen_succeeded": codegen_result.succeeded,
                    "codegen_attempts": len(codegen_result.attempts),
                    "rows_extracted": len(codegen_result.rows) if codegen_result.rows else 0,
                    "rows": codegen_result.rows,
                    "last_error": codegen_result.attempts[-1].error if codegen_result.attempts else None,
                }
            )

    result = {
        "mode": "per_document_codegen",
        "model": model,
        "elapsed_s": round(time.time() - t0, 2),
        "schema_attempts": len(schema_result.attempts),
        "schema_succeeded": schema_result.succeeded,
        "schema": schema_dict,
        "per_document": per_document,
    }

    _write_log(
        log_path,
        {**result, "schema_attempt_detail": [{"attempt": a.attempt_number, "error": a.error, "raw_response": a.raw_response[:2000]} for a in schema_result.attempts]},
    )
    return result


def run_pipeline_shared_script(host: str, model: str, docling_json_paths: list[Path], log_path: Path | None = None) -> dict:
    """The real generalization test: detect a schema, generate+validate
    ONE script against the first document, then run that *exact same*
    script (no regeneration, no per-document LLM calls) against every
    other document in the group. Reports, per document, whether the
    shared script found real rows - this is what "does it generalize"
    actually means, not "did codegen succeed independently N times."
    """
    t0 = time.time()
    schema_result = schema_detect.detect_schema(host, model, docling_json_paths)

    if not schema_result.succeeded:
        result = {"mode": "shared_script", "model": model, "elapsed_s": round(time.time() - t0, 2), "schema_succeeded": False, "schema": None, "per_document": []}
        _write_log(log_path, result)
        return result

    schema = schema_result.schema
    training_doc, *rest_docs = docling_json_paths
    codegen_result = codegen.generate_and_validate(host, model, schema, training_doc)

    per_document = [
        {
            "document": str(training_doc),
            "role": "training (script generated against this document)",
            "codegen_succeeded": codegen_result.succeeded,
            "codegen_attempts": len(codegen_result.attempts),
            "rows_extracted": len(codegen_result.rows) if codegen_result.rows else 0,
            "rows": codegen_result.rows,
        }
    ]

    if codegen_result.succeeded:
        for path in rest_docs:
            stdout, stderr, exec_error = codegen._run_generated_script(codegen_result.code, path)  # noqa: SLF001 - intentional reuse, not a new public API for one internal call
            if exec_error:
                per_document.append({"document": str(path), "role": "generalization test", "codegen_succeeded": False, "rows_extracted": 0, "rows": None, "error": exec_error})
                continue
            parsed, parse_error = codegen.ollama_client.parse_json_response(stdout)
            if parse_error:
                per_document.append({"document": str(path), "role": "generalization test", "codegen_succeeded": False, "rows_extracted": 0, "rows": None, "error": parse_error})
                continue
            rows, validation_error = codegen._validate_rows(parsed, schema)  # noqa: SLF001
            per_document.append(
                {
                    "document": str(path),
                    "role": "generalization test",
                    "codegen_succeeded": rows is not None,
                    "rows_extracted": len(rows) if rows else 0,
                    "rows": rows,
                    "error": validation_error,
                }
            )
    else:
        for path in rest_docs:
            per_document.append({"document": str(path), "role": "generalization test", "codegen_succeeded": False, "rows_extracted": 0, "rows": None, "error": "training document's own codegen never succeeded - shared script was never produced"})

    generalized_count = sum(1 for d in per_document[1:] if d["rows_extracted"] > 0)
    result = {
        "mode": "shared_script",
        "model": model,
        "elapsed_s": round(time.time() - t0, 2),
        "schema_succeeded": True,
        "schema": schema.as_dict(),
        "training_codegen_succeeded": codegen_result.succeeded,
        "generalized_to_n_of_m_other_documents": f"{generalized_count}/{len(rest_docs)}",
        "per_document": per_document,
    }
    _write_log(log_path, result)
    return result
