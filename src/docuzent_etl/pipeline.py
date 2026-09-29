"""Orchestrates the full pipeline: schema detection across a document
group (issue #1), then codegen+validation per document (issue #2). Logs
every attempt to a JSONL file when `log_path` is given - this is exactly
the data the model-evaluation issue (#4) needs, so it's built in from
the start rather than bolted on later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import codegen, schema_detect


def run_pipeline(host: str, model: str, docling_json_paths: list[Path], log_path: Path | None = None) -> dict:
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
        "model": model,
        "elapsed_s": round(time.time() - t0, 2),
        "schema_attempts": len(schema_result.attempts),
        "schema_succeeded": schema_result.succeeded,
        "schema": schema_dict,
        "per_document": per_document,
    }

    if log_path:
        log_entry = {
            **result,
            "schema_attempt_detail": [
                {"attempt": a.attempt_number, "error": a.error, "raw_response": a.raw_response[:2000]} for a in schema_result.attempts
            ],
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    return result
