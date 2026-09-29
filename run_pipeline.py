#!/usr/bin/env python3
"""Real, runnable entry point for the schema-detection + codegen pipeline
against a document group already converted to Docling JSON.

Usage:
    python run_pipeline.py --model qwen2.5:3b data/docling_json/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuzent_etl.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", help="Docling JSON files (a document group)")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--log", default="runs/pipeline_log.jsonl")
    args = parser.parse_args()

    paths = [Path(p) for p in args.documents]
    for p in paths:
        if not p.exists():
            print(f"error: {p} does not exist", file=sys.stderr)
            sys.exit(1)

    print(f"Running schema detection + codegen for model={args.model} over {len(paths)} document(s)...")
    result = run_pipeline(args.host, args.model, paths, log_path=Path(args.log))

    print(f"\nSchema detection: {'OK' if result['schema_succeeded'] else 'FAILED'} after {result['schema_attempts']} attempt(s)")
    if result["schema"]:
        print(json.dumps(result["schema"], indent=2))

    print("\nPer-document codegen results:")
    for doc in result["per_document"]:
        status = "OK" if doc["codegen_succeeded"] else "FAILED"
        print(f"  [{status}] {Path(doc['document']).name} - {doc['codegen_attempts']} attempt(s), {doc['rows_extracted']} row(s) extracted")
        if doc["rows"]:
            print(f"    sample row: {json.dumps(doc['rows'][0])}")
        if not doc["codegen_succeeded"]:
            print(f"    last error: {doc['last_error']}")

    print(f"\nFull log appended to {args.log}")


if __name__ == "__main__":
    main()
