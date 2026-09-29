#!/usr/bin/env python3
"""Real, runnable entry point for the schema-detection + codegen pipeline
against a document group already converted to Docling JSON.

Usage:
    python run_pipeline.py --model qwen2.5:3b data/docling_json/*.json
    python run_pipeline.py --shared-script --model qwen2.5-coder:14b data/docling_json/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from docuzent_etl import context_sizing  # noqa: E402
from docuzent_etl.docling_json import load_group_tables  # noqa: E402
from docuzent_etl.pipeline import run_pipeline, run_pipeline_shared_script  # noqa: E402
from docuzent_etl.schema_detect import _build_prompt as _build_schema_detect_prompt  # noqa: E402


def _print_per_document(result: dict, shared: bool) -> None:
    print("\nPer-document results:")
    for doc in result["per_document"]:
        status = "OK" if doc["codegen_succeeded"] else "FAILED"
        role = f" [{doc['role']}]" if shared and "role" in doc else ""
        attempts = f", {doc['codegen_attempts']} attempt(s)" if "codegen_attempts" in doc else ""
        print(f"  [{status}]{role} {Path(doc['document']).name}{attempts} - {doc['rows_extracted']} row(s) extracted")
        if doc.get("rows"):
            print(f"    sample row: {json.dumps(doc['rows'][0])}")
        if not doc["codegen_succeeded"]:
            err = doc.get("last_error") or doc.get("error")
            print(f"    error: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", help="Docling JSON files (a document group)")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--log", default="runs/pipeline_log.jsonl")
    parser.add_argument(
        "--shared-script",
        action="store_true",
        help="Generate ONE script against the first document and apply it, unmodified, to every other document - the real generalization test.",
    )
    parser.add_argument(
        "--auto-model",
        action="store_true",
        help="Measure the real token requirement for this document group first, then pick a local model whose real trained "
        "context fits it (smallest that fits; largest available + a warning if none fit). Overrides --model. See issue #7.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.documents]
    for p in paths:
        if not p.exists():
            print(f"error: {p} does not exist", file=sys.stderr)
            sys.exit(1)

    model = args.model
    if args.auto_model:
        group_tables = load_group_tables(paths)
        prompt = _build_schema_detect_prompt(group_tables)
        required_tokens = context_sizing.estimate_required_tokens(prompt)
        print(f"Estimated real token requirement for this document group's schema-detection prompt: ~{required_tokens:,} tokens")

        selection = context_sizing.pick_model_for_tokens(required_tokens, args.host)
        model = selection.model.name
        fit_note = "fits comfortably" if selection.fits else "does NOT fully fit"
        print(f"Selected model: {model} (real trained context {selection.model.trained_context_length:,} tokens, {fit_note})")
        if selection.warning:
            print(f"WARNING: {selection.warning}")
        if selection.ram_offload_warning:
            print(f"WARNING: {selection.ram_offload_warning}")

    mode = "shared-script (generate once, apply to the group)" if args.shared_script else "per-document codegen"
    print(f"Running [{mode}] for model={model} over {len(paths)} document(s)...")

    if args.shared_script:
        result = run_pipeline_shared_script(args.host, model, paths, log_path=Path(args.log))
    else:
        result = run_pipeline(args.host, model, paths, log_path=Path(args.log))

    print(f"\nSchema detection: {'OK' if result['schema_succeeded'] else 'FAILED'}")
    if result["schema"]:
        print(json.dumps(result["schema"], indent=2))

    _print_per_document(result, shared=args.shared_script)

    if args.shared_script and result.get("schema_succeeded"):
        print(f"\nGeneralized to {result['generalized_to_n_of_m_other_documents']} other document(s) in the group (real rows extracted with the SAME unmodified script).")

    print(f"\nFull log appended to {args.log}")


if __name__ == "__main__":
    main()
