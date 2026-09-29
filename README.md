# docuzent-etl (private prototype)

An LLM-driven pipeline: given a **group** of related documents (already parsed by Docling into JSON), detect the valuable structured tables that appear consistently across the group, propose one JSON schema that fits all of them, then generate, execute, and validate Python extraction code against that schema - with a real retry loop when the model's own output is malformed.

This is a **prototype for a working ETL system**, kept private. It is not part of the public [`docuzent`](https://github.com/no-mans-code/docuzent) repo - it reuses the idea of shelling out to Docling but has its own codebase, since the actual work here (LLM-driven schema inference, generated-code execution, financial-data extraction) is a different problem from that repo's document Q&A tool.

## Status: real, working POC - not production-hardened

Every claim below is backed by a real run against real data, not a design intention. See "Known limitations" for what's honestly still broken or missing.

## Pipeline

1. **Ingest** (`docuzent_etl.docling_json`): reads Docling's own JSON output and extracts a compact `{rows: [[cell_text, ...]]}` representation per table, filtering out 1x1/empty layout artifacts before anything is shown to an LLM.
2. **Schema detection** (`docuzent_etl.schema_detect`): given a document group's tables, asks an LLM to find the one kind of table that's genuinely consistent across the group and propose a JSON schema for it - `{table_name, description, table_anchor_text, fields: [{name, type}]}`. Validated and retried (bounded attempts) on malformed output.
3. **Codegen** (`docuzent_etl.codegen`): given the schema, asks an LLM to write a standalone Python script that extracts matching rows from one document's Docling JSON. Executes it (subprocess, timeout, no network) and validates the output is a JSON array of objects with every required field. Retries with the real error (parse failure, exception, schema mismatch) fed back to the model.
4. **Orchestration + logging** (`docuzent_etl.pipeline`): runs 2-3 across a whole document group, logging every attempt (prompt, response, pass/fail) to a JSONL file - this is the data the model-evaluation work needs, built in from the start rather than added later.

## Real test: SEC EDGAR Form 4 filings (Apple, 5 real filings)

Fetched for real via EDGAR's own Fair Access-compliant API (`data.sec.gov`, identifying User-Agent, no scraping - EDGAR explicitly permits this) - see `data/samples/edgar_form4/`. Converted to real Docling JSON via a Docker image with Docling actually installed (`data/docling_json/`).

**Real findings from running this, not assumed:**

- **Schema detection succeeded on the first attempt** with both models tried (`qwen2.5:3b`, `qwen2.5-coder:14b`), correctly identifying Form 4's "Table I/II - Securities..." structure as the valuable table amid ~14-17 total tables per document (most of the rest are layout artifacts: address blocks, checkboxes, signature blocks).
- **Model choice materially changes schema quality**: `qwen2.5:3b` initially proposed field names that were raw, unedited table header text (e.g. `"1. Title of Security (Instr. 3)"`) - technically valid JSON, but useless as real field names. `qwen2.5-coder:14b` produced clean, human-usable names (`"Security Type"`, `"Transaction Date"`). This is exactly the comparison issue #4 (model evaluation) exists to make systematic, not anecdotal.
- **A real bug found and fixed**: the first version of `_strip_code_fence` didn't handle a model adding prose *after* the closing markdown fence (`"...\`\`\`\n\nThis script assumes..."`) - left the fence and trailing prose glued onto the executable code, causing real syntax errors. Fixed; regression test added (`tests/test_codegen.py`).
- **A real design gap found and fixed**: the schema alone doesn't tell codegen *which* of 14-17 tables in a real document is the target - codegen was blindly guessing. Added `table_anchor_text` (a literal snippet from the target table's own header/title, since a raw table *index* isn't stable across documents in the same group when boilerplate table counts differ slightly).
- **Real extraction still doesn't reliably work end-to-end yet**, even with the anchor fix: Form 4's real table has THREE header-like rows (a merged title row, a numbered-label row, an abbreviated-label row) before real data starts. Generated code routinely fails to distinguish "header row" from "data row" correctly, or has ordinary bugs (one generated script called `re.match` without `import re` - a real `NameError`, correctly caught by the validation/retry loop, just not always successfully self-corrected within the attempt budget). This is the single biggest open problem - see "Known limitations."

## Known limitations (honest, not hidden)

- **Real-world multi-header-row tables are the current bottleneck.** Codegen's prompt now explicitly warns about this, but doesn't reliably solve it yet - a real, unsolved problem, not a documented-and-fixed one.
- **Minimal sandboxing.** Generated code runs in a plain subprocess with a timeout - no container/VM isolation, no network namespace, no filesystem restriction beyond convention. Acceptable risk for now, restricted to Docling-JSON-only input (see below); real sandboxing is required before raw-document input is ever supported, and is deliberately not built yet.
- **Docling JSON only, not raw documents.** By design, for now - raw (PDF/HTML/etc.) input needs real sandboxing first, since generated code would then run closer to attacker-controlled content.
- **Amazon test corpus not built.** Amazon's Terms of Service explicitly prohibit automated data collection for building a compilation/database - a real consideration for anything beyond a one-off, non-republished local test. Flagged rather than silently scraped; needs an explicit decision (a small one-time fetch vs. a public dataset vs. already-owned pages) before building this test.
- **Concept/schema drift is not handled.** See below - documented as a known future problem, not solved here.

## Concept/schema drift

Direct framing from this prototype's purpose (financial/hedge-fund-relevant data extraction): a document group's real structure can drift over time - a filing's line items change between fiscal years, a company changes its Form 4 layout, a product category gains new spec fields. A schema (and the code generated against it) built from one snapshot of a group can silently degrade in correctness as new documents diverge from that snapshot, rather than failing loudly. For a use case where the extracted data feeds financial decisions, "silently wrong structured data" is a materially worse failure mode than "visibly broken."

**Deliberately not solved in this pass** (explicit instruction: build the POC first, deal with drift later). Real future work this points to:
- Schema versioning - detect when a new document's tables no longer match a previously-accepted schema, rather than forcing a fit.
- Automatic re-validation - periodically re-run schema detection against recent documents and diff against the stored schema.
- Alerting - a codegen output that suddenly starts returning far fewer/more rows than its historical average, or starts failing its own schema validation, should be surfaced, not silently logged.

## Sandboxing note

Generated Python currently runs with only a subprocess timeout as isolation - a real, accepted risk for now, scoped down by only ever running it against Docling-JSON input (not raw documents an attacker-adjacent source could shape more directly). Real sandboxing (container isolation, no network namespace, a read-only filesystem view) is required before raw-document support is added - tracked as future work, not built here.

## Repo layout

```
src/docuzent_etl/
  docling_json.py   - Docling JSON -> compact table extraction
  ollama_client.py  - minimal Ollama client + robust JSON-from-model parsing
  schema_detect.py  - schema detection + validate/retry
  codegen.py        - codegen + execute + validate/retry
  pipeline.py       - orchestration + JSONL attempt logging
tests/              - model-free regression tests (25+ passing)
data/samples/       - real EDGAR Form 4 filings (fetched via SEC's own API)
data/docling_json/  - their real Docling JSON output
run_pipeline.py     - real, runnable CLI entry point
```

## Running it

```bash
pip install -r requirements.txt
python run_pipeline.py --model qwen2.5-coder:14b data/docling_json/*.json
python -m pytest tests/
```

Requires a local `ollama serve` with the chosen model pulled.
