"""Reads Docling's own JSON output and extracts a compact table
representation - a plain grid of cell text per table, not the full
Docling document schema (which also carries layout/provenance metadata
irrelevant to schema detection and would bloat every LLM prompt for no
benefit). This is the only supported input shape for now: raw
(non-Docling) documents are explicitly out of scope until real sandboxing
exists for that path - see the repo README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedTable:
    table_index: int
    num_rows: int
    num_cols: int
    rows: list[list[str]]

    def is_trivial(self) -> bool:
        """A 1x1 table, or one where every cell is empty, is Docling
        picking up a layout artifact (a spacer, a single-cell wrapper),
        not real structured data - cheap to filter before ever asking an
        LLM to look at it."""
        if self.num_rows <= 1 and self.num_cols <= 1:
            return True
        return not any(cell.strip() for row in self.rows for cell in row)

    def to_markdown(self, max_rows: int = 30) -> str:
        """A compact, LLM-friendly rendering - full tables can be long,
        so this caps how many rows are shown per table in a prompt
        (schema detection needs to see the *shape*, not every row)."""
        rows = self.rows[:max_rows]
        lines = ["| " + " | ".join(c.strip() or " " for c in row) + " |" for row in rows]
        if len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows) - max_rows} more rows)")
        return "\n".join(lines)


def load_tables(docling_json_path: Path) -> list[ExtractedTable]:
    """Extracts every non-trivial table from one Docling JSON file, in
    document order. Docling's own table index (position in `tables`) is
    preserved as `table_index` so a later step can point back at exactly
    which table in the source document a schema/extraction came from."""
    doc = json.loads(docling_json_path.read_text(encoding="utf-8"))
    extracted: list[ExtractedTable] = []
    for i, table in enumerate(doc.get("tables", [])):
        data = table.get("data", {})
        grid = data.get("grid", [])
        rows = [[cell.get("text", "") for cell in row] for row in grid]
        num_rows = data.get("num_rows", len(rows))
        num_cols = data.get("num_cols", len(rows[0]) if rows else 0)
        t = ExtractedTable(table_index=i, num_rows=num_rows, num_cols=num_cols, rows=rows)
        if not t.is_trivial():
            extracted.append(t)
    return extracted


def load_group_tables(docling_json_paths: list[Path]) -> dict[str, list[ExtractedTable]]:
    """Same as `load_tables`, but for a whole document group - schema
    detection needs to see tables across *multiple* documents to find
    the shape that's actually consistent, not just what one document
    happens to contain."""
    return {str(p): load_tables(p) for p in docling_json_paths}
