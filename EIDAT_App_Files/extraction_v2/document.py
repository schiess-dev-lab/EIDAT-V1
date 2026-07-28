"""Document-level orchestration -> document.json (schema v2).

Every value in the output carries provenance: page, bbox, source
(text-layer or OCR), confidence. That is what makes trending trustworthy
and every extraction auditable back to pixels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .engines import Token, default_engine
from .grid import detect_tables
from .render import render_pages
from .tables import (
    apply_label_rules,
    build_table,
    find_title,
    merge_continuations,
    refine_cells,
)
from .kv import detect_kv_pairs
from .values import parse_value

SCHEMA_VERSION = "2.0"


def _kv_pairs_from_tables(tables) -> dict[str, str]:
    """Two-column Field/Value-style tables double as metadata kv sources."""
    kv: dict[str, str] = {}
    for t in tables:
        if t.n_cols != 2:
            continue
        hdr = [h.strip().lower() for h in t.header]
        if hdr and hdr[0] in {"field", "parameter", "item", "key"}:
            for row in t.grid_text()[1:]:
                if row[0]:
                    kv.setdefault(row[0].strip(), row[1].strip())
    return kv


def extract_document(
    pdf_path: str | Path,
    dpi: int = 300,
    engine=None,
    label_rules: list[dict] | None = None,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    engine = engine or default_engine()
    label_rules = label_rules or []

    pages_meta: list[dict] = []
    all_tables = []
    text_blocks: list[dict] = []
    kv_groups: list[dict] = []
    page_h_px = 0

    for pr in render_pages(str(pdf_path), dpi=dpi):
        page_h_px = pr.gray.shape[0]
        if pr.source == "text":
            tokens: list[Token] = pr.text_tokens
        else:
            tokens = engine.recognize(pr.gray)

        grids = detect_tables(pr.gray, dpi=dpi)
        page_tables = []
        for g in grids:
            t = build_table(g, tokens, pr.number)
            if pr.source == "ocr":
                refine_cells(t, pr.gray, engine)
            t.title = find_title(t, tokens)
            t.label = apply_label_rules(t, label_rules)
            page_tables.append(t)

        # non-table text -> plain text blocks (line-grouped)
        in_table = lambda tok: any(
            g.x0 <= tok.cx < g.x1 and g.y0 <= tok.cy < g.y1 for g in grids
        )
        free = [t for t in tokens if not in_table(t)]
        free_lines = _group_lines(free)
        for group in detect_kv_pairs(free_lines, pr.number, pr.gray.shape[1]):
            kv_groups.append(
                {
                    "group_id": f"p{pr.number}_kv{len(kv_groups)}",
                    "page": pr.number,
                    "bbox": [round(v, 1) for v in group.bbox],
                    "pairs": [
                        {
                            "label": p.label,
                            "value": p.value,
                            "value_typed": p.value_typed,
                            "bbox": [round(v, 1) for v in p.bbox],
                            "conf": round(p.conf, 3),
                        }
                        for p in group.pairs
                    ],
                }
            )
        for line in free_lines:
            text_blocks.append(
                {
                    "page": pr.number,
                    "text": " ".join(w.text for w in line),
                    "bbox": [
                        min(w.x0 for w in line),
                        min(w.y0 for w in line),
                        max(w.x1 for w in line),
                        max(w.y1 for w in line),
                    ],
                    "conf": round(min(w.conf for w in line), 3),
                }
            )

        pages_meta.append(
            {
                "n": pr.number,
                "px": [pr.gray.shape[1], pr.gray.shape[0]],
                "dpi": dpi,
                "source": pr.source,
                "tables": len(page_tables),
                "tokens": len(tokens),
            }
        )
        all_tables.extend(page_tables)

    all_tables = merge_continuations(all_tables, page_h_px, dpi)
    kv = _kv_pairs_from_tables(all_tables)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "engine": getattr(engine, "name", "unknown"),
        "source": {
            "file": pdf_path.name,
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        },
        "pages": pages_meta,
        "metadata": kv,
        "tables": [
            {
                "table_id": f"p{t.page}_t{i}",
                "page": t.page,
                "pages": t.merged_from_pages or [t.page],
                "title": t.title,
                "label": t.label,
                "bbox": list(t.bbox),
                "n_rows": t.n_rows,
                "n_cols": t.n_cols,
                "header": t.header,
                "rows": t.grid_text(),
                "cells": [
                    {
                        "r": c.r,
                        "c": c.c,
                        "text": c.text,
                        "value": c.value,
                        "bbox": [round(v, 1) for v in c.bbox],
                        "conf": round(c.conf, 3),
                    }
                    for c in t.cells
                ],
            }
            for i, t in enumerate(all_tables)
        ],
        "kv_groups": kv_groups,
        "text_blocks": text_blocks,
    }
    return doc


def _group_lines(tokens: list[Token]) -> list[list[Token]]:
    toks = sorted(tokens, key=lambda t: (t.y0, t.x0))
    lines: list[list[Token]] = []
    for t in toks:
        placed = False
        for ln in lines:
            ref = ln[-1]
            if abs(t.cy - ref.cy) < (ref.y1 - ref.y0) * 0.7:
                ln.append(t)
                placed = True
                break
        if not placed:
            lines.append([t])
    return [sorted(ln, key=lambda t: t.x0) for ln in lines]


def _jsonable(o):
    """Coerce numpy scalars that leak out of OpenCV/pypdfium2."""
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(doc: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, default=_jsonable)
    )
    return out_path
