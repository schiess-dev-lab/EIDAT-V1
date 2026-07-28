"""Assemble structured tables from grid geometry + tokens.

Tokens (from OCR or the PDF text layer — identical shape either way) are
assigned to cells by center-point containment. Titles are the nearest text
line above the table. Same-structure tables that continue across a page
break are merged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engines import Token
from .grid import TableGrid
from .values import parse_value


@dataclass
class TableCell:
    r: int
    c: int
    text: str
    value: dict[str, Any]
    bbox: tuple[float, float, float, float]
    conf: float


@dataclass
class Table:
    page: int
    bbox: tuple[float, float, float, float]
    n_rows: int
    n_cols: int
    cells: list[TableCell]
    title: str = ""
    label: str = ""
    merged_from_pages: list[int] = field(default_factory=list)

    @property
    def header(self) -> list[str]:
        return [cell.text for cell in self.cells if cell.r == 0]

    def grid_text(self) -> list[list[str]]:
        rows: list[list[str]] = [[""] * self.n_cols for _ in range(self.n_rows)]
        for cell in self.cells:
            rows[cell.r][cell.c] = cell.text
        return rows


def _tokens_in(tokens: list[Token], x0, y0, x1, y1) -> list[Token]:
    return [t for t in tokens if x0 <= t.cx < x1 and y0 <= t.cy < y1]


import re

_EDGE_ARTIFACTS = re.compile(r"^[|_\[\]{}~—-]+\s*|\s*[|_\[\]{}~]+$")


def _clean_cell_text(s: str) -> str:
    """Strip border-glyph OCR artifacts from cell edges."""
    s = _EDGE_ARTIFACTS.sub("", s.strip())
    return " ".join(s.split())


def _join(tokens: list[Token]) -> tuple[str, float]:
    if not tokens:
        return "", 1.0
    # order: line by line (cluster on y), then x
    toks = sorted(tokens, key=lambda t: (t.y0, t.x0))
    lines: list[list[Token]] = [[toks[0]]]
    for t in toks[1:]:
        if t.cy - lines[-1][-1].cy < (lines[-1][-1].y1 - lines[-1][-1].y0) * 0.6:
            lines[-1].append(t)
        else:
            lines.append([t])
    parts = [" ".join(w.text for w in sorted(ln, key=lambda t: t.x0)) for ln in lines]
    conf = min(t.conf for t in tokens)
    return " ".join(parts), conf


def build_table(grid: TableGrid, tokens: list[Token], page: int) -> Table:
    cells: list[TableCell] = []
    for cell in grid.cells:
        inside = _tokens_in(tokens, cell.x0, cell.y0, cell.x1, cell.y1)
        text, conf = _join(inside)
        text = _clean_cell_text(text)
        cells.append(
            TableCell(
                r=cell.r,
                c=cell.c,
                text=text,
                value=parse_value(text),
                bbox=(cell.x0, cell.y0, cell.x1, cell.y1),
                conf=conf,
            )
        )
    return Table(
        page=page,
        bbox=(grid.x0, grid.y0, grid.x1, grid.y1),
        n_rows=grid.n_rows,
        n_cols=grid.n_cols,
        cells=cells,
    )


def refine_cells(table: Table, gray, engine, conf_floor: float = 0.60) -> int:
    """Targeted per-cell re-OCR (single-line mode) for suspicious cells.

    A cell is suspicious if its confidence is low, or it failed to parse as
    a typed value while containing digits. Re-OCR of the isolated crop
    removes neighbouring-border interference and usually reads better.
    Returns the number of cells replaced.
    """
    if not hasattr(engine, "recognize_line"):
        return 0
    import numpy as np

    fixed = 0
    for cell in table.cells:
        x0, y0, x1, y1 = (int(v) for v in cell.bbox)
        pad = 3
        crop = gray[max(0, y0 + pad) : y1 - pad, max(0, x0 + pad) : x1 - pad]
        if crop.size == 0:
            continue
        has_ink = bool(np.mean(crop < 128) > 0.001)
        digity = any(ch.isdigit() for ch in cell.text)
        suspicious = (
            (cell.conf < conf_floor)
            or (cell.value.get("kind") == "text" and digity)
            or (not cell.text and has_ink)  # page-level OCR missed the cell
        )
        if not suspicious:
            continue
        new_text, new_conf = engine.recognize_line(crop)
        new_text = _clean_cell_text(new_text)
        if new_text and (new_conf > cell.conf or not cell.text):
            cell.text = new_text
            cell.conf = new_conf
            cell.value = parse_value(new_text)
            fixed += 1
    return fixed


def find_title(table: Table, tokens: list[Token], max_gap_px: int = 120) -> str:
    """Nearest text line fully above the table within max_gap_px."""
    above = [
        t
        for t in tokens
        if t.y1 <= table.bbox[1]
        and table.bbox[1] - t.y1 < max_gap_px
        and t.x1 > table.bbox[0] - 20
        and t.x0 < table.bbox[2] + 20
    ]
    if not above:
        return ""
    bottom = max(t.y1 for t in above)
    line = [t for t in above if bottom - t.y1 < (t.y1 - t.y0) * 1.2]
    return " ".join(t.text for t in sorted(line, key=lambda t: t.x0))


def apply_label_rules(table: Table, rules: list[dict]) -> str:
    """Port of the table_label_rules idea: match on title/header keywords,
    highest priority wins."""
    best = ("", -1)
    hay = (table.title + " | " + " | ".join(table.header)).lower()
    for rule in rules:
        pri = rule.get("priority", 0)
        keywords = [k.lower() for k in rule.get("keywords", [])]
        if keywords and all(k in hay for k in keywords) and pri > best[1]:
            best = (rule.get("label", ""), pri)
    return best[0] or table.title


def merge_continuations(tables: list[Table], page_h: int, dpi: int) -> list[Table]:
    """Merge a table that resumes at the top of the next page.

    Conditions: previous table ends in the bottom margin zone, next starts in
    the top zone, same column count, column x-bounds within tolerance.
    """
    if not tables:
        return tables
    tol = dpi // 20
    out: list[Table] = [tables[0]]
    for t in tables[1:]:
        prev = out[-1]
        same_cols = prev.n_cols == t.n_cols and all(
            abs((prev.bbox[0] + a) - (t.bbox[0] + b)) < tol
            for a, b in zip(_col_offsets(prev), _col_offsets(t))
        )
        if (
            t.page == prev.page + 1
            and same_cols
            and prev.bbox[3] > page_h * 0.82
            and t.bbox[1] < page_h * 0.18
        ):
            # drop repeated header row if identical
            rows = t.grid_text()
            skip_header = rows and rows[0] == prev.grid_text()[0]
            base = prev.n_rows
            for cell in t.cells:
                if skip_header and cell.r == 0:
                    continue
                cell.r = base + cell.r - (1 if skip_header else 0)
                prev.cells.append(cell)
            prev.n_rows = max(c.r for c in prev.cells) + 1
            prev.merged_from_pages = sorted(
                set(prev.merged_from_pages or [prev.page]) | {t.page}
            )
        else:
            out.append(t)
    return out


def _col_offsets(t: Table) -> list[float]:
    xs = sorted({round(c.bbox[0] - t.bbox[0], 1) for c in t.cells})
    return xs
