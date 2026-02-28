"""
Multi-page ASCII table merger for combined.txt (post-processing only).

This module scans a finished `combined.txt`, detects bordered ASCII tables that
continue across page boundaries, merges continuation segments into the first page
where the table begins, and blanks out the continuation regions on later pages.

It intentionally operates on `combined.txt` lines only (no PDFs/images/JSON).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional


PAGE_MARKER_RE = re.compile(r"^=== Page (\d+) ===\s*$")
TABLE_TITLE_MARKER = "[Table/Chart Title]"
TABLE_MARKER_RE = re.compile(r"^\[Table(?:\s+\d+)?\]\s*$")
TABLE_LABEL_MARKER = "[TABLE_LABEL]"

ASCII_BORDER_RE = re.compile(r"^\s*\+[-=+]+\+\s*$")
ASCII_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_LABEL_SUFFIX_RE = re.compile(r"\s*\(\s*\d+\s*\)\s*$")


_DEFAULT_CFG: dict[str, Any] = {
    "version": 1,
    "min_body_rows": 8,
    "max_col_width_delta_ratio": 0.15,
    "max_col_width_delta_abs": 3,
    "max_plus_pos_delta_chars": 2,
    "candidate_tables_prev": 2,
    "candidate_tables_next": 2,
    "header_block_similarity_below": 0.2,
    "title_block_fuzzy_ratio_below": 0.5,
}


def load_table_merge_heuristics(path: Path) -> dict[str, Any]:
    """
    Best-effort load of table merge heuristics (schema v1).

    Returns {} on missing/invalid.
    """
    try:
        p = Path(path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(data, dict):
            return {}
        if int(data.get("version") or 0) != 1:
            return {}
        return data
    except Exception:
        return {}


def _cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(_DEFAULT_CFG)
    if isinstance(cfg, dict):
        out.update({k: v for k, v in cfg.items() if v is not None})
    return out


def _newline_for_line(ln: str) -> str:
    if str(ln).endswith("\r\n"):
        return "\r\n"
    if str(ln).endswith("\n"):
        return "\n"
    return "\n"


def _is_border(ln: str) -> bool:
    s = str(ln).rstrip("\r\n")
    return bool(ASCII_BORDER_RE.match(s.strip())) and ("-" in s or "=" in s)


def _is_pipe_row(ln: str) -> bool:
    s = str(ln).rstrip("\r\n")
    return bool(ASCII_PIPE_ROW_RE.match(s))


def _norm_cell(s: str) -> str:
    s2 = str(s or "").lower()
    s2 = s2.replace("\u00A0", " ")
    s2 = re.sub(r"[\t\r\n]+", " ", s2)
    s2 = re.sub(r"[\"'`“”‘’]", "", s2)
    s2 = re.sub(r"[^\w\s./+-]+", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _split_cells_from_pipe_row(ln: str) -> list[str]:
    s = str(ln).rstrip("\r\n").strip()
    if not s.startswith("|"):
        return []
    parts = [p.strip() for p in s.strip().strip("|").split("|")]
    while parts and not parts[-1]:
        parts.pop()
    return parts


def _header_similarity(a_cells: list[str], b_cells: list[str]) -> float:
    if not a_cells or not b_cells:
        return 0.0
    if len(a_cells) != len(b_cells):
        return 0.0
    a_n = [_norm_cell(c) for c in a_cells]
    b_n = [_norm_cell(c) for c in b_cells]
    if not any(a_n) or not any(b_n):
        return 0.0
    matches = sum(1 for x, y in zip(a_n, b_n) if x and y and x == y)
    return matches / max(len(a_n), 1)


def _title_ratio(a: str, b: str) -> float:
    an = _norm_cell(a)
    bn = _norm_cell(b)
    if not an or not bn:
        return 0.0
    if an == bn:
        return 1.0
    return float(SequenceMatcher(None, an, bn).ratio())


@dataclass(frozen=True)
class _PageRange:
    page: int
    start: int
    end: int


@dataclass
class _AsciiTable:
    page: int
    ascii_start: int
    ascii_end: int
    title: str
    table_label: str
    table_label_base: str
    indent: int
    plus_positions: tuple[int, ...]
    col_widths: tuple[int, ...]
    header_cells: list[str]
    pipe_row_count: int
    body_row_count_est: int


def _iter_page_ranges(lines: list[str]) -> list[_PageRange]:
    pages: list[tuple[int, int]] = []
    for i, ln in enumerate(lines):
        m = PAGE_MARKER_RE.match(str(ln).rstrip("\r\n"))
        if m:
            pages.append((int(m.group(1)), i))
    pages.sort(key=lambda x: x[1])
    if not pages:
        return [_PageRange(page=0, start=0, end=len(lines))]
    out: list[_PageRange] = []
    for idx, (p, start) in enumerate(pages):
        end = pages[idx + 1][1] if idx + 1 < len(pages) else len(lines)
        out.append(_PageRange(page=p, start=start, end=end))
    return out


def _find_preceding_title(lines: list[str], *, page_start: int, before_idx: int) -> str:
    # Best-effort: find the closest preceding [Table/Chart Title] payload line in-page.
    i = before_idx - 1
    while i >= page_start:
        if str(lines[i]).rstrip("\r\n").strip() == TABLE_TITLE_MARKER:
            j = i + 1
            while j < before_idx and not str(lines[j]).strip():
                j += 1
            if j < before_idx:
                return str(lines[j]).rstrip("\r\n").strip()
            return ""
        i -= 1
    return ""


def _base_table_label(label: str) -> str:
    s = str(label or "").strip()
    if not s:
        return ""
    s2 = _LABEL_SUFFIX_RE.sub("", s).strip()
    return s2 or s


def _find_preceding_table_label(lines: list[str], *, page_start: int, before_idx: int) -> str:
    """
    Best-effort: find the closest preceding [TABLE_LABEL] payload line in-page.

    Only returns a label if the marker/payload block appears to apply to the
    table that begins at `before_idx` (i.e., only blank lines between payload and table start).
    """
    i = before_idx - 1
    while i >= page_start:
        if str(lines[i]).rstrip("\r\n").strip() == TABLE_LABEL_MARKER:
            j = i + 1
            while j < before_idx and not str(lines[j]).strip():
                j += 1
            if j >= before_idx:
                return ""
            payload = str(lines[j]).rstrip("\r\n").strip()
            if not payload:
                return ""
            k = j + 1
            while k < before_idx:
                if str(lines[k]).strip():
                    return ""
                k += 1
            return payload
        i -= 1
    return ""


def _extract_layout_signature(border_line: str) -> tuple[int, tuple[int, ...], int, int]:
    b = str(border_line).rstrip("\r\n")
    plus = [k for k, ch in enumerate(b) if ch == "+"]
    indent = plus[0] if plus else 0
    widths = tuple((plus[k + 1] - plus[k] - 1) for k in range(len(plus) - 1))
    return indent, tuple(plus), widths, len(b)


def _extract_header_cells(block_lines: list[str]) -> list[str]:
    for ln in block_lines:
        if _is_pipe_row(ln):
            return _split_cells_from_pipe_row(ln)
    return []


def _estimate_body_rows(block_lines: list[str]) -> int:
    pipe_idxs = [i for i, ln in enumerate(block_lines) if _is_pipe_row(ln)]
    if not pipe_idxs:
        return 0
    # If there is a header separator using '=', treat pipe rows before it as headers.
    eq_sep = next((i for i, ln in enumerate(block_lines) if _is_border(ln) and "=" in str(ln)), None)
    header_rows = 1
    if eq_sep is not None:
        # Count pipe rows that occur before the '=' separator.
        header_rows = sum(1 for i in pipe_idxs if i < eq_sep)
        header_rows = max(header_rows, 1)
    return max(len(pipe_idxs) - header_rows, 0)


def _iter_ascii_tables(lines: list[str], pages: list[_PageRange]) -> list[_AsciiTable]:
    out: list[_AsciiTable] = []
    page_by_idx: list[tuple[int, int, int]] = [(p.page, p.start, p.end) for p in pages]
    # Scan per-page to avoid accidentally spanning across a page marker.
    for page, page_start, page_end in page_by_idx:
        i = page_start
        while i < page_end:
            ln = lines[i]
            if _is_border(ln):
                start = i
                j = i + 1
                while j < page_end:
                    s2 = str(lines[j]).rstrip("\r\n")
                    if not s2.strip():
                        break
                    if PAGE_MARKER_RE.match(s2.strip()):
                        break
                    if not s2.strip().startswith(("+", "|")):
                        break
                    j += 1
                block = lines[start:j]
                header_cells = _extract_header_cells(block)
                pipe_row_count = sum(1 for ln2 in block if _is_pipe_row(ln2))
                title = _find_preceding_title(lines, page_start=page_start, before_idx=start)
                table_label = _find_preceding_table_label(lines, page_start=page_start, before_idx=start)
                indent, plus_positions, col_widths, _total_len = _extract_layout_signature(block[0])
                out.append(
                    _AsciiTable(
                        page=page,
                        ascii_start=start,
                        ascii_end=j,
                        title=title,
                        table_label=table_label,
                        table_label_base=_base_table_label(table_label),
                        indent=int(indent),
                        plus_positions=tuple(plus_positions),
                        col_widths=tuple(col_widths),
                        header_cells=header_cells,
                        pipe_row_count=int(pipe_row_count),
                        body_row_count_est=int(_estimate_body_rows(block)),
                    )
                )
                i = j + 1
                continue
            i += 1
    return out


def _layout_compatible(a: _AsciiTable, b: _AsciiTable, cfg: dict[str, Any]) -> bool:
    if len(a.col_widths) != len(b.col_widths) or not a.col_widths or not b.col_widths:
        return False
    try:
        ratio = float(cfg.get("max_col_width_delta_ratio", 0.15) or 0.15)
    except Exception:
        ratio = 0.15
    try:
        abs_max = int(cfg.get("max_col_width_delta_abs", 3) or 3)
    except Exception:
        abs_max = 3
    for wa, wb in zip(a.col_widths, b.col_widths):
        tol = max(abs_max, int(round(ratio * max(int(wa), int(wb), 1))))
        if abs(int(wa) - int(wb)) > tol:
            return False
    rel_a = [p - int(a.indent) for p in a.plus_positions]
    rel_b = [p - int(b.indent) for p in b.plus_positions]
    if len(rel_a) != len(rel_b) or not rel_a:
        return False
    mean_delta = sum(abs(x - y) for x, y in zip(rel_a, rel_b)) / float(len(rel_a))
    try:
        max_plus = float(cfg.get("max_plus_pos_delta_chars", 2) or 2)
    except Exception:
        max_plus = 2.0
    return mean_delta <= max_plus


def _layout_distance(a: _AsciiTable, b: _AsciiTable) -> float:
    if len(a.col_widths) != len(b.col_widths) or not a.col_widths:
        return 1e9
    diffs = [abs(int(x) - int(y)) / float(max(int(x), int(y), 1)) for x, y in zip(a.col_widths, b.col_widths)]
    wdist = max(diffs) if diffs else 1e9
    rel_a = [p - int(a.indent) for p in a.plus_positions]
    rel_b = [p - int(b.indent) for p in b.plus_positions]
    if len(rel_a) != len(rel_b) or not rel_a:
        return 1e9
    pdist = sum(abs(x - y) for x, y in zip(rel_a, rel_b)) / float(len(rel_a))
    return float(wdist) + (float(pdist) * 0.01)


def _pick_boundary_pair(
    prev_tables: list[_AsciiTable],
    next_tables: list[_AsciiTable],
    *,
    cfg: dict[str, Any],
) -> Optional[tuple[_AsciiTable, _AsciiTable]]:
    best: Optional[tuple[float, _AsciiTable, _AsciiTable]] = None
    for a in prev_tables:
        for b in next_tables:
            if not _layout_compatible(a, b, cfg):
                continue
            # Negative content guardrails (only to block obvious false merges).
            hs = _header_similarity(a.header_cells, b.header_cells)
            try:
                header_block_below = float(cfg.get("header_block_similarity_below", 0.2) or 0.2)
            except Exception:
                header_block_below = 0.2
            if a.header_cells and b.header_cells and hs < header_block_below:
                continue

            if a.title and b.title:
                tr = _title_ratio(a.title, b.title)
                try:
                    title_block_below = float(cfg.get("title_block_fuzzy_ratio_below", 0.5) or 0.5)
                except Exception:
                    title_block_below = 0.5
                if tr < title_block_below:
                    continue

            dist = _layout_distance(a, b)
            if best is None or dist < best[0]:
                best = (dist, a, b)
    if best is None:
        return None
    return best[1], best[2]


def _find_dash_border_line(block_lines: list[str]) -> str:
    for ln in block_lines:
        s = str(ln).rstrip("\r\n")
        if _is_border(ln) and "-" in s and "=" not in s:
            return ln
    # Fall back to last border line.
    for ln in reversed(block_lines):
        if _is_border(ln):
            return ln
    return "+\n"


def _find_bottom_border(block_lines: list[str]) -> tuple[int, str]:
    for idx in range(len(block_lines) - 1, -1, -1):
        if _is_border(block_lines[idx]):
            return idx, block_lines[idx]
    return len(block_lines) - 1, block_lines[-1] if block_lines else "\n"


def _extract_body_segment_lines(
    block_lines: list[str],
    *,
    leader_header_cells: list[str],
    leader_dash_border: str,
) -> list[str]:
    pipe_idxs = [i for i, ln in enumerate(block_lines) if _is_pipe_row(ln)]
    if not pipe_idxs:
        return []

    eq_sep = next((i for i, ln in enumerate(block_lines) if _is_border(ln) and "=" in str(ln)), None)
    body_pipe_idx: Optional[int] = None
    if eq_sep is not None:
        # First pipe row after the '=' separator.
        body_pipe_idx = next((i for i in pipe_idxs if i > eq_sep), None)
    else:
        # If the first row is a repeated header, drop it.
        first_cells = _split_cells_from_pipe_row(block_lines[pipe_idxs[0]])
        if leader_header_cells and _header_similarity(leader_header_cells, first_cells) >= 0.8:
            body_pipe_idx = pipe_idxs[1] if len(pipe_idxs) >= 2 else None
        else:
            body_pipe_idx = pipe_idxs[0]

    if body_pipe_idx is None:
        return []

    # Find border immediately preceding first body pipe row.
    start = None
    for i in range(body_pipe_idx - 1, -1, -1):
        if _is_border(block_lines[i]):
            start = i
            break
    if start is None:
        start = 0

    seg = list(block_lines[start:])
    if seg and _is_border(seg[0]) and "=" in str(seg[0]):
        # Replace header separator with normal '-' border for body continuation.
        seg[0] = leader_dash_border
    return seg


def _find_blank_span_start(
    lines: list[str],
    *,
    page_start: int,
    table_ascii_start: int,
) -> int:
    """
    Choose where to start blanking out a continuation region.

    Preference order (within page slice):
      1) [Table/Chart Title] (and its payload) if directly precedes the table
      2) [Table] / [Table N] if directly precedes the table
      3) ASCII border start
    """
    start = table_ascii_start
    # Only consider nearby markers; avoid reaching into unrelated sections.
    window = 40
    lo = max(page_start, table_ascii_start - window)

    def _directly_precedes(marker_idx: int) -> bool:
        j = marker_idx + 1
        while j < table_ascii_start:
            s = str(lines[j]).rstrip("\r\n").strip()
            if not s:
                j += 1
                continue
            if s == TABLE_TITLE_MARKER:
                return False
            if s == TABLE_LABEL_MARKER:
                j += 1
                while j < table_ascii_start and not str(lines[j]).strip():
                    j += 1
                if j < table_ascii_start:
                    j += 1  # label payload line
                while j < table_ascii_start and not str(lines[j]).strip():
                    j += 1
                continue
            if TABLE_MARKER_RE.match(s) or s == "[Table]":
                j += 1
                continue
            return False
        return True

    def _title_directly_precedes(marker_idx: int) -> bool:
        # Title marker must have a payload line; after payload, only allow blanks and [Table] markers.
        j = marker_idx + 1
        while j < table_ascii_start and not str(lines[j]).strip():
            j += 1
        if j >= table_ascii_start:
            return False
        # j is payload (title text); allow it even though it's non-marker content.
        k = j + 1
        while k < table_ascii_start:
            s = str(lines[k]).rstrip("\r\n").strip()
            if not s:
                k += 1
                continue
            if s == TABLE_LABEL_MARKER:
                k += 1
                while k < table_ascii_start and not str(lines[k]).strip():
                    k += 1
                if k < table_ascii_start:
                    k += 1  # label payload line
                while k < table_ascii_start and not str(lines[k]).strip():
                    k += 1
                continue
            if TABLE_MARKER_RE.match(s) or s == "[Table]":
                k += 1
                continue
            return False
        return True

    def _label_directly_precedes(marker_idx: int) -> bool:
        # Label marker must have a payload line; after payload, only allow blanks until the table.
        j = marker_idx + 1
        while j < table_ascii_start and not str(lines[j]).strip():
            j += 1
        if j >= table_ascii_start:
            return False
        # j is payload (label text); allow it even though it's non-marker content.
        k = j + 1
        while k < table_ascii_start:
            if str(lines[k]).strip():
                return False
            k += 1
        return True

    # Prefer [Table/Chart Title]
    for i in range(table_ascii_start - 1, lo - 1, -1):
        if str(lines[i]).rstrip("\r\n").strip() == TABLE_TITLE_MARKER and _title_directly_precedes(i):
            return i

    # Else [Table] / [Table N]
    for i in range(table_ascii_start - 1, lo - 1, -1):
        s = str(lines[i]).rstrip("\r\n").strip()
        if TABLE_MARKER_RE.match(s) or s == "[Table]":
            if _directly_precedes(i):
                return i

    # Else [TABLE_LABEL]
    for i in range(table_ascii_start - 1, lo - 1, -1):
        if str(lines[i]).rstrip("\r\n").strip() == TABLE_LABEL_MARKER and _label_directly_precedes(i):
            return i

    return start


def _norm_phrase_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for p in raw:
        s = _norm_cell(str(p or ""))
        if s:
            out.append(s)
    return out


def _norm_table_text(lines: list[str], t: _AsciiTable) -> str:
    cells: list[str] = []
    for ln in lines[t.ascii_start : t.ascii_end]:
        if _is_pipe_row(ln):
            cells.extend(_split_cells_from_pipe_row(ln))
    return _norm_cell(" ".join(cells))


def _find_blank_span_end(lines: list[str], *, page_end: int, table_ascii_end: int) -> int:
    end = table_ascii_end
    i = end
    while i < page_end:
        s = str(lines[i]).rstrip("\r\n")
        if PAGE_MARKER_RE.match(s.strip()):
            break
        if s.strip():
            break
        i += 1
    return i


def merge_multipage_tables_in_combined_lines(lines: list[str], cfg: dict[str, Any] | None = None) -> list[str]:
    """
    Merge multi-page ASCII table continuations into their first appearance page.

    Input should be `splitlines(True)` to preserve newline styles.
    Returns new lines list (may be identical to input if no merges).
    """
    cfg2 = _cfg(cfg)
    pages = _iter_page_ranges(lines)
    tables = _iter_ascii_tables(lines, pages)
    if not tables:
        return lines

    tables_by_page: dict[int, list[_AsciiTable]] = {}
    for t in tables:
        tables_by_page.setdefault(int(t.page), []).append(t)
    for p in list(tables_by_page.keys()):
        tables_by_page[p].sort(key=lambda x: x.ascii_start)

    page_range_by_page = {p.page: p for p in pages}
    sorted_pages = sorted(page_range_by_page.keys())

    try:
        k_prev = int(cfg2.get("candidate_tables_prev", 2) or 2)
    except Exception:
        k_prev = 2
    try:
        k_next = int(cfg2.get("candidate_tables_next", 2) or 2)
    except Exception:
        k_next = 2
    try:
        min_body = int(cfg2.get("min_body_rows", 8) or 8)
    except Exception:
        min_body = 8

    label_merge_rules = cfg2.get("label_merge_rules")
    label_rules: list[dict[str, Any]] = label_merge_rules if isinstance(label_merge_rules, list) else []
    try:
        protect_labeled = bool(cfg2.get("protect_labeled_tables", False))
    except Exception:
        protect_labeled = False
    allowed_label_bases: set[str] = {_base_table_label(r.get("label")) for r in label_rules if isinstance(r, dict)}
    allowed_label_bases = {x for x in allowed_label_bases if x}

    debug_dump_path = str(cfg2.get("debug_dump_path") or "").strip()
    debug_enabled = bool(debug_dump_path)
    debug_report: dict[str, Any] = {"version": 1, "pages": [], "decisions": []}
    if debug_enabled:
        # Precompute per-page table summaries.
        for p in sorted(tables_by_page.keys()):
            debug_report["pages"].append(
                {
                    "page": int(p),
                    "tables": [
                        {
                            "ascii_start": int(t.ascii_start),
                            "ascii_end": int(t.ascii_end),
                            "cols": int(len(t.col_widths)),
                            "body_rows_est": int(t.body_row_count_est),
                            "title": str(t.title or ""),
                            "table_label": str(t.table_label or ""),
                            "table_label_base": str(t.table_label_base or ""),
                        }
                        for t in tables_by_page.get(p, [])
                    ],
                }
            )

    # Build continuation groups: leader -> [continuations...]
    leader_for: dict[tuple[int, int], tuple[int, int]] = {}
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    used_as_continuation: set[tuple[int, int]] = set()

    def _table_key(t: _AsciiTable) -> tuple[int, int]:
        return (int(t.page), int(t.ascii_start))

    for idx, p in enumerate(sorted_pages[:-1]):
        p_next = sorted_pages[idx + 1]
        prev_list = tables_by_page.get(p, [])
        next_list = tables_by_page.get(p_next, [])
        if not prev_list or not next_list:
            continue
        decision: dict[str, Any] = {"boundary": f"{int(p)}->{int(p_next)}", "forced": [], "legacy": None}

        # 0) Label-driven forced merges (best-effort, per allowlisted label).
        for rr in label_rules:
            if not isinstance(rr, dict):
                continue
            rule_label = _base_table_label(rr.get("label"))
            if not rule_label:
                continue
            try:
                sp = int(rr.get("search_prev_tables", 0) or 0)
            except Exception:
                sp = 0
            try:
                sn = int(rr.get("search_next_tables", 0) or 0)
            except Exception:
                sn = 0
            sp = max(sp, 1) if sp else max(k_prev, 1)
            sn = max(sn, 1) if sn else max(k_next, 1)
            try:
                min_override_raw = rr.get("min_body_rows_override", None)
                min_override = None if min_override_raw is None else int(min_override_raw)
            except Exception:
                min_override = None
            try:
                require_layout = bool(rr.get("require_layout_compatible", True))
            except Exception:
                require_layout = True
            try:
                allow_unlabeled = bool(rr.get("allow_unlabeled_continuation", False))
            except Exception:
                allow_unlabeled = False
            must_all = _norm_phrase_list(rr.get("union_must_contain_all") or [])

            prev_cands_r = prev_list[-max(sp, 1) :]
            next_cands_r = next_list[: max(sn, 1)]
            leader_opts = [t for t in prev_cands_r if t.table_label_base and t.table_label_base == rule_label]
            if not leader_opts:
                decision["forced"].append({"label": rule_label, "ok": False, "reason": "leader_not_found"})
                continue
            a = leader_opts[-1]
            ak = _table_key(a)
            min_body_rule = min_body if min_override is None else max(int(min_override), 0)
            if ak not in leader_for and int(a.body_row_count_est) < min_body_rule:
                decision["forced"].append(
                    {"label": rule_label, "ok": False, "reason": "leader_too_small", "body_rows_est": int(a.body_row_count_est)}
                )
                continue

            cont_opts = [t for t in next_cands_r if t.table_label_base and t.table_label_base == rule_label]
            b: _AsciiTable | None = None
            if cont_opts:
                b = cont_opts[0]
            elif allow_unlabeled:
                # Conservative: only accept an unlabeled continuation if it is the only compatible candidate.
                unl = [t for t in next_cands_r if not t.table_label_base]
                compat = []
                for cand in unl:
                    if require_layout and not _layout_compatible(a, cand, cfg2):
                        continue
                    compat.append(cand)
                if len(compat) == 1:
                    b = compat[0]

            if b is None:
                decision["forced"].append({"label": rule_label, "ok": False, "reason": "continuation_not_found"})
                continue

            bk = _table_key(b)
            if bk in used_as_continuation:
                decision["forced"].append({"label": rule_label, "ok": False, "reason": "continuation_already_used"})
                continue
            if require_layout and not _layout_compatible(a, b, cfg2):
                decision["forced"].append({"label": rule_label, "ok": False, "reason": "layout_incompatible"})
                continue
            if must_all:
                hay = f"{_norm_table_text(lines, a)} {_norm_table_text(lines, b)}".strip()
                if not hay or any(ph not in hay for ph in must_all):
                    decision["forced"].append({"label": rule_label, "ok": False, "reason": "union_keywords_missing"})
                    continue

            leader = leader_for.get(ak, ak)
            groups.setdefault(leader, [])
            groups[leader].append(bk)
            leader_for[bk] = leader
            used_as_continuation.add(bk)
            decision["forced"].append({"label": rule_label, "ok": True, "leader": ak, "cont": bk})

        # 1) Legacy heuristic merge for unlabeled tables (or label-allowed if protection disabled).
        prev_cands = prev_list[-max(k_prev, 1) :]
        next_cands = next_list[: max(k_next, 1)]
        if protect_labeled:
            # Disallow legacy merges involving labeled tables unless the label is explicitly allowlisted and matches.
            def _legacy_ok(a: _AsciiTable, b: _AsciiTable) -> bool:
                if not a.table_label_base and not b.table_label_base:
                    return True
                if a.table_label_base and b.table_label_base and a.table_label_base == b.table_label_base:
                    return a.table_label_base in allowed_label_bases
                return False

            # Reduce the cartesian search space by filtering obvious no-gos.
            prev_cands = [t for t in prev_cands if (not t.table_label_base) or (t.table_label_base in allowed_label_bases)]
            next_cands = [t for t in next_cands if (not t.table_label_base) or (t.table_label_base in allowed_label_bases)]
        else:
            _legacy_ok = None  # type: ignore[assignment]

        pair = _pick_boundary_pair(prev_cands, next_cands, cfg=cfg2) if (prev_cands and next_cands) else None
        if pair:
            a, b = pair
            if protect_labeled and _legacy_ok is not None and not _legacy_ok(a, b):
                decision["legacy"] = {"ok": False, "reason": "protected_labels_blocked"}
            else:
                ak = _table_key(a)
                if ak not in leader_for and int(a.body_row_count_est) < min_body:
                    decision["legacy"] = {"ok": False, "reason": "leader_too_small", "body_rows_est": int(a.body_row_count_est)}
                else:
                    bk = _table_key(b)
                    if bk in used_as_continuation:
                        decision["legacy"] = {"ok": False, "reason": "continuation_already_used"}
                    else:
                        leader = leader_for.get(ak, ak)
                        groups.setdefault(leader, [])
                        groups[leader].append(bk)
                        leader_for[bk] = leader
                        used_as_continuation.add(bk)
                        decision["legacy"] = {"ok": True, "leader": ak, "cont": bk}
        else:
            decision["legacy"] = {"ok": False, "reason": "no_pair"}

        if debug_enabled:
            debug_report["decisions"].append(decision)

    if not groups:
        if debug_enabled:
            try:
                dp = Path(debug_dump_path).expanduser()
                if not dp.is_absolute():
                    repo_root = Path(__file__).resolve().parents[2]
                    dp = repo_root / dp
                dp.parent.mkdir(parents=True, exist_ok=True)
                dp.write_text(json.dumps(debug_report, indent=2), encoding="utf-8")
            except Exception:
                pass
        return lines

    # Build replacements as slice operations on the original lines indices.
    replacements: list[tuple[int, int, list[str]]] = []

    # Helper to fetch table object by key.
    table_by_key: dict[tuple[int, int], _AsciiTable] = {_table_key(t): t for t in tables}

    # 1) Leader table replacements (merged blocks)
    for leader_key, cont_keys in groups.items():
        leader = table_by_key.get(leader_key)
        if leader is None:
            continue
        cont_tables = [table_by_key[k] for k in cont_keys if k in table_by_key]
        if not cont_tables:
            continue

        leader_block = list(lines[leader.ascii_start : leader.ascii_end])
        if not leader_block:
            continue
        dash_border = _find_dash_border_line(leader_block)
        bottom_idx, bottom_border = _find_bottom_border(leader_block)
        # Remove bottom border from leader block (we'll re-add after appending rows).
        if 0 <= bottom_idx < len(leader_block):
            leader_block = leader_block[:bottom_idx] + leader_block[bottom_idx + 1 :]

        merged = list(leader_block)
        for ct in cont_tables:
            seg = _extract_body_segment_lines(
                list(lines[ct.ascii_start : ct.ascii_end]),
                leader_header_cells=leader.header_cells,
                leader_dash_border=dash_border,
            )
            if not seg:
                continue
            # Drop segment's final border; leader's final border will close the merged table.
            if seg and _is_border(seg[-1]):
                seg = seg[:-1]
            # If the merged block currently ends with a border, and seg begins with a border, drop seg's first border.
            if merged and seg and _is_border(merged[-1]) and _is_border(seg[0]):
                seg = seg[1:]
            merged.extend(seg)

        # Close the merged table with the leader's original bottom border.
        merged.append(bottom_border)
        replacements.append((leader.ascii_start, leader.ascii_end, merged))

    # 2) Continuation page blank-outs (replace span with empty lines)
    for leader_key, cont_keys in groups.items():
        for ck in cont_keys:
            ct = table_by_key.get(ck)
            if ct is None:
                continue
            pr = page_range_by_page.get(int(ct.page))
            if pr is None:
                continue
            blank_start = _find_blank_span_start(lines, page_start=pr.start, table_ascii_start=ct.ascii_start)
            blank_end = _find_blank_span_end(lines, page_end=pr.end, table_ascii_end=ct.ascii_end)
            if blank_end <= blank_start:
                continue
            blanks = [_newline_for_line(lines[i]) for i in range(blank_start, blank_end)]
            replacements.append((blank_start, blank_end, blanks))

    # Apply replacements from bottom to top (descending start index).
    replacements.sort(key=lambda x: (x[0], x[1]), reverse=True)

    out_lines = list(lines)
    for start, end, new_lines in replacements:
        if start < 0 or end < start or start > len(out_lines):
            continue
        end2 = min(end, len(out_lines))
        out_lines[start:end2] = list(new_lines)

    if debug_enabled:
        try:
            dp = Path(debug_dump_path).expanduser()
            if not dp.is_absolute():
                repo_root = Path(__file__).resolve().parents[2]
                dp = repo_root / dp
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text(json.dumps(debug_report, indent=2), encoding="utf-8")
        except Exception:
            pass

    return out_lines


def _cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Merge multi-page ASCII table continuations in combined.txt (in-place optional).")
    ap.add_argument("--combined", type=str, required=True, help="Path to combined.txt.")
    ap.add_argument("--in-place", action="store_true", help="Write changes back to file if any merges occur.")
    ap.add_argument(
        "--heuristics",
        type=str,
        default="user_inputs/table_merge_heuristics.json",
        help="Heuristics JSON path (relative to repo root by default).",
    )
    args = ap.parse_args(argv)

    combined_path = Path(args.combined).expanduser()
    if not combined_path.exists():
        raise FileNotFoundError(f"combined.txt not found: {combined_path}")

    heur_path = Path(args.heuristics).expanduser()
    if not heur_path.is_absolute():
        # Assume repo root two levels up from this file (EIDAT_App_Files/extraction/..)
        repo_root = Path(__file__).resolve().parents[2]
        heur_path = repo_root / heur_path
    cfg = load_table_merge_heuristics(heur_path)

    raw_lines = combined_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    merged = merge_multipage_tables_in_combined_lines(raw_lines, cfg=cfg)
    changed = merged != raw_lines

    if args.in_place and changed:
        combined_path.write_text("".join(merged), encoding="utf-8")

    print(f"[table_multipage_merger] changed={changed} lines_in={len(raw_lines)} lines_out={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
