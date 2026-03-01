"""
Combined.txt table cell healer (post-processing).

Operates on bordered ASCII tables inside a finished combined.txt.

Pipeline (per table):
  0) Prune: drop fully-empty narrow columns + fully-empty rows
  1) Spacing heuristic: fix common glued tokens (e.g., "1 to4" -> "1 to 4")
  2) Numeric healing: use header roles + above/below reference in numeric columns

Includes a disabled-by-default scaffold for future cross-document learning keyed
by [TABLE_LABEL].
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol


_APP_ROOT = Path(__file__).resolve().parents[1]  # EIDAT_App_Files/
_REPO_ROOT = _APP_ROOT.parent  # repo root (holds user_inputs/)
DEFAULT_TABLE_CELL_HEAL_PATH = _REPO_ROOT / "user_inputs" / "table_cell_heal_heuristics.json"


TABLE_LABEL_MARKER = "[TABLE_LABEL]"

_BORDER_RE = re.compile(r"^\s*\+[-=+]+\+\s*$")
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _newline_for_line(ln: str) -> str:
    if str(ln).endswith("\r\n"):
        return "\r\n"
    if str(ln).endswith("\n"):
        return "\n"
    return "\n"


def _strip_nl(ln: str) -> str:
    return str(ln).rstrip("\r\n")


def _is_border(ln: str) -> bool:
    return bool(_BORDER_RE.match(_strip_nl(ln)))


def _is_pipe_row(ln: str) -> bool:
    return bool(_PIPE_ROW_RE.match(_strip_nl(ln)))


def _plus_positions(border_line: str) -> list[int]:
    s = _strip_nl(border_line)
    return [i for i, ch in enumerate(s) if ch == "+"]


def _norm_text(s: str) -> str:
    s2 = str(s or "").lower().replace("\u00A0", " ").strip()
    s2 = re.sub(r"[\t\r\n]+", " ", s2)
    s2 = re.sub(r"[\"'`“”‘’]", "", s2)
    s2 = re.sub(r"[^a-z0-9]+", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


_DEFAULT_CFG: dict[str, Any] = {
    "version": 1,
    "enabled": True,
    "prune": {
        "enabled": True,
        "drop_empty_rows": True,
        "drop_empty_cols": True,
        "max_empty_col_width": 5,
    },
    "spacing": {
        "enabled": True,
        "default_scope": "all_cells",  # all_cells | numeric_cols_only
    },
    "numeric": {
        "enabled": True,
        "numeric_ratio_min": 0.6,
        "role_synonyms": {
            "value": ["value", "measured", "measured value", "actual", "reading"],
            "min": ["min", "minimum", "lower", "low"],
            "max": ["max", "maximum", "upper", "high"],
            "requirement": ["requirement", "criteria", "spec", "acceptance", "limit", "threshold"],
            "target": ["target", "setpoint", "set point", "goal"],
        },
        "numeric_roles": ["value", "min", "max", "requirement", "target"],
        "apply_to_header": False,
        "unicode_minus_to_dash": True,
        "allow_o_to_zero": True,
        "allow_i_l_to_one": True,
    },
    "history": {
        "enabled": False,
    },
}


def load_table_cell_heal_heuristics(path: Path = DEFAULT_TABLE_CELL_HEAL_PATH) -> dict[str, Any]:
    """
    Best-effort load of combined.txt table cell heal heuristics (schema v1).

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
    # Deep-copy defaults via JSON for simplicity.
    out: dict[str, Any] = json.loads(json.dumps(_DEFAULT_CFG))
    if not isinstance(cfg, dict):
        return out

    def _merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)  # type: ignore[index]
            else:
                dst[k] = v

    _merge(out, cfg)

    # Seed role synonyms from acceptance_heuristics.json if present and caller didn't override.
    try:
        numeric = out.get("numeric")
        if isinstance(numeric, dict):
            rs = numeric.get("role_synonyms")
            if not isinstance(rs, dict) or not rs:
                from extraction.term_value_extractor import DEFAULT_ACCEPTANCE_HEURISTICS_PATH  # type: ignore

                try:
                    acc = json.loads(DEFAULT_ACCEPTANCE_HEURISTICS_PATH.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    acc = {}
                headers = (acc.get("headers") if isinstance(acc, dict) else {}) or {}
                if isinstance(headers, dict):
                    seeded: dict[str, list[str]] = {}
                    for role, syns in headers.items():
                        if not isinstance(role, str) or not isinstance(syns, list):
                            continue
                        seeded[str(role).strip().lower()] = [str(s).strip() for s in syns if str(s).strip()]
                    seeded.setdefault("target", []).extend(["target", "setpoint", "set point", "goal"])
                    numeric["role_synonyms"] = seeded
    except Exception:
        pass

    return out


class TableHealHistory(Protocol):
    def lookup(
        self,
        *,
        table_label: str,
        column_role: str,
        header_text: str,
        raw_text: str,
    ) -> Optional[str]: ...

    def record(
        self,
        *,
        table_label: str,
        column_role: str,
        header_text: str,
        raw_text: str,
        healed_text: str,
    ) -> None: ...


@dataclass(frozen=True)
class _AsciiTableBlock:
    start: int
    end: int
    table_label: str
    lines: list[str]


def _find_next_non_empty(lines: list[str], start: int, end: int) -> int | None:
    i = int(start)
    while i < end:
        if str(lines[i]).strip():
            return i
        i += 1
    return None


def _iter_tables(lines: list[str]) -> Iterable[_AsciiTableBlock]:
    i = 0
    pending_label = ""
    n = len(lines)
    while i < n:
        s = str(lines[i]).strip()
        if s == TABLE_LABEL_MARKER:
            j = _find_next_non_empty(lines, i + 1, n)
            if j is not None:
                pending_label = str(lines[j]).strip()
                i = j + 1
                continue
        if _is_border(lines[i]):
            j = i
            tbl_lines: list[str] = []
            while j < n:
                cur = _strip_nl(lines[j])
                if not cur.strip():
                    break
                cs = cur.lstrip()
                if not (cs.startswith(("+", "|"))):
                    break
                tbl_lines.append(lines[j])
                j += 1
            if tbl_lines:
                yield _AsciiTableBlock(start=i, end=j, table_label=pending_label, lines=tbl_lines)
                pending_label = ""
                i = j
                continue
        i += 1


def _slice_cell_slots_from_row(row_line: str, plus_pos: list[int]) -> tuple[str, list[str]]:
    s = _strip_nl(row_line)
    indent_len = s.find("|")
    if indent_len < 0:
        indent_len = 0
    indent = s[:indent_len]
    if not plus_pos or len(plus_pos) < 2:
        return indent, []

    need_len = max(plus_pos) + 1
    if len(s) < need_len:
        s = s.ljust(need_len)

    slots: list[str] = []
    for a, b in zip(plus_pos[:-1], plus_pos[1:]):
        start = int(a) + 1
        end = int(b)
        slots.append(s[start:end])
    return indent, slots


def _extract_trimmed_cells_from_row(row_line: str, plus_pos: list[int]) -> tuple[str, list[str], list[int]]:
    indent, slots = _slice_cell_slots_from_row(row_line, plus_pos)
    widths = [max(0, int(b - a - 1)) for a, b in zip(plus_pos[:-1], plus_pos[1:])]
    trimmed = [str(s).strip() for s in slots]
    return indent, trimmed, widths


def _render_row(indent: str, widths: list[int], texts: list[str], *, newline: str) -> str:
    out = [str(indent), "|"]
    for w, t in zip(widths, texts):
        w = max(0, int(w))
        txt = str(t or "").strip()
        if w <= 0:
            seg = ""
        elif w == 1:
            seg = (txt[:1]).ljust(1)
        else:
            inner = w - 2
            if inner <= 0:
                seg = (txt[:w]).ljust(w)
            else:
                seg = " " + (txt[:inner]).ljust(inner) + " "
        out.append(seg)
        out.append("|")
    return "".join(out) + newline


_NUM_TOKEN = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NUM_ONLY_RE = re.compile(rf"^\s*{_NUM_TOKEN}\s*%?\s*$")
_NUM_RANGE_RE = re.compile(rf"^\s*({_NUM_TOKEN})\s*(?:to|[-–—])\s*({_NUM_TOKEN})\s*%?\s*$", re.IGNORECASE)


def _is_numeric_like(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return bool(_NUM_ONLY_RE.match(s) or _NUM_RANGE_RE.match(s))


def _split_range(text: str) -> tuple[str, str, str] | None:
    s = str(text or "").strip()
    m = _NUM_RANGE_RE.match(s)
    if not m:
        return None
    a = m.group(1)
    b = m.group(2)
    if re.search(r"\bto\b", s, flags=re.IGNORECASE):
        return a, "to", b
    return a, "-", b


def _normalize_number_token(
    token: str,
    *,
    unicode_minus_to_dash: bool,
    allow_o_to_zero: bool,
    allow_i_l_to_one: bool,
    prefer_decimal_dot: bool,
) -> str:
    s = str(token or "").replace("\u00A0", " ").strip()
    if unicode_minus_to_dash:
        s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    if allow_o_to_zero:
        s = s.replace("O", "0").replace("o", "0")
    if allow_i_l_to_one:
        s = s.replace("I", "1").replace("l", "1")

    if "," in s:
        if "." in s:
            s = s.replace(",", "")
        else:
            if re.match(r"^\d{1,3}(,\d{3})+$", s):
                s = s.replace(",", "")
            elif re.match(r"^\d+,\d{1,2}$", s):
                s = s.replace(",", ".")
            else:
                parts = s.split(",")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    tail = parts[1]
                    if len(tail) in (1, 2):
                        s = parts[0] + "." + tail
                    elif len(tail) == 3:
                        s = parts[0] + tail
                    else:
                        s = parts[0] + (("." if prefer_decimal_dot else ",") + tail)
                else:
                    s = s.replace(",", "")
    return s


def _infer_prefer_decimal_dot(neighbors: list[str]) -> bool:
    for v in neighbors:
        if "." in str(v or ""):
            return True
    return True


def _heal_numeric_text(
    text: str,
    *,
    neighbors: list[str],
    unicode_minus_to_dash: bool,
    allow_o_to_zero: bool,
    allow_i_l_to_one: bool,
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    prefer_dot = _infer_prefer_decimal_dot(neighbors)
    rng = _split_range(raw)
    if rng is not None:
        a, sep, b = rng
        a2 = _normalize_number_token(
            a,
            unicode_minus_to_dash=unicode_minus_to_dash,
            allow_o_to_zero=allow_o_to_zero,
            allow_i_l_to_one=allow_i_l_to_one,
            prefer_decimal_dot=prefer_dot,
        )
        b2 = _normalize_number_token(
            b,
            unicode_minus_to_dash=unicode_minus_to_dash,
            allow_o_to_zero=allow_o_to_zero,
            allow_i_l_to_one=allow_i_l_to_one,
            prefer_decimal_dot=prefer_dot,
        )
        return f"{a2} {sep} {b2}" if sep == "to" else f"{a2}-{b2}"

    suffix = ""
    core = raw
    if core.endswith("%"):
        suffix = "%"
        core = core[:-1].strip()
    core2 = _normalize_number_token(
        core,
        unicode_minus_to_dash=unicode_minus_to_dash,
        allow_o_to_zero=allow_o_to_zero,
        allow_i_l_to_one=allow_i_l_to_one,
        prefer_decimal_dot=prefer_dot,
    )
    return (core2 + suffix).strip()


_GLUED_TO_RE = re.compile(r"(\d)\s*to\s*(\d)", re.IGNORECASE)
_DIGIT_LETTER_RE = re.compile(r"(\d)([a-zA-Z])")
_LETTER_DIGIT_RE = re.compile(r"([a-zA-Z])(\d)")
_OP_GLUED_RE = re.compile(r"([<>]=?)\s*(\d)")


def _apply_spacing(text: str) -> str:
    s = str(text or "")
    if not s.strip():
        return s.strip()

    protected: list[str] = []

    def _protect(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"__EIDAT_SCI_{len(protected) - 1}__"

    sci_re = re.compile(r"\b\d+(?:\.\d+)?[eE][-+]?\d+\b")
    s2 = sci_re.sub(_protect, s)

    s2 = s2.replace("\u00A0", " ")
    s2 = _GLUED_TO_RE.sub(r"\1 to \2", s2)
    s2 = _OP_GLUED_RE.sub(r"\1 \2", s2)
    s2 = _DIGIT_LETTER_RE.sub(r"\1 \2", s2)
    s2 = _LETTER_DIGIT_RE.sub(r"\1 \2", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()

    for idx, token in enumerate(protected):
        s2 = s2.replace(f"__EIDAT_SCI_{idx}__", token)
    return s2


def _header_roles_for_columns(header_cells: list[str], role_synonyms: dict[str, list[str]]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for col_idx, raw in enumerate(header_cells):
        hn = _norm_text(raw)
        if not hn:
            continue
        best_role = ""
        best_len = 0
        for role, syns in (role_synonyms or {}).items():
            if not isinstance(role, str) or not isinstance(syns, list):
                continue
            for syn in syns:
                sn = _norm_text(syn)
                if not sn:
                    continue
                if sn == hn or (sn in hn and len(sn) >= 3):
                    if len(sn) > best_len:
                        best_role = str(role).strip().lower()
                        best_len = len(sn)
        if best_role:
            roles[col_idx] = best_role
    return roles


def _numeric_columns(
    rows: list[list[str]],
    *,
    header_roles: dict[int, str],
    numeric_roles: set[str],
    numeric_ratio_min: float,
) -> set[int]:
    n_cols = max((len(r) for r in rows), default=0)
    out: set[int] = set()
    for c in range(n_cols):
        role = header_roles.get(c, "")
        if role and role in numeric_roles:
            out.add(c)

    for c in range(n_cols):
        if c in out:
            continue
        filled = 0
        numeric_like = 0
        for r in rows[1:]:
            if c >= len(r):
                continue
            val = str(r[c] or "").strip()
            if not val:
                continue
            filled += 1
            if _is_numeric_like(val):
                numeric_like += 1
        if filled >= 2 and (numeric_like / max(1, filled)) >= float(numeric_ratio_min or 0.0):
            out.add(c)
    return out


def _prune_empty_narrow_columns(
    table_lines: list[str],
    *,
    max_empty_col_width: int,
) -> tuple[list[str], dict[str, int]]:
    stats = {"cols_dropped": 0}
    border0 = next((ln for ln in table_lines if _is_border(ln)), "")
    plus_pos = _plus_positions(border0) if border0 else []
    if len(plus_pos) < 2:
        return table_lines, stats

    widths = [max(0, int(b - a - 1)) for a, b in zip(plus_pos[:-1], plus_pos[1:])]
    pipe_rows = [ln for ln in table_lines if _is_pipe_row(ln)]
    if not pipe_rows:
        return table_lines, stats

    all_empty = [True for _ in widths]
    for ln in pipe_rows:
        _, cells, _ = _extract_trimmed_cells_from_row(ln, plus_pos)
        for c in range(min(len(all_empty), len(cells))):
            if str(cells[c]).strip():
                all_empty[c] = False

    drop_cols = [
        idx for idx, empty in enumerate(all_empty)
        if empty and int(widths[idx]) <= int(max_empty_col_width)
    ]
    if not drop_cols or len(drop_cols) >= len(widths):
        return table_lines, stats

    out_lines: list[str] = []
    for ln in table_lines:
        nl = _newline_for_line(ln)
        s = _strip_nl(ln)
        need_len = max(plus_pos) + 1
        if len(s) < need_len:
            s = s.ljust(need_len)
        for col in sorted(drop_cols, reverse=True):
            a = plus_pos[col]
            b = plus_pos[col + 1]
            s = s[: a + 1] + s[b + 1 :]
        out_lines.append(s + nl)

    stats["cols_dropped"] = int(len(drop_cols))
    return out_lines, stats


def _prune_empty_rows(table_lines: list[str]) -> tuple[list[str], dict[str, int]]:
    stats = {"rows_dropped": 0}
    border0 = next((ln for ln in table_lines if _is_border(ln)), "")
    plus_pos = _plus_positions(border0) if border0 else []
    if len(plus_pos) < 2:
        return table_lines, stats

    drop: set[int] = set()
    pipe_indices = [idx for idx, ln in enumerate(table_lines) if _is_pipe_row(ln)]
    if not pipe_indices:
        return table_lines, stats

    for idx in pipe_indices:
        _, cells, _ = _extract_trimmed_cells_from_row(table_lines[idx], plus_pos)
        if cells and all(not str(c).strip() for c in cells):
            drop.add(idx)
            if idx + 1 < len(table_lines) and _is_border(table_lines[idx + 1]):
                drop.add(idx + 1)

    remaining_pipe = sum(1 for idx in pipe_indices if idx not in drop)
    if remaining_pipe <= 0 or not drop:
        return table_lines, stats

    out = [ln for k, ln in enumerate(table_lines) if k not in drop]
    stats["rows_dropped"] = int(len([idx for idx in pipe_indices if idx in drop]))
    return out, stats


def _heal_table_lines(
    table_lines: list[str],
    *,
    cfg: dict[str, Any],
    table_label: str,
    history: Optional[TableHealHistory],
) -> tuple[list[str], dict[str, int]]:
    stats: dict[str, int] = {
        "tables_seen": 1,
        "cols_dropped": 0,
        "rows_dropped": 0,
        "cells_spaced": 0,
        "cells_numeric_healed": 0,
    }
    prune_cfg = cfg.get("prune") if isinstance(cfg.get("prune"), dict) else {}
    spacing_cfg = cfg.get("spacing") if isinstance(cfg.get("spacing"), dict) else {}
    numeric_cfg = cfg.get("numeric") if isinstance(cfg.get("numeric"), dict) else {}

    if bool(prune_cfg.get("enabled", True)):
        if bool(prune_cfg.get("drop_empty_cols", True)):
            try:
                maxw = int(prune_cfg.get("max_empty_col_width", 5))
            except Exception:
                maxw = 5
            table_lines, st = _prune_empty_narrow_columns(table_lines, max_empty_col_width=maxw)
            stats["cols_dropped"] += int(st.get("cols_dropped", 0))
        if bool(prune_cfg.get("drop_empty_rows", True)):
            table_lines, st = _prune_empty_rows(table_lines)
            stats["rows_dropped"] += int(st.get("rows_dropped", 0))

    border0 = next((ln for ln in table_lines if _is_border(ln)), "")
    plus_pos = _plus_positions(border0) if border0 else []
    if len(plus_pos) < 2:
        return table_lines, stats
    widths = [max(0, int(b - a - 1)) for a, b in zip(plus_pos[:-1], plus_pos[1:])]

    pipe_indices = [idx for idx, ln in enumerate(table_lines) if _is_pipe_row(ln)]
    if not pipe_indices:
        return table_lines, stats

    indent0 = ""
    matrix: list[list[str]] = []
    matrix_orig: list[list[str]] = []
    for idx in pipe_indices:
        indent, cells, _ = _extract_trimmed_cells_from_row(table_lines[idx], plus_pos)
        if not indent0:
            indent0 = indent
        fixed = [cells[c] if c < len(cells) else "" for c in range(len(widths))]
        matrix.append(list(fixed))
        matrix_orig.append(list(fixed))

    spacing_enabled = bool(spacing_cfg.get("enabled", True))
    scope = str(spacing_cfg.get("default_scope", "all_cells") or "all_cells").strip().lower()
    spacing_all = spacing_enabled and scope in ("all_cells", "all", "everything")

    numeric_enabled = bool(numeric_cfg.get("enabled", True))
    try:
        numeric_ratio_min = float(numeric_cfg.get("numeric_ratio_min", 0.6))
    except Exception:
        numeric_ratio_min = 0.6
    role_synonyms = numeric_cfg.get("role_synonyms") if isinstance(numeric_cfg.get("role_synonyms"), dict) else {}
    numeric_roles = set(
        str(x).strip().lower()
        for x in (numeric_cfg.get("numeric_roles") or [])
        if str(x).strip()
    ) or {"value", "min", "max", "requirement", "target"}

    header_cells = matrix[0] if matrix else []
    header_roles = _header_roles_for_columns(header_cells, role_synonyms) if header_cells else {}
    numeric_cols = _numeric_columns(matrix, header_roles=header_roles, numeric_roles=numeric_roles, numeric_ratio_min=numeric_ratio_min) if numeric_enabled else set()

    def _neighbors_for(r_idx: int, c_idx: int) -> list[str]:
        vals: list[tuple[int, str]] = []
        for rr in range(1, len(matrix)):
            if rr == r_idx:
                continue
            v = str(matrix[rr][c_idx] or "").strip()
            if not v:
                continue
            if _is_numeric_like(v):
                vals.append((rr, v))
        above = [v for rr, v in vals if rr < r_idx]
        below = [v for rr, v in vals if rr > r_idx]
        out: list[str] = []
        if above:
            out.append(above[-1])
        if below:
            out.append(below[0])
        return out

    unicode_minus_to_dash = bool(numeric_cfg.get("unicode_minus_to_dash", True))
    allow_o_to_zero = bool(numeric_cfg.get("allow_o_to_zero", True))
    allow_i_l_to_one = bool(numeric_cfg.get("allow_i_l_to_one", True))

    for r in range(len(matrix)):
        for c in range(len(widths)):
            orig = str(matrix[r][c] or "").strip()
            if not orig:
                continue

            do_spacing = spacing_enabled and (spacing_all or (scope in ("numeric_cols_only", "numeric") and (c in numeric_cols)))
            spaced = _apply_spacing(orig) if do_spacing else orig
            if spaced != orig:
                matrix[r][c] = spaced
                stats["cells_spaced"] += 1

            if not numeric_enabled or c not in numeric_cols or r == 0:
                continue
            cur = str(matrix[r][c] or "").strip()
            neigh = _neighbors_for(r, c)
            healed = _heal_numeric_text(
                cur,
                neighbors=neigh,
                unicode_minus_to_dash=unicode_minus_to_dash,
                allow_o_to_zero=allow_o_to_zero,
                allow_i_l_to_one=allow_i_l_to_one,
            ).strip()
            if healed and healed != cur and _is_numeric_like(healed):
                matrix[r][c] = healed
                stats["cells_numeric_healed"] += 1
                if history is not None and bool(cfg.get("history", {}).get("enabled", False)):  # type: ignore[union-attr]
                    role = header_roles.get(c, "")
                    try:
                        history.record(
                            table_label=table_label,
                            column_role=str(role),
                            header_text=str(header_cells[c] if c < len(header_cells) else ""),
                            raw_text=cur,
                            healed_text=healed,
                        )
                    except Exception:
                        pass

    out_lines = list(table_lines)
    for row_idx, tbl_idx in enumerate(pipe_indices):
        nl = _newline_for_line(out_lines[tbl_idx])
        original_cells = matrix_orig[row_idx]
        new_cells = matrix[row_idx]
        safe_cells: list[str] = []
        for w, orig_txt, new_txt in zip(widths, original_cells, new_cells):
            w = int(w)
            orig_t = str(orig_txt or "").strip()
            new_t = str(new_txt or "").strip()
            inner = max(0, w - 2) if w >= 2 else w
            if inner > 0 and len(new_t) > inner and len(orig_t) <= inner:
                safe_cells.append(orig_t)
            else:
                safe_cells.append(new_t)
        out_lines[tbl_idx] = _render_row(indent0, widths, safe_cells, newline=nl)

    return out_lines, stats


def heal_combined_tables_in_lines(
    lines: list[str],
    cfg: dict[str, Any] | None = None,
    *,
    history: Optional[TableHealHistory] = None,
) -> tuple[list[str], dict[str, int]]:
    cfg2 = _cfg(cfg)
    stats: dict[str, int] = {
        "tables_seen": 0,
        "cols_dropped": 0,
        "rows_dropped": 0,
        "cells_spaced": 0,
        "cells_numeric_healed": 0,
    }
    if not bool(cfg2.get("enabled", True)):
        return lines, stats

    out = list(lines)
    # Replace from the end so index ranges remain valid even if we drop rows.
    for block in reversed(list(_iter_tables(lines))):
        healed, st = _heal_table_lines(block.lines, cfg=cfg2, table_label=block.table_label, history=history)
        stats["tables_seen"] += int(st.get("tables_seen", 0))
        stats["cols_dropped"] += int(st.get("cols_dropped", 0))
        stats["rows_dropped"] += int(st.get("rows_dropped", 0))
        stats["cells_spaced"] += int(st.get("cells_spaced", 0))
        stats["cells_numeric_healed"] += int(st.get("cells_numeric_healed", 0))
        if healed != block.lines:
            out[block.start : block.end] = healed
    return out, stats


def heal_combined_txt_file_inplace(
    combined_txt_path: Path,
    cfg: dict[str, Any] | None = None,
    *,
    history: Optional[TableHealHistory] = None,
) -> dict[str, int]:
    """
    Heal ASCII tables inside combined.txt in-place.

    Returns stats dict. If file is missing/unreadable, returns {}.
    """
    try:
        p = Path(combined_txt_path)
        if not p.exists():
            return {}
        raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)
        healed, stats = heal_combined_tables_in_lines(raw_lines, cfg=cfg, history=history)
        if healed != raw_lines:
            p.write_text("".join(healed), encoding="utf-8")
        return stats
    except Exception:
        return {}
