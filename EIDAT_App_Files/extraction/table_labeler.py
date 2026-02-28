"""
Rules-driven table labeling for combined.txt.

Scans a combined.txt (ASCII bordered tables), identifies tables matching user-defined
rules, and inserts a stable label block immediately before each matching table:

    [TABLE_LABEL]
    Acceptance Test Data (2)

    +----+----+
    | .. | .. |
    +----+----+
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_TABLE_BORDER_RE = re.compile(r"^\s*\+.*\+\s*$")


def load_table_label_rules(path: Path) -> Dict[str, Any]:
    """
    Best-effort load of table label rules JSON (schema v1).

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
        rules = data.get("rules")
        if rules is None:
            data["rules"] = []
        if not isinstance(data.get("rules"), list):
            return {}
        return data
    except Exception:
        return {}


def _norm_text(s: str) -> str:
    s2 = str(s or "").lower()
    s2 = s2.replace("\u00A0", " ")
    s2 = re.sub(r"[\t\r\n]+", " ", s2)
    s2 = re.sub(r"[\"'`“”‘’]", "", s2)
    s2 = re.sub(r"[^\w\s./+-]+", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _table_rows_from_lines(table_lines: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for ln in table_lines:
        s = str(ln).rstrip("\n")
        if not s.strip().startswith("|"):
            continue
        parts = [p.strip() for p in s.strip().strip("|").split("|")]
        # Drop empty trailing columns
        while parts and not parts[-1]:
            parts.pop()
        if parts:
            rows.append(parts)
    return rows


def _iter_tables(lines: List[str]) -> List[Tuple[int, int, List[str], List[List[str]]]]:
    """
    Return a list of tables:
      (start_index, end_index_exclusive, table_lines, rows)
    """
    out: List[Tuple[int, int, List[str], List[List[str]]]] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        s = str(ln).strip()
        # Some ASCII tables use '=' borders (or mixed '='/'-'); accept either as a table start.
        if s.startswith("+") and s.endswith("+") and (("-" in s) or ("=" in s)):
            tbl_lines: List[str] = []
            j = i
            while j < n:
                cur = str(lines[j]).rstrip("\n")
                if not cur.strip():
                    break
                cs = cur.strip()
                if not (cs.startswith(("+", "|"))):
                    break
                tbl_lines.append(cur)
                j += 1
            rows = _table_rows_from_lines(tbl_lines)
            if rows:
                out.append((i, j, tbl_lines, rows))
            i = j + 1
            continue
        i += 1
    return out


def _remove_existing_label_blocks(lines: List[str], marker: str) -> List[str]:
    want = f"[{str(marker or 'TABLE_LABEL').strip()}]"
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if str(lines[i]).strip() == want:
            i += 1
            # Skip blank lines
            while i < n and not str(lines[i]).strip():
                i += 1
            # Skip label line
            if i < n:
                i += 1
            # Skip trailing blank lines
            while i < n and not str(lines[i]).strip():
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def _rule_matches(rule: Dict[str, Any], *, rows: List[List[str]]) -> bool:
    try:
        min_rows = int(rule.get("min_rows") or 0)
    except Exception:
        min_rows = 0
    try:
        min_cols = int(rule.get("min_cols") or 0)
    except Exception:
        min_cols = 0
    try:
        max_rows = int(rule.get("max_rows") or 0)
    except Exception:
        max_rows = 0
    try:
        max_cols = int(rule.get("max_cols") or 0)
    except Exception:
        max_cols = 0

    row_count = len(rows)
    col_count = max((len(r) for r in rows), default=0)
    if min_rows and row_count < min_rows:
        return False
    if min_cols and col_count < min_cols:
        return False
    if max_rows and row_count > max_rows:
        return False
    if max_cols and col_count > max_cols:
        return False

    scope = str(rule.get("match_scope") or "any_cell").strip().lower()
    cells: List[str] = []
    if scope == "header_row":
        if rows:
            cells = [str(c or "") for c in rows[0]]
    elif scope == "first_col":
        cells = [str(r[0] or "") for r in rows if r]
    else:
        for r in rows:
            cells.extend(str(c or "") for c in r)

    hay = _norm_text(" ".join(cells))
    if not hay:
        return False

    def _phrases(key: str) -> List[str]:
        raw = rule.get(key) or []
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for p in raw:
            s = _norm_text(str(p or ""))
            if s:
                out.append(s)
        return out

    must_all = _phrases("must_contain_all")
    must_any = _phrases("must_contain_any")
    must_not = _phrases("must_not_contain")

    for ph in must_all:
        if ph not in hay:
            return False
    if must_any:
        if not any(ph in hay for ph in must_any):
            return False
    for ph in must_not:
        if ph in hay:
            return False
    return True


def _rule_specificity(rule: Dict[str, Any]) -> int:
    score = 0

    def _count_phrases(key: str) -> int:
        raw = rule.get(key) or []
        if not isinstance(raw, list):
            return 0
        return sum(1 for p in raw if _norm_text(str(p or "")))

    score += _count_phrases("must_contain_all")
    score += _count_phrases("must_contain_any")
    score += _count_phrases("must_not_contain")

    for k in ("min_rows", "min_cols", "max_rows", "max_cols"):
        try:
            if int(rule.get(k) or 0):
                score += 1
        except Exception:
            continue

    scope = str(rule.get("match_scope") or "any_cell").strip().lower()
    if scope and scope != "any_cell":
        score += 1

    return int(score)


def _pick_rule(
    rules: List[Dict[str, Any]],
    *,
    rows: List[List[str]],
    tie_breaker: str = "priority_then_order",
) -> Optional[Dict[str, Any]]:
    matches: List[Tuple[int, int, Dict[str, Any]]] = []
    tb = str(tie_breaker or "priority_then_order").strip().lower()
    use_specificity = tb in (
        "priority_then_specificity",
        "specificity",
        "specific",
        "most_specific",
    )
    for idx, rule in enumerate(rules or []):
        if not isinstance(rule, dict):
            continue
        label = str(rule.get("label") or "").strip()
        if not label:
            continue
        if not _rule_matches(rule, rows=rows):
            continue
        try:
            priority = int(rule.get("priority") or 0)
        except Exception:
            priority = 0
        if use_specificity:
            # Higher specificity wins on priority ties; rule order is last-resort tie-breaker.
            spec = _rule_specificity(rule)
            matches.append((priority, spec, -idx, rule))  # type: ignore[list-item]
        else:
            matches.append((priority, -idx, rule))
    if not matches:
        return None
    matches.sort(reverse=True)
    top = matches[0]
    return top[-1]  # type: ignore[return-value]


def label_combined_lines(lines: List[str], rules_cfg: Dict[str, Any]) -> List[str]:
    """
    Return new combined.txt lines with [TABLE_LABEL] blocks inserted (idempotent).
    """
    if not isinstance(rules_cfg, dict) or int(rules_cfg.get("version") or 0) != 1:
        return list(lines)

    marker = str(rules_cfg.get("marker") or "TABLE_LABEL").strip() or "TABLE_LABEL"
    marker_line = f"[{marker}]"
    append_index_after_first = bool(rules_cfg.get("append_index_after_first", True))
    tie_breaker = str(rules_cfg.get("tie_breaker") or "priority_then_order").strip() or "priority_then_order"
    rules = rules_cfg.get("rules") if isinstance(rules_cfg.get("rules"), list) else []

    cleaned = _remove_existing_label_blocks(list(lines), marker)
    tables = _iter_tables(cleaned)
    if not tables:
        return cleaned

    label_counts: Dict[str, int] = {}
    start_to_label: Dict[int, str] = {}
    for start_idx, _end_idx, _tbl_lines, rows in tables:
        rule = _pick_rule(rules, rows=rows, tie_breaker=tie_breaker)
        if not rule:
            continue
        base = str(rule.get("label") or "").strip()
        if not base:
            continue
        label_counts[base] = label_counts.get(base, 0) + 1
        n = label_counts[base]
        if append_index_after_first:
            full = base if n == 1 else f"{base} ({n})"
        else:
            full = f"{base} ({n})"
        start_to_label[start_idx] = full

    if not start_to_label:
        return cleaned

    out: List[str] = []
    for i, ln in enumerate(cleaned):
        if i in start_to_label:
            out.append(f"{marker_line}\n")
            out.append(f"{start_to_label[i]}\n")
            out.append("\n")
        out.append(ln if ln.endswith("\n") else f"{ln}\n")

    # Preserve trailing newline behavior (combined.txt usually ends with newline)
    while out and out[-1] == "\n" and (len(out) >= 2 and out[-2] == "\n"):
        break
    return out


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Label tables in combined.txt using JSON rules.")
    parser.add_argument("--combined", type=str, required=True, help="Path to combined.txt to label (in-place).")
    parser.add_argument("--rules", type=str, required=True, help="Path to user_inputs/table_label_rules.json")
    args = parser.parse_args(argv)

    combined_path = Path(args.combined).expanduser()
    rules_path = Path(args.rules).expanduser()
    if not combined_path.exists():
        raise FileNotFoundError(f"combined.txt not found: {combined_path}")
    rules_cfg = load_table_label_rules(rules_path)
    if not rules_cfg:
        print(f"[SKIP] rules not found/invalid: {rules_path}")
        return 0

    lines = combined_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    out_lines = label_combined_lines(lines, rules_cfg)
    combined_path.write_text("".join(out_lines), encoding="utf-8")
    print(f"[DONE] Labeled tables in {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
