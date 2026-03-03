from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    # Common execution mode: `python EIDAT_App_Files/Application/eidat_manager.py ...`
    # where `Application/` is not a package.
    from eidat_manager_db import support_paths  # type: ignore
except Exception:  # pragma: no cover
    # Package mode fallback.
    from .eidat_manager_db import support_paths  # type: ignore


EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}

_NUM_RE = re.compile(r"^[\s\+\-]?\d[\d,\s]*(\.\d+)?([eE][\+\-]?\d+)?\s*$")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_DATA_ENV_PATH = _REPO_ROOT / "user_inputs" / "test_data.env"
_EXCEL_TREND_CONFIG_PATH = _REPO_ROOT / "user_inputs" / "excel_trend_config.json"


def _now_ns() -> int:
    try:
        return int(time.time_ns())
    except Exception:
        return int(time.time() * 1e9)


def _stderr(line: str) -> None:
    s = str(line or "").rstrip("\n")
    if not s:
        return
    print(s, file=sys.stderr, flush=True)


def export_sqlite_text_mirror(
    sqlite_path: Path,
    *,
    out_txt: Path | None = None,
    max_rows_per_table: int = 80,
    max_cell_chars: int = 240,
    max_total_chars: int = 2_000_000,
) -> Path:
    """
    Write a human-readable text mirror of a workbook SQLite DB for debugging.
    Intended for TD artifacts folders so we can inspect row/column issues without opening SQLite.
    """
    db = Path(sqlite_path).expanduser()
    if not db.exists() or not db.is_file():
        raise FileNotFoundError(str(db))

    out = Path(out_txt) if out_txt is not None else db.with_suffix(db.suffix + ".txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _quote_ident(name: str) -> str:
        return "[" + str(name).replace("]", "]]") + "]"

    def _safe_cell(v: object) -> str:
        if v is None:
            return ""
        s = str(v)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        if len(s) > int(max_cell_chars):
            s = s[: int(max_cell_chars) - 3] + "..."
        return s

    chunks: list[str] = []
    total = 0

    def _append(s: str) -> None:
        nonlocal total
        if total >= int(max_total_chars):
            return
        if total + len(s) > int(max_total_chars):
            s = s[: max(0, int(max_total_chars) - total)]
        chunks.append(s)
        total += len(s)

    _append(f"sqlite_path: {db}\n")
    try:
        st = db.stat()
        _append(f"size_bytes: {int(st.st_size)}\n")
        _append(f"mtime_ns: {int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)))}\n")
    except Exception:
        pass
    _append("\n")

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            schema_rows = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
        except Exception:
            schema_rows = []

        _append("== schema ==\n")
        for r in schema_rows:
            try:
                typ = str(r["type"] or "")
                name = str(r["name"] or "")
                sql = str(r["sql"] or "").strip()
            except Exception:
                continue
            if not name:
                continue
            _append(f"-- {typ}: {name}\n")
            if sql:
                _append(sql + "\n")
            _append("\n")

        tables: list[str] = []
        for r in schema_rows:
            try:
                if str(r["type"] or "") == "table":
                    tables.append(str(r["name"] or ""))
            except Exception:
                continue
        _append("== table_previews ==\n")
        for t in tables:
            if not t or t.startswith("sqlite_"):
                continue
            if total >= int(max_total_chars):
                break

            q_t = _quote_ident(t)
            try:
                info = conn.execute(f"PRAGMA table_info({q_t})").fetchall()
                cols = [str(rr[1] or "") for rr in info if rr and rr[1]]
            except Exception:
                cols = []

            try:
                n = int(conn.execute(f"SELECT COUNT(*) FROM {q_t}").fetchone()[0] or 0)
            except Exception:
                n = -1

            _append(f"-- table: {t} rows={n}\n")
            if cols:
                _append("columns: " + ", ".join(cols) + "\n")

            if cols:
                try:
                    rows = conn.execute(
                        f"SELECT * FROM {q_t} ORDER BY rowid ASC LIMIT {int(max_rows_per_table)}"
                    ).fetchall()
                except Exception:
                    rows = []
                if rows:
                    _append("\t".join(cols) + "\n")
                    for rr in rows:
                        try:
                            _append("\t".join(_safe_cell(rr[c]) for c in cols) + "\n")
                        except Exception:
                            continue
            _append("\n")

    out.write_text("".join(chunks), encoding="utf-8", errors="ignore")
    return out


def export_sqlite_excel_mirror(
    sqlite_path: Path,
    *,
    out_xlsx: Path | None = None,
    max_rows_per_table: int | None = None,
    max_cell_chars: int = 32_000,
) -> Path:
    """
    Export a SQLite DB to an Excel workbook for one-to-one inspection.

    - Adds a `__schema` sheet with sqlite_master SQL.
    - Adds one sheet per table (name truncated/deduped to Excel's 31-char limit).
    """
    try:
        import openpyxl  # type: ignore
        from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openpyxl is required to export SQLite mirrors to .xlsx") from exc

    db = Path(sqlite_path).expanduser()
    if not db.exists() or not db.is_file():
        raise FileNotFoundError(str(db))

    out = Path(out_xlsx) if out_xlsx is not None else db.with_suffix(db.suffix + ".xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def _safe_sheet_name(name: str, used: set[str]) -> str:
        base = (name or "sheet").strip() or "sheet"
        # Excel sheet name constraints: max 31 chars, cannot contain : \ / ? * [ ]
        base = re.sub(r"[:\\\\/\\?\\*\\[\\]]+", "_", base)
        base = base[:31].rstrip() or "sheet"
        cand = base
        i = 2
        while cand.lower() in {u.lower() for u in used}:
            suffix = f"_{i}"
            cand = (base[: max(0, 31 - len(suffix))] + suffix).rstrip() or f"sheet{i}"
            i += 1
        used.add(cand)
        return cand

    def _safe_cell(v: object) -> object:
        if v is None:
            return None
        if isinstance(v, (int, float, bool)):
            return v
        s = str(v)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        if len(s) > int(max_cell_chars):
            s = s[: int(max_cell_chars) - 3] + "..."
        # Avoid writing raw bytes reprs as huge strings
        return s

    with sqlite3.connect(str(db)) as conn:
        cur = conn.cursor()
        schema_rows = cur.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        table_names = [r[1] for r in schema_rows if str(r[0] or "") == "table" and str(r[1] or "").strip()]

        wb = openpyxl.Workbook()
        # Remove default sheet
        try:
            wb.remove(wb.active)
        except Exception:
            pass

        used_names: set[str] = set()

        # Schema sheet
        ws_schema = wb.create_sheet(title=_safe_sheet_name("__schema", used_names))
        ws_schema.append(["type", "name", "tbl_name", "sql"])
        for c in range(1, 5):
            cell = ws_schema.cell(row=1, column=c)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
        ws_schema.freeze_panes = "A2"
        for r in schema_rows:
            ws_schema.append([_safe_cell(r[0]), _safe_cell(r[1]), _safe_cell(r[2]), _safe_cell(r[3])])

        # Table sheets
        excel_max_rows = 1_048_576  # includes header row
        for t in table_names:
            if not t:
                continue
            cols = [c[1] for c in cur.execute(f"PRAGMA table_info([{t}])").fetchall()]
            ws = wb.create_sheet(title=_safe_sheet_name(str(t), used_names))
            ws.append([_safe_cell(c) for c in cols])
            for col_idx in range(1, len(cols) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_align

            limit = None
            if max_rows_per_table is not None:
                limit = int(max_rows_per_table)
            # Header occupies row 1
            hard_cap = excel_max_rows - 1
            if limit is None or limit > hard_cap:
                limit = hard_cap

            try:
                rows = cur.execute(f"SELECT * FROM [{t}] LIMIT {int(limit)}").fetchall()
            except Exception:
                rows = []

            for row in rows:
                ws.append([_safe_cell(v) for v in row])

            ws.freeze_panes = "A2"
            # Approx auto-fit widths (capped)
            try:
                for col_idx, col_name in enumerate(cols, start=1):
                    max_len = min(60, len(str(col_name or "")))
                    for r_idx in range(2, min(ws.max_row, 200) + 1):
                        v = ws.cell(row=r_idx, column=col_idx).value
                        if v is None:
                            continue
                        max_len = max(max_len, min(60, len(str(v))))
                    ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 3
            except Exception:
                pass

        wb.save(str(out))
        try:
            wb.close()
        except Exception:
            pass

    return out


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _try_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            fv = float(v)
            if fv == fv:  # NaN guard
                return fv
        except Exception:
            return None
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Fast reject before float() (handles commas and scientific notation).
        if not _NUM_RE.match(s):
            return None
        s = s.replace(",", "").replace(" ", "")
        try:
            fv = float(s)
            if fv == fv:
                return fv
        except Exception:
            return None
    return None


def _is_header_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return False
    s = str(v).strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    # Must contain at least one letter; avoids picking up numeric-ish labels.
    if not any(ch.isalpha() for ch in s):
        return False
    # Avoid rows like "1.23" or "2025-01-01".
    if _try_float(s) is not None:
        return False
    return True


def _safe_ident(name: str, *, prefix: str = "col") -> str:
    raw = str(name or "").strip()
    if not raw:
        return prefix
    # Keep ASCII-ish identifier (SQLite allows more, but we want portable column names).
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:80]


def _dedupe_idents(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for n in names:
        base = _safe_ident(n, prefix="c")
        k = base.lower()
        if k not in seen:
            seen[k] = 1
            out.append(base)
            continue
        seen[k] += 1
        out.append(f"{base}_{seen[k]}")
    return out


def _safe_table_name(sheet_name: str) -> str:
    base = _safe_ident(sheet_name, prefix="sheet")
    return f"sheet__{base}"


@dataclass(frozen=True)
class DetectedSheet:
    sheet_name: str
    table_name: str
    header_row: int
    excel_col_indices: list[int]
    headers: list[str]
    mapped_headers: list[str]
    col_idents: list[str]


def _parse_env_file(path: Path) -> dict[str, str]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        out[k] = v.strip()
    return out


def _load_test_data_env() -> dict[str, str]:
    env = _parse_env_file(_TEST_DATA_ENV_PATH)
    # Environment variables win
    for k in list(env.keys()):
        if k in os.environ:
            env[k] = str(os.environ.get(k) or "").strip()
    for k, v in os.environ.items():
        if k.startswith("EIDAT_TEST_DATA_") and k not in env:
            env[k] = str(v or "").strip()
    return env


def _truthy(s: Any) -> bool:
    v = str(s or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _float(s: Any, default: float) -> float:
    try:
        return float(str(s or "").strip())
    except Exception:
        return float(default)


def _normalize_header(s: Any) -> str:
    v = str(s or "").strip().lower()
    v = re.sub(r"\(.*?\)", " ", v)
    v = re.sub(r"[^0-9a-z]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _load_trend_column_names() -> list[str]:
    try:
        if not _EXCEL_TREND_CONFIG_PATH.exists():
            return []
        cfg = json.loads(_EXCEL_TREND_CONFIG_PATH.read_text(encoding="utf-8"))
        cols = cfg.get("columns") if isinstance(cfg, dict) else None
        if not isinstance(cols, list):
            return []
        out: list[str] = []
        for c in cols:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            if name:
                out.append(name)
        return out
    except Exception:
        return []


def _fuzzy_score(target: str, candidate: str) -> float:
    t = _normalize_header(target)
    c = _normalize_header(candidate)
    if not t or not c:
        return 0.0
    if t == c:
        return 1.0
    if t in c or c in t:
        return 0.92
    return float(difflib.SequenceMatcher(a=t, b=c).ratio())


def _map_headers_to_config(headers: list[str], config_cols: list[str], *, min_ratio: float) -> list[str]:
    """
    Return mapped header labels (same length as headers).
    Each config column is used at most once per sheet.
    """
    if not headers or not config_cols:
        return list(headers)

    pairs: list[tuple[float, int, str]] = []
    for hi, h in enumerate(headers):
        for c in config_cols:
            pairs.append((_fuzzy_score(c, h), hi, c))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2].lower()))

    assigned_header: dict[int, str] = {}
    used_config: set[str] = set()
    for score, hi, canon in pairs:
        if score < float(min_ratio):
            break
        if hi in assigned_header:
            continue
        if canon in used_config:
            continue
        assigned_header[hi] = canon
        used_config.add(canon)

    return [assigned_header.get(i, h) for i, h in enumerate(headers)]


def _detect_header_row(
    ws,
    *,
    max_scan_rows: int = 200,
    max_cols: int = 200,
    lookahead_rows: int = 60,
    min_numeric_count: int = 8,
    min_numeric_ratio: float = 0.60,
    min_data_cols: int = 1,
) -> tuple[int | None, list[tuple[int, str]]]:
    """
    Return (header_row_index_1_based, [(excel_col_index_1_based, header_text), ...]).
    Picks the row that maximizes "header cells with numeric-heavy columns beneath".
    """
    try:
        max_row = int(getattr(ws, "max_row", 0) or 0)
        max_col = int(getattr(ws, "max_column", 0) or 0)
    except Exception:
        max_row = 0
        max_col = 0
    if max_row <= 0 or max_col <= 0:
        return None, []

    scan_rows = max(1, min(int(max_scan_rows), max_row))
    scan_cols = max(1, min(int(max_cols), max_col))

    best_row: int | None = None
    best_cols: list[tuple[int, str]] = []
    best_score: float = -1.0

    for r in range(1, scan_rows + 1):
        try:
            header_tuple = next(
                ws.iter_rows(min_row=r, max_row=r, min_col=1, max_col=scan_cols, values_only=True)
            )
        except Exception:
            continue

        header_vals = list(header_tuple)
        if not any(_is_header_value(v) for v in header_vals):
            continue

        la_start = r + 1
        la_end = min(max_row, r + int(lookahead_rows))
        if la_start > la_end:
            continue

        try:
            lookahead = list(
                ws.iter_rows(min_row=la_start, max_row=la_end, min_col=1, max_col=scan_cols, values_only=True)
            )
        except Exception:
            lookahead = []
        if not lookahead:
            continue

        cols: list[tuple[int, str]] = []
        score = 0.0
        for ci, hv in enumerate(header_vals, start=1):
            if not _is_header_value(hv):
                continue
            filled = 0
            numeric = 0
            for row_vals in lookahead:
                try:
                    v = row_vals[ci - 1]
                except Exception:
                    v = None
                if _is_blank(v):
                    continue
                filled += 1
                if _try_float(v) is not None:
                    numeric += 1
            if numeric < int(min_numeric_count):
                continue
            ratio = float(numeric) / float(max(1, filled))
            if ratio < float(min_numeric_ratio):
                continue
            cols.append((ci, str(hv).strip()))
            # favor rows with many strong numeric columns
            score += float(numeric) + 3.0 * float(ratio)

        if len(cols) < int(min_data_cols):
            continue
        # require at least some total evidence of numeric columns
        if score < float(min_numeric_count) * float(max(1, min_data_cols)):
            continue

        # prefer earlier rows when scores tie
        bump = 0.001 * (scan_rows - r)
        score += bump
        if score > best_score:
            best_score = score
            best_row = r
            best_cols = cols

    return best_row, best_cols


def _iter_data_rows(
    ws,
    *,
    header_row: int,
    excel_col_indices: Sequence[int],
    max_consecutive_empty: int = 500,
    headers: Sequence[str] | None = None,
    sheet_name: str = "",
    debug: bool = False,
    row_id_min_run: int = 5,
    row_id_probe_limit: int = 400,
) -> Iterator[tuple[int, list[float | None]]]:
    """
    Yield rows as (excel_row_index_1_based, [values...]) for the selected columns.
    Starts after the header row and stops after a long run of empty rows (post-start).
    """
    try:
        max_row = int(getattr(ws, "max_row", 0) or 0)
    except Exception:
        max_row = 0
    if max_row <= header_row:
        return

    cols = [int(c) for c in excel_col_indices if int(c) > 0]
    if not cols:
        return

    min_col = min(cols)
    max_col = max(cols)
    col_offsets = [c - min_col for c in cols]

    def _int_like(v: float | None) -> bool:
        if v is None:
            return False
        try:
            rv = round(float(v))
            return abs(float(v) - float(rv)) < 1e-9
        except Exception:
            return False

    def _find_row_id_run(rows: list[tuple[int, list[float | None]]]) -> tuple[int, int] | None:
        """
        Return (row_id_col_idx, start_offset_in_rows) for a detected run of consecutive integers.
        rows are the numeric-bearing rows (any numeric across selected columns) encountered so far.
        """
        if not rows:
            return None
        ncols = len(rows[0][1]) if rows[0][1] else 0
        if ncols <= 0:
            return None
        min_run = max(3, int(row_id_min_run))
        if len(rows) < min_run:
            return None

        # Search for the earliest run of length >= min_run in any column.
        best: tuple[int, int] | None = None  # (start_offset, col_idx)
        for ci in range(ncols):
            for start in range(0, len(rows) - min_run + 1):
                window = [rows[start + k][1][ci] for k in range(min_run)]
                if not all(_int_like(v) for v in window):
                    continue
                base = int(round(float(window[0] or 0.0)))
                if any(int(round(float(window[k] or 0.0))) != base + k for k in range(min_run)):
                    continue
                if best is None or start < best[0] or (start == best[0] and ci < best[1]):
                    best = (start, ci)
                    break
        if best is None:
            return None
        start, ci = best
        return ci, start

    def _format_preview(
        rows: list[tuple[int, list[float | None]]],
        *,
        keep_excel_row_min: int | None,
        row_id_ci: int | None,
        max_rows: int = 30,
    ) -> str:
        hdrs = list(headers) if headers else [f"col_{i+1}" for i in range(len(cols))]
        hdrs = [str(h or "").strip() for h in hdrs]
        col_tags = [f"{c}:{h}" for c, h in zip(cols, hdrs)]
        head = "excel_row\tkeep\t" + "\t".join(col_tags)
        lines = [head]
        for excel_row, values in rows[: int(max_rows)]:
            keep = True
            if keep_excel_row_min is not None and excel_row < int(keep_excel_row_min):
                keep = False
            if keep and row_id_ci is not None:
                keep = _int_like(values[row_id_ci])
            vtxt = []
            for v in values:
                if v is None:
                    vtxt.append("")
                else:
                    try:
                        vtxt.append(str(float(v)))
                    except Exception:
                        vtxt.append("")
            lines.append(f"{excel_row}\t{'KEEP' if keep else 'DROP'}\t" + "\t".join(vtxt))
        return "\n".join(lines)

    started = False
    empty_run = 0
    buffer_any_numeric: list[tuple[int, list[float | None]]] = []
    row_id_ci: int | None = None
    keep_excel_row_min: int | None = None

    for r in range(int(header_row) + 1, max_row + 1):
        try:
            row_vals = next(ws.iter_rows(min_row=r, max_row=r, min_col=min_col, max_col=max_col, values_only=True))
        except Exception:
            continue
        values: list[float | None] = []
        any_num = False
        for off in col_offsets:
            try:
                v = row_vals[off]
            except Exception:
                v = None
            fv = _try_float(v)
            if fv is not None:
                any_num = True
            values.append(fv)

        if not started:
            if not any_num:
                continue
            started = True

        if any_num:
            empty_run = 0
            # Buffer "any numeric" rows until we can detect a row-id column to filter out summary blocks.
            if row_id_ci is None and keep_excel_row_min is None:
                buffer_any_numeric.append((int(r), values))
                found = _find_row_id_run(buffer_any_numeric)
                if found is not None:
                    row_id_ci, start_off = found
                    keep_excel_row_min = int(buffer_any_numeric[start_off][0])
                    if debug:
                        try:
                            hdr = ""
                            if headers and 0 <= int(row_id_ci) < len(list(headers)):
                                hdr = str(list(headers)[int(row_id_ci)] or "").strip()
                            _stderr(
                                f"[TEST_DATA] row-id detected: sheet={sheet_name!r} "
                                f"col_idx={row_id_ci} header={hdr!r} start_excel_row={keep_excel_row_min}"
                            )
                            _stderr("[TEST_DATA] row preview (excel_col:header):\n" + _format_preview(
                                buffer_any_numeric,
                                keep_excel_row_min=keep_excel_row_min,
                                row_id_ci=row_id_ci,
                                max_rows=30,
                            ))
                        except Exception:
                            pass

                    # Flush buffered rows from the detected start row, keeping only those with an int-like row id.
                    for br, bvals in buffer_any_numeric[start_off:]:
                        if row_id_ci is not None and _int_like(bvals[row_id_ci]):
                            yield int(br), bvals
                    buffer_any_numeric.clear()
                    continue

                # Avoid unbounded buffering on sheets with no detectable row-id column.
                if len(buffer_any_numeric) >= int(row_id_probe_limit):
                    if debug:
                        try:
                            _stderr(
                                f"[TEST_DATA] row-id probe limit hit: sheet={sheet_name!r} "
                                f"(buffer={len(buffer_any_numeric)}). Using legacy row iteration."
                            )
                        except Exception:
                            pass
                    for br, bvals in buffer_any_numeric:
                        yield int(br), bvals
                    buffer_any_numeric.clear()
                    keep_excel_row_min = 0  # sentinel: stop buffering, no filtering
                    continue

                continue

            if row_id_ci is not None and keep_excel_row_min is not None:
                if int(r) < int(keep_excel_row_min):
                    continue
                if not _int_like(values[row_id_ci]):
                    continue
            yield r, values
        else:
            empty_run += 1
            if empty_run >= int(max_consecutive_empty):
                break

    # Fallback: if we never detected a row-id run, yield buffered numeric rows as-is (legacy behavior).
    if buffer_any_numeric:
        if debug:
            try:
                _stderr(
                    f"[TEST_DATA] row-id not detected: sheet={sheet_name!r} "
                    f"(yielding {len(buffer_any_numeric)} buffered rows; legacy behavior)"
                )
                _stderr("[TEST_DATA] row preview (legacy):\n" + _format_preview(
                    buffer_any_numeric,
                    keep_excel_row_min=None,
                    row_id_ci=None,
                    max_rows=30,
                ))
            except Exception:
                pass
        for br, bvals in buffer_any_numeric:
            yield int(br), bvals


def _write_workbook_sqlite(
    *,
    excel_path: Path,
    sqlite_path: Path,
    overwrite: bool,
    max_scan_rows: int,
    max_cols: int,
    lookahead_rows: int,
    min_numeric_count: int,
    min_numeric_ratio: float,
    min_data_cols: int,
) -> dict[str, Any]:
    if excel_path.suffix.lower() not in EXCEL_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {excel_path.suffix}")
    if excel_path.suffix.lower() == ".xls":
        # openpyxl cannot read .xls; xlrd is not shipped in this runtime by default.
        raise RuntimeError("Unsupported .xls input (convert to .xlsx or install xlrd and add an .xls reader).")

    if not excel_path.exists():
        raise FileNotFoundError(str(excel_path))

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        if not overwrite:
            return {
                "source_file": str(excel_path),
                "sqlite_path": str(sqlite_path),
                "skipped": True,
                "reason": "exists",
            }
        try:
            sqlite_path.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError("openpyxl is required to read Excel files in this runtime.") from exc

    _stderr(f"[EXCEL] {excel_path}")

    env = _load_test_data_env()
    fuzzy_enabled = _truthy(env.get("EIDAT_TEST_DATA_FUZZY_HEADER_STICK", "1"))
    fuzzy_min_ratio = _float(env.get("EIDAT_TEST_DATA_FUZZY_HEADER_MIN_RATIO", "0.82"), 0.82)
    debug_fuzzy = _truthy(env.get("EIDAT_TEST_DATA_FUZZY_DEBUG", "0"))
    debug_table = _truthy(env.get("EIDAT_TEST_DATA_DEBUG_TABLE", "0")) or debug_fuzzy
    trend_cols = _load_trend_column_names() if fuzzy_enabled else []

    wb = load_workbook(str(excel_path), data_only=True, read_only=True)
    try:
        sheetnames = list(getattr(wb, "sheetnames", []) or [])
        if not sheetnames:
            raise RuntimeError("Workbook has no sheets.")
        sheets: list[DetectedSheet] = []
        for sheet_name in sheetnames:
            ws = wb[sheet_name]
            header_row, cols = _detect_header_row(
                ws,
                max_scan_rows=max_scan_rows,
                max_cols=max_cols,
                lookahead_rows=lookahead_rows,
                min_numeric_count=min_numeric_count,
                min_numeric_ratio=min_numeric_ratio,
                min_data_cols=min_data_cols,
            )
            if not header_row or not cols:
                _stderr(f"[SHEET] {sheet_name}: no data headers detected (skipped)")
                continue
            excel_col_indices = [c for c, _ in cols]
            headers = [h for _, h in cols]
            mapped_headers = (
                _map_headers_to_config(headers, trend_cols, min_ratio=float(fuzzy_min_ratio))
                if trend_cols
                else list(headers)
            )
            if debug_fuzzy and mapped_headers != headers:
                for oh, mh in zip(headers, mapped_headers):
                    if oh != mh:
                        _stderr(f"[TEST_DATA] header map: {sheet_name}: {oh!r} -> {mh!r}")
            col_idents = _dedupe_idents(mapped_headers)
            sheets.append(
                DetectedSheet(
                    sheet_name=str(sheet_name),
                    table_name=_safe_table_name(str(sheet_name)),
                    header_row=int(header_row),
                    excel_col_indices=excel_col_indices,
                    headers=headers,
                    mapped_headers=mapped_headers,
                    col_idents=col_idents,
                )
            )
            _stderr(f"[SHEET] {sheet_name}: header_row={header_row} cols={len(cols)}")

        if not sheets:
            raise RuntimeError("No sheets contained detectable numeric columns.")

        conn = sqlite3.connect(str(sqlite_path))
        try:
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA journal_mode=DELETE;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS __workbook (
                  source_file TEXT NOT NULL,
                  imported_epoch_ns INTEGER NOT NULL,
                  excel_size_bytes INTEGER NOT NULL,
                  excel_mtime_ns INTEGER NOT NULL
                );
                """
            )
            st = excel_path.stat()
            conn.execute(
                "INSERT INTO __workbook(source_file, imported_epoch_ns, excel_size_bytes, excel_mtime_ns) VALUES(?, ?, ?, ?)",
                (str(excel_path), int(_now_ns()), int(st.st_size), int(st.st_mtime_ns)),
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS __sheet_info (
                  sheet_name TEXT PRIMARY KEY,
                  table_name TEXT NOT NULL,
                  header_row INTEGER NOT NULL,
                  excel_col_indices_json TEXT NOT NULL,
                  headers_json TEXT NOT NULL,
                  columns_json TEXT NOT NULL,
                  rows_inserted INTEGER NOT NULL
                );
                """
            )
            # Back/forward compat: add mapped headers column if missing.
            try:
                existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(__sheet_info)").fetchall()}
                if "mapped_headers_json" not in existing_cols:
                    conn.execute("ALTER TABLE __sheet_info ADD COLUMN mapped_headers_json TEXT;")
            except Exception:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS __column_map (
                  sheet_name TEXT NOT NULL,
                  header TEXT NOT NULL,
                  mapped_header TEXT NOT NULL,
                  sqlite_column TEXT NOT NULL,
                  PRIMARY KEY(sheet_name, header)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS __meta_cells (
                  sheet_name TEXT NOT NULL,
                  excel_row INTEGER NOT NULL,
                  excel_col INTEGER NOT NULL,
                  value TEXT NOT NULL
                );
                """
            )

            outputs: list[dict[str, Any]] = []
            for s in sheets:
                ws = wb[s.sheet_name]

                # Store metadata cells above the detected header row for later parsing/trending.
                try:
                    meta_max_row = max(1, min(int(s.header_row) - 1, 250))
                    meta_max_col = max(1, min(int(getattr(ws, "max_column", 0) or 0), 80))
                except Exception:
                    meta_max_row = 0
                    meta_max_col = 0
                if meta_max_row and meta_max_col:
                    meta_rows = ws.iter_rows(min_row=1, max_row=meta_max_row, min_col=1, max_col=meta_max_col, values_only=True)
                    meta_to_insert: list[tuple[str, int, int, str]] = []
                    for rr, row_vals in enumerate(meta_rows, start=1):
                        for cc, v in enumerate(list(row_vals), start=1):
                            if _is_blank(v):
                                continue
                            meta_to_insert.append((s.sheet_name, int(rr), int(cc), str(v)))
                    if meta_to_insert:
                        conn.executemany(
                            "INSERT INTO __meta_cells(sheet_name, excel_row, excel_col, value) VALUES(?, ?, ?, ?)",
                            meta_to_insert,
                        )

                col_defs = ", ".join([f"\"{c}\" REAL" for c in s.col_idents])
                conn.execute(f"DROP TABLE IF EXISTS \"{s.table_name}\";")
                conn.execute(f"CREATE TABLE \"{s.table_name}\" (excel_row INTEGER NOT NULL, {col_defs});")

                rows_inserted = 0
                placeholders = ", ".join(["?"] * (1 + len(s.col_idents)))
                ins_sql = f"INSERT INTO \"{s.table_name}\" (excel_row, " + ", ".join([f"\"{c}\"" for c in s.col_idents]) + f") VALUES({placeholders})"

                batch: list[tuple[Any, ...]] = []
                for excel_row, values in _iter_data_rows(
                    ws,
                    header_row=s.header_row,
                    excel_col_indices=s.excel_col_indices,
                    headers=list(s.headers),
                    sheet_name=str(s.sheet_name),
                    debug=bool(debug_table),
                ):
                    batch.append((int(excel_row), *values))
                    if len(batch) >= 1000:
                        conn.executemany(ins_sql, batch)
                        rows_inserted += len(batch)
                        batch.clear()
                if batch:
                    conn.executemany(ins_sql, batch)
                    rows_inserted += len(batch)
                    batch.clear()

                conn.execute(
                    """
                    INSERT INTO __sheet_info(
                      sheet_name, table_name, header_row,
                      excel_col_indices_json, headers_json, columns_json,
                      mapped_headers_json,
                      rows_inserted
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.sheet_name,
                        s.table_name,
                        int(s.header_row),
                        json.dumps(list(s.excel_col_indices)),
                        json.dumps(list(s.headers), ensure_ascii=False),
                        json.dumps({h: c for h, c in zip(s.headers, s.col_idents)}, ensure_ascii=False),
                        json.dumps(list(getattr(s, "mapped_headers", list(s.headers))), ensure_ascii=False),
                        int(rows_inserted),
                    ),
                )
                try:
                    conn.executemany(
                        "INSERT OR REPLACE INTO __column_map(sheet_name, header, mapped_header, sqlite_column) VALUES(?, ?, ?, ?)",
                        [
                            (s.sheet_name, h, mh, ci)
                            for h, mh, ci in zip(
                                s.headers, getattr(s, "mapped_headers", list(s.headers)), s.col_idents
                            )
                        ],
                    )
                except Exception:
                    pass
                outputs.append(
                    {
                        "sheet": s.sheet_name,
                        "table": s.table_name,
                        "header_row": int(s.header_row),
                        "columns": list(s.headers),
                        "mapped_columns": list(getattr(s, "mapped_headers", list(s.headers))),
                        "rows_inserted": int(rows_inserted),
                    }
                )
                _stderr(f"[SHEET DONE] {s.sheet_name}: rows={rows_inserted}")

            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        _stderr(f"[DONE] {excel_path.name} -> {sqlite_path}")
        return {
            "source_file": str(excel_path),
            "sqlite_path": str(sqlite_path),
            "skipped": False,
            "sheets": outputs,
        }
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _should_skip_dir(p: Path) -> bool:
    # Avoid scanning app/runtime internals by default when users provide broad roots.
    low = p.name.strip().lower()
    return low in {
        ".git",
        ".venv",
        "__pycache__",
        "eidat",
        "eidat support",
        "master_database",
        "ui_next",
        "lib",
        "scripts",
    }


def find_excel_files(root: Path) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists() or not base.is_dir():
        return []
    out: list[Path] = []
    stack = [base]
    while stack:
        d = stack.pop()
        try:
            for child in d.iterdir():
                try:
                    if child.is_dir():
                        if _should_skip_dir(child):
                            continue
                        stack.append(child)
                    elif child.is_file():
                        suf = child.suffix.lower()
                        if suf in EXCEL_EXTENSIONS and not child.name.startswith("~$"):
                            out.append(child)
                except Exception:
                    continue
        except Exception:
            continue
    out.sort(key=lambda p: p.name.lower())
    return out


def excel_to_sqlite(
    *,
    global_repo: Path,
    excel_files: Sequence[Path] | None = None,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = True,
    max_scan_rows: int = 200,
    max_cols: int = 200,
    lookahead_rows: int = 60,
    min_numeric_count: int = 8,
    min_numeric_ratio: float = 0.60,
    min_data_cols: int = 1,
) -> dict[str, Any]:
    repo = Path(global_repo).expanduser()
    if not repo.exists() or not repo.is_dir():
        raise RuntimeError(f"Global repo not found: {repo}")

    if excel_files:
        targets = [Path(p).expanduser() for p in excel_files]
    else:
        scan_root = Path(data_dir).expanduser() if data_dir is not None else repo
        targets = find_excel_files(scan_root)

    if not targets:
        raise RuntimeError("No Excel files found.")

    if out_dir is not None:
        out_root = Path(out_dir).expanduser()
    else:
        out_root = support_paths(repo).support_dir / "excel_sqlite"
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    ok = 0
    failed = 0
    for fp in targets:
        try:
            db = out_root / f"{fp.stem}.sqlite3"
            res = _write_workbook_sqlite(
                excel_path=fp,
                sqlite_path=db,
                overwrite=bool(overwrite),
                max_scan_rows=int(max_scan_rows),
                max_cols=int(max_cols),
                lookahead_rows=int(lookahead_rows),
                min_numeric_count=int(min_numeric_count),
                min_numeric_ratio=float(min_numeric_ratio),
                min_data_cols=int(min_data_cols),
            )
            results.append(res)
            if res.get("skipped"):
                continue
            ok += 1
        except Exception as exc:
            failed += 1
            _stderr(f"[FAIL] {fp}: {exc}")
            results.append(
                {
                    "source_file": str(fp),
                    "sqlite_path": "",
                    "skipped": False,
                    "error": str(exc),
                }
            )

    return {
        "global_repo": str(repo),
        "data_dir": str(Path(data_dir).expanduser() if data_dir is not None else repo),
        "out_dir": str(out_root),
        "excel_count": len(targets),
        "sqlite_ok": int(ok),
        "sqlite_failed": int(failed),
        "results": results,
    }
