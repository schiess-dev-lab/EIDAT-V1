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

    started = False
    empty_run = 0

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
            yield r, values
        else:
            empty_run += 1
            if empty_run >= int(max_consecutive_empty):
                break


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
    debug = _truthy(env.get("EIDAT_TEST_DATA_FUZZY_DEBUG", "0"))
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
            if debug and mapped_headers != headers:
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
                for excel_row, values in _iter_data_rows(ws, header_row=s.header_row, excel_col_indices=s.excel_col_indices):
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

    out_root = Path(out_dir).expanduser() if out_dir is not None else (repo / "EIDAT Support" / "excel_sqlite")
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
