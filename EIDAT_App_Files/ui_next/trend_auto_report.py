from __future__ import annotations

"""
Auto-report generation for Test Data Trend / Analyze.

This module is intentionally imported lazily from backend/UI because it depends
on optional plotting/scientific libraries (matplotlib, numpy).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math
import sqlite3
import statistics
import time


def _now_datestr() -> str:
    try:
        return time.strftime("%Y-%m-%d")
    except Exception:
        return "unknown-date"


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s or "").strip() if ch.isalnum())


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    Path(path).expanduser().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def default_trend_auto_report_config() -> dict:
    return {
        "version": 1,
        "model": {
            "type": "poly",
            "degree": 3,
            "aggregate": "median",
            "grid_points": 200,
            "normalize_x": True,
        },
        "watch": {
            "curve_deviation": {
                "max_abs": None,
                "max_pct": 10.0,
                "rms_pct": 5.0,
            }
        },
        "grading": {
            "zscore_pass_max": 2.0,
            "zscore_watch_max": 3.0,
        },
        "report": {
            "statistics": "from_excel_trend_config",
            "include_metrics": True,
            "graphs_at_end": True,
            "max_findings": 12,
        },
        "highlight": {
            "default_serials": [],
            "policy": "watch_only",
            "colors": ["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316"],
        },
    }


def load_trend_auto_report_config(*, project_dir: Path | None, central_path: Path) -> dict:
    cfg = default_trend_auto_report_config()
    cpath = Path(central_path).expanduser()
    if cpath.exists():
        try:
            raw = _read_json(cpath)
            if isinstance(raw, dict):
                cfg = _deep_merge(cfg, raw)
        except Exception:
            pass
    if project_dir is not None:
        ppath = Path(project_dir).expanduser() / "trend_auto_report_config.json"
        if ppath.exists():
            try:
                raw = _read_json(ppath)
                if isinstance(raw, dict):
                    cfg = _deep_merge(cfg, raw)
            except Exception:
                pass
    return cfg


def _td_list_serials(conn: sqlite3.Connection) -> list[str]:
    try:
        rows = conn.execute("SELECT serial FROM td_sources ORDER BY serial").fetchall()
    except Exception:
        return []
    out: list[str] = []
    for r in rows:
        try:
            s = str(r[0] or "").strip()
        except Exception:
            s = ""
        if s:
            out.append(s)
    return out


def _td_list_runs(conn: sqlite3.Connection) -> list[dict]:
    try:
        rows = conn.execute("SELECT run_name, default_x, display_name FROM td_runs ORDER BY run_name").fetchall()
    except Exception:
        return []
    out: list[dict] = []
    for rn, dx, dn in rows:
        out.append(
            {
                "run_name": str(rn or "").strip(),
                "default_x": str(dx or "").strip(),
                "display_name": str(dn or "").strip(),
            }
        )
    return out


def _td_list_y_columns(conn: sqlite3.Connection, run_name: str) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT name, units FROM td_columns WHERE run_name=? AND kind='y' ORDER BY name",
            (str(run_name or "").strip(),),
        ).fetchall()
    except Exception:
        return []
    out: list[dict] = []
    for name, units in rows:
        out.append({"name": str(name or "").strip(), "units": str(units or "").strip()})
    return out


def _resolve_td_y_col(conn: sqlite3.Connection, run_name: str, target: str) -> tuple[str, str]:
    """Return (actual_column_name, units) for a desired Y column in a run (best-effort)."""
    tgt = str(target or "").strip()
    if not tgt:
        return "", ""
    want = _norm_key(tgt)
    for c in _td_list_y_columns(conn, run_name):
        name = str(c.get("name") or "").strip()
        if name and _norm_key(name) == want:
            return name, str(c.get("units") or "").strip()
    return tgt, ""


def _td_metric_map(conn: sqlite3.Connection, run_name: str, column_name: str, stat: str) -> dict[str, float]:
    run = str(run_name or "").strip()
    col = str(column_name or "").strip()
    st = str(stat or "").strip().lower()
    if not run or not col or not st:
        return {}
    try:
        rows = conn.execute(
            """
            SELECT serial, value_num
            FROM td_metrics
            WHERE run_name=? AND column_name=? AND lower(stat)=?
            """,
            (run, col, st),
        ).fetchall()
    except Exception:
        rows = []
    out: dict[str, float] = {}
    for sn, val in rows:
        s = str(sn or "").strip()
        fv = _safe_float(val)
        if s and fv is not None and math.isfinite(float(fv)):
            out[s] = float(fv)
    return out


def _td_units_for_y(conn: sqlite3.Connection) -> dict[str, list[str]]:
    try:
        rows = conn.execute("SELECT name, units FROM td_columns WHERE kind='y'").fetchall()
    except Exception:
        rows = []
    by: dict[str, list[str]] = {}
    for name, units in rows:
        n = str(name or "").strip()
        u = str(units or "").strip()
        if not n or not u:
            continue
        by.setdefault(_norm_key(n), []).append(u)
    return by


def _td_ranges_for_y(conn: sqlite3.Connection) -> dict[str, tuple[float | None, float | None]]:
    try:
        rows = conn.execute(
            """
            SELECT column_name,
                   MIN(CASE WHEN lower(stat)='min' THEN value_num END) AS mn,
                   MAX(CASE WHEN lower(stat)='max' THEN value_num END) AS mx
            FROM td_metrics
            GROUP BY column_name
            """
        ).fetchall()
    except Exception:
        rows = []
    out: dict[str, tuple[float | None, float | None]] = {}
    for col, mn, mx in rows:
        name = str(col or "").strip()
        if not name:
            continue
        out[_norm_key(name)] = (_safe_float(mn), _safe_float(mx))
    return out


def _ensure_backup(path: Path) -> Path:
    p = Path(path).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = p.with_name(f"{p.stem}.backup.{stamp}{p.suffix}")
    backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def autofill_excel_trend_config_from_td_cache(
    db_path: Path,
    excel_trend_config_path: Path,
    *,
    fill_units: bool = True,
    fill_ranges: bool = True,
    add_missing_columns: bool = False,
) -> tuple[dict, str]:
    cfg_path = Path(excel_trend_config_path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Excel trend config not found: {cfg_path}")
    raw = _read_json(cfg_path)
    if not isinstance(raw, dict):
        raise ValueError("excel_trend_config.json must be a JSON object")
    cols = raw.get("columns")
    if not isinstance(cols, list):
        cols = []
    raw["columns"] = cols

    changed: list[str] = []
    with sqlite3.connect(str(Path(db_path).expanduser())) as conn:
        units_by = _td_units_for_y(conn) if fill_units else {}
        ranges_by = _td_ranges_for_y(conn) if fill_ranges else {}

        existing_by_norm: dict[str, dict] = {}
        for c in cols:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            if name:
                existing_by_norm[_norm_key(name)] = c

        try:
            all_y_rows = conn.execute("SELECT DISTINCT name FROM td_columns WHERE kind='y' ORDER BY name").fetchall()
        except Exception:
            all_y_rows = []
        all_y = [str(r[0] or "").strip() for r in all_y_rows if str(r[0] or "").strip()]

        for y_name in all_y:
            nk = _norm_key(y_name)
            col_cfg = existing_by_norm.get(nk)
            if col_cfg is None:
                if not add_missing_columns:
                    continue
                col_cfg = {"name": y_name, "units": "", "range_min": None, "range_max": None}
                cols.append(col_cfg)
                existing_by_norm[nk] = col_cfg
                changed.append(f"Added column '{y_name}'")

            if fill_units:
                cur_u = str(col_cfg.get("units") or "").strip()
                if not cur_u:
                    cand = units_by.get(nk) or []
                    if cand:
                        pick = max(set(cand), key=lambda s: cand.count(s))
                        col_cfg["units"] = pick
                        changed.append(f"Filled units for '{col_cfg.get('name')}' -> {pick}")

            if fill_ranges:
                mn_cur = col_cfg.get("range_min")
                mx_cur = col_cfg.get("range_max")
                if (mn_cur is None) or (mx_cur is None):
                    mn, mx = ranges_by.get(nk, (None, None))
                    if mn_cur is None and mn is not None:
                        col_cfg["range_min"] = mn
                        changed.append(f"Filled range_min for '{col_cfg.get('name')}' -> {mn:g}")
                    if mx_cur is None and mx is not None:
                        col_cfg["range_max"] = mx
                        changed.append(f"Filled range_max for '{col_cfg.get('name')}' -> {mx:g}")

    if changed:
        backup = _ensure_backup(cfg_path)
        _write_json(cfg_path, raw)
        changed.insert(0, f"Backup created: {backup}")
    summary = "\n".join(changed) if changed else "No changes needed."
    return raw, summary


@dataclass(frozen=True)
class CurveSeries:
    serial: str
    x: list[float]
    y: list[float]


def _load_curves(conn: sqlite3.Connection, run: str, y_name: str, x_name: str) -> list[CurveSeries]:
    rows = conn.execute(
        """
        SELECT serial, x_json, y_json
        FROM td_curves
        WHERE run_name=? AND y_name=? AND x_name=?
        ORDER BY serial
        """,
        (run, y_name, x_name),
    ).fetchall()
    out: list[CurveSeries] = []
    for serial, xj, yj in rows:
        sn = str(serial or "").strip()
        if not sn:
            continue
        try:
            xs = json.loads(xj) if isinstance(xj, str) else []
            ys = json.loads(yj) if isinstance(yj, str) else []
        except Exception:
            continue
        if not isinstance(xs, list) or not isinstance(ys, list) or not xs or not ys:
            continue
        pts = min(len(xs), len(ys))
        x: list[float] = []
        y: list[float] = []
        for i in range(pts):
            xf = _safe_float(xs[i])
            yf = _safe_float(ys[i])
            if xf is None or yf is None:
                continue
            x.append(float(xf))
            y.append(float(yf))
        if len(x) < 2 or len(y) < 2:
            continue
        out.append(CurveSeries(serial=sn, x=x, y=y))
    return out


def _interp_linear(x_src: list[float], y_src: list[float], x_grid: list[float]) -> list[float]:
    pts = list(zip(x_src, y_src))
    pts.sort(key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    out: list[float] = []
    j = 0
    n = len(xs)
    for x in x_grid:
        if x < xs[0] or x > xs[-1]:
            out.append(float("nan"))
            continue
        while j + 1 < n and xs[j + 1] < x:
            j += 1
        x0 = xs[j]
        y0 = ys[j]
        if j + 1 >= n:
            out.append(y0)
            continue
        x1 = xs[j + 1]
        y1 = ys[j + 1]
        if x1 == x0:
            out.append(y0)
            continue
        t = (x - x0) / (x1 - x0)
        out.append(y0 + t * (y1 - y0))
    return out


def _nan_median(cols: list[list[float]]) -> list[float]:
    if not cols:
        return []
    m = len(cols[0])
    out: list[float] = []
    for i in range(m):
        vs = [c[i] for c in cols if i < len(c) and isinstance(c[i], (int, float)) and not math.isnan(float(c[i]))]
        out.append(float(statistics.median(vs)) if vs else float("nan"))
    return out


def _nan_mean(cols: list[list[float]]) -> list[float]:
    if not cols:
        return []
    m = len(cols[0])
    out: list[float] = []
    for i in range(m):
        vs = [c[i] for c in cols if i < len(c) and isinstance(c[i], (int, float)) and not math.isnan(float(c[i]))]
        out.append(float(statistics.mean(vs)) if vs else float("nan"))
    return out


def _nan_std(cols: list[list[float]]) -> list[float]:
    if not cols:
        return []
    m = len(cols[0])
    out: list[float] = []
    for i in range(m):
        vs = [c[i] for c in cols if i < len(c) and isinstance(c[i], (int, float)) and not math.isnan(float(c[i]))]
        if len(vs) <= 1:
            out.append(0.0 if vs else float("nan"))
        else:
            out.append(float(statistics.pstdev(vs)))
    return out


def _poly_fit(x: list[float], y: list[float], degree: int, *, normalize_x: bool) -> dict:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("numpy is required for polynomial model fitting.") from exc
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    if len(xs) < max(2, degree + 1):
        return {"degree": int(degree), "coeffs": [], "rmse": None, "x0": None, "sx": None}
    if normalize_x:
        x0 = float(np.mean(xs))
        sx = float(np.std(xs)) or 1.0
        xn = [(v - x0) / sx for v in xs]
    else:
        x0 = 0.0
        sx = 1.0
        xn = xs
    coeffs = np.polyfit(np.array(xn, dtype=float), np.array(ys, dtype=float), int(degree)).tolist()
    p = np.poly1d(coeffs)
    yhat = p(np.array(xn, dtype=float))
    rmse = float(np.sqrt(np.mean((np.array(ys, dtype=float) - yhat) ** 2)))
    return {"degree": int(degree), "coeffs": [float(c) for c in coeffs], "rmse": rmse, "x0": float(x0), "sx": float(sx)}


def _fmt_equation(poly: dict) -> str:
    coeffs = poly.get("coeffs") or []
    deg = int(poly.get("degree") or 0)
    if not coeffs:
        return ""
    parts: list[str] = []
    for i, c in enumerate(coeffs):
        power = deg - i
        try:
            cf = float(c)
        except Exception:
            continue
        if power == 0:
            parts.append(f"{cf:+.4g}")
        elif power == 1:
            parts.append(f"{cf:+.4g}·x")
        else:
            parts.append(f"{cf:+.4g}·x^{power}")
    expr = " ".join(parts).lstrip("+").strip()
    x0 = poly.get("x0")
    sx = poly.get("sx")
    if x0 is not None and sx is not None:
        return f"y = {expr}  (x'=(x-{float(x0):.4g})/{float(sx):.4g})"
    return f"y = {expr}"


def _grade_from_z(z: float, pass_max: float, watch_max: float) -> str:
    az = abs(float(z))
    if az <= float(pass_max):
        return "PASS"
    if az <= float(watch_max):
        return "WATCH"
    return "FAIL"


def _summarize_units(excel_cfg: dict, *, params_used: set[str]) -> dict[str, int]:
    cols = excel_cfg.get("columns") or []
    out: dict[str, int] = {}
    for c in cols:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if not name or _norm_key(name) not in params_used:
            continue
        u = str(c.get("units") or "").strip() or "(blank)"
        out[u] = int(out.get(u, 0)) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _summarize_units_from_td(conn: sqlite3.Connection, *, runs: list[str], params: list[str]) -> dict[str, int]:
    want = {_norm_key(p) for p in params if str(p).strip()}
    counts: dict[str, int] = {}
    for run in runs:
        for c in _td_list_y_columns(conn, run):
            name = str(c.get("name") or "").strip()
            if not name or _norm_key(name) not in want:
                continue
            u = str(c.get("units") or "").strip() or "(blank)"
            counts[u] = int(counts.get(u, 0)) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _figure_text_page(title: str, lines: list[str]):
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(8.5, 11.0), dpi=120)
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=16, fontweight="bold", va="top")
    y = 0.92
    for line in lines:
        fig.text(0.06, y, str(line), fontsize=10, va="top")
        y -= 0.018
        if y < 0.06:
            break
    return fig


def _figure_table_page(title: str, columns: list[str], rows: list[list[object]]):
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(8.5, 11.0), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
    cell_text = [[("" if v is None else str(v)) for v in r] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="left", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    try:
        for (r, _c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f1f5f9")
            if r % 2 == 0 and r != 0:
                cell.set_facecolor("#fafafa")
    except Exception:
        pass
    return fig


def generate_test_data_auto_report(
    project_dir: Path,
    workbook_path: Path,
    output_pdf: Path,
    *,
    highlighted_serials: list[str],
    options: dict,
) -> dict:
    try:
        from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required to generate PDF reports.") from exc

    from . import backend as be  # local import to reuse existing cache builder

    proj = Path(project_dir).expanduser()
    wb = Path(workbook_path).expanduser()
    out_pdf = Path(output_pdf).expanduser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    cfg_excel_path = Path(options.get("excel_trend_config_path") or be.DEFAULT_EXCEL_TREND_CONFIG).expanduser()
    excel_cfg = be.load_excel_trend_config(cfg_excel_path)

    report_cfg = be.load_trend_auto_report_config(proj)
    model_cfg = report_cfg.get("model") or {}
    watch_cfg = (report_cfg.get("watch") or {}).get("curve_deviation") or {}
    grade_cfg = report_cfg.get("grading") or {}
    report_opts = report_cfg.get("report") or {}
    hi_cfg = report_cfg.get("highlight") or {}

    rebuild = bool(options.get("rebuild_cache"))
    db_path = be.ensure_test_data_project_cache(proj, wb, rebuild=rebuild)

    if bool(options.get("update_excel_trend_config", True)):
        _, change_summary = be.autofill_excel_trend_config_from_td_cache(
            db_path,
            cfg_excel_path,
            fill_units=True,
            fill_ranges=True,
            add_missing_columns=bool(options.get("add_missing_columns")),
        )
    else:
        change_summary = "excel_trend_config.json update disabled."

    with sqlite3.connect(str(Path(db_path).expanduser())) as conn:
        all_serials = _td_list_serials(conn)
        run_rows = _td_list_runs(conn)
        run_by_name = {str(r.get("run_name") or ""): r for r in run_rows if str(r.get("run_name") or "")}

        selected_runs = options.get("runs") or []
        runs = [str(r).strip() for r in selected_runs if str(r).strip()] if isinstance(selected_runs, list) else []
        if not runs:
            runs = [str(r.get("run_name") or "").strip() for r in run_rows if str(r.get("run_name") or "").strip()]
        runs = [r for r in runs if r in run_by_name]

        excel_cols = excel_cfg.get("columns") or []
        excel_names = {
            _norm_key(str(c.get("name") or "")): str(c.get("name") or "").strip()
            for c in excel_cols
            if isinstance(c, dict) and str(c.get("name") or "").strip()
        }

        selected_params = options.get("params") or []
        params = [str(p).strip() for p in selected_params if str(p).strip()] if isinstance(selected_params, list) else []
        if not params:
            # Auto-detect from cache/workbook (td_columns). Do not require excel_trend_config intersection.
            seen: set[str] = set()
            auto: list[str] = []
            for run in runs:
                for c in _td_list_y_columns(conn, run):
                    name = str(c.get("name") or "").strip()
                    if not name:
                        continue
                    nk = _norm_key(name)
                    if nk in seen:
                        continue
                    seen.add(nk)
                    auto.append(name)
            params = sorted(auto, key=lambda s: s.lower())

        params_norm = {_norm_key(p) for p in params if p}
        hi = [s for s in highlighted_serials if s in all_serials]

        stats = excel_cfg.get("statistics") or ["mean", "min", "max", "std", "median", "count"]
        stats = [str(s).strip().lower() for s in stats if str(s).strip()]
        include_metrics = bool(options.get("include_metrics", bool(report_opts.get("include_metrics", True))))

        grid_points = int(model_cfg.get("grid_points") or 200) or 200
        degree = int(model_cfg.get("degree") or 3) or 3
        normalize_x = bool(model_cfg.get("normalize_x", True))

        max_abs_thr = _safe_float(watch_cfg.get("max_abs"))
        max_pct_thr = _safe_float(watch_cfg.get("max_pct"))
        rms_pct_thr = _safe_float(watch_cfg.get("rms_pct"))

        z_pass = float(grade_cfg.get("zscore_pass_max") or 2.0)
        z_watch = float(grade_cfg.get("zscore_watch_max") or 3.0)

        colors = hi_cfg.get("colors") or ["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316"]
        colors = [str(c) for c in colors if str(c).strip()] or ["#ef4444"]

        curves_summary: dict[str, dict[str, dict]] = {}
        watch_items: list[dict] = []
        grading_rows: list[dict] = []

        for run in runs:
            run_meta = run_by_name.get(run) or {}
            x_name = str(run_meta.get("default_x") or "").strip() or "Time"
            y_cols = _td_list_y_columns(conn, run)
            y_by_norm = {_norm_key(str(c.get("name") or "")): c for c in y_cols if str(c.get("name") or "").strip()}

            for p in params:
                nk = _norm_key(p)
                if nk not in params_norm or nk not in y_by_norm:
                    continue
                y_name = str(y_by_norm[nk].get("name") or "").strip()
                units = str(y_by_norm[nk].get("units") or "").strip()

                series = _load_curves(conn, run, y_name, x_name)
                if not series:
                    continue

                mins = [min(s.x) for s in series if s.x]
                maxs = [max(s.x) for s in series if s.x]
                if not mins or not maxs:
                    continue
                overlap_lo = max(mins)
                overlap_hi = min(maxs)
                global_lo = min(mins)
                global_hi = max(maxs)
                lo, hi_dom = overlap_lo, overlap_hi
                if not (math.isfinite(lo) and math.isfinite(hi_dom)) or (hi_dom - lo) <= 1e-12:
                    lo, hi_dom = global_lo, global_hi
                if not (math.isfinite(lo) and math.isfinite(hi_dom)) or (hi_dom - lo) <= 1e-12:
                    continue

                x_grid = [lo + (hi_dom - lo) * (i / (grid_points - 1)) for i in range(grid_points)]
                y_resampled_by_sn = {s.serial: _interp_linear(s.x, s.y, x_grid) for s in series}
                y_matrix = list(y_resampled_by_sn.values())
                master_y = _nan_median(y_matrix)
                std_y = _nan_std(y_matrix)

                fit_x = []
                fit_y = []
                for x, yv in zip(x_grid, master_y):
                    if isinstance(yv, (int, float)) and not math.isnan(float(yv)):
                        fit_x.append(float(x))
                        fit_y.append(float(yv))
                poly = _poly_fit(fit_x, fit_y, degree, normalize_x=normalize_x) if fit_x else {"degree": degree, "coeffs": [], "rmse": None, "x0": None, "sx": None}
                eqn = _fmt_equation(poly)

                denom = max((abs(v) for v in master_y if isinstance(v, (int, float)) and not math.isnan(float(v))), default=0.0)
                denom = float(denom) if denom > 0 else 1.0

                score_by_sn: dict[str, float] = {}
                dev_by_sn: dict[str, dict] = {}
                for sn, yv in y_resampled_by_sn.items():
                    residual: list[float] = []
                    for a, b in zip(yv, master_y):
                        if not (isinstance(a, (int, float)) and not math.isnan(float(a))):
                            continue
                        if not (isinstance(b, (int, float)) and not math.isnan(float(b))):
                            continue
                        residual.append(float(a) - float(b))
                    if not residual:
                        continue
                    max_abs = max(abs(r) for r in residual)
                    rms = math.sqrt(sum(r * r for r in residual) / max(1, len(residual)))
                    max_pct = (max_abs / denom) * 100.0
                    rms_pct = (rms / denom) * 100.0
                    idx = 0
                    best = -1.0
                    for i, (a, b) in enumerate(zip(yv, master_y)):
                        if not (isinstance(a, (int, float)) and not math.isnan(float(a))):
                            continue
                        if not (isinstance(b, (int, float)) and not math.isnan(float(b))):
                            continue
                        vabs = abs(float(a) - float(b))
                        if vabs > best:
                            best = vabs
                            idx = i
                    x_at = float(x_grid[idx]) if 0 <= idx < len(x_grid) else None
                    score_by_sn[sn] = float(max_abs)
                    dev_by_sn[sn] = {"max_abs": float(max_abs), "rms": float(rms), "max_pct": float(max_pct), "rms_pct": float(rms_pct), "x_at_max_abs": x_at}

                scores = [v for v in score_by_sn.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
                mean_score = float(statistics.mean(scores)) if scores else 0.0
                std_score = float(statistics.pstdev(scores)) if len(scores) > 1 else 1.0
                if std_score == 0.0:
                    std_score = 1.0

                for sn in hi:
                    dv = dev_by_sn.get(sn)
                    if not dv:
                        continue
                    z = (float(dv.get("max_abs") or 0.0) - mean_score) / std_score if std_score else 0.0
                    grade = _grade_from_z(z, z_pass, z_watch)
                    grading_rows.append(
                        {
                            "serial": sn,
                            "run": run,
                            "param": y_name,
                            "units": units,
                            "max_pct": dv.get("max_pct"),
                            "rms_pct": dv.get("rms_pct"),
                            "x_at_max_abs": dv.get("x_at_max_abs"),
                            "z": float(z),
                            "grade": grade,
                            "poly_rmse": poly.get("rmse"),
                        }
                    )

                    watch = False
                    if max_abs_thr is not None and float(dv.get("max_abs") or 0.0) >= float(max_abs_thr):
                        watch = True
                    if max_pct_thr is not None and float(dv.get("max_pct") or 0.0) >= float(max_pct_thr):
                        watch = True
                    if rms_pct_thr is not None and float(dv.get("rms_pct") or 0.0) >= float(rms_pct_thr):
                        watch = True
                    if watch:
                        watch_items.append({"serial": sn, "run": run, "param": y_name, "units": units, **dv, "z": float(z), "grade": grade})

                curves_summary.setdefault(run, {})[y_name] = {
                    "x_name": x_name,
                    "units": units,
                    "domain": [float(lo), float(hi_dom)],
                    "grid_points": int(grid_points),
                    "poly": poly,
                    "equation": eqn,
                    "watch_any": any(w.get("run") == run and w.get("param") == y_name for w in watch_items),
                }

        metrics_summary: dict[str, dict[str, dict]] = {}
        if include_metrics and runs and params:
            for run in runs:
                y_cols = _td_list_y_columns(conn, run)
                y_by_norm = {_norm_key(str(c.get("name") or "")): c for c in y_cols if str(c.get("name") or "").strip()}
                for p in params:
                    nk = _norm_key(p)
                    if nk not in y_by_norm:
                        continue
                    y_name = str(y_by_norm[nk].get("name") or "").strip()
                    units = str(y_by_norm[nk].get("units") or "").strip()
                    for st in stats:
                        try:
                            rows = conn.execute(
                                "SELECT serial, value_num FROM td_metrics WHERE run_name=? AND column_name=? AND lower(stat)=?",
                                (run, y_name, st.lower()),
                            ).fetchall()
                        except Exception:
                            rows = []
                        vmap: dict[str, float] = {}
                        for sn, val in rows:
                            s = str(sn or "").strip()
                            fv = _safe_float(val)
                            if s and fv is not None and math.isfinite(float(fv)):
                                vmap[s] = float(fv)
                        metrics_summary.setdefault(run, {}).setdefault(y_name, {"units": units, "stats": {}})
                        metrics_summary[run][y_name]["stats"][st] = vmap

        units_summary = _summarize_units_from_td(conn, runs=runs, params=params)
        max_findings = int(report_opts.get("max_findings") or 12) or 12
        findings = sorted(
            watch_items,
            key=lambda d: (float(d.get("max_pct") or 0.0), abs(float(d.get("z") or 0.0))),
            reverse=True,
        )[:max_findings]

        grade_counts = {"PASS": 0, "WATCH": 0, "FAIL": 0}
        for r in grading_rows:
            g = str(r.get("grade") or "").strip().upper()
            if g in grade_counts:
                grade_counts[g] += 1

        summary_lines = [
            f"Date: {_now_datestr()}",
            f"Project: {proj}",
            f"Workbook: {wb}",
            f"Cache DB: {db_path}",
            f"Runs: {len(runs)}",
            f"Serials: {len(all_serials)}",
            f"Highlighted Serials: {', '.join(hi) if hi else '(none)'}",
            f"Watch Items: {len(watch_items)}",
            f"Grades (highlighted): PASS={grade_counts['PASS']}  WATCH={grade_counts['WATCH']}  FAIL={grade_counts['FAIL']}",
            "",
            "excel_trend_config.json updates:",
            *[f"  {ln}" for ln in str(change_summary).splitlines()[:14]],
        ]

        performance_models: list[dict] = []

        sidecar = {
            "version": 1,
            "generated_date": _now_datestr(),
            "project_dir": str(proj),
            "workbook_path": str(wb),
            "db_path": str(db_path),
            "output_pdf": str(out_pdf),
            "report_config": report_cfg,
            "options": options,
            "highlighted_serials": hi,
            "runs": runs,
            "params": params,
            "statistics": stats,
            "curve_models": curves_summary,
            "watch_items": watch_items,
            "grading": grading_rows,
        }
        sidecar_path = out_pdf.with_suffix(".summary.json")
        # Sidecar written after PDF generation (includes performance models).

        with PdfPages(out_pdf) as pdf:
            pdf.savefig(_figure_text_page("Auto Report — Executive Summary", summary_lines))
            if units_summary:
                pdf.savefig(_figure_table_page("Units Summary", ["Units", "Parameters"], [[u, n] for u, n in units_summary.items()]))
            if findings:
                cols = ["Serial", "Run", "Param", "Max %", "RMS %", "x@max", "Grade"]
                rows = [
                    [
                        f.get("serial"),
                        f.get("run"),
                        f.get("param"),
                        f"{float(f.get('max_pct') or 0.0):.3g}",
                        f"{float(f.get('rms_pct') or 0.0):.3g}",
                        f"{float(f.get('x_at_max_abs') or 0.0):.4g}" if f.get("x_at_max_abs") is not None else "",
                        f.get("grade"),
                    ]
                    for f in findings
                ]
                pdf.savefig(_figure_table_page("Top Findings (Watch Items)", cols, rows))

            if hi:
                by_sn: dict[str, list[dict]] = {}
                for r in grading_rows:
                    by_sn.setdefault(str(r.get("serial") or ""), []).append(r)
                for sn in hi:
                    rows = by_sn.get(sn, [])
                    if not rows:
                        pdf.savefig(_figure_text_page(f"Highlighted Serial — {sn}", ["No data found for this serial in cache."]))
                        continue
                    rows = sorted(rows, key=lambda d: (str(d.get("run") or ""), str(d.get("param") or "")))
                    cols = ["Run", "Param", "Units", "Max %", "RMS %", "x@max", "z", "Grade"]
                    tab = []
                    for r in rows:
                        tab.append(
                            [
                                r.get("run"),
                                r.get("param"),
                                r.get("units"),
                                f"{float(r.get('max_pct') or 0.0):.3g}",
                                f"{float(r.get('rms_pct') or 0.0):.3g}",
                                f"{float(r.get('x_at_max_abs') or 0.0):.4g}" if r.get("x_at_max_abs") is not None else "",
                                f"{float(r.get('z') or 0.0):+.3g}",
                                r.get("grade"),
                            ]
                        )
                    pdf.savefig(_figure_table_page(f"Highlighted Serial — {sn} (Family Grading)", cols, tab))

            for run in runs:
                run_meta = run_by_name.get(run) or {}
                title = str(run_meta.get("display_name") or "").strip() or run
                params_run = curves_summary.get(run, {}) or {}
                cols = ["Parameter", "Units", "x-axis", "Domain", "Poly RMSE", "Equation"]
                rows = []
                for param_name, d in sorted(params_run.items(), key=lambda kv: kv[0].lower()):
                    poly = d.get("poly") or {}
                    rows.append(
                        [
                            param_name,
                            d.get("units") or "",
                            d.get("x_name") or "",
                            "…".join(f"{float(x):.4g}" for x in (d.get("domain") or [])[:2]) if d.get("domain") else "",
                            f"{float(poly.get('rmse') or 0.0):.4g}" if poly.get("rmse") is not None else "",
                            (d.get("equation") or "")[:120],
                        ]
                    )
                if rows:
                    pdf.savefig(_figure_table_page(f"Run Details — {title}", cols, rows))

            import matplotlib.pyplot as plt  # type: ignore

            watch_set = {(str(w.get("run") or ""), str(w.get("param") or ""), str(w.get("serial") or "")) for w in watch_items}
            watch_any_set = {(str(w.get("run") or ""), str(w.get("param") or "")) for w in watch_items}

            for run in runs:
                run_meta = run_by_name.get(run) or {}
                run_title = str(run_meta.get("display_name") or "").strip() or run
                x_name = str(run_meta.get("default_x") or "").strip() or "Time"
                params_run = curves_summary.get(run, {}) or {}
                for param_name, model in sorted(params_run.items(), key=lambda kv: kv[0].lower()):
                    series = _load_curves(conn, run, param_name, x_name)
                    if not series:
                        continue
                    dom = model.get("domain") or []
                    if not isinstance(dom, list) or len(dom) < 2:
                        continue
                    lo = float(dom[0])
                    hi_dom = float(dom[1])
                    x_grid = [lo + (hi_dom - lo) * (i / (grid_points - 1)) for i in range(grid_points)]
                    y_resampled_by_sn = {s.serial: _interp_linear(s.x, s.y, x_grid) for s in series}
                    y_matrix = list(y_resampled_by_sn.values())
                    master_y = _nan_median(y_matrix)
                    std_y = _nan_std(y_matrix)
                    units = str(model.get("units") or "").strip()

                    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.set_title(f"{run_title} — {param_name}")
                    ax.set_xlabel(x_name)
                    ax.set_ylabel(f"{param_name} ({units})" if units else param_name)
                    ax.plot(x_grid, master_y, linewidth=2.2, color="#0f172a", label="Master (median)")
                    try:
                        band_lo = [a - b if (isinstance(a, (int, float)) and not math.isnan(float(a)) and isinstance(b, (int, float)) and not math.isnan(float(b))) else float("nan") for a, b in zip(master_y, std_y)]
                        band_hi = [a + b if (isinstance(a, (int, float)) and not math.isnan(float(a)) and isinstance(b, (int, float)) and not math.isnan(float(b))) else float("nan") for a, b in zip(master_y, std_y)]
                        ax.fill_between(x_grid, band_lo, band_hi, color="#93c5fd", alpha=0.25, label="±1σ band")
                    except Exception:
                        pass

                    if (run, param_name) in watch_any_set and hi:
                        for idx, sn in enumerate(hi):
                            if (run, param_name, sn) not in watch_set:
                                continue
                            yv = y_resampled_by_sn.get(sn)
                            if not yv:
                                continue
                            ax.plot(x_grid, yv, linewidth=1.6, color=colors[idx % len(colors)], label=f"{sn} (watch)")

                    eqn = str(model.get("equation") or "").strip()
                    rmse = (model.get("poly") or {}).get("rmse")
                    notes = []
                    if eqn:
                        notes.append(eqn)
                    if rmse is not None:
                        try:
                            notes.append(f"RMSE: {float(rmse):.4g}")
                        except Exception:
                            pass
                    if notes:
                        ax.text(0.01, 0.01, "\n".join(notes), transform=ax.transAxes, fontsize=8, va="bottom", ha="left", color="#334155")
                    ax.grid(True, alpha=0.25)
                    try:
                        ax.legend(fontsize=8, loc="best")
                    except Exception:
                        pass
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass
                    pdf.savefig(fig)
                    plt.close(fig)

            if include_metrics and metrics_summary:
                watch_any_param = {(str(w.get("run") or ""), str(w.get("param") or "")) for w in watch_items}
                for run in runs:
                    run_meta = run_by_name.get(run) or {}
                    run_title = str(run_meta.get("display_name") or "").strip() or run
                    for param_name, d in sorted((metrics_summary.get(run) or {}).items(), key=lambda kv: kv[0].lower()):
                        stats_map = (d or {}).get("stats") or {}
                        if not isinstance(stats_map, dict) or not stats_map:
                            continue
                        units = str((d or {}).get("units") or "").strip()
                        fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
                        ax = fig.add_subplot(111)
                        ax.set_title(f"{run_title} — Metrics: {param_name}")
                        ax.set_xlabel("Serial Number")
                        ax.set_ylabel(f"Value ({units})" if units else "Value")
                        serials = all_serials
                        x_idx = list(range(len(serials)))
                        plotted_any = False
                        for st in stats:
                            vmap = stats_map.get(st) or {}
                            if not isinstance(vmap, dict):
                                continue
                            yv = [(float(vmap.get(sn)) if isinstance(vmap.get(sn), (int, float)) else float("nan")) for sn in serials]
                            try:
                                ax.plot(x_idx, yv, marker="o", linewidth=1.2, alpha=0.85, label=st)
                                plotted_any = True
                            except Exception:
                                continue
                        if not plotted_any:
                            plt.close(fig)
                            continue
                        if (run, param_name) in watch_any_param and hi:
                            for idx, sn in enumerate(hi):
                                if sn not in serials:
                                    continue
                                xi = serials.index(sn)
                                ax.axvline(xi, color=colors[idx % len(colors)], linewidth=1.1, alpha=0.35)
                        ax.set_xticks(x_idx)
                        ax.set_xticklabels(serials, rotation=45, ha="right", fontsize=7)
                        ax.grid(True, alpha=0.25)
                        try:
                            ax.legend(fontsize=8, loc="best")
                        except Exception:
                            pass
                        try:
                            fig.tight_layout()
                        except Exception:
                            pass
                        pdf.savefig(fig)
                        plt.close(fig)

            # Performance plotters (config-driven X vs Y metrics per serial across runs)
            perf_defs = excel_cfg.get("performance_plotters") if isinstance(excel_cfg, dict) else []
            if isinstance(perf_defs, list) and perf_defs:
                for pd in perf_defs:
                    if not isinstance(pd, dict):
                        continue
                    name = str(pd.get("name") or "Performance").strip() or "Performance"
                    x_spec = pd.get("x") or {}
                    y_spec = pd.get("y") or {}
                    x_target = str((x_spec.get("column") if isinstance(x_spec, dict) else "") or "").strip()
                    y_target = str((y_spec.get("column") if isinstance(y_spec, dict) else "") or "").strip()
                    stats_list = pd.get("stats")
                    if isinstance(stats_list, list) and all(isinstance(s, str) for s in stats_list):
                        stats_list = [str(s).strip().lower() for s in stats_list if str(s).strip()]
                    else:
                        legacy = str((x_spec.get("stat") if isinstance(x_spec, dict) else "mean") or "mean").strip().lower()
                        stats_list = [legacy] if legacy else ["mean"]
                    if not stats_list:
                        stats_list = ["mean"]
                    if not x_target or not y_target:
                        continue

                    require_min_points = int(pd.get("require_min_points") or 2)
                    require_min_points = max(2, require_min_points)

                    fit_cfg = pd.get("fit") or {}
                    fit_degree = int((fit_cfg.get("degree") if isinstance(fit_cfg, dict) else 0) or 0)
                    fit_degree = max(0, fit_degree)
                    fit_norm = bool((fit_cfg.get("normalize_x") if isinstance(fit_cfg, dict) else True))

                    for st in stats_list:
                        # Build per-run metric maps once for this stat.
                        per_run: list[tuple[str, str, dict[str, float], dict[str, float], str, str]] = []
                        for rn in runs:
                            x_col, x_units = _resolve_td_y_col(conn, rn, x_target)
                            y_col, y_units = _resolve_td_y_col(conn, rn, y_target)
                            xmap = _td_metric_map(conn, rn, x_col, st)
                            ymap = _td_metric_map(conn, rn, y_col, st)
                            dn = str((run_by_name.get(rn) or {}).get("display_name") or "").strip()
                            per_run.append((rn, dn or rn, xmap, ymap, x_units, y_units))

                        # Assemble per-serial points across runs + pooled points for master fit.
                        curves: dict[str, list[tuple[float, float, str]]] = {}
                        pooled_x: list[float] = []
                        pooled_y: list[float] = []
                        for sn in all_serials:
                            pts: list[tuple[float, float, str]] = []
                            for _rn, rdisp, xmap, ymap, _xu, _yu in per_run:
                                if sn not in xmap or sn not in ymap:
                                    continue
                                pts.append((float(xmap[sn]), float(ymap[sn]), rdisp))
                            if len(pts) >= require_min_points:
                                pts.sort(key=lambda t: t[0])
                                curves[sn] = pts
                                pooled_x.extend([p[0] for p in pts])
                                pooled_y.extend([p[1] for p in pts])
                        if not curves:
                            continue

                        # Units: prefer first non-empty
                        x_units = next((u for *_rest, u, _ in per_run if str(u).strip()), "")
                        y_units = next((u for *_rest, _, u in per_run if str(u).strip()), "")

                        # Fits (master + highlighted)
                        master_poly: dict = {"degree": int(fit_degree), "coeffs": [], "rmse": None, "x0": None, "sx": None}
                        master_eqn = ""
                        if fit_degree > 0 and pooled_x:
                            try:
                                master_poly = _poly_fit(pooled_x, pooled_y, int(fit_degree), normalize_x=fit_norm)
                                master_eqn = _fmt_equation(master_poly)
                            except Exception:
                                master_poly = {"degree": int(fit_degree), "coeffs": [], "rmse": None, "x0": None, "sx": None}
                                master_eqn = ""

                        highlighted_models: dict[str, dict] = {}
                        for sn in hi:
                            pts = curves.get(sn)
                            if not pts or fit_degree <= 0:
                                continue
                            try:
                                xs = [p[0] for p in pts]
                                ys = [p[1] for p in pts]
                                poly = _poly_fit(xs, ys, int(fit_degree), normalize_x=fit_norm)
                                highlighted_models[sn] = {"poly": poly, "equation": _fmt_equation(poly), "rmse": poly.get("rmse")}
                            except Exception:
                                continue

                        performance_models.append(
                            {
                                "name": name,
                                "x": {"column": x_target},
                                "y": {"column": y_target},
                                "stat": st,
                                "fit": {"degree": int(fit_degree), "normalize_x": bool(fit_norm)},
                                "require_min_points": int(require_min_points),
                                "points_total": int(len(pooled_x)),
                                "serials_curves": int(len(curves)),
                                "master": {"poly": master_poly, "equation": master_eqn, "rmse": master_poly.get("rmse")},
                                "highlighted": highlighted_models,
                            }
                        )

                        # Plot overlay for all serials (faint), with highlighted serials emphasized.
                        fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
                        ax = fig.add_subplot(111)
                        ax.set_title(f"Performance — {name} — {st}")

                        ax.set_xlabel(f"{x_target}.{st}" + (f" ({x_units})" if x_units else ""))
                        ax.set_ylabel(f"{y_target}.{st}" + (f" ({y_units})" if y_units else ""))

                        hi_set = set(hi)
                        # Draw non-highlighted first (gray).
                        for sn, pts in curves.items():
                            if sn in hi_set:
                                continue
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            ax.plot(xs, ys, linewidth=0.9, alpha=0.12, color="#64748b")

                        # Master fit line (optional)
                        if fit_degree > 0 and master_poly.get("coeffs"):
                            try:
                                import numpy as np  # type: ignore

                                x_min = float(min(pooled_x))
                                x_max = float(max(pooled_x))
                                xfit = np.linspace(x_min, x_max, 240)
                                coeffs = master_poly.get("coeffs") or []
                                pfit = np.poly1d(coeffs)
                                if fit_norm:
                                    x0 = float(master_poly.get("x0") or 0.0)
                                    sx = float(master_poly.get("sx") or 1.0) or 1.0
                                    xfit_n = (xfit - x0) / sx
                                else:
                                    xfit_n = xfit
                                yfit = pfit(xfit_n)
                                ax.plot(xfit.tolist(), yfit.tolist(), linestyle="--", linewidth=1.6, alpha=0.65, color="#0f172a", label="master fit")
                            except Exception:
                                pass

                        # Draw highlighted serials in color (and optional fit).
                        for idx, sn in enumerate(hi):
                            pts = curves.get(sn)
                            if not pts:
                                continue
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            color = colors[idx % len(colors)]
                            ax.plot(xs, ys, marker="o", linewidth=2.2, alpha=0.95, color=color, label=sn)
                            for x, y, rdisp in pts:
                                ax.annotate(str(rdisp), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.75, color=color)

                            hm = highlighted_models.get(sn) or {}
                            poly = hm.get("poly") if isinstance(hm, dict) else None
                            if fit_degree > 0 and isinstance(poly, dict) and poly.get("coeffs"):
                                try:
                                    import numpy as np  # type: ignore

                                    x_min = float(min(xs))
                                    x_max = float(max(xs))
                                    xfit = np.linspace(x_min, x_max, 200)
                                    pfit = np.poly1d(poly.get("coeffs") or [])
                                    if fit_norm:
                                        x0 = float(poly.get("x0") or 0.0)
                                        sx = float(poly.get("sx") or 1.0) or 1.0
                                        xfit_n = (xfit - x0) / sx
                                    else:
                                        xfit_n = xfit
                                    yfit = pfit(xfit_n)
                                    ax.plot(xfit.tolist(), yfit.tolist(), linestyle="--", linewidth=1.4, alpha=0.75, color=color)
                                except Exception:
                                    pass

                        # Equation text block
                        notes: list[str] = []
                        if master_eqn:
                            rmse = master_poly.get("rmse")
                            notes.append(
                                f"Master: {master_eqn}"
                                + (f"  RMSE={float(rmse):.4g}" if isinstance(rmse, (int, float)) else "")
                            )
                        if highlighted_models:
                            shown = 0
                            for sn in hi:
                                hm = highlighted_models.get(sn)
                                if not isinstance(hm, dict):
                                    continue
                                eqn = str(hm.get("equation") or "").strip()
                                rmse = hm.get("rmse")
                                if eqn:
                                    notes.append(
                                        f"{sn}: {eqn}"
                                        + (f"  RMSE={float(rmse):.4g}" if isinstance(rmse, (int, float)) else "")
                                    )
                                    shown += 1
                                if shown >= 4 and len(highlighted_models) > shown:
                                    notes.append(f"(+{len(highlighted_models) - shown} more highlighted fits)")
                                    break
                        if notes:
                            try:
                                ax.text(
                                    0.01,
                                    0.99,
                                    "\n".join(notes),
                                    transform=ax.transAxes,
                                    va="top",
                                    ha="left",
                                    fontsize=8,
                                    color="#0f172a",
                                    bbox=dict(
                                        boxstyle="round,pad=0.35",
                                        facecolor="white",
                                        edgecolor="#cbd5e1",
                                        alpha=0.9,
                                    ),
                                )
                            except Exception:
                                pass

                        ax.grid(True, alpha=0.25)
                        try:
                            if hi or master_eqn:
                                ax.legend(fontsize=9, loc="best")
                        except Exception:
                            pass
                        try:
                            fig.tight_layout()
                        except Exception:
                            pass
                        pdf.savefig(fig)
                        plt.close(fig)

        sidecar["performance_models"] = performance_models
        _write_json(sidecar_path, sidecar)

    return {
        "output_pdf": str(out_pdf),
        "summary_json": str(sidecar_path),
        "db_path": str(db_path),
        "runs": runs,
        "params": params,
        "highlighted_serials": hi,
        "watch_items": len(watch_items),
    }
