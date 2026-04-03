from __future__ import annotations

"""
Auto-report generation for Test Data Trend / Analyze.

This module is intentionally imported lazily from backend/UI because it depends
on optional plotting/scientific libraries (matplotlib, numpy).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import math
import sqlite3
import statistics
import textwrap
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


def _fmt_num(v: object, *, sig: int = 4) -> str:
    fv = _safe_float(v)
    if fv is None:
        return "—"
    try:
        x = float(fv)
    except Exception:
        return "—"
    if not math.isfinite(x):
        return "—"
    try:
        s = f"{x:.{max(1, int(sig))}g}"
    except Exception:
        s = str(x)
    return "0" if s in {"-0", "-0.0", "-0.00"} else s


def _norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in str(s or "").strip() if ch.isalnum())


def _td_display_program_title(value: object) -> str:
    return str(value or "").strip() or "Unknown Program"


def _td_serial_value(row: Mapping[str, object] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("serial") or row.get("serial_number") or "").strip()


def _td_compact_filter_value(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        try:
            num = float(value)
        except Exception:
            return ""
        if not math.isfinite(num):
            return ""
        return f"{num:g}"
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        num = float(raw)
    except Exception:
        return raw
    if not math.isfinite(num):
        return raw
    return f"{num:g}"


def _td_control_period_filter_value(row: Mapping[str, object] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return _td_compact_filter_value(row.get("control_period"))


def _td_suppression_voltage_filter_value(row: Mapping[str, object] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    return _td_compact_filter_value(row.get("suppression_voltage"))


def _filter_state_values(filter_state: Mapping[str, object] | None, key: str) -> list[str]:
    if not isinstance(filter_state, Mapping):
        return []
    raw_values = filter_state.get(key) or []
    if not isinstance(raw_values, list):
        return []
    return [str(value).strip() for value in raw_values if str(value).strip()]


def _filter_state_has_key(filter_state: Mapping[str, object] | None, key: str) -> bool:
    return isinstance(filter_state, Mapping) and key in filter_state


def _row_matches_filter_state(row: Mapping[str, object] | None, filter_state: Mapping[str, object] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    selected_programs = set(_filter_state_values(filter_state, "programs"))
    if _filter_state_has_key(filter_state, "programs"):
        if _td_display_program_title(row.get("program_title")) not in selected_programs:
            return False
    selected_serials = set(_filter_state_values(filter_state, "serials"))
    if _filter_state_has_key(filter_state, "serials"):
        serial = _td_serial_value(row)
        if not serial or serial not in selected_serials:
            return False
    selected_control_periods = set(_filter_state_values(filter_state, "control_periods"))
    if _filter_state_has_key(filter_state, "control_periods"):
        control_period = _td_control_period_filter_value(row)
        if not control_period or control_period not in selected_control_periods:
            return False
    selected_suppression = set(_filter_state_values(filter_state, "suppression_voltages"))
    if _filter_state_has_key(filter_state, "suppression_voltages"):
        suppression_voltage = _td_suppression_voltage_filter_value(row)
        if not suppression_voltage or suppression_voltage not in selected_suppression:
            return False
    return True


def _filter_rows_for_filter_state(
    rows: list[dict],
    filter_state: Mapping[str, object] | None,
) -> list[dict]:
    if not isinstance(filter_state, Mapping) or not filter_state:
        return [dict(row) for row in (rows or []) if isinstance(row, dict)]
    return [dict(row) for row in (rows or []) if _row_matches_filter_state(row, filter_state)]


def _selection_matches_observation_row(
    selection: Mapping[str, object] | None,
    row: Mapping[str, object] | None,
) -> bool:
    if not isinstance(selection, Mapping) or not isinstance(row, Mapping):
        return False
    sequence_values: list[str] = []
    raw_sequences = selection.get("member_sequences") or []
    if isinstance(raw_sequences, list):
        sequence_values = [str(value).strip() for value in raw_sequences if str(value).strip()]
    if not sequence_values:
        for candidate in (
            selection.get("source_run_name"),
            selection.get("sequence_name"),
            selection.get("run_name"),
        ):
            text = str(candidate or "").strip()
            if text:
                sequence_values = [text]
                break
    sequence_set = {value.casefold() for value in sequence_values if value}
    row_sequence = str(row.get("source_run_name") or "").strip()
    if sequence_set:
        if not row_sequence or row_sequence.casefold() not in sequence_set:
            return False

    member_programs: list[str] = []
    seen_programs: set[str] = set()
    raw_programs = selection.get("member_programs") or []
    if isinstance(raw_programs, list):
        for value in raw_programs:
            label = _td_display_program_title(value)
            if not label or label.casefold() in seen_programs:
                continue
            seen_programs.add(label.casefold())
            member_programs.append(label)
    if not member_programs and "program_title" in selection:
        member_programs = [_td_display_program_title(selection.get("program_title"))]
    if member_programs:
        if _td_display_program_title(row.get("program_title")) not in set(member_programs):
            return False
    return True


def _resolve_filtered_serials(
    be: Any,
    db_path: Path,
    ordered_serials: list[str],
    options: dict,
) -> list[str]:
    provided = options.get("filtered_serials") or []
    if isinstance(provided, list):
        chosen = {str(value).strip() for value in provided if str(value).strip()}
        if chosen:
            return [serial for serial in ordered_serials if serial in chosen]

    filter_state = options.get("filter_state")
    run_selections = options.get("run_selections") or []
    if not isinstance(filter_state, Mapping):
        filter_state = {}
    if not isinstance(run_selections, list):
        run_selections = []
    if not filter_state and not run_selections:
        return list(ordered_serials)

    try:
        filter_rows = be.td_read_observation_filter_rows_from_cache(db_path)
    except Exception:
        filter_rows = []
    if not isinstance(filter_rows, list) or not filter_rows:
        selected_serials = set(_filter_state_values(filter_state, "serials"))
        if _filter_state_has_key(filter_state, "serials"):
            return [serial for serial in ordered_serials if serial in selected_serials]
        return list(ordered_serials)

    matched_serials: set[str] = set()
    valid_run_selections = [selection for selection in run_selections if isinstance(selection, Mapping)]
    for row in filter_rows:
        if not _row_matches_filter_state(row, filter_state):
            continue
        if valid_run_selections and not any(_selection_matches_observation_row(selection, row) for selection in valid_run_selections):
            continue
        serial = _td_serial_value(row)
        if serial:
            matched_serials.add(serial)
    return [serial for serial in ordered_serials if serial in matched_serials]


def _ceil_div(a: int, b: int) -> int:
    try:
        ai = int(a)
        bi = int(b)
    except Exception:
        return 0
    return int((ai + bi - 1) // bi) if bi else 0


# Backwards-compat: some callers used the non-underscored name.
ceil_div = _ceil_div


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
            "max_pages": 30,
            "metrics_stats": ["median"],
            "metadata_columns": "extended",
            "appendix_include_grade_matrix": True,
            "appendix_include_pass_details": True,
            "plot_only_nonpass": True,
            "max_plots": None,
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


def _selection_observation_filters(selection: dict | None) -> tuple[str, str]:
    if not isinstance(selection, dict):
        return "", ""
    if str(selection.get("mode") or "sequence").strip().lower() != "sequence":
        return "", ""
    program_title = str(selection.get("program_title") or "").strip()
    source_run_name = str(selection.get("source_run_name") or selection.get("sequence_name") or "").strip()
    return program_title, source_run_name


def _selection_for_run(run_name: str, options: dict) -> dict:
    run = str(run_name or "").strip()
    if not run:
        return {}
    run_selections = options.get("run_selections") or []
    if not isinstance(run_selections, list):
        return {}

    best: dict = {}
    for selection in run_selections:
        if not isinstance(selection, dict):
            continue
        members = [str(v).strip() for v in (selection.get("member_runs") or []) if str(v).strip()]
        sel_run = str(selection.get("run_name") or "").strip()
        if run not in members and run != sel_run:
            continue
        if str(selection.get("mode") or "sequence").strip().lower() == "sequence":
            return dict(selection)
        if not best:
            best = dict(selection)
    return best


def _td_serial_metadata_by_serial(rows: list[dict]) -> dict[str, dict]:
    by_sn: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sn = str(row.get("serial") or row.get("serial_number") or "").strip()
        if sn and sn not in by_sn:
            by_sn[sn] = dict(row)
    return by_sn


def _td_order_metric_serials(labels: list[str], serial_rows: list[dict]) -> list[str]:
    meta_by_sn = _td_serial_metadata_by_serial(serial_rows)
    serials: list[str] = []
    seen: set[str] = set()
    for raw_sn in labels or []:
        sn = str(raw_sn or "").strip()
        if not sn or sn in seen:
            continue
        seen.add(sn)
        serials.append(sn)

    def _sort_key(sn: str) -> tuple[int, str, str]:
        row = meta_by_sn.get(sn) or {}
        program = str(row.get("program_title") or "").strip() or "Unknown Program"
        return (
            1 if program == "Unknown Program" else 0,
            program.casefold(),
            sn.casefold(),
        )

    return sorted(serials, key=_sort_key)


def _series_rows_to_metric_map(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sn = str(row.get("serial") or "").strip()
        val = row.get("value_num")
        if not sn or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        out[sn] = float(val)
    return out


def _resolve_td_y_col_from_rows(rows: list[dict], target: str) -> tuple[str, str]:
    tgt = str(target or "").strip()
    if not tgt:
        return "", ""
    want = _norm_key(tgt)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name and _norm_key(name) == want:
            return name, str(row.get("units") or "").strip()
    return tgt, ""


def _resolve_curve_x_key(be: Any, db_path: Path, run_name: str, x_label: str) -> str:
    run = str(run_name or "").strip()
    label = str(x_label or "").strip()
    if not run or not label:
        return label

    def _norm_name(value: object) -> str:
        return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())

    time_norms = {_norm_name(x) for x in ("time", "time_s", "time(sec)", "time(s)", "time (s)", "time_sec", "times")}
    pulse_norms = {_norm_name(x) for x in ("pulse number", "pulse#", "pulse #", "pulse_number", "pulsenumber", "cycle")}

    try:
        xs = be.td_list_x_columns(db_path, run)
    except Exception:
        xs = []
    xs = [str(x or "").strip() for x in (xs or []) if str(x or "").strip()]
    if label in xs:
        return label

    by_norm: dict[str, str] = {}
    for x in xs:
        nk = _norm_name(x)
        if nk and nk not in by_norm:
            by_norm[nk] = x

    want = _norm_name(label)
    if want == _norm_name("excel_row"):
        for pref in ("Time", "Time (s)", "Time(s)", "time_s", "time"):
            resolved = by_norm.get(_norm_name(pref))
            if resolved:
                return resolved
        for pref in ("Pulse Number", "Pulse #", "cycle", "Cycle", "pulse_number", "pulsenumber"):
            resolved = by_norm.get(_norm_name(pref))
            if resolved:
                return resolved
    if want in time_norms:
        for pref in ("Time", "Time (s)", "Time(s)", "time_s", "time"):
            resolved = by_norm.get(_norm_name(pref))
            if resolved:
                return resolved
    if want in pulse_norms:
        for pref in ("Pulse Number", "Pulse #", "cycle", "Cycle", "pulse_number", "pulsenumber"):
            resolved = by_norm.get(_norm_name(pref))
            if resolved:
                return resolved
    return label


def _load_metric_series_for_selection(
    be: Any,
    db_path: Path,
    run_name: str,
    column_name: str,
    stat: str,
    *,
    selection: dict | None = None,
    control_period_filter: object = None,
    filter_state: Mapping[str, object] | None = None,
) -> list[dict]:
    program_title, source_run_name = _selection_observation_filters(selection)
    try:
        rows = be.td_load_metric_series(
            db_path,
            run_name,
            column_name,
            stat,
            program_title=(program_title or None),
            source_run_name=(source_run_name or None),
            control_period_filter=control_period_filter,
        )
    except Exception:
        rows = []
    return _filter_rows_for_filter_state(rows, filter_state)


def _load_metric_map_for_selection(
    be: Any,
    db_path: Path,
    run_name: str,
    column_name: str,
    stat: str,
    *,
    selection: dict | None = None,
    control_period_filter: object = None,
    filter_state: Mapping[str, object] | None = None,
) -> dict[str, float]:
    rows = _load_metric_series_for_selection(
        be,
        db_path,
        run_name,
        column_name,
        stat,
        selection=selection,
        control_period_filter=control_period_filter,
        filter_state=filter_state,
    )
    return _series_rows_to_metric_map(rows)


def _load_perf_equation_metric_series(
    be: Any,
    db_path: Path,
    run_name: str,
    column_name: str,
    stat: str,
    *,
    selection: dict | None = None,
    control_period_filter: object = None,
    filter_state: Mapping[str, object] | None = None,
) -> list[dict]:
    st = str(stat or "").strip().lower()
    if not st:
        return []
    if st in {"min_3sigma", "max_3sigma"}:
        resolver = getattr(be, "td_perf_mean_3sigma_value", None)
        if not callable(resolver):
            return []
        mean_rows = _load_metric_series_for_selection(
            be,
            db_path,
            run_name,
            column_name,
            "mean",
            selection=selection,
            control_period_filter=control_period_filter,
            filter_state=filter_state,
        )
        std_rows = _load_metric_series_for_selection(
            be,
            db_path,
            run_name,
            column_name,
            "std",
            selection=selection,
            control_period_filter=control_period_filter,
            filter_state=filter_state,
        )
        mean_by_obs = {
            str(row.get("observation_id") or "").strip(): dict(row)
            for row in mean_rows
            if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
        }
        std_by_obs = {
            str(row.get("observation_id") or "").strip(): dict(row)
            for row in std_rows
            if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
        }
        out: list[dict] = []
        for obs_id in sorted(set(mean_by_obs.keys()) | set(std_by_obs.keys())):
            mean_row = mean_by_obs.get(obs_id) or {}
            std_row = std_by_obs.get(obs_id) or {}
            try:
                val = resolver(
                    {
                        "mean": (mean_row or {}).get("value_num"),
                        "std": (std_row or {}).get("value_num"),
                    },
                    st,
                )
            except Exception:
                val = None
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                continue
            base_row = dict(mean_row or std_row)
            base_row["value_num"] = float(val)
            out.append(base_row)
        return out
    return _load_metric_series_for_selection(
        be,
        db_path,
        run_name,
        column_name,
        st,
        selection=selection,
        control_period_filter=control_period_filter,
        filter_state=filter_state,
    )


def _curve_rows_to_series(rows: list[dict]) -> list[CurveSeries]:
    out: list[CurveSeries] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sn = str(row.get("serial") or "").strip()
        xs = row.get("x")
        ys = row.get("y")
        if not sn or not isinstance(xs, list) or not isinstance(ys, list) or not xs or not ys:
            continue
        pts = min(len(xs), len(ys))
        x_vals: list[float] = []
        y_vals: list[float] = []
        for idx in range(pts):
            xf = _safe_float(xs[idx])
            yf = _safe_float(ys[idx])
            if xf is None or yf is None:
                continue
            x_vals.append(float(xf))
            y_vals.append(float(yf))
        if len(x_vals) < 2 or len(y_vals) < 2:
            continue
        out.append(CurveSeries(serial=sn, x=x_vals, y=y_vals))
    return out


def _load_curves_for_selection(
    be: Any,
    db_path: Path,
    run_name: str,
    y_name: str,
    x_name: str,
    *,
    selection: dict | None = None,
    serials: list[str] | None = None,
    filter_state: Mapping[str, object] | None = None,
) -> list[CurveSeries]:
    program_title, source_run_name = _selection_observation_filters(selection)
    try:
        rows = be.td_load_curves(
            db_path,
            run_name,
            y_name,
            x_name,
            serials=serials,
            program_title=(program_title or None),
            source_run_name=(source_run_name or None),
        )
    except Exception:
        rows = []
    return _curve_rows_to_series(_filter_rows_for_filter_state(rows, filter_state))


def _read_gui_source_metadata(be: Any, workbook_path: Path) -> tuple[dict[str, dict[str, str]], str]:
    reader = getattr(be, "td_read_sources_metadata", None)
    loader = getattr(be, "_load_td_source_metadata", None)
    if not callable(reader):
        return {}, "GUI source metadata unavailable (td_read_sources_metadata missing)."
    wb_path = Path(workbook_path).expanduser()
    try:
        source_rows = reader(wb_path)
    except Exception as exc:
        return {}, f"GUI source metadata unavailable ({exc})."
    if not isinstance(source_rows, list) or not source_rows:
        return {}, "GUI source metadata unavailable (Sources sheet empty)."

    meta_by_sn: dict[str, dict[str, str]] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        sn = str(row.get("serial") or row.get("serial_number") or "").strip()
        if not sn:
            continue
        payload = dict(row)
        if callable(loader):
            try:
                loaded = loader(wb_path, row)
                if isinstance(loaded, dict):
                    payload = dict(loaded)
            except Exception:
                payload = dict(row)
        meta_by_sn[sn] = {
            "program_title": str(payload.get("program_title") or "").strip(),
            "asset_type": str(payload.get("asset_type") or "").strip(),
            "asset_specific_type": str(payload.get("asset_specific_type") or "").strip(),
            "vendor": str(payload.get("vendor") or "").strip(),
            "acceptance_test_plan_number": str(payload.get("acceptance_test_plan_number") or "").strip(),
            "part_number": str(payload.get("part_number") or "").strip(),
            "revision": str(payload.get("revision") or "").strip(),
            "test_date": str(payload.get("test_date") or "").strip(),
            "report_date": str(payload.get("report_date") or "").strip(),
            "document_type": str(payload.get("document_type") or "").strip(),
            "document_type_acronym": str(payload.get("document_type_acronym") or "").strip(),
            "similarity_group": str(payload.get("similarity_group") or "").strip(),
        }
    if not meta_by_sn:
        return {}, "GUI source metadata unavailable (resolved metadata empty)."
    return meta_by_sn, ""


def _series_by_observation(rows: list[dict], serial_set: set[str] | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        obs_id = str(row.get("observation_id") or "").strip()
        sn = str(row.get("serial") or "").strip()
        val = row.get("value_num")
        if not obs_id or not sn:
            continue
        if serial_set is not None and sn not in serial_set:
            continue
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue
        out[obs_id] = dict(row)
    return out


def _perf_observation_label(run_display: str, run_name: str, row: dict) -> str:
    parts: list[str] = [str(run_display or run_name).strip() or str(run_name or "").strip()]
    for key in ("program_title", "source_run_name"):
        value = str((row or {}).get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    return " | ".join([part for part in parts if str(part).strip()])


def _collect_performance_curves_for_stat(
    *,
    be: Any,
    db_path: Path,
    conn: sqlite3.Connection,
    run_by_name: dict[str, dict],
    runs: list[str],
    serials: list[str],
    x_target: str,
    y_target: str,
    stat: str,
    options: dict,
    require_min_points: int,
    control_period_filter: object = None,
    filter_state: Mapping[str, object] | None = None,
) -> tuple[dict[str, list[tuple[float, float, str]]], list[float], list[float], str, str]:
    serial_set = {str(sn).strip() for sn in serials if str(sn).strip()}
    per_run: list[tuple[str, str, dict[str, dict], dict[str, dict], str, str]] = []
    for rn in runs:
        run_selection = _selection_for_run(rn, options)
        metric_cols = []
        try:
            metric_cols = be.td_list_metric_y_columns(db_path, rn)
        except Exception:
            metric_cols = []
        if not metric_cols:
            metric_cols = _td_list_y_columns(conn, rn)
        x_col, x_units = _resolve_td_y_col_from_rows(metric_cols, x_target)
        y_col, y_units = _resolve_td_y_col_from_rows(metric_cols, y_target)
        if not x_col or not y_col:
            continue
        x_rows = _load_perf_equation_metric_series(
            be,
            db_path,
            rn,
            x_col,
            stat,
            selection=run_selection,
            control_period_filter=control_period_filter,
            filter_state=filter_state,
        )
        y_rows = _load_perf_equation_metric_series(
            be,
            db_path,
            rn,
            y_col,
            stat,
            selection=run_selection,
            control_period_filter=control_period_filter,
            filter_state=filter_state,
        )
        x_map = _series_by_observation(x_rows, serial_set)
        y_map = _series_by_observation(y_rows, serial_set)
        if not x_map or not y_map:
            continue
        run_display = str((run_by_name.get(rn) or {}).get("display_name") or "").strip() or rn
        per_run.append((rn, run_display, x_map, y_map, x_units, y_units))

    curves: dict[str, list[tuple[float, float, str]]] = {}
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for sn in serials:
        pts: list[tuple[float, float, str]] = []
        for rn, run_display, x_map, y_map, _xu, _yu in per_run:
            obs_ids = sorted(set(x_map.keys()) & set(y_map.keys()))
            for obs_id in obs_ids:
                row_x = x_map.get(obs_id) or {}
                row_y = y_map.get(obs_id) or {}
                if str(row_y.get("serial") or row_x.get("serial") or "").strip() != sn:
                    continue
                pts.append(
                    (
                        float(row_x.get("value_num") or 0.0),
                        float(row_y.get("value_num") or 0.0),
                        _perf_observation_label(run_display, rn, row_y or row_x),
                    )
                )
        if len(pts) >= require_min_points:
            pts.sort(key=lambda t: t[0])
            curves[sn] = pts
            pooled_x.extend([p[0] for p in pts])
            pooled_y.extend([p[1] for p in pts])

    x_units = next((str(xu).strip() for *_rest, xu, _yu in per_run if str(xu).strip()), "")
    y_units = next((str(yu).strip() for *_rest, _xu, yu in per_run if str(yu).strip()), "")
    return curves, pooled_x, pooled_y, x_units, y_units


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


def _overall_cert_status(grades: list[str]) -> str:
    gs = [str(g or "").strip().upper() for g in (grades or [])]
    if any(g == "FAIL" for g in gs):
        return "FAILED"
    if any(g == "WATCH" for g in gs):
        return "WATCH"
    if any(g == "NO_DATA" for g in gs) or not gs:
        return "NO_DATA"
    return "CERTIFIED"


def _resolve_selected_runs(run_rows: list[dict], options: dict) -> list[str]:
    run_by_name = {str(r.get("run_name") or ""): r for r in (run_rows or []) if str(r.get("run_name") or "")}
    runs: list[str] = []

    run_selections = options.get("run_selections") or []
    if isinstance(run_selections, list):
        seen: set[str] = set()
        for selection in run_selections:
            if not isinstance(selection, dict):
                continue
            members = selection.get("member_runs") or []
            if isinstance(members, list):
                for run in members:
                    rn = str(run or "").strip()
                    if not rn or rn in seen:
                        continue
                    seen.add(rn)
                    runs.append(rn)
            rn = str(selection.get("run_name") or "").strip()
            if rn and rn not in seen:
                seen.add(rn)
                runs.append(rn)

    if not runs:
        selected_runs = options.get("runs") or []
        runs = [str(r).strip() for r in selected_runs if str(r).strip()] if isinstance(selected_runs, list) else []
    if not runs:
        runs = [str(r.get("run_name") or "").strip() for r in (run_rows or []) if str(r.get("run_name") or "").strip()]
    return [r for r in runs if r in run_by_name]


def _resolve_selected_params(
    be_or_conn: Any,
    db_path: Path | None = None,
    conn: sqlite3.Connection | None = None,
    *,
    runs: list[str],
    options: dict,
) -> list[str]:
    selected_params = options.get("params") or []
    params = [str(p).strip() for p in selected_params if str(p).strip()] if isinstance(selected_params, list) else []
    if params:
        return params

    be = be_or_conn
    if conn is None and isinstance(be_or_conn, sqlite3.Connection):
        conn = be_or_conn
        be = None

    # Auto-detect from the same raw-curve dataset that backs the GUI Curves tab.
    def _norm_name(value: object) -> str:
        return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())

    x_exclude_norms = {
        _norm_name(x)
        for x in (
            "time",
            "time_s",
            "time(sec)",
            "time(s)",
            "time (s)",
            "time_sec",
            "times",
            "pulse number",
            "pulse#",
            "pulse #",
            "pulse_number",
            "pulsenumber",
            "cycle",
            "excel_row",
        )
    }
    seen: set[str] = set()
    auto: list[str] = []
    for run in runs:
        cols: list[dict] = []
        if be is not None and db_path is not None:
            try:
                cols = be.td_list_raw_y_columns(db_path, run)
            except Exception:
                cols = []
            if not cols:
                try:
                    cols = be.td_list_curve_y_columns(db_path, run)
                except Exception:
                    cols = []
        if not cols and conn is not None:
            cols = _td_list_y_columns(conn, run)
        for c in cols:
            name = str(c.get("name") or "").strip()
            if not name:
                continue
            nk = _norm_key(name)
            if _norm_name(name) in x_exclude_norms:
                continue
            if nk in seen:
                continue
            seen.add(nk)
            auto.append(name)
    return sorted(auto, key=lambda s: s.lower())


def _finding_sort_key(r: dict) -> tuple[int, float, float]:
    g = str(r.get("grade") or "").strip().upper()
    gk = 0 if g == "FAIL" else 1
    try:
        z = abs(float(r.get("z") or 0.0))
    except Exception:
        z = 0.0
    try:
        mp = float(r.get("max_pct") or 0.0)
    except Exception:
        mp = 0.0
    return (gk, -z, -mp)


def _build_chart_specs(
    *,
    run_param_pairs: list[tuple[str, str]],
    nonpass_findings: list[dict],
    max_plots: int | None = None,
) -> list[tuple[tuple[int, float, float], str, str]]:
    nonpass_run_param = {(str(r.get("run") or ""), str(r.get("param") or "")) for r in (nonpass_findings or [])}
    out: list[tuple[tuple[int, float, float], str, str]] = []
    for run, param in (run_param_pairs or []):
        if (run, param) not in nonpass_run_param:
            continue
        rows = [rr for rr in (nonpass_findings or []) if str(rr.get("run") or "") == run and str(rr.get("param") or "") == param]
        worst_grade = "WATCH"
        best_abs_z = 0.0
        best_max_pct = 0.0
        for rr in rows:
            g = str(rr.get("grade") or "").strip().upper()
            if g == "FAIL":
                worst_grade = "FAIL"
            try:
                best_abs_z = max(best_abs_z, abs(float(rr.get("z") or 0.0)))
            except Exception:
                pass
            try:
                best_max_pct = max(best_max_pct, float(rr.get("max_pct") or 0.0))
            except Exception:
                pass
        gk = 0 if worst_grade == "FAIL" else 1
        out.append(((gk, -best_abs_z, -best_max_pct), run, param))
    out.sort(key=lambda t: t[0])
    if isinstance(max_plots, int) and max_plots >= 0:
        out = out[:max_plots]
    return out


def _plan_page_selections(
    *,
    max_pages: int,
    appendix_include_grade_matrix: bool,
    appendix_include_pass_details: bool,
    include_metrics: bool,
    grading_rows: list[dict],
    run_param_pairs: list[tuple[str, str]],
    serials_nonpass_sorted: list[str],
    perf_defs_all: list[dict],
    run_details_all: list[str],
    chart_specs_all: list[tuple[tuple[int, float, float], str, str]],
    metrics_pairs_all: list[tuple[str, ...]],
    metrics_pairs_nonpass: list[tuple[str, ...]],
) -> dict:
    omitted_items: list[str] = []

    matrix_pages = _ceil_div(len(run_param_pairs), 30) if appendix_include_grade_matrix else 0
    by_serial_pages = _ceil_div(len(serials_nonpass_sorted), 2)
    perf_defs_sel = list(perf_defs_all or [])
    run_details_sel = list(run_details_all or [])
    serials_sel = list(serials_nonpass_sorted or [])

    perf_pages = len(perf_defs_sel) * 2
    base_pages = 3 + by_serial_pages + 1 + perf_pages + len(run_details_sel) + matrix_pages
    if base_pages > max_pages:
        while base_pages > max_pages and perf_defs_sel:
            perf_defs_sel.pop()
            perf_pages = len(perf_defs_sel) * 2
            base_pages = 3 + by_serial_pages + 1 + perf_pages + len(run_details_sel) + matrix_pages
            omitted_items.append("Performance equations: truncated to meet page cap.")
        while base_pages > max_pages and run_details_sel:
            dropped = run_details_sel.pop()
            base_pages -= 1
            omitted_items.append(f"Run details omitted: {dropped}")
        if base_pages > max_pages and by_serial_pages:
            max_by_serial_pages = max_pages - (3 + 1 + perf_pages + len(run_details_sel) + matrix_pages)
            max_by_serial_pages = max(0, int(max_by_serial_pages))
            keep_serials = int(max_by_serial_pages) * 2
            if keep_serials < len(serials_sel):
                serials_sel = serials_sel[:keep_serials]
                omitted_items.append("Non-PASS by-serial: truncated to meet page cap.")
            by_serial_pages = _ceil_div(len(serials_sel), 2)
            base_pages = 3 + by_serial_pages + 1 + perf_pages + len(run_details_sel) + matrix_pages

    include_deviations = bool(appendix_include_pass_details and grading_rows)
    deviations_pages = 1 if include_deviations else 0

    metrics_sel = list(metrics_pairs_all) if include_metrics else []
    if base_pages + deviations_pages + len(metrics_sel) > max_pages and deviations_pages:
        include_deviations = False
        deviations_pages = 0
        omitted_items.append("Appendix: full deviations table omitted to meet page cap.")
    if base_pages + deviations_pages + len(metrics_sel) > max_pages and metrics_sel:
        metrics_sel = list(metrics_pairs_nonpass)
        omitted_items.append("Metrics: PASS-only run/param pages omitted to meet page cap.")
    if base_pages + deviations_pages + len(metrics_sel) > max_pages and metrics_sel:
        metrics_sel = []
        omitted_items.append("Metrics: omitted remaining pages to meet page cap.")

    remaining_for_charts = max_pages - (base_pages + deviations_pages + len(metrics_sel))
    charts_sel = [t[1:] for t in (chart_specs_all or [])[: max(0, remaining_for_charts)]]
    if len(charts_sel) < len(chart_specs_all or []):
        omitted_items.append(f"Charts: omitted {len(chart_specs_all or []) - len(charts_sel)} lower-severity plots to meet page cap.")

    include_omitted_page = bool(omitted_items)
    if include_omitted_page:
        total_no_omitted = base_pages + deviations_pages + len(metrics_sel) + len(charts_sel)
        if total_no_omitted >= max_pages:
            if charts_sel:
                charts_sel.pop()
            elif metrics_sel:
                metrics_sel.pop()
            elif deviations_pages:
                deviations_pages = 0
                include_deviations = False
            include_omitted_page = True

    return {
        "perf_defs_sel": perf_defs_sel,
        "run_details_sel": run_details_sel,
        "serials_nonpass_sorted": serials_sel,
        "include_deviations": include_deviations,
        "deviations_pages": int(deviations_pages),
        "metrics_sel": metrics_sel,
        "charts_sel": charts_sel,
        "include_omitted_page": include_omitted_page,
        "omitted_items": omitted_items,
    }


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


TD_SOURCE_METADATA_FIELDS = (
    "program_title",
    "asset_type",
    "asset_specific_type",
    "vendor",
    "acceptance_test_plan_number",
    "part_number",
    "revision",
    "test_date",
    "report_date",
    "document_type",
    "document_type_acronym",
    "similarity_group",
)


def _read_cached_source_metadata(conn: sqlite3.Connection) -> tuple[dict[str, dict[str, str]], str]:
    try:
        rows = conn.execute(
            """
            SELECT
                serial,
                program_title,
                asset_type,
                asset_specific_type,
                vendor,
                acceptance_test_plan_number,
                part_number,
                revision,
                test_date,
                report_date,
                document_type,
                document_type_acronym,
                similarity_group
            FROM td_source_metadata
            ORDER BY serial
            """
        ).fetchall()
    except Exception:
        return {}, "Project cache metadata unavailable (td_source_metadata missing)."

    meta_by_sn: dict[str, dict[str, str]] = {}
    for row in rows:
        sn = str(row[0] or "").strip()
        if not sn:
            continue
        meta_by_sn[sn] = {
            key: str(row[idx + 1] or "").strip()
            for idx, key in enumerate(TD_SOURCE_METADATA_FIELDS)
        }
    if not meta_by_sn:
        return {}, "Project cache metadata unavailable (td_source_metadata empty)."
    return meta_by_sn, ""


def _read_workbook_metadata(workbook_path: Path) -> tuple[dict[str, dict[str, str]], str]:
    """
    Best-effort read of the project's trending workbook metadata sheet.

    Returns:
      (metadata_by_serial, note)
    """
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception:
        return {}, "Workbook metadata unavailable (openpyxl not installed)."

    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        return {}, f"Workbook metadata unavailable (missing workbook: {wb_path})."

    try:
        wb = load_workbook(str(wb_path), data_only=True, read_only=True)
    except Exception as exc:
        return {}, f"Workbook metadata unavailable (failed to load workbook: {exc})."

    try:
        def _read_metadata_sheet(sheet) -> tuple[dict[str, dict[str, str]], str]:
            rows = sheet.iter_rows(values_only=True)
            try:
                headers = next(rows)
            except StopIteration:
                return {}, "Workbook metadata unavailable (empty 'metadata' sheet)."

            header_norm_to_idx: dict[str, int] = {}
            for idx, h in enumerate(headers or []):
                hn = _norm_key("" if h is None else str(h))
                if hn and hn not in header_norm_to_idx:
                    header_norm_to_idx[hn] = int(idx)

            def _get(row: tuple, *header_aliases: str) -> str:
                for ha in header_aliases:
                    idx = header_norm_to_idx.get(_norm_key(ha))
                    if idx is None or idx >= len(row):
                        continue
                    v = row[idx]
                    s = "" if v is None else str(v)
                    if s.strip():
                        return s.strip()
                return ""

            meta_by_sn: dict[str, dict[str, str]] = {}
            for r in rows:
                if not r:
                    continue
                sn = _get(r, "Serial Number", "Serial", "SN")
                if not sn:
                    continue
                if sn not in meta_by_sn:
                    meta_by_sn[sn] = {
                        "program_title": _get(r, "Program"),
                        "asset_type": _get(r, "Asset Type"),
                        "asset_specific_type": _get(r, "Asset Specific Type"),
                        "vendor": _get(r, "Vendor"),
                        "acceptance_test_plan_number": _get(r, "Acceptance Test Plan", "Acceptance Test Plan Number"),
                        "part_number": _get(r, "Part Number", "PN"),
                        "revision": _get(r, "Revision", "Revision Letter"),
                        "test_date": _get(r, "Test Date"),
                        "report_date": _get(r, "Report Date"),
                        "document_type": _get(r, "Document Type"),
                        "document_type_acronym": _get(r, "Document Acronym", "Document Type Acronym"),
                        "similarity_group": _get(r, "Similarity Group"),
                    }
            return meta_by_sn, ""

        def _read_master_sheet_metadata(sheet) -> dict[str, dict[str, str]]:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                return {}

            headers = rows[0] or ()
            header_norm_to_idx: dict[str, int] = {}
            for idx, h in enumerate(headers):
                hn = _norm_key("" if h is None else str(h))
                if hn and hn not in header_norm_to_idx:
                    header_norm_to_idx[hn] = int(idx)

            data_group_idx = header_norm_to_idx.get(_norm_key("Data Group"))
            term_idx = header_norm_to_idx.get(_norm_key("Term"))
            term_label_idx = header_norm_to_idx.get(_norm_key("Term Label"))
            max_idx = header_norm_to_idx.get(_norm_key("Max"))
            if data_group_idx is None or max_idx is None:
                return {}

            field_by_term = {
                _norm_key("Program"): "program_title",
                _norm_key("Asset Type"): "asset_type",
                _norm_key("Asset Specific Type"): "asset_specific_type",
                _norm_key("Vendor"): "vendor",
                _norm_key("Acceptance Test Plan"): "acceptance_test_plan_number",
                _norm_key("Acceptance Test Plan Number"): "acceptance_test_plan_number",
                _norm_key("Part Number"): "part_number",
                _norm_key("PN"): "part_number",
                _norm_key("Revision"): "revision",
                _norm_key("Revision Letter"): "revision",
                _norm_key("Test Date"): "test_date",
                _norm_key("Report Date"): "report_date",
                _norm_key("Document Type"): "document_type",
                _norm_key("Document Acronym"): "document_type_acronym",
                _norm_key("Document Type Acronym"): "document_type_acronym",
                _norm_key("Similarity Group"): "similarity_group",
            }

            serial_cols: list[tuple[int, str]] = []
            for idx in range(int(max_idx) + 1, len(headers)):
                sn = "" if headers[idx] is None else str(headers[idx]).strip()
                if sn:
                    serial_cols.append((idx, sn))
            if not serial_cols:
                return {}

            meta_by_sn: dict[str, dict[str, str]] = {}
            for r in rows[1:]:
                if not r:
                    continue
                dg = ""
                if data_group_idx < len(r):
                    dg = "" if r[data_group_idx] is None else str(r[data_group_idx]).strip()
                if _norm_key(dg) != _norm_key("Metadata"):
                    continue

                term = ""
                if term_idx is not None and term_idx < len(r):
                    term = "" if r[term_idx] is None else str(r[term_idx]).strip()
                if not term and term_label_idx is not None and term_label_idx < len(r):
                    term = "" if r[term_label_idx] is None else str(r[term_label_idx]).strip()
                field = field_by_term.get(_norm_key(term))
                if not field:
                    continue

                for col_idx, sn in serial_cols:
                    if col_idx >= len(r):
                        continue
                    val = "" if r[col_idx] is None else str(r[col_idx]).strip()
                    if sn not in meta_by_sn:
                        meta_by_sn[sn] = {}
                    if val or field not in meta_by_sn[sn]:
                        meta_by_sn[sn][field] = val
            return meta_by_sn

        for name in wb.sheetnames:
            if _norm_key(name) == _norm_key("metadata"):
                return _read_metadata_sheet(wb[name])

        for name in wb.sheetnames:
            meta_by_sn = _read_master_sheet_metadata(wb[name])
            if meta_by_sn:
                return meta_by_sn, ""

        return {}, "Workbook metadata unavailable (no metadata sheet or metadata rows found)."
    finally:
        try:
            wb.close()
        except Exception:
            pass


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


def _figure_table_page2(
    title: str,
    columns: list[str],
    rows: list[list[object]],
    *,
    landscape: bool = False,
    font_size: int = 8,
    title_size: int = 14,
    scale_y: float = 1.2,
    col_widths: list[float] | None = None,
):
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(11.0, 8.5) if landscape else (8.5, 11.0), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=title_size, fontweight="bold", pad=12)
    cell_text = [[("" if v is None else str(v)) for v in r] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="left", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(int(font_size))
    table.scale(1.0, float(scale_y))
    try:
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f1f5f9")
            if r % 2 == 0 and r != 0:
                cell.set_facecolor("#fafafa")
            if col_widths and c < len(col_widths):
                try:
                    cell.set_width(float(col_widths[c]))
                except Exception:
                    pass
    except Exception:
        pass
    return fig


def _figure_two_tables_page(
    title: str,
    *,
    left_title: str,
    left_columns: list[str],
    left_rows: list[list[object]],
    right_title: str,
    right_columns: list[str],
    right_rows: list[list[object]],
):
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=15, fontweight="bold", va="top")

    ax_l = fig.add_axes([0.06, 0.10, 0.42, 0.80])
    ax_r = fig.add_axes([0.52, 0.10, 0.42, 0.80])
    for ax, sub_title in ((ax_l, left_title), (ax_r, right_title)):
        ax.axis("off")
        ax.set_title(sub_title, loc="left", fontsize=11, fontweight="bold", pad=6)

    def _add_table(ax, cols, rows):
        cell_text = [[("" if v is None else str(v)) for v in r] for r in rows]
        tab = ax.table(cellText=cell_text, colLabels=cols, cellLoc="left", loc="upper left")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7)
        tab.scale(1.0, 1.15)
        try:
            for (r, _c), cell in tab.get_celld().items():
                if r == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#f1f5f9")
                if r % 2 == 0 and r != 0:
                    cell.set_facecolor("#fafafa")
        except Exception:
            pass

    _add_table(ax_l, left_columns, left_rows)
    _add_table(ax_r, right_columns, right_rows)
    return fig


def _wrap_cell(s: str, *, max_chars: int, max_lines: int) -> str:
    txt = str(s or "").strip()
    if not txt:
        return ""
    wrapped = textwrap.wrap(txt, width=max_chars) or [txt]
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        if wrapped:
            wrapped[-1] = wrapped[-1].rstrip()
            if not wrapped[-1].endswith("…"):
                wrapped[-1] = (wrapped[-1][: max(0, max_chars - 1)]).rstrip() + "…"
    return "\n".join(wrapped)


def _figure_performance_equation_page(
    *,
    title: str,
    x_label: str,
    y_label: str,
    curves: dict[str, list[tuple[float, float, str]]],
    highlighted_serials: list[str],
    highlighted_models: dict[str, dict],
    master_poly: dict,
    master_eqn: str,
    fit_norm: bool,
    colors: list[str],
):
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=14, fontweight="bold", va="top")

    ax = fig.add_axes([0.06, 0.12, 0.60, 0.76])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    hi_set = {str(sn).strip() for sn in highlighted_serials if str(sn).strip()}
    pooled_x: list[float] = []

    for sn, pts in curves.items():
        if sn in hi_set:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pooled_x.extend(xs)
        ax.plot(xs, ys, linewidth=0.9, alpha=0.12, color="#64748b")

    for idx, sn in enumerate(highlighted_serials):
        pts = curves.get(sn)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pooled_x.extend(xs)
        color = colors[idx % len(colors)] if colors else "#2563eb"
        ax.plot(xs, ys, marker="o", linewidth=2.1, alpha=0.95, color=color, label=sn)
        for x, y, run_label in pts:
            ax.annotate(str(run_label), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.75, color=color)

        hm = highlighted_models.get(sn) or {}
        poly = hm.get("poly") if isinstance(hm, dict) else None
        if isinstance(poly, dict) and poly.get("coeffs"):
            try:
                import numpy as np  # type: ignore

                xfit = np.linspace(float(min(xs)), float(max(xs)), 200)
                pfit = np.poly1d(poly.get("coeffs") or [])
                if fit_norm:
                    x0 = float(poly.get("x0") or 0.0)
                    sx = float(poly.get("sx") or 1.0) or 1.0
                    xfit_n = (xfit - x0) / sx
                else:
                    xfit_n = xfit
                yfit = pfit(xfit_n)
                ax.plot(xfit.tolist(), yfit.tolist(), linestyle="--", linewidth=1.4, alpha=0.85, color=color)
            except Exception:
                pass

    if master_poly.get("coeffs") and pooled_x:
        try:
            import numpy as np  # type: ignore

            xfit = np.linspace(float(min(pooled_x)), float(max(pooled_x)), 240)
            pfit = np.poly1d(master_poly.get("coeffs") or [])
            if fit_norm:
                x0 = float(master_poly.get("x0") or 0.0)
                sx = float(master_poly.get("sx") or 1.0) or 1.0
                xfit_n = (xfit - x0) / sx
            else:
                xfit_n = xfit
            yfit = pfit(xfit_n)
            ax.plot(xfit.tolist(), yfit.tolist(), linestyle="--", linewidth=1.8, alpha=0.75, color="#0f172a", label="Family fit")
        except Exception:
            pass

    ax.grid(True, alpha=0.25)
    try:
        if highlighted_serials or master_poly.get("coeffs"):
            ax.legend(fontsize=8, loc="best")
    except Exception:
        pass

    ax_txt = fig.add_axes([0.70, 0.12, 0.26, 0.76])
    ax_txt.axis("off")
    text_lines: list[str] = []
    if master_eqn:
        text_lines.append("Family Equation")
        text_lines.append(_wrap_cell(master_eqn, max_chars=34, max_lines=6))
        text_lines.append(f"RMSE: {_fmt_num(master_poly.get('rmse'))}")
        text_lines.append("")
    for sn in highlighted_serials:
        hm = highlighted_models.get(sn)
        if not isinstance(hm, dict):
            continue
        eqn = str(hm.get("equation") or "").strip()
        if not eqn:
            continue
        text_lines.append(f"{sn}")
        text_lines.append(_wrap_cell(eqn, max_chars=34, max_lines=6))
        text_lines.append(f"RMSE: {_fmt_num(hm.get('rmse'))}  Points: {int(hm.get('points') or 0)}")
        text_lines.append("")
    if not text_lines:
        text_lines = ["No fitted equations available."]
    ax_txt.text(
        0.0,
        1.0,
        "\n".join(text_lines).rstrip(),
        va="top",
        ha="left",
        fontsize=8,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
    )

    return fig


def _grade_cell_color(g: str) -> str:
    gg = str(g or "").strip().upper()
    if gg == "PASS":
        return "#dcfce7"
    if gg == "WATCH":
        return "#fef3c7"
    if gg == "FAIL":
        return "#fee2e2"
    return "#f1f5f9"


def _figure_grade_matrix_page(
    title: str,
    *,
    columns: list[str],
    rows: list[list[object]],
    grade_cells: set[tuple[int, int]],
):
    """
    grade_cells: set of (r_idx, c_idx) for rows/cols within `rows` (0-based in rows, 0-based in columns),
    where the value is a grade token (PASS/WATCH/FAIL/NO_DATA).
    """
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
    cell_text = [[("" if v is None else str(v)) for v in r] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    try:
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f1f5f9")
                continue
            if r % 2 == 0:
                cell.set_facecolor("#fafafa")
        for rr, cc in grade_cells:
            # +1 because matplotlib table has header at row 0.
            cell = table.get_celld().get((rr + 1, cc))
            if not cell:
                continue
            try:
                cell.set_facecolor(_grade_cell_color(cell.get_text().get_text()))
            except Exception:
                pass
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
    # In production-node mode, config files live under the node repo's `EIDAT\\UserData\\user_inputs`,
    # but the authoritative copies are kept in the central runtime `<runtime_root>\\user_inputs`.
    # Seed missing node-local configs on-demand so Auto Report can run without a separate deploy step.
    try:
        if getattr(be, "DATA_ROOT", None) != getattr(be, "ROOT", None):
            runtime_user_inputs = Path(getattr(be, "ROOT")) / "user_inputs"
            node_user_inputs = Path(getattr(be, "DATA_ROOT")) / "user_inputs"

            def _seed_user_input_if_missing(dst: Path) -> None:
                try:
                    p = Path(dst).expanduser()
                    if p.exists():
                        return
                    if not p.is_relative_to(node_user_inputs):
                        return
                    src = runtime_user_inputs / p.name
                    if not src.exists():
                        return
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(src.read_bytes())
                except Exception:
                    return

            _seed_user_input_if_missing(cfg_excel_path)
            _seed_user_input_if_missing(Path(be.DEFAULT_TREND_AUTO_REPORT_CONFIG).expanduser())
    except Exception:
        pass
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
    if True:
        if not bool(options.get("update_excel_trend_config", True)):
            change_summary = "excel_trend_config.json update disabled."

        conn = sqlite3.connect(str(Path(db_path).expanduser()))
        source_rows = []
        try:
            source_rows = be.td_read_sources_metadata(wb)
        except Exception:
            source_rows = []
        ordered_serials = _td_order_metric_serials(be.td_list_serials(db_path), source_rows) or _td_list_serials(conn)
        filter_state = options.get("filter_state")
        if not isinstance(filter_state, Mapping):
            filter_state = {}
        all_serials = _resolve_filtered_serials(be, db_path, ordered_serials, options)
        run_rows = _td_list_runs(conn)
        run_by_name = {str(r.get("run_name") or "").strip(): r for r in (run_rows or []) if str(r.get("run_name") or "").strip()}

        if not all_serials:
            if filter_state:
                raise RuntimeError("Auto Report filters excluded all serials in the current project cache.")
            raise RuntimeError(
                "Auto Report found no usable Test Data sources in the current project cache. "
                "Build / Refresh Cache again and verify the workbook Sources sheet points at the active node path."
            )

        runs = _resolve_selected_runs(run_rows, options)
        if not runs:
            raise RuntimeError(
                "Auto Report found no usable Test Data runs in the current project cache. "
                "Build / Refresh Cache again and verify TD source resolution for this project."
            )

        excel_cols = excel_cfg.get("columns") or []
        excel_names = {
            _norm_key(str(c.get("name") or "")): str(c.get("name") or "").strip()
            for c in excel_cols
            if isinstance(c, dict) and str(c.get("name") or "").strip()
        }

        params = _resolve_selected_params(be, db_path, conn, runs=runs, options=options)
        if not params:
            raise RuntimeError(
                "Auto Report found no reportable Test Data parameters in the current project cache. "
                "Check the workbook Sources sheet, cache diagnostics, and configured TD columns."
            )

        params_norm = {_norm_key(p) for p in params if p}
        hi = [s for s in highlighted_serials if s in all_serials]

        stats = excel_cfg.get("statistics") or ["mean", "min", "max", "std", "median", "count"]
        stats = [str(s).strip().lower() for s in stats if str(s).strip()]
        if not stats:
            stats = ["mean", "min", "max", "std", "median", "count"]
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
            run_selection = _selection_for_run(run, options)
            x_name = _resolve_curve_x_key(be, db_path, run, str(run_meta.get("default_x") or "").strip() or "Time")
            y_cols = []
            try:
                y_cols = be.td_list_curve_y_columns(db_path, run, x_name)
            except Exception:
                y_cols = []
            if not y_cols:
                try:
                    y_cols = be.td_list_raw_y_columns(db_path, run)
                except Exception:
                    y_cols = []
            if not y_cols:
                y_cols = _td_list_y_columns(conn, run)
            y_by_norm = {_norm_key(str(c.get("name") or "")): c for c in y_cols if str(c.get("name") or "").strip()}

            for p in params:
                nk = _norm_key(p)
                if nk not in params_norm or nk not in y_by_norm:
                    continue
                y_name = str(y_by_norm[nk].get("name") or "").strip()
                units = str(y_by_norm[nk].get("units") or "").strip()

                series = _load_curves_for_selection(
                    be,
                    db_path,
                    run,
                    y_name,
                    x_name,
                    selection=run_selection,
                    filter_state=filter_state,
                )
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
                            "max_abs": dv.get("max_abs"),
                            "rms": dv.get("rms"),
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

        cache_meta_by_sn, cache_meta_note = _read_cached_source_metadata(conn)
        workbook_meta_by_sn, workbook_meta_note = _read_workbook_metadata(wb)
        gui_meta_by_sn, gui_meta_note = _read_gui_source_metadata(be, wb)
        meta_by_sn: dict[str, dict[str, str]] = {}
        for sn in sorted(set(cache_meta_by_sn.keys()) | set(workbook_meta_by_sn.keys()) | set(gui_meta_by_sn.keys())):
            merged = dict(workbook_meta_by_sn.get(sn) or {})
            for key, value in (cache_meta_by_sn.get(sn) or {}).items():
                if str(value or "").strip():
                    merged[key] = str(value).strip()
            for key, value in (gui_meta_by_sn.get(sn) or {}).items():
                if str(value or "").strip():
                    merged[key] = str(value).strip()
            meta_by_sn[sn] = merged
        meta_note = ""
        if not meta_by_sn:
            meta_note = gui_meta_note or cache_meta_note or workbook_meta_note

        def _meta(sn: str, key: str) -> str:
            try:
                return str((meta_by_sn.get(sn) or {}).get(key) or "").strip()
            except Exception:
                return ""

        def _metric_tick_label(sn: str) -> str:
            program = _meta(sn, "program_title")
            if program:
                return f"{program}\n{sn}"
            return sn

        def _metric_map_for_run(run_name: str, column_name: str, stat: str, *, control_period_filter: object = None) -> dict[str, float]:
            selection = _selection_for_run(run_name, options)
            return _load_metric_map_for_selection(
                be,
                db_path,
                run_name,
                column_name,
                stat,
                selection=selection,
                control_period_filter=control_period_filter,
                filter_state=filter_state,
            )

        grade_map: dict[tuple[str, str, str], str] = {}
        for r in grading_rows:
            sn = str(r.get("serial") or "").strip()
            run = str(r.get("run") or "").strip()
            param = str(r.get("param") or "").strip()
            grade = str(r.get("grade") or "").strip().upper()
            if sn and run and param and grade:
                grade_map[(run, param, sn)] = grade

        run_param_pairs: list[tuple[str, str]] = []
        for run in runs:
            params_run = curves_summary.get(run, {}) or {}
            by_norm = {_norm_key(pn): pn for pn in params_run.keys()}
            for p in params:
                actual = by_norm.get(_norm_key(p))
                if actual:
                    run_param_pairs.append((run, actual))

        overall_by_sn = {sn: _overall_cert_status([grade_map.get((run, param, sn), "NO_DATA") for run, param in run_param_pairs]) for sn in hi}

        nonpass_findings = [r for r in grading_rows if str(r.get("grade") or "").strip().upper() in ("WATCH", "FAIL")]
        nonpass_findings = sorted(nonpass_findings, key=_finding_sort_key)
        nonpass_run_param = {(str(r.get("run") or ""), str(r.get("param") or "")) for r in nonpass_findings}

        by_sn_nonpass: dict[str, list[dict]] = {}
        for r in nonpass_findings:
            sn = str(r.get("serial") or "").strip()
            if sn:
                by_sn_nonpass.setdefault(sn, []).append(r)

        serials_nonpass_sorted = sorted(
            by_sn_nonpass.keys(),
            key=lambda sn: (
                0
                if any(str(rr.get("grade") or "").strip().upper() == "FAIL" for rr in (by_sn_nonpass.get(sn) or []))
                else 1,
                -max((abs(float(rr.get("z") or 0.0)) for rr in (by_sn_nonpass.get(sn) or [])), default=0.0),
            ),
        )

        metrics_summary: dict[str, dict[str, dict]] = {}
        if False and include_metrics and runs and params:
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

        max_pages = int((report_opts.get("max_pages") or 30) or 30)
        max_pages = max(4, max_pages)
        metrics_stats_cfg = options.get("metric_stats")
        if not isinstance(metrics_stats_cfg, list) or not metrics_stats_cfg:
            metrics_stats_cfg = report_opts.get("metrics_stats")
        metric_stats: list[str] = []
        if isinstance(metrics_stats_cfg, list):
            for st in metrics_stats_cfg:
                ss = str(st or "").strip().lower()
                if ss and ss in stats and ss not in metric_stats:
                    metric_stats.append(ss)
        elif isinstance(metrics_stats_cfg, str) and metrics_stats_cfg.strip():
            ss = metrics_stats_cfg.strip().lower()
            if ss in stats:
                metric_stats.append(ss)
        if not metric_stats:
            metric_stats = [stats[0] if stats else "median"]
        summary_metric_stat = metric_stats[0]

        appendix_include_grade_matrix = bool(report_opts.get("appendix_include_grade_matrix", True))
        appendix_include_pass_details = bool(report_opts.get("appendix_include_pass_details", True))

        assets_counts: dict[tuple[str, str, str, str], int] = {}
        for sn in hi:
            pn = _meta(sn, "part_number") or "—"
            rev = _meta(sn, "revision") or "—"
            at = _meta(sn, "asset_type") or "—"
            vd = _meta(sn, "vendor") or "—"
            assets_counts[(pn, rev, at, vd)] = int(assets_counts.get((pn, rev, at, vd), 0)) + 1
        assets_lines = [f"  • PN {pn}  Rev {rev}  Asset {at}  Vendor {vd}  (x{n})" for (pn, rev, at, vd), n in sorted(assets_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

        programs = sorted({p for p in (_meta(sn, "program_title") for sn in hi) if p})
        program_disp = programs[0] if len(programs) == 1 else ("(multiple)" if programs else "(unknown)")
        sim_groups = sorted({g for g in (_meta(sn, "similarity_group") for sn in hi) if g})
        sim_disp = ", ".join(sim_groups) if sim_groups else "(unknown)"

        cover_lines = [
            "Acceptance Test Certification of test data against family baseline.",
            "",
            "Assets involved:",
            *(assets_lines if assets_lines else ["  • (metadata unavailable)"]),
            "",
            "Scope:",
            f"  • Program: {program_disp}",
            f"  • Similarity Group(s): {sim_disp}",
            f"  • Serials under certification ({len(hi)}): {', '.join(hi)}",
            f"  • Runs included: {', '.join(runs)}",
            f"  • Params included: {', '.join(params)}",
            f"  • Generated: {_now_datestr()}",
        ]
        if meta_note:
            cover_lines += ["", f"Note: {meta_note}"]

        performance_models: list[dict] = []
        omitted_items: list[str] = []

        params_norm_set = {_norm_key(p) for p in params if str(p).strip()}
        run_details_runs = []
        for run in runs:
            params_run = curves_summary.get(run, {}) or {}
            if any(_norm_key(k) in params_norm_set for k in params_run.keys()):
                run_details_runs.append(run)

        perf_defs_all = []
        try:
            raw_perf_opt = options.get("performance_plotters")
            if isinstance(raw_perf_opt, list):
                raw_perf = raw_perf_opt
            else:
                raw_perf = excel_cfg.get("performance_plotters") if isinstance(excel_cfg, dict) else []
            if isinstance(raw_perf, list):
                for pd in raw_perf:
                    if not isinstance(pd, dict):
                        continue
                    x_spec = pd.get("x") or {}
                    y_spec = pd.get("y") or {}
                    x_target = str((x_spec.get("column") if isinstance(x_spec, dict) else "") or "").strip()
                    y_target = str((y_spec.get("column") if isinstance(y_spec, dict) else "") or "").strip()
                    if x_target and y_target:
                        perf_defs_all.append(pd)
        except Exception:
            perf_defs_all = []

        max_plots_cfg = report_opts.get("max_plots")
        try:
            max_plots = int(max_plots_cfg) if max_plots_cfg not in (None, "") else None
        except Exception:
            max_plots = None
        chart_specs_all = _build_chart_specs(run_param_pairs=run_param_pairs, nonpass_findings=nonpass_findings, max_plots=max_plots)

        metric_params_opt = options.get("metric_params")
        if isinstance(metric_params_opt, list):
            metric_params = [str(p).strip() for p in metric_params_opt if str(p).strip()]
        else:
            metric_params = list(params)

        metrics_pairs_all: list[tuple[str, str, str]] = []
        for run in runs:
            params_run = curves_summary.get(run, {}) or {}
            by_norm = {_norm_key(pn): pn for pn in params_run.keys()}
            for p in metric_params:
                actual = by_norm.get(_norm_key(p))
                if not actual:
                    continue
                for st in metric_stats:
                    metrics_pairs_all.append((run, actual, st))
        metrics_pairs_nonpass = [(run, param, st) for (run, param, st) in metrics_pairs_all if (run, param) in nonpass_run_param]

        plan_sel = _plan_page_selections(
            max_pages=max_pages,
            appendix_include_grade_matrix=appendix_include_grade_matrix,
            appendix_include_pass_details=appendix_include_pass_details,
            include_metrics=include_metrics,
            grading_rows=grading_rows,
            run_param_pairs=run_param_pairs,
            serials_nonpass_sorted=serials_nonpass_sorted,
            perf_defs_all=perf_defs_all,
            run_details_all=run_details_runs,
            chart_specs_all=chart_specs_all,
            metrics_pairs_all=metrics_pairs_all,
            metrics_pairs_nonpass=metrics_pairs_nonpass,
        )
        perf_defs_sel = plan_sel["perf_defs_sel"]
        run_details_sel = plan_sel["run_details_sel"]
        serials_nonpass_sorted = plan_sel["serials_nonpass_sorted"]
        include_deviations = bool(plan_sel["include_deviations"])
        deviations_pages = int(plan_sel["deviations_pages"])
        metrics_sel = plan_sel["metrics_sel"]
        charts_sel = plan_sel["charts_sel"]
        include_omitted_page = bool(plan_sel["include_omitted_page"])
        omitted_items = plan_sel["omitted_items"]

        sidecar = {
            "version": 2,
            "generated_date": _now_datestr(),
            "project_dir": str(proj),
            "workbook_path": str(wb),
            "db_path": str(db_path),
            "output_pdf": str(out_pdf),
            "report_config": report_cfg,
            "options": options,
            "investigated_serials": hi,
            "metadata_by_serial": {sn: (meta_by_sn.get(sn) or {}) for sn in hi},
            "overall_results_by_serial": overall_by_sn,
            "non_pass_findings": nonpass_findings,
            "runs": runs,
            "params": params,
            "curve_models": curves_summary,
            "watch_items": watch_items,
            "grading": grading_rows,
            "page_cap": int(max_pages),
            "omitted_items": omitted_items,
        }
        sidecar_path = out_pdf.with_suffix(".summary.json")
        # Sidecar written after PDF generation (includes performance models).

        with PdfPages(out_pdf) as pdf:
            import matplotlib.pyplot as plt  # type: ignore

            # 1) Cover
            pdf.savefig(_figure_text_page("Acceptance Test Certification Report", cover_lines))

            # 2) Executive summary (table + grading explanation)
            exec_rows = []
            for sn in hi:
                exec_rows.append(
                    [
                        sn,
                        overall_by_sn.get(sn, "NO_DATA"),
                        _meta(sn, "part_number"),
                        _meta(sn, "revision"),
                        _meta(sn, "asset_type"),
                        _meta(sn, "vendor"),
                        _meta(sn, "acceptance_test_plan_number"),
                        _meta(sn, "similarity_group"),
                        _meta(sn, "test_date"),
                        _meta(sn, "report_date"),
                    ]
                )

            fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
            fig.patch.set_facecolor("white")
            fig.text(0.06, 0.96, "Executive Summary", fontsize=15, fontweight="bold", va="top")
            ax_t = fig.add_axes([0.06, 0.36, 0.92, 0.56])
            ax_t.axis("off")
            cols = ["Serial", "Overall", "Part #", "Rev", "Asset Type", "Vendor", "ATP", "Similarity Group", "Test Date", "Report Date"]
            tab = ax_t.table(cellText=[[("" if v is None else str(v)) for v in r] for r in exec_rows], colLabels=cols, cellLoc="left", loc="upper left")
            tab.auto_set_font_size(False)
            tab.set_fontsize(7)
            tab.scale(1.0, 1.2)
            try:
                for (r, _c), cell in tab.get_celld().items():
                    if r == 0:
                        cell.set_text_props(weight="bold")
                        cell.set_facecolor("#f1f5f9")
                    if r % 2 == 0 and r != 0:
                        cell.set_facecolor("#fafafa")
            except Exception:
                pass

            expl = [
                "Grading metrics (family comparison):",
                "  • Family curve: median across all serials in the project cache (per run/param).",
                "  • Residual metrics: max_abs, max_pct, rms_pct, x@max.",
                "  • Grade basis: z-score of max_abs vs family distribution.",
                f"  • Thresholds: PASS if |z|≤{_fmt_num(z_pass)}; WATCH if |z|≤{_fmt_num(z_watch)}; else FAIL.",
                "  • Main body shows WATCH/FAIL only; PASS details are in the appendix.",
            ]
            y = 0.30
            for line in expl:
                fig.text(0.06, y, line, fontsize=9, va="top", color="#0f172a")
                y -= 0.03
            pdf.savefig(fig)
            plt.close(fig)

            # 3) Non-PASS findings overview
            if not nonpass_findings:
                pdf.savefig(_figure_text_page("Non‑PASS Findings (Overview)", ["All investigated serials are CERTIFIED (all PASS)."]))
            else:
                cols = ["Serial", "Run", "Param", "Grade", "Max %", "RMS %", "x@max", "z"]
                rows = [
                    [
                        r.get("serial"),
                        r.get("run"),
                        r.get("param"),
                        r.get("grade"),
                        _fmt_num(r.get("max_pct")),
                        _fmt_num(r.get("rms_pct")),
                        _fmt_num(r.get("x_at_max_abs"), sig=5),
                        _fmt_num(r.get("z"), sig=4),
                    ]
                    for r in nonpass_findings
                ]
                pdf.savefig(_figure_table_page2("Non‑PASS Findings (Overview)", cols, rows, landscape=True, font_size=7))

            # 4) Non-PASS findings by serial (two per page)
            cols_by_sn = ["Run", "Param", "Grade", "Max %", "RMS %", "x@max", "z"]
            for i in range(0, len(serials_nonpass_sorted), 2):
                left_sn = serials_nonpass_sorted[i]
                right_sn = serials_nonpass_sorted[i + 1] if i + 1 < len(serials_nonpass_sorted) else ""
                left_rows = [
                    [
                        rr.get("run"),
                        rr.get("param"),
                        rr.get("grade"),
                        _fmt_num(rr.get("max_pct")),
                        _fmt_num(rr.get("rms_pct")),
                        _fmt_num(rr.get("x_at_max_abs"), sig=5),
                        _fmt_num(rr.get("z"), sig=4),
                    ]
                    for rr in (by_sn_nonpass.get(left_sn) or [])
                ]
                right_rows = [
                    [
                        rr.get("run"),
                        rr.get("param"),
                        rr.get("grade"),
                        _fmt_num(rr.get("max_pct")),
                        _fmt_num(rr.get("rms_pct")),
                        _fmt_num(rr.get("x_at_max_abs"), sig=5),
                        _fmt_num(rr.get("z"), sig=4),
                    ]
                    for rr in (by_sn_nonpass.get(right_sn) or [])
                ]
                fig = _figure_two_tables_page(
                    "Non‑PASS Findings (By Serial)",
                    left_title=left_sn or "—",
                    left_columns=cols_by_sn,
                    left_rows=left_rows or [["", "", "", "", "", "", ""]],
                    right_title=right_sn or "—",
                    right_columns=cols_by_sn,
                    right_rows=right_rows or [["", "", "", "", "", "", ""]],
                )
                pdf.savefig(fig)
                plt.close(fig)

            # 5) Unit-to-family summary: requested param medians (pooled across runs)
            metric_map_cache: dict[tuple[str, str], dict[str, float]] = {}
            units_by_param_norm: dict[str, str] = {}
            col_by_run_param_norm: dict[tuple[str, str], str] = {}
            for run in runs:
                for p in params:
                    nk = _norm_key(p)
                    metric_cols = []
                    try:
                        metric_cols = be.td_list_metric_y_columns(db_path, run)
                    except Exception:
                        metric_cols = []
                    if not metric_cols:
                        metric_cols = _td_list_y_columns(conn, run)
                    col, units = _resolve_td_y_col_from_rows(metric_cols, p)
                    if not col:
                        continue
                    col_by_run_param_norm[(run, nk)] = col
                    if units and nk not in units_by_param_norm:
                        units_by_param_norm[nk] = units
                    key = (run, col)
                    if key not in metric_map_cache:
                        metric_map_cache[key] = _metric_map_for_run(run, col, summary_metric_stat)

            def _pooled_median(sn: str, p: str) -> float | None:
                nk = _norm_key(p)
                vals: list[float] = []
                for run in runs:
                    col = col_by_run_param_norm.get((run, nk))
                    if not col:
                        continue
                    vmap = metric_map_cache.get((run, col)) or {}
                    v = vmap.get(sn)
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        vals.append(float(v))
                if not vals:
                    return None
                try:
                    return float(statistics.median(vals))
                except Exception:
                    return None

            family_median_by_param: dict[str, float | None] = {}
            for p in params:
                vals = []
                for sn in all_serials:
                    pv = _pooled_median(sn, p)
                    if pv is not None and math.isfinite(float(pv)):
                        vals.append(float(pv))
                family_median_by_param[p] = float(statistics.median(vals)) if vals else None

            cols = ["Parameter", "Units", "Family Median (pooled)"] + hi
            rows = []
            for p in params:
                nk = _norm_key(p)
                units = units_by_param_norm.get(nk, "")
                fam = family_median_by_param.get(p)
                row: list[object] = [p, units, _fmt_num(fam)]
                for sn in hi:
                    row.append(_fmt_num(_pooled_median(sn, p)))
                rows.append(row)
            pdf.savefig(_figure_table_page2("Unit‑to‑Family Summary — Requested Param Medians", cols, rows, landscape=True, font_size=7))

            # 5B) Unit-to-family summary: performance equations (from excel_trend_config.performance_plotters)
            perf_defs = perf_defs_sel
            if isinstance(perf_defs, list) and perf_defs:
                for pd in perf_defs:
                    if not isinstance(pd, dict):
                        continue
                    name = str(pd.get("name") or "Performance").strip() or "Performance"
                    x_spec = pd.get("x") or {}
                    y_spec = pd.get("y") or {}
                    x_target = str((x_spec.get("column") if isinstance(x_spec, dict) else "") or "").strip()
                    y_target = str((y_spec.get("column") if isinstance(y_spec, dict) else "") or "").strip()
                    if not x_target or not y_target:
                        continue

                    stats_list = pd.get("stats")
                    if isinstance(stats_list, list) and all(isinstance(s, str) for s in stats_list):
                        stats_list = [str(s).strip().lower() for s in stats_list if str(s).strip()]
                    else:
                        legacy = str((x_spec.get("stat") if isinstance(x_spec, dict) else "mean") or "mean").strip().lower()
                        stats_list = [legacy] if legacy else ["mean"]
                    if not stats_list:
                        stats_list = ["mean"]
                    st = (str(stats_list[0]).strip().lower() if stats_list else "mean") or "mean"

                    require_min_points = int(pd.get("require_min_points") or 2)
                    require_min_points = max(2, require_min_points)

                    fit_cfg = pd.get("fit") or {}
                    fit_degree = int((fit_cfg.get("degree") if isinstance(fit_cfg, dict) else 0) or 0)
                    fit_degree = max(0, fit_degree)
                    fit_norm = bool((fit_cfg.get("normalize_x") if isinstance(fit_cfg, dict) else True))

                    curves, pooled_x, pooled_y, x_units, y_units = _collect_performance_curves_for_stat(
                        be=be,
                        db_path=db_path,
                        conn=conn,
                        run_by_name=run_by_name,
                        runs=runs,
                        serials=all_serials,
                        x_target=x_target,
                        y_target=y_target,
                        stat=st,
                        options=options,
                        require_min_points=require_min_points,
                        filter_state=filter_state,
                    )
                    if not curves:
                        continue

                    # Fits (master + investigated)
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
                        if not pts:
                            continue
                        row: dict[str, object] = {"points": int(len(pts))}
                        if fit_degree > 0:
                            try:
                                xs = [p[0] for p in pts]
                                ys = [p[1] for p in pts]
                                poly = _poly_fit(xs, ys, int(fit_degree), normalize_x=fit_norm)
                                row.update({"poly": poly, "equation": _fmt_equation(poly), "rmse": poly.get("rmse")})
                            except Exception:
                                pass
                        highlighted_models[sn] = row

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

                    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
                    fig.patch.set_facecolor("white")
                    fig.text(0.06, 0.96, f"Unit‑to‑Family Summary — Performance Equation — {name} ({st})", fontsize=14, fontweight="bold", va="top")
                    master_lines = []
                    if master_eqn:
                        master_lines.append(f"Master (family): {master_eqn}")
                    if master_poly.get("rmse") is not None:
                        master_lines.append(f"Master RMSE: {_fmt_num(master_poly.get('rmse'))}")
                    if master_lines:
                        fig.text(0.06, 0.90, "\n".join(master_lines), fontsize=9, va="top", color="#0f172a")

                    ax = fig.add_axes([0.06, 0.10, 0.92, 0.74])
                    ax.axis("off")
                    tcols = ["Serial", "Points", "Equation", "RMSE"]
                    trows = []
                    for sn in hi:
                        hm = highlighted_models.get(sn)
                        if not isinstance(hm, dict):
                            trows.append([sn, "", "", ""])
                            continue
                        eqn = _wrap_cell(str(hm.get("equation") or ""), max_chars=52, max_lines=2)
                        trows.append([sn, str(hm.get("points") or ""), eqn, _fmt_num(hm.get("rmse"))])
                    tab = ax.table(cellText=trows, colLabels=tcols, cellLoc="left", loc="upper left")
                    tab.auto_set_font_size(False)
                    tab.set_fontsize(7)
                    tab.scale(1.0, 1.2)
                    try:
                        for (r, _c), cell in tab.get_celld().items():
                            if r == 0:
                                cell.set_text_props(weight="bold")
                                cell.set_facecolor("#f1f5f9")
                            if r % 2 == 0 and r != 0:
                                cell.set_facecolor("#fafafa")
                    except Exception:
                        pass

                    pdf.savefig(fig)
                    plt.close(fig)

                    fig = _figure_performance_equation_page(
                        title=f"Unitâ€‘toâ€‘Family Summary â€” Performance Equation Chart â€” {name} ({st})",
                        x_label=f"{x_target}.{st}" + (f" ({x_units})" if x_units else ""),
                        y_label=f"{y_target}.{st}" + (f" ({y_units})" if y_units else ""),
                        curves=curves,
                        highlighted_serials=hi,
                        highlighted_models=highlighted_models,
                        master_poly=master_poly,
                        master_eqn=master_eqn,
                        fit_norm=fit_norm,
                        colors=colors,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

            for run in run_details_sel:
                run_meta = run_by_name.get(run) or {}
                title = str(run_meta.get("display_name") or "").strip() or run
                params_run = curves_summary.get(run, {}) or {}
                cols = ["Parameter", "Units", "x-axis", "Domain", "Poly RMSE", "Equation"]
                rows = []
                by_norm = {_norm_key(k): k for k in params_run.keys()}
                for p in params:
                    actual = by_norm.get(_norm_key(p))
                    if not actual:
                        continue
                    d = params_run.get(actual) or {}
                    poly = d.get("poly") or {}
                    eqn = _wrap_cell(str(d.get("equation") or ""), max_chars=60, max_lines=2)
                    rows.append(
                        [
                            actual,
                            d.get("units") or "",
                            d.get("x_name") or "",
                            "…".join(f"{float(x):.4g}" for x in (d.get("domain") or [])[:2]) if d.get("domain") else "",
                            _fmt_num(poly.get("rmse"), sig=4),
                            eqn,
                        ]
                    )
                if rows:
                    pdf.savefig(_figure_table_page2(f"Run Details — {title}", cols, rows, landscape=True, font_size=7))

            # 7) Charts — only non-PASS overlays (WATCH/FAIL)
            if include_metrics:
                serials = all_serials
                x_idx = list(range(len(serials)))
                x_labels = [_metric_tick_label(sn) for sn in serials]
                for run, param_name, metric_stat in metrics_sel:
                    run_meta = run_by_name.get(run) or {}
                    run_title = str(run_meta.get("display_name") or "").strip() or run
                    units = str(((curves_summary.get(run, {}) or {}).get(param_name) or {}).get("units") or "").strip()
                    vmap = _metric_map_for_run(run, param_name, metric_stat)
                    yv = [(float(vmap.get(sn)) if isinstance(vmap.get(sn), (int, float)) else float("nan")) for sn in serials]
                    if not any(isinstance(v, (int, float)) and not math.isnan(float(v)) for v in yv):
                        continue

                    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.set_title(f"Metrics â€” {run_title} â€” {param_name} ({metric_stat})")
                    ax.set_xlabel("Program + Serial Number")
                    ax.set_ylabel(f"{param_name} ({units})" if units else param_name)
                    ax.plot(x_idx, yv, marker="o", linewidth=1.0, alpha=0.45, color="#64748b")
                    for sn in hi:
                        if sn not in serials:
                            continue
                        xi = serials.index(sn)
                        g = grade_map.get((run, param_name, sn), "NO_DATA")
                        color = "#3b82f6"
                        if g == "WATCH":
                            color = "#f59e0b"
                        elif g == "FAIL":
                            color = "#ef4444"
                        ax.scatter([xi], [yv[xi]], s=36, color=color, zorder=5)
                        ax.axvline(xi, color=color, linewidth=1.0, alpha=0.12)

                    try:
                        finite_vals = [float(v) for v in yv if isinstance(v, (int, float)) and not math.isnan(float(v))]
                        if finite_vals:
                            fam_med = float(statistics.median(finite_vals))
                            ax.axhline(fam_med, color="#0f172a", linestyle="--", linewidth=1.1, alpha=0.6, label="family median")
                    except Exception:
                        pass

                    ax.set_xticks(x_idx)
                    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
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
                metrics_sel = []

            for run, param_name in charts_sel:
                run_meta = run_by_name.get(run) or {}
                run_title = str(run_meta.get("display_name") or "").strip() or run
                run_selection = _selection_for_run(run, options)
                x_name = _resolve_curve_x_key(be, db_path, run, str(run_meta.get("default_x") or "").strip() or "Time")
                model = (curves_summary.get(run, {}) or {}).get(param_name) or {}
                series = _load_curves_for_selection(
                    be,
                    db_path,
                    run,
                    param_name,
                    x_name,
                    selection=run_selection,
                    filter_state=filter_state,
                )
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
                ax.plot(x_grid, master_y, linewidth=2.2, color="#0f172a", label="Family (median)")
                try:
                    band_lo = [a - b if (isinstance(a, (int, float)) and not math.isnan(float(a)) and isinstance(b, (int, float)) and not math.isnan(float(b))) else float("nan") for a, b in zip(master_y, std_y)]
                    band_hi = [a + b if (isinstance(a, (int, float)) and not math.isnan(float(a)) and isinstance(b, (int, float)) and not math.isnan(float(b))) else float("nan") for a, b in zip(master_y, std_y)]
                    ax.fill_between(x_grid, band_lo, band_hi, color="#93c5fd", alpha=0.25, label="±1σ band")
                except Exception:
                    pass

                nonpass_sns = [sn for sn in hi if grade_map.get((run, param_name, sn), "NO_DATA") in ("WATCH", "FAIL")]
                for idx, sn in enumerate(nonpass_sns):
                    yv = y_resampled_by_sn.get(sn)
                    if not yv:
                        continue
                    g = grade_map.get((run, param_name, sn), "NO_DATA")
                    ax.plot(x_grid, yv, linewidth=1.7, color=colors[idx % len(colors)], label=f"{sn} ({g})")

                eqn = str(model.get("equation") or "").strip()
                rmse = (model.get("poly") or {}).get("rmse")
                notes = []
                if eqn:
                    notes.append(eqn)
                if rmse is not None:
                    notes.append(f"RMSE: {_fmt_num(rmse)}")
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

            # 8) Metrics — SN vs Value (single stat)
            if include_metrics:
                for run, param_name, metric_stat in metrics_sel:
                    run_meta = run_by_name.get(run) or {}
                    run_title = str(run_meta.get("display_name") or "").strip() or run
                    units = str(((curves_summary.get(run, {}) or {}).get(param_name) or {}).get("units") or "").strip()
                    vmap = _metric_map_for_run(run, param_name, metric_stat)
                    serials = all_serials
                    x_idx = list(range(len(serials)))
                    yv = [(float(vmap.get(sn)) if isinstance(vmap.get(sn), (int, float)) else float("nan")) for sn in serials]
                    if not any(isinstance(v, (int, float)) and not math.isnan(float(v)) for v in yv):
                        continue

                    fig = plt.figure(figsize=(11.0, 8.5), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.set_title(f"Metrics — {run_title} — {param_name} ({metric_stat})")
                    ax.set_xlabel("Serial Number")
                    ax.set_ylabel(f"{param_name} ({units})" if units else param_name)

                    ax.plot(x_idx, yv, marker="o", linewidth=1.0, alpha=0.45, color="#64748b")
                    for sn in hi:
                        if sn not in serials:
                            continue
                        xi = serials.index(sn)
                        g = grade_map.get((run, param_name, sn), "NO_DATA")
                        color = "#3b82f6"
                        if g == "WATCH":
                            color = "#f59e0b"
                        elif g == "FAIL":
                            color = "#ef4444"
                        ax.scatter([xi], [yv[xi]], s=36, color=color, zorder=5)
                        ax.axvline(xi, color=color, linewidth=1.0, alpha=0.12)

                    try:
                        finite_vals = [float(v) for v in yv if isinstance(v, (int, float)) and not math.isnan(float(v))]
                        if finite_vals:
                            fam_med = float(statistics.median(finite_vals))
                            ax.axhline(fam_med, color="#0f172a", linestyle="--", linewidth=1.1, alpha=0.6, label="family median")
                    except Exception:
                        pass

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

            # 9) Appendix — Grade matrix
            if appendix_include_grade_matrix and run_param_pairs and hi:
                cols = ["Run", "Param"] + hi
                matrix_rows_all: list[list[object]] = []
                for run, param in run_param_pairs:
                    matrix_rows_all.append([run, param] + [grade_map.get((run, param, sn), "NO_DATA") for sn in hi])
                for idx, start in enumerate(range(0, len(matrix_rows_all), 30), start=1):
                    rows = matrix_rows_all[start : start + 30]
                    grade_cells = {(rr, cc) for rr in range(len(rows)) for cc in range(2, len(cols))}
                    title = "Appendix — Grade Matrix (PASS/WATCH/FAIL)"
                    if len(matrix_rows_all) > 30:
                        total_pages = _ceil_div(len(matrix_rows_all), 30)
                        title += f" ({idx}/{total_pages})"
                    fig = _figure_grade_matrix_page(title, columns=cols, rows=rows, grade_cells=grade_cells)
                    pdf.savefig(fig)
                    plt.close(fig)

            # Optional appendix: full deviations table (kept tight)
            if include_deviations and grading_rows:
                cols = ["Serial", "Run", "Param", "Grade", "z", "Max Abs", "RMS", "Max %", "RMS %", "x@max"]
                max_rows = 60
                rows = []
                for r in grading_rows[:max_rows]:
                    rows.append(
                        [
                            r.get("serial"),
                            r.get("run"),
                            r.get("param"),
                            r.get("grade"),
                            _fmt_num(r.get("z"), sig=4),
                            _fmt_num(r.get("max_abs"), sig=5),
                            _fmt_num(r.get("rms"), sig=5),
                            _fmt_num(r.get("max_pct")),
                            _fmt_num(r.get("rms_pct")),
                            _fmt_num(r.get("x_at_max_abs"), sig=5),
                        ]
                    )
                note = f"(Showing first {len(rows)} rows.)" if len(grading_rows) > len(rows) else ""
                pdf.savefig(_figure_table_page2(f"Appendix — Full Deviations {note}".strip(), cols, rows, landscape=True, font_size=7))

            # Omitted items note (if we had to downselect to meet page cap)
            if include_omitted_page and omitted_items:
                lines = ["The following items were omitted to meet the max page cap:", *[f"  • {x}" for x in omitted_items]]
                pdf.savefig(_figure_text_page("Omitted Items", lines))

            # Performance plotters (config-driven X vs Y metrics per serial across runs)
            perf_defs: list[object] = []
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
                        curves, pooled_x, pooled_y, x_units, y_units = _collect_performance_curves_for_stat(
                            be=be,
                            db_path=db_path,
                            conn=conn,
                            run_by_name=run_by_name,
                            runs=runs,
                            serials=all_serials,
                            x_target=x_target,
                            y_target=y_target,
                            stat=st,
                            options=options,
                            require_min_points=require_min_points,
                            filter_state=filter_state,
                        )
                        if not curves:
                            continue

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
        try:
            conn.close()
        except Exception:
            pass

    return {
        "output_pdf": str(out_pdf),
        "summary_json": str(sidecar_path),
        "db_path": str(db_path),
        "runs": runs,
        "params": params,
        "highlighted_serials": hi,
        "watch_items": len(watch_items),
    }
