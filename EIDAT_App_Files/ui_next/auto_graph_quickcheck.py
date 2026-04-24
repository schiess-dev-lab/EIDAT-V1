from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import statistics
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


LIBRARY_VERSION = 1
STORE_DIRNAME = "auto_graph_quickchecks"
LIBRARY_FILENAME = "library.json"
SNAPSHOT_DIRNAME = "snapshots"
BASELINE_DB_FILENAME = "baseline.sqlite3"
FILTER_KEYS = (
    "programs",
    "serials",
    "control_periods",
    "suppression_voltages",
    "valve_voltages",
)
STATUS_ORDER = {"FAIL": 3, "WATCH": 2, "PASS": 1, "NO_DATA": 0}


def _backend() -> Any:
    try:
        from . import backend as be  # type: ignore
    except Exception:  # pragma: no cover
        import ui_next.backend as be  # type: ignore
    return be


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(value: object, *, fallback: str = "quickcheck") -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip()).strip("_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw or fallback


def _hash_text(value: object) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8", "ignore")).hexdigest()[:12]


def _pack_id(name: object) -> str:
    return f"pack_{_slug(name, fallback='quickcheck')}_{_hash_text(f'{name}|{time.time_ns()}')}"


def _plot_id(plot_definition: Mapping[str, object] | None, name: object) -> str:
    payload = json.dumps(dict(plot_definition or {}), sort_keys=True, ensure_ascii=False)
    return f"plot_{_slug(name, fallback='graph')}_{_hash_text(payload)}"


def quickcheck_root(project_dir: Path) -> Path:
    return Path(project_dir).expanduser() / STORE_DIRNAME


def quickcheck_library_path(project_dir: Path) -> Path:
    return quickcheck_root(project_dir) / LIBRARY_FILENAME


def quickcheck_snapshot_path(project_dir: Path, pack_id: str) -> Path:
    return quickcheck_root(project_dir) / SNAPSHOT_DIRNAME / str(pack_id or "").strip() / BASELINE_DB_FILENAME


def _normalize_text_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def normalize_filter_state(filter_state: Mapping[str, object] | None) -> dict[str, list[str]]:
    source = filter_state if isinstance(filter_state, Mapping) else {}
    return {key: _normalize_text_list(source.get(key)) for key in FILTER_KEYS}


def _compact_value(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return ""
        return f"{number:g}" if math.isfinite(number) else ""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        number = float(raw)
    except Exception:
        return raw
    return f"{number:g}" if math.isfinite(number) else raw


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _filter_row_matches(serial_row: Mapping[str, object], filter_state: Mapping[str, object] | None, *, ignore_serials: bool = False) -> bool:
    state = normalize_filter_state(filter_state)
    serial = str(serial_row.get("serial") or "").strip()
    program = str(serial_row.get("program_title") or "").strip()
    if state["programs"] and program not in set(state["programs"]):
        return False
    if not ignore_serials and state["serials"] and serial not in set(state["serials"]):
        return False
    value_sets = {
        "control_periods": {str(value).strip() for value in (serial_row.get("control_periods") or []) if str(value).strip()},
        "suppression_voltages": {str(value).strip() for value in (serial_row.get("suppression_voltages") or []) if str(value).strip()},
        "valve_voltages": {str(value).strip() for value in (serial_row.get("valve_voltages") or []) if str(value).strip()},
    }
    for key in ("control_periods", "suppression_voltages", "valve_voltages"):
        wanted = set(state[key])
        if wanted and not (wanted & value_sets[key]):
            return False
    return True


def _default_curve_threshold(value: object, *, ratio: float) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return float(number) * float(ratio)


def default_finding_rule(mode: object, *, report_config: Mapping[str, object] | None = None) -> dict[str, object]:
    raw_mode = str(mode or "").strip().lower()
    cfg = dict(report_config or {})
    watch_cfg = (((cfg.get("watch") or {}) if isinstance(cfg, Mapping) else {}).get("curve_deviation") or {})
    grading_cfg = (cfg.get("grading") or {}) if isinstance(cfg, Mapping) else {}
    pass_max = float(grading_cfg.get("zscore_pass_max") or 2.0)
    watch_max = float(grading_cfg.get("zscore_watch_max") or 3.0)
    if raw_mode == "curves":
        max_abs_fail = _safe_float(watch_cfg.get("max_abs"))
        max_pct_fail = _safe_float(watch_cfg.get("max_pct"))
        rms_pct_fail = _safe_float(watch_cfg.get("rms_pct"))
        return {
            "mode": "curve_thresholds",
            "max_abs_watch": _default_curve_threshold(max_abs_fail, ratio=0.5),
            "max_abs_fail": max_abs_fail,
            "max_pct_watch": _default_curve_threshold(max_pct_fail, ratio=0.5),
            "max_pct_fail": max_pct_fail,
            "rms_pct_watch": _default_curve_threshold(rms_pct_fail, ratio=0.5),
            "rms_pct_fail": rms_pct_fail,
        }
    return {
        "mode": "zscore",
        "zscore_pass_max": pass_max,
        "zscore_watch_max": watch_max,
        "abs_watch_max": None,
        "abs_fail_max": None,
        "pct_watch_max": None,
        "pct_fail_max": None,
    }


def _normalize_plot_entry(raw_entry: Mapping[str, object] | None, *, report_config: Mapping[str, object] | None = None) -> dict[str, object] | None:
    if not isinstance(raw_entry, Mapping):
        return None
    plot_definition = raw_entry.get("plot_definition")
    if not isinstance(plot_definition, Mapping):
        plot_definition = {
            str(key): value
            for key, value in raw_entry.items()
            if str(key) not in {"id", "name", "plot_definition", "finding_rule"}
        }
    plot_definition = dict(plot_definition or {})
    mode = str(plot_definition.get("mode") or "").strip().lower()
    if mode not in {"curves", "metrics", "life_metrics", "performance"}:
        return None
    name = str(raw_entry.get("name") or "").strip() or str(plot_definition.get("name") or "").strip() or f"{mode.title()} Plot"
    plot_id = str(raw_entry.get("id") or "").strip() or _plot_id(plot_definition, name)
    finding_rule_raw = raw_entry.get("finding_rule")
    finding_rule = dict(finding_rule_raw) if isinstance(finding_rule_raw, Mapping) else default_finding_rule(mode, report_config=report_config)
    return {
        "id": plot_id,
        "name": name,
        "plot_definition": plot_definition,
        "finding_rule": finding_rule,
    }


def _normalize_pack(raw_pack: Mapping[str, object] | None, *, project_dir: Path | None = None) -> dict[str, object]:
    be = _backend()
    report_config = {}
    if project_dir is not None:
        try:
            report_config = be.load_trend_auto_report_config(Path(project_dir).expanduser())
        except Exception:
            report_config = {}
    source = dict(raw_pack or {}) if isinstance(raw_pack, Mapping) else {}
    name = str(source.get("name") or "").strip() or "Quick-Check Pack"
    pack_id = str(source.get("id") or "").strip() or _pack_id(name)
    created_at = str(source.get("created_at") or "").strip() or _now_text()
    updated_at = str(source.get("updated_at") or "").strip() or created_at
    baseline_snapshot_raw = source.get("baseline_snapshot")
    baseline_snapshot = dict(baseline_snapshot_raw) if isinstance(baseline_snapshot_raw, Mapping) else {}
    normalized_baseline = {
        "db_path": str(baseline_snapshot.get("db_path") or quickcheck_snapshot_path(project_dir or Path("."), pack_id)).strip(),
        "captured_at_epoch_ns": int(baseline_snapshot.get("captured_at_epoch_ns") or 0),
        "source_db_mtime_ns": int(baseline_snapshot.get("source_db_mtime_ns") or 0),
        "baseline_filters": normalize_filter_state(baseline_snapshot.get("baseline_filters")),
        "baseline_serials": _normalize_text_list(baseline_snapshot.get("baseline_serials")),
    }
    eligibility_filters = normalize_filter_state(source.get("eligibility_filters") or normalized_baseline["baseline_filters"])
    plots_raw = source.get("plots") if isinstance(source.get("plots"), list) else []
    plots = [
        plot
        for plot in (
            _normalize_plot_entry(item if isinstance(item, Mapping) else None, report_config=report_config)
            for item in plots_raw
        )
        if plot is not None
    ]
    finding_rules_raw = source.get("finding_rules")
    finding_rules_input = dict(finding_rules_raw) if isinstance(finding_rules_raw, Mapping) else {}
    finding_rules: dict[str, dict[str, object]] = {}
    normalized_plots: list[dict[str, object]] = []
    for plot in plots:
        plot_id = str(plot.get("id") or "").strip()
        existing_rule = finding_rules_input.get(plot_id)
        rule = dict(existing_rule) if isinstance(existing_rule, Mapping) else dict(plot.get("finding_rule") or {})
        if not rule:
            rule = default_finding_rule((plot.get("plot_definition") or {}).get("mode"), report_config=report_config)
        finding_rules[plot_id] = rule
        normalized_plot = dict(plot)
        normalized_plot["finding_rule"] = dict(rule)
        normalized_plots.append(normalized_plot)
    target_policy = {"mode": "auto_new_arrivals"}
    if isinstance(source.get("target_policy"), Mapping):
        manual_targets = _normalize_text_list((source.get("target_policy") or {}).get("manual_target_serials"))
        if manual_targets:
            target_policy["manual_target_serials"] = manual_targets
    return {
        "id": pack_id,
        "name": name,
        "created_at": created_at,
        "updated_at": updated_at,
        "baseline_snapshot": normalized_baseline,
        "eligibility_filters": eligibility_filters,
        "target_policy": target_policy,
        "plots": normalized_plots,
        "finding_rules": finding_rules,
    }


def load_auto_graph_quickcheck_library(project_dir: Path) -> dict[str, object]:
    path = quickcheck_library_path(project_dir)
    if not path.exists():
        return {"version": LIBRARY_VERSION, "packs": [], "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    packs_raw = payload.get("packs") if isinstance(payload, Mapping) and isinstance(payload.get("packs"), list) else []
    packs = [
        _normalize_pack(item if isinstance(item, Mapping) else None, project_dir=project_dir)
        for item in packs_raw
    ]
    packs.sort(
        key=lambda item: (
            str(item.get("updated_at") or ""),
            str(item.get("name") or "").lower(),
        ),
        reverse=True,
    )
    return {"version": LIBRARY_VERSION, "packs": packs, "path": str(path)}


def _write_library(project_dir: Path, packs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    path = quickcheck_library_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": LIBRARY_VERSION,
        "packs": [_normalize_pack(pack if isinstance(pack, Mapping) else None, project_dir=project_dir) for pack in packs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"version": LIBRARY_VERSION, "packs": payload["packs"], "path": str(path)}


def save_auto_graph_quickcheck_pack(project_dir: Path, pack_payload: Mapping[str, object]) -> dict[str, object]:
    library = load_auto_graph_quickcheck_library(project_dir)
    normalized = _normalize_pack(pack_payload, project_dir=project_dir)
    normalized["updated_at"] = _now_text()
    packs = [dict(item) for item in (library.get("packs") or []) if isinstance(item, Mapping)]
    target_id = str(normalized.get("id") or "").strip()
    replaced = False
    for index, item in enumerate(packs):
        if str(item.get("id") or "").strip() == target_id:
            normalized["created_at"] = str(item.get("created_at") or normalized.get("created_at") or _now_text())
            packs[index] = normalized
            replaced = True
            break
    if not replaced:
        packs.append(normalized)
    _write_library(project_dir, packs)
    return normalized


def delete_auto_graph_quickcheck_pack(project_dir: Path, pack_id: str) -> dict[str, object]:
    target = str(pack_id or "").strip()
    library = load_auto_graph_quickcheck_library(project_dir)
    kept = [dict(item) for item in (library.get("packs") or []) if isinstance(item, Mapping) and str(item.get("id") or "").strip() != target]
    if len(kept) == len(library.get("packs") or []):
        raise RuntimeError("Quick-check pack was not found.")
    snapshot_dir = quickcheck_snapshot_path(project_dir, target).parent
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    _write_library(project_dir, kept)
    return {"deleted": True, "pack_id": target}


def rename_auto_graph_quickcheck_pack(project_dir: Path, pack_id: str, new_name: str) -> dict[str, object]:
    target = str(pack_id or "").strip()
    cleaned_name = str(new_name or "").strip() or "Quick-Check Pack"
    library = load_auto_graph_quickcheck_library(project_dir)
    packs = [dict(item) for item in (library.get("packs") or []) if isinstance(item, Mapping)]
    for index, item in enumerate(packs):
        if str(item.get("id") or "").strip() != target:
            continue
        item["name"] = cleaned_name
        item["updated_at"] = _now_text()
        packs[index] = _normalize_pack(item, project_dir=project_dir)
        _write_library(project_dir, packs)
        return {"pack_id": target, "name": cleaned_name}
    raise RuntimeError("Quick-check pack was not found.")


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    quoted = table_name.replace('"', '""')
    try:
        rows = conn.execute(f'PRAGMA table_info("{quoted}")').fetchall()
    except Exception:
        return set()
    return {str(row[1] or "").strip() for row in rows if str(row[1] or "").strip()}


def _quoted_ident(value: str) -> str:
    return '"' + str(value or "").replace('"', '""') + '"'


def _collect_observation_ids(conn: sqlite3.Connection, baseline_serials: Sequence[str]) -> set[str]:
    serials = [str(value).strip() for value in (baseline_serials or []) if str(value).strip()]
    if not serials:
        return set()
    placeholders = ",".join("?" for _ in serials)
    observation_ids: set[str] = set()
    for table_name in _table_names(conn):
        columns = _table_columns(conn, table_name)
        if not {"serial", "observation_id"}.issubset(columns):
            continue
        sql = f"SELECT DISTINCT observation_id FROM {_quoted_ident(table_name)} WHERE serial IN ({placeholders})"
        try:
            rows = conn.execute(sql, tuple(serials)).fetchall()
        except Exception:
            continue
        for row in rows:
            obs_id = str(row[0] or "").strip()
            if obs_id:
                observation_ids.add(obs_id)
    return observation_ids


def _prune_snapshot_db(snapshot_db: Path, baseline_serials: Sequence[str]) -> None:
    serials = [str(value).strip() for value in (baseline_serials or []) if str(value).strip()]
    if not serials:
        raise RuntimeError("Baseline snapshot requires at least one baseline serial.")
    placeholders = ",".join("?" for _ in serials)
    with closing(sqlite3.connect(str(snapshot_db))) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        observation_ids = _collect_observation_ids(conn, serials)
        observation_list = sorted(observation_ids)
        observation_placeholders = ",".join("?" for _ in observation_list)
        for table_name in _table_names(conn):
            columns = _table_columns(conn, table_name)
            if "serial" in columns:
                conn.execute(
                    f"DELETE FROM {_quoted_ident(table_name)} WHERE COALESCE(serial, '') NOT IN ({placeholders})",
                    tuple(serials),
                )
                continue
            if "observation_id" in columns:
                if observation_list:
                    conn.execute(
                        f"DELETE FROM {_quoted_ident(table_name)} WHERE COALESCE(observation_id, '') NOT IN ({observation_placeholders})",
                        tuple(observation_list),
                    )
                else:
                    conn.execute(f"DELETE FROM {_quoted_ident(table_name)}")
        conn.commit()


def _serial_catalog(db_path: Path) -> list[dict[str, object]]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    catalog: dict[str, dict[str, object]] = {}
    with closing(sqlite3.connect(str(path))) as conn:
        try:
            source_rows = conn.execute(
                "SELECT serial, COALESCE(last_ingested_epoch_ns, 0), COALESCE(status, '') FROM td_sources"
            ).fetchall()
        except Exception:
            source_rows = []
        for serial, last_ingested, status in source_rows:
            sn = str(serial or "").strip()
            if not sn:
                continue
            catalog[sn] = {
                "serial": sn,
                "last_ingested_epoch_ns": int(last_ingested or 0),
                "status": str(status or "").strip().lower(),
                "program_title": "",
                "control_periods": set(),
                "suppression_voltages": set(),
                "valve_voltages": set(),
            }
        try:
            meta_rows = conn.execute(
                "SELECT serial, COALESCE(program_title, '') FROM td_source_metadata"
            ).fetchall()
        except Exception:
            meta_rows = []
        for serial, program_title in meta_rows:
            sn = str(serial or "").strip()
            if not sn:
                continue
            entry = catalog.setdefault(
                sn,
                {
                    "serial": sn,
                    "last_ingested_epoch_ns": 0,
                    "status": "",
                    "program_title": "",
                    "control_periods": set(),
                    "suppression_voltages": set(),
                    "valve_voltages": set(),
                },
            )
            if not str(entry.get("program_title") or "").strip():
                entry["program_title"] = str(program_title or "").strip()
        for table_name in ("td_condition_observations", "td_condition_observations_sequences"):
            try:
                rows = conn.execute(
                    f"""
                    SELECT serial, control_period, suppression_voltage, valve_voltage
                    FROM {_quoted_ident(table_name)}
                    """
                ).fetchall()
            except Exception:
                rows = []
            for serial, control_period, suppression_voltage, valve_voltage in rows:
                sn = str(serial or "").strip()
                if not sn:
                    continue
                entry = catalog.setdefault(
                    sn,
                    {
                        "serial": sn,
                        "last_ingested_epoch_ns": 0,
                        "status": "",
                        "program_title": "",
                        "control_periods": set(),
                        "suppression_voltages": set(),
                        "valve_voltages": set(),
                    },
                )
                cp_text = _compact_value(control_period)
                if cp_text:
                    entry["control_periods"].add(cp_text)
                suppression_text = _compact_value(suppression_voltage)
                if suppression_text:
                    entry["suppression_voltages"].add(suppression_text)
                valve_text = _compact_value(valve_voltage)
                if valve_text:
                    entry["valve_voltages"].add(valve_text)
    out: list[dict[str, object]] = []
    for serial in sorted(catalog.keys(), key=str.casefold):
        entry = catalog[serial]
        if str(entry.get("status") or "").strip().lower() == "invalid":
            continue
        out.append(
            {
                "serial": str(entry.get("serial") or "").strip(),
                "last_ingested_epoch_ns": int(entry.get("last_ingested_epoch_ns") or 0),
                "program_title": str(entry.get("program_title") or "").strip(),
                "control_periods": sorted(str(value) for value in (entry.get("control_periods") or set()) if str(value).strip()),
                "suppression_voltages": sorted(str(value) for value in (entry.get("suppression_voltages") or set()) if str(value).strip()),
                "valve_voltages": sorted(str(value) for value in (entry.get("valve_voltages") or set()) if str(value).strip()),
            }
        )
    return out


def list_auto_graph_quickcheck_target_candidates(
    source_db_path: Path,
    pack: Mapping[str, object],
) -> list[str]:
    baseline_serials = set(_normalize_text_list(((pack.get("baseline_snapshot") or {}) if isinstance(pack.get("baseline_snapshot"), Mapping) else {}).get("baseline_serials")))
    filters = normalize_filter_state(pack.get("eligibility_filters"))
    out: list[str] = []
    for row in _serial_catalog(source_db_path):
        serial = str(row.get("serial") or "").strip()
        if not serial or serial in baseline_serials:
            continue
        if _filter_row_matches(row, filters, ignore_serials=True):
            out.append(serial)
    return out


def resolve_auto_graph_quickcheck_target_serials(
    source_db_path: Path,
    pack: Mapping[str, object],
    target_serials: Sequence[object] | None = None,
) -> list[str]:
    snapshot = dict(pack.get("baseline_snapshot") or {}) if isinstance(pack.get("baseline_snapshot"), Mapping) else {}
    captured_at_epoch_ns = int(snapshot.get("captured_at_epoch_ns") or 0)
    baseline_serials = set(_normalize_text_list(snapshot.get("baseline_serials")))
    filters = normalize_filter_state(pack.get("eligibility_filters"))
    available_rows = _serial_catalog(source_db_path)
    if target_serials is not None:
        wanted = {str(value or "").strip() for value in (target_serials or []) if str(value or "").strip()}
        resolved = [
            str(row.get("serial") or "").strip()
            for row in available_rows
            if str(row.get("serial") or "").strip() in wanted
            and str(row.get("serial") or "").strip() not in baseline_serials
            and _filter_row_matches(row, filters, ignore_serials=True)
        ]
        resolved.sort(key=str.casefold)
        return resolved
    out: list[str] = []
    for row in available_rows:
        serial = str(row.get("serial") or "").strip()
        if not serial or serial in baseline_serials:
            continue
        if int(row.get("last_ingested_epoch_ns") or 0) <= captured_at_epoch_ns:
            continue
        if _filter_row_matches(row, filters, ignore_serials=True):
            out.append(serial)
    out.sort(key=str.casefold)
    return out


def build_auto_graph_quickcheck_snapshot(
    project_dir: Path,
    source_db_path: Path,
    pack_id: str,
    baseline_filters: Mapping[str, object] | None,
    baseline_serials: Sequence[object] | None,
) -> dict[str, object]:
    target_pack_id = str(pack_id or "").strip()
    source_db = Path(source_db_path).expanduser()
    if not source_db.exists():
        raise FileNotFoundError(f"Project cache was not found: {source_db}")
    normalized_serials = _normalize_text_list(list(baseline_serials or []))
    if not normalized_serials:
        raise RuntimeError("Select at least one baseline serial before freezing a quick-check baseline.")
    library = load_auto_graph_quickcheck_library(project_dir)
    packs = [dict(item) for item in (library.get("packs") or []) if isinstance(item, Mapping)]
    for index, item in enumerate(packs):
        if str(item.get("id") or "").strip() != target_pack_id:
            continue
        snapshot_db = quickcheck_snapshot_path(project_dir, target_pack_id)
        snapshot_dir = snapshot_db.parent
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(source_db))) as src_conn, closing(sqlite3.connect(str(snapshot_db))) as dest_conn:
            src_conn.backup(dest_conn)
        _prune_snapshot_db(snapshot_db, normalized_serials)
        normalized_filters = normalize_filter_state(baseline_filters)
        eligibility_filters = dict(normalized_filters)
        eligibility_filters["serials"] = []
        baseline_snapshot = {
            "db_path": str(snapshot_db),
            "captured_at_epoch_ns": time.time_ns(),
            "source_db_mtime_ns": int(source_db.stat().st_mtime_ns),
            "baseline_filters": normalized_filters,
            "baseline_serials": list(normalized_serials),
        }
        item["baseline_snapshot"] = baseline_snapshot
        item["eligibility_filters"] = eligibility_filters
        item["updated_at"] = _now_text()
        packs[index] = _normalize_pack(item, project_dir=project_dir)
        _write_library(project_dir, packs)
        return dict(packs[index])
    raise RuntimeError("Quick-check pack was not found.")


def _status_counts(plot_results: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"FAIL": 0, "WATCH": 0, "PASS": 0, "NO_DATA": 0}
    for item in plot_results:
        status = str(item.get("status") or "NO_DATA").strip().upper()
        if status not in counts:
            status = "NO_DATA"
        counts[status] += 1
    return counts


def _overall_status(plot_results: Sequence[Mapping[str, object]]) -> str:
    if not plot_results:
        return "NO_DATA"
    best = "NO_DATA"
    for item in plot_results:
        status = str(item.get("status") or "NO_DATA").strip().upper()
        if STATUS_ORDER.get(status, 0) > STATUS_ORDER.get(best, 0):
            best = status
    return best


def _plot_runs(plot_definition: Mapping[str, object]) -> list[str]:
    member_runs = _normalize_text_list(plot_definition.get("member_runs"))
    if member_runs:
        return member_runs
    run_name = str(plot_definition.get("run") or "").strip()
    return [run_name] if run_name else []


def _zscore_status(rule: Mapping[str, object], *, zscore: float | None, abs_delta: float | None, pct_delta: float | None) -> str:
    watch_limit = _safe_float(rule.get("zscore_watch_max"))
    pass_limit = _safe_float(rule.get("zscore_pass_max"))
    abs_fail = _safe_float(rule.get("abs_fail_max"))
    abs_watch = _safe_float(rule.get("abs_watch_max"))
    pct_fail = _safe_float(rule.get("pct_fail_max"))
    pct_watch = _safe_float(rule.get("pct_watch_max"))
    if zscore is not None and watch_limit is not None and zscore > watch_limit:
        return "FAIL"
    if abs_delta is not None and abs_fail is not None and abs_delta > abs_fail:
        return "FAIL"
    if pct_delta is not None and pct_fail is not None and pct_delta > pct_fail:
        return "FAIL"
    if zscore is not None and pass_limit is not None and zscore > pass_limit:
        return "WATCH"
    if abs_delta is not None and abs_watch is not None and abs_delta > abs_watch:
        return "WATCH"
    if pct_delta is not None and pct_watch is not None and pct_delta > pct_watch:
        return "WATCH"
    return "PASS"


def _curve_status(rule: Mapping[str, object], metrics: Mapping[str, object]) -> str:
    for name in ("max_abs", "max_pct", "rms_pct"):
        value = _safe_float(metrics.get(name))
        fail_limit = _safe_float(rule.get(f"{name}_fail"))
        if value is not None and fail_limit is not None and value > fail_limit:
            return "FAIL"
    for name in ("max_abs", "max_pct", "rms_pct"):
        value = _safe_float(metrics.get(name))
        watch_limit = _safe_float(rule.get(f"{name}_watch"))
        if value is not None and watch_limit is not None and value > watch_limit:
            return "WATCH"
    return "PASS"


def _finite_curve_points(series: Mapping[str, object]) -> list[tuple[float, float]]:
    xs = series.get("x") if isinstance(series, Mapping) else []
    ys = series.get("y") if isinstance(series, Mapping) else []
    if not isinstance(xs, list) or not isinstance(ys, list):
        return []
    out: list[tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x_value = _safe_float(raw_x)
        y_value = _safe_float(raw_y)
        if x_value is None or y_value is None:
            continue
        out.append((float(x_value), float(y_value)))
    out.sort(key=lambda item: item[0])
    return out


def _interpolate(points: Sequence[tuple[float, float]], x_value: float) -> float | None:
    if len(points) < 2:
        return None
    if x_value < points[0][0] or x_value > points[-1][0]:
        return None
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        if x1 == x0:
            continue
        if x0 <= x_value <= x1:
            ratio = (x_value - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return None


def _compare_curve_sets(baseline_series: Sequence[Mapping[str, object]], target_series: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    baseline_points = [_finite_curve_points(item) for item in baseline_series]
    target_points = [_finite_curve_points(item) for item in target_series]
    baseline_points = [item for item in baseline_points if len(item) >= 2]
    target_points = [item for item in target_points if len(item) >= 2]
    if not baseline_points or not target_points:
        return None
    low = max(min(points[0][0] for points in baseline_points), min(points[0][0] for points in target_points))
    high = min(max(points[-1][0] for points in baseline_points), max(points[-1][0] for points in target_points))
    if high <= low:
        return None
    grid = [low + (high - low) * index / 39.0 for index in range(40)]
    baseline_mean: list[float] = []
    target_mean: list[float] = []
    for x_value in grid:
        baseline_values = [value for value in (_interpolate(points, x_value) for points in baseline_points) if value is not None]
        target_values = [value for value in (_interpolate(points, x_value) for points in target_points) if value is not None]
        if not baseline_values or not target_values:
            continue
        baseline_mean.append(float(sum(baseline_values) / max(1, len(baseline_values))))
        target_mean.append(float(sum(target_values) / max(1, len(target_values))))
    if len(baseline_mean) < 2 or len(target_mean) < 2:
        return None
    diffs = [abs(target - baseline) for baseline, target in zip(baseline_mean, target_mean)]
    baseline_scale = max(max(abs(value) for value in baseline_mean), 1e-9)
    baseline_rms_scale = max(sum(abs(value) for value in baseline_mean) / max(1, len(baseline_mean)), 1e-9)
    max_abs = max(diffs)
    max_pct = max((diff / max(abs(baseline), 1e-9)) * 100.0 for diff, baseline in zip(diffs, baseline_mean))
    rms_pct = math.sqrt(sum(diff * diff for diff in diffs) / max(1, len(diffs))) / baseline_rms_scale * 100.0
    return {
        "max_abs": float(max_abs),
        "max_pct": float(max_pct),
        "rms_pct": float(rms_pct),
        "grid_point_count": len(baseline_mean),
        "baseline_scale": float(baseline_scale),
    }


def _distribution_compare(baseline_values: Sequence[float], target_values: Sequence[float]) -> dict[str, object] | None:
    base = [float(value) for value in baseline_values if _safe_float(value) is not None]
    target = [float(value) for value in target_values if _safe_float(value) is not None]
    if not base or not target:
        return None
    baseline_mean = float(sum(base) / max(1, len(base)))
    target_mean = float(sum(target) / max(1, len(target)))
    abs_delta = abs(target_mean - baseline_mean)
    pct_delta = abs_delta / max(abs(baseline_mean), 1e-9) * 100.0 if baseline_mean != 0.0 else (0.0 if abs_delta == 0.0 else math.inf)
    baseline_std = float(statistics.pstdev(base)) if len(base) >= 2 else 0.0
    if baseline_std <= 1e-12:
        zscore = 0.0 if abs_delta <= 1e-12 else math.inf
    else:
        zscore = abs_delta / baseline_std
    return {
        "baseline_mean": baseline_mean,
        "target_mean": target_mean,
        "baseline_std": baseline_std,
        "abs_delta": abs_delta,
        "pct_delta": pct_delta,
        "zscore": zscore,
        "baseline_count": len(base),
        "target_count": len(target),
    }


def _curve_plot_result(
    source_db_path: Path,
    baseline_db_path: Path,
    plot: Mapping[str, object],
    baseline_serials: Sequence[str],
    target_serials: Sequence[str],
) -> dict[str, object]:
    be = _backend()
    plot_definition = dict(plot.get("plot_definition") or {})
    rule = dict(plot.get("finding_rule") or {})
    runs = _plot_runs(plot_definition)
    y_names = _normalize_text_list(plot_definition.get("y"))
    x_name = str(plot_definition.get("x") or "Time").strip() or "Time"
    best_metrics: dict[str, object] | None = None
    warning_text = ""
    for run_name in runs:
        for y_name in y_names:
            try:
                baseline_series = be.td_load_curves(baseline_db_path, run_name, y_name, x_name, serials=list(baseline_serials))
                target_series = be.td_load_curves(source_db_path, run_name, y_name, x_name, serials=list(target_serials))
            except Exception as exc:
                warning_text = str(exc)
                continue
            metrics = _compare_curve_sets(baseline_series, target_series)
            if metrics is None:
                continue
            if best_metrics is None or _safe_float(metrics.get("max_pct") or 0.0) >= _safe_float(best_metrics.get("max_pct") or 0.0):
                best_metrics = {
                    **metrics,
                    "run_name": run_name,
                    "y_name": y_name,
                    "baseline_series_count": len(baseline_series),
                    "target_series_count": len(target_series),
                }
    if best_metrics is None:
        return {
            "plot_id": str(plot.get("id") or "").strip(),
            "plot_name": str(plot.get("name") or "").strip(),
            "mode": "curves",
            "status": "NO_DATA",
            "summary_text": "No qualifying curve data was available.",
            "grading_metrics": {},
            "warning_text": warning_text or "No qualifying curve data was available.",
            "render_context": {},
        }
    status = _curve_status(rule, best_metrics)
    summary_text = (
        f"{best_metrics.get('y_name') or ''} | {best_metrics.get('run_name') or ''} | "
        f"max_abs={best_metrics.get('max_abs'):.4g}, max_pct={best_metrics.get('max_pct'):.4g}, rms_pct={best_metrics.get('rms_pct'):.4g}"
    )
    return {
        "plot_id": str(plot.get("id") or "").strip(),
        "plot_name": str(plot.get("name") or "").strip(),
        "mode": "curves",
        "status": status,
        "summary_text": summary_text,
        "grading_metrics": best_metrics,
        "warning_text": warning_text,
        "render_context": {
            "plot_definition": plot_definition,
        },
    }


def _metric_like_plot_result(
    source_db_path: Path,
    baseline_db_path: Path,
    plot: Mapping[str, object],
    baseline_serials: Sequence[str],
    target_serials: Sequence[str],
) -> dict[str, object]:
    be = _backend()
    plot_definition = dict(plot.get("plot_definition") or {})
    rule = dict(plot.get("finding_rule") or {})
    mode = str(plot_definition.get("mode") or "").strip().lower()
    runs = _plot_runs(plot_definition)
    best_compare: dict[str, object] | None = None
    warning_text = ""
    if mode == "metrics":
        stats = _normalize_text_list(plot_definition.get("stats")) or [str(plot_definition.get("view_stat") or "mean").strip().lower() or "mean"]
        y_names = _normalize_text_list(plot_definition.get("y"))
        metric_source = plot_definition.get("metric_plot_source")
        for run_name in runs:
            for y_name in y_names:
                for stat in stats:
                    try:
                        baseline_rows = be.td_load_metric_series(
                            baseline_db_path,
                            run_name,
                            y_name,
                            stat,
                            metric_source=metric_source,
                        )
                        target_rows = be.td_load_metric_series(
                            source_db_path,
                            run_name,
                            y_name,
                            stat,
                            metric_source=metric_source,
                        )
                    except Exception as exc:
                        warning_text = str(exc)
                        continue
                    compare = _distribution_compare(
                        [_safe_float(row.get("value_num")) for row in baseline_rows if isinstance(row, Mapping)],
                        [_safe_float(row.get("value_num")) for row in target_rows if isinstance(row, Mapping) and str(row.get("serial") or "").strip() in set(target_serials)],
                    )
                    if compare is None:
                        continue
                    compare.update({"run_name": run_name, "y_name": y_name, "stat": stat})
                    if best_compare is None or _safe_float(compare.get("zscore")) >= _safe_float(best_compare.get("zscore")):
                        best_compare = compare
    elif mode == "life_metrics":
        plot_type = str(plot_definition.get("plot_type") or "life_axis").strip().lower()
        y_name = str(plot_definition.get("y_parameter") or "").strip()
        try:
            if plot_type == "metric_xy":
                x_name = str(plot_definition.get("x_parameter") or "").strip()
                baseline_rows = be.td_load_life_metric_xy(baseline_db_path, runs, x_name, y_name, serials=list(baseline_serials))
                target_rows = be.td_load_life_metric_xy(source_db_path, runs, x_name, y_name, serials=list(target_serials))
                compare = _distribution_compare(
                    [_safe_float(row.get("y_value")) for row in baseline_rows if isinstance(row, Mapping)],
                    [_safe_float(row.get("y_value")) for row in target_rows if isinstance(row, Mapping)],
                )
                if compare is not None:
                    compare.update({"run_name": ", ".join(runs), "y_name": y_name, "x_name": x_name})
                    best_compare = compare
            else:
                life_axis = str(plot_definition.get("life_axis") or "sequence_index").strip() or "sequence_index"
                baseline_rows = be.td_load_life_metric_series(baseline_db_path, runs, y_name, life_axis, serials=list(baseline_serials))
                target_rows = be.td_load_life_metric_series(source_db_path, runs, y_name, life_axis, serials=list(target_serials))
                compare = _distribution_compare(
                    [_safe_float(row.get("y_value")) for row in baseline_rows if isinstance(row, Mapping)],
                    [_safe_float(row.get("y_value")) for row in target_rows if isinstance(row, Mapping)],
                )
                if compare is not None:
                    compare.update({"run_name": ", ".join(runs), "y_name": y_name, "life_axis": life_axis})
                    best_compare = compare
        except Exception as exc:
            warning_text = str(exc)
    else:
        stat = str(plot_definition.get("view_stat") or ((plot_definition.get("stats") or ["mean"])[0] if isinstance(plot_definition.get("stats"), list) else "mean")).strip().lower() or "mean"
        output_name = str(plot_definition.get("output") or "").strip()
        run_type_filter = plot_definition.get("performance_run_type_mode")
        control_period_filter = None
        if (
            str(plot_definition.get("performance_filter_mode") or "").strip().lower() == "match_control_period"
            and str(plot_definition.get("performance_run_type_mode") or "").strip().lower() == "pulsed_mode"
        ):
            control_period_filter = plot_definition.get("selected_control_period")
        for run_name in runs:
            try:
                baseline_rows = be.td_load_metric_series(
                    baseline_db_path,
                    run_name,
                    output_name,
                    stat,
                    run_type_filter=run_type_filter,
                    control_period_filter=control_period_filter,
                    metric_source=getattr(be, "TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES", "all_sequences"),
                )
                target_rows = be.td_load_metric_series(
                    source_db_path,
                    run_name,
                    output_name,
                    stat,
                    run_type_filter=run_type_filter,
                    control_period_filter=control_period_filter,
                    metric_source=getattr(be, "TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES", "all_sequences"),
                )
            except Exception as exc:
                warning_text = str(exc)
                continue
            compare = _distribution_compare(
                [_safe_float(row.get("value_num")) for row in baseline_rows if isinstance(row, Mapping)],
                [_safe_float(row.get("value_num")) for row in target_rows if isinstance(row, Mapping) and str(row.get("serial") or "").strip() in set(target_serials)],
            )
            if compare is None:
                continue
            compare.update({"run_name": run_name, "output": output_name, "stat": stat})
            if best_compare is None or _safe_float(compare.get("zscore")) >= _safe_float(best_compare.get("zscore")):
                best_compare = compare
    if best_compare is None:
        return {
            "plot_id": str(plot.get("id") or "").strip(),
            "plot_name": str(plot.get("name") or "").strip(),
            "mode": mode,
            "status": "NO_DATA",
            "summary_text": "No qualifying comparison data was available.",
            "grading_metrics": {},
            "warning_text": warning_text or "No qualifying comparison data was available.",
            "render_context": {"plot_definition": plot_definition},
        }
    status = _zscore_status(
        rule,
        zscore=_safe_float(best_compare.get("zscore")),
        abs_delta=_safe_float(best_compare.get("abs_delta")),
        pct_delta=_safe_float(best_compare.get("pct_delta")),
    )
    summary_text = (
        f"{best_compare.get('run_name') or ''} | "
        f"z={best_compare.get('zscore'):.4g}, abs={best_compare.get('abs_delta'):.4g}, pct={best_compare.get('pct_delta'):.4g}"
    )
    return {
        "plot_id": str(plot.get("id") or "").strip(),
        "plot_name": str(plot.get("name") or "").strip(),
        "mode": mode,
        "status": status,
        "summary_text": summary_text,
        "grading_metrics": best_compare,
        "warning_text": warning_text,
        "render_context": {"plot_definition": plot_definition},
    }


def run_auto_graph_quickcheck_pack(
    project_dir: Path,
    source_db_path: Path,
    pack_id: str,
    target_serials: Sequence[object] | None = None,
) -> dict[str, object]:
    library = load_auto_graph_quickcheck_library(project_dir)
    packs = [dict(item) for item in (library.get("packs") or []) if isinstance(item, Mapping)]
    pack = next((dict(item) for item in packs if str(item.get("id") or "").strip() == str(pack_id or "").strip()), None)
    if pack is None:
        raise RuntimeError("Quick-check pack was not found.")
    snapshot = dict(pack.get("baseline_snapshot") or {})
    baseline_db_path = Path(str(snapshot.get("db_path") or "")).expanduser()
    baseline_serials = _normalize_text_list(snapshot.get("baseline_serials"))
    resolved_targets = resolve_auto_graph_quickcheck_target_serials(source_db_path, pack, target_serials=target_serials)
    warnings: list[str] = []
    if not baseline_db_path.exists():
        warnings.append("Baseline snapshot is missing. Rebuild the baseline before running this quick check.")
    if not resolved_targets:
        warnings.append("No qualifying target serials were found after the last Update Project refresh.")
    plot_results: list[dict[str, object]] = []
    for plot in [dict(item) for item in (pack.get("plots") or []) if isinstance(item, Mapping)]:
        plot_definition = dict(plot.get("plot_definition") or {})
        mode = str(plot_definition.get("mode") or "").strip().lower()
        if not baseline_db_path.exists() or not resolved_targets:
            plot_results.append(
                {
                    "plot_id": str(plot.get("id") or "").strip(),
                    "plot_name": str(plot.get("name") or "").strip(),
                    "mode": mode,
                    "status": "NO_DATA",
                    "summary_text": "No qualifying target serials were found." if resolved_targets == [] else "Baseline snapshot is missing.",
                    "grading_metrics": {},
                    "warning_text": warnings[-1] if warnings else "",
                    "render_context": {"plot_definition": plot_definition},
                }
            )
            continue
        if mode == "curves":
            result = _curve_plot_result(source_db_path, baseline_db_path, plot, baseline_serials, resolved_targets)
        else:
            result = _metric_like_plot_result(source_db_path, baseline_db_path, plot, baseline_serials, resolved_targets)
        plot_results.append(result)
        if str(result.get("warning_text") or "").strip():
            warnings.append(str(result.get("warning_text") or "").strip())
    warnings = list(dict.fromkeys([item for item in warnings if str(item).strip()]))
    summary_counts = _status_counts(plot_results)
    return {
        "pack_id": str(pack.get("id") or "").strip(),
        "pack_name": str(pack.get("name") or "").strip(),
        "baseline_snapshot": snapshot,
        "target_serials": resolved_targets,
        "overall_status": _overall_status(plot_results),
        "summary_counts": summary_counts,
        "warnings": warnings,
        "plot_results": plot_results,
    }
