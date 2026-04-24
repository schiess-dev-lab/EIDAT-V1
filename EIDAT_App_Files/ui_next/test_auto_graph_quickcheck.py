import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ui_next import auto_graph_quickcheck as agq  # type: ignore
from ui_next import backend  # type: ignore


def _insert_serial_rows(
    conn: sqlite3.Connection,
    *,
    serial: str,
    last_ingested_epoch_ns: int,
    program_title: str,
    control_period: float = 10.0,
    suppression_voltage: float = 5.0,
    valve_voltage: float = 28.0,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO td_sources(
            serial, sqlite_path, mtime_ns, size_bytes, status, last_ingested_epoch_ns, raw_fingerprint
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (serial, f"{serial}.sqlite3", 1, 1, "ok", int(last_ingested_epoch_ns), f"fp-{serial}"),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO td_source_metadata(
            serial, program_title, document_type, metadata_rel, artifacts_rel, excel_sqlite_rel, metadata_mtime_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (serial, program_title, "TD", f"{serial}.json", f"{serial}/artifacts", f"{serial}.sqlite3", 1),
    )
    conn.execute(
        "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
        ("RunA", "time", "Run A", "PM", float(control_period), 0.5),
    )
    conn.execute(
        "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
        ("RunA", "Pressure", "psi", "y"),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO td_condition_observations(
            observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width,
            control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (f"agg-{serial}", serial, "RunA", program_title, "Aggregate", "PM", 0.5, control_period, suppression_voltage, valve_voltage, 1, 1),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO td_condition_observations_sequences(
            observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width,
            control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (f"seq-{serial}", serial, "RunA", program_title, f"Seq-{serial}", "PM", 0.5, control_period, suppression_voltage, valve_voltage, 1, 1),
    )
    conn.commit()


def _build_cache_db(db_path: Path) -> Path:
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        _insert_serial_rows(
            conn,
            serial="SN-BASE",
            last_ingested_epoch_ns=1,
            program_title="Program A",
        )
    return db_path


class TestAutoGraphQuickcheck(unittest.TestCase):
    def test_snapshot_prunes_to_baseline_and_resolves_new_arrivals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            db_path = _build_cache_db(root / "cache.sqlite3")

            saved = agq.save_auto_graph_quickcheck_pack(
                project_dir,
                {"name": "Baseline Pack", "plots": []},
            )
            frozen = agq.build_auto_graph_quickcheck_snapshot(
                project_dir,
                db_path,
                str(saved.get("id") or "").strip(),
                {
                    "programs": ["Program A"],
                    "serials": ["SN-BASE"],
                    "control_periods": ["10"],
                    "suppression_voltages": ["5"],
                    "valve_voltages": ["28"],
                },
                ["SN-BASE"],
            )

            snapshot_db = Path(str((frozen.get("baseline_snapshot") or {}).get("db_path") or "")).expanduser()
            self.assertTrue(snapshot_db.exists())
            with closing(sqlite3.connect(str(snapshot_db))) as conn:
                tables = {
                    str(row[0] or "").strip()
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                self.assertIn("td_sources", tables)
                self.assertIn("td_source_metadata", tables)
                self.assertIn("td_condition_observations", tables)
                self.assertIn("td_condition_observations_sequences", tables)
                source_serials = [
                    str(row[0] or "").strip()
                    for row in conn.execute("SELECT serial FROM td_sources ORDER BY serial").fetchall()
                ]
                self.assertEqual(source_serials, ["SN-BASE"])

            captured_at = int((frozen.get("baseline_snapshot") or {}).get("captured_at_epoch_ns") or 0)
            with closing(sqlite3.connect(str(db_path))) as conn:
                _insert_serial_rows(
                    conn,
                    serial="SN-NEW",
                    last_ingested_epoch_ns=captured_at + 1_000,
                    program_title="Program A",
                )
                _insert_serial_rows(
                    conn,
                    serial="SN-SKIP",
                    last_ingested_epoch_ns=captured_at + 2_000,
                    program_title="Program B",
                )

            candidates = agq.list_auto_graph_quickcheck_target_candidates(db_path, frozen)
            self.assertEqual(candidates, ["SN-NEW"])
            resolved = agq.resolve_auto_graph_quickcheck_target_serials(db_path, frozen)
            self.assertEqual(resolved, ["SN-NEW"])

    def test_run_pack_grades_modes_and_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            db_path = _build_cache_db(root / "cache.sqlite3")
            pack = agq.save_auto_graph_quickcheck_pack(
                project_dir,
                {
                    "name": "Runner",
                    "plots": [
                        {
                            "id": "curve-1",
                            "name": "Curve Check",
                            "plot_definition": {
                                "mode": "curves",
                                "run": "RunA",
                                "y": ["Pressure"],
                                "x": "time",
                            },
                            "finding_rule": {
                                "mode": "curve_thresholds",
                                "max_abs_watch": 2.0,
                                "max_abs_fail": 4.0,
                                "max_pct_watch": 10.0,
                                "max_pct_fail": 20.0,
                                "rms_pct_watch": 10.0,
                                "rms_pct_fail": 20.0,
                            },
                        },
                        {
                            "id": "metric-1",
                            "name": "Metric Check",
                            "plot_definition": {
                                "mode": "metrics",
                                "run": "RunA",
                                "y": ["Pressure"],
                                "stats": ["mean"],
                                "metric_plot_source": backend.TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES,
                            },
                            "finding_rule": {
                                "mode": "zscore",
                                "zscore_pass_max": 2.0,
                                "zscore_watch_max": 3.0,
                                "abs_watch_max": None,
                                "abs_fail_max": None,
                                "pct_watch_max": None,
                                "pct_fail_max": None,
                            },
                        },
                        {
                            "id": "life-1",
                            "name": "Life Check",
                            "plot_definition": {
                                "mode": "life_metrics",
                                "member_runs": ["RunA"],
                                "plot_type": "life_axis",
                                "y_parameter": "LifeMetric",
                                "life_axis": "sequence_index",
                            },
                            "finding_rule": {
                                "mode": "zscore",
                                "zscore_pass_max": 2.0,
                                "zscore_watch_max": 3.0,
                                "abs_watch_max": None,
                                "abs_fail_max": None,
                                "pct_watch_max": None,
                                "pct_fail_max": None,
                            },
                        },
                        {
                            "id": "perf-1",
                            "name": "Performance Check",
                            "plot_definition": {
                                "mode": "performance",
                                "run": "RunA",
                                "member_runs": ["RunA"],
                                "output": "Pressure",
                                "view_stat": "mean",
                                "performance_run_type_mode": "pulsed_mode",
                                "performance_filter_mode": "match_control_period",
                                "selected_control_period": 10,
                            },
                            "finding_rule": {
                                "mode": "zscore",
                                "zscore_pass_max": 2.0,
                                "zscore_watch_max": 3.0,
                                "abs_watch_max": None,
                                "abs_fail_max": None,
                                "pct_watch_max": None,
                                "pct_fail_max": None,
                            },
                        },
                    ],
                },
            )
            frozen = agq.build_auto_graph_quickcheck_snapshot(
                project_dir,
                db_path,
                str(pack.get("id") or "").strip(),
                {"programs": ["Program A"], "serials": ["SN-BASE"]},
                ["SN-BASE"],
            )
            snapshot_db = Path(str((frozen.get("baseline_snapshot") or {}).get("db_path") or "")).expanduser()
            captured_at = int((frozen.get("baseline_snapshot") or {}).get("captured_at_epoch_ns") or 0)
            with closing(sqlite3.connect(str(db_path))) as conn:
                _insert_serial_rows(
                    conn,
                    serial="SN-NEW",
                    last_ingested_epoch_ns=captured_at + 1_000,
                    program_title="Program A",
                )

            def _fake_curves(db_arg, run_name, y_name, x_name, serials=None):
                _ = (run_name, y_name, x_name, serials)
                if Path(db_arg).expanduser() == snapshot_db:
                    return [{"x": [0.0, 1.0, 2.0], "y": [10.0, 10.0, 10.0]}]
                return [{"x": [0.0, 1.0, 2.0], "y": [16.0, 16.0, 16.0]}]

            def _fake_metric_series(
                db_arg,
                run_name,
                column_name,
                stat,
                metric_source=None,
                run_type_filter=None,
                control_period_filter=None,
                **_kwargs,
            ):
                _ = (run_name, column_name, stat, metric_source, control_period_filter)
                if run_type_filter is not None:
                    return []
                if Path(db_arg).expanduser() == snapshot_db:
                    return [
                        {"serial": "SN-BASE", "value_num": 10.0},
                        {"serial": "SN-BASE", "value_num": 12.0},
                    ]
                return [{"serial": "SN-NEW", "value_num": 30.0}]

            def _fake_life_metric_series(db_arg, run_names, parameter, life_axis, serials=None):
                _ = (run_names, parameter, life_axis, serials)
                if Path(db_arg).expanduser() == snapshot_db:
                    return [{"y_value": 10.0}, {"y_value": 11.0}]
                return [{"y_value": 10.4}, {"y_value": 10.6}]

            with mock.patch.object(backend, "td_load_curves", side_effect=_fake_curves), mock.patch.object(
                backend,
                "td_load_metric_series",
                side_effect=_fake_metric_series,
            ), mock.patch.object(
                backend,
                "td_load_life_metric_series",
                side_effect=_fake_life_metric_series,
            ):
                result = agq.run_auto_graph_quickcheck_pack(
                    project_dir,
                    db_path,
                    str(pack.get("id") or "").strip(),
                )

            self.assertEqual(result.get("target_serials"), ["SN-NEW"])
            self.assertEqual(result.get("overall_status"), "FAIL")
            self.assertEqual(result.get("summary_counts"), {"FAIL": 2, "WATCH": 0, "PASS": 1, "NO_DATA": 1})
            modes = {str(item.get("mode") or ""): str(item.get("status") or "") for item in (result.get("plot_results") or [])}
            self.assertEqual(modes.get("curves"), "FAIL")
            self.assertEqual(modes.get("metrics"), "FAIL")
            self.assertEqual(modes.get("life_metrics"), "PASS")
            self.assertEqual(modes.get("performance"), "NO_DATA")


if __name__ == "__main__":
    unittest.main()
