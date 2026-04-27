import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


def _create_metric_source_db(tmpdir: str) -> Path:
    db_path = Path(tmpdir) / "metric_sources.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("CondA", "time", "Condition A", "PM", 10.0, 0.5),
                ("CondSS", "time", "Condition SS", "SS", None, None),
            ],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
            [
                ("CondA", "Pressure", "psi", "y"),
                ("CondSS", "Pressure", "psi", "y"),
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_condition_observations(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("agg-1", "SN-001", "CondA", "Program Alpha", "Aggregate", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
                ("agg-ss-1", "SN-002", "CondSS", "Program Beta", "Aggregate", "SS", None, None, None, None, 1, 1),
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_metrics_calc(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("agg-1", "SN-001", "CondA", "Pressure", "mean", 1.5, 1, 1, "Program Alpha", "Aggregate"),
                ("agg-ss-1", "SN-002", "CondSS", "Pressure", "mean", 3.5, 1, 1, "Program Beta", "Aggregate"),
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_condition_observations_sequences(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("seq-1", "SN-001", "CondA", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
                ("seq-2", "SN-001", "CondA", "Program Alpha", "Seq-2", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
                ("seq-ss-1", "SN-002", "CondSS", "Program Beta", "Seq-SS-1", "SS", None, None, None, None, 1, 1),
            ],
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_metrics_calc_sequences(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("seq-1", "SN-001", "CondA", "Pressure", "mean", 1.0, 1, 1, "Program Alpha", "Seq-1"),
                ("seq-2", "SN-001", "CondA", "Pressure", "mean", 2.0, 1, 1, "Program Alpha", "Seq-2"),
                ("seq-ss-1", "SN-002", "CondSS", "Pressure", "mean", 3.0, 1, 1, "Program Beta", "Seq-SS-1"),
            ],
        )
        conn.commit()
    return db_path


def _create_plotter_curve_db(tmpdir: str) -> Path:
    db_path = Path(tmpdir) / "plotter_curves.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {backend.TD_PLOTTER_CURVE_CATALOG_TABLE}(
                run_name, parameter_name, units, x_axis_kind, display_name, source_kind, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("CondA", "Pressure", "psi", "time", "Pressure", "raw_cache", 1),
        )
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {backend.TD_PLOTTER_OBSERVATIONS_TABLE}(
                observation_id, run_name, serial, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("plot-obs-1", "CondA", "SN-001", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
        )
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {backend.TD_PLOTTER_CURVES_TABLE}(
                run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points,
                source_mtime_ns, computed_epoch_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "CondA",
                "Pressure",
                "time",
                "plot-obs-1",
                "SN-001",
                json.dumps([0.0, 1.0, 2.0]),
                json.dumps([1.0, 1.1, 1.2]),
                3,
                1,
                1,
                "Program Alpha",
                "Seq-1",
            ),
        )
        conn.commit()
    return db_path


def _create_raw_curve_db(tmpdir: str) -> Path:
    db_path = Path(tmpdir) / "raw_curves.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_raw_cache_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO td_raw_condition_observations(
                observation_id, run_name, serial, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-obs-1", "CondA", "SN-001", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_raw_curve_catalog(
                run_name, parameter_name, units, x_axis_kind, table_name, display_name, source_kind, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("CondA", "Pressure", "psi", "time", "td_curves_raw", "Pressure", "raw_cache", 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_curves_raw(
                run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points,
                source_mtime_ns, computed_epoch_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "CondA",
                "Pressure",
                "time",
                "raw-obs-1",
                "SN-001",
                json.dumps([0.0, 1.0, 2.0]),
                json.dumps([1.0, 1.1, 1.2]),
                3,
                1,
                1,
                "Program Alpha",
                "Seq-1",
            ),
        )
        conn.commit()
    return db_path


def _create_legacy_curve_db(tmpdir: str) -> Path:
    db_path = Path(tmpdir) / "legacy_curves.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO td_condition_observations(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("legacy-obs-1", "SN-001", "CondA", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_curves(
                run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points,
                source_mtime_ns, computed_epoch_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "CondA",
                "Pressure",
                "time",
                "legacy-obs-1",
                "SN-001",
                json.dumps([0.0, 1.0, 2.0]),
                json.dumps([1.0, 1.1, 1.2]),
                3,
                1,
                1,
                "Program Alpha",
                "Seq-1",
            ),
        )
        conn.commit()
    return db_path


def _create_life_metric_db(tmpdir: str) -> Path:
    db_path = Path(tmpdir) / "life_metrics.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_condition_observations_sequences(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("life-1", "SN-001", "CondLife", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
                ("life-2", "SN-001", "CondLife", "Program Alpha", "Seq-2", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
            ],
        )
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {backend.TD_LIFE_METRICS_TABLE}(
                observation_id, serial, sequence_index, sequence_label, condition_key, condition_display,
                program_title, source_run_name, parameter_name, stat, value_num, units,
                sequence_pulses, cumulative_pulses, sequence_on_time, cumulative_on_time,
                sequence_elapsed_time, cumulative_elapsed_time, sequence_throughput, cumulative_throughput,
                sequence_impulse, cumulative_impulse, diagnostics, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "life-1",
                    "SN-001",
                    1,
                    "Seq-1",
                    "CondLife",
                    "Condition Life",
                    "Program Alpha",
                    "Seq-1",
                    "Thrust",
                    "mean",
                    10.0,
                    "mN",
                    100.0,
                    100.0,
                    1.0,
                    1.0,
                    10.0,
                    10.0,
                    0.5,
                    0.5,
                    1.5,
                    1.5,
                    "",
                    1,
                    1,
                ),
                (
                    "life-1",
                    "SN-001",
                    1,
                    "Seq-1",
                    "CondLife",
                    "Condition Life",
                    "Program Alpha",
                    "Seq-1",
                    "Pulse Count",
                    "mean",
                    100.0,
                    "count",
                    100.0,
                    100.0,
                    1.0,
                    1.0,
                    10.0,
                    10.0,
                    0.5,
                    0.5,
                    1.5,
                    1.5,
                    "",
                    1,
                    1,
                ),
                (
                    "life-2",
                    "SN-001",
                    2,
                    "Seq-2",
                    "CondLife",
                    "Condition Life",
                    "Program Alpha",
                    "Seq-2",
                    "Thrust",
                    "mean",
                    12.0,
                    "mN",
                    150.0,
                    250.0,
                    2.0,
                    3.0,
                    20.0,
                    30.0,
                    0.75,
                    1.25,
                    2.5,
                    4.0,
                    "",
                    1,
                    1,
                ),
                (
                    "life-2",
                    "SN-001",
                    2,
                    "Seq-2",
                    "CondLife",
                    "Condition Life",
                    "Program Alpha",
                    "Seq-2",
                    "Pulse Count",
                    "mean",
                    150.0,
                    "count",
                    150.0,
                    250.0,
                    2.0,
                    3.0,
                    20.0,
                    30.0,
                    0.75,
                    1.25,
                    2.5,
                    4.0,
                    "",
                    1,
                    1,
                ),
            ],
        )
        conn.commit()
    return db_path


class TestBackendTdMetricSources(unittest.TestCase):
    def test_aggregate_metric_source_ignores_sequence_filters(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            rows = backend.td_load_metric_series(
                db_path,
                "CondA",
                "Pressure",
                "mean",
                program_title="Program Alpha",
                source_run_name="Seq-1",
                metric_source=backend.TD_METRIC_PLOT_SOURCE_AGGREGATE,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["observation_id"], "agg-1")
            self.assertEqual(rows[0]["source_run_name"], "Aggregate")
            self.assertEqual(rows[0]["valve_voltage"], 28.0)

    def test_all_sequences_metric_source_returns_per_sequence_rows_and_filters(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            rows = backend.td_load_metric_series(
                db_path,
                "CondA",
                "Pressure",
                "mean",
                metric_source=backend.TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES,
            )
            self.assertEqual([row["observation_id"] for row in rows], ["seq-1", "seq-2"])
            self.assertEqual([row["valve_voltage"] for row in rows], [28.0, 28.0])

            filtered = backend.td_load_metric_series(
                db_path,
                "CondA",
                "Pressure",
                "mean",
                program_title="Program Alpha",
                source_run_name="Seq-1",
                metric_source=backend.TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES,
            )
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["observation_id"], "seq-1")
            self.assertEqual(filtered[0]["source_run_name"], "Seq-1")
            self.assertEqual(filtered[0]["valve_voltage"], 28.0)

    def test_run_selection_views_and_filter_rows_prefer_sequence_observations(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            views = backend.td_list_run_selection_views(db_path, Path("project.xlsx"))
            self.assertEqual([item["source_run_name"] for item in views["sequence"]], ["Seq-1", "Seq-2", "Seq-SS-1"])
            self.assertEqual([item["run_name"] for item in views["condition"]], ["CondA", "CondSS"])
            self.assertEqual(views["condition"][0]["display_text"], "Condition A")
            self.assertEqual(views["condition"][0]["member_sequences"], ["Seq-1", "Seq-2"])
            self.assertEqual(views["condition"][0]["member_valve_voltages"], ["28"])
            self.assertEqual(views["condition"][0]["valve_voltage"], "28")
            self.assertEqual(views["condition"][1]["display_text"], "Condition SS")
            self.assertEqual(views["condition"][1]["member_sequences"], ["Seq-SS-1"])

            filter_rows = backend.td_read_observation_filter_rows_from_cache(db_path)
            self.assertEqual([row["source_run_name"] for row in filter_rows], ["Seq-1", "Seq-2", "Seq-SS-1"])
            self.assertEqual([row["valve_voltage"] for row in filter_rows], [28.0, 28.0, None])

    def test_steady_state_rows_remain_available_in_aggregate_and_sequence_metric_sources(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)

            aggregate_rows = backend.td_load_metric_series(
                db_path,
                "CondSS",
                "Pressure",
                "mean",
                run_type_filter="steady_state",
                metric_source=backend.TD_METRIC_PLOT_SOURCE_AGGREGATE,
            )
            self.assertEqual(len(aggregate_rows), 1)
            self.assertEqual(aggregate_rows[0]["observation_id"], "agg-ss-1")
            self.assertEqual(aggregate_rows[0]["run_type"], "SS")
            self.assertIsNone(aggregate_rows[0]["valve_voltage"])

            sequence_rows = backend.td_load_metric_series(
                db_path,
                "CondSS",
                "Pressure",
                "mean",
                run_type_filter="steady_state",
                metric_source=backend.TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES,
            )
            self.assertEqual(len(sequence_rows), 1)
            self.assertEqual(sequence_rows[0]["observation_id"], "seq-ss-1")
            self.assertEqual(sequence_rows[0]["source_run_name"], "Seq-SS-1")
            self.assertEqual(sequence_rows[0]["run_type"], "SS")
            self.assertIsNone(sequence_rows[0]["valve_voltage"])

    def test_control_period_filter_does_not_exclude_steady_state_sequence_rows(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            rows = backend.td_load_metric_series(
                db_path,
                "CondSS",
                "Pressure",
                "mean",
                control_period_filter=10.0,
                metric_source=backend.TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["observation_id"], "seq-ss-1")
            self.assertIsNone(rows[0]["valve_voltage"])

    def test_performance_run_type_modes_still_report_steady_state_and_pulsed_mode(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            self.assertEqual(
                backend.td_list_performance_run_type_modes(db_path),
                ["steady_state", "pulsed_mode"],
            )

    def test_curve_loaders_return_valve_voltage_across_plotter_raw_and_legacy_paths(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            plotter_db = _create_plotter_curve_db(tmpdir)
            found_plotter, plotter_rows = backend._td_try_load_plotter_curves(
                plotter_db,
                "CondA",
                "Pressure",
                "time",
                serials=["SN-001"],
                program_title="Program Alpha",
                source_run_name="Seq-1",
            )
            self.assertTrue(found_plotter)
            self.assertEqual(len(plotter_rows), 1)
            self.assertEqual(plotter_rows[0]["valve_voltage"], 28.0)

            raw_db = _create_raw_curve_db(tmpdir)
            found_raw, raw_rows = backend._td_try_load_raw_curves(
                raw_db,
                "CondA",
                "Pressure",
                "time",
                serials=["SN-001"],
                program_title="Program Alpha",
                source_run_name="Seq-1",
            )
            self.assertTrue(found_raw)
            self.assertEqual(len(raw_rows), 1)
            self.assertEqual(raw_rows[0]["valve_voltage"], 28.0)

            legacy_db = _create_legacy_curve_db(tmpdir)
            legacy_rows = backend._td_load_legacy_curves(
                legacy_db,
                "CondA",
                "Pressure",
                "time",
                serials=["SN-001"],
                program_title="Program Alpha",
                source_run_name="Seq-1",
            )
            self.assertEqual(len(legacy_rows), 1)
            self.assertEqual(legacy_rows[0]["valve_voltage"], 28.0)

    def test_life_parameter_options_include_synthetic_cumulative_impulse(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_life_metric_db(tmpdir)
            options = backend.td_list_life_parameter_options(db_path, ["CondLife"])
            self.assertEqual(
                [item["name"] for item in options],
                ["Cumulative Impulse", "Pulse Count", "Thrust"],
            )

    def test_load_life_metric_xy_uses_cumulative_impulse_rollup_for_synthetic_parameter(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_life_metric_db(tmpdir)

            x_rollup_rows = backend.td_load_life_metric_xy(
                db_path,
                ["CondLife"],
                "Cumulative Impulse",
                "Thrust",
            )
            self.assertEqual(
                [(row["sequence_index"], row["x_value"], row["y_value"]) for row in x_rollup_rows],
                [(1, 1.5, 10.0), (2, 4.0, 12.0)],
            )

            y_rollup_rows = backend.td_load_life_metric_xy(
                db_path,
                ["CondLife"],
                "Thrust",
                "Total Impulse",
            )
            self.assertEqual(
                [(row["sequence_index"], row["x_value"], row["y_value"]) for row in y_rollup_rows],
                [(1, 10.0, 1.5), (2, 12.0, 4.0)],
            )

    def test_load_life_metric_series_uses_cumulative_impulse_rollup_for_synthetic_parameter(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_life_metric_db(tmpdir)
            rows = backend.td_load_life_metric_series(
                db_path,
                ["CondLife"],
                "Cumulative Impulse",
                "sequence_index",
            )
            self.assertEqual(
                [(row["sequence_index"], row["x_value"], row["y_value"]) for row in rows],
                [(1, 1.0, 1.5), (2, 2.0, 4.0)],
            )

    def test_cumulative_impulse_alias_match_accepts_embedded_cum_imp_headers(self) -> None:
        aliases = backend._td_life_cumulative_impulse_aliases()
        self.assertTrue(backend._td_life_matches_cumulative_impulse_alias("Main Cum Imp", aliases))
        self.assertTrue(backend._td_life_matches_cumulative_impulse_alias("seq_cum_imp_total", aliases))
        self.assertTrue(backend._td_life_matches_cumulative_impulse_alias("thruster cumulative impulse value", aliases))


if __name__ == "__main__":
    unittest.main()
