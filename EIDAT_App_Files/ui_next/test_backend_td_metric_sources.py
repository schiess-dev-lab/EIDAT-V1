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
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("agg-1", "SN-001", "CondA", "Program Alpha", "Aggregate", "PM", 0.5, 10.0, 5.0, 1, 1),
                ("agg-ss-1", "SN-002", "CondSS", "Program Beta", "Aggregate", "SS", None, None, None, 1, 1),
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
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("seq-1", "SN-001", "CondA", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 1, 1),
                ("seq-2", "SN-001", "CondA", "Program Alpha", "Seq-2", "PM", 0.5, 10.0, 5.0, 1, 1),
                ("seq-ss-1", "SN-002", "CondSS", "Program Beta", "Seq-SS-1", "SS", None, None, None, 1, 1),
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

    def test_run_selection_views_and_filter_rows_prefer_sequence_observations(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            views = backend.td_list_run_selection_views(db_path, Path("project.xlsx"))
            self.assertEqual([item["source_run_name"] for item in views["sequence"]], ["Seq-1", "Seq-2", "Seq-SS-1"])
            self.assertEqual([item["run_name"] for item in views["condition"]], ["CondA", "CondSS"])
            self.assertEqual(views["condition"][0]["display_text"], "Condition A")
            self.assertEqual(views["condition"][0]["member_sequences"], ["Seq-1", "Seq-2"])
            self.assertEqual(views["condition"][1]["display_text"], "Condition SS")
            self.assertEqual(views["condition"][1]["member_sequences"], ["Seq-SS-1"])

            filter_rows = backend.td_read_observation_filter_rows_from_cache(db_path)
            self.assertEqual([row["source_run_name"] for row in filter_rows], ["Seq-1", "Seq-2", "Seq-SS-1"])

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

    def test_performance_run_type_modes_still_report_steady_state_and_pulsed_mode(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_metric_source_db(tmpdir)
            self.assertEqual(
                backend.td_list_performance_run_type_modes(db_path),
                ["steady_state", "pulsed_mode"],
            )


if __name__ == "__main__":
    unittest.main()
