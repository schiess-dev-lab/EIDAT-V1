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
        conn.execute(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            ("CondA", "time", "Condition A", "PM", 10.0, 0.5),
        )
        conn.execute(
            "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
            ("CondA", "Pressure", "psi", "y"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_condition_observations(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("agg-1", "SN-001", "CondA", "Program Alpha", "Aggregate", "PM", 0.5, 10.0, 5.0, 1, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_metrics_calc(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("agg-1", "SN-001", "CondA", "Pressure", "mean", 1.5, 1, 1, "Program Alpha", "Aggregate"),
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
            self.assertEqual([item["source_run_name"] for item in views["sequence"]], ["Seq-1", "Seq-2"])
            self.assertEqual(views["condition"][0]["run_name"], "CondA")
            self.assertEqual(views["condition"][0]["display_text"], "Condition A")
            self.assertEqual(views["condition"][0]["member_sequences"], ["Seq-1", "Seq-2"])

            filter_rows = backend.td_read_observation_filter_rows_from_cache(db_path)
            self.assertEqual([row["source_run_name"] for row in filter_rows], ["Seq-1", "Seq-2"])


if __name__ == "__main__":
    unittest.main()
