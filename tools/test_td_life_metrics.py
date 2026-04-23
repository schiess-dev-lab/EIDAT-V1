import json
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from EIDAT_App_Files.ui_next import backend as be  # type: ignore


class TestTDLifeMetrics(unittest.TestCase):
    def _make_paths(self, td: str) -> tuple[Path, Path, Path]:
        root = Path(td)
        return root / "implementation_trending.sqlite3", root / "test_data_raw_cache.sqlite3", root / "missing.xlsx"

    def _init_dbs(self, db_path: Path, raw_db_path: Path) -> None:
        with sqlite3.connect(str(db_path)) as conn:
            be._ensure_test_data_impl_tables(conn)  # type: ignore[attr-defined]
            conn.commit()
        with sqlite3.connect(str(raw_db_path)) as conn:
            be._ensure_test_data_raw_cache_tables(conn)  # type: ignore[attr-defined]
            conn.commit()

    def _insert_sequence(
        self,
        conn: sqlite3.Connection,
        *,
        obs_id: str,
        serial: str = "SN1",
        run_name: str,
        display_name: str,
        source_run_name: str,
        run_type: str,
        pulse_width: float | None,
        control_period: float | None,
        metrics: dict[str, float],
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            (run_name, "Time", display_name, run_type, control_period, pulse_width),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_condition_observations_sequences(
                observation_id, serial, run_name, program_title, source_run_name, run_type,
                pulse_width, control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (obs_id, serial, run_name, "Program A", source_run_name, run_type, pulse_width, control_period, None, None, 1, 1),
        )
        for name, value in metrics.items():
            conn.execute(
                "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, 'y')",
                (run_name, name, "u"),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO td_metrics_calc_sequences(
                    observation_id, serial, run_name, column_name, stat, value_num,
                    computed_epoch_ns, source_mtime_ns, program_title, source_run_name
                ) VALUES (?, ?, ?, ?, 'mean', ?, ?, ?, ?, ?)
                """,
                (obs_id, serial, run_name, name, value, 1, 1, "Program A", source_run_name),
            )

    def _insert_raw_pulse_axis(self, raw_conn: sqlite3.Connection, *, obs_id: str, run_name: str, serial: str, pulses: list[int]) -> None:
        self._insert_raw_curve(
            raw_conn,
            obs_id=obs_id,
            run_name=run_name,
            serial=serial,
            x_name="Pulse Number",
            y_name="pc",
            xs=pulses,
            ys=[0 for _ in pulses],
        )

    def _insert_raw_curve(
        self,
        raw_conn: sqlite3.Connection,
        *,
        obs_id: str,
        run_name: str,
        serial: str,
        x_name: str,
        y_name: str,
        xs: list[float | int],
        ys: list[float | int],
    ) -> None:
        raw_conn.execute(
            """
            INSERT OR REPLACE INTO td_curves_raw(
                run_name, y_name, x_name, observation_id, serial, x_json, y_json,
                n_points, source_mtime_ns, computed_epoch_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                y_name,
                x_name,
                obs_id,
                serial,
                json.dumps(xs),
                json.dumps(ys),
                max(len(xs), len(ys)),
                1,
                1,
                "Program A",
                run_name,
            ),
        )

    def test_pm_life_accumulates_by_serial_and_sequence(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path, raw_db_path, workbook_path = self._make_paths(td)
            self._init_dbs(db_path, raw_db_path)
            with sqlite3.connect(str(db_path)) as conn:
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__Seq1",
                    run_name="cond_a",
                    display_name="Test Block A",
                    source_run_name="Seq1",
                    run_type="PM",
                    pulse_width=0.1,
                    control_period=1.0,
                    metrics={"pc": 35.0, "prop_per_pulse": 2.0, "impulse bit": 3.0, "thrust": 9.0},
                )
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__Seq2",
                    run_name="cond_b",
                    display_name="Test Block B",
                    source_run_name="Seq2",
                    run_type="PM",
                    pulse_width=0.1,
                    control_period=1.0,
                    metrics={"pc": 40.0, "prop_per_pulse": 1.0, "impulse bit": 4.0, "thrust": 10.0},
                )
                conn.commit()
            with sqlite3.connect(str(raw_db_path)) as raw_conn:
                self._insert_raw_pulse_axis(raw_conn, obs_id="SN1__Program_A__Seq1", run_name="cond_a", serial="SN1", pulses=[1, 2, 3, 4, 5])
                self._insert_raw_pulse_axis(raw_conn, obs_id="SN1__Program_A__Seq2", run_name="cond_b", serial="SN1", pulses=[1, 2])
                self._insert_raw_curve(
                    raw_conn,
                    obs_id="SN1__Program_A__Seq1",
                    run_name="cond_a",
                    serial="SN1",
                    x_name="Pulse Number",
                    y_name="I bit",
                    xs=[1, 2, 3, 4, 5],
                    ys=[1.0, 2.0, 3.0, 4.0, 5.0],
                )
                self._insert_raw_curve(
                    raw_conn,
                    obs_id="SN1__Program_A__Seq2",
                    run_name="cond_b",
                    serial="SN1",
                    x_name="Pulse Number",
                    y_name="i-bit",
                    xs=[1, 2],
                    ys=[10.0, 20.0],
                )
                raw_conn.commit()

            with sqlite3.connect(str(db_path)) as conn:
                count = be._td_rebuild_life_metrics(  # type: ignore[attr-defined]
                    conn,
                    workbook_path=workbook_path,
                    project_cfg={},
                    raw_db_path=raw_db_path,
                    computed_epoch_ns=99,
                )
                self.assertEqual(count, 8)
                rows = conn.execute(
                    """
                    SELECT sequence_index, sequence_label, sequence_pulses, cumulative_pulses,
                           sequence_on_time, cumulative_on_time, sequence_throughput,
                           cumulative_throughput, sequence_impulse, cumulative_impulse
                    FROM td_life_metrics
                    WHERE parameter_name='pc'
                    ORDER BY sequence_index
                    """
                ).fetchall()

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], 1)
            self.assertEqual(rows[0][1], "Test Block A")
            self.assertAlmostEqual(rows[0][2], 5.0)
            self.assertAlmostEqual(rows[0][3], 5.0)
            self.assertAlmostEqual(rows[0][4], 0.5)
            self.assertAlmostEqual(rows[0][5], 0.5)
            self.assertAlmostEqual(rows[0][6], 10.0)
            self.assertAlmostEqual(rows[0][7], 10.0)
            self.assertAlmostEqual(rows[0][8], 15.0)
            self.assertAlmostEqual(rows[0][9], 15.0)
            self.assertEqual(rows[1][0], 2)
            self.assertAlmostEqual(rows[1][3], 7.0)
            self.assertAlmostEqual(rows[1][5], 0.7)
            self.assertAlmostEqual(rows[1][7], 12.0)
            self.assertAlmostEqual(rows[1][8], 30.0)
            self.assertAlmostEqual(rows[1][9], 45.0)

            series = be.td_load_life_metric_series(db_path, ["cond_a", "cond_b"], "pc", "cumulative_pulses", serials=["SN1"])
            self.assertEqual([row["x_value"] for row in series], [5.0, 7.0])
            self.assertEqual([row["y_value"] for row in series], [35.0, 40.0])
            xy = be.td_load_life_metric_xy(db_path, ["cond_a", "cond_b"], "pc", "thrust", serials=["SN1"])
            self.assertEqual([(row["x_value"], row["y_value"]) for row in xy], [(35.0, 9.0), (40.0, 10.0)])

    def test_missing_pm_pulse_count_does_not_use_row_count_fallback(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path, raw_db_path, workbook_path = self._make_paths(td)
            self._init_dbs(db_path, raw_db_path)
            with sqlite3.connect(str(db_path)) as conn:
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__Seq1",
                    run_name="cond_a",
                    display_name="Test Block A",
                    source_run_name="Seq1",
                    run_type="PM",
                    pulse_width=0.2,
                    control_period=1.0,
                    metrics={"pc": 35.0, "prop_per_pulse": 2.0, "thrust": 9.0},
                )
                be._td_rebuild_life_metrics(  # type: ignore[attr-defined]
                    conn,
                    workbook_path=workbook_path,
                    project_cfg={},
                    raw_db_path=raw_db_path,
                    computed_epoch_ns=99,
                )
                row = conn.execute(
                    """
                    SELECT sequence_pulses, sequence_on_time, sequence_throughput,
                           sequence_impulse, diagnostics
                    FROM td_life_metrics
                    WHERE parameter_name='pc'
                    """
                ).fetchone()

            self.assertIsNone(row[0])
            self.assertIsNone(row[1])
            self.assertIsNone(row[2])
            self.assertIsNone(row[3])
            self.assertIn("missing pulse count", row[4])

    def test_pm_impulse_falls_back_to_mean_impulse_bit_when_raw_curve_missing(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path, raw_db_path, workbook_path = self._make_paths(td)
            self._init_dbs(db_path, raw_db_path)
            with sqlite3.connect(str(db_path)) as conn:
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__Seq1",
                    run_name="cond_a",
                    display_name="Test Block A",
                    source_run_name="Seq1",
                    run_type="PM",
                    pulse_width=0.2,
                    control_period=1.0,
                    metrics={"pc": 35.0, "impulse bit": 3.0, "thrust": 9.0},
                )
                conn.commit()
            with sqlite3.connect(str(raw_db_path)) as raw_conn:
                self._insert_raw_pulse_axis(raw_conn, obs_id="SN1__Program_A__Seq1", run_name="cond_a", serial="SN1", pulses=[1, 2, 3, 4])
                raw_conn.commit()

            with sqlite3.connect(str(db_path)) as conn:
                be._td_rebuild_life_metrics(  # type: ignore[attr-defined]
                    conn,
                    workbook_path=workbook_path,
                    project_cfg={},
                    raw_db_path=raw_db_path,
                    computed_epoch_ns=99,
                )
                row = conn.execute(
                    """
                    SELECT sequence_pulses, sequence_impulse, cumulative_impulse
                    FROM td_life_metrics
                    WHERE parameter_name='pc'
                    """
                ).fetchone()

            self.assertEqual(tuple(float(value) for value in row), (4.0, 12.0, 12.0))

    def test_steady_state_uses_raw_thrust_time_integration(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path, raw_db_path, workbook_path = self._make_paths(td)
            self._init_dbs(db_path, raw_db_path)
            with sqlite3.connect(str(db_path)) as conn:
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__SS1",
                    run_name="ss_a",
                    display_name="Steady State A",
                    source_run_name="SS1",
                    run_type="SS",
                    pulse_width=None,
                    control_period=None,
                    metrics={"pc": 100.0, "elapsed time": 3.0, "flow rate": 2.0, "thrust": 99.0},
                )
                conn.commit()
            with sqlite3.connect(str(raw_db_path)) as raw_conn:
                self._insert_raw_curve(
                    raw_conn,
                    obs_id="SN1__Program_A__SS1",
                    run_name="ss_a",
                    serial="SN1",
                    x_name="Time",
                    y_name="thrust",
                    xs=[0.0, 1.0, 2.0, 3.0],
                    ys=[10.0, 10.0, 20.0, 20.0],
                )
                raw_conn.commit()

            with sqlite3.connect(str(db_path)) as conn:
                be._td_rebuild_life_metrics(  # type: ignore[attr-defined]
                    conn,
                    workbook_path=workbook_path,
                    project_cfg={},
                    raw_db_path=raw_db_path,
                    computed_epoch_ns=99,
                )
                row = conn.execute(
                    """
                    SELECT sequence_on_time, sequence_elapsed_time, sequence_throughput,
                           sequence_impulse, cumulative_impulse
                    FROM td_life_metrics
                    WHERE parameter_name='pc'
                    """
                ).fetchone()

            self.assertEqual(tuple(float(value) for value in row), (3.0, 3.0, 6.0, 45.0, 45.0))

    def test_steady_state_uses_elapsed_time_flowrate_and_thrust(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path, raw_db_path, workbook_path = self._make_paths(td)
            self._init_dbs(db_path, raw_db_path)
            with sqlite3.connect(str(db_path)) as conn:
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__SS1",
                    run_name="ss_a",
                    display_name="Steady State A",
                    source_run_name="SS1",
                    run_type="SS",
                    pulse_width=None,
                    control_period=None,
                    metrics={"pc": 100.0, "elapsed time": 10.0, "flow rate": 2.0, "thrust": 5.0},
                )
                be._td_rebuild_life_metrics(  # type: ignore[attr-defined]
                    conn,
                    workbook_path=workbook_path,
                    project_cfg={},
                    raw_db_path=raw_db_path,
                    computed_epoch_ns=99,
                )
                row = conn.execute(
                    """
                    SELECT sequence_on_time, sequence_elapsed_time, sequence_throughput,
                           sequence_impulse, cumulative_on_time, cumulative_throughput,
                           cumulative_impulse
                    FROM td_life_metrics
                    WHERE parameter_name='pc'
                    """
                ).fetchone()

            self.assertEqual(tuple(float(value) for value in row), (10.0, 10.0, 20.0, 50.0, 10.0, 20.0, 50.0))

    def test_steady_state_uses_on_time_alias_for_duration(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path, raw_db_path, workbook_path = self._make_paths(td)
            self._init_dbs(db_path, raw_db_path)
            with sqlite3.connect(str(db_path)) as conn:
                self._insert_sequence(
                    conn,
                    obs_id="SN1__Program_A__SS1",
                    run_name="ss_a",
                    display_name="Steady State A",
                    source_run_name="SS1",
                    run_type="SS",
                    pulse_width=None,
                    control_period=None,
                    metrics={"pc": 100.0, "on time": 4.0, "flow rate": 2.0, "thrust": 5.0},
                )
                be._td_rebuild_life_metrics(  # type: ignore[attr-defined]
                    conn,
                    workbook_path=workbook_path,
                    project_cfg={},
                    raw_db_path=raw_db_path,
                    computed_epoch_ns=99,
                )
                row = conn.execute(
                    """
                    SELECT sequence_on_time, sequence_elapsed_time, sequence_throughput,
                           sequence_impulse, cumulative_impulse
                    FROM td_life_metrics
                    WHERE parameter_name='pc'
                    """
                ).fetchone()

            self.assertEqual(tuple(float(value) for value in row), (4.0, 4.0, 8.0, 20.0, 20.0))


if __name__ == "__main__":
    unittest.main()
