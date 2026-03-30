import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class _FakeArray(list):
    @property
    def size(self) -> int:
        return len(self)

    def tolist(self) -> list[float]:
        return list(self)


class _FakeNumpy:
    @staticmethod
    def asarray(values, dtype=float):  # noqa: ANN001
        return _FakeArray(dtype(value) for value in values)

    @staticmethod
    def median(values) -> float:  # noqa: ANN001
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _create_smart_solver_db(tmpdir: str, rows: list[dict[str, object]]) -> Path:
    db_path = Path(tmpdir) / "smart_solver.sqlite3"
    runs = sorted({str(row.get("run_name") or "").strip() for row in rows if str(row.get("run_name") or "").strip()})
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            [(run_name, "time", run_name, "PM", None, 0.5) for run_name in runs],
        )
        column_rows: list[tuple[str, str, str, str]] = []
        for run_name in runs:
            column_rows.extend(
                [
                    (run_name, "Input1", "arb", "y"),
                    (run_name, "Output", "arb", "y"),
                ]
            )
        conn.executemany(
            "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
            column_rows,
        )

        observation_rows: list[tuple[object, ...]] = []
        metric_rows: list[tuple[object, ...]] = []
        for idx, row in enumerate(rows, start=1):
            observation_id = str(row.get("observation_id") or "").strip()
            serial = str(row.get("serial") or "").strip()
            run_name = str(row.get("run_name") or "").strip()
            program_title = str(row.get("program_title") or "").strip()
            source_run_name = str(row.get("source_run_name") or "")
            control_period = float(row.get("control_period") or 0.0)
            suppression_voltage = float(row.get("suppression_voltage") or 5.0)
            input_1 = float(row.get("input_1") or 0.0)
            output = float(row.get("output") or 0.0)
            observation_rows.append(
                (
                    observation_id,
                    serial,
                    run_name,
                    program_title,
                    source_run_name,
                    "PM",
                    0.5,
                    control_period,
                    suppression_voltage,
                    idx,
                    idx,
                )
            )
            metric_rows.extend(
                [
                    (
                        observation_id,
                        serial,
                        run_name,
                        "Input1",
                        "mean",
                        input_1,
                        idx,
                        idx,
                        program_title,
                        source_run_name,
                    ),
                    (
                        observation_id,
                        serial,
                        run_name,
                        "Output",
                        "mean",
                        output,
                        idx,
                        idx,
                        program_title,
                        source_run_name,
                    ),
                ]
            )

        conn.executemany(
            """
            INSERT OR REPLACE INTO td_condition_observations_sequences(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            observation_rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_metrics_calc_sequences(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            metric_rows,
        )
        conn.commit()
    return db_path


def _run_solver_with_captures(
    db_path: Path,
    *,
    keep_first_sequences_per_serial: int = 0,
    runs: list[str],
    serials: list[str],
    program_filters: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    captured: dict[str, list[float]] = {}

    def _fake_support(xs, ys, control_periods):
        cp_values = [float(value) for value in control_periods]
        x_values = [float(value) for value in xs]
        captured["support_xs"] = list(x_values)
        captured["support_cps"] = list(cp_values)
        distinct_cps = sorted({float(value) for value in cp_values})
        slice_rows = []
        for cp_value in distinct_cps:
            indices = [idx for idx, current in enumerate(cp_values) if float(current) == cp_value]
            x_slice = [x_values[idx] for idx in indices]
            slice_rows.append(
                {
                    "control_period": cp_value,
                    "point_count": len(indices),
                    "distinct_x1": len({float(value) for value in x_slice}),
                    "eligible": True,
                    "reason": "",
                }
            )
        return {
            "eligible_control_period_values": distinct_cps,
            "slice_rows": slice_rows,
            "ignored_control_periods": [],
        }

    def _fake_fit(xs, ys, control_periods):
        captured["fit_xs"] = [float(value) for value in xs]
        captured["fit_ys"] = [float(value) for value in ys]
        captured["fit_cps"] = [float(value) for value in control_periods]
        return {
            "fit_family": "quadratic_curve_control_period",
            "equation": "y = x",
            "x_norm_equation": "x' = x",
            "rmse": 0.0,
            "fit_domain_control_period": [min(captured["fit_cps"]), max(captured["fit_cps"])],
        }

    def _fake_predict(_model, xs, control_periods):
        captured["predict_xs"] = [float(value) for value in xs]
        captured["predict_cps"] = [float(value) for value in control_periods]
        return list(captured.get("fit_ys") or [])

    with patch.object(
        backend,
        "_td_perf_analyze_quadratic_curve_control_period_fit_support",
        side_effect=_fake_support,
    ), patch.object(
        backend,
        "_td_perf_fit_quadratic_curve_control_period_model",
        side_effect=_fake_fit,
    ), patch.object(
        backend,
        "_td_perf_predict_quadratic_curve_control_period",
        side_effect=_fake_predict,
    ), patch.object(
        backend,
        "_td_perf_import_numpy",
        return_value=_FakeNumpy,
    ):
        result = backend.td_smart_solver_run(
            db_path,
            output_target="Output",
            input1_target="Input1",
            keep_first_sequences_per_serial=keep_first_sequences_per_serial,
            runs=runs,
            serials=serials,
            program_filters=program_filters,
        )
    return result, captured


class TestBackendTdSmartSolver(unittest.TestCase):
    def test_sequence_cap_keeps_first_n_sequences_per_serial_and_drops_later_sequences(self) -> None:
        rows: list[dict[str, object]] = []
        for serial in ("SN-001", "SN-002"):
            for run_name, control_period in (("CondA", 10.0), ("CondB", 20.0)):
                for sequence_name, x_value in (("Seq-1", 1.0), ("Seq-2", 2.0), ("Seq-10", 10.0)):
                    rows.append(
                        {
                            "observation_id": f"{serial}-{sequence_name}-{run_name}",
                            "serial": serial,
                            "run_name": run_name,
                            "program_title": "Program Alpha",
                            "source_run_name": sequence_name,
                            "control_period": control_period,
                            "input_1": x_value,
                            "output": x_value * 10.0 + control_period,
                        }
                    )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            result, captured = _run_solver_with_captures(
                db_path,
                keep_first_sequences_per_serial=2,
                runs=["CondA", "CondB"],
                serials=["SN-001", "SN-002"],
            )

        self.assertEqual(result["keep_first_sequences_per_serial"], 2)
        self.assertEqual(result["dropped_sequence_count"], 2)
        self.assertEqual(result["dropped_point_count"], 4)
        self.assertEqual(result["sample_count"], 8)
        self.assertEqual(sorted(set(captured["fit_xs"])), [1.0, 2.0])
        self.assertNotIn(10.0, captured["fit_xs"])

    def test_sequence_cap_keeps_all_run_bucket_points_for_a_retained_sequence(self) -> None:
        rows = [
            {
                "observation_id": "SN-001-Seq-1-CondA",
                "serial": "SN-001",
                "run_name": "CondA",
                "program_title": "Program Alpha",
                "source_run_name": "Seq-1",
                "control_period": 10.0,
                "input_1": 1.0,
                "output": 11.0,
            },
            {
                "observation_id": "SN-001-Seq-1-CondB",
                "serial": "SN-001",
                "run_name": "CondB",
                "program_title": "Program Alpha",
                "source_run_name": "Seq-1",
                "control_period": 20.0,
                "input_1": 1.0,
                "output": 21.0,
            },
            {
                "observation_id": "SN-001-Seq-2-CondA",
                "serial": "SN-001",
                "run_name": "CondA",
                "program_title": "Program Alpha",
                "source_run_name": "Seq-2",
                "control_period": 10.0,
                "input_1": 2.0,
                "output": 12.0,
            },
            {
                "observation_id": "SN-001-Seq-2-CondB",
                "serial": "SN-001",
                "run_name": "CondB",
                "program_title": "Program Alpha",
                "source_run_name": "Seq-2",
                "control_period": 20.0,
                "input_1": 2.0,
                "output": 22.0,
            },
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            result, captured = _run_solver_with_captures(
                db_path,
                keep_first_sequences_per_serial=1,
                runs=["CondA", "CondB"],
                serials=["SN-001"],
            )

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["run_count"], 2)
        self.assertEqual(sorted(captured["fit_cps"]), [10.0, 20.0])
        self.assertEqual(captured["fit_xs"], [1.0, 1.0])

    def test_sequence_cap_falls_back_to_observation_id_when_source_run_name_is_blank(self) -> None:
        rows = [
            {
                "observation_id": "Obs-1",
                "serial": "SN-001",
                "run_name": "CondA",
                "program_title": "Program Alpha",
                "source_run_name": "",
                "control_period": 10.0,
                "input_1": 1.0,
                "output": 11.0,
            },
            {
                "observation_id": "Obs-2",
                "serial": "SN-001",
                "run_name": "CondA",
                "program_title": "Program Alpha",
                "source_run_name": "",
                "control_period": 20.0,
                "input_1": 2.0,
                "output": 22.0,
            },
            {
                "observation_id": "Obs-10",
                "serial": "SN-001",
                "run_name": "CondA",
                "program_title": "Program Alpha",
                "source_run_name": "",
                "control_period": 20.0,
                "input_1": 10.0,
                "output": 30.0,
            },
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            result, captured = _run_solver_with_captures(
                db_path,
                keep_first_sequences_per_serial=2,
                runs=["CondA"],
                serials=["SN-001"],
            )

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["dropped_sequence_count"], 1)
        self.assertEqual(sorted(captured["fit_xs"]), [1.0, 2.0])
        self.assertNotIn(10.0, captured["fit_xs"])

    def test_sequence_cap_is_applied_after_program_filters(self) -> None:
        rows = []
        for run_name, control_period in (("CondA", 10.0), ("CondB", 20.0)):
            rows.extend(
                [
                    {
                        "observation_id": f"SN-001-Seq-1-{run_name}",
                        "serial": "SN-001",
                        "run_name": run_name,
                        "program_title": "Program Skip",
                        "source_run_name": "Seq-1",
                        "control_period": control_period,
                        "input_1": 1.0,
                        "output": 10.0 + control_period,
                    },
                    {
                        "observation_id": f"SN-001-Seq-2-{run_name}",
                        "serial": "SN-001",
                        "run_name": run_name,
                        "program_title": "Program Keep",
                        "source_run_name": "Seq-2",
                        "control_period": control_period,
                        "input_1": 2.0,
                        "output": 20.0 + control_period,
                    },
                    {
                        "observation_id": f"SN-001-Seq-10-{run_name}",
                        "serial": "SN-001",
                        "run_name": run_name,
                        "program_title": "Program Keep",
                        "source_run_name": "Seq-10",
                        "control_period": control_period,
                        "input_1": 10.0,
                        "output": 30.0 + control_period,
                    },
                ]
            )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            result, captured = _run_solver_with_captures(
                db_path,
                keep_first_sequences_per_serial=1,
                runs=["CondA", "CondB"],
                serials=["SN-001"],
                program_filters=["Program Keep"],
            )

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["dropped_sequence_count"], 1)
        self.assertEqual(captured["fit_xs"], [2.0, 2.0])
        self.assertNotIn(1.0, captured["fit_xs"])
        self.assertNotIn(10.0, captured["fit_xs"])


if __name__ == "__main__":
    unittest.main()
