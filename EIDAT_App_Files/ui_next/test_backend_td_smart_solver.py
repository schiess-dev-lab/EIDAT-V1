import sqlite3
import sys
import tempfile
import unittest
import math
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


def _have_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


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
    has_input2 = any("input_2" in row for row in rows)
    has_input3 = any("input_3" in row for row in rows)
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            [(run_name, "time", run_name, "PM", None, 0.5) for run_name in runs],
        )
        column_rows: list[tuple[str, str, str, str]] = []
        for run_name in runs:
            run_columns = [
                (run_name, "Input1", "arb", "y"),
                (run_name, "Output", "arb", "y"),
            ]
            if has_input2:
                run_columns.insert(1, (run_name, "Input2", "arb", "y"))
            if has_input3:
                run_columns.insert(2 if has_input2 else 1, (run_name, "Input3", "arb", "y"))
            column_rows.extend(run_columns)
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
            input_2 = float(row.get("input_2") or 0.0) if "input_2" in row else None
            input_3 = float(row.get("input_3") or 0.0) if "input_3" in row else None
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
            if has_input2 and input_2 is not None:
                metric_rows.append(
                    (
                        observation_id,
                        serial,
                        run_name,
                        "Input2",
                        "mean",
                        input_2,
                        idx,
                        idx,
                        program_title,
                        source_run_name,
                    )
                )
            if has_input3 and input_3 is not None:
                metric_rows.append(
                    (
                        observation_id,
                        serial,
                        run_name,
                        "Input3",
                        "mean",
                        input_3,
                        idx,
                        idx,
                        program_title,
                        source_run_name,
                    )
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

    def _fake_hybrid_fit(xs, ys, control_periods, **_kwargs):
        return _fake_fit(xs, ys, control_periods)

    def _fake_predict(_model, xs, *, control_period=None):
        captured["predict_xs"] = [float(value) for value in xs]
        captured["predict_cps"] = [float(value) for value in (control_period or [])]
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
        "_td_perf_fit_hybrid_quadratic_residual_control_period_model",
        side_effect=_fake_hybrid_fit,
    ), patch.object(
        backend,
        "td_perf_predict_model",
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


def _build_curve_cp_rows(
    *,
    control_periods: list[float],
    xs: list[float],
    low_x_bend: bool,
    serial: str = "SN-001",
    run_name: str = "CondA",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cp_index, control_period in enumerate(control_periods):
        for point_index, x_value in enumerate(xs):
            base = 160.0 + (42.0 * (1.0 - math.exp(-1700.0 * x_value))) + (220.0 * x_value) + (0.05 * control_period)
            if low_x_bend:
                bend = (18.0 + (0.04 * control_period)) * math.exp(-((x_value - 0.0009) / 0.00055) ** 2)
                tail_noise = (0.35 if point_index % 2 == 0 else -0.35) if x_value >= 0.0035 else 0.0
            else:
                bend = 0.0
                tail_noise = (0.08 if point_index % 2 == 0 else -0.08)
            rows.append(
                {
                    "observation_id": f"{serial}-{run_name}-{int(control_period)}-{point_index}",
                    "serial": serial,
                    "run_name": run_name,
                    "program_title": "Program Alpha",
                    "source_run_name": f"Seq-{cp_index + 1}-{point_index + 1}",
                    "control_period": float(control_period),
                    "input_1": float(x_value),
                    "output": float(base + bend + tail_noise),
                }
            )
    return rows


def _negative_interval_count(xs: list[float], ys: list[float], *, breakpoint: float | None) -> int:
    intervals = 0
    in_interval = False
    positive_slopes = [
        (ys[idx + 1] - ys[idx]) / (xs[idx + 1] - xs[idx])
        for idx in range(len(xs) - 1)
        if xs[idx + 1] > xs[idx] and (ys[idx + 1] - ys[idx]) > 0.0
    ]
    positive_slopes = [float(value) for value in positive_slopes if math.isfinite(float(value))]
    median_positive = sorted(positive_slopes)[len(positive_slopes) // 2] if positive_slopes else 0.0
    threshold = abs(float(median_positive)) * 0.02
    for idx in range(len(xs) - 1):
        x_mid = (xs[idx] + xs[idx + 1]) / 2.0
        if breakpoint is not None and x_mid <= breakpoint:
            continue
        dx = xs[idx + 1] - xs[idx]
        if dx <= 0.0:
            continue
        slope = (ys[idx + 1] - ys[idx]) / dx
        is_negative = float(slope) < -threshold
        if is_negative and not in_interval:
            intervals += 1
            in_interval = True
        elif not is_negative:
            in_interval = False
    return intervals


def _build_three_input_rows(
    *,
    control_periods: list[float],
    serial: str = "SN-001",
    run_name: str = "CondA",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lattice = [
        (0.8, 1.0),
        (1.2, 1.3),
        (1.6, 1.6),
        (2.0, 1.9),
        (2.4, 2.2),
        (2.8, 2.5),
        (3.2, 2.8),
        (3.6, 3.1),
        (4.0, 3.4),
        (4.4, 3.7),
    ]
    for cp_index, control_period in enumerate(control_periods):
        for point_index, (x1_value, x2_value) in enumerate(lattice):
            x3_value = (0.6 * x1_value) + (1.1 * x2_value) + (0.015 * control_period)
            output = (2.8 * x1_value) - (1.4 * x2_value) + (0.9 * x3_value) + (0.08 * control_period)
            rows.append(
                {
                    "observation_id": f"{serial}-{run_name}-{int(control_period)}-{point_index:02d}",
                    "serial": serial,
                    "run_name": run_name,
                    "program_title": "Program Alpha",
                    "source_run_name": f"Seq-{cp_index + 1}-{point_index + 1}",
                    "control_period": float(control_period),
                    "input_1": float(x1_value),
                    "input_2": float(x2_value),
                    "input_3": float(x3_value),
                    "output": float(output),
                }
            )
    return rows


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

    def test_three_input_solver_prefers_direct_branch_when_staged_is_not_materially_better(self) -> None:
        rows = _build_three_input_rows(control_periods=[40.0, 80.0])
        ordered_rows = sorted(rows, key=lambda row: str(row["observation_id"]))
        direct_support = {
            "eligible_control_period_values": [40.0, 80.0],
            "ignored_control_periods": [],
            "slice_rows": [
                {
                    "control_period": 40.0,
                    "point_count": 10,
                    "distinct_x1": 10,
                    "distinct_x2": 10,
                    "distinct_x3": 10,
                    "eligible": True,
                    "reason": "",
                },
                {
                    "control_period": 80.0,
                    "point_count": 10,
                    "distinct_x1": 10,
                    "distinct_x2": 10,
                    "distinct_x3": 10,
                    "eligible": True,
                    "reason": "",
                },
            ],
        }
        direct_model = {
            "fit_family": backend.TD_PERF_FIT_FAMILY_QUADRATIC_3INPUT_CONTROL_PERIOD,
            "equation": "y = direct",
            "x_norm_equation": "x' = direct",
            "rmse": 1.0,
        }
        staged_model = {
            "fit_family": backend.TD_PERF_FIT_FAMILY_STAGED_MEDIATOR_CONTROL_PERIOD,
            "equation": "y = staged",
            "x_norm_equation": "x' = staged",
            "rmse": 0.99,
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            with patch.object(
                backend,
                "_td_perf_analyze_quadratic_3input_control_period_fit_support",
                return_value=direct_support,
            ), patch.object(
                backend,
                "_td_perf_fit_quadratic_3input_control_period_model",
                return_value=direct_model,
            ), patch.object(
                backend,
                "_td_perf_fit_staged_mediator_control_period_model",
                return_value=staged_model,
            ), patch.object(
                backend,
                "_td_perf_predict_quadratic_3input_control_period",
                return_value=[float(row["output"]) for row in ordered_rows],
            ), patch.object(
                backend,
                "_td_perf_import_numpy",
                return_value=_FakeNumpy,
            ):
                result = backend.td_smart_solver_run(
                    db_path,
                    output_target="Output",
                    input1_target="Input1",
                    input2_target="Input2",
                    input3_target="Input3",
                    runs=["CondA"],
                    serials=["SN-001"],
                )

        self.assertEqual(result["fit_family"], backend.TD_PERF_FIT_FAMILY_QUADRATIC_3INPUT_CONTROL_PERIOD)
        self.assertEqual(result["solver_branch"], backend.TD_PERF_FIT_FAMILY_QUADRATIC_3INPUT_CONTROL_PERIOD)
        self.assertEqual(result["input3_target"], "Input3")
        self.assertEqual(result["candidate_scores"]["selected"], backend.TD_PERF_FIT_FAMILY_QUADRATIC_3INPUT_CONTROL_PERIOD)
        self.assertIn("did not improve RMSE by at least 5%", result["selection_reason"])
        self.assertEqual(result["slice_rows"][0]["distinct_input_3"], 10)
        self.assertIn("Correlation helper", result["variable_descriptors"]["input3"])
        self.assertTrue(all("input_3" in point for point in result["fit_points"]))

    def test_three_input_solver_uses_staged_branch_when_it_beats_direct_by_five_percent(self) -> None:
        rows = _build_three_input_rows(control_periods=[40.0, 80.0])
        ordered_rows = sorted(rows, key=lambda row: str(row["observation_id"]))
        direct_support = {
            "eligible_control_period_values": [40.0, 80.0],
            "ignored_control_periods": [],
            "slice_rows": [
                {
                    "control_period": 40.0,
                    "point_count": 10,
                    "distinct_x1": 10,
                    "distinct_x2": 10,
                    "distinct_x3": 10,
                    "eligible": True,
                    "reason": "",
                },
                {
                    "control_period": 80.0,
                    "point_count": 10,
                    "distinct_x1": 10,
                    "distinct_x2": 10,
                    "distinct_x3": 10,
                    "eligible": True,
                    "reason": "",
                },
            ],
        }
        staged_model = {
            "fit_family": backend.TD_PERF_FIT_FAMILY_STAGED_MEDIATOR_CONTROL_PERIOD,
            "equation": "x3_hat = f(...); y = g(...)",
            "x_norm_equation": "x3' = f(...); y' = g(...)",
            "rmse": 4.0,
            "stage1_model": {
                "fit_family": backend.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD,
            },
            "stage2_model": {
                "fit_family": backend.TD_PERF_FIT_FAMILY_QUADRATIC_CURVE_CONTROL_PERIOD,
            },
            "stage1_slice_rows": [
                {"control_period": 40.0, "eligible": True, "reason": ""},
                {"control_period": 80.0, "eligible": True, "reason": ""},
            ],
            "stage2_slice_rows": [
                {"control_period": 40.0, "eligible": True, "reason": ""},
                {"control_period": 80.0, "eligible": True, "reason": ""},
            ],
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            with patch.object(
                backend,
                "_td_perf_analyze_quadratic_3input_control_period_fit_support",
                return_value=direct_support,
            ), patch.object(
                backend,
                "_td_perf_fit_quadratic_3input_control_period_model",
                return_value={
                    "fit_family": backend.TD_PERF_FIT_FAMILY_QUADRATIC_3INPUT_CONTROL_PERIOD,
                    "equation": "y = direct",
                    "x_norm_equation": "x' = direct",
                    "rmse": 10.0,
                },
            ), patch.object(
                backend,
                "_td_perf_fit_staged_mediator_control_period_model",
                return_value=staged_model,
            ), patch.object(
                backend,
                "td_perf_predict_surface",
                return_value=[float(row["input_3"]) for row in ordered_rows],
            ), patch.object(
                backend,
                "td_perf_predict_model",
                return_value=[float(row["output"]) for row in ordered_rows],
            ), patch.object(
                backend,
                "_td_perf_import_numpy",
                return_value=_FakeNumpy,
            ):
                result = backend.td_smart_solver_run(
                    db_path,
                    output_target="Output",
                    input1_target="Input1",
                    input2_target="Input2",
                    input3_target="Input3",
                    runs=["CondA"],
                    serials=["SN-001"],
                )

        self.assertEqual(result["fit_family"], backend.TD_PERF_FIT_FAMILY_STAGED_MEDIATOR_CONTROL_PERIOD)
        self.assertEqual(result["candidate_scores"]["selected"], backend.TD_PERF_FIT_FAMILY_STAGED_MEDIATOR_CONTROL_PERIOD)
        self.assertIn("improved RMSE by at least 5%", result["selection_reason"])
        self.assertEqual(result["slice_rows"][0]["distinct_input_3"], 10)
        self.assertEqual(result["master_model"]["stage1_model"]["fit_family"], backend.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD)
        self.assertEqual(result["master_model"]["stage2_model"]["fit_family"], backend.TD_PERF_FIT_FAMILY_QUADRATIC_CURVE_CONTROL_PERIOD)

    def test_three_input_solver_requires_input2_before_input3(self) -> None:
        rows = _build_three_input_rows(control_periods=[40.0, 80.0])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            with self.assertRaisesRegex(RuntimeError, "Select Input 2 before using Input 3"):
                backend.td_smart_solver_run(
                    db_path,
                    output_target="Output",
                    input1_target="Input1",
                    input3_target="Input3",
                    runs=["CondA"],
                    serials=["SN-001"],
                )

    @unittest.skipUnless(_have_scipy(), "scipy is required")
    def test_smart_solver_prefers_stabilized_curve_family_for_low_x_bend(self) -> None:
        xs = [0.0003, 0.0006, 0.0009, 0.0012, 0.0016, 0.0022, 0.0035, 0.0060, 0.0120]
        rows = _build_curve_cp_rows(control_periods=[40.0, 80.0, 120.0], xs=xs, low_x_bend=True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            result = backend.td_smart_solver_run(
                db_path,
                output_target="Output",
                input1_target="Input1",
                runs=["CondA"],
                serials=["SN-001"],
            )

        self.assertEqual(
            result["fit_family"],
            backend.TD_PERF_FIT_FAMILY_HYBRID_QUADRATIC_RESIDUAL_CONTROL_PERIOD,
        )
        self.assertTrue(bool(result.get("low_x_window_enabled")))
        breakpoint = float(result.get("low_x_breakpoint") or 0.0)
        self.assertGreater(breakpoint, 0.0003)
        self.assertLess(breakpoint, 0.0025)
        self.assertFalse(bool(result.get("fallback_used")))

        model = dict(result.get("master_model") or {})
        dense_x = [xs[0] + ((xs[-1] - xs[0]) * idx / 119.0) for idx in range(120)]
        for control_period in (40.0, 60.0, 80.0, 100.0, 120.0):
            predictions = backend.td_perf_predict_model(
                model,
                dense_x,
                control_period=[control_period] * len(dense_x),
            )
            self.assertEqual(len(predictions), len(dense_x))
            self.assertLessEqual(
                _negative_interval_count(dense_x, predictions, breakpoint=breakpoint),
                1,
            )

    @unittest.skipUnless(_have_scipy(), "scipy is required")
    def test_smart_solver_smooth_curve_disables_low_x_window_or_falls_back_cleanly(self) -> None:
        xs = [0.0003, 0.0006, 0.0009, 0.0012, 0.0016, 0.0022, 0.0035, 0.0060, 0.0120]
        rows = _build_curve_cp_rows(control_periods=[40.0, 80.0, 120.0], xs=xs, low_x_bend=False)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = _create_smart_solver_db(tmpdir, rows)
            result = backend.td_smart_solver_run(
                db_path,
                output_target="Output",
                input1_target="Input1",
                runs=["CondA"],
                serials=["SN-001"],
            )

        self.assertIn(
            result["fit_family"],
            {
                backend.TD_PERF_FIT_FAMILY_HYBRID_QUADRATIC_RESIDUAL_CONTROL_PERIOD,
                backend.TD_PERF_FIT_FAMILY_QUADRATIC_CURVE_CONTROL_PERIOD,
            },
        )
        if result["fit_family"] == backend.TD_PERF_FIT_FAMILY_HYBRID_QUADRATIC_RESIDUAL_CONTROL_PERIOD:
            self.assertFalse(bool(result.get("low_x_window_enabled")))
            self.assertFalse(bool(result.get("fallback_used")))
        else:
            self.assertTrue(bool(result.get("fallback_used")))
            self.assertIsInstance(result.get("support_profile"), dict)

    @unittest.skipUnless(_have_scipy(), "scipy is required")
    def test_hybrid_curve_cp_fit_uses_linear_cp_degree_with_two_periods(self) -> None:
        xs = [0.0003, 0.0006, 0.0009, 0.0012, 0.0016, 0.0022, 0.0035, 0.0060]
        rows = _build_curve_cp_rows(control_periods=[60.0, 120.0], xs=xs, low_x_bend=True)
        x_values = [float(row["input_1"]) for row in rows]
        y_values = [float(row["output"]) for row in rows]
        cp_values = [float(row["control_period"]) for row in rows]

        model = backend._td_perf_fit_hybrid_quadratic_residual_control_period_model(
            x_values,
            y_values,
            cp_values,
        )

        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(
            str(model.get("fit_family") or ""),
            backend.TD_PERF_FIT_FAMILY_HYBRID_QUADRATIC_RESIDUAL_CONTROL_PERIOD,
        )
        self.assertEqual(int(model.get("control_period_degree") or 0), 1)
        residual_cp_models = model.get("residual_cp_models") or []
        self.assertEqual(len(residual_cp_models), 3)
        self.assertTrue(all(len(coeffs) == 2 for coeffs in residual_cp_models))
        predictions = backend.td_perf_predict_model(
            model,
            x_values,
            control_period=cp_values,
        )
        self.assertEqual(len(predictions), len(x_values))
        self.assertTrue(all(math.isfinite(float(value)) for value in predictions))


if __name__ == "__main__":
    unittest.main()
