import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtCore, QtWidgets
    from ui_next.qt_main import ProjectTaskWorker, TestDataTrendDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    ProjectTaskWorker = None  # type: ignore[assignment]
    TestDataTrendDialog = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or TestDataTrendDialog is None or ProjectTaskWorker is None, "PySide6 is required")
class TestQtMainSmartSolverConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _make_window(self) -> TestDataTrendDialog:
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        workbook_path = project_dir / "project.xlsx"
        workbook_path.write_text("", encoding="utf-8")
        with patch.object(TestDataTrendDialog, "_load_cache", lambda self, *, rebuild: None), patch.object(
            TestDataTrendDialog, "_load_auto_plots", lambda self: None
        ):
            window = TestDataTrendDialog(project_dir, workbook_path)
        window._test_tmpdir = tmpdir  # type: ignore[attr-defined]
        return window

    def test_smart_solver_config_summary_and_backend_call_include_sequence_cap(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._smart_solver_config = {
                    "run_type_mode": "pulsed_mode",
                    "output_target": "Output",
                    "input1_target": "Input1",
                    "input2_target": "",
                    "input3_target": "",
                    "control_period_hard_input": True,
                    "keep_first_sequences_per_serial": 0,
                }
                self.assertIn("Condition Family = Pulsed mode", window._smart_solver_config_text())
                self.assertIn("Seq cap = Unlimited", window._smart_solver_config_text())

                window._smart_solver_config["keep_first_sequences_per_serial"] = 3
                config_text = window._smart_solver_config_text()
                self.assertIn("Seq cap = first 3 / serial", config_text)
                self.assertIn("2D Solver", config_text)

                solver_calls: list[dict[str, object]] = []

                def _fake_solver(*_args, **kwargs):
                    solver_calls.append(dict(kwargs))
                    return {
                        "equation": "y = x",
                        "x_norm_equation": "x' = x",
                        "rmse": 0.0,
                        "residual_threshold": 0.0,
                        "in_fit_percent": 100.0,
                        "fell_out_count": 0,
                        "sample_count": 2,
                        "warning_text": "",
                        "slice_rows": [],
                        "keep_first_sequences_per_serial": int(kwargs.get("keep_first_sequences_per_serial") or 0),
                        "dropped_sequence_count": 1,
                        "dropped_point_count": 2,
                    }

                with patch.object(TestDataTrendDialog, "_active_serials", return_value=["SN-001"]), patch.object(
                    TestDataTrendDialog, "_active_program_filter_values", return_value=[]
                ), patch.object(
                    TestDataTrendDialog, "_active_suppression_voltage_filter_values", return_value=[]
                ), patch.object(
                    TestDataTrendDialog, "_active_control_period_filter_values", return_value=[]
                ), patch(
                    "ui_next.qt_main.be.td_smart_solver_has_sequence_rows", return_value=True
                ), patch(
                    "ui_next.qt_main.be.td_list_runs", return_value=["CondA", "CondB"]
                ), patch(
                    "ui_next.qt_main.be.td_smart_solver_run", side_effect=_fake_solver
                ), patch.object(
                    ProjectTaskWorker, "start", lambda self: self.run()
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.warning"
                ) as warning_mock, patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._run_smart_solver(user_initiated=True)

                warning_mock.assert_not_called()
                info_mock.assert_not_called()
                self.assertEqual(len(solver_calls), 1)
                self.assertEqual(solver_calls[0]["keep_first_sequences_per_serial"], 3)
                self.assertEqual(solver_calls[0]["input3_target"], "")
                self.assertEqual(solver_calls[0]["run_type_mode"], "pulsed_mode")
                self.assertIn("Dropped: 1 seq / 2 pts", window.lbl_smart_solver_summary.text())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_smart_solver_config_popup_shows_input3_guidance(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._perf_available_columns = [
                    {"name": "Output"},
                    {"name": "Input1"},
                    {"name": "Input2"},
                    {"name": "Input3"},
                ]
                captured: dict[str, str] = {}

                def _inspect_dialog():
                    dialog = window._smart_solver_popup
                    captured["condition_family"] = dialog.findChild(
                        QtWidgets.QComboBox, "smart_solver_combo_condition_family"
                    ).currentData()
                    captured["guide"] = dialog.findChild(QtWidgets.QLabel, "smart_solver_input_guide").text()
                    captured["output"] = dialog.findChild(QtWidgets.QLabel, "smart_solver_help_output").text()
                    captured["input3"] = dialog.findChild(QtWidgets.QLabel, "smart_solver_help_input3").text()
                    captured["control_period"] = dialog.findChild(
                        QtWidgets.QLabel, "smart_solver_help_control_period"
                    ).text()
                    dialog.reject()
                    return 0

                with patch(
                    "ui_next.qt_main.QtWidgets.QDialog.exec",
                    side_effect=_inspect_dialog,
                ):
                    window._open_smart_solver_config_popup()

                self.assertEqual(captured["condition_family"], "pulsed_mode")
                self.assertIn("direct 3D solve", captured["guide"])
                self.assertEqual(captured["output"], "Metric the solver predicts.")
                self.assertIn("Correlation helper", captured["input3"])
                self.assertIn("remains unchanged", captured["control_period"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_smart_solver_config_popup_steady_state_hides_cp_controls_and_updates_copy(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._perf_available_columns = [
                    {"name": "Output"},
                    {"name": "Input1"},
                    {"name": "Input2"},
                    {"name": "Input3"},
                ]
                captured: dict[str, object] = {}

                def _run_dialog():
                    dialog = window._smart_solver_popup
                    combo = dialog.findChild(QtWidgets.QComboBox, "smart_solver_combo_condition_family")
                    combo.setCurrentIndex(1)
                    self._app.processEvents()
                    captured["guide"] = dialog.findChild(QtWidgets.QLabel, "smart_solver_input_guide").text()
                    captured["cp_help_visible"] = dialog.findChild(
                        QtWidgets.QLabel, "smart_solver_help_control_period"
                    ).isVisible()
                    captured["cp_filters"] = [
                        label.text()
                        for label in dialog.findChildren(QtWidgets.QLabel)
                        if "Ignored in steady-state mode." in str(label.text() or "")
                    ]
                    dialog.reject()
                    return 0

                with patch(
                    "ui_next.qt_main.QtWidgets.QDialog.exec",
                    side_effect=_run_dialog,
                ):
                    window._open_smart_solver_config_popup()

                self.assertIn("steady-state 3D solve", str(captured["guide"]))
                self.assertFalse(bool(captured["cp_help_visible"]))
                self.assertTrue(bool(captured["cp_filters"]))
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_render_smart_solver_result_shows_stability_and_clamp_details(self) -> None:
        window = self._make_window()
        try:
            window._render_smart_solver_result(
                {
                    "equation": "y = x",
                    "x_norm_equation": "x' = x",
                    "rmse": 1.5,
                    "residual_threshold": 3.0,
                    "in_fit_percent": 80.0,
                    "fell_out_count": 2,
                    "sample_count": 10,
                    "warning_text": "",
                    "slice_rows": [{"scope": "steady_state", "point_count": 10, "distinct_input_1": 3, "eligible": True, "reason": ""}],
                    "solver_branch": "staged_mediator_control_period",
                    "selection_reason": "Test stability summary.",
                    "uses_staged_mediator": True,
                    "uses_control_period": False,
                    "stability_ok": False,
                    "stage2_fit_source": "stage1_pred_input_3",
                    "mediator_clamp_count": 3,
                }
            )

            self.assertIn("Stability: Review", window.lbl_smart_solver_summary.text())
            self.assertIn("Mediator clamps: 3", window.lbl_smart_solver_summary.text())
            self.assertIn("stage1_pred_input_3", window.lbl_smart_solver_warning.text())
            self.assertIn("Mediator clamp hits: 3.", window.lbl_smart_solver_warning.text())
            self.assertEqual(window.tbl_smart_solver_diagnostics.horizontalHeaderItem(0).text(), "scope")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_smart_solver_config_popup_rejects_duplicate_input3_selection(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._perf_available_columns = [
                    {"name": "Output"},
                    {"name": "Input1"},
                    {"name": "Input2"},
                    {"name": "Input3"},
                ]

                def _select_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
                    for index in range(combo.count()):
                        if combo.itemText(index) == value:
                            combo.setCurrentIndex(index)
                            return
                    raise AssertionError(f"missing combo value {value!r}")

                def _run_dialog():
                    dialog = window._smart_solver_popup
                    _select_combo_value(dialog.findChild(QtWidgets.QComboBox, "smart_solver_combo_output"), "Output")
                    _select_combo_value(dialog.findChild(QtWidgets.QComboBox, "smart_solver_combo_input1"), "Input1")
                    _select_combo_value(dialog.findChild(QtWidgets.QComboBox, "smart_solver_combo_input2"), "Input2")
                    _select_combo_value(dialog.findChild(QtWidgets.QComboBox, "smart_solver_combo_input3"), "Input2")
                    dialog.findChild(QtWidgets.QPushButton, "smart_solver_btn_solve").click()
                    dialog.reject()
                    return 0

                with patch(
                    "ui_next.qt_main.QtWidgets.QDialog.exec",
                    side_effect=_run_dialog,
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.warning"
                ) as warning_mock, patch.object(
                    TestDataTrendDialog,
                    "_run_smart_solver",
                ) as run_mock:
                    window._open_smart_solver_config_popup()

                warning_mock.assert_called_once()
                self.assertIn("must be different", warning_mock.call_args[0][2])
                run_mock.assert_not_called()
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_smart_solver_exportability_accepts_supported_solver_families(self) -> None:
        window = self._make_window()
        try:
            window._smart_solver_result = {
                "master_model": {
                    "fit_family": "quadratic_3input",
                }
            }
            self.assertTrue(window._smart_solver_has_exportable_model())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_smart_solver_matlab_export_button_tracks_exportable_state(self) -> None:
        class _Worker:
            def __init__(self, running: bool) -> None:
                self._running = running

            def isRunning(self) -> bool:
                return self._running

        window = self._make_window()
        try:
            window._mode = "smart_solver"
            window._db_path = Path("C:/temp/cache.sqlite3")
            window._smart_solver_result = {
                "master_model": {
                    "fit_family": "quadratic_3input",
                }
            }

            self.assertEqual(window.btn_solver_export_matlab.text(), "Export Equation to MATLAB")

            window._export_worker = None
            window._refresh_smart_solver_ui()
            self.assertTrue(window.btn_solver_export_equations.isEnabled())
            self.assertTrue(window.btn_solver_export_matlab.isEnabled())

            window._export_worker = _Worker(True)
            window._refresh_smart_solver_ui()
            self.assertFalse(window.btn_solver_export_equations.isEnabled())
            self.assertFalse(window.btn_solver_export_matlab.isEnabled())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_export_smart_solver_equations_to_excel_uses_dedicated_solver_export_path(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                out_path = Path(tmpdir) / "smart_solver_export.xlsx"
                window._db_path = db_path
                window._run_display_by_name = {"CondA": "Condition A"}
                window._smart_solver_config = {
                    "run_type_mode": "pulsed_mode",
                    "output_target": "Output",
                    "input1_target": "Input1",
                    "input2_target": "",
                    "input3_target": "",
                    "control_period_hard_input": True,
                    "keep_first_sequences_per_serial": 2,
                }
                window._smart_solver_result = {
                    "fit_family": "quadratic_curve_control_period",
                    "run_type_mode": "pulsed_mode",
                    "uses_control_period": True,
                    "master_model": {
                        "fit_family": "quadratic_curve_control_period",
                        "coeff_cp_models": [[1.0], [2.0], [3.0]],
                        "x_center": 0.0,
                        "x_scale": 1.0,
                        "cp_center": 0.0,
                        "cp_scale": 1.0,
                    },
                    "fit_points": [
                        {
                            "run_name": "CondA",
                            "serial": "SN-001",
                            "observation_id": "obs-1",
                            "program_title": "Program Alpha",
                            "source_run_name": "Seq-1",
                            "suppression_voltage": 5.0,
                            "control_period": 30.0,
                            "condition_label": "Condition A",
                            "input_1": 1.0,
                            "input_2": None,
                            "actual_mean": 11.0,
                            "sample_count": 1,
                        }
                    ],
                    "solver_variables": [
                        {
                            "key": "input_1",
                            "target": "Input1",
                            "units": "u",
                            "role": "Primary",
                            "is_optional": False,
                        }
                    ],
                    "output_target": "Output",
                    "output_units": "u",
                    "input1_target": "Input1",
                    "input1_units": "u",
                    "input2_target": "",
                    "input2_units": "",
                    "input3_target": "",
                    "input3_units": "",
                }
                export_calls: list[dict[str, object]] = []
                dialog_defaults: list[str] = []

                def _capture_start(output_path: Path, **kwargs):
                    export_calls.append({"output_path": Path(output_path), **kwargs})

                def _capture_save_dialog(*args, **kwargs):
                    dialog_defaults.append(str(args[2]))
                    return (str(out_path), "Excel Files (*.xlsx)")

                with patch.object(
                    TestDataTrendDialog,
                    "_active_control_period_filter_values",
                    return_value=[30.0],
                ), patch.object(
                    TestDataTrendDialog,
                    "_start_smart_solver_equation_excel_export",
                    side_effect=_capture_start,
                ), patch(
                    "ui_next.qt_main.be.td_perf_collect_asset_metadata",
                    return_value={},
                ), patch(
                    "ui_next.qt_main.QtWidgets.QFileDialog.getSaveFileName",
                    side_effect=_capture_save_dialog,
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock, patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.warning"
                ) as warning_mock:
                    window._export_smart_solver_equations_to_excel()

                info_mock.assert_not_called()
                warning_mock.assert_not_called()
                self.assertEqual(len(export_calls), 1)
                call = export_calls[0]
                self.assertEqual(call["output_path"], out_path)
                self.assertEqual(call["plot_metadata"]["selected_control_period"], 30.0)
                self.assertEqual(call["result"]["master_model"]["fit_family"], "quadratic_curve_control_period")
                self.assertEqual(call["plot_metadata"]["solver_variables"][0]["target"], "Input1")
                self.assertTrue(dialog_defaults)
                self.assertIn("Output_vs_Input1", dialog_defaults[0])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_export_smart_solver_equations_to_matlab_uses_dedicated_solver_export_path(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                out_path = Path(tmpdir) / "smart_solver_export.m"
                window._db_path = db_path
                window._run_display_by_name = {"CondA": "Condition A"}
                window._smart_solver_config = {
                    "run_type_mode": "pulsed_mode",
                    "output_target": "Output",
                    "input1_target": "Input1",
                    "input2_target": "",
                    "input3_target": "",
                    "control_period_hard_input": True,
                    "keep_first_sequences_per_serial": 2,
                }
                window._smart_solver_result = {
                    "fit_family": "quadratic_curve_control_period",
                    "run_type_mode": "pulsed_mode",
                    "uses_control_period": True,
                    "master_model": {
                        "fit_family": "quadratic_curve_control_period",
                        "coeff_cp_models": [[1.0], [2.0], [3.0]],
                        "x_center": 0.0,
                        "x_scale": 1.0,
                        "cp_center": 0.0,
                        "cp_scale": 1.0,
                    },
                    "fit_points": [
                        {
                            "run_name": "CondA",
                            "serial": "SN-001",
                            "observation_id": "obs-1",
                            "program_title": "Program Alpha",
                            "source_run_name": "Seq-1",
                            "suppression_voltage": 5.0,
                            "control_period": 30.0,
                            "condition_label": "Condition A",
                            "input_1": 1.0,
                            "input_2": None,
                            "actual_mean": 11.0,
                            "sample_count": 1,
                        }
                    ],
                    "solver_variables": [
                        {
                            "key": "input_1",
                            "target": "Input1",
                            "units": "u",
                            "role": "Primary",
                            "is_optional": False,
                        }
                    ],
                    "output_target": "Output",
                    "output_units": "u",
                    "input1_target": "Input1",
                    "input1_units": "u",
                    "input2_target": "",
                    "input2_units": "",
                    "input3_target": "",
                    "input3_units": "",
                }
                export_calls: list[dict[str, object]] = []
                dialog_defaults: list[str] = []

                def _capture_start(output_path: Path, **kwargs):
                    export_calls.append({"output_path": Path(output_path), **kwargs})

                def _capture_save_dialog(*args, **kwargs):
                    dialog_defaults.append(str(args[2]))
                    return (str(out_path), "MATLAB Files (*.m)")

                with patch.object(
                    TestDataTrendDialog,
                    "_active_control_period_filter_values",
                    return_value=[30.0],
                ), patch(
                    "ui_next.qt_main.QtWidgets.QInputDialog.getItem",
                    return_value=("Clean export", True),
                ), patch.object(
                    TestDataTrendDialog,
                    "_start_smart_solver_equation_matlab_export",
                    side_effect=_capture_start,
                ), patch(
                    "ui_next.qt_main.be.td_perf_collect_asset_metadata",
                    return_value={},
                ), patch(
                    "ui_next.qt_main.QtWidgets.QFileDialog.getSaveFileName",
                    side_effect=_capture_save_dialog,
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock, patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.warning"
                ) as warning_mock:
                    window._export_smart_solver_equations_to_matlab()

                info_mock.assert_not_called()
                warning_mock.assert_not_called()
                self.assertEqual(len(export_calls), 1)
                call = export_calls[0]
                self.assertEqual(call["output_path"], out_path)
                self.assertEqual(call["export_mode"], "clean")
                self.assertEqual(call["plot_metadata"]["selected_control_period"], 30.0)
                self.assertEqual(call["result"]["master_model"]["fit_family"], "quadratic_curve_control_period")
                self.assertEqual(call["plot_metadata"]["solver_variables"][0]["target"], "Input1")
                self.assertTrue(dialog_defaults)
                self.assertTrue(dialog_defaults[0].endswith(".m"))
                self.assertIn("Output_vs_Input1", dialog_defaults[0])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_export_smart_solver_equations_to_excel_omits_selected_control_period_for_steady_state(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                out_path = Path(tmpdir) / "smart_solver_export.xlsx"
                window._db_path = db_path
                window._smart_solver_config = {
                    "run_type_mode": "steady_state",
                    "output_target": "Output",
                    "input1_target": "Input1",
                    "input2_target": "Input2",
                    "input3_target": "Input3",
                    "control_period_hard_input": True,
                    "keep_first_sequences_per_serial": 0,
                }
                window._smart_solver_result = {
                    "fit_family": "quadratic_3input",
                    "run_type_mode": "steady_state",
                    "uses_control_period": False,
                    "master_model": {
                        "fit_family": "quadratic_3input",
                        "coeffs": [0.0] * 10,
                        "x1_center": 0.0,
                        "x1_scale": 1.0,
                        "x2_center": 0.0,
                        "x2_scale": 1.0,
                        "x3_center": 0.0,
                        "x3_scale": 1.0,
                    },
                    "fit_points": [
                        {
                            "run_name": "CondSS",
                            "serial": "SN-001",
                            "observation_id": "obs-1",
                            "program_title": "Program Alpha",
                            "source_run_name": "Seq-1",
                            "suppression_voltage": 5.0,
                            "condition_label": "Condition A",
                            "input_1": 1.0,
                            "input_2": 2.0,
                            "input_3": 3.0,
                            "actual_mean": 11.0,
                            "sample_count": 1,
                        }
                    ],
                    "solver_variables": [
                        {
                            "key": "input_1",
                            "target": "Input1",
                            "units": "u",
                            "role": "Primary",
                            "is_optional": False,
                        },
                        {
                            "key": "input_2",
                            "target": "Input2",
                            "units": "u",
                            "role": "Secondary",
                            "is_optional": True,
                        },
                        {
                            "key": "input_3",
                            "target": "Input3",
                            "units": "u",
                            "role": "Helper",
                            "is_optional": True,
                        }
                    ],
                    "output_target": "Output",
                    "output_units": "u",
                    "input1_target": "Input1",
                    "input1_units": "u",
                    "input2_target": "Input2",
                    "input2_units": "u",
                    "input3_target": "Input3",
                    "input3_units": "u",
                }
                export_calls: list[dict[str, object]] = []

                def _capture_start(output_path: Path, **kwargs):
                    export_calls.append({"output_path": Path(output_path), **kwargs})

                with patch.object(
                    TestDataTrendDialog,
                    "_active_control_period_filter_values",
                    return_value=[30.0],
                ), patch.object(
                    TestDataTrendDialog,
                    "_start_smart_solver_equation_excel_export",
                    side_effect=_capture_start,
                ), patch(
                    "ui_next.qt_main.be.td_perf_collect_asset_metadata",
                    return_value={},
                ), patch(
                    "ui_next.qt_main.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(str(out_path), "Excel Files (*.xlsx)"),
                ):
                    window._export_smart_solver_equations_to_excel()

                self.assertEqual(export_calls[0]["plot_metadata"]["selected_control_period"], None)
                self.assertEqual(export_calls[0]["plot_metadata"]["run_type_mode"], "steady_state")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_export_smart_solver_equations_to_matlab_omits_selected_control_period_for_steady_state(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                out_path = Path(tmpdir) / "smart_solver_export.m"
                window._db_path = db_path
                window._smart_solver_config = {
                    "run_type_mode": "steady_state",
                    "output_target": "Output",
                    "input1_target": "Input1",
                    "input2_target": "Input2",
                    "input3_target": "Input3",
                    "control_period_hard_input": True,
                    "keep_first_sequences_per_serial": 0,
                }
                window._smart_solver_result = {
                    "fit_family": "quadratic_3input",
                    "run_type_mode": "steady_state",
                    "uses_control_period": False,
                    "master_model": {
                        "fit_family": "quadratic_3input",
                        "coeffs": [0.0] * 10,
                        "x1_center": 0.0,
                        "x1_scale": 1.0,
                        "x2_center": 0.0,
                        "x2_scale": 1.0,
                        "x3_center": 0.0,
                        "x3_scale": 1.0,
                    },
                    "fit_points": [
                        {
                            "run_name": "CondSS",
                            "serial": "SN-001",
                            "observation_id": "obs-1",
                            "program_title": "Program Alpha",
                            "source_run_name": "Seq-1",
                            "suppression_voltage": 5.0,
                            "condition_label": "Condition A",
                            "input_1": 1.0,
                            "input_2": 2.0,
                            "input_3": 3.0,
                            "actual_mean": 11.0,
                            "sample_count": 1,
                        }
                    ],
                    "solver_variables": [
                        {
                            "key": "input_1",
                            "target": "Input1",
                            "units": "u",
                            "role": "Primary",
                            "is_optional": False,
                        },
                        {
                            "key": "input_2",
                            "target": "Input2",
                            "units": "u",
                            "role": "Secondary",
                            "is_optional": True,
                        },
                        {
                            "key": "input_3",
                            "target": "Input3",
                            "units": "u",
                            "role": "Helper",
                            "is_optional": True,
                        }
                    ],
                    "output_target": "Output",
                    "output_units": "u",
                    "input1_target": "Input1",
                    "input1_units": "u",
                    "input2_target": "Input2",
                    "input2_units": "u",
                    "input3_target": "Input3",
                    "input3_units": "u",
                }
                export_calls: list[dict[str, object]] = []

                def _capture_start(output_path: Path, **kwargs):
                    export_calls.append({"output_path": Path(output_path), **kwargs})

                with patch.object(
                    TestDataTrendDialog,
                    "_active_control_period_filter_values",
                    return_value=[30.0],
                ), patch(
                    "ui_next.qt_main.QtWidgets.QInputDialog.getItem",
                    return_value=("Export with regression checks", True),
                ), patch.object(
                    TestDataTrendDialog,
                    "_start_smart_solver_equation_matlab_export",
                    side_effect=_capture_start,
                ), patch(
                    "ui_next.qt_main.be.td_perf_collect_asset_metadata",
                    return_value={},
                ), patch(
                    "ui_next.qt_main.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(str(out_path), "MATLAB Files (*.m)"),
                ):
                    window._export_smart_solver_equations_to_matlab()

                self.assertEqual(export_calls[0]["plot_metadata"]["selected_control_period"], None)
                self.assertEqual(export_calls[0]["plot_metadata"]["run_type_mode"], "steady_state")
                self.assertEqual(export_calls[0]["export_mode"], "regression_checks")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_trend_analyze_left_panel_width_stays_bounded_across_modes(self) -> None:
        window = self._make_window()
        try:
            window.show()
            self._app.processEvents()

            preferred_curves_width = window._preferred_left_panel_width()
            self.assertLess(preferred_curves_width, window.width())
            self.assertLessEqual(window._left_panel_scroll.width(), preferred_curves_width)

            window._set_mode("performance")
            self._app.processEvents()

            preferred_perf_width = window._preferred_left_panel_width()
            self.assertLess(preferred_perf_width, window.width())
            self.assertLessEqual(window._left_panel_scroll.width(), preferred_perf_width)
            self.assertLess(window._tabs.currentWidget().sizeHint().width(), window.width())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_top_auto_report_button_is_placed_under_global_filters(self) -> None:
        window = self._make_window()
        try:
            self.assertEqual(window.btn_auto_report.text(), "Auto Report...")
            self.assertFalse(window.btn_auto_report.isEnabled())

            root_layout = window.layout()
            self.assertIsNotNone(root_layout)
            self.assertLess(root_layout.indexOf(window.filter_frame), root_layout.indexOf(window.auto_report_frame))
            self.assertLess(root_layout.indexOf(window.auto_report_frame), root_layout.indexOf(window.main_splitter))

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._plot_ready = True
                window._sync_main_auto_plot_actions()
                self.assertTrue(window.btn_auto_report.isEnabled())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_filter_helpers_limit_sequences_and_serials_to_matching_scope(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program A", "Program B"]
            window._available_control_period_filters = ["10", "20"]
            window._available_suppression_voltage_filters = ["5", "7"]
            window._available_serial_filter_rows = [
                {"serial": "SN-001", "program_title": "Program A"},
                {"serial": "SN-002", "program_title": "Program B"},
            ]
            window._global_filter_rows = [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq A",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-002",
                    "program_title": "Program B",
                    "source_run_name": "Seq B",
                    "control_period": 20.0,
                    "suppression_voltage": 7.0,
                },
            ]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:run_a|Program A|Seq A",
                        "run_name": "run_a",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_sequences": ["Seq A"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:run_b|Program B|Seq B",
                        "run_name": "run_b",
                        "program_title": "Program B",
                        "member_programs": ["Program B"],
                        "member_sequences": ["Seq B"],
                        "member_control_periods": ["20"],
                        "member_suppression_voltages": ["7"],
                    },
                ],
                "condition": [],
            }
            filter_state = {
                "programs": ["Program A"],
                "serials": ["SN-001"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            }

            items = window._visible_run_selection_items_for_filter_state(
                "sequence",
                filter_state=filter_state,
                require_active_serial_match=True,
            )
            self.assertEqual([item["id"] for item in items], ["sequence:run_a|Program A|Seq A"])

            serial_rows = window._serial_rows_for_run_selections(items, filter_state=filter_state)
            self.assertEqual([row["serial"] for row in serial_rows], ["SN-001"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_control_period_filter_keeps_steady_state_sequences_visible(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program A"]
            window._available_control_period_filters = ["10", "20"]
            window._available_suppression_voltage_filters = ["5"]
            window._available_serial_filter_rows = [{"serial": "SN-001", "program_title": "Program A"}]
            window._global_filter_rows = [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq Pulse",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq Steady",
                    "control_period": None,
                    "suppression_voltage": 5.0,
                },
            ]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:pulse",
                        "run_name": "run_pulse",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_sequences": ["Seq Pulse"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:steady",
                        "run_name": "run_steady",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_sequences": ["Seq Steady"],
                        "member_control_periods": [],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["steady_state"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:pulse-other",
                        "run_name": "run_other",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_sequences": ["Seq Pulse Other"],
                        "member_control_periods": ["20"],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                ],
                "condition": [],
            }
            filter_state = {
                "programs": ["Program A"],
                "serials": ["SN-001"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            }

            items = window._visible_auto_report_run_selection_items_for_filter_state(
                "sequence",
                filter_state=filter_state,
                require_active_serial_match=True,
            )
            self.assertEqual(
                [item["id"] for item in items],
                ["sequence:pulse", "sequence:steady"],
            )
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_control_period_filter_keeps_steady_state_run_conditions_visible(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program A"]
            window._available_control_period_filters = ["10", "20"]
            window._available_suppression_voltage_filters = ["5"]
            window._available_serial_filter_rows = [{"serial": "SN-001", "program_title": "Program A"}]
            window._global_filter_rows = [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq Pulse",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq Steady",
                    "control_period": None,
                    "suppression_voltage": 5.0,
                },
            ]
            window._run_selection_views = {
                "sequence": [],
                "condition": [
                    {
                        "mode": "condition",
                        "id": "condition:pulse",
                        "run_name": "run_pulse",
                        "run_condition": "250 psia",
                        "member_programs": ["Program A"],
                        "member_runs": ["run_pulse"],
                        "member_sequences": ["Seq Pulse"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                    {
                        "mode": "condition",
                        "id": "condition:steady",
                        "run_name": "run_steady",
                        "run_condition": "250 psia",
                        "member_programs": ["Program A"],
                        "member_runs": ["run_steady"],
                        "member_sequences": ["Seq Steady"],
                        "member_control_periods": [],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["steady_state"],
                    },
                    {
                        "mode": "condition",
                        "id": "condition:pulse-other",
                        "run_name": "run_other",
                        "run_condition": "250 psia",
                        "member_programs": ["Program A"],
                        "member_runs": ["run_other"],
                        "member_sequences": ["Seq Pulse Other"],
                        "member_control_periods": ["20"],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                ],
            }
            filter_state = {
                "programs": ["Program A"],
                "serials": ["SN-001"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            }

            items = window._visible_auto_report_run_selection_items_for_filter_state(
                "condition",
                filter_state=filter_state,
                require_active_serial_match=True,
            )
            self.assertEqual(
                [item["id"] for item in items],
                ["condition:pulse", "condition:steady"],
            )
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_certification_helpers_ignore_suppression_but_keep_family_filter_scope(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program Alpha", "Program Beta"]
            window._checked_program_filters = ["Program Alpha", "Program Beta"]
            window._available_control_period_filters = ["10"]
            window._checked_control_period_filters = ["10"]
            window._available_suppression_voltage_filters = ["5", "10"]
            window._checked_suppression_voltage_filters = ["5"]
            window._available_serial_filter_rows = [
                {"serial": "SN-002", "program_title": "Program Alpha"},
                {"serial": "SN-101", "program_title": "Program Beta"},
            ]
            window._checked_serial_filters = ["SN-002", "SN-101"]
            window._global_filter_rows = [
                {
                    "serial": "SN-002",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq 1",
                    "control_period": 10.0,
                    "suppression_voltage": 10.0,
                },
                {
                    "serial": "SN-101",
                    "program_title": "Program Beta",
                    "source_run_name": "Seq 1",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
            ]
            filter_state = {
                "programs": ["Program Alpha", "Program Beta"],
                "serials": ["SN-002", "SN-101"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            }

            self.assertEqual(window._active_serials(filter_state=filter_state), ["SN-101"])
            self.assertEqual(
                window._auto_report_certifying_program_options(filter_state=filter_state),
                ["Program Alpha", "Program Beta"],
            )
            self.assertEqual(
                [row["serial"] for row in window._auto_report_serial_rows_for_certifying_program("Program Alpha", filter_state=filter_state)],
                ["SN-002"],
            )
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_run_selection_visibility_ignores_suppression_membership(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program Alpha"]
            window._available_control_period_filters = ["10"]
            window._available_suppression_voltage_filters = ["5", "10"]
            window._available_serial_filter_rows = [{"serial": "SN-002", "program_title": "Program Alpha"}]
            window._global_filter_rows = [
                {
                    "serial": "SN-002",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq 10V",
                    "control_period": 10.0,
                    "suppression_voltage": 10.0,
                }
            ]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:seq10v",
                        "run_name": "run_seq10v",
                        "program_title": "Program Alpha",
                        "member_programs": ["Program Alpha"],
                        "member_sequences": ["Seq 10V"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["10"],
                        "member_run_type_modes": ["pulsed_mode"],
                    }
                ],
                "condition": [],
            }
            filter_state = {
                "programs": ["Program Alpha"],
                "serials": ["SN-002"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            }

            items = window._visible_auto_report_run_selection_items_for_filter_state(
                "sequence",
                filter_state=filter_state,
                require_active_serial_match=True,
            )
            self.assertEqual([item["id"] for item in items], ["sequence:seq10v"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_certification_run_selection_visibility_tracks_selected_serials(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program Alpha", "Program Beta"]
            window._available_control_period_filters = ["10"]
            window._available_suppression_voltage_filters = ["5", "10"]
            window._available_serial_filter_rows = [
                {"serial": "SN-001", "program_title": "Program Alpha"},
                {"serial": "SN-002", "program_title": "Program Alpha"},
                {"serial": "SN-101", "program_title": "Program Beta"},
            ]
            window._global_filter_rows = [
                {
                    "serial": "SN-001",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq Alpha 1",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-002",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq Alpha 2",
                    "control_period": 10.0,
                    "suppression_voltage": 10.0,
                },
                {
                    "serial": "SN-101",
                    "program_title": "Program Beta",
                    "source_run_name": "Seq Beta 1",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
            ]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:alpha1",
                        "run_name": "run_alpha_1",
                        "program_title": "Program Alpha",
                        "member_programs": ["Program Alpha"],
                        "member_sequences": ["Seq Alpha 1"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:alpha2",
                        "run_name": "run_alpha_2",
                        "program_title": "Program Alpha",
                        "member_programs": ["Program Alpha"],
                        "member_sequences": ["Seq Alpha 2"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["10"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:beta1",
                        "run_name": "run_beta_1",
                        "program_title": "Program Beta",
                        "member_programs": ["Program Beta"],
                        "member_sequences": ["Seq Beta 1"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                        "member_run_type_modes": ["pulsed_mode"],
                    },
                ],
                "condition": [],
            }
            filter_state = {
                "programs": ["Program Alpha", "Program Beta"],
                "serials": ["SN-001", "SN-002", "SN-101"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            }

            items = window._visible_auto_report_certification_run_selection_items_for_filter_state(
                "sequence",
                certifying_program="Program Alpha",
                certification_serials=["SN-002"],
                filter_state=filter_state,
            )
            self.assertEqual([item["id"] for item in items], ["sequence:alpha2"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_dialog_emits_certification_payload_and_default_output(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as td:
                repo = Path(td) / "repo"
                repo.mkdir(parents=True, exist_ok=True)
                db_path = Path(td) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._plot_ready = True
                window._db_path = db_path
                window._available_program_filters = ["Program Alpha", "Program Beta"]
                window._checked_program_filters = ["Program Alpha", "Program Beta"]
                window._available_control_period_filters = ["10"]
                window._checked_control_period_filters = ["10"]
                window._available_suppression_voltage_filters = ["5", "10"]
                window._checked_suppression_voltage_filters = ["5", "10"]
                window._available_serial_filter_rows = [
                    {"serial": "SN-001", "program_title": "Program Alpha"},
                    {"serial": "SN-002", "program_title": "Program Alpha"},
                    {"serial": "SN-101", "program_title": "Program Beta"},
                ]
                window._checked_serial_filters = ["SN-001", "SN-002", "SN-101"]
                window._global_filter_rows = [
                    {
                        "serial": "SN-001",
                        "program_title": "Program Alpha",
                        "source_run_name": "Seq 1",
                        "control_period": 10.0,
                        "suppression_voltage": 5.0,
                    },
                    {
                        "serial": "SN-002",
                        "program_title": "Program Alpha",
                        "source_run_name": "Seq 1",
                        "control_period": 10.0,
                        "suppression_voltage": 10.0,
                    },
                    {
                        "serial": "SN-101",
                        "program_title": "Program Beta",
                        "source_run_name": "Seq 1",
                        "control_period": 10.0,
                        "suppression_voltage": 5.0,
                    },
                ]
                window._run_selection_views = {
                    "sequence": [
                        {
                            "mode": "sequence",
                            "id": "sequence:seq1",
                            "run_name": "run_seq1",
                            "program_title": "Program Alpha",
                            "member_programs": ["Program Alpha", "Program Beta"],
                            "member_runs": ["run_seq1"],
                            "member_sequences": ["Seq 1"],
                            "member_control_periods": ["10"],
                            "member_suppression_voltages": ["5", "10"],
                            "member_run_type_modes": ["pulsed_mode"],
                        }
                    ],
                    "condition": [],
                }

                captured: dict[str, object] = {}

                def _run_dialog() -> int:
                    cert_popup = next(
                        (
                            widget
                            for widget in self._app.topLevelWidgets()
                            if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == "Certification Specifics"
                        ),
                        None,
                    )
                    if cert_popup is not None:
                        popup_program = cert_popup.findChild(QtWidgets.QComboBox, "auto_report_cert_popup_program")
                        popup_serials = cert_popup.findChild(QtWidgets.QListWidget, "auto_report_cert_popup_serials")
                        popup_runs = cert_popup.findChild(QtWidgets.QListWidget, "auto_report_cert_popup_runs")
                        self.assertIsNotNone(popup_program)
                        self.assertIsNotNone(popup_serials)
                        self.assertIsNotNone(popup_runs)
                        popup_program.setCurrentText("Program Alpha")
                        self._app.processEvents()
                        self.assertEqual(
                            [popup_serials.item(i).text() for i in range(popup_serials.count())],
                            ["SN-001", "SN-002"],
                        )
                        popup_serials.item(1).setSelected(True)
                        self._app.processEvents()
                        self.assertGreaterEqual(popup_runs.count(), 1)
                        apply_btn = next(
                            btn for btn in cert_popup.findChildren(QtWidgets.QPushButton) if btn.text() == "Apply"
                        )
                        apply_btn.click()
                        return int(QtWidgets.QDialog.DialogCode.Accepted)

                    dialog = next(
                        widget
                        for widget in self._app.topLevelWidgets()
                        if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == "Auto Report Options"
                    )
                    cert_button = dialog.findChild(
                        QtWidgets.QPushButton, "auto_report_certification_popup_button"
                    )
                    report_name = dialog.findChild(QtWidgets.QLineEdit, "auto_report_report_name")
                    output_dir = dialog.findChild(QtWidgets.QLineEdit, "auto_report_output_dir")
                    grade_summary = dialog.findChild(QtWidgets.QLabel, "auto_report_grade_scoring_summary")
                    params_title = dialog.findChild(QtWidgets.QLabel, "auto_report_certification_params_title")
                    params_list = dialog.findChild(QtWidgets.QListWidget, "auto_report_certification_params")
                    self.assertIsNotNone(cert_button)
                    self.assertIsNotNone(report_name)
                    self.assertIsNotNone(output_dir)
                    self.assertIsNotNone(grade_summary)
                    self.assertIsNotNone(params_title)
                    self.assertIsNotNone(params_list)
                    self.assertIn("Family Serials...", [btn.text() for btn in dialog.findChildren(QtWidgets.QPushButton)])
                    self.assertIn("Certification Specifics...", [btn.text() for btn in dialog.findChildren(QtWidgets.QPushButton)])
                    self.assertEqual(params_title.text(), "Certification Parameter Selection (required)")
                    self.assertEqual(params_list.count(), 2)
                    self.assertTrue(
                        all(
                            params_list.item(i).checkState() == QtCore.Qt.CheckState.Unchecked
                            for i in range(params_list.count())
                        )
                    )
                    for i in range(params_list.count()):
                        params_list.item(i).setCheckState(QtCore.Qt.CheckState.Checked)
                    self.assertIn("z =", grade_summary.text())
                    self.assertIn("PASS if |z| <= 1.5", grade_summary.text())
                    self.assertIn("WATCH if |z| <= 2.5", grade_summary.text())
                    cert_button.click()
                    self._app.processEvents()
                    self.assertIn("Program Alpha", report_name.text())
                    self.assertIn("SN-002", report_name.text())
                    self.assertTrue(
                        output_dir.text().endswith(
                            str(Path("EDIN Program Folders") / "Program Alpha" / "EDAT reports")
                        )
                    )
                    gen_btn = next(btn for btn in dialog.findChildren(QtWidgets.QPushButton) if btn.text() == "Generate Report")
                    gen_btn.click()
                    return int(QtWidgets.QDialog.DialogCode.Accepted)

                def _capture_payload(*args, **_kwargs) -> None:
                    payload = args[-1] if args else {}
                    captured["payload"] = dict(payload)

                with patch("ui_next.qt_main.be.get_repo_root", return_value=repo), patch(
                    "ui_next.qt_main.be.td_cached_statistics", return_value=["mean"]
                ), patch(
                    "ui_next.qt_main.be.load_trend_auto_report_config",
                    return_value={"grading": {"zscore_pass_max": 1.5, "zscore_watch_max": 2.5}},
                ), patch(
                    "ui_next.qt_main.be.load_excel_trend_config", return_value={}
                ), patch(
                    "ui_next.qt_main.be.td_list_y_columns", return_value=[{"name": "Thrust"}, {"name": "Flow"}]
                ), patch.object(
                    TestDataTrendDialog, "_run_auto_report", side_effect=_capture_payload
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock, patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.warning"
                ) as warn_mock, patch(
                    "ui_next.qt_main.QtWidgets.QDialog.exec",
                    side_effect=_run_dialog,
                ), patch(
                    "ui_next.qt_main._fit_widget_to_screen",
                    lambda *_args, **_kwargs: None,
                ):
                    window._open_auto_report_options()
                info_mock.assert_not_called()
                warn_mock.assert_not_called()

                payload = captured["payload"]
                self.assertEqual(payload["certifying_program"], "Program Alpha")
                self.assertEqual(payload["highlighted_serials"], ["SN-002"])
                self.assertEqual(payload["filtered_serials"], ["SN-001", "SN-002", "SN-101"])
                self.assertEqual(payload["params"], ["Flow", "Thrust"])
                self.assertIn("Program Alpha", str(payload["output_pdf"]))
                self.assertIn("EDAT reports", str(payload["output_pdf"]))
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_run_selection_items_honor_valve_voltage_filter_membership(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program A"]
            window._available_control_period_filters = ["10"]
            window._available_valve_voltage_filters = ["28", "32"]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:seq28",
                        "run_name": "run_28",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_control_periods": ["10"],
                        "member_valve_voltages": ["28"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:seq32",
                        "run_name": "run_32",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_control_periods": ["10"],
                        "member_valve_voltages": ["32"],
                    },
                ],
                "condition": [],
            }

            items = window._visible_run_selection_items_for_filter_state(
                "sequence",
                filter_state={
                    "programs": ["Program A"],
                    "control_periods": ["10"],
                    "valve_voltages": ["28"],
                },
            )

            self.assertEqual([item["id"] for item in items], ["sequence:seq28"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_report_visibility_ignores_suppression_but_keeps_valve_scope(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program Alpha"]
            window._available_control_period_filters = ["10"]
            window._available_suppression_voltage_filters = ["5", "10"]
            window._available_valve_voltage_filters = ["28", "32"]
            window._available_serial_filter_rows = [{"serial": "SN-002", "program_title": "Program Alpha"}]
            window._global_filter_rows = [
                {
                    "serial": "SN-002",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq 32V",
                    "control_period": 10.0,
                    "suppression_voltage": 10.0,
                    "valve_voltage": 32.0,
                }
            ]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:seq32v",
                        "run_name": "run_seq32v",
                        "program_title": "Program Alpha",
                        "member_programs": ["Program Alpha"],
                        "member_sequences": ["Seq 32V"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["10"],
                        "member_valve_voltages": ["32"],
                        "member_run_type_modes": ["pulsed_mode"],
                    }
                ],
                "condition": [],
            }

            visible = window._visible_auto_report_run_selection_items_for_filter_state(
                "sequence",
                filter_state={
                    "programs": ["Program Alpha"],
                    "serials": ["SN-002"],
                    "control_periods": ["10"],
                    "suppression_voltages": ["5"],
                    "valve_voltages": ["32"],
                },
                require_active_serial_match=True,
            )
            hidden = window._visible_auto_report_run_selection_items_for_filter_state(
                "sequence",
                filter_state={
                    "programs": ["Program Alpha"],
                    "serials": ["SN-002"],
                    "control_periods": ["10"],
                    "suppression_voltages": ["5"],
                    "valve_voltages": ["28"],
                },
                require_active_serial_match=True,
            )

            self.assertEqual([item["id"] for item in visible], ["sequence:seq32v"])
            self.assertEqual(hidden, [])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_performance_and_solver_modes_ignore_live_valve_filters(self) -> None:
        window = self._make_window()
        try:
            window._available_program_filters = ["Program A"]
            window._checked_program_filters = ["Program A"]
            window._available_control_period_filters = ["10"]
            window._checked_control_period_filters = ["10"]
            window._available_suppression_voltage_filters = ["5"]
            window._checked_suppression_voltage_filters = ["5"]
            window._available_valve_voltage_filters = ["28", "32"]
            window._checked_valve_voltage_filters = ["28"]
            window._available_serial_filter_rows = [
                {"serial": "SN-001", "program_title": "Program A"},
                {"serial": "SN-002", "program_title": "Program A"},
            ]
            window._checked_serial_filters = ["SN-001", "SN-002"]
            window._global_filter_rows = [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq 28V",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                    "valve_voltage": 28.0,
                },
                {
                    "serial": "SN-002",
                    "program_title": "Program A",
                    "source_run_name": "Seq 32V",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                    "valve_voltage": 32.0,
                },
            ]
            window._run_selection_views = {
                "sequence": [
                    {
                        "mode": "sequence",
                        "id": "sequence:seq28",
                        "run_name": "run_28",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_sequences": ["Seq 28V"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                        "member_valve_voltages": ["28"],
                    },
                    {
                        "mode": "sequence",
                        "id": "sequence:seq32",
                        "run_name": "run_32",
                        "program_title": "Program A",
                        "member_programs": ["Program A"],
                        "member_sequences": ["Seq 32V"],
                        "member_control_periods": ["10"],
                        "member_suppression_voltages": ["5"],
                        "member_valve_voltages": ["32"],
                    },
                ],
                "condition": [],
            }

            window._mode = "curves"
            self.assertEqual(window._active_serials(), ["SN-001"])
            self.assertEqual(
                [item["id"] for item in window._visible_run_selection_items("sequence")],
                ["sequence:seq28"],
            )

            window._mode = "performance"
            self.assertEqual(window._active_serials(), ["SN-001", "SN-002"])
            self.assertEqual(
                [item["id"] for item in window._visible_run_selection_items("sequence")],
                ["sequence:seq28", "sequence:seq32"],
            )

            window._mode = "smart_solver"
            self.assertEqual(window._active_serials(), ["SN-001", "SN-002"])
            self.assertEqual(
                [item["id"] for item in window._visible_run_selection_items("sequence")],
                ["sequence:seq28", "sequence:seq32"],
            )
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
