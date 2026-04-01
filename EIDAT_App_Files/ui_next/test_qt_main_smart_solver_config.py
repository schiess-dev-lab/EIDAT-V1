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
    from PySide6 import QtWidgets
    from ui_next.qt_main import ProjectTaskWorker, TestDataTrendDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
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


if __name__ == "__main__":
    unittest.main()
