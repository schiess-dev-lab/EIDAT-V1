import os
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
    from ui_next.qt_main import MainWindow, ProjectTaskWorker, TestDataTrendDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]
    ProjectTaskWorker = None  # type: ignore[assignment]
    TestDataTrendDialog = None  # type: ignore[assignment]


class _DummyRepoScanDialog:
    def __init__(self) -> None:
        self.finished: list[tuple[str, bool]] = []

    def finish(self, text: str, *, success: bool) -> None:
        self.finished.append((str(text), bool(success)))


class _DummyMainWindow:
    def __init__(self) -> None:
        self._project_worker = object()
        self._project_popup_active = True
        self._repo_scan_dialog = _DummyRepoScanDialog()
        self.logs: list[str] = []
        self.toasts: list[str] = []
        self.actions_updated = 0

    def _update_project_actions(self) -> None:
        self.actions_updated += 1

    def _append_log(self, text: str) -> None:
        self.logs.append(str(text))

    def _show_toast(self, text: str) -> None:
        self.toasts.append(str(text))


@unittest.skipIf(QtWidgets is None or ProjectTaskWorker is None or MainWindow is None, "PySide6 is required")
class TestProjectTaskWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_worker_failure_writes_all_error_lines_to_task_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "project_task.log"
            progress_messages: list[str] = []
            failed_messages: list[str] = []

            def _task(report) -> None:
                report("before failure")
                raise RuntimeError("Line one\nLine two")

            worker = ProjectTaskWorker(_task, log_path=log_path)
            worker.progress.connect(progress_messages.append)
            worker.failed.connect(failed_messages.append)
            worker.run()

            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("Log file:", log_text)
            self.assertIn("before failure", log_text)
            self.assertIn("ERROR: Line one", log_text)
            self.assertIn("ERROR: Line two", log_text)
            self.assertTrue(any("before failure" == msg for msg in progress_messages))
            self.assertEqual(len(failed_messages), 1)
            self.assertIn("Line one\nLine two", failed_messages[0])
            self.assertIn(f"Log: {log_path}", failed_messages[0])

    def test_project_task_error_uses_full_message_in_log_and_popup(self) -> None:
        dummy = _DummyMainWindow()
        message = (
            "Detailed failure reason\n"
            "TD cache debug: C:\\temp\\td_cache_debug.json\n"
            "Log: C:\\temp\\project_task.log"
        )

        with patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
            MainWindow._on_project_task_error(dummy, message, "Update Project")

        self.assertIsNone(dummy._project_worker)
        self.assertEqual(dummy.actions_updated, 1)
        self.assertTrue(
            any(log == f"[PROJECT TASK ERROR] Update Project: {message}" for log in dummy.logs)
        )
        warning_mock.assert_called_once_with(dummy, "Update Project", message)
        self.assertEqual(
            dummy._repo_scan_dialog.finished,
            [(f"Update Project failed: {message}", False)],
        )
        self.assertFalse(dummy._project_popup_active)

    def test_project_update_success_popup_includes_excluded_source_warning(self) -> None:
        dummy = _DummyMainWindow()
        payload = {
            "updated_cells": 12,
            "missing_source": 1,
            "missing_value": 0,
            "serials_in_workbook": 2,
            "serials_with_source": 1,
            "serials_added": 0,
            "added_serials": [],
            "compiled_serials": ["SN-001"],
            "compiled_serials_count": 1,
            "excluded_sources": [
                {
                    "serial": "SN-002",
                    "status": "missing",
                    "reason": "No usable raw curves were written for any discovered run.",
                    "metadata_rel": "docs/sn002.json",
                    "artifacts_rel": "debug/ocr/sn002",
                    "excel_sqlite_rel": "missing_source.sqlite3",
                }
            ],
            "excluded_sources_count": 1,
            "warning_summary": "Excluded 1 TD source from compilation (missing=1, invalid=0). They will be ignored until fixed.",
            "workbook": "C:\\temp\\project.xlsx",
            "cache_sync_mode": "full_rebuild",
            "cache_sync_reason": "context changed",
            "cache_sync_counts": {
                "added": 0,
                "changed": 0,
                "removed": 0,
                "unchanged": 1,
                "missing": 1,
                "invalid": 0,
                "reingested": 0,
            },
            "cache_state": {
                "impl_counts": {"td_runs": 1, "td_columns_calc_y": 1, "td_metrics_calc": 2},
                "raw_counts": {"td_raw_sequences": 1, "td_columns_raw_y": 1, "td_curves_raw": 1},
                "source_status_counts": {"missing": 1, "invalid": 0, "non_ok": 1},
            },
            "cache_validation_ok": True,
            "cache_validation_error": "",
            "cache_validation_summary": "mode=none, compiled_serials=1, excluded_sources=1",
            "cache_debug_path": "C:\\temp\\td_cache_debug.json",
            "backend_module_path": "C:\\temp\\backend.py",
            "saved_equation_refresh": {"refreshed_count": 0, "failed_count": 0, "errors": []},
            "debug_json": "",
        }

        with patch.object(QtWidgets.QMessageBox, "information") as info_mock:
            result = MainWindow._handle_project_update_success(
                dummy,
                payload,
                wb_path=Path("C:/temp/project.xlsx"),
                ptype="Test Data Trending",
                started=0.0,
            )

        info_mock.assert_called_once()
        popup_text = info_mock.call_args.args[2]
        self.assertIn("Excluded 1 TD source from compilation", popup_text)
        self.assertIn("SN-002 (docs/sn002.json)", popup_text)
        self.assertIn("will be ignored until fixed", popup_text)
        self.assertTrue(any("Excluded TD sources: 1" in log for log in dummy.logs))
        self.assertTrue(any("SN-002 (docs/sn002.json)" in log for log in dummy.logs))
        self.assertEqual(dummy.toasts, ["Project updated with warnings: 12 cell(s)"])
        self.assertIn("with warnings", result)

    def test_test_data_trend_load_cache_uses_fast_open_validator(self) -> None:
        class _DummyTrendDialog:
            def __init__(self) -> None:
                self._project_dir = Path("C:/temp/project")
                self._workbook_path = Path("C:/temp/project/project.xlsx")
                self.lbl_source = QtWidgets.QLabel("")
                self.lbl_cache = QtWidgets.QLabel("")
                self._db_path = None
                self.refresh_calls = 0
                self.zoom_calls = 0

            def _refresh_from_cache(self) -> None:
                self.refresh_calls += 1

            def _update_plot_zoom_actions(self) -> None:
                self.zoom_calls += 1

        dummy = _DummyTrendDialog()
        expected_db = Path("C:/temp/project/implementation_trending.sqlite3")

        with patch("ui_next.qt_main.be.validate_test_data_project_cache_for_open", return_value=expected_db) as open_validate_mock:
            TestDataTrendDialog._load_cache(dummy, rebuild=False)

        open_validate_mock.assert_called_once_with(dummy._project_dir, dummy._workbook_path)
        self.assertEqual(dummy._db_path, expected_db)
        self.assertEqual(dummy.lbl_source.text(), str(expected_db))
        self.assertEqual(dummy.lbl_cache.text(), f"Cache DB: {expected_db}")
        self.assertEqual(dummy.refresh_calls, 1)
        self.assertEqual(dummy.zoom_calls, 1)

    def test_start_perf_equation_excel_export_uses_project_task(self) -> None:
        class _DummyExportWindow:
            def __init__(self) -> None:
                self._db_path = Path("C:/temp/cache.sqlite3")
                self._project_dir = Path("C:/temp/project")
                self.started: dict[str, object] | None = None
                self.opened: list[Path] = []

            def _start_perf_export_task(self, **kwargs) -> None:
                self.started = dict(kwargs)

            def _open_spreadsheet_path(self, file_path: Path) -> None:
                self.opened.append(Path(file_path))

            def _handle_perf_excel_export_success(self, payload: object, *, heading: str) -> str:
                return TestDataTrendDialog._handle_perf_excel_export_success(self, payload, heading=heading)

        dummy = _DummyExportWindow()
        out_path = Path("C:/temp/performance.xlsx")

        with patch("ui_next.qt_main.be.td_perf_export_equation_workbook", return_value=out_path) as export_mock:
            TestDataTrendDialog._start_perf_equation_excel_export(
                dummy,
                out_path,
                plot_metadata={"output_target": "Efficiency"},
                results_by_stat={"mean": {"master_model": {"fit_family": "polynomial"}}},
                run_specs=[{"run_name": "cond_a"}],
                control_period_filter=12,
                run_type_filter="pulsed_mode",
            )

            self.assertIsNotNone(dummy.started)
            task_factory = dummy.started["task_factory"]
            on_success = dummy.started["on_success"]
            self.assertEqual(dummy.started["heading"], "Export Equation to Excel")
            self.assertEqual(dummy.started["status_text"], "Exporting equation workbook to performance.xlsx")

            progress_messages: list[str] = []
            payload = task_factory(progress_messages.append)
            export_mock.assert_called_once()
            export_mock.call_args.kwargs["progress_cb"]("step one")
            self.assertEqual(progress_messages, ["step one"])
            self.assertEqual(payload, out_path)

            result = on_success(payload)
            self.assertEqual(dummy.opened, [out_path])
            self.assertIn("Export Equation to Excel complete", result)

    def test_start_perf_interactive_equation_export_uses_project_task(self) -> None:
        class _DummyExportWindow:
            def __init__(self) -> None:
                self._db_path = Path("C:/temp/cache.sqlite3")
                self._project_dir = Path("C:/temp/project")
                self.started: dict[str, object] | None = None
                self.opened: list[Path] = []

            def _start_perf_export_task(self, **kwargs) -> None:
                self.started = dict(kwargs)

            def _open_spreadsheet_path(self, file_path: Path) -> None:
                self.opened.append(Path(file_path))

            def _handle_perf_excel_export_success(self, payload: object, *, heading: str) -> str:
                return TestDataTrendDialog._handle_perf_excel_export_success(self, payload, heading=heading)

        dummy = _DummyExportWindow()
        out_path = Path("C:/temp/interactive_performance.xlsx")

        with patch("ui_next.qt_main.be.td_perf_export_interactive_equation_workbook", return_value=out_path) as export_mock:
            TestDataTrendDialog._start_perf_interactive_equation_export(
                dummy,
                out_path,
                plot_metadata={"output_target": "Efficiency"},
                results_by_stat={"mean": {"master_model": {"fit_family": "polynomial"}}},
                run_specs=[{"run_name": "cond_a"}],
                control_period_filter=12,
                run_type_filter="pulsed_mode",
                include_regression_checker=False,
            )

            self.assertIsNotNone(dummy.started)
            task_factory = dummy.started["task_factory"]
            on_success = dummy.started["on_success"]
            self.assertEqual(dummy.started["heading"], "Export Interactive Workbook")
            self.assertEqual(dummy.started["status_text"], "Exporting interactive workbook to interactive_performance.xlsx")

            progress_messages: list[str] = []
            payload = task_factory(progress_messages.append)
            export_mock.assert_called_once()
            export_mock.call_args.kwargs["progress_cb"]("step one")
            self.assertEqual(progress_messages, ["step one"])
            self.assertEqual(export_mock.call_args.kwargs["include_regression_checker"], False)
            self.assertEqual(payload, out_path)

            result = on_success(payload)
            self.assertEqual(dummy.opened, [out_path])
            self.assertIn("Export Interactive Workbook complete", result)

    def test_update_perf_export_button_state_toggles_new_interactive_button(self) -> None:
        class _Worker:
            def __init__(self, running: bool) -> None:
                self._running = running

            def isRunning(self) -> bool:
                return self._running

        class _DummyExportButtons:
            def __init__(self) -> None:
                self.btn_perf_export_equations = QtWidgets.QPushButton()
                self.btn_perf_export_interactive = QtWidgets.QPushButton()
                self.btn_perf_save_equation = QtWidgets.QPushButton()
                self.btn_perf_saved_equations = QtWidgets.QPushButton()
                self._project_dir = Path("C:/temp/project")
                self._export_worker = None

            def _perf_has_exportable_models(self) -> bool:
                return True

        dummy = _DummyExportButtons()

        TestDataTrendDialog._update_perf_export_button_state(dummy)
        self.assertTrue(dummy.btn_perf_export_equations.isEnabled())
        self.assertTrue(dummy.btn_perf_export_interactive.isEnabled())
        self.assertTrue(dummy.btn_perf_save_equation.isEnabled())
        self.assertTrue(dummy.btn_perf_saved_equations.isEnabled())

        dummy._export_worker = _Worker(True)
        TestDataTrendDialog._update_perf_export_button_state(dummy)
        self.assertFalse(dummy.btn_perf_export_equations.isEnabled())
        self.assertFalse(dummy.btn_perf_export_interactive.isEnabled())
        self.assertFalse(dummy.btn_perf_save_equation.isEnabled())
        self.assertFalse(dummy.btn_perf_saved_equations.isEnabled())

    def test_start_saved_perf_equations_excel_export_uses_project_task(self) -> None:
        class _DummyExportWindow:
            def __init__(self) -> None:
                self._db_path = Path("C:/temp/cache.sqlite3")
                self._project_dir = Path("C:/temp/project")
                self.started: dict[str, object] | None = None
                self.opened: list[Path] = []

            def _start_perf_export_task(self, **kwargs) -> None:
                self.started = dict(kwargs)

            def _open_spreadsheet_path(self, file_path: Path) -> None:
                self.opened.append(Path(file_path))

            def _handle_perf_excel_export_success(self, payload: object, *, heading: str) -> str:
                return TestDataTrendDialog._handle_perf_excel_export_success(self, payload, heading=heading)

        dummy = _DummyExportWindow()
        out_path = Path("C:/temp/saved_performance.xlsx")
        entries = [{"name": "Saved Equation A"}]

        with patch("ui_next.qt_main.be.td_perf_export_saved_equations_workbook", return_value=out_path) as export_mock:
            TestDataTrendDialog._start_saved_perf_equations_excel_export(dummy, out_path, entries=entries)

            self.assertIsNotNone(dummy.started)
            task_factory = dummy.started["task_factory"]
            on_success = dummy.started["on_success"]
            self.assertEqual(dummy.started["heading"], "Export Saved Performance Equations to Excel")
            self.assertEqual(dummy.started["status_text"], "Exporting 1 saved equation(s) to saved_performance.xlsx")

            progress_messages: list[str] = []
            payload = task_factory(progress_messages.append)
            export_mock.assert_called_once_with(
                dummy._db_path,
                out_path,
                entries=entries,
                progress_cb=export_mock.call_args.kwargs["progress_cb"],
            )
            export_mock.call_args.kwargs["progress_cb"]("step one")
            self.assertEqual(progress_messages, ["step one"])
            self.assertEqual(payload, out_path)

            result = on_success(payload)
            self.assertEqual(dummy.opened, [out_path])
            self.assertIn("Export Saved Performance Equations to Excel complete", result)

    def test_start_smart_solver_equation_matlab_export_uses_project_task(self) -> None:
        class _DummyExportWindow:
            def __init__(self) -> None:
                self._db_path = Path("C:/temp/cache.sqlite3")
                self._project_dir = Path("C:/temp/project")
                self.started: dict[str, object] | None = None

            def _start_perf_export_task(self, **kwargs) -> None:
                self.started = dict(kwargs)

            def _handle_path_export_success(self, payload: object, *, heading: str) -> str:
                return TestDataTrendDialog._handle_path_export_success(self, payload, heading=heading)

        dummy = _DummyExportWindow()
        out_path = Path("C:/temp/smart_solver_equation.m")
        result_payload = {"equation": "y = x"}
        plot_metadata = {"output_target": "Output"}

        with patch("ui_next.qt_main.be.td_smart_solver_export_equation_matlab", return_value=out_path) as export_mock, patch(
            "ui_next.qt_main.be.open_path"
        ) as open_mock:
            TestDataTrendDialog._start_smart_solver_equation_matlab_export(
                dummy,
                out_path,
                result=result_payload,
                plot_metadata=plot_metadata,
            )

            self.assertIsNotNone(dummy.started)
            task_factory = dummy.started["task_factory"]
            on_success = dummy.started["on_success"]
            self.assertEqual(dummy.started["heading"], "Export Equation to MATLAB")
            self.assertEqual(dummy.started["status_text"], "Exporting MATLAB equation file to smart_solver_equation.m")

            progress_messages: list[str] = []
            payload = task_factory(progress_messages.append)
            export_mock.assert_called_once_with(
                out_path,
                result=result_payload,
                plot_metadata=plot_metadata,
                progress_cb=export_mock.call_args.kwargs["progress_cb"],
            )
            export_mock.call_args.kwargs["progress_cb"]("step one")
            self.assertEqual(progress_messages, ["step one"])
            self.assertEqual(payload, out_path)

            result = on_success(payload)
            open_mock.assert_called_once_with(out_path)
            self.assertIn("Export Equation to MATLAB complete", result)
