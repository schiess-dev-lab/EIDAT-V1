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
    from ui_next.qt_main import MainWindow, ProjectTaskWorker
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]
    ProjectTaskWorker = None  # type: ignore[assignment]


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
        self.actions_updated = 0

    def _update_project_actions(self) -> None:
        self.actions_updated += 1

    def _append_log(self, text: str) -> None:
        self.logs.append(str(text))


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
