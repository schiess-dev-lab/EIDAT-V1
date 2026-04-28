import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtWidgets
    from ui_next.qt_main import ProjectTaskWorker, TDParameterNormalizationDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtWidgets = None  # type: ignore[assignment]
    ProjectTaskWorker = None  # type: ignore[assignment]
    TDParameterNormalizationDialog = None  # type: ignore[assignment]


def _start_worker_sync(worker: ProjectTaskWorker) -> None:
    worker.run()


@unittest.skipIf(
    QtWidgets is None or ProjectTaskWorker is None or TDParameterNormalizationDialog is None,
    "PySide6 is required",
)
class TestTDParameterNormalizationDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _base_context(self) -> dict[str, object]:
        return {
            "repo_parameter_rows": [
                {
                    "program_title": "Program A",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "PulsePressure",
                    "default_display_parameter": "Pulse Pressure",
                    "displayed_parameter": "Pulse Pressure",
                    "preferred_units": "psia-second",
                    "default_preferred_units": "psia-second",
                    "enabled": True,
                    "edited": False,
                }
            ],
            "inventory": [
                {
                    "program_title": "Program A",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "raw_name": "PulsePressure",
                    "displayed_parameter": "Pulse Pressure",
                    "default_display_parameter": "Pulse Pressure",
                    "preferred_units": "psia-second",
                    "default_preferred_units": "psia-second",
                    "units": ["psia-second"],
                    "enabled": True,
                    "source_count": 1,
                    "source_run_names": ["Run-1"],
                    "surfaces": ["metrics"],
                    "run_names": ["Run-1"],
                    "status": "default",
                }
            ],
        }

    def _cleanup_dialog(self, dialog: TDParameterNormalizationDialog) -> None:
        try:
            dialog._allow_direct_close = True
        except Exception:
            pass
        try:
            dialog.close()
        except Exception:
            pass
        self._app.processEvents()

    def _make_dialog(self, *, sync_worker: bool = False, context: dict[str, object] | None = None):
        stack = ExitStack()
        context = dict(context or self._base_context())
        stack.enter_context(
            patch("ui_next.qt_main.be.validate_test_data_project_cache_for_open", return_value=Path("dummy.sqlite3"))
        )
        stack.enter_context(
            patch("ui_next.qt_main.be.td_load_parameter_runtime_context", return_value=context)
        )
        if sync_worker:
            stack.enter_context(patch.object(ProjectTaskWorker, "start", _start_worker_sync))
        dialog = TDParameterNormalizationDialog(Path("."), Path("dummy.xlsx"))
        self.addCleanup(stack.close)
        self.addCleanup(self._cleanup_dialog, dialog)
        return dialog

    def _set_dirty_units(self, dialog: TDParameterNormalizationDialog, value: str = "psia second") -> None:
        dialog.tbl.selectRow(0)
        self._app.processEvents()
        with patch.object(dialog, "_prompt_units", return_value=value):
            dialog._act_set_units()
        self._app.processEvents()
        if dialog._autosave_timer.isActive():
            dialog._autosave_timer.stop()

    def test_set_units_keeps_exact_text_and_marks_row_edited(self) -> None:
        dialog = self._make_dialog()

        self._set_dirty_units(dialog, "psia second")

        self.assertTrue(dialog._dirty)
        self.assertEqual(str(dialog._working_rows[0].get("preferred_units") or ""), "psia second")
        self.assertIn("psia second", dialog.tbl.item(0, dialog.COL_UNITS).text())
        self.assertEqual(dialog.tbl.item(0, dialog.COL_STATUS).text(), "Edited")

    def test_window_close_save_closes_and_requests_update_without_runtime_refresh(self) -> None:
        save_mock = Mock(side_effect=lambda _project_dir, rows: [dict(item) for item in rows])
        runtime_refresh_mock = Mock(return_value={"mode": "rebuilt"})
        support_refresh_mock = Mock(return_value={"updated": True})
        dialog = self._make_dialog(sync_worker=True)
        self._set_dirty_units(dialog, "psia second")
        dialog.show()
        self._app.processEvents()

        with patch("ui_next.qt_main.be.save_td_repo_parameter_mappings", save_mock), patch(
            "ui_next.qt_main.be.refresh_td_parameter_runtime_cache",
            runtime_refresh_mock,
        ), patch(
            "ui_next.qt_main.be.refresh_td_parameter_support_workbook",
            support_refresh_mock,
        ), patch(
            "ui_next.qt_main.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ):
            dialog.close()
            self._app.processEvents()

        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), int(QtWidgets.QDialog.DialogCode.Accepted))
        self.assertTrue(dialog.update_requested)
        save_mock.assert_called_once()
        runtime_refresh_mock.assert_not_called()
        support_refresh_mock.assert_not_called()

    def test_window_close_discard_closes_without_update(self) -> None:
        dialog = self._make_dialog()
        self._set_dirty_units(dialog, "psia second")
        dialog.show()
        self._app.processEvents()

        with patch(
            "ui_next.qt_main.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Discard,
        ):
            dialog.close()
            self._app.processEvents()

        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), int(QtWidgets.QDialog.DialogCode.Rejected))
        self.assertFalse(dialog.update_requested)

    def test_window_close_cancel_leaves_dialog_open(self) -> None:
        dialog = self._make_dialog()
        self._set_dirty_units(dialog, "psia second")
        dialog.show()
        self._app.processEvents()

        with patch(
            "ui_next.qt_main.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
        ):
            dialog.close()
            self._app.processEvents()

        self.assertTrue(dialog.isVisible())
        self.assertEqual(dialog.result(), 0)
        self.assertFalse(dialog.update_requested)

    def test_close_button_save_matches_window_close_save_behavior(self) -> None:
        save_mock = Mock(side_effect=lambda _project_dir, rows: [dict(item) for item in rows])
        runtime_refresh_mock = Mock(return_value={"mode": "rebuilt"})
        support_refresh_mock = Mock(return_value={"updated": True})
        dialog = self._make_dialog(sync_worker=True)
        self._set_dirty_units(dialog, "psia second")
        dialog.show()
        self._app.processEvents()

        with patch("ui_next.qt_main.be.save_td_repo_parameter_mappings", save_mock), patch(
            "ui_next.qt_main.be.refresh_td_parameter_runtime_cache",
            runtime_refresh_mock,
        ), patch(
            "ui_next.qt_main.be.refresh_td_parameter_support_workbook",
            support_refresh_mock,
        ), patch(
            "ui_next.qt_main.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ):
            dialog.btn_close.click()
            self._app.processEvents()

        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), int(QtWidgets.QDialog.DialogCode.Accepted))
        self.assertTrue(dialog.update_requested)
        save_mock.assert_called_once()
        runtime_refresh_mock.assert_not_called()
        support_refresh_mock.assert_not_called()

    def test_close_update_saves_and_closes_without_prompt(self) -> None:
        save_mock = Mock(side_effect=lambda _project_dir, rows: [dict(item) for item in rows])
        runtime_refresh_mock = Mock(return_value={"mode": "rebuilt"})
        support_refresh_mock = Mock(return_value={"updated": True})
        dialog = self._make_dialog(sync_worker=True)
        self._set_dirty_units(dialog, "psia second")
        dialog.show()
        self._app.processEvents()

        with patch("ui_next.qt_main.be.save_td_repo_parameter_mappings", save_mock), patch(
            "ui_next.qt_main.be.refresh_td_parameter_runtime_cache",
            runtime_refresh_mock,
        ), patch(
            "ui_next.qt_main.be.refresh_td_parameter_support_workbook",
            support_refresh_mock,
        ), patch("ui_next.qt_main.QtWidgets.QMessageBox.question") as question_mock:
            dialog.btn_close_update.click()
            self._app.processEvents()

        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), int(QtWidgets.QDialog.DialogCode.Accepted))
        self.assertTrue(dialog.update_requested)
        save_mock.assert_called_once()
        runtime_refresh_mock.assert_not_called()
        support_refresh_mock.assert_not_called()
        question_mock.assert_not_called()

    def test_save_now_runs_support_and_runtime_refresh(self) -> None:
        save_mock = Mock(side_effect=lambda _project_dir, rows: [dict(item) for item in rows])
        runtime_refresh_mock = Mock(return_value={"mode": "rebuilt"})
        support_refresh_mock = Mock(return_value={"updated": True})
        dialog = self._make_dialog(sync_worker=True)
        self._set_dirty_units(dialog, "psia second")

        with patch("ui_next.qt_main.be.save_td_repo_parameter_mappings", save_mock), patch(
            "ui_next.qt_main.be.refresh_td_parameter_runtime_cache",
            runtime_refresh_mock,
        ), patch(
            "ui_next.qt_main.be.refresh_td_parameter_support_workbook",
            support_refresh_mock,
        ):
            dialog.btn_save.click()
            self._app.processEvents()

        self.assertTrue(dialog.saved)
        self.assertFalse(dialog.update_requested)
        save_mock.assert_called_once()
        runtime_refresh_mock.assert_called_once()
        support_refresh_mock.assert_called_once()
