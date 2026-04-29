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

    def _merged_context(self) -> dict[str, object]:
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
                },
                {
                    "program_title": "Program B",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "PulsePressureAvg",
                    "default_display_parameter": "Pulse Pressure",
                    "displayed_parameter": "Pulse Pressure",
                    "preferred_units": "psia-second",
                    "default_preferred_units": "psia-second",
                    "enabled": True,
                    "edited": False,
                },
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
                },
                {
                    "program_title": "Program B",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "raw_name": "PulsePressureAvg",
                    "displayed_parameter": "Pulse Pressure",
                    "default_display_parameter": "Pulse Pressure",
                    "preferred_units": "psia-second",
                    "default_preferred_units": "psia-second",
                    "units": ["psia-second"],
                    "enabled": True,
                    "source_count": 1,
                    "source_run_names": ["Run-2"],
                    "surfaces": ["metrics"],
                    "run_names": ["Run-2"],
                    "status": "default",
                },
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

    def _advance_to_phase2(self, dialog: TDParameterNormalizationDialog, *, merged: bool = False) -> None:
        repo_rows = self._merged_context()["repo_parameter_rows"] if merged else self._base_context()["repo_parameter_rows"]
        phase2_rows = [
            {
                "canonical_id": "display:pulsepressure",
                "displayed_parameter": "Pulse Pressure",
                "raw_names": ["PulsePressure", "PulsePressureAvg"] if merged else ["PulsePressure"],
                "raw_names_text": "PulsePressure, PulsePressureAvg" if merged else "PulsePressure",
                "program_titles": ["Program A", "Program B"] if merged else ["Program A"],
                "program_titles_text": "Program A, Program B" if merged else "Program A",
                "source_units": ["psia-second"],
                "preferred_units": "psia-second",
                "unit_conflict": False,
            }
        ]
        with patch("ui_next.qt_main.be.save_td_repo_parameter_mappings", return_value=[dict(row) for row in repo_rows]), patch(
            "ui_next.qt_main.be.td_rebuild_project_parameter_units_catalog",
            return_value={"groups": []},
        ), patch(
            "ui_next.qt_main.be.td_build_project_parameter_units_rows",
            return_value=phase2_rows,
        ):
            dialog.btn_save.click()
            self._app.processEvents()

    def test_phase1_hides_units_and_allows_display_edits(self) -> None:
        dialog = self._make_dialog()

        headers = [dialog.tbl.horizontalHeaderItem(idx).text() for idx in range(dialog.tbl.columnCount())]
        self.assertEqual(headers, dialog.PHASE1_COLS)
        self.assertTrue(dialog.btn_set_units.isHidden())

        dialog.tbl.selectRow(0)
        self._app.processEvents()
        with patch.object(dialog, "_prompt_display_name", return_value="Pulse Pressure Display"):
            dialog._act_set_display_name()
        self._app.processEvents()

        self.assertTrue(dialog._dirty)
        self.assertEqual(str(dialog._working_rows[0].get("displayed_parameter") or ""), "Pulse Pressure Display")
        self.assertEqual(dialog.tbl.item(0, dialog.COL_STATUS).text(), "Edited")

    def test_save_phase1_advances_to_units_phase(self) -> None:
        dialog = self._make_dialog(sync_worker=True, context=self._merged_context())

        self._advance_to_phase2(dialog, merged=True)

        self.assertEqual(dialog._phase, dialog.PHASE_UNITS)
        self.assertEqual(dialog.result(), 0)
        self.assertFalse(dialog.update_requested)
        headers = [dialog.tbl.horizontalHeaderItem(idx).text() for idx in range(dialog.tbl.columnCount())]
        self.assertEqual(headers, dialog.PHASE2_COLS)
        self.assertFalse(dialog.btn_set_units.isHidden())

    def test_phase2_merges_matching_displayed_parameters(self) -> None:
        dialog = self._make_dialog(sync_worker=True, context=self._merged_context())

        self._advance_to_phase2(dialog, merged=True)

        self.assertEqual(dialog.tbl.rowCount(), 1)
        self.assertIn("PulsePressure", dialog.tbl.item(0, dialog.UNIT_COL_RAW_NAMES).text())
        self.assertIn("PulsePressureAvg", dialog.tbl.item(0, dialog.UNIT_COL_RAW_NAMES).text())
        self.assertIn("Program A", dialog.tbl.item(0, dialog.UNIT_COL_PROGRAMS).text())
        self.assertIn("Program B", dialog.tbl.item(0, dialog.UNIT_COL_PROGRAMS).text())

    def test_save_phase2_sets_update_requested(self) -> None:
        dialog = self._make_dialog(sync_worker=True, context=self._merged_context())
        self._advance_to_phase2(dialog, merged=True)
        dialog.tbl.selectRow(0)
        self._app.processEvents()

        with patch.object(dialog, "_prompt_units", return_value="psi"):
            dialog._act_set_units()
        self._app.processEvents()

        save_units_mock = Mock(return_value={"saved_rows": [dict(row) for row in self._merged_context()["repo_parameter_rows"]]})
        with patch("ui_next.qt_main.be.td_save_project_parameter_units", save_units_mock):
            dialog.btn_save.click()
            self._app.processEvents()

        self.assertEqual(dialog.result(), int(QtWidgets.QDialog.DialogCode.Accepted))
        self.assertTrue(dialog.update_requested)
        save_units_mock.assert_called_once()

    def test_close_phase2_discard_keeps_saved_phase1_without_update(self) -> None:
        dialog = self._make_dialog(sync_worker=True, context=self._merged_context())
        self._advance_to_phase2(dialog, merged=True)
        dialog.tbl.selectRow(0)
        self._app.processEvents()

        with patch.object(dialog, "_prompt_units", return_value="psi"):
            dialog._act_set_units()
        self._app.processEvents()

        with patch(
            "ui_next.qt_main.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Discard,
        ):
            dialog.close()
            self._app.processEvents()

        self.assertEqual(dialog.result(), int(QtWidgets.QDialog.DialogCode.Rejected))
        self.assertFalse(dialog.update_requested)
        self.assertTrue(dialog.saved)
