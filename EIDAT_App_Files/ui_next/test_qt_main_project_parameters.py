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

    def test_phase1_keeps_same_named_mappings_separate_by_program(self) -> None:
        dialog = self._make_dialog(context=self._merged_context())

        self.assertEqual(dialog.tbl.rowCount(), 2)
        self.assertEqual(dialog.tbl.item(0, dialog.COL_PROGRAM).text(), "Program A")
        self.assertEqual(dialog.tbl.item(1, dialog.COL_PROGRAM).text(), "Program B")
        self.assertEqual(dialog.tbl.item(0, dialog.COL_DISPLAYED).text(), "Pulse Pressure")
        self.assertEqual(dialog.tbl.item(1, dialog.COL_DISPLAYED).text(), "Pulse Pressure")

    def test_phase1_editing_one_row_does_not_change_sibling(self) -> None:
        dialog = self._make_dialog(context=self._merged_context())

        dialog.tbl.selectRow(0)
        self._app.processEvents()
        with patch.object(dialog, "_prompt_display_name", return_value="Pulse Pressure A"):
            dialog._act_set_display_name()
        self._app.processEvents()

        rows_by_program = {
            str(row.get("program_title") or ""): str(row.get("displayed_parameter") or "")
            for row in dialog._working_rows
        }
        self.assertEqual(rows_by_program["Program A"], "Pulse Pressure A")
        self.assertEqual(rows_by_program["Program B"], "Pulse Pressure")

    def test_phase1_disabling_one_row_does_not_change_sibling(self) -> None:
        dialog = self._make_dialog(context=self._merged_context())

        dialog.tbl.selectRow(0)
        self._app.processEvents()
        dialog._act_disable_rows()
        self._app.processEvents()

        enabled_by_program = {
            str(row.get("program_title") or ""): bool(row.get("enabled", True))
            for row in dialog._working_rows
        }
        self.assertFalse(enabled_by_program["Program A"])
        self.assertTrue(enabled_by_program["Program B"])

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

    def test_reopen_after_saved_phase1_discarded_phase2_uses_committed_mapping(self) -> None:
        state = {
            "rows": [dict(row) for row in self._base_context()["repo_parameter_rows"]],
            "inventory": [dict(item) for item in self._base_context()["inventory"]],
        }

        def _context_for_reload(*_args, **_kwargs):
            rows = [dict(row) for row in state["rows"]]
            row_by_scope = {
                (
                    str(row.get("program_title") or ""),
                    str(row.get("asset_type") or ""),
                    str(row.get("asset_specific_type") or ""),
                    str(row.get("ingested_parameter") or ""),
                ): dict(row)
                for row in rows
            }
            inventory: list[dict[str, object]] = []
            for raw_item in state["inventory"]:
                item = dict(raw_item)
                scope_key = (
                    str(item.get("program_title") or ""),
                    str(item.get("asset_type") or ""),
                    str(item.get("asset_specific_type") or ""),
                    str(item.get("raw_name") or ""),
                )
                row = row_by_scope.get(scope_key) or {}
                if row:
                    item["displayed_parameter"] = str(row.get("displayed_parameter") or item.get("displayed_parameter") or "").strip()
                    item["default_display_parameter"] = str(
                        row.get("default_display_parameter") or item.get("default_display_parameter") or ""
                    ).strip()
                    item["preferred_units"] = str(row.get("preferred_units") or item.get("preferred_units") or "").strip()
                    item["default_preferred_units"] = str(
                        row.get("default_preferred_units") or item.get("default_preferred_units") or ""
                    ).strip()
                    item["enabled"] = bool(row.get("enabled", True))
                    item["edited"] = bool(row.get("edited"))
                inventory.append(item)
            return {
                "repo_parameter_rows": rows,
                "inventory": inventory,
            }

        def _save_rows(_project_dir, rows):
            state["rows"] = [dict(row) for row in rows]
            return [dict(row) for row in state["rows"]]

        def _build_units_rows(_project_dir, saved_rows, _context):
            display_name = str((saved_rows or [{}])[0].get("displayed_parameter") or "")
            return [
                {
                    "canonical_id": "display:pulsepressuresaved",
                    "displayed_parameter": display_name,
                    "raw_names": ["PulsePressure"],
                    "raw_names_text": "PulsePressure",
                    "program_titles": ["Program A"],
                    "program_titles_text": "Program A",
                    "source_units": ["psia-second"],
                    "preferred_units": "psia-second",
                    "unit_conflict": False,
                }
            ]

        stack = ExitStack()
        stack.enter_context(
            patch("ui_next.qt_main.be.validate_test_data_project_cache_for_open", return_value=Path("dummy.sqlite3"))
        )
        stack.enter_context(
            patch("ui_next.qt_main.be.td_load_parameter_runtime_context", side_effect=_context_for_reload)
        )
        stack.enter_context(patch.object(ProjectTaskWorker, "start", _start_worker_sync))
        stack.enter_context(
            patch("ui_next.qt_main.be.save_td_repo_parameter_mappings", side_effect=_save_rows)
        )
        stack.enter_context(
            patch("ui_next.qt_main.be.td_rebuild_project_parameter_units_catalog", return_value={"groups": []})
        )
        stack.enter_context(
            patch("ui_next.qt_main.be.td_build_project_parameter_units_rows", side_effect=_build_units_rows)
        )
        self.addCleanup(stack.close)

        dialog = TDParameterNormalizationDialog(Path("."), Path("dummy.xlsx"))
        self.addCleanup(self._cleanup_dialog, dialog)
        dialog.tbl.selectRow(0)
        self._app.processEvents()
        with patch.object(dialog, "_prompt_display_name", return_value="Pulse Pressure Saved"):
            dialog._act_set_display_name()
        self._app.processEvents()
        dialog.btn_save.click()
        self._app.processEvents()
        dialog.close()
        self._app.processEvents()

        reopened = TDParameterNormalizationDialog(Path("."), Path("dummy.xlsx"))
        self.addCleanup(self._cleanup_dialog, reopened)
        self.assertEqual(str(reopened._working_rows[0].get("displayed_parameter") or ""), "Pulse Pressure Saved")
        self.assertEqual(
            str(reopened._working_rows[0].get("default_display_parameter") or ""),
            "Pulse Pressure Saved",
        )
        self.assertEqual(reopened.tbl.item(0, reopened.COL_DISPLAYED).text(), "Pulse Pressure Saved")
        reopened.tbl.selectRow(0)
        self._app.processEvents()
        with patch.object(reopened, "_prompt_display_name", return_value="Pulse Pressure Temp"):
            reopened._act_set_display_name()
        self._app.processEvents()
        with patch(
            "ui_next.qt_main.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ):
            reopened._act_reset_rows()
            self._app.processEvents()
        self.assertEqual(str(reopened._working_rows[0].get("displayed_parameter") or ""), "Pulse Pressure Saved")
