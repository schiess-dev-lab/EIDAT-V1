import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtCore, QtWidgets
    from ui_next import backend as be  # type: ignore
    from ui_next.qt_main import MainWindow, ProjectDocumentManagerDialog  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    be = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]
    ProjectDocumentManagerDialog = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or MainWindow is None, "PySide6 is required")
class TestProjectDocumentManagerUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_project_report_and_graph_buttons_follow_selection_and_busy_state(self) -> None:
        class _Harness:
            pass

        harness = _Harness()
        harness.btn_project_implementation = QtWidgets.QPushButton()
        harness.btn_project_view_reports = QtWidgets.QPushButton()
        harness.btn_project_view_graphs = QtWidgets.QPushButton()
        harness._project_worker = None
        harness._selected_project_record = lambda: {
            "name": "Proj",
            "type": be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
            "folder": "C:/tmp/repo/projects/Proj",
            "workbook": "C:/tmp/repo/projects/Proj/Proj.xlsx",
        }

        MainWindow._update_project_actions(harness)
        self.assertTrue(harness.btn_project_view_reports.isEnabled())
        self.assertTrue(harness.btn_project_view_graphs.isEnabled())

        class _Worker:
            def isRunning(self) -> bool:
                return True

        harness._project_worker = _Worker()
        MainWindow._update_project_actions(harness)
        self.assertFalse(harness.btn_project_view_reports.isEnabled())
        self.assertFalse(harness.btn_project_view_graphs.isEnabled())

    def test_view_reports_requires_global_repo(self) -> None:
        class _Harness:
            pass

        harness = _Harness()
        harness.ed_global_repo = QtWidgets.QLineEdit("")

        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
            MainWindow._act_open_project_document_manager(harness, "reports")

        warning_mock.assert_called_once()
        self.assertEqual(warning_mock.call_args.args[1], "View Reports")
        self.assertIn("Global Repo", warning_mock.call_args.args[2])

    def test_manager_update_action_delegates_to_project_update_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "Proj.xlsx"
            workbook.write_text("placeholder", encoding="utf-8")
            captured: dict[str, object] = {}

            dlg = ProjectDocumentManagerDialog(
                mode="graphs",
                global_repo=root,
                record={
                    "name": "Proj",
                    "type": be.EIDAT_PROJECT_TYPE_TRENDING,
                    "folder": str(root),
                    "workbook": str(workbook),
                },
                request_update=lambda refresh: captured.update({"refresh": refresh}),
                open_graph_library=lambda item=None: None,
            )
            try:
                with mock.patch.object(dlg, "refresh_items") as refresh_mock:
                    dlg._act_update()
                    refresh = captured.get("refresh")
                    self.assertTrue(callable(refresh))
                    refresh()  # type: ignore[operator]
                    refresh_mock.assert_called_once()
            finally:
                dlg.close()
                dlg.deleteLater()

    def test_graph_definition_open_passes_selected_item_to_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "Proj.xlsx"
            workbook.write_text("placeholder", encoding="utf-8")
            captured: dict[str, object] = {}
            item = {
                "id": "graph_definition:legacy_graph_file:id:graph-1",
                "type": "graph_definition",
                "kind": "Auto-Graph File",
                "name": "Saved Graph",
                "graph_key": "legacy_graph_file:id:graph-1",
                "path": str(root / "auto_plots_test_data.json"),
            }

            dlg = ProjectDocumentManagerDialog(
                mode="graphs",
                global_repo=root,
                record={
                    "name": "Proj",
                    "type": be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
                    "folder": str(root),
                    "workbook": str(workbook),
                },
                request_update=lambda refresh: None,
                open_graph_library=lambda selected_item=None: captured.update({"item": dict(selected_item or {})}),
            )
            try:
                dlg.tbl.setRowCount(1)
                cell = QtWidgets.QTableWidgetItem(str(item.get("name") or ""))
                cell.setData(QtCore.Qt.ItemDataRole.UserRole, dict(item))
                dlg.tbl.setItem(0, 0, cell)
                for col in range(1, dlg.tbl.columnCount()):
                    dlg.tbl.setItem(0, col, QtWidgets.QTableWidgetItem(""))
                dlg.tbl.selectRow(0)

                dlg._act_open()

                self.assertEqual((captured.get("item") or {}).get("graph_key"), "legacy_graph_file:id:graph-1")
            finally:
                dlg.close()
                dlg.deleteLater()


if __name__ == "__main__":
    unittest.main()
