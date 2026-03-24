import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtWidgets
    from ui_next.qt_main import MainWindow
except Exception:  # pragma: no cover
    QtWidgets = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or MainWindow is None, "PySide6 is required")
class TestQtMainProductCenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _make_window(self) -> MainWindow:
        with patch.object(MainWindow, "_scan_refresh", lambda self: None), patch.object(
            MainWindow, "_start_workspace_sync", lambda self, *, auto, show_popup, heading: None
        ):
            window = MainWindow()
        return window

    def test_product_center_tab_exists_and_empty_refresh_is_safe(self) -> None:
        window = self._make_window()
        try:
            window.ed_global_repo.setText("")
            window._refresh_product_center()
            self.assertEqual(window.tabs.count(), 4)
            self.assertEqual(window.tabs.tabText(2), "Product Center")
            self.assertIn("Select a Global Repo", window.lbl_product_center_status.text())
        finally:
            window.close()

    def test_product_center_selection_populates_detail(self) -> None:
        window = self._make_window()
        product = {
            "display_name": "Pump Model",
            "asset_type": "Pump",
            "asset_specific_type": "Pump Model",
            "vendor": "Vendor A",
            "image_path": "",
            "counts": {
                "documents": 2,
                "eidp_documents": 1,
                "td_documents": 1,
                "projects": 1,
                "saved_performance_equations": 1,
            },
            "part_numbers": ["PN-001"],
            "acceptance_test_plan_numbers": ["ATP-001"],
            "serial_numbers": ["SN-001"],
            "document_types": ["EIDP", "TD"],
            "documents": [
                {
                    "rel_path": "source/doc_a.pdf",
                    "metadata_rel": "docs/doc_a.json",
                    "artifacts_rel": "debug/ocr/doc_a",
                    "display_document_type": "EIDP",
                    "serial_number": "SN-001",
                    "part_number": "PN-001",
                    "acceptance_test_plan_number": "ATP-001",
                    "program_title": "Program Alpha",
                }
            ],
            "projects": [
                {
                    "name": "TD Project",
                    "type": "Test Data Trending",
                    "project_dir": "C:/repo/projects/TD Project",
                    "workbook": "C:/repo/projects/TD Project/TD Project.xlsx",
                }
            ],
            "saved_performance_equations": [
                {
                    "project_name": "TD Project",
                    "name": "Flow Fit",
                    "summary": "Pressure vs Flow",
                    "saved_at": "2026-03-01 10:00:00",
                    "updated_at": "2026-03-02 10:00:00",
                }
            ],
        }
        try:
            window.ed_global_repo.setText("C:/repo")
            with patch("ui_next.qt_main.be.list_product_center_products", return_value=[product]):
                window._refresh_product_center()
            self.assertEqual(window.list_product_center.count(), 1)
            self.assertEqual(window.lbl_product_center_title.text(), "Pump Model")
            self.assertIn("Asset Type: Pump", window.lbl_product_center_subtitle.text())
            self.assertEqual(window.tree_product_center_docs.topLevelItemCount(), 1)
            self.assertEqual(window.tbl_product_center_projects.rowCount(), 1)
            self.assertEqual(window.tbl_product_center_equations.rowCount(), 1)
        finally:
            window.close()

    def test_product_center_show_in_files_applies_and_clears_exact_subset(self) -> None:
        window = self._make_window()
        try:
            window._files_data = [
                {"rel_path": "source/doc_a.pdf", "program_title": "Program Alpha"},
                {"rel_path": "source/doc_b.pdf", "program_title": "Program Alpha"},
            ]
            window._files_filtered = list(window._files_data)
            window._product_center_current = {
                "asset_specific_type": "Pump Model",
                "documents": [
                    {"rel_path": "source/doc_a.pdf"},
                    {"rel_path": "source/doc_a.pdf"},
                ],
            }
            with patch.object(MainWindow, "_switch_tab", lambda self, idx: None):
                window._act_product_center_show_files()
            self.assertEqual(window.tbl_files.rowCount(), 1)
            self.assertIn("Product Center", window.lbl_files_subset.text())
            window._clear_files_external_subset()
            self.assertEqual(window.tbl_files.rowCount(), 2)
        finally:
            window.close()

    def test_product_center_show_in_projects_applies_and_clears_exact_subset(self) -> None:
        window = self._make_window()
        try:
            window._projects_all = [
                {
                    "name": "TD Project",
                    "type": "Test Data Trending",
                    "project_dir": "C:/repo/projects/TD Project",
                    "project_dir_display": "projects/TD Project",
                    "workbook": "C:/repo/projects/TD Project/TD Project.xlsx",
                    "workbook_display": "projects/TD Project/TD Project.xlsx",
                },
                {
                    "name": "Other Project",
                    "type": "EIDP Trending",
                    "project_dir": "C:/repo/projects/Other Project",
                    "project_dir_display": "projects/Other Project",
                    "workbook": "C:/repo/projects/Other Project/Other Project.xlsx",
                    "workbook_display": "projects/Other Project/Other Project.xlsx",
                },
            ]
            window._apply_projects_filter()
            window._product_center_current = {
                "asset_specific_type": "Pump Model",
                "projects": [{"project_dir": "C:/repo/projects/TD Project"}],
            }
            with patch.object(MainWindow, "_switch_tab", lambda self, idx: None):
                window._act_product_center_show_projects()
            self.assertEqual(window.tbl_projects.rowCount(), 1)
            self.assertIn("Product Center", window.lbl_projects_subset.text())
            window._clear_projects_external_subset()
            self.assertEqual(window.tbl_projects.rowCount(), 2)
        finally:
            window.close()
