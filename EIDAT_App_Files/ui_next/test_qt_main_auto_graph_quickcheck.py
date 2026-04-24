import json
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
    from PySide6 import QtWidgets
    from ui_next import backend as be  # type: ignore
    from ui_next.qt_main import TestDataTrendDialog  # type: ignore
except Exception:  # pragma: no cover
    QtWidgets = None  # type: ignore[assignment]
    be = None  # type: ignore[assignment]
    TestDataTrendDialog = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or TestDataTrendDialog is None or be is None, "PySide6 is required")
class TestQtMainAutoGraphQuickcheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _make_window(self, project_dir: Path, workbook_path: Path) -> TestDataTrendDialog:
        with mock.patch.object(TestDataTrendDialog, "_load_cache", lambda self, *, rebuild: None):
            window = TestDataTrendDialog(project_dir, workbook_path)
        return window

    def test_load_auto_plots_reads_quickcheck_library_not_legacy_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")
            (project_dir / "auto_plots_test_data.json").write_text(
                json.dumps({"version": 3, "graph_files": [{"id": "legacy", "name": "Legacy File", "plots": []}]}),
                encoding="utf-8",
            )
            saved = be.save_auto_graph_quickcheck_pack(project_dir, {"name": "Quick Pack", "plots": []})

            window = self._make_window(project_dir, workbook_path)
            try:
                names = [str(item.get("name") or "") for item in (window._auto_plots or []) if isinstance(item, dict)]
                self.assertIn("Quick Pack", names)
                self.assertNotIn("Legacy File", names)
            finally:
                window.close()
                window.deleteLater()

            self.assertEqual(str(saved.get("name") or ""), "Quick Pack")

    def test_open_selected_auto_plot_uses_quickcheck_viewer_for_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")
            be.save_auto_graph_quickcheck_pack(project_dir, {"name": "Quick Pack", "plots": []})

            window = self._make_window(project_dir, workbook_path)
            try:
                window._plot_ready = True
                window._db_path = project_dir / "cache.sqlite3"
                window._db_path.write_text("", encoding="utf-8")
                pack = next(item for item in (window._auto_plots or []) if isinstance(item, dict))
                with mock.patch.object(window, "_open_auto_graph_quickcheck_viewer") as viewer_mock, mock.patch.object(
                    window,
                    "_open_auto_graph_file_viewer",
                ) as legacy_viewer_mock:
                    window._open_selected_auto_plot(pack)
                viewer_mock.assert_called_once()
                legacy_viewer_mock.assert_not_called()
                payload = viewer_mock.call_args.args[0]
                self.assertEqual(str(payload.get("name") or ""), "Quick Pack")
            finally:
                window.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main()
