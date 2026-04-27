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

    def test_open_project_graph_item_uses_auto_graph_file_viewer_for_legacy_graph_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")
            (project_dir / "auto_plots_test_data.json").write_text(
                json.dumps(
                    {
                        "version": 4,
                        "graph_files": [
                            {
                                "id": "legacy",
                                "name": "Legacy File",
                                "global_selection": {
                                    "run_scope": "sequence",
                                    "selected_selection_ids": [],
                                    "filters": {
                                        "programs": [],
                                        "serials": [],
                                        "control_periods": [],
                                        "suppression_voltages": [],
                                        "valve_voltages": [],
                                    },
                                },
                                "plots": [
                                    {
                                        "name": "Pressure Mean",
                                        "plot_definition": {
                                            "mode": "metrics",
                                            "selector_mode": "sequence",
                                            "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                                            "stats": ["mean"],
                                            "y": ["Pressure"],
                                            "metric_plot_source": "all_sequences",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            graph_item = next(
                item
                for item in be.list_project_graph_items(project_dir, be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING)
                if str(item.get("kind") or "") == "Auto-Graph File"
            )

            window = self._make_window(project_dir, workbook_path)
            try:
                with mock.patch.object(window, "_open_auto_graph_file_viewer") as legacy_viewer_mock, mock.patch.object(
                    window,
                    "_open_auto_graph_quickcheck_viewer",
                ) as quickcheck_viewer_mock:
                    window._open_project_graph_item(graph_item)
                legacy_viewer_mock.assert_called_once()
                quickcheck_viewer_mock.assert_not_called()
                payload = legacy_viewer_mock.call_args.args[0]
                self.assertEqual(str(payload.get("name") or ""), "Legacy File")
                self.assertEqual(
                    str(((payload.get("plots") or [])[0] or {}).get("plot_definition", {}).get("metric_plot_source") or ""),
                    "all_sequences",
                )
            finally:
                window.close()
                window.deleteLater()

    def test_upsert_auto_graph_file_entry_refreshes_linked_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")
            pdf_path = project_dir / "linked.pdf"

            window = self._make_window(project_dir, workbook_path)
            try:
                refreshed_paths: list[str] = []
                graph_file = {
                    "id": "graph-1",
                    "name": "Linked Graph",
                    "pdf_export_path": str(pdf_path),
                    "global_selection": {
                        "run_scope": "sequence",
                        "selected_selection_ids": [],
                        "filters": {
                            "programs": [],
                            "serials": [],
                            "control_periods": [],
                            "suppression_voltages": [],
                            "valve_voltages": [],
                        },
                    },
                    "track_program_serials": False,
                    "plots": [
                        {
                            "name": "Pressure Mean",
                            "plot_definition": {
                                "mode": "metrics",
                                "selector_mode": "sequence",
                                "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                                "stats": ["mean"],
                                "y": ["Pressure"],
                            },
                        }
                    ],
                }

                with mock.patch.object(
                    window,
                    "_save_auto_graph_file_pdf_to_path",
                    side_effect=lambda entry, path, **kwargs: (
                        refreshed_paths.append(str(Path(path).expanduser())),
                        dict(entry),
                    )[1],
                ):
                    saved = window._upsert_auto_graph_file_entry(graph_file)

                self.assertIsNotNone(saved)
                self.assertEqual(refreshed_paths, [str(pdf_path)])
                payload = json.loads((project_dir / "auto_plots_test_data.json").read_text(encoding="utf-8"))
                self.assertEqual(str((payload.get("graph_files") or [])[0].get("pdf_export_path") or ""), str(pdf_path))
            finally:
                window.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main()
