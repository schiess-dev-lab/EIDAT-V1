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

    class _FakePdfPages:
        saved_figures: list[object] = []

        def __init__(self, _path: str) -> None:
            self._saved: list[object] = []

        def __enter__(self):
            type(self).saved_figures = self._saved
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def savefig(self, fig: object) -> None:
            self._saved.append(fig)

    def test_load_auto_plots_reads_graph_snapshot_store_and_ignores_quickcheck_library(self) -> None:
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
                                "global_selection": {"filters": {}},
                                "track_program_serials": False,
                                "plots": [{"plot_definition": {"mode": "metrics", "stats": ["mean"], "y": ["Pressure"]}}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            saved = be.save_auto_graph_quickcheck_pack(project_dir, {"name": "Quick Pack", "plots": []})

            window = self._make_window(project_dir, workbook_path)
            try:
                names = [str(item.get("name") or "") for item in (window._auto_plots or []) if isinstance(item, dict)]
                self.assertIn("Legacy File", names)
                self.assertNotIn("Quick Pack", names)
            finally:
                window.close()
                window.deleteLater()

            self.assertEqual(str(saved.get("name") or ""), "Quick Pack")

    def test_open_selected_auto_plot_uses_graph_file_viewer_for_saved_snapshot(self) -> None:
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
                                "id": "graph-1",
                                "name": "Saved Graph",
                                "global_selection": {"filters": {}},
                                "track_program_serials": False,
                                "plots": [{"plot_definition": {"mode": "metrics", "stats": ["mean"], "y": ["Pressure"]}}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

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
                viewer_mock.assert_not_called()
                legacy_viewer_mock.assert_called_once()
                payload = legacy_viewer_mock.call_args.args[0]
                self.assertEqual(str(payload.get("name") or ""), "Saved Graph")
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

    def test_open_project_graph_item_ignores_quickcheck_library_entries(self) -> None:
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
                                "global_selection": {"filters": {}},
                                "track_program_serials": False,
                                "plots": [{"plot_definition": {"mode": "metrics", "stats": ["mean"], "y": ["Pressure"]}}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            be.save_auto_graph_quickcheck_pack(project_dir, {"name": "Quick Pack", "plots": []})

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

    def test_quickcheck_render_uses_live_plot_title_format_and_keeps_status_out_of_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")

            window = self._make_window(project_dir, workbook_path)
            try:
                captured: dict[str, object] = {}
                window._available_program_filters = ["Program Alpha"]
                window._available_serial_filter_rows = [
                    {"serial": "SN-001", "program_title": "Program Alpha"},
                ]
                pack = {"eligibility_filters": {"programs": ["Program Alpha"]}}
                plot_result = {
                    "status": "PASS",
                    "plot_name": "Pressure Plot",
                    "render_context": {
                        "plot_definition": {
                            "mode": "metrics",
                            "selector_mode": "sequence",
                            "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                            "stats": ["mean"],
                            "y": ["Pressure"],
                        }
                    },
                }

                def _fake_render(plot_definition: dict[str, object], *, filter_state=None, title_override: str = "") -> object:
                    captured["plot_definition"] = dict(plot_definition)
                    captured["filter_state"] = dict(filter_state or {})
                    captured["title_override"] = title_override
                    return object()

                with mock.patch.object(
                    window,
                    "_available_auto_plot_selection_items",
                    return_value=[
                        {
                            "id": "sequence:CondA|Program Alpha|Seq-1",
                            "mode": "sequence",
                            "run_name": "CondA",
                            "run_conditions": ["Condition A"],
                            "member_runs": ["CondA"],
                        }
                    ],
                ), mock.patch.object(window, "_render_plot_def_to_figure", side_effect=_fake_render):
                    _fig, warning_text, title_text, tab_text = window._render_auto_graph_quickcheck_result_figure(
                        pack,
                        plot_result,
                        target_serials=["SN-001"],
                    )

                self.assertEqual(warning_text, "")
                self.assertEqual(
                    title_text,
                    "Run Condition: CondA | Graph Type: Serial Number vs Pressure (mean)",
                )
                self.assertEqual(captured.get("title_override"), title_text)
                self.assertNotIn("PASS | Pressure Plot", str(captured.get("title_override") or ""))
                self.assertEqual(tab_text, "PASS | Pressure Plot")
                self.assertEqual((captured.get("filter_state") or {}).get("serials"), ["SN-001"])
            finally:
                window.close()
                window.deleteLater()

    def test_snapshot_from_quickcheck_freezes_target_serials_into_saved_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")

            window = self._make_window(project_dir, workbook_path)
            try:
                db_path = project_dir / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._plot_ready = True
                window._available_program_filters = ["Program Alpha"]
                window._available_serial_filter_rows = [
                    {"serial": "SN-001", "program_title": "Program Alpha"},
                    {"serial": "SN-002", "program_title": "Program Alpha"},
                ]

                def _fake_plot_metrics() -> None:
                    window._last_plot_def = {
                        "mode": "metrics",
                        "selector_mode": "sequence",
                        "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                        "stats": ["mean"],
                        "y": ["Pressure"],
                    }

                plot_entry = {
                    "plot_definition": {
                        "mode": "metrics",
                        "selector_mode": "sequence",
                        "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                        "stats": ["mean"],
                        "y": ["Pressure"],
                    },
                    "snapshot_filter_state": {
                        "programs": ["Program Alpha"],
                        "serials": [],
                        "control_periods": [],
                        "suppression_voltages": [],
                        "valve_voltages": [],
                    },
                    "snapshot_target_serials": ["SN-001", "SN-002"],
                }

                with mock.patch.object(window, "_selection_from_plot_def", return_value={"id": "sequence:CondA|Program Alpha|Seq-1"}), mock.patch.object(
                    window, "_set_mode", return_value=None
                ), mock.patch.object(
                    window, "_select_run_by_id", return_value=None
                ), mock.patch.object(
                    window, "_set_metric_plot_source", return_value=None
                ), mock.patch.object(
                    window, "_plot_metrics", side_effect=_fake_plot_metrics
                ):
                    window._apply_auto_graph_quickcheck_plot_to_live_gui(plot_entry)

                snapshot_file = window._current_auto_graph_snapshot_file()
                self.assertIsNotNone(snapshot_file)
                filters = (((snapshot_file or {}).get("global_selection") or {}).get("filters") or {})
                self.assertEqual(filters.get("programs"), ["Program Alpha"])
                self.assertEqual(filters.get("serials"), ["SN-001", "SN-002"])
            finally:
                window.close()
                window.deleteLater()

    def test_saved_snapshot_render_uses_frozen_serial_filter_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")

            window = self._make_window(project_dir, workbook_path)
            try:
                captured: dict[str, object] = {}
                window._available_program_filters = ["Program Alpha"]
                window._available_serial_filter_rows = [
                    {"serial": "SN-001", "program_title": "Program Alpha"},
                    {"serial": "SN-002", "program_title": "Program Alpha"},
                ]
                graph_file = {
                    "name": "Saved Graph",
                    "global_selection": {
                        "filters": {
                            "programs": ["Program Alpha"],
                            "serials": ["SN-001", "SN-002"],
                            "control_periods": [],
                            "suppression_voltages": [],
                            "valve_voltages": [],
                        }
                    },
                    "track_program_serials": False,
                    "plots": [
                        {
                            "name": "Pressure Plot",
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
                plot_entry = dict((graph_file.get("plots") or [])[0])

                def _fake_render(plot_definition: dict[str, object], *, filter_state=None, title_override: str = "") -> object:
                    captured["plot_definition"] = dict(plot_definition)
                    captured["filter_state"] = dict(filter_state or {})
                    captured["title_override"] = title_override
                    return object()

                with mock.patch.object(
                    window,
                    "_available_auto_plot_selection_items",
                    return_value=[
                        {
                            "id": "sequence:CondA|Program Alpha|Seq-1",
                            "mode": "sequence",
                            "run_name": "CondA",
                            "run_conditions": ["Condition A"],
                            "member_runs": ["CondA"],
                        }
                    ],
                ), mock.patch.object(window, "_render_plot_def_to_figure", side_effect=_fake_render):
                    _fig, warning_text = window._render_auto_graph_plot_figure(graph_file, plot_entry)

                self.assertEqual(warning_text, "")
                self.assertEqual((captured.get("filter_state") or {}).get("serials"), ["SN-001", "SN-002"])
                self.assertEqual(
                    captured.get("title_override"),
                    "Run Condition: CondA | Graph Type: Serial Number vs Pressure (mean)",
                )
            finally:
                window.close()
                window.deleteLater()

    def test_snapshot_pdf_export_reuses_snapshot_render_title_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("", encoding="utf-8")

            window = self._make_window(project_dir, workbook_path)
            try:
                db_path = project_dir / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._plot_ready = True
                window._available_program_filters = ["Program Alpha"]
                window._available_serial_filter_rows = [
                    {"serial": "SN-001", "program_title": "Program Alpha"},
                ]
                capture: dict[str, object] = {}
                pdf_path = project_dir / "snapshot.pdf"
                graph_file = {
                    "id": "graph-1",
                    "name": "Saved Graph",
                    "global_selection": {
                        "filters": {
                            "programs": ["Program Alpha"],
                            "serials": ["SN-001"],
                            "control_periods": [],
                            "suppression_voltages": [],
                            "valve_voltages": [],
                        }
                    },
                    "track_program_serials": False,
                    "plots": [
                        {
                            "name": "Pressure Plot",
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

                def _fake_render(plot_definition: dict[str, object], *, filter_state=None, title_override: str = "") -> object:
                    capture["plot_definition"] = dict(plot_definition)
                    capture["filter_state"] = dict(filter_state or {})
                    capture["title_override"] = title_override
                    return object()

                with mock.patch.object(
                    window,
                    "_available_auto_plot_selection_items",
                    return_value=[
                        {
                            "id": "sequence:CondA|Program Alpha|Seq-1",
                            "mode": "sequence",
                            "run_name": "CondA",
                            "run_conditions": ["Condition A"],
                            "member_runs": ["CondA"],
                        }
                    ],
                ), mock.patch.object(
                    window,
                    "_render_plot_def_to_figure",
                    side_effect=_fake_render,
                ), mock.patch.dict(
                    sys.modules,
                    {"matplotlib.backends.backend_pdf": mock.Mock(PdfPages=self._FakePdfPages)},
                ):
                    saved = window._save_auto_graph_file_pdf_to_path(graph_file, pdf_path, persist_path=False, show_result=False)

                self.assertIsNotNone(saved)
                self.assertEqual(capture.get("title_override"), "Run Condition: CondA | Graph Type: Serial Number vs Pressure (mean)")
                self.assertEqual((capture.get("filter_state") or {}).get("serials"), ["SN-001"])
                self.assertEqual(len(self._FakePdfPages.saved_figures), 1)
            finally:
                window.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main()
