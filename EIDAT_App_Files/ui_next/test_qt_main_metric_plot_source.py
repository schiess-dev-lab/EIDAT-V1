import os
import shutil
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
    from PySide6 import QtWidgets
    from ui_next.qt_main import TestDataTrendDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtWidgets = None  # type: ignore[assignment]
    TestDataTrendDialog = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or TestDataTrendDialog is None, "PySide6 is required")
class TestQtMainMetricPlotSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _make_window(self) -> TestDataTrendDialog:
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        workbook_path = project_dir / "project.xlsx"
        workbook_path.write_text("", encoding="utf-8")
        with patch.object(TestDataTrendDialog, "_load_cache", lambda self, *, rebuild: None), patch.object(
            TestDataTrendDialog, "_load_auto_plots", lambda self: None
        ):
            window = TestDataTrendDialog(project_dir, workbook_path)
        window._test_tmpdir = tmpdir  # type: ignore[attr-defined]
        return window

    @staticmethod
    def _ensure_selected_item(list_widget: QtWidgets.QListWidget, text: str) -> None:
        matches = [
            list_widget.item(i)
            for i in range(list_widget.count())
            if str(list_widget.item(i).text() or "").strip().lower() == str(text or "").strip().lower()
        ]
        if not matches:
            list_widget.addItem(text)
            matches = [
                list_widget.item(i)
                for i in range(list_widget.count())
                if str(list_widget.item(i).text() or "").strip().lower() == str(text or "").strip().lower()
            ]
        list_widget.clearSelection()
        if matches:
            matches[0].setSelected(True)

    def test_plot_metrics_uses_selected_metric_source_and_persists_it(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                class _DummyLine:
                    def get_color(self) -> str:
                        return "#1f77b4"

                class _DummyAxes:
                    def clear(self) -> None:
                        return None

                    def set_title(self, _value: str) -> None:
                        return None

                    def set_xlabel(self, _value: str) -> None:
                        return None

                    def set_ylabel(self, _value: str) -> None:
                        return None

                    def plot(self, *_args, **_kwargs):
                        return [_DummyLine()]

                    def set_xlim(self, *_args) -> None:
                        return None

                    def set_xticks(self, *_args) -> None:
                        return None

                    def set_xticklabels(self, *_args, **_kwargs) -> None:
                        return None

                    def grid(self, *_args, **_kwargs) -> None:
                        return None

                class _DummyFigure:
                    def tight_layout(self) -> None:
                        return None

                class _DummyCanvas:
                    def draw(self) -> None:
                        return None

                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._plot_ready = True
                window._figure = _DummyFigure()
                window._axes = _DummyAxes()
                window._canvas = _DummyCanvas()
                self._ensure_selected_item(window.list_stats, "Mean")
                self._ensure_selected_item(window.list_y_metrics, "Pressure")
                selection = {
                    "mode": "sequence",
                    "id": "sequence:CondA|Program Alpha|Seq-1",
                    "run_name": "CondA",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq-1",
                    "member_runs": ["CondA"],
                }
                metric_sources: list[str] = []

                def _fake_loader(
                    run_name: str,
                    column_name: str,
                    stat: str,
                    *,
                    selection: dict | None = None,
                    control_period_filter: object = None,
                    run_type_filter: object = None,
                    metric_source: object = None,
                ) -> list[dict]:
                    del run_name, column_name, stat, selection, control_period_filter, run_type_filter
                    metric_sources.append(str(metric_source or ""))
                    return [
                        {
                            "serial": "SN-001",
                            "value_num": 1.0,
                            "observation_id": "seq-1",
                            "program_title": "Program Alpha",
                            "source_run_name": "Seq-1",
                        }
                    ]

                with patch.object(window, "_current_run_selection", return_value=selection), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_active_serial_rows", return_value=[{"serial": "SN-001"}]
                ), patch.object(
                    window, "_metric_bounds_for_run", return_value={}
                ), patch.object(
                    window, "_compose_run_title", return_value="Metric Plot"
                ), patch.object(
                    window, "_apply_metric_program_segments", return_value=None
                ), patch.object(
                    window, "_apply_interactive_legend_policy", return_value=[]
                ), patch.object(
                    window, "_apply_plot_view_bands_to_axes", return_value=None
                ), patch.object(
                    window, "_refresh_plot_note", return_value=None
                ), patch.object(
                    window, "_capture_main_plot_base_view", return_value=None
                ), patch.object(
                    window, "_update_perf_primary_equation_banner", return_value=None
                ), patch.object(
                    window, "_refresh_stats_preview", return_value=None
                ), patch.object(
                    window, "_load_metric_series_for_selection", side_effect=_fake_loader
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._set_metric_plot_source("all_sequences")
                    window._plot_metrics()

                info_mock.assert_not_called()
                self.assertTrue(metric_sources)
                self.assertTrue(all(source == "all_sequences" for source in metric_sources))
                self.assertEqual(window._last_plot_def["metric_plot_source"], "all_sequences")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_open_selected_auto_plot_restores_metric_source(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._plot_ready = True
                self._ensure_selected_item(window.list_stats, "Mean")
                self._ensure_selected_item(window.list_y_metrics, "Pressure")
                plot_sources: list[str] = []
                plot_def = {
                    "mode": "metrics",
                    "selector_mode": "sequence",
                    "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                    "stats": ["mean"],
                    "y": ["Pressure"],
                    "plot_bounds": False,
                    "metric_plot_source": "all_sequences",
                }

                with patch.object(
                    window,
                    "_selection_from_plot_def",
                    return_value={"mode": "sequence", "id": "sequence:CondA|Program Alpha|Seq-1"},
                ), patch.object(
                    window, "_select_run_by_id", return_value=None
                ), patch.object(
                    window, "_set_mode", side_effect=lambda mode: setattr(window, "_mode", str(mode))
                ), patch.object(
                    window, "_plot_metrics", side_effect=lambda: plot_sources.append(window._selected_metric_plot_source())
                ):
                    window._set_metric_plot_source("aggregate")
                    window._open_selected_auto_plot(plot_def=plot_def)

                self.assertEqual(plot_sources, ["all_sequences"])
                self.assertEqual(window._selected_metric_plot_source(), "all_sequences")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
