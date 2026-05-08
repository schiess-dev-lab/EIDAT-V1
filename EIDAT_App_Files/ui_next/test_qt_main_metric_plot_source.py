import json
import os
import sqlite3
import shutil
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtCore, QtWidgets
    from ui_next import backend
    from ui_next.qt_main import TestDataTrendDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    backend = None  # type: ignore[assignment]
    TestDataTrendDialog = None  # type: ignore[assignment]


class _DummyLine:
    def __init__(self) -> None:
        self._visible = True
        self._gid = ""

    def get_color(self) -> str:
        return "#1f77b4"

    def set_gid(self, value: object) -> None:
        self._gid = str(value or "")

    def get_gid(self) -> str:
        return self._gid

    def get_visible(self) -> bool:
        return self._visible

    def set_visible(self, value: object) -> None:
        self._visible = bool(value)


class _DummyAxes:
    def __init__(self) -> None:
        self.plot_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.lines: list[_DummyLine] = []
        self._xlim = (0.0, 1.0)
        self._ylim = (0.0, 1.0)

    def clear(self) -> None:
        self.lines = []
        return None

    def set_title(self, _value: str) -> None:
        return None

    def set_xlabel(self, _value: str) -> None:
        return None

    def set_ylabel(self, _value: str) -> None:
        return None

    def plot(self, *args: object, **kwargs: object):
        self.plot_calls.append((args, dict(kwargs)))
        line = _DummyLine()
        self.lines.append(line)
        return [line]

    def set_xlim(self, *_args) -> None:
        if len(_args) >= 2:
            self._xlim = (float(_args[0]), float(_args[1]))
        return None

    def set_xticks(self, *_args) -> None:
        return None

    def set_xticklabels(self, *_args, **_kwargs) -> None:
        return None

    def grid(self, *_args, **_kwargs) -> None:
        return None

    def get_xlim(self) -> tuple[float, float]:
        return self._xlim

    def get_ylim(self) -> tuple[float, float]:
        return self._ylim


class _DummyFigure:
    def tight_layout(self) -> None:
        return None


class _DummyCanvas:
    def draw(self) -> None:
        return None

    def draw_idle(self) -> None:
        return None


class _DummyComboBox:
    def findData(self, _value: object) -> int:
        return -1

    def setCurrentIndex(self, _index: int) -> None:
        return None

    def currentData(self) -> str:
        return "sequence"

    def currentText(self) -> str:
        return "sequence"


class _LegendStub:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _LegendArtist:
    def __init__(
        self,
        *,
        label: str,
        linewidth: float = 1.0,
        alpha: float | None = None,
        color: str = "#1f77b4",
        markersize: float = 6.0,
        markeredgewidth: float = 1.0,
        zorder: float = 2.0,
    ) -> None:
        self._label = str(label)
        self._linewidth = float(linewidth)
        self._alpha = alpha
        self._color = str(color)
        self._markersize = float(markersize)
        self._markeredgewidth = float(markeredgewidth)
        self._zorder = float(zorder)

    def get_label(self) -> str:
        return self._label

    def get_color(self) -> str:
        return self._color

    def get_alpha(self) -> float | None:
        return self._alpha

    def set_alpha(self, value: float | None) -> None:
        self._alpha = value

    def get_zorder(self) -> float:
        return self._zorder

    def set_zorder(self, value: float) -> None:
        self._zorder = float(value)

    def get_linewidth(self) -> float:
        return self._linewidth

    def set_linewidth(self, value: float) -> None:
        self._linewidth = float(value)

    def get_markersize(self) -> float:
        return self._markersize

    def set_markersize(self, value: float) -> None:
        self._markersize = float(value)

    def get_markeredgewidth(self) -> float:
        return self._markeredgewidth

    def set_markeredgewidth(self, value: float) -> None:
        self._markeredgewidth = float(value)


class _LegendAxes:
    def __init__(self) -> None:
        self.lines: list[_LegendArtist] = []
        self.collections: list[object] = []
        self.patches: list[object] = []
        self.containers: list[object] = []
        self.artists: list[object] = []
        self._legend: _LegendStub | None = None

    def plot(self, *_args: object, **kwargs: object) -> list[_LegendArtist]:
        artist = _LegendArtist(
            label=str(kwargs.get("label") or ""),
            linewidth=float(kwargs.get("linewidth") or 1.0),
            alpha=(float(kwargs["alpha"]) if kwargs.get("alpha") is not None else None),
            color=str(kwargs.get("color") or "#1f77b4"),
            markersize=float(kwargs.get("markersize") or 6.0),
        )
        self.lines.append(artist)
        return [artist]

    def get_legend_handles_labels(self) -> tuple[list[_LegendArtist], list[str]]:
        return list(self.lines), [artist.get_label() for artist in self.lines]

    def get_legend(self) -> _LegendStub | None:
        return self._legend

    def legend(self, **_kwargs: object) -> _LegendStub:
        self._legend = _LegendStub()
        return self._legend

    def grid(self, *_args: object, **_kwargs: object) -> None:
        return None


class _ResetAxes:
    name = "rectilinear"

    def __init__(
        self,
        *,
        xlim: tuple[float, float] = (2.0, 4.0),
        ylim: tuple[float, float] = (3.0, 5.0),
        ignore_limit_updates: bool = False,
    ) -> None:
        self._xlim = tuple(float(v) for v in xlim)
        self._ylim = tuple(float(v) for v in ylim)
        self.ignore_limit_updates = bool(ignore_limit_updates)
        self.autoscale_enabled = False
        self.relim_calls = 0
        self.autoscale_view_calls = 0

    def get_xlim(self) -> tuple[float, float]:
        return self._xlim

    def get_ylim(self) -> tuple[float, float]:
        return self._ylim

    def set_xlim(self, lo: float, hi: float) -> None:
        if not self.ignore_limit_updates:
            self._xlim = (float(lo), float(hi))

    def set_ylim(self, lo: float, hi: float) -> None:
        if not self.ignore_limit_updates:
            self._ylim = (float(lo), float(hi))

    def set_autoscale_on(self, enabled: bool) -> None:
        self.autoscale_enabled = bool(enabled)

    def relim(self) -> None:
        self.relim_calls += 1

    def autoscale_view(self) -> None:
        self.autoscale_view_calls += 1


class _ResetCanvas:
    def __init__(self) -> None:
        self.draw_idle_calls = 0

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1


class _ResetButton:
    def __init__(self) -> None:
        self.checked: bool | None = None

    def setChecked(self, value: bool) -> None:
        self.checked = bool(value)


class _ResetRect:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _ResetHarness:
    _axis_bounds_valid = staticmethod(TestDataTrendDialog._axis_bounds_valid)
    _axis_bounds_match = staticmethod(TestDataTrendDialog._axis_bounds_match)

    def __init__(self, axes: _ResetAxes) -> None:
        self._axes = axes
        self._canvas = _ResetCanvas()
        self._plot_base_xlim = (0.0, 10.0)
        self._plot_base_ylim = (0.0, 20.0)
        self._zone_zoom_press_xy = (1.0, 1.0)
        self._zone_zoom_rect = _ResetRect()
        self.btn_zone_zoom = _ResetButton()
        self._last_plot_def = {"mode": "metrics", "selection_id": "sequence:test"}
        self.restore_calls = 0

    def _restore_main_plot_from_last_definition(self) -> bool:
        self.restore_calls += 1
        self._axes._xlim = tuple(float(v) for v in self._plot_base_xlim)
        self._axes._ylim = tuple(float(v) for v in self._plot_base_ylim)
        return True


def _create_plot_filter_db(tmpdir: str) -> Path:
    db_path = Path(tmpdir) / "plot_filter_cache.sqlite3"
    with closing(sqlite3.connect(str(db_path))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.execute(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            ("CondA", "time", "Condition A", "PM", 10.0, 0.5),
        )
        conn.execute(
            "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
            ("CondA", "Pressure", "psi", "y"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_condition_observations(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("agg-1", "SN-001", "CondA", "Program Alpha", "Aggregate", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_metrics_calc(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns,
                program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("agg-1", "SN-001", "CondA", "Pressure", "mean", 1.5, 1, 1, "Program Alpha", "Aggregate"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_condition_observations_sequences(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("seq-1", "SN-001", "CondA", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_metrics_calc_sequences(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns,
                program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("seq-1", "SN-001", "CondA", "Pressure", "mean", 1.0, 1, 1, "Program Alpha", "Seq-1"),
        )
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {backend.TD_PLOTTER_CURVE_CATALOG_TABLE}(
                run_name, parameter_name, units, x_axis_kind, display_name, source_kind, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("CondA", "Pressure", "psi", "time", "Pressure", "raw_cache", 1),
        )
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {backend.TD_PLOTTER_OBSERVATIONS_TABLE}(
                observation_id, run_name, serial, program_title, source_run_name, run_type, pulse_width,
                control_period, suppression_voltage, valve_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("seq-1", "CondA", "SN-001", "Program Alpha", "Seq-1", "PM", 0.5, 10.0, 5.0, 28.0, 1, 1),
        )
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {backend.TD_PLOTTER_CURVES_TABLE}(
                run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points,
                source_mtime_ns, computed_epoch_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "CondA",
                "Pressure",
                "time",
                "seq-1",
                "SN-001",
                json.dumps([0.0, 1.0, 2.0]),
                json.dumps([1.0, 1.1, 1.2]),
                3,
                1,
                1,
                "Program Alpha",
                "Seq-1",
            ),
        )
        conn.commit()
    return db_path


@unittest.skipIf(QtWidgets is None or TestDataTrendDialog is None or backend is None, "PySide6 is required")
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

    @staticmethod
    def _ensure_selected_items(list_widget: QtWidgets.QListWidget, texts: list[str]) -> None:
        wanted = {str(text or "").strip().lower() for text in texts if str(text or "").strip()}
        list_widget.clearSelection()
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item is not None and str(item.text() or "").strip().lower() in wanted:
                item.setSelected(True)

    @staticmethod
    def _prepare_plot_window(window: TestDataTrendDialog, db_path: Path, *, mode: str) -> _DummyAxes:
        axes = _DummyAxes()
        window._db_path = db_path
        window._plot_ready = True
        window._figure = _DummyFigure()
        window._axes = axes
        window._canvas = _DummyCanvas()
        window._mode = str(mode)
        window._highlight_sns = []
        window._highlight_sn = ""
        window._available_program_filters = ["Program Alpha"]
        window._checked_program_filters = ["Program Alpha"]
        window._available_serial_filter_rows = [{"serial": "SN-001", "program_title": "Program Alpha"}]
        window._checked_serial_filters = ["SN-001"]
        window._available_control_period_filters = []
        window._checked_control_period_filters = []
        window._available_suppression_voltage_filters = []
        window._checked_suppression_voltage_filters = []
        window._available_valve_voltage_filters = ["28"]
        window._checked_valve_voltage_filters = ["28"]
        window._serial_source_rows = [{"serial": "SN-001", "program_title": "Program Alpha"}]
        window._serial_source_by_serial = {"SN-001": {"serial": "SN-001", "program_title": "Program Alpha"}}
        window._global_filter_rows = [
            {
                "serial": "SN-001",
                "program_title": "Program Alpha",
                "source_run_name": "Seq-1",
                "control_period": 10.0,
                "suppression_voltage": 5.0,
                "valve_voltage": 28.0,
            }
        ]
        return axes

    @staticmethod
    def _make_real_legend_axes():
        return object(), _LegendAxes()

    def test_reset_main_plot_zoom_restores_saved_inline_limits(self) -> None:
        harness = _ResetHarness(_ResetAxes())

        TestDataTrendDialog._reset_main_plot_zoom(harness)  # type: ignore[arg-type]

        self.assertEqual(harness._axes.get_xlim(), harness._plot_base_xlim)
        self.assertEqual(harness._axes.get_ylim(), harness._plot_base_ylim)
        self.assertEqual(harness.restore_calls, 0)
        self.assertEqual(harness._canvas.draw_idle_calls, 1)
        self.assertEqual(harness.btn_zone_zoom.checked, False)
        self.assertTrue(harness._zone_zoom_rect is None)
        self.assertTrue(harness._zone_zoom_press_xy is None)

    def test_reset_main_plot_zoom_replays_plot_when_inline_limits_do_not_restore(self) -> None:
        harness = _ResetHarness(_ResetAxes(ignore_limit_updates=True))

        TestDataTrendDialog._reset_main_plot_zoom(harness)  # type: ignore[arg-type]

        self.assertEqual(harness.restore_calls, 1)
        self.assertEqual(harness._axes.get_xlim(), harness._plot_base_xlim)
        self.assertEqual(harness._axes.get_ylim(), harness._plot_base_ylim)
        self.assertEqual(harness.btn_zone_zoom.checked, False)
        self.assertTrue(harness._zone_zoom_rect is None)
        self.assertTrue(harness._zone_zoom_press_xy is None)

    def test_plot_metrics_uses_selected_metric_source_and_persists_it(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
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
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                window._db_path = db_path
                window._plot_ready = True
                opened_graph_files: list[dict[str, object]] = []
                graph_file = {
                    "name": "Metric Graph",
                    "global_selection": {
                        "run_scope": "sequence",
                        "selection_ids": [],
                        "selection_labels": [],
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
                                "plot_bounds": False,
                                "metric_plot_source": "all_sequences",
                            },
                        }
                    ],
                }

                with patch.object(
                    window,
                    "_normalize_auto_plot_global_selection",
                    side_effect=lambda selection, default_to_current=True: dict(selection or {}),
                ), patch.object(
                    window,
                    "_open_auto_graph_file_viewer",
                    side_effect=lambda normalized: opened_graph_files.append(dict(normalized or {})),
                ), patch.object(
                    window, "_refresh_auto_plots_list", return_value=None
                ):
                    window._open_selected_auto_plot(plot_def=graph_file)

                self.assertEqual(len(opened_graph_files), 1)
                opened_plots = opened_graph_files[0].get("plots") or []
                self.assertEqual(len(opened_plots), 1)
                plot_definition = dict((opened_plots[0] or {}).get("plot_definition") or {})
                self.assertEqual(plot_definition.get("metric_plot_source"), "all_sequences")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_plot_metrics_keeps_visible_rows_when_live_valve_filter_is_active(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = _create_plot_filter_db(tmpdir)
                axes = self._prepare_plot_window(window, db_path, mode="metrics")
                self._ensure_selected_item(window.list_stats, "Mean")
                self._ensure_selected_item(window.list_y_metrics, "Pressure")
                selection = {
                    "mode": "sequence",
                    "id": "sequence:CondA|Program Alpha|Seq-1",
                    "run_name": "CondA",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq-1",
                    "member_runs": ["CondA"],
                    "member_sequences": ["Seq-1"],
                }

                with patch.object(window, "_current_run_selection", return_value=selection), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
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
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._set_metric_plot_source("aggregate")
                    window._plot_metrics()

                info_mock.assert_not_called()
                self.assertEqual(window._last_plot_def["mode"], "metrics")
                self.assertTrue(axes.plot_calls)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_plot_life_metrics_uses_selected_stats_and_persists_them(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                axes = self._prepare_plot_window(window, db_path, mode="life_metrics")
                selection = {
                    "mode": "condition",
                    "id": "condition:CondA",
                    "run_name": "CondA",
                    "display_text": "Condition A",
                    "run_condition": "Condition A",
                    "member_runs": ["CondA"],
                    "member_sequences": ["Seq-1"],
                }
                requested_stats: list[str] = []
                window.cb_life_y_param.clear()
                window.cb_life_y_param.addItem("LifeMetric", "LifeMetric")
                window.cb_life_axis.clear()
                window.cb_life_axis.addItem("Sequence", "sequence_index")
                self._ensure_selected_items(window.list_life_stats, ["mean", "max"])

                def _fake_loader(
                    run_names: list[str],
                    parameter_value: object,
                    life_axis: str,
                    *,
                    stat: object = "mean",
                    serials: list[str] | None = None,
                    filter_state: object = None,
                ) -> list[dict]:
                    del run_names, parameter_value, life_axis, serials, filter_state
                    requested_stats.append(str(stat))
                    return [
                        {
                            "serial": "SN-001",
                            "observation_id": f"obs-{stat}",
                            "sequence_index": 1,
                            "condition_key": "CondA",
                            "x_value": 1.0,
                            "y_value": 1.0 if str(stat) == "mean" else 2.0,
                            "units": "psi",
                            "program_title": "Program Alpha",
                        }
                    ]

                with patch.object(window, "_current_run_selector_mode", return_value="condition"), patch.object(
                    window, "_current_run_selection", return_value=selection
                ), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_active_serials", return_value=["SN-001"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_compose_run_title", return_value="Life Plot"
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
                    window, "_load_life_metric_series_for_selection", side_effect=_fake_loader
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._plot_life_metrics()

                info_mock.assert_not_called()
                self.assertEqual(requested_stats, ["mean", "max"])
                self.assertEqual(window._last_plot_def.get("stats"), ["mean", "max"])
                self.assertEqual(len(axes.plot_calls), 2)
                labels = [str(kwargs.get("label") or "") for _, kwargs in axes.plot_calls]
                self.assertIn("SN-001 | Program Alpha (mean)", labels)
                self.assertIn("SN-001 | Program Alpha (max)", labels)
                snapshot = window._last_life_metrics_excel_snapshot
                self.assertIsInstance(snapshot, dict)
                self.assertEqual(snapshot.get("plot_metadata", {}).get("plot_type"), "life_axis")
                self.assertEqual(snapshot.get("plot_metadata", {}).get("stats"), ["mean", "max"])
                self.assertEqual(len(snapshot.get("traces") or []), 2)
                self.assertEqual(
                    [str(trace.get("stat") or "") for trace in (snapshot.get("traces") or [])],
                    ["mean", "max"],
                )
                self.assertEqual(
                    [str(row.get("observation_id") or "") for row in (snapshot.get("rows") or [])],
                    ["obs-mean", "obs-max"],
                )
                self.assertTrue(window.btn_export_life_metrics_excel.isEnabled())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_plot_life_metrics_allows_multiple_y_parameters_over_cumulative_impulse(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                axes = self._prepare_plot_window(window, db_path, mode="life_metrics")
                selection = {
                    "mode": "condition",
                    "id": "condition:CondA",
                    "run_name": "CondA",
                    "display_text": "Condition A",
                    "run_condition": "Condition A",
                    "member_runs": ["CondA"],
                    "member_sequences": ["Seq-1"],
                }
                requested_params: list[str] = []
                requested_axes: list[str] = []
                window.cb_life_y_param.clear()
                window.list_life_y_params.clear()
                for name in ("feed pressure", "thrust"):
                    window.cb_life_y_param.addItem(name, name)
                    window.list_life_y_params.addItem(name)
                window.cb_life_axis.clear()
                window.cb_life_axis.addItem("Cumulative Impulse", "cumulative_impulse")
                self._ensure_selected_items(window.list_life_stats, ["mean"])
                window._set_life_y_parameter_selection(["feed pressure", "thrust"])

                def _fake_loader(
                    run_names: list[str],
                    parameter_value: object,
                    life_axis: str,
                    *,
                    stat: object = "mean",
                    serials: list[str] | None = None,
                    filter_state: object = None,
                ) -> list[dict]:
                    del run_names, stat, serials, filter_state
                    param = str(parameter_value)
                    requested_params.append(param)
                    requested_axes.append(str(life_axis))
                    return [
                        {
                            "serial": "SN-001",
                            "observation_id": f"obs-{param}",
                            "sequence_index": 1,
                            "condition_key": "CondA",
                            "x_value": 1.0,
                            "y_value": 1.0 if param == "feed pressure" else 2.0,
                            "units": "psi",
                            "program_title": "Program Alpha",
                        }
                    ]

                with patch.object(window, "_current_run_selector_mode", return_value="condition"), patch.object(
                    window, "_current_run_selection", return_value=selection
                ), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_active_serials", return_value=["SN-001"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_compose_run_title", return_value="Life Plot"
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
                    window, "_load_life_metric_series_for_selection", side_effect=_fake_loader
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._plot_life_metrics()

                info_mock.assert_not_called()
                self.assertEqual(requested_params, ["feed pressure", "thrust"])
                self.assertEqual(requested_axes, ["cumulative_impulse", "cumulative_impulse"])
                self.assertEqual(window._last_plot_def.get("y_parameters"), ["feed pressure", "thrust"])
                self.assertEqual(window._last_plot_def.get("y_parameter"), "feed pressure")
                self.assertEqual(len(axes.plot_calls), 2)
                labels = [str(kwargs.get("label") or "") for _, kwargs in axes.plot_calls]
                self.assertIn("feed pressure | SN-001 | Program Alpha", labels)
                self.assertIn("thrust | SN-001 | Program Alpha", labels)
                self.assertEqual(window._last_plot_def.get("life_axis"), "cumulative_impulse")
                self.assertEqual(window._last_plot_def.get("life_axis_label"), "Cumulative Impulse")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_plot_life_metrics_metric_xy_allows_multiple_y_parameters(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                axes = self._prepare_plot_window(window, db_path, mode="life_metrics")
                selection = {
                    "mode": "condition",
                    "id": "condition:CondA",
                    "run_name": "CondA",
                    "display_text": "Condition A",
                    "run_condition": "Condition A",
                    "member_runs": ["CondA"],
                    "member_sequences": ["Seq-1"],
                }
                requested_pairs: list[tuple[str, str]] = []
                window.cb_life_plot_type.clear()
                window.cb_life_plot_type.addItem("Parameter vs Life", "life_axis")
                window.cb_life_plot_type.addItem("Parameter vs Parameter", "metric_xy")
                window.cb_life_plot_type.setCurrentIndex(window.cb_life_plot_type.findData("metric_xy"))
                window.cb_life_y_param.clear()
                window.list_life_y_params.clear()
                for name in ("feed pressure", "thrust"):
                    window.cb_life_y_param.addItem(name, name)
                    window.list_life_y_params.addItem(name)
                window.cb_life_x_param.clear()
                window.cb_life_x_param.addItem("Cumulative impulse", "Cumulative impulse")
                window.cb_life_x_param.setCurrentIndex(0)
                self._ensure_selected_items(window.list_life_stats, ["mean"])
                window._set_life_y_parameter_selection(["feed pressure", "thrust"])

                def _fake_loader(
                    run_names: list[str],
                    x_parameter_value: object,
                    y_parameter_value: object,
                    *,
                    stat: object = "mean",
                    serials: list[str] | None = None,
                    filter_state: object = None,
                ) -> list[dict]:
                    del run_names, stat, serials, filter_state
                    x_param = str(x_parameter_value)
                    y_param = str(y_parameter_value)
                    requested_pairs.append((x_param, y_param))
                    return [
                        {
                            "serial": "SN-001",
                            "observation_id": f"obs-{y_param}",
                            "sequence_index": 1,
                            "condition_key": "CondA",
                            "x_value": 10.0,
                            "y_value": 1.0 if y_param == "feed pressure" else 2.0,
                            "x_units": "lbf-s",
                            "y_units": "psi",
                            "program_title": "Program Alpha",
                        }
                    ]

                with patch.object(window, "_current_run_selector_mode", return_value="condition"), patch.object(
                    window, "_current_run_selection", return_value=selection
                ), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_active_serials", return_value=["SN-001"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_compose_run_title", return_value="Life XY Plot"
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
                    window, "_load_life_metric_xy_for_selection", side_effect=_fake_loader
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._plot_life_metrics()

                info_mock.assert_not_called()
                self.assertEqual(
                    requested_pairs,
                    [("Cumulative impulse", "feed pressure"), ("Cumulative impulse", "thrust")],
                )
                self.assertEqual(window._last_plot_def.get("plot_type"), "metric_xy")
                self.assertEqual(window._last_plot_def.get("x_parameter"), "Cumulative impulse")
                self.assertEqual(window._last_plot_def.get("y_parameters"), ["feed pressure", "thrust"])
                self.assertEqual(window._last_plot_def.get("y_parameter"), "feed pressure")
                self.assertEqual(len(axes.plot_calls), 2)
                labels = [str(kwargs.get("label") or "") for _, kwargs in axes.plot_calls]
                self.assertIn("feed pressure | SN-001 | Program Alpha", labels)
                self.assertIn("thrust | SN-001 | Program Alpha", labels)
                snapshot = window._last_life_metrics_excel_snapshot
                self.assertIsInstance(snapshot, dict)
                self.assertEqual(snapshot.get("plot_metadata", {}).get("plot_type"), "metric_xy")
                self.assertEqual(snapshot.get("plot_metadata", {}).get("x_parameter"), "Cumulative impulse")
                self.assertEqual(snapshot.get("plot_metadata", {}).get("y_parameters"), ["feed pressure", "thrust"])
                self.assertEqual(len(snapshot.get("traces") or []), 2)
                self.assertEqual(
                    [str(trace.get("y_parameter") or "") for trace in (snapshot.get("traces") or [])],
                    ["feed pressure", "thrust"],
                )
                self.assertEqual(
                    [str(row.get("observation_id") or "") for row in (snapshot.get("rows") or [])],
                    ["obs-feed pressure", "obs-thrust"],
                )
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_life_metrics_excel_export_button_tracks_snapshot_state(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                self._prepare_plot_window(window, db_path, mode="life_metrics")
                selection = {
                    "mode": "condition",
                    "id": "condition:CondA",
                    "run_name": "CondA",
                    "display_text": "Condition A",
                    "run_condition": "Condition A",
                    "member_runs": ["CondA"],
                    "member_sequences": ["Seq-1"],
                }
                window.cb_life_y_param.clear()
                window.cb_life_y_param.addItem("LifeMetric", "LifeMetric")
                window.cb_life_axis.clear()
                window.cb_life_axis.addItem("Sequence", "sequence_index")
                self.assertFalse(window.btn_export_life_metrics_excel.isEnabled())

                def _fake_loader(
                    run_names: list[str],
                    parameter_value: object,
                    life_axis: str,
                    *,
                    stat: object = "mean",
                    serials: list[str] | None = None,
                    filter_state: object = None,
                ) -> list[dict]:
                    del run_names, parameter_value, life_axis, serials, filter_state
                    return [
                        {
                            "serial": "SN-001",
                            "observation_id": f"obs-{stat}",
                            "sequence_index": 1,
                            "condition_key": "CondA",
                            "x_value": 1.0,
                            "y_value": 1.0,
                            "units": "psi",
                            "program_title": "Program Alpha",
                        }
                    ]

                with patch.object(window, "_current_run_selector_mode", return_value="condition"), patch.object(
                    window, "_current_run_selection", return_value=selection
                ), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_active_serials", return_value=["SN-001"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_compose_run_title", return_value="Life Plot"
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
                    window, "_load_life_metric_series_for_selection", side_effect=_fake_loader
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._plot_life_metrics()

                info_mock.assert_not_called()
                self.assertTrue(window.btn_export_life_metrics_excel.isEnabled())
                window._last_plot_def = {"mode": "metrics"}
                window._sync_life_metrics_excel_export_action()
                self.assertFalse(window.btn_export_life_metrics_excel.isEnabled())
                window._clear_life_metrics_excel_snapshot()
                self.assertFalse(window.btn_export_life_metrics_excel.isEnabled())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_plot_curves_keeps_visible_rows_when_live_valve_filter_is_active(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = _create_plot_filter_db(tmpdir)
                axes = self._prepare_plot_window(window, db_path, mode="curves")
                selection = {
                    "mode": "sequence",
                    "id": "sequence:CondA|Program Alpha|Seq-1",
                    "run_name": "CondA",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq-1",
                    "member_runs": ["CondA"],
                    "member_sequences": ["Seq-1"],
                }

                with patch.object(window, "_current_run_selection", return_value=selection), patch.object(
                    window, "_current_member_runs", return_value=["CondA"]
                ), patch.object(
                    window, "_current_run_selections", return_value=[selection]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_selected_curve_y_names", return_value=["Pressure"]
                ), patch.object(
                    window, "_current_curve_x_key", return_value="time"
                ), patch.object(
                    window, "_current_curve_x_label", return_value="Time"
                ), patch.object(
                    window, "_resolve_curve_x_key", return_value="time"
                ), patch.object(
                    window, "_compose_run_title", return_value="Curve Plot"
                ), patch.object(
                    window, "_apply_interactive_legend_policy", return_value=[]
                ), patch.object(
                    window, "_apply_plot_view_bands_to_axes", return_value=None
                ), patch.object(
                    window, "_capture_main_plot_base_view", return_value=None
                ), patch.object(
                    window, "_populate_stats_table", return_value=None
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._plot_curves()

                info_mock.assert_not_called()
                self.assertEqual(window._last_plot_def["mode"], "curves")
                self.assertTrue(axes.plot_calls)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_plot_curves_supports_multiple_runs_and_y_columns(self) -> None:
        window = self._make_window()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                db_path = Path(tmpdir) / "cache.sqlite3"
                db_path.write_text("", encoding="utf-8")
                axes = self._prepare_plot_window(window, db_path, mode="curves")
                selections = [
                    {
                        "mode": "condition",
                        "id": "condition:CondA",
                        "run_name": "CondA",
                        "display_text": "Condition A",
                        "member_runs": ["CondA"],
                    },
                    {
                        "mode": "condition",
                        "id": "condition:CondB",
                        "run_name": "CondB",
                        "display_text": "Condition B",
                        "member_runs": ["CondB"],
                    },
                ]
                combined_selection = {
                    "mode": "condition",
                    "id": "condition:multi:condition:CondA|condition:CondB",
                    "display_text": "Condition A, Condition B",
                    "run_condition": "Condition A, Condition B",
                    "member_runs": ["CondA", "CondB"],
                }

                def _fake_load_curves(run_name: str, y_name: str, *_args, **_kwargs) -> list[dict]:
                    return [
                        {
                            "serial": f"{run_name}-{y_name}-SN-001",
                            "x": [0.0, 1.0],
                            "y": [1.0, 2.0],
                            "program_title": "Program Alpha",
                            "source_run_name": f"{run_name}-Seq",
                        }
                    ]

                with patch.object(window, "_current_run_selection", return_value=combined_selection), patch.object(
                    window, "_current_run_selections", return_value=selections
                ), patch.object(
                    window, "_current_member_runs", return_value=["CondA", "CondB"]
                ), patch.object(
                    window, "_ensure_main_axes", return_value=None
                ), patch.object(
                    window, "_selected_curve_y_names", return_value=["Pressure", "Voltage"]
                ), patch.object(
                    window, "_current_curve_x_key", return_value="time"
                ), patch.object(
                    window, "_current_curve_x_label", return_value="Time"
                ), patch.object(
                    window, "_resolve_curve_x_key", return_value="time"
                ), patch.object(
                    window, "_compose_run_title", return_value="Curve Plot"
                ), patch.object(
                    window, "_load_curves_for_selection", side_effect=_fake_load_curves
                ), patch.object(
                    window, "_apply_interactive_legend_policy", return_value=[]
                ), patch.object(
                    window, "_apply_plot_view_bands_to_axes", return_value=None
                ), patch.object(
                    window, "_capture_main_plot_base_view", return_value=None
                ), patch.object(
                    window, "_populate_stats_table", return_value=None
                ), patch(
                    "ui_next.qt_main.QtWidgets.QMessageBox.information"
                ) as info_mock:
                    window._plot_curves()

                info_mock.assert_not_called()
                self.assertEqual(len(axes.plot_calls), 4)
                labels = [str(kwargs.get("label") or "") for _, kwargs in axes.plot_calls]
                self.assertTrue(any(label.startswith("Pressure | ") for label in labels))
                self.assertTrue(any(label.startswith("Voltage | ") for label in labels))
                self.assertEqual(window._last_plot_def["member_runs"], ["CondA", "CondB"])
                self.assertEqual(window._last_plot_def["y"], ["Pressure", "Voltage"])
                self.assertEqual(window._last_plot_def["selection_ids"], ["condition:CondA", "condition:CondB"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_auto_plot_display_name_lists_multiple_curve_y_columns(self) -> None:
        window = self._make_window()
        try:
            name = window._auto_plot_display_name(
                {
                    "plot_definition": {
                        "mode": "curves",
                        "x": "Time",
                        "y": ["Pressure", "Voltage"],
                    }
                }
            )
            self.assertIn("Pressure", name)
            self.assertIn("Voltage", name)
            self.assertIn("Time", name)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_curve_trace_label_uses_serial_number_from_composite_source_key(self) -> None:
        window = self._make_window()
        try:
            label = window._curve_trace_label(
                "CondA",
                {
                    "serial": "Program Alpha / Valve / Primary / SN-001 / source_a",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq-1",
                },
                multi_run=False,
            )
            self.assertEqual(label, "SN-001 | Program Alpha | Seq-1")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_curve_trace_label_uses_serial_number_from_prefixed_composite_source_key(self) -> None:
        window = self._make_window()
        try:
            label = window._curve_trace_label(
                "CondA",
                {
                    "serial": r"C:\repo\EIDAT Support\doc_a / Program Alpha / Valve / Primary / SN-001 / source_a",
                    "program_title": "Program Alpha",
                    "source_run_name": "Seq-1",
                },
                multi_run=False,
            )
            self.assertEqual(label, "SN-001 | Program Alpha | Seq-1")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_stats_panel_displays_serial_number_from_composite_source_key(self) -> None:
        window = self._make_window()
        try:
            composite_serial = "Program Alpha / Valve / Primary / SN-001 / source_a"
            db_path = Path(getattr(window, "_test_tmpdir", "")) / "cache.sqlite3"
            db_path.write_text("", encoding="utf-8")
            window._db_path = db_path

            with patch.object(window, "_current_run_selection", return_value={}), patch.object(
                window, "_selected_metric_plot_source", return_value="sequence"
            ), patch.object(
                window,
                "_load_metric_series_for_selection",
                return_value=[{"serial": composite_serial, "value_num": 1.0}],
            ):
                window._populate_stats_table("CondA", "Pressure", composite_serial)

            self.assertEqual(window._stats_values["serial"].text(), "SN-001")
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_interactive_legend_policy_shows_button_without_overflow(self) -> None:
        window = self._make_window()
        try:
            _fig, ax = self._make_real_legend_axes()
            btn = QtWidgets.QPushButton()
            btn.setVisible(False)
            btn.setEnabled(False)
            ax.plot([0.0, 1.0], [1.0, 2.0], label="SN-001")
            ax.plot([0.0, 1.0], [2.0, 3.0], label="SN-002")

            entries = window._apply_interactive_legend_policy(ax, overflow_button=btn)

            self.assertEqual([str(entry.get("label") or "") for entry in entries], ["SN-001", "SN-002"])
            self.assertFalse(btn.isHidden())
            self.assertTrue(btn.isEnabled())
            self.assertIsNotNone(ax.get_legend())
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_apply_legend_highlight_labels_emphasizes_selected_labels_only(self) -> None:
        window = self._make_window()
        try:
            _fig, ax = self._make_real_legend_axes()
            line_a = ax.plot([0.0, 1.0], [1.0, 2.0], linewidth=1.1, alpha=0.72, label="SN-001")[0]
            line_b = ax.plot([0.0, 1.0], [2.0, 3.0], linewidth=1.2, alpha=0.74, label="SN-002")[0]
            base_a = line_a.get_linewidth()
            base_b = line_b.get_linewidth()

            filtered = window._apply_legend_highlight_labels_to_axes(ax, ["Missing", "SN-002"])

            self.assertEqual(filtered, ["SN-002"])
            self.assertAlmostEqual(line_a.get_linewidth(), base_a)
            self.assertGreater(line_b.get_linewidth(), base_b)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_main_plot_legend_popup_seeds_checks_and_highlights_multiple_labels(self) -> None:
        window = self._make_window()
        try:
            _fig, ax = self._make_real_legend_axes()
            line_a = ax.plot([0.0, 1.0], [1.0, 2.0], linewidth=1.0, alpha=0.75, label="SN-001")[0]
            line_b = ax.plot([0.0, 1.0], [2.0, 3.0], linewidth=1.0, alpha=0.75, label="SN-002")[0]
            base_a = line_a.get_linewidth()
            base_b = line_b.get_linewidth()
            window._axes = ax
            window._canvas = _DummyCanvas()
            window._last_plot_def = {"mode": "curves", "legend_highlight_labels": ["SN-001"]}
            window._main_plot_legend_entries = window._apply_interactive_legend_policy(
                ax,
                overflow_button=QtWidgets.QPushButton(),
                highlighted_labels=["SN-001"],
            )
            interaction: dict[str, object] = {"found": False, "checks": {}}

            def _interact() -> None:
                dialogs = [
                    widget
                    for widget in QtWidgets.QApplication.topLevelWidgets()
                    if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == "Plot Legend"
                ]
                if not dialogs:
                    return
                interaction["found"] = True
                dlg = dialogs[-1]
                listw = dlg.findChild(QtWidgets.QListWidget)
                if listw is not None:
                    checks: dict[str, bool] = {}
                    for idx in range(listw.count()):
                        item = listw.item(idx)
                        if item is None:
                            continue
                        checks[str(item.text() or "")] = item.checkState() == QtCore.Qt.CheckState.Checked
                        if str(item.text() or "") == "SN-002":
                            item.setCheckState(QtCore.Qt.CheckState.Checked)
                    interaction["checks"] = checks
                for button in dlg.findChildren(QtWidgets.QPushButton):
                    if button.text() == "Highlight":
                        button.click()
                    if button.text() == "Close":
                        button.click()

            QtCore.QTimer.singleShot(0, _interact)
            window._open_main_plot_legend_popup()

            self.assertTrue(interaction["found"])
            self.assertEqual(interaction["checks"], {"SN-001": True, "SN-002": False})
            self.assertEqual(window._last_plot_def.get("legend_highlight_labels"), ["SN-001", "SN-002"])
            self.assertGreater(line_a.get_linewidth(), base_a)
            self.assertGreater(line_b.get_linewidth(), base_b)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_main_plot_legend_popup_clear_restores_base_style(self) -> None:
        window = self._make_window()
        try:
            _fig, ax = self._make_real_legend_axes()
            line_a = ax.plot([0.0, 1.0], [1.0, 2.0], linewidth=1.05, alpha=0.72, label="SN-001")[0]
            line_b = ax.plot([0.0, 1.0], [2.0, 3.0], linewidth=1.05, alpha=0.72, label="SN-002")[0]
            base_a = line_a.get_linewidth()
            base_b = line_b.get_linewidth()
            window._axes = ax
            window._canvas = _DummyCanvas()
            window._last_plot_def = {"mode": "curves", "legend_highlight_labels": ["SN-001", "SN-002"]}
            window._main_plot_legend_entries = window._apply_interactive_legend_policy(
                ax,
                overflow_button=QtWidgets.QPushButton(),
                highlighted_labels=["SN-001", "SN-002"],
            )
            self.assertGreater(line_a.get_linewidth(), base_a)
            self.assertGreater(line_b.get_linewidth(), base_b)
            interaction = {"found": False}

            def _interact() -> None:
                dialogs = [
                    widget
                    for widget in QtWidgets.QApplication.topLevelWidgets()
                    if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == "Plot Legend"
                ]
                if not dialogs:
                    return
                interaction["found"] = True
                dlg = dialogs[-1]
                for button in dlg.findChildren(QtWidgets.QPushButton):
                    if button.text() == "Clear Highlights":
                        button.click()
                    if button.text() == "Close":
                        button.click()

            QtCore.QTimer.singleShot(0, _interact)
            window._open_main_plot_legend_popup()

            self.assertTrue(interaction["found"])
            self.assertEqual(window._last_plot_def.get("legend_highlight_labels"), [])
            self.assertAlmostEqual(line_a.get_linewidth(), base_a)
            self.assertAlmostEqual(line_b.get_linewidth(), base_b)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_apply_auto_graph_quickcheck_plot_to_live_gui_seeds_legend_highlights(self) -> None:
        window = self._make_window()
        try:
            db_path = Path(getattr(window, "_test_tmpdir", "")) / "cache.sqlite3"
            db_path.write_text("", encoding="utf-8")
            window._db_path = db_path
            window._plot_ready = True
            captured: dict[str, object] = {}

            def _fake_plot_metrics() -> None:
                captured["labels"] = window._legend_highlight_seed_labels()
                window._last_plot_def = {
                    "mode": "metrics",
                    "legend_highlight_labels": window._legend_highlight_seed_labels(),
                }

            plot_entry = {
                "plot_definition": {
                    "mode": "metrics",
                    "selector_mode": "sequence",
                    "selection_id": "sequence:CondA|Program Alpha|Seq-1",
                    "stats": ["mean"],
                    "y": ["Pressure"],
                    "legend_highlight_labels": ["Pressure.mean", "Bounds"],
                }
            }

            with patch.object(window, "_selection_from_plot_def", return_value={"id": "sequence:CondA|Program Alpha|Seq-1"}), patch.object(
                window, "_set_mode", return_value=None
            ), patch.object(
                window, "_select_run_by_id", return_value=None
            ), patch.object(
                window, "_set_metric_plot_source", return_value=None
            ), patch.object(
                window, "_plot_metrics", side_effect=_fake_plot_metrics
            ):
                window._apply_auto_graph_quickcheck_plot_to_live_gui(plot_entry)

            self.assertEqual(captured.get("labels"), ["Pressure.mean", "Bounds"])
            self.assertEqual(window._last_plot_def.get("legend_highlight_labels"), ["Pressure.mean", "Bounds"])
            self.assertIsNone(window._pending_legend_highlight_labels)
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)

    def test_apply_auto_graph_quickcheck_plot_to_live_gui_restores_multiple_life_y_parameters(self) -> None:
        window = self._make_window()
        try:
            db_path = Path(getattr(window, "_test_tmpdir", "")) / "cache.sqlite3"
            db_path.write_text("", encoding="utf-8")
            window._db_path = db_path
            window._plot_ready = True
            captured: dict[str, object] = {}
            window.cb_run_mode.blockSignals(True)
            window.cb_run_mode.clear()
            window.cb_run_mode.addItem("Run Conditions", "condition")
            window.cb_run_mode.setCurrentIndex(0)
            window.cb_run_mode.blockSignals(False)
            window.cb_life_y_param.clear()
            window.list_life_y_params.clear()
            for name in ("feed pressure", "thrust"):
                window.cb_life_y_param.addItem(name, name)
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                window.list_life_y_params.addItem(item)
            window.cb_life_axis.clear()
            window.cb_life_axis.addItem("Cumulative Impulse", "cumulative_impulse")

            def _fake_plot_life_metrics() -> None:
                captured["plot_type"] = window._selected_life_plot_type()
                captured["y_values"] = window._selected_life_y_parameters()
                captured["y_summary"] = window.lbl_life_y_params_summary.text()
                captured["combo_value"] = str(window.cb_life_y_param.currentData() or "").strip()
                window._last_plot_def = {
                    "mode": "life_metrics",
                    "plot_type": window._selected_life_plot_type(),
                    "y_parameters": window._selected_life_y_parameters(),
                }

            plot_entry = {
                "plot_definition": {
                    "mode": "life_metrics",
                    "selector_mode": "condition",
                    "selection_id": "condition:CondA",
                    "plot_type": "life_axis",
                    "life_axis": "cumulative_impulse",
                    "stats": ["mean"],
                    "y_parameters": ["feed pressure", "thrust"],
                    "y_parameter": "feed pressure",
                }
            }

            with patch.object(window, "_selection_from_plot_def", return_value={"id": "condition:CondA"}), patch.object(
                window, "_set_mode", return_value=None
            ), patch.object(
                window, "_set_metric_condition_selection_ids", return_value=None
            ), patch.object(
                window, "_plot_life_metrics", side_effect=_fake_plot_life_metrics
            ):
                window._apply_auto_graph_quickcheck_plot_to_live_gui(plot_entry)

            self.assertEqual(captured.get("plot_type"), "life_axis")
            self.assertEqual(captured.get("y_values"), ["feed pressure", "thrust"])
            self.assertEqual(captured.get("combo_value"), "feed pressure")
            self.assertEqual(captured.get("y_summary"), "Y Parameters: All (2)")
            self.assertEqual(window._last_plot_def.get("y_parameters"), ["feed pressure", "thrust"])
        finally:
            window.close()
            tmpdir = getattr(window, "_test_tmpdir", "")
            if tmpdir:
                shutil.rmtree(str(tmpdir), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
