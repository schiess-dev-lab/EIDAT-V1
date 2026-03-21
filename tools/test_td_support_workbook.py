import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _have_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return False
    return True


def _have_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


def _have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


def _have_pyside6() -> bool:
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _PerfAxisHarness:
    def __init__(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self.cb_perf_x_col = QtWidgets.QComboBox()
        self.cb_perf_y_col = QtWidgets.QComboBox()
        self.cb_perf_z_col = QtWidgets.QComboBox()
        self.lbl_perf_axes = QtWidgets.QLabel()
        self.lbl_perf_common_runs = QtWidgets.QLabel()
        self.clear_calls = 0

        self._perf_available_columns = [
            {"name": "A", "units": "ua"},
            {"name": "B", "units": "ub"},
            {"name": "C", "units": "uc"},
            {"name": "D", "units": "ud"},
        ]
        self._perf_col_runs = {
            "a": {"r1", "r2", "r3"},
            "b": {"r1", "r2", "r3"},
            "c": {"r1", "r2", "r3"},
            "d": {"r2"},
        }
        self._perf_all_runs = ["r1", "r2", "r3"]
        self._perf_axis_update_in_progress = False

        self._perf_norm_name = TestDataTrendDialog._perf_norm_name

        for name in (
            "_perf_current_col_name",
            "_fill_perf_axis_combo",
            "_set_perf_axis_combo_by_norm",
            "_common_runs_for_perf_vars",
            "_perf_var_names",
            "_update_perf_axes_label",
            "_update_perf_pair_summary",
            "_filter_perf_axis_options",
            "_on_perf_axis_changed",
            "_set_combo_to_value",
        ):
            setattr(self, name, getattr(TestDataTrendDialog, name).__get__(self, _PerfAxisHarness))

        self._update_perf_fit_controls = lambda: None
        self._update_perf_control_period_state = lambda: None
        self._clear_perf_results = lambda: setattr(self, "clear_calls", self.clear_calls + 1)

        self._fill_perf_axis_combo(self.cb_perf_x_col)
        self._fill_perf_axis_combo(self.cb_perf_y_col)
        self._fill_perf_axis_combo(self.cb_perf_z_col, allow_blank=True)
        self._set_perf_axis_combo_by_norm(self.cb_perf_x_col, "a")
        self._set_perf_axis_combo_by_norm(self.cb_perf_y_col, "b")
        self._set_perf_axis_combo_by_norm(self.cb_perf_z_col, "", allow_blank=True)

        self.cb_perf_x_col.currentIndexChanged.connect(lambda *_: self._on_perf_axis_changed("x"))
        self.cb_perf_y_col.currentIndexChanged.connect(lambda *_: self._on_perf_axis_changed("y"))
        self.cb_perf_z_col.currentIndexChanged.connect(lambda *_: self._on_perf_axis_changed("z"))

    def set_user_value(self, cb, value: str) -> None:
        want_norm = self._perf_norm_name(value)
        for i in range(cb.count()):
            if self._perf_norm_name(cb.itemData(i)) == want_norm:
                cb.setCurrentIndex(i)
                return
        raise AssertionError(f"missing combo value {value!r}")


class _PerfControlPeriodHarness:
    def __init__(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self.rb_perf_steady_state = QtWidgets.QRadioButton()
        self.rb_perf_pulsed_mode = QtWidgets.QRadioButton()
        self.rb_perf_pulsed_mode.setChecked(True)
        self.cb_perf_filter_mode = QtWidgets.QComboBox()
        self.cb_perf_filter_mode.addItem("All run conditions", "all_conditions")
        self.cb_perf_filter_mode.addItem("Match control period", "match_control_period")
        self.cb_perf_control_period = QtWidgets.QComboBox()
        self.cb_perf_control_period.addItem("60", 60.0)
        self.cb_perf_control_period.addItem("120", 120.0)
        self.cb_perf_surface_model = QtWidgets.QComboBox()
        self.cb_perf_surface_model.addItem("Auto Surface", "auto_surface")
        self.cb_perf_surface_model.addItem("Quadratic Surface", "quadratic_surface")
        self.cb_perf_surface_model.addItem("Quadratic Surface + Control Period", "quadratic_surface_control_period")
        self.cb_perf_x_col = QtWidgets.QComboBox()
        self.cb_perf_y_col = QtWidgets.QComboBox()
        self.cb_perf_z_col = QtWidgets.QComboBox()
        for cb, values in (
            (self.cb_perf_x_col, [("Input1", "input1")]),
            (self.cb_perf_y_col, [("Output", "output")]),
            (self.cb_perf_z_col, [("None", ""), ("Input2", "input2")]),
        ):
            for text, data in values:
                cb.addItem(text, data)
        self.cb_perf_z_col.setCurrentIndex(1)

        for name in (
            "_selected_perf_run_type_mode",
            "_selected_perf_filter_mode",
            "_perf_current_col_name",
            "_perf_var_names",
            "_perf_requested_surface_family",
            "_selected_perf_control_period",
            "_update_perf_control_period_state",
        ):
            setattr(self, name, getattr(TestDataTrendDialog, name).__get__(self, _PerfControlPeriodHarness))


class _PlotViewBandHarness:
    def __init__(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._mode = "curves"
        self._last_plot_def = None
        self._plot_note_base_text = ""
        self.lbl_plot_note = QtWidgets.QLabel()
        self.plot_band_frame = QtWidgets.QFrame()
        self.plot_band_x_row = QtWidgets.QHBoxLayout()
        self.plot_band_y_row = QtWidgets.QHBoxLayout()
        self.ed_plot_x_band_min = QtWidgets.QLineEdit()
        self.ed_plot_x_band_max = QtWidgets.QLineEdit()
        self.ed_plot_y_band_min = QtWidgets.QLineEdit()
        self.ed_plot_y_band_max = QtWidgets.QLineEdit()
        self._axes = None
        self._canvas = None

        type(self)._view_band_active = staticmethod(TestDataTrendDialog._view_band_active)
        type(self)._parse_view_band_value = staticmethod(TestDataTrendDialog._parse_view_band_value)
        type(self)._normalize_view_band = classmethod(TestDataTrendDialog._normalize_view_band.__func__)
        type(self)._plot_view_band_axes = staticmethod(TestDataTrendDialog._plot_view_band_axes)
        type(self)._plot_view_band_note = staticmethod(TestDataTrendDialog._plot_view_band_note)

        for row, widgets in (
            (self.plot_band_x_row, (QtWidgets.QLabel("x"), self.ed_plot_x_band_min, QtWidgets.QLabel("to"), self.ed_plot_x_band_max)),
            (self.plot_band_y_row, (QtWidgets.QLabel("y"), self.ed_plot_y_band_min, QtWidgets.QLabel("to"), self.ed_plot_y_band_max)),
        ):
            for widget in widgets:
                row.addWidget(widget)

        for name in (
            "_set_plot_note",
            "_displayed_plot_mode",
            "_plot_view_band_mode_for_display",
            "_current_plot_view_bands",
            "_refresh_plot_note",
            "_refresh_plot_view_band_controls",
            "_apply_plot_view_bands_to_axes",
            "_apply_current_plot_view_bands",
        ):
            setattr(self, name, getattr(TestDataTrendDialog, name).__get__(self, _PlotViewBandHarness))


class _MockPlotAxis:
    def __init__(self, *, xlim: tuple[float, float] = (0.0, 10.0), ylim: tuple[float, float] = (0.0, 100.0)) -> None:
        self._xlim = tuple(xlim)
        self._ylim = tuple(ylim)

    def get_xlim(self) -> tuple[float, float]:
        return self._xlim

    def get_ylim(self) -> tuple[float, float]:
        return self._ylim

    def set_xlim(self, lo: float, hi: float) -> None:
        self._xlim = (float(lo), float(hi))

    def set_ylim(self, lo: float, hi: float) -> None:
        self._ylim = (float(lo), float(hi))


class _MockPlotCanvas:
    def __init__(self) -> None:
        self.draw_idle_calls = 0
        self.draw_calls = 0

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1

    def draw(self) -> None:
        self.draw_calls += 1


class _PlotPerformanceHarness:
    def __init__(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._plot_ready = True
        self._db_path = Path("C:/tmp/fake.sqlite3")
        self._perf_require_min_points = 2
        self._perf_all_runs = ["RunA"]
        self._perf_col_runs = {"input1": {"RunA"}, "output": {"RunA"}}
        self._perf_results_by_stat = {}
        self._highlight_sn = ""
        self._last_plot_def = None
        self._plot_note = ""

        self.cb_perf_x_col = QtWidgets.QComboBox()
        self.cb_perf_x_col.addItem("Input 1", "input1")
        self.cb_perf_y_col = QtWidgets.QComboBox()
        self.cb_perf_y_col.addItem("Output", "output")
        self.cb_perf_z_col = QtWidgets.QComboBox()
        self.cb_perf_z_col.addItem("None", "")
        self.cb_perf_fit = QtWidgets.QCheckBox()
        self.cb_perf_fit.setChecked(True)
        self.cb_perf_view_stat = QtWidgets.QComboBox()
        self.btn_save_plot_pdf = QtWidgets.QPushButton()
        self.btn_add_auto_plot = QtWidgets.QPushButton()

        self._perf_current_col_name = lambda cb: str(cb.currentData() or cb.currentText() or "").strip()
        self._perf_norm_name = lambda value: "".join(ch.lower() for ch in str(value or "") if ch.isalnum())
        self._perf_checked_stats = lambda: ["mean"]
        self._perf_plot_stat_candidates = lambda: ["mean"]
        self._selected_perf_runs = lambda: ["RunA"]
        self._selected_perf_serials = lambda: ["SN1"]
        self._selected_perf_run_type_mode = lambda: "steady_state"
        self._selected_perf_filter_mode = lambda: "all_conditions"
        self._selected_perf_control_period = lambda: None
        self._perf_var_names = lambda: ("output", "input1", "")
        self._common_runs_for_perf_vars = lambda output, input1, input2: ["RunA"]
        self._perf_collect_results = lambda *args, **kwargs: (
            {
                "mean": {
                    "plot_dimension": "2d",
                    "master_model": {},
                    "curves": {"SN1": [(1.0, 2.0, "RunA"), (2.0, 3.0, "RunA")]},
                }
            },
            ["mean"],
            "",
        )
        self._populate_perf_stats = lambda stats: None
        self._update_perf_pair_summary = lambda **kwargs: None
        self._set_plot_note = lambda text="": setattr(self, "_plot_note", str(text or ""))
        self._update_perf_highlight_models = lambda: None
        self._fill_perf_equations_table = lambda: None
        self._redraw_performance_view = lambda: None
        self._current_run_selection = lambda: {
            "mode": "sequence",
            "id": "sequence:RunA",
            "member_sequences": ["RunA"],
            "member_runs": ["RunA"],
        }
        self._selection_display_text = lambda selection: "RunA"
        self._selection_condition_label = lambda selection: "RunA"
        self._perf_requested_surface_family = lambda: "auto_surface"
        self._perf_requested_fit_mode = lambda: "auto"

        self._plot_performance = getattr(TestDataTrendDialog, "_plot_performance").__get__(self, _PlotPerformanceHarness)


class _RenderPlotDefHarness:
    def __init__(self) -> None:
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._db_path = Path("C:/tmp/fake.sqlite3")
        self._serial_source_rows = []
        self._run_name_by_display = {}
        self._perf_all_runs = ["RunA"]
        self._perf_require_min_points = 2
        self.collect_calls: list[dict] = []

        def _collect_results(
            output_name: str,
            input1_name: str,
            input2_name: str,
            plot_stats: list[str],
            runs: list[str],
            serials: list[str],
            *,
            fit_enabled: bool,
            require_min_points: int,
            control_period_filter=None,
            display_control_period=None,
            run_type_filter=None,
        ):
            self.collect_calls.append(
                {
                    "output_name": output_name,
                    "input1_name": input1_name,
                    "input2_name": input2_name,
                    "plot_stats": list(plot_stats),
                    "runs": list(runs),
                    "serials": list(serials),
                    "fit_enabled": bool(fit_enabled),
                    "require_min_points": int(require_min_points),
                    "control_period_filter": control_period_filter,
                    "display_control_period": display_control_period,
                    "run_type_filter": run_type_filter,
                }
            )
            return ({"mean": {"plot_dimension": "2d"}}, ["mean"], "")

        self._selection_from_plot_def = lambda d: {"member_runs": ["RunA"]}
        self._selected_perf_runs = lambda: ["RunA"]
        self._selected_perf_serials = lambda: ["SN1"]
        self._selected_perf_run_type_mode = lambda: "steady_state"
        self._common_runs_for_perf_vars = lambda output, input1, input2: ["RunA"]
        self._perf_collect_results = _collect_results
        self._render_performance_result = lambda ax, result, **kwargs: ax.plot([0.0, 1.0], [0.0, 1.0])

        self._render_plot_def_to_figure = getattr(TestDataTrendDialog, "_render_plot_def_to_figure").__get__(self, _RenderPlotDefHarness)


class _RunSelectionHarness:
    def __init__(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._mode = "curves"
        self.cb_run_mode = QtWidgets.QComboBox()
        self.cb_run_mode.addItem("Sequence", "sequence")
        self.cb_run_mode.addItem("Run Conditions", "condition")
        self.lbl_run_combo = QtWidgets.QLabel()
        self.cb_run = QtWidgets.QComboBox()
        self.metrics_condition_frame = QtWidgets.QFrame()
        self.list_metric_run_conditions = QtWidgets.QListWidget()
        self.lbl_run_details = QtWidgets.QLabel()
        self.cb_metric_average = QtWidgets.QCheckBox()
        self.cb_plot_metric_bounds = QtWidgets.QCheckBox()
        self.list_stats = QtWidgets.QListWidget()
        for st in ("mean", "min", "max", "std"):
            self.list_stats.addItem(QtWidgets.QListWidgetItem(st))
        self.list_y_metrics = QtWidgets.QListWidget()
        for name in ("thrust", "current"):
            self.list_y_metrics.addItem(QtWidgets.QListWidgetItem(name))
        self.list_auto_plots = QtWidgets.QListWidget()
        self.btn_open_auto = QtWidgets.QPushButton()
        self.btn_open_all_auto = QtWidgets.QPushButton()
        self.btn_delete_auto = QtWidgets.QPushButton()
        self.btn_save_all_auto = QtWidgets.QPushButton()
        self.btn_view_auto_plots = QtWidgets.QPushButton()
        self._plot_ready = True
        self._db_path = "db.sqlite3"
        self._auto_plots = []
        self._auto_plot_path = Path(tempfile.mkdtemp()) / "auto_plots_test_data.json"
        self._run_selection_views = {"sequence": [], "condition": []}
        self._run_display_by_name = {}
        self._run_name_by_display = {}
        self._available_program_filters = ["Program A", "Program B", "Unknown Program"]
        self._checked_program_filters = list(self._available_program_filters)
        self._is_internal_run_label = TestDataTrendDialog._is_internal_run_label
        self._metric_title_suffix = TestDataTrendDialog._metric_title_suffix
        self._selection_summary_text = TestDataTrendDialog._selection_summary_text
        self._plot_metrics_called = False

        for name in (
            "_current_run_selector_mode",
            "_metrics_condition_multiselect_active",
            "_checked_metric_condition_selections",
            "_set_metric_condition_selection_ids",
            "_combine_run_selections",
            "_current_run_selections",
            "_current_run_selection",
            "_current_member_runs",
            "_run_display_text",
            "_selection_condition_label",
            "_selection_display_text",
            "_selection_title_parts",
            "_compose_run_title",
            "_select_run_by_id",
            "_selection_from_plot_def",
            "_populate_metric_condition_list",
            "_sync_main_auto_plot_actions",
            "_auto_plot_display_name",
            "_selected_auto_plot_definitions",
            "_active_program_filter_values",
            "_visible_run_selection_items",
            "_sync_run_mode_availability",
            "_refresh_run_selection_visibility",
            "_on_metric_condition_selection_changed",
            "_refresh_run_dropdown",
            "_refresh_auto_plots_list",
            "_update_auto_actions",
            "_open_selected_auto_plot",
            "_delete_selected_auto_plots",
        ):
            setattr(self, name, getattr(TestDataTrendDialog, name).__get__(self, _RunSelectionHarness))

        self._refresh_columns_for_run = lambda: None
        self._refresh_stats_preview = lambda: None
        self._set_mode = lambda mode: setattr(self, "_mode", str(mode or "").strip().lower())
        self._plot_metrics = lambda: setattr(self, "_plot_metrics_called", True)
        self._plot_curves = lambda: None
        self._refresh_performance_ui = lambda: None
        self._set_combo_to_value = lambda *args, **kwargs: None
        self._on_perf_axis_changed = lambda *args, **kwargs: None
        self.cb_run_mode.currentIndexChanged.connect(lambda *_: self._refresh_run_dropdown())

    def checked_condition_ids(self) -> list[str]:
        from PySide6 import QtCore

        out: list[str] = []
        for i in range(self.list_metric_run_conditions.count()):
            it = self.list_metric_run_conditions.item(i)
            if not it:
                continue
            data = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if it.checkState() == QtCore.Qt.CheckState.Checked and isinstance(data, dict):
                out.append(str(data.get("id") or "").strip())
        return out


class _GlobalFilterHarness:
    def __init__(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._db_path = Path("C:/tmp/fake.sqlite3")
        self._mode = "curves"
        self._highlight_sn = ""
        self._highlight_sns: list[str] = []
        self._perf_results_by_stat = {}
        self._serial_source_rows = [
            {"serial": "SN1", "serial_number": "SN1", "program_title": "Program A", "document_type": "TD"},
            {"serial": "SN2", "serial_number": "SN2", "program_title": "Program B", "document_type": "TD"},
        ]
        self._serial_source_by_serial = {"SN1": dict(self._serial_source_rows[0]), "SN2": dict(self._serial_source_rows[1])}
        self._available_program_filters = ["Program A", "Program B"]
        self._available_serial_filter_rows = list(self._serial_source_rows)
        self._checked_program_filters = ["Program A"]
        self._checked_serial_filters = ["SN1", "SN2"]
        self.lbl_highlight_serials = QtWidgets.QLabel()
        self._refresh_stats_preview_calls = 0
        self._refresh_stats_preview = lambda: setattr(
            self,
            "_refresh_stats_preview_calls",
            int(getattr(self, "_refresh_stats_preview_calls", 0)) + 1,
        )
        self._update_perf_highlight_models = lambda: None
        self._fill_perf_equations_table = lambda: None
        self._redraw_performance_view = lambda: None
        self._selection_observation_filters = lambda selection=None: ("", "")

        for name in (
            "_active_program_filter_values",
            "_active_serial_rows",
            "_active_serials",
            "_row_program_label",
            "_row_matches_global_filters",
            "_filter_rows_for_global_selection",
            "_load_metric_series_for_selection",
            "_load_curves_for_selection",
            "_selected_perf_serials",
            "_highlight_summary_text",
            "_set_highlight_serials",
        ):
            setattr(self, name, getattr(TestDataTrendDialog, name).__get__(self, _GlobalFilterHarness))


class _SavedPerfActionHarness:
    def __init__(self, project_dir: Path) -> None:
        _qt_app()
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._project_dir = Path(project_dir)
        self._db_path = self._project_dir / "cache.sqlite3"
        self._toast_message = ""
        self._perf_results_by_stat = {"mean": {"master_model": {"fit_family": "polynomial"}}}

        for name in ("_save_current_performance_equation", "_format_saved_performance_entry_detail"):
            setattr(self, name, getattr(TestDataTrendDialog, name).__get__(self, _SavedPerfActionHarness))

        self._perf_has_exportable_models = lambda: True
        self._perf_default_saved_name = lambda: "Performance: thrust vs impulse bit"
        self._build_current_saved_performance_entry = lambda name, existing_entry=None: {
            "id": "entry1",
            "name": name,
            "slug": "entry1",
            "saved_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-01 00:00:00",
            "plot_definition": {},
            "plot_metadata": {},
            "run_specs": [],
            "results_by_stat": {},
            "equation_rows": [],
            "asset_metadata": {},
        }
        self._show_toast = lambda message: setattr(self, "_toast_message", str(message))


class _MetricBoundsHarness:
    def __init__(self, project_dir: Path, workbook_path: Path) -> None:
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        self._project_dir = Path(project_dir)
        self._workbook_path = Path(workbook_path)
        self._metric_bounds_for_run = TestDataTrendDialog._metric_bounds_for_run.__get__(self, _MetricBoundsHarness)


def _build_test_data_dialog():
    _qt_app()
    from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

    with mock.patch.object(TestDataTrendDialog, "_load_cache", lambda self, rebuild=False: None):
        with mock.patch.object(TestDataTrendDialog, "_load_auto_plots", lambda self: None):
            return TestDataTrendDialog(Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "project.xlsx")


@unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
class TestTDTrendDialogCacheLoading(unittest.TestCase):
    def test_load_cache_sync_fallback_validates_existing_cache(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness._project_dir = Path("C:/tmp/project")
        harness._workbook_path = Path("C:/tmp/project/project.xlsx")
        harness.lbl_source = QtWidgets.QLabel()
        harness.lbl_cache = QtWidgets.QLabel()
        harness._refresh_from_cache_calls = 0
        harness._update_plot_zoom_actions_calls = 0
        harness._refresh_from_cache = lambda: setattr(harness, "_refresh_from_cache_calls", harness._refresh_from_cache_calls + 1)
        harness._update_plot_zoom_actions = lambda: setattr(
            harness,
            "_update_plot_zoom_actions_calls",
            harness._update_plot_zoom_actions_calls + 1,
        )
        load_cache = TestDataTrendDialog._load_cache.__get__(harness, _Harness)
        fake_db = harness._project_dir / "implementation_trending.sqlite3"

        with mock.patch.object(be, "validate_existing_test_data_project_cache", return_value=fake_db) as validate_mock:
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
                load_cache(rebuild=False)

        warning_mock.assert_not_called()
        validate_mock.assert_called_once_with(
            harness._project_dir,
            harness._workbook_path,
        )
        self.assertEqual(harness._db_path, fake_db)
        self.assertEqual(harness._refresh_from_cache_calls, 1)
        self.assertEqual(harness._update_plot_zoom_actions_calls, 1)

    def test_load_cache_sync_fallback_warns_on_error(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness._project_dir = Path("C:/tmp/project")
        harness._workbook_path = Path("C:/tmp/project/project.xlsx")
        harness.lbl_source = QtWidgets.QLabel()
        harness.lbl_cache = QtWidgets.QLabel()
        harness._refresh_from_cache_calls = 0
        harness._update_plot_zoom_actions_calls = 0
        harness._refresh_from_cache = lambda: setattr(harness, "_refresh_from_cache_calls", harness._refresh_from_cache_calls + 1)
        harness._update_plot_zoom_actions = lambda: setattr(
            harness,
            "_update_plot_zoom_actions_calls",
            harness._update_plot_zoom_actions_calls + 1,
        )
        load_cache = TestDataTrendDialog._load_cache.__get__(harness, _Harness)

        with mock.patch.object(be, "validate_existing_test_data_project_cache", side_effect=RuntimeError("boom")) as validate_mock:
            with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
                load_cache(rebuild=False)

        validate_mock.assert_called_once_with(
            harness._project_dir,
            harness._workbook_path,
        )
        warning_mock.assert_called_once()
        self.assertEqual(harness._refresh_from_cache_calls, 0)
        self.assertEqual(harness._update_plot_zoom_actions_calls, 0)

    def test_load_cache_progress_path_validates_existing_cache(self) -> None:
        _qt_app()
        from PySide6 import QtCore, QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next import qt_main as qm  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        class _Signal:
            def __init__(self) -> None:
                self._callbacks: list = []

            def connect(self, callback) -> None:
                self._callbacks.append(callback)

            def emit(self, *args) -> None:
                for callback in list(self._callbacks):
                    callback(*args)

        class _ImmediateWorker:
            def __init__(self, task_factory, parent=None):
                self._task_factory = task_factory
                self._running = False
                self.progress = _Signal()
                self.completed = _Signal()
                self.failed = _Signal()

            def isRunning(self) -> bool:
                return self._running

            def start(self) -> None:
                self._running = True
                try:
                    payload = self._task_factory(lambda message: self.progress.emit(message))
                except Exception as exc:
                    self.failed.emit(str(exc))
                else:
                    self.completed.emit(payload)
                finally:
                    self._running = False

        class _ProgressStub:
            def __init__(self) -> None:
                self.lbl_heading = QtWidgets.QLabel()
                self.detail_label = QtWidgets.QLabel()
                self.btn_cancel = QtWidgets.QPushButton()
                self.begin_calls: list[str] = []
                self.finish_calls: list[tuple[str, bool]] = []

            def begin(self, status_text: str) -> None:
                self.begin_calls.append(status_text)

            def finish(self, message: str, success: bool = True) -> None:
                self.finish_calls.append((message, success))

        class _Harness:
            pass

        harness = _Harness()
        harness._project_dir = Path("C:/tmp/project")
        harness._workbook_path = Path("C:/tmp/project/project.xlsx")
        harness.lbl_source = QtWidgets.QLabel()
        harness.lbl_cache = QtWidgets.QLabel()
        harness.btn_refresh_cache = QtWidgets.QPushButton()
        harness.btn_open_support = QtWidgets.QPushButton()
        harness.btn_export_debug_excels = QtWidgets.QPushButton()
        harness.btn_plot = QtWidgets.QPushButton()
        harness._cache_worker = None
        harness._cache_progress_visible = False
        harness._cache_progress_heading = ""
        harness._cache_progress_status = ""
        harness._cache_progress_detail = ""
        harness._cache_progress_timer = QtCore.QTimer()
        harness._report_progress = _ProgressStub()
        harness._refresh_from_cache_calls = 0
        harness._update_plot_zoom_actions_calls = 0
        harness._refresh_from_cache = lambda: setattr(harness, "_refresh_from_cache_calls", harness._refresh_from_cache_calls + 1)
        harness._update_plot_zoom_actions = lambda: setattr(
            harness,
            "_update_plot_zoom_actions_calls",
            harness._update_plot_zoom_actions_calls + 1,
        )

        for name in (
            "_load_cache",
            "_set_cache_controls_enabled",
            "_show_cache_progress_dialog",
            "_on_cache_task_progress",
            "_on_cache_task_done",
            "_on_cache_task_error",
        ):
            setattr(harness, name, getattr(TestDataTrendDialog, name).__get__(harness, _Harness))
        harness._cache_progress_timer.timeout.connect(harness._show_cache_progress_dialog)

        fake_db = harness._project_dir / "implementation_trending.sqlite3"
        with mock.patch.object(qm, "ProjectTaskWorker", _ImmediateWorker):
            with mock.patch.object(be, "validate_existing_test_data_project_cache", return_value=fake_db) as validate_mock:
                with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
                    harness._load_cache(rebuild=False)

        warning_mock.assert_not_called()
        validate_mock.assert_called_once_with(
            harness._project_dir,
            harness._workbook_path,
        )
        self.assertEqual(harness._db_path, fake_db)
        self.assertEqual(harness._refresh_from_cache_calls, 1)
        self.assertEqual(harness._update_plot_zoom_actions_calls, 1)

    def test_load_cache_progress_path_warns_only_when_build_fails(self) -> None:
        _qt_app()
        from PySide6 import QtCore, QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next import qt_main as qm  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        class _Signal:
            def __init__(self) -> None:
                self._callbacks: list = []

            def connect(self, callback) -> None:
                self._callbacks.append(callback)

            def emit(self, *args) -> None:
                for callback in list(self._callbacks):
                    callback(*args)

        class _ImmediateWorker:
            def __init__(self, task_factory, parent=None):
                self._task_factory = task_factory
                self._running = False
                self.progress = _Signal()
                self.completed = _Signal()
                self.failed = _Signal()

            def isRunning(self) -> bool:
                return self._running

            def start(self) -> None:
                self._running = True
                try:
                    _ = self._task_factory(lambda message: self.progress.emit(message))
                except Exception as exc:
                    self.failed.emit(str(exc))
                else:
                    self.completed.emit(None)
                finally:
                    self._running = False

        class _ProgressStub:
            def __init__(self) -> None:
                self.lbl_heading = QtWidgets.QLabel()
                self.detail_label = QtWidgets.QLabel()
                self.btn_cancel = QtWidgets.QPushButton()

            def begin(self, status_text: str) -> None:
                return None

            def finish(self, message: str, success: bool = True) -> None:
                return None

        class _Harness:
            pass

        harness = _Harness()
        harness._project_dir = Path("C:/tmp/project")
        harness._workbook_path = Path("C:/tmp/project/project.xlsx")
        harness.lbl_source = QtWidgets.QLabel()
        harness.lbl_cache = QtWidgets.QLabel()
        harness.btn_refresh_cache = QtWidgets.QPushButton()
        harness.btn_open_support = QtWidgets.QPushButton()
        harness.btn_export_debug_excels = QtWidgets.QPushButton()
        harness.btn_plot = QtWidgets.QPushButton()
        harness._cache_worker = None
        harness._cache_progress_visible = False
        harness._cache_progress_heading = ""
        harness._cache_progress_status = ""
        harness._cache_progress_detail = ""
        harness._cache_progress_timer = QtCore.QTimer()
        harness._report_progress = _ProgressStub()
        harness._refresh_from_cache = lambda: None
        harness._update_plot_zoom_actions = lambda: None

        for name in (
            "_load_cache",
            "_set_cache_controls_enabled",
            "_show_cache_progress_dialog",
            "_on_cache_task_progress",
            "_on_cache_task_done",
            "_on_cache_task_error",
        ):
            setattr(harness, name, getattr(TestDataTrendDialog, name).__get__(harness, _Harness))
        harness._cache_progress_timer.timeout.connect(harness._show_cache_progress_dialog)

        with mock.patch.object(qm, "ProjectTaskWorker", _ImmediateWorker):
            with mock.patch.object(be, "validate_existing_test_data_project_cache", side_effect=RuntimeError("build failed")):
                with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
                    harness._load_cache(rebuild=False)

        warning_mock.assert_called_once()
        self.assertIn("build failed", str(warning_mock.call_args.args[2]))
        self.assertEqual(harness.lbl_cache.text(), "Cache DB: unavailable")

    def test_load_cache_rebuild_path_uses_project_update_pipeline(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness._project_dir = Path("C:/tmp/project")
        harness._workbook_path = Path("C:/tmp/project/project.xlsx")
        harness.lbl_source = QtWidgets.QLabel()
        harness.lbl_cache = QtWidgets.QLabel()
        harness._refresh_from_cache_calls = 0
        harness._update_plot_zoom_actions_calls = 0
        harness._refresh_from_cache = lambda: setattr(harness, "_refresh_from_cache_calls", harness._refresh_from_cache_calls + 1)
        harness._update_plot_zoom_actions = lambda: setattr(
            harness,
            "_update_plot_zoom_actions_calls",
            harness._update_plot_zoom_actions_calls + 1,
        )
        load_cache = TestDataTrendDialog._load_cache.__get__(harness, _Harness)
        fake_db = harness._project_dir / "implementation_trending.sqlite3"
        fake_repo = Path("C:/tmp/repo")
        fake_payload = {"db_path": str(fake_db)}

        with mock.patch.object(be, "resolve_test_data_project_global_repo", return_value=fake_repo) as repo_mock:
            with mock.patch.object(be, "update_test_data_trending_project_workbook", return_value=fake_payload) as update_mock:
                with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
                    load_cache(rebuild=True)

        warning_mock.assert_not_called()
        repo_mock.assert_called_once_with(
            harness._project_dir,
            harness._workbook_path,
        )
        update_mock.assert_called_once_with(
            fake_repo,
            harness._workbook_path,
            overwrite=False,
        )
        self.assertEqual(harness._db_path, fake_db)
        self.assertEqual(harness._refresh_from_cache_calls, 1)
        self.assertEqual(harness._update_plot_zoom_actions_calls, 1)

    def test_generate_debug_excel_files_uses_manual_export_only(self) -> None:
        _qt_app()
        from pathlib import Path
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import TestDataTrendDialog  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness._project_dir = Path("C:/tmp/project")
        harness._workbook_path = Path("C:/tmp/project/project.xlsx")
        generate_debug = TestDataTrendDialog._generate_debug_excel_files.__get__(harness, _Harness)
        exported = {
            "implementation_excel": Path("C:/tmp/project/implementation_trending.xlsx"),
            "raw_cache_excel": Path("C:/tmp/project/test_data_raw_cache.xlsx"),
            "raw_points_excel": Path("C:/tmp/project/test_data_raw_points.xlsx"),
        }

        with mock.patch.object(be, "export_test_data_project_debug_excels", return_value=exported) as export_mock:
            with mock.patch.object(QtWidgets.QMessageBox, "information") as info_mock:
                generate_debug()

        export_mock.assert_called_once_with(harness._project_dir, harness._workbook_path, force=True)
        info_mock.assert_called_once()
        info_args = info_mock.call_args.args
        self.assertEqual(info_args[1], "Debug Excel Files")
        self.assertIn("implementation_trending.xlsx", info_args[2])
        self.assertIn("test_data_raw_cache.xlsx", info_args[2])
        self.assertIn("test_data_raw_points.xlsx", info_args[2])


@unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
class TestTDTrendDialogLayout(unittest.TestCase):
    def test_metrics_y_columns_section_preserves_selection_when_collapsed(self) -> None:
        dlg = _build_test_data_dialog()
        try:
            from PySide6 import QtWidgets

            for name in ("thrust", "current"):
                dlg.list_y_metrics.addItem(QtWidgets.QListWidgetItem(name))
            dlg.list_y_metrics.item(1).setSelected(True)

            dlg.section_metrics_y_columns.set_expanded(False)
            dlg.section_metrics_y_columns.set_expanded(True)

            self.assertTrue(dlg.list_y_metrics.item(1).isSelected())
        finally:
            dlg.close()

    def test_curve_axes_section_preserves_combo_values_when_collapsed(self) -> None:
        dlg = _build_test_data_dialog()
        try:
            dlg.cb_y_curve.addItems(["thrust", "current"])
            dlg.cb_x.addItems(["Time", "Pulse Number"])
            dlg.cb_y_curve.setCurrentText("current")
            dlg.cb_x.setCurrentText("Pulse Number")

            dlg.section_curve_axes.set_expanded(False)
            dlg.section_curve_axes.set_expanded(True)

            self.assertEqual(dlg.cb_y_curve.currentText(), "current")
            self.assertEqual(dlg.cb_x.currentText(), "Pulse Number")
        finally:
            dlg.close()

    def test_performance_equations_section_defaults_collapsed_without_fixed_minimum_height(self) -> None:
        dlg = _build_test_data_dialog()
        try:
            self.assertFalse(dlg.section_perf_equations.is_expanded())
            self.assertEqual(dlg.tbl_perf_equations.minimumHeight(), 0)
        finally:
            dlg.close()

    def test_view_auto_plots_button_tracks_saved_plot_availability_and_opens_popup(self) -> None:
        dlg = _build_test_data_dialog()
        try:
            dlg._plot_ready = True
            dlg._db_path = Path(tempfile.mkdtemp()) / "cache.sqlite3"
            dlg._auto_plots = []
            dlg._sync_main_auto_plot_actions()
            self.assertFalse(dlg.btn_view_auto_plots.isEnabled())

            dlg._auto_plots = [{"name": "Plot 1", "mode": "curves", "y": ["thrust"], "x": "Time"}]
            dlg._sync_main_auto_plot_actions()
            self.assertTrue(dlg.btn_view_auto_plots.isEnabled())

            with mock.patch("PySide6.QtWidgets.QDialog.exec", return_value=0) as exec_mock:
                dlg._open_auto_plots_popup()

            exec_mock.assert_called_once()
        finally:
            dlg.close()

    def test_test_data_dialog_show_event_fits_to_screen(self) -> None:
        dlg = _build_test_data_dialog()
        try:
            from PySide6 import QtGui

            with mock.patch("EIDAT_App_Files.ui_next.qt_main._fit_widget_to_screen") as fit_mock:
                dlg.showEvent(QtGui.QShowEvent())

            fit_mock.assert_called_once_with(dlg)
        finally:
            dlg.close()

    def test_popup_auto_plot_open_selected_uses_supplied_list_widget(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets

        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "350 psia, SS",
                "run_condition": "350 psia, SS",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            },
            {
                "mode": "condition",
                "id": "condition:seq3",
                "run_name": "Seq3",
                "display_text": "410 psia, PM",
                "run_condition": "410 psia, PM",
                "member_runs": ["Seq3"],
                "member_sequences": ["Seq3"],
                "details_text": "Source Sequences: Seq3",
            },
        ]
        harness._auto_plots = [
            {
                "mode": "metrics",
                "selector_mode": "condition",
                "selection_id": "condition:multi:condition:seq1|condition:seq3",
                "selection_ids": ["condition:seq1", "condition:seq3"],
                "selection_labels": ["350 psia, SS", "410 psia, PM"],
                "run_conditions": ["350 psia, SS", "410 psia, PM"],
                "member_runs": ["Seq1", "Seq3"],
                "member_sequences": ["Seq1", "Seq2", "Seq3"],
                "stats": ["mean"],
                "y": ["thrust"],
            }
        ]
        popup_list = QtWidgets.QListWidget()
        popup_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        harness._refresh_auto_plots_list(popup_list)
        popup_list.item(0).setSelected(True)

        harness._open_selected_auto_plot(list_widget=popup_list)

        self.assertEqual(harness.checked_condition_ids(), ["condition:seq1", "condition:seq3"])
        self.assertTrue(harness._plot_metrics_called)

    def test_popup_auto_plot_delete_updates_saved_definitions_and_list(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets

        harness = _RunSelectionHarness()
        harness._auto_plots = [
            {"name": "Plot 1", "mode": "metrics", "stats": ["mean"], "y": ["thrust"]},
            {"name": "Plot 2", "mode": "curves", "y": ["current"], "x": "Time"},
        ]
        popup_list = QtWidgets.QListWidget()
        popup_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        harness._refresh_auto_plots_list(popup_list)
        popup_list.item(0).setSelected(True)

        harness._delete_selected_auto_plots(list_widget=popup_list)

        self.assertEqual([d.get("name") for d in harness._auto_plots], ["Plot 2"])
        self.assertEqual(popup_list.count(), 1)
        self.assertEqual(popup_list.item(0).text(), "Plot 2")

    @unittest.skipUnless(_have_matplotlib(), "matplotlib not installed")
    def test_open_all_auto_plots_panel_still_opens(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from matplotlib.figure import Figure

        class _DummyCanvas(QtWidgets.QWidget):
            def __init__(self, figure) -> None:
                super().__init__()
                self.figure = figure

            def mpl_connect(self, *_args, **_kwargs) -> int:
                return 0

            def draw_idle(self) -> None:
                return None

        dlg = _build_test_data_dialog()
        try:
            dlg._plot_ready = True
            dlg._db_path = Path(tempfile.mkdtemp()) / "cache.sqlite3"
            dlg._auto_plots = [{"name": "Plot 1", "mode": "metrics", "stats": ["mean"], "y": ["thrust"]}]

            with mock.patch.object(dlg, "_render_plot_def_to_figure", return_value=Figure()):
                with mock.patch("matplotlib.backends.backend_qtagg.FigureCanvasQTAgg", _DummyCanvas):
                    with mock.patch("PySide6.QtWidgets.QDialog.exec", return_value=0) as exec_mock:
                        dlg._open_all_auto_plots_panel()

            exec_mock.assert_called_once()
        finally:
            dlg.close()

    @unittest.skipUnless(_have_matplotlib(), "matplotlib not installed")
    def test_save_all_auto_plots_pdf_still_exports_all_saved_plots(self) -> None:
        saved_figures: list[object] = []

        class _DummyPdfPages:
            def __init__(self, _path: str) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def savefig(self, fig) -> None:
                saved_figures.append(fig)

        dlg = _build_test_data_dialog()
        try:
            dlg._plot_ready = True
            dlg._db_path = Path(tempfile.mkdtemp()) / "cache.sqlite3"
            dlg._auto_plots = [{"name": "Plot 1", "mode": "metrics", "stats": ["mean"], "y": ["thrust"]}]

            with mock.patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=("C:/tmp/auto.pdf", "PDF")):
                with mock.patch("matplotlib.backends.backend_pdf.PdfPages", _DummyPdfPages):
                    with mock.patch.object(dlg, "_render_plot_def_to_figure", return_value=object()) as render_mock:
                        dlg._save_all_auto_plots_pdf()

            render_mock.assert_called_once()
            self.assertEqual(len(saved_figures), 1)
        finally:
            dlg.close()


@unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
class TestProjectUpdateUI(unittest.TestCase):
    def test_prepare_dialog_keeps_explicit_launch_size_when_adjust_size_grows(self) -> None:
        _qt_app()
        from PySide6 import QtCore, QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import MainWindow  # type: ignore

        class _Harness:
            pass

        class _LargeHintDialog(QtWidgets.QDialog):
            def sizeHint(self) -> QtCore.QSize:
                return QtCore.QSize(1600, 1200)

        harness = _Harness()
        dlg = _LargeHintDialog()
        dlg.resize(960, 640)

        with mock.patch("EIDAT_App_Files.ui_next.qt_main._fit_widget_to_screen") as fit_mock:
            MainWindow._prepare_dialog(harness, dlg)

        self.assertEqual(dlg.size().width(), 960)
        self.assertEqual(dlg.size().height(), 640)
        fit_mock.assert_called_once_with(dlg)

    def test_project_update_uses_background_task_for_td_projects(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import MainWindow  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness.ed_global_repo = QtWidgets.QLineEdit("C:/tmp/repo")
        harness.cb_project_overwrite = QtWidgets.QCheckBox()
        harness.cb_project_overwrite.setChecked(True)
        harness._selected_project_record = lambda: {
            "name": "Proj",
            "type": getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
            "folder": "C:/tmp/repo/projects/Proj",
            "workbook": "C:/tmp/repo/projects/Proj/project.xlsx",
        }
        harness._append_log = lambda *_args, **_kwargs: None
        captured: dict[str, object] = {}
        harness._start_project_task = lambda **kwargs: captured.update(kwargs)
        act = MainWindow._act_update_project.__get__(harness, _Harness)

        with mock.patch.object(be, "update_test_data_trending_project_workbook", return_value={"workbook": "x"}) as update_mock:
            act()
            self.assertEqual(captured.get("heading"), "Update Project")
            self.assertEqual(captured.get("log_prefix"), "project_update")
            task_factory = captured.get("task_factory")
            self.assertTrue(callable(task_factory))
            progress_lines: list[str] = []
            result = task_factory(progress_lines.append)  # type: ignore[misc]

        self.assertEqual(result.get("workbook"), "x")
        update_mock.assert_called_once()
        args, kwargs = update_mock.call_args
        self.assertEqual(Path(args[1]), Path("C:/tmp/repo/projects/Proj/project.xlsx"))
        self.assertTrue(bool(kwargs.get("overwrite")))
        self.assertFalse(bool(kwargs.get("include_performance_sheets")))
        self.assertTrue(callable(kwargs.get("progress_cb")))

    def test_generate_performance_sheets_uses_background_task_for_td_projects(self) -> None:
        _qt_app()
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import MainWindow  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness._selected_project_record = lambda: {
            "name": "Proj",
            "type": getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
            "folder": "C:/tmp/repo/projects/Proj",
            "workbook": "C:/tmp/repo/projects/Proj/project.xlsx",
        }
        harness._append_log = lambda *_args, **_kwargs: None
        captured: dict[str, object] = {}
        harness._start_project_task = lambda **kwargs: captured.update(kwargs)
        act = MainWindow._act_generate_project_performance_sheets.__get__(harness, _Harness)

        with mock.patch.object(be, "generate_test_data_project_performance_sheets", return_value={"workbook": "x"}) as perf_mock:
            act()
            self.assertEqual(captured.get("heading"), "Generate Performance Sheets")
            self.assertEqual(captured.get("log_prefix"), "project_performance_sheets")
            task_factory = captured.get("task_factory")
            self.assertTrue(callable(task_factory))
            result = task_factory(lambda *_args: None)  # type: ignore[misc]

        self.assertEqual(result.get("workbook"), "x")
        perf_mock.assert_called_once()

    def test_generate_debug_excel_files_uses_background_task_for_td_projects(self) -> None:
        _qt_app()
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import MainWindow  # type: ignore

        class _Harness:
            pass

        harness = _Harness()
        harness._selected_project_record = lambda: {
            "name": "Proj",
            "type": getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
            "folder": "C:/tmp/repo/projects/Proj",
            "workbook": "C:/tmp/repo/projects/Proj/project.xlsx",
        }
        harness._append_log = lambda *_args, **_kwargs: None
        captured: dict[str, object] = {}
        harness._start_project_task = lambda **kwargs: captured.update(kwargs)
        act = MainWindow._act_generate_project_debug_excels.__get__(harness, _Harness)

        with mock.patch.object(
            be,
            "export_test_data_project_debug_excels",
            return_value={"implementation_excel": Path("C:/tmp/repo/projects/Proj/implementation_trending.xlsx")},
        ) as export_mock:
            act()
            self.assertEqual(captured.get("heading"), "Generate Debug Excel Files")
            self.assertEqual(captured.get("log_prefix"), "project_debug_excel_files")
            task_factory = captured.get("task_factory")
            self.assertTrue(callable(task_factory))
            result = task_factory(lambda *_args: None)  # type: ignore[misc]

        self.assertIn("implementation_excel", result)
        export_mock.assert_called_once_with(
            Path("C:/tmp/repo/projects/Proj"),
            Path("C:/tmp/repo/projects/Proj/project.xlsx"),
            force=True,
        )

    def test_handle_project_update_success_logs_td_cache_summary_and_debug_path(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from EIDAT_App_Files.ui_next.qt_main import MainWindow  # type: ignore

        class _Harness:
            pass

        logs: list[str] = []
        harness = _Harness()
        harness._append_log = logs.append
        harness._show_toast = lambda message: setattr(harness, "_toast_message", str(message))
        handle = MainWindow._handle_project_update_success.__get__(harness, _Harness)
        payload = {
            "workbook": "C:/tmp/repo/projects/Proj/project.xlsx",
            "updated_cells": 5,
            "missing_source": 0,
            "missing_value": 0,
            "serials_in_workbook": 1,
            "serials_added": 0,
            "added_serials": [],
            "cache_sync_mode": "noop",
            "cache_sync_reason": "",
            "cache_sync_counts": {"added": 0, "changed": 0, "removed": 0, "unchanged": 1, "invalid": 0, "reingested": 0},
            "cache_state": {
                "impl_counts": {"td_runs": 1, "td_columns_calc_y": 1, "td_metrics_calc": 1},
                "raw_counts": {"td_raw_sequences": 1, "td_columns_raw_y": 1, "td_curves_raw": 1},
            },
            "cache_validation_ok": True,
            "cache_validation_error": "",
            "cache_validation_summary": "mode=none, reason=n/a, impl_complete=True, raw_complete=True",
            "cache_debug_path": "C:/tmp/repo/projects/Proj/td_cache_debug.json",
            "backend_module_path": "C:/tmp/repo/EIDAT_App_Files/ui_next/backend.py",
            "timings": {"total_s": 1.23},
            "saved_equation_refresh": {"refreshed_count": 0, "failed_count": 0, "errors": []},
            "debug_json": json.dumps({"timings_s": {"total_s": 1.23}}),
        }

        with mock.patch.object(QtWidgets.QMessageBox, "information") as info_mock:
            result = handle(
                payload,
                wb_path=Path("C:/tmp/repo/projects/Proj/project.xlsx"),
                ptype=getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
                started=time.perf_counter() - 1.0,
            )

        self.assertTrue(any("TD cache validation=ok" in line for line in logs))
        self.assertTrue(any("TD cache summary:" in line for line in logs))
        self.assertTrue(any("TD cache debug:" in line for line in logs))
        self.assertTrue(any("Backend module:" in line for line in logs))
        self.assertIn("Project updated in", result)
        info_mock.assert_called_once()

    def test_on_project_task_error_logs_failure_message(self) -> None:
        _qt_app()
        from PySide6 import QtWidgets
        from EIDAT_App_Files.ui_next.qt_main import MainWindow  # type: ignore

        class _Harness:
            pass

        logs: list[str] = []
        harness = _Harness()
        harness._project_worker = object()
        harness._project_popup_active = False
        harness._update_project_actions = lambda: setattr(harness, "_actions_updated", True)
        harness._append_log = logs.append
        handler = MainWindow._on_project_task_error.__get__(harness, _Harness)

        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
            handler("final TD cache validation failed: boom", "Update Project")

        self.assertIsNone(harness._project_worker)
        self.assertTrue(bool(getattr(harness, "_actions_updated", False)))
        self.assertTrue(any("final TD cache validation failed: boom" in line for line in logs))
        warning_mock.assert_called_once()


@unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
class TestTDSupportWorkbook(unittest.TestCase):
    def _make_source_sqlite(self, path: Path) -> None:
        rows = [
            (1, 0.0, 100.0, 5.0, 10.0),
            (2, 1.0, 100.0, 5.0, 20.0),
            (3, 2.0, 100.0, 5.0, 30.0),
            (4, 3.0, 100.0, 5.0, 40.0),
            (5, 4.0, 100.0, 5.0, 50.0),
            (6, 5.0, 100.0, 5.0, 60.0),
        ]
        self._make_source_sqlite_with_rows(path, rows)

    def _make_source_sqlite_with_rows(self, path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                """
                CREATE TABLE "sheet__RunA" (
                    excel_row INTEGER NOT NULL,
                    "Time" REAL,
                    "feed pressure" REAL,
                    "pulse width on" REAL,
                    thrust REAL
                )
                """
            )
            conn.executemany(
                'INSERT INTO "sheet__RunA"(excel_row,"Time","feed pressure","pulse width on",thrust) VALUES(?,?,?,?,?)',
                rows,
            )
            conn.commit()

    def _make_config(self) -> dict:
        return {
            "description": "support test",
            "data_group": "Excel Data",
            "columns": [{"name": "thrust", "units": "lbf", "range_min": None, "range_max": None}],
            "statistics": ["mean", "min", "max", "std"],
            "statistics_ignore_first_n": 0,
            "performance_plotters": [],
            "sheet_name": None,
            "header_row": 0,
        }

    def _default_program_sheet_name(self, be) -> str:
        return str(be._td_support_program_sheet_name(be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, 0))

    def _refresh_support_conditions(self, be, wb_path: Path, root: Path) -> None:
        be._refresh_td_support_run_conditions_sheet(
            wb_path,
            project_dir=root,
            param_defs=[{"name": "thrust", "units": "lbf"}],
        )

    def _seed_perf_candidate_db(
        self,
        root: Path,
        *,
        rows: list[tuple[str, str, float, float]],
        support_settings: dict[str, object] | None = None,
        legacy_support_only: bool = False,
    ) -> Path:
        from openpyxl import Workbook, load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        wb_path = root / "project.xlsx"
        Workbook().save(str(wb_path))
        support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)

        if legacy_support_only:
            wb = Workbook()
            ws_settings = wb.active
            ws_settings.title = "Settings"
            ws_settings.append(["key", "value"])
            ws_settings.append(["exclude_first_n_default", ""])
            ws_settings.append(["last_n_rows_default", 10])
            ws_seq = wb.create_sheet("Sequences")
            ws_seq.append(
                [
                    "sequence_name",
                    "source_run_name",
                    "feed_pressure",
                    "pulse_width_on",
                    "exclude_first_n",
                    "last_n_rows",
                    "enabled",
                ]
            )
            ws_bounds = wb.create_sheet("ParameterBounds")
            ws_bounds.append(["sequence_name", "parameter_name", "units", "min_value", "max_value", "enabled"])
            wb.save(str(support_path))
            wb.close()
        else:
            be._write_td_support_workbook(
                support_path,
                sequence_names=sorted({run for _sn, run, _x, _y in rows}),
                param_defs=[
                    {"name": "impulse bit", "units": "mN-s"},
                    {"name": "thrust", "units": "lbf"},
                ],
            )

        if support_settings:
            wb = load_workbook(str(support_path))
            try:
                ws = wb["Settings"]
                row_by_key = {
                    str(ws.cell(r, 1).value or "").strip(): r
                    for r in range(2, (ws.max_row or 0) + 1)
                    if str(ws.cell(r, 1).value or "").strip()
                }
                for key, value in support_settings.items():
                    row = row_by_key.get(str(key))
                    if row is None:
                        row = int((ws.max_row or 0) + 1)
                        ws.cell(row, 1).value = str(key)
                        row_by_key[str(key)] = row
                    ws.cell(row, 2).value = value
                wb.save(str(support_path))
            finally:
                wb.close()

        db_path = root / "implementation_trending.sqlite3"
        with sqlite3.connect(str(db_path)) as conn:
            be._ensure_test_data_tables(conn)
            conn.execute(
                "INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)",
                ("workbook_path", str(wb_path)),
            )
            for run in sorted({run for _sn, run, _x, _y in rows}):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run, "Time", run, "", None, None),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run, "impulse bit", "mN-s", "y"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run, "thrust", "lbf", "y"),
                )
            for serial, run, x_val, y_val in rows:
                observation_id = f"{serial}__{run}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run, "", None, None, 0, 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_metrics_calc
                    (observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, "impulse bit", "mean", float(x_val), 0, 0, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_metrics_calc
                    (observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, "thrust", "mean", float(y_val), 0, 0, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run),
                )
            conn.commit()
        return db_path

    def _seed_perf_export_db_2d(
        self,
        root: Path,
        *,
        rows: list[tuple[str, str, float, float, float | None]],
    ) -> Path:
        from openpyxl import Workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        wb_path = root / "project.xlsx"
        Workbook().save(str(wb_path))
        support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
        be._write_td_support_workbook(
            support_path,
            sequence_names=sorted({run for _sn, run, _x, _y, _cp in rows}),
            param_defs=[
                {"name": "impulse bit", "units": "mN-s"},
                {"name": "thrust", "units": "lbf"},
            ],
        )
        db_path = root / "perf_export_2d.sqlite3"
        with sqlite3.connect(str(db_path)) as conn:
            be._ensure_test_data_tables(conn)
            conn.execute("INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)", ("workbook_path", str(wb_path)))
            for run in sorted({run for _sn, run, _x, _y, _cp in rows}):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run, "Time", run, "pm", None, None),
                )
                for name, units in (("impulse bit", "mN-s"), ("thrust", "lbf")):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run, name, units, "y"),
                    )
            for serial, run, x_val, y_val, cp in rows:
                observation_id = f"{serial}__{run}__{x_val:.6f}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run, "pm", None, cp, 0, 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_metrics_calc
                    (observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, "impulse bit", "mean", float(x_val), 0, 0, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_metrics_calc
                    (observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, "thrust", "mean", float(y_val), 0, 0, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run),
                )
            conn.commit()
        return db_path

    def _seed_perf_export_db_3d(
        self,
        root: Path,
        *,
        rows: list[tuple[str, str, float, float, float, float | None]],
    ) -> Path:
        from openpyxl import Workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        wb_path = root / "project.xlsx"
        Workbook().save(str(wb_path))
        support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
        be._write_td_support_workbook(
            support_path,
            sequence_names=sorted({run for _sn, run, _x1, _x2, _y, _cp in rows}),
            param_defs=[
                {"name": "impulse bit", "units": "mN-s"},
                {"name": "feed pressure", "units": "psia"},
                {"name": "thrust", "units": "lbf"},
            ],
        )
        db_path = root / "perf_export_3d.sqlite3"
        with sqlite3.connect(str(db_path)) as conn:
            be._ensure_test_data_tables(conn)
            conn.execute("INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)", ("workbook_path", str(wb_path)))
            for run in sorted({run for _sn, run, _x1, _x2, _y, _cp in rows}):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run, "Time", run, "pm", None, None),
                )
                for name, units in (("impulse bit", "mN-s"), ("feed pressure", "psia"), ("thrust", "lbf")):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run, name, units, "y"),
                    )
            for serial, run, x1_val, x2_val, y_val, cp in rows:
                observation_id = f"{serial}__{run}__{x1_val:.6f}__{x2_val:.6f}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (observation_id, serial, run, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run, "pm", None, cp, 0, 0),
                )
                for column_name, value_num in (
                    ("impulse bit", float(x1_val)),
                    ("feed pressure", float(x2_val)),
                    ("thrust", float(y_val)),
                ):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO td_metrics_calc
                        (observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (observation_id, serial, run, column_name, "mean", value_num, 0, 0, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, run),
                    )
            conn.commit()
        return db_path

    def _seed_perf_source_metadata(
        self,
        db_path: Path,
        rows: list[tuple[str, str, str]],
    ) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with sqlite3.connect(str(db_path)) as conn:
            be._ensure_test_data_impl_tables(conn)
            for serial, asset_type, asset_specific_type in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_sources(serial, sqlite_path, mtime_ns, size_bytes, status, last_ingested_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (serial, "", 0, 0, "ok", 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_source_metadata(
                        serial,
                        program_title,
                        asset_type,
                        asset_specific_type,
                        vendor,
                        acceptance_test_plan_number,
                        part_number,
                        revision,
                        test_date,
                        report_date,
                        document_type,
                        document_type_acronym,
                        similarity_group,
                        metadata_rel,
                        artifacts_rel,
                        excel_sqlite_rel,
                        metadata_mtime_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        serial,
                        be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE,
                        asset_type,
                        asset_specific_type,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "TD",
                        "TD",
                        "",
                        "",
                        "",
                        "",
                        0,
                    ),
                )
            conn.commit()

    def test_write_support_workbook_seeds_expected_sheets(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            support = root / "proj.support.xlsx"
            be._write_td_support_workbook(
                support,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support), read_only=False, data_only=True)
            try:
                self.assertEqual(
                    wb.sheetnames,
                    [
                        "Settings",
                        "Programs",
                        self._default_program_sheet_name(be),
                    ],
                )
                ws_settings = wb["Settings"]
                self.assertEqual(ws_settings.cell(2, 1).value, "exclude_first_n_default")
                self.assertEqual(ws_settings.cell(3, 1).value, "last_n_rows_default")
                self.assertEqual(ws_settings.cell(3, 2).value, 10)
                self.assertEqual(ws_settings.cell(4, 1).value, "perf_eq_strictness")
                self.assertEqual(str(ws_settings.cell(4, 2).value or "").strip().lower(), "medium")
                self.assertEqual(ws_settings.cell(5, 1).value, "perf_eq_point_count")
                self.assertEqual(str(ws_settings.cell(5, 2).value or "").strip().lower(), "medium")
                self.assertIsNotNone(ws_settings["A4"].comment)
                self.assertIn("tight, medium, loose", str(ws_settings["A4"].comment.text or ""))
                self.assertIsNotNone(ws_settings["A5"].comment)
                self.assertIn("tight = 4 points", str(ws_settings["A5"].comment.text or ""))

                ws_programs = wb["Programs"]
                self.assertEqual(ws_programs.cell(2, 1).value, be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE)
                self.assertEqual(ws_programs.cell(2, 2).value, self._default_program_sheet_name(be))
                ws_prog = wb[self._default_program_sheet_name(be)]

                self.assertNotIn("RunConditions", wb.sheetnames)
                self.assertNotIn("RunConditionBounds", wb.sheetnames)
                self.assertEqual(ws_prog.cell(2, 1).value, "RunA")
                self.assertEqual(ws_prog.cell(2, 2).value, "RunA")
                self.assertEqual(ws_prog.cell(1, 8).value, "control_period")
                self.assertTrue(bool(ws_prog.cell(2, 11).value))
            finally:
                wb.close()

    def test_write_support_workbook_new_schema_seeds_program_tabs_only(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            support = root / "proj.support.xlsx"
            be._write_td_support_workbook(
                support,
                sequence_names=["Seq1", "Seq2", "Seq3"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
                program_titles=["Program A", "Program B"],
                sequences_by_program={"Program A": ["Seq1", "Seq2"], "Program B": ["Seq3"]},
            )

            wb = load_workbook(str(support), read_only=False, data_only=True)
            try:
                self.assertIn("Settings", wb.sheetnames)
                self.assertIn("Programs", wb.sheetnames)
                self.assertNotIn("RunConditionBounds", wb.sheetnames)
                self.assertNotIn("RunConditions", wb.sheetnames)
                ws_programs = wb["Programs"]
                sheet_a = str(ws_programs.cell(2, 2).value or "").strip()
                sheet_b = str(ws_programs.cell(3, 2).value or "").strip()
                self.assertTrue(sheet_a.startswith("Program_"))
                self.assertTrue(sheet_b.startswith("Program_"))
                ws_a = wb[sheet_a]
                ws_b = wb[sheet_b]
                self.assertEqual([ws_a.cell(r, 1).value for r in range(2, 4)], ["Seq1", "Seq2"])
                self.assertEqual([ws_b.cell(r, 1).value for r in range(2, 3)], ["Seq3"])
                cond_sheet_names = [name for name in wb.sheetnames if str(name).startswith(be.TD_SUPPORT_CONDITION_SHEET_PREFIX)]
                self.assertEqual(cond_sheet_names, [])
            finally:
                wb.close()

    def test_update_project_generates_deferred_run_conditions_sheet(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db), "program_title": be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                self.assertNotIn("RunConditions", wb.sheetnames)
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 3).value = "100 psia, PM, 5 Sec ON / 20 Sec OFF"
                ws_prog.cell(2, 4).value = 100
                ws_prog.cell(2, 5).value = "psia"
                ws_prog.cell(2, 6).value = "PM"
                ws_prog.cell(2, 7).value = 5
                ws_prog.cell(2, 8).value = 20
                wb.save(str(support_path))
            finally:
                wb.close()

            be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True, require_existing_cache=False)

            wb = load_workbook(str(support_path), read_only=True, data_only=True)
            try:
                self.assertEqual(wb.sheetnames[-1], "RunConditions")
                ws_cond = wb["RunConditions"]
                self.assertEqual(
                    [ws_cond.cell(1, c).value for c in range(1, 15)],
                    [
                        "condition_key",
                        "display_name",
                        "feed_pressure",
                        "feed_pressure_units",
                        "run_type",
                        "pulse_width_on",
                        "control_period",
                        "member_sequences",
                        "member_programs",
                        "parameter_name",
                        "units",
                        "min_value",
                        "max_value",
                        "enabled",
                    ],
                )
                self.assertEqual(str(ws_cond.cell(2, 1).value or "").strip(), "Seq1")
                self.assertEqual(str(ws_cond.cell(2, 10).value or "").strip(), "thrust")
            finally:
                wb.close()

    def test_update_project_only_ensures_cache_once(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db), "program_title": be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            with mock.patch.object(be, "ensure_test_data_project_cache", wraps=be.ensure_test_data_project_cache) as mocked_ensure:
                be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True, require_existing_cache=False)
            self.assertEqual(mocked_ensure.call_count, 1)

    def test_update_project_preserves_existing_run_condition_bounds(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db), "program_title": be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 3).value = "100 psia, PM, 5 Sec ON / 20 Sec OFF"
                ws_prog.cell(2, 4).value = 100
                ws_prog.cell(2, 5).value = "psia"
                ws_prog.cell(2, 6).value = "PM"
                ws_prog.cell(2, 7).value = 5
                ws_prog.cell(2, 8).value = 20
                wb.save(str(support_path))
            finally:
                wb.close()

            be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True, require_existing_cache=False)

            wb = load_workbook(str(support_path))
            try:
                ws_cond = wb["RunConditions"]
                ws_cond.cell(2, 12).value = 15
                ws_cond.cell(2, 13).value = 45
                wb.save(str(support_path))
            finally:
                wb.close()

            be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True, require_existing_cache=False)

            wb = load_workbook(str(support_path), read_only=True, data_only=True)
            try:
                ws_cond = wb["RunConditions"]
                self.assertEqual(ws_cond.cell(2, 12).value, 15)
                self.assertEqual(ws_cond.cell(2, 13).value, 45)
            finally:
                wb.close()

    def test_refresh_support_conditions_is_noop_when_generated_sheet_is_unchanged(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db), "program_title": be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 1).value = "RunA"
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 3).value = "100 psia, PM, 5 Sec ON / 20 Sec OFF"
                ws_prog.cell(2, 4).value = 100
                ws_prog.cell(2, 5).value = "psia"
                ws_prog.cell(2, 6).value = "PM"
                ws_prog.cell(2, 7).value = 5
                ws_prog.cell(2, 8).value = 20
                wb.save(str(support_path))
            finally:
                wb.close()

            first = be._refresh_td_support_run_conditions_sheet(
                wb_path,
                project_dir=root,
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            mtime_before = support_path.stat().st_mtime_ns
            second = be._refresh_td_support_run_conditions_sheet(
                wb_path,
                project_dir=root,
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            mtime_after = support_path.stat().st_mtime_ns

            self.assertTrue(first.get("updated"))
            self.assertFalse(second.get("updated"))
            self.assertEqual(mtime_before, mtime_after)

    def test_refresh_support_conditions_normalizes_overlong_program_sheet_names(self) -> None:
        import warnings

        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": ""}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            long_title = "Program_01_This_is_a_very_long_program_sheet_name_that_exceeds_excel_limits"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wb = load_workbook(str(support_path))
                try:
                    ws_programs = wb["Programs"]
                    program_sheet = wb[self._default_program_sheet_name(be)]
                    program_sheet.title = long_title
                    ws_programs.cell(2, 2).value = long_title
                    wb.save(str(support_path))
                finally:
                    wb.close()

            self._refresh_support_conditions(be, wb_path, root)

            wb = load_workbook(str(support_path), read_only=True, data_only=True)
            try:
                self.assertIn("RunConditions", wb.sheetnames)
                self.assertTrue(all(len(str(name or "")) <= 31 for name in wb.sheetnames))
                ws_programs = wb["Programs"]
                normalized_name = str(ws_programs.cell(2, 2).value or "").strip()
                self.assertTrue(normalized_name)
                self.assertLessEqual(len(normalized_name), 31)
                self.assertIn(normalized_name, wb.sheetnames)
            finally:
                wb.close()

    def test_sync_sqlite_excel_mirror_uses_unique_sheet_titles_for_long_table_names(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = root / "mirror.sqlite3"
            table_a = "table_name_that_is_definitely_longer_than_thirty_one_chars_A"
            table_b = "table_name_that_is_definitely_longer_than_thirty_one_chars_B"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(f'CREATE TABLE "{table_a}" (id INTEGER, value TEXT)')
                conn.execute(f'CREATE TABLE "{table_b}" (id INTEGER, value TEXT)')
                conn.execute(f'INSERT INTO "{table_a}"(id, value) VALUES (1, "a")')
                conn.execute(f'INSERT INTO "{table_b}"(id, value) VALUES (2, "b")')
                conn.commit()

            out_path = be._sync_sqlite_excel_mirror(db_path, force=True)
            self.assertIsNotNone(out_path)

            wb = load_workbook(str(out_path), read_only=True, data_only=True)
            try:
                self.assertEqual(len(wb.sheetnames), 2)
                self.assertEqual(len(set(wb.sheetnames)), 2)
                self.assertTrue(all(len(str(name or "")) <= 31 for name in wb.sheetnames))
            finally:
                wb.close()

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_metric_bounds_for_run_reads_deferred_run_conditions_sheet(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db), "program_title": be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 1).value = "RunA"
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 3).value = "100 psia, PM, 5 Sec ON / 20 Sec OFF"
                ws_prog.cell(2, 4).value = 100
                ws_prog.cell(2, 5).value = "psia"
                ws_prog.cell(2, 6).value = "PM"
                ws_prog.cell(2, 7).value = 5
                ws_prog.cell(2, 8).value = 20
                wb.save(str(support_path))
            finally:
                wb.close()

            be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True, require_existing_cache=False)

            wb = load_workbook(str(support_path))
            try:
                ws_cond = wb["RunConditions"]
                ws_cond.cell(2, 12).value = 15
                ws_cond.cell(2, 13).value = 45
                wb.save(str(support_path))
            finally:
                wb.close()

            harness = _MetricBoundsHarness(root, wb_path)
            seq_bounds = harness._metric_bounds_for_run("Seq1")
            source_bounds = harness._metric_bounds_for_run("RunA")
            self.assertEqual((seq_bounds.get("thrust") or {}).get("min_value"), 15.0)
            self.assertEqual((seq_bounds.get("thrust") or {}).get("max_value"), 45.0)
            self.assertEqual((source_bounds.get("thrust") or {}).get("min_value"), 15.0)
            self.assertEqual((source_bounds.get("thrust") or {}).get("max_value"), 45.0)

    def test_read_support_workbook_new_schema_groups_by_condition_and_splits_control_period(self) -> None:
        from openpyxl import Workbook, load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            Workbook().save(str(wb_path))
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["Seq1", "Seq2", "Seq3"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
                program_titles=["Program A", "Program B"],
                sequences_by_program={"Program A": ["Seq1", "Seq2"], "Program B": ["Seq3"]},
            )

            wb = load_workbook(str(support_path))
            try:
                ws_programs = wb["Programs"]
                sheet_a = str(ws_programs.cell(2, 2).value or "").strip()
                sheet_b = str(ws_programs.cell(3, 2).value or "").strip()
                ws_a = wb[sheet_a]
                ws_b = wb[sheet_b]
                for ws, row in ((ws_a, 2), (ws_b, 2)):
                    ws.cell(row, 2).value = "CondA"
                    ws.cell(row, 3).value = "350 psia, PM, 60 ON / 120 OFF"
                    ws.cell(row, 4).value = 350
                    ws.cell(row, 5).value = "psia"
                    ws.cell(row, 6).value = "PM"
                    ws.cell(row, 7).value = 60
                    ws.cell(row, 8).value = 120
                ws_a.cell(3, 2).value = "CondA"
                ws_a.cell(3, 3).value = "350 psia, PM, 60 ON / 240 OFF"
                ws_a.cell(3, 4).value = 350
                ws_a.cell(3, 5).value = "psia"
                ws_a.cell(3, 6).value = "PM"
                ws_a.cell(3, 7).value = 60
                ws_a.cell(3, 8).value = 240
                wb.save(str(support_path))
            finally:
                wb.close()

            support_cfg = be._read_td_support_workbook(wb_path, project_dir=root)
            conds = support_cfg.get("run_conditions") or []
            self.assertEqual(len(conds), 2)
            sequences = support_cfg.get("sequences") or []
            cond_keys = {str(row.get("condition_key") or "") for row in sequences}
            self.assertEqual(len(cond_keys), 2)

    def test_run_condition_views_group_default_program_rows_by_condition_tuple(self) -> None:
        from openpyxl import Workbook, load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            Workbook().save(str(wb_path))
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["Seq1", "Seq2"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
                program_titles=["Program A"],
                sequences_by_program={"Program A": ["Seq1", "Seq2"]},
            )

            wb = load_workbook(str(support_path))
            try:
                ws_programs = wb["Programs"]
                sheet_a = str(ws_programs.cell(2, 2).value or "").strip()
                ws_a = wb[sheet_a]
                for row in (2, 3):
                    ws_a.cell(row, 3).value = "350 psia, PM, 60 ON / 120 OFF"
                    ws_a.cell(row, 4).value = 350
                    ws_a.cell(row, 5).value = "psia"
                    ws_a.cell(row, 6).value = "PM"
                    ws_a.cell(row, 7).value = 60
                    ws_a.cell(row, 8).value = 120
                wb.save(str(support_path))
            finally:
                wb.close()

            support_cfg = be._read_td_support_workbook(wb_path, project_dir=root)
            run_conditions = support_cfg.get("run_conditions") or []
            self.assertEqual(len(run_conditions), 1)
            condition_key = str(run_conditions[0].get("condition_key") or "").strip()
            self.assertTrue(condition_key)

            db_path = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_impl_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
                    (condition_key, "Time", "350 psia, PM, 60 ON / 120 OFF", "PM", 120, 60),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN1__Seq1", "SN1", condition_key, "Program A", "Seq1", "PM", 60, 120, 0, 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN2__Seq2", "SN2", condition_key, "Program A", "Seq2", "PM", 60, 120, 0, 0),
                )
                conn.commit()

            views = be.td_list_run_selection_views(db_path, wb_path, project_dir=root)
            cond_items = views.get("condition") or []
            self.assertEqual(len(cond_items), 1)
            self.assertEqual(cond_items[0].get("run_name"), condition_key)
            self.assertEqual(cond_items[0].get("member_sequences"), ["Seq1", "Seq2"])
            self.assertEqual(cond_items[0].get("member_programs"), ["Program A"])

    def test_run_selection_views_normalize_blank_programs_to_unknown_program(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = root / "implementation_trending.sqlite3"
            wb_path = root / "project.xlsx"

            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": ""}],
                config=self._make_config(),
            )

            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_impl_tables(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("SeqBlank", "Time", "SeqBlank", "", None, None),
                )
                conn.commit()

            views = be.td_list_run_selection_views(db_path, wb_path, project_dir=root)
            seq_items = views.get("sequence") or []
            cond_items = views.get("condition") or []
            self.assertEqual(seq_items[0].get("member_programs"), ["Unknown Program"])
            self.assertEqual(cond_items[0].get("member_programs"), ["Unknown Program"])

    def test_run_condition_label_prefers_derived_pm_fields_over_display_name(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        row = {
            "display_name": "sequence",
            "feed_pressure": 350,
            "feed_pressure_units": "psia",
            "run_type": "pulsed mode",
            "pulse_width_on": 60,
            "control_period": 120,
            "condition_key": "cond_a",
        }
        self.assertEqual(
            be._td_effective_run_condition_label(row),
            "350 psia, PM, 60 Sec ON / 120 Sec OFF",
        )

    def test_run_condition_label_prefers_derived_non_pm_fields_over_display_name(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        row = {
            "display_name": "sequence",
            "feed_pressure": 410,
            "feed_pressure_units": "psia",
            "run_type": "steady state",
            "condition_key": "cond_b",
        }
        self.assertEqual(be._td_effective_run_condition_label(row), "410 psia, SS")

    def test_run_selection_views_use_derived_condition_display_name(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = root / "implementation_trending.sqlite3"
            wb_path = root / "project.xlsx"

            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": ""}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["Seq1"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 3).value = "sequence"
                ws_prog.cell(2, 4).value = 350
                ws_prog.cell(2, 5).value = "psia"
                ws_prog.cell(2, 6).value = "pulsed mode"
                ws_prog.cell(2, 7).value = 60
                ws_prog.cell(2, 8).value = 120
                wb.save(str(support_path))
            finally:
                wb.close()

            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_impl_tables(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("Seq1", "Time", "sequence", "pulsed mode", 120, 60),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN1__Seq1", "SN1", "Seq1", be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, "Seq1", "pulsed mode", 60, 120, 0, 0),
                )
                conn.commit()

            views = be.td_list_run_selection_views(db_path, wb_path, project_dir=root)
            cond_items = views.get("condition") or []
            self.assertEqual(len(cond_items), 1)
            self.assertEqual(cond_items[0].get("display_text"), "350 psia, PM, 60 Sec ON / 120 Sec OFF")
            self.assertEqual(cond_items[0].get("run_condition"), "350 psia, PM, 60 Sec ON / 120 Sec OFF")

    def test_rebuild_new_schema_aggregates_across_programs_and_skips_disabled_rows(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db_a = root / "src_a.sqlite3"
            src_db_b = root / "src_b.sqlite3"
            self._make_source_sqlite(src_db_a)
            self._make_source_sqlite(src_db_b)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[
                    {"serial_number": "SN1", "excel_sqlite_rel": str(src_db_b), "program_title": "Program B"},
                    {"serial_number": "SN1", "excel_sqlite_rel": str(src_db_a), "program_title": "Program A"},
                ],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
                program_titles=["Program A", "Program B"],
                sequences_by_program={"Program A": ["RunA"], "Program B": ["RunA"]},
            )

            wb = load_workbook(str(support_path))
            try:
                ws_programs = wb["Programs"]
                sheet_a = str(ws_programs.cell(2, 2).value or "").strip()
                sheet_b = str(ws_programs.cell(3, 2).value or "").strip()
                ws_a = wb[sheet_a]
                ws_b = wb[sheet_b]
                for ws in (ws_a, ws_b):
                    ws.cell(2, 2).value = "CondMerged"
                    ws.cell(2, 3).value = "100 psia, PM, 5 ON / 20 OFF"
                    ws.cell(2, 4).value = 100
                    ws.cell(2, 5).value = "psia"
                    ws.cell(2, 6).value = "PM"
                    ws.cell(2, 7).value = 5
                    ws.cell(2, 8).value = 20
                ws_b.cell(2, 11).value = False
                wb.save(str(support_path))
            finally:
                wb.close()

            out_db = root / "implementation_trending.sqlite3"
            payload = be.rebuild_test_data_project_cache(out_db, wb_path)
            self.assertGreater(int(payload.get("metrics_written") or 0), 0)
            with sqlite3.connect(str(out_db)) as conn:
                run_names = [str(row[0] or "").strip() for row in conn.execute("SELECT run_name FROM td_runs ORDER BY run_name").fetchall()]
                self.assertEqual(run_names, ["CondMerged"])
                obs = conn.execute(
                    "SELECT program_title, source_run_name FROM td_condition_observations WHERE run_name=? AND serial=?",
                    ("CondMerged", "SN1"),
                ).fetchone()
                self.assertIsNotNone(obs)
                self.assertIn("Program A", str(obs[0] or ""))
                self.assertNotIn("Program B", str(obs[0] or ""))

    def test_rebuild_uses_support_sequence_name_without_filtering_parameter_bounds(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore
        from openpyxl import load_workbook  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            from openpyxl import load_workbook  # type: ignore

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 2
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 1).value = "RunA"
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 4).value = 100
                ws_prog.cell(2, 7).value = 5
                wb.save(str(support_path))
            finally:
                wb.close()
            self._refresh_support_conditions(be, wb_path, root)
            wb = load_workbook(str(support_path))
            try:
                ws_cond = wb["RunConditions"]
                ws_cond.cell(2, 12).value = 15
                ws_cond.cell(2, 13).value = 45
                wb.save(str(support_path))
            finally:
                wb.close()

            out_db = root / "implementation_trending.sqlite3"
            payload = be.rebuild_test_data_project_cache(out_db, wb_path)
            self.assertIn("Seq1", payload.get("runs") or [])

            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                row = conn.execute(
                    "SELECT x_json, y_json FROM td_curves_raw WHERE run_name=? AND y_name=? AND x_name=? AND serial=?",
                    ("Seq1", "thrust", "Time", "SN1"),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(json.loads(row[0]), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                self.assertEqual(json.loads(row[1]), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            with sqlite3.connect(str(out_db)) as conn:
                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics_calc WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
                    ("Seq1", "thrust", "mean", "SN1"),
                ).fetchone()
                self.assertIsNotNone(mean_row)
                self.assertAlmostEqual(float(mean_row[0]), 50.0)

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertEqual(values.get("Seq1.thrust.mean"), 50.0)
            finally:
                wb2.close()

    def test_last_n_rows_excludes_final_row_and_stays_consistent_between_rebuild_and_refresh(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            rows = [(idx + 1, float(idx), 100.0, 5.0, float((idx + 1) * 10)) for idx in range(12)]
            self._make_source_sqlite_with_rows(src_db, rows)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 10
                wb.save(str(support_path))
            finally:
                wb.close()

            out_db = root / "implementation_trending.sqlite3"
            be.rebuild_test_data_project_cache(out_db, wb_path)

            with sqlite3.connect(str(out_db)) as conn:
                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics_calc WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
                    ("RunA", "thrust", "mean", "SN1"),
                ).fetchone()
            self.assertIsNotNone(mean_row)
            self.assertAlmostEqual(float(mean_row[0]), 70.0)

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            with sqlite3.connect(str(out_db)) as conn:
                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics_calc WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
                    ("RunA", "thrust", "mean", "SN1"),
                ).fetchone()
            self.assertIsNotNone(mean_row)
            self.assertAlmostEqual(float(mean_row[0]), 70.0)

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertEqual(values.get("RunA.thrust.mean"), 70.0)
            finally:
                wb2.close()

    def test_last_n_rows_of_one_yields_empty_stats(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 1
                wb.save(str(support_path))
            finally:
                wb.close()

            out_db = root / "implementation_trending.sqlite3"
            be.rebuild_test_data_project_cache(out_db, wb_path)

            with sqlite3.connect(str(out_db)) as conn:
                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics_calc WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
                    ("RunA", "thrust", "mean", "SN1"),
                ).fetchone()
            self.assertIsNotNone(mean_row)
            self.assertIsNone(mean_row[0])

    def test_metric_bound_line_specs_use_red_lines_for_enabled_bounds(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        self.assertEqual(
            be.td_metric_bound_line_specs({"enabled": True, "min_value": 10.0, "max_value": 20.0}),
            [
                {"value": 10.0, "color": "red", "linestyle": "--", "alpha": 0.8, "linewidth": 1.2},
                {"value": 20.0, "color": "red", "linestyle": "--", "alpha": 0.8, "linewidth": 1.2},
            ],
        )
        self.assertEqual(
            be.td_metric_bound_line_specs({"enabled": True, "min_value": 10.0, "max_value": None}),
            [
                {"value": 10.0, "color": "red", "linestyle": "--", "alpha": 0.8, "linewidth": 1.2},
            ],
        )
        self.assertEqual(be.td_metric_bound_line_specs({"enabled": False, "min_value": 10.0, "max_value": 20.0}), [])
        self.assertEqual(be.td_metric_bound_line_specs({}), [])

    def test_run_selection_views_group_exact_run_conditions(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = root / "implementation_trending.sqlite3"
            wb_path = root / "project.xlsx"

            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": ""}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["Seq1", "Seq2", "Seq3"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 1).value = "Seq1"
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 3).value = "350 psia, SS"
                ws_prog.cell(2, 4).value = 350
                ws_prog.cell(2, 5).value = "psia"
                ws_prog.cell(2, 6).value = "steady state"
                ws_prog.cell(3, 1).value = "Seq2"
                ws_prog.cell(3, 2).value = "Seq1"
                ws_prog.cell(3, 3).value = "350 psia, SS"
                ws_prog.cell(3, 4).value = 350
                ws_prog.cell(3, 5).value = "psia"
                ws_prog.cell(3, 6).value = "steady state"
                ws_prog.cell(4, 1).value = "Seq3"
                ws_prog.cell(4, 2).value = "Seq3"
                ws_prog.cell(4, 3).value = "350 psia, PM, 60 Sec ON / 120 Sec OFF"
                ws_prog.cell(4, 4).value = 350
                ws_prog.cell(4, 5).value = "psia"
                ws_prog.cell(4, 6).value = "pulsed mode"
                ws_prog.cell(4, 7).value = 60
                ws_prog.cell(4, 8).value = 120
                wb.save(str(support_path))
            finally:
                wb.close()

            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_impl_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
                    ("Seq1", "Time", "350 psia, SS", "steady state", None, None),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
                    ("Seq3", "Time", "350 psia, PM, 60 Sec ON / 120 Sec OFF", "pulsed mode", 120, 60),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN1__Seq1", "SN1", "Seq1", be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, "Seq1", "steady state", None, None, 0, 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN1__Seq2", "SN1", "Seq1", be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, "Seq2", "steady state", None, None, 0, 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_condition_observations
                    (observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN1__Seq3", "SN1", "Seq3", be.TD_SUPPORT_DEFAULT_PROGRAM_TITLE, "Seq3", "pulsed mode", 60, 120, 0, 0),
                )
                conn.commit()

            views = be.td_list_run_selection_views(db_path, wb_path, project_dir=root)
            seq_items = views.get("sequence") or []
            cond_items = views.get("condition") or []

            self.assertEqual(
                [item.get("display_text") for item in seq_items],
                ["Default Program - Seq3", "Default Program - Seq1", "Default Program - Seq2"],
            )
            self.assertEqual([item.get("display_text") for item in cond_items], ["350 psia, PM, 60 Sec ON / 120 Sec OFF", "350 psia, SS"])
            ss_group = next(item for item in cond_items if item.get("display_text") == "350 psia, SS")
            self.assertEqual(ss_group.get("member_runs"), ["Seq1"])
            self.assertEqual(ss_group.get("member_sequences"), ["Seq1", "Seq2"])

    def test_rebuild_prefers_workbook_config_columns_over_runtime_config(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            with mock.patch.object(
                be,
                "_load_runtime_td_trend_config",
                return_value={
                    "config": {"columns": [{"name": "bogus", "units": ""}], "statistics": ["mean"]},
                    "columns": [{"name": "bogus", "units": ""}],
                    "statistics": ["mean"],
                    "path": "bogus.json",
                    "fallback_used": False,
                },
            ):
                payload = be.rebuild_test_data_project_cache(root / "implementation_trending.sqlite3", wb_path)

            self.assertGreater(int(payload.get("curves_written") or 0), 0)
            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM td_curves_raw
                    WHERE run_name=? AND y_name=? AND x_name=? AND serial=?
                    """,
                    ("RunA", "thrust", "Time", "SN1"),
                ).fetchone()
            self.assertEqual(int(row[0] or 0), 1)

    def test_update_workbook_adds_support_named_rows(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 4).value = 100
                ws_prog.cell(2, 7).value = 5
                wb.save(str(support_path))
            finally:
                wb.close()

            be.rebuild_test_data_project_cache(root / "implementation_trending.sqlite3", wb_path)
            src_db.unlink()
            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                metrics = [str(ws_calc.cell(r, 1).value or "").strip() for r in range(1, (ws_calc.max_row or 0) + 1)]
                self.assertIn("Seq1.thrust.mean", metrics)
                self.assertNotIn("Data", wb2.sheetnames)
            finally:
                wb2.close()

    def test_rebuild_surfaces_support_pulse_width_as_trend_metric(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 2).value = "Seq1"
                ws_prog.cell(2, 7).value = 5
                wb.save(str(support_path))
            finally:
                wb.close()

            be.rebuild_test_data_project_cache(root / "implementation_trending.sqlite3", wb_path)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                cols = {
                    str(row[0] or "").strip()
                    for row in conn.execute(
                        "SELECT name FROM td_columns_calc WHERE run_name=? AND kind='y' ORDER BY name",
                        ("Seq1",),
                    ).fetchall()
                }
                self.assertIn("pulse_width", cols)
                pulse_rows = conn.execute(
                    """
                    SELECT stat, value_num
                    FROM td_metrics_calc
                    WHERE run_name=? AND column_name=?
                    ORDER BY stat
                    """,
                    ("Seq1", "pulse_width"),
                ).fetchall()
            self.assertEqual(pulse_rows, [("max", 5.0), ("mean", 5.0), ("min", 5.0), ("std", 0.0)])

            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                raw_row = conn.execute(
                    "SELECT pulse_width FROM td_raw_sequences WHERE run_name=?",
                    ("Seq1",),
                ).fetchone()
            self.assertIsNotNone(raw_row)
            self.assertAlmostEqual(float(raw_row[0]), 5.0)

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertEqual(values.get("Seq1.pulse_width.mean"), 5.0)
                self.assertEqual(values.get("Seq1.pulse_width.std"), 0.0)
            finally:
                wb2.close()

    def test_rebuild_maps_legacy_pulse_width_on_to_canonical_pulse_width(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            wb = load_workbook(str(support_path))
            try:
                ws_prog = wb[self._default_program_sheet_name(be)]
                ws_prog.cell(2, 7).value = 7
                wb.save(str(support_path))
            finally:
                wb.close()

            be.rebuild_test_data_project_cache(root / "implementation_trending.sqlite3", wb_path)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                cols = {
                    str(row[0] or "").strip()
                    for row in conn.execute(
                        "SELECT name FROM td_columns_calc WHERE run_name=? AND kind='y' ORDER BY name",
                        ("RunA",),
                    ).fetchall()
                }
                pulse_rows = conn.execute(
                    """
                    SELECT stat, value_num
                    FROM td_metrics_calc
                    WHERE run_name=? AND column_name=?
                    ORDER BY stat
                    """,
                    ("RunA", "pulse_width"),
                ).fetchall()
            self.assertIn("pulse_width", cols)
            self.assertNotIn("pulse_width_on", cols)
            self.assertEqual(pulse_rows, [("max", 7.0), ("mean", 7.0), ("min", 7.0), ("std", 0.0)])

    def test_rebuild_skips_synthetic_pulse_width_when_support_value_missing(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            be.rebuild_test_data_project_cache(root / "implementation_trending.sqlite3", wb_path)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                cols = {
                    str(row[0] or "").strip()
                    for row in conn.execute(
                        "SELECT name FROM td_columns_calc WHERE run_name=? AND kind='y' ORDER BY name",
                        ("RunA",),
                    ).fetchall()
                }
                pulse_rows = conn.execute(
                    "SELECT COUNT(*) FROM td_metrics_calc WHERE run_name=? AND column_name=?",
                    ("RunA", "pulse_width"),
                ).fetchone()
            self.assertNotIn("pulse_width", cols)
            self.assertEqual(int(pulse_rows[0] or 0), 0)

    def test_update_workbook_rebuilds_cache_after_support_change(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 2
                wb.save(str(support_path))
            finally:
                wb.close()

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics_calc WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
                    ("RunA", "thrust", "mean", "SN1"),
                ).fetchone()
            self.assertIsNotNone(mean_row)
            self.assertAlmostEqual(float(mean_row[0]), 50.0)

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertEqual(values.get("RunA.thrust.mean"), 50.0)
            finally:
                wb2.close()

    def test_update_sync_support_workbook_adds_new_program_sheet(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
                program_titles=["Program Alpha"],
                sequences_by_program={"Program Alpha": ["RunA"]},
            )

            docs = [
                {"metadata_rel": "alpha.json", "program_title": "Program Alpha"},
                {"metadata_rel": "beta.json", "program_title": "Program Beta"},
            ]
            with mock.patch.object(be, "read_eidat_index_documents", return_value=docs):
                with mock.patch.object(be, "_project_selected_metadata_rels", return_value={"alpha.json", "beta.json"}):
                    with mock.patch.object(be, "_discover_td_runs_for_docs", return_value=["RunA", "RunB"]):
                        with mock.patch.object(
                            be,
                            "_discover_td_runs_by_program_for_docs",
                            return_value={"Program Alpha": ["RunA"], "Program Beta": ["RunB"]},
                        ):
                            result = be._sync_td_support_workbook_program_sheets(
                                wb_path,
                                global_repo=root,
                                project_dir=root,
                                param_defs=[{"name": "thrust", "units": "lbf"}],
                            )

            self.assertTrue(bool(result.get("updated")))
            support_cfg = be._read_td_support_workbook(wb_path, project_dir=root)
            program_titles = {
                str(row.get("program_title") or "").strip()
                for row in (support_cfg.get("programs") or [])
                if isinstance(row, dict)
            }
            self.assertIn("Program Alpha", program_titles)
            self.assertIn("Program Beta", program_titles)
            beta_rows = support_cfg.get("program_mappings", {}).get("Program Beta") or []
            self.assertEqual(
                [str(row.get("source_run_name") or "").strip() for row in beta_rows],
                ["RunB"],
            )

    def test_ensure_cache_uses_calc_only_refresh_after_support_change(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 3
                wb.save(str(support_path))
            finally:
                wb.close()

            with mock.patch.object(be, "rebuild_test_data_project_cache") as rebuild_mock:
                with mock.patch.object(be, "_rebuild_test_data_project_calc_cache_from_raw", return_value={}) as calc_mock:
                    db_path = be.ensure_test_data_project_cache(root, wb_path, rebuild=False)

            self.assertEqual(db_path, root / "implementation_trending.sqlite3")
            rebuild_mock.assert_not_called()
            calc_mock.assert_called_once_with(root / "implementation_trending.sqlite3", wb_path, progress_cb=None)

    def test_validate_existing_cache_allows_calc_only_staleness_after_support_change(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 3
                wb.save(str(support_path))
            finally:
                wb.close()

            validated = be.validate_existing_test_data_project_cache(root, wb_path)
            readiness = be._td_collect_project_readiness(root, wb_path)

            self.assertEqual(str(validated), str(root / "implementation_trending.sqlite3"))
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(
                any("support workbook changed" in str(warning).lower() for warning in (readiness.get("warnings") or []))
            )

    def test_ensure_cache_uses_calc_only_refresh_after_statistics_change(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            cfg = self._make_config()
            cfg["statistics"] = ["mean", "max"]
            with mock.patch.object(be, "_load_project_td_trend_config", return_value=cfg):
                with mock.patch.object(be, "rebuild_test_data_project_cache") as rebuild_mock:
                    with mock.patch.object(be, "_rebuild_test_data_project_calc_cache_from_raw", return_value={}) as calc_mock:
                        db_path = be.ensure_test_data_project_cache(root, wb_path, rebuild=False)

            self.assertEqual(db_path, root / "implementation_trending.sqlite3")
            rebuild_mock.assert_not_called()
            calc_mock.assert_called_once_with(root / "implementation_trending.sqlite3", wb_path, progress_cb=None)

    def test_sync_cache_uses_calc_only_refresh_when_impl_cache_is_incomplete(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                conn.execute("DELETE FROM td_runs")
                conn.execute("DELETE FROM td_columns_calc")
                conn.execute("DELETE FROM td_metrics_calc")
                conn.execute("DELETE FROM td_condition_observations")
                conn.commit()

            with mock.patch.object(be, "rebuild_test_data_project_cache") as rebuild_mock:
                with mock.patch.object(be, "_rebuild_test_data_project_calc_cache_from_raw", return_value={"db_path": str(root / "implementation_trending.sqlite3")}) as calc_mock:
                    payload = be.sync_test_data_project_cache(root, wb_path)

            self.assertEqual(str(payload.get("mode") or ""), "calc_only")
            rebuild_mock.assert_not_called()
            calc_mock.assert_called_once_with(root / "implementation_trending.sqlite3", wb_path, progress_cb=None)

    def test_ensure_cache_rebuilds_raw_cache_after_raw_column_change(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            cfg = self._make_config()
            cfg["columns"] = [
                {"name": "thrust", "units": "lbf", "range_min": None, "range_max": None},
                {"name": "feed pressure", "units": "psi", "range_min": None, "range_max": None},
            ]
            with mock.patch.object(be, "_load_project_td_trend_config", return_value=cfg):
                with mock.patch.object(be, "rebuild_test_data_project_cache", return_value={}) as rebuild_mock:
                    with mock.patch.object(be, "_rebuild_test_data_project_calc_cache_from_raw") as calc_mock:
                        db_path = be.ensure_test_data_project_cache(root, wb_path, rebuild=False)

            self.assertEqual(db_path, root / "implementation_trending.sqlite3")
            rebuild_mock.assert_called_once_with(root / "implementation_trending.sqlite3", wb_path, progress_cb=None)
            calc_mock.assert_not_called()

    def test_full_rebuild_avoids_reloading_raw_cache_for_calc(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            with mock.patch.object(be, "_rebuild_test_data_project_calc_cache_from_raw") as calc_from_raw_mock:
                payload = be.rebuild_test_data_project_cache(root / "implementation_trending.sqlite3", wb_path)

            self.assertGreater(int(payload.get("metrics_written") or 0), 0)
            calc_from_raw_mock.assert_not_called()

    def test_sync_cache_noops_when_inputs_are_unchanged(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            with mock.patch.object(be, "rebuild_test_data_project_cache") as rebuild_mock:
                with mock.patch.object(be, "_rebuild_test_data_project_calc_cache_from_raw") as calc_mock:
                    payload = be.sync_test_data_project_cache(root, wb_path)

            self.assertEqual(str(payload.get("mode") or ""), "noop")
            rebuild_mock.assert_not_called()
            calc_mock.assert_not_called()

    def test_sync_cache_incremental_reingests_only_changed_serial(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db_1 = root / "src1.sqlite3"
            src_db_2 = root / "src2.sqlite3"
            self._make_source_sqlite(src_db_1)
            self._make_source_sqlite(src_db_2)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1", "SN2"],
                docs=[
                    {"serial_number": "SN1", "excel_sqlite_rel": str(src_db_1)},
                    {"serial_number": "SN2", "excel_sqlite_rel": str(src_db_2)},
                ],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                before = {
                    str(serial or ""): int(epoch or 0)
                    for serial, epoch in conn.execute(
                        "SELECT serial, last_ingested_epoch_ns FROM td_sources ORDER BY serial"
                    ).fetchall()
                }

            time.sleep(0.05)
            src_db_1.unlink()
            self._make_source_sqlite_with_rows(
                src_db_1,
                [
                    (1, 0.0, 100.0, 5.0, 11.0),
                    (2, 1.0, 100.0, 5.0, 22.0),
                    (3, 2.0, 100.0, 5.0, 33.0),
                    (4, 3.0, 100.0, 5.0, 44.0),
                    (5, 4.0, 100.0, 5.0, 55.0),
                    (6, 5.0, 100.0, 5.0, 66.0),
                ],
            )

            payload = be.sync_test_data_project_cache(root, wb_path)
            self.assertEqual(str(payload.get("mode") or ""), "incremental_raw")
            counts = payload.get("counts") or {}
            self.assertEqual(int(counts.get("changed") or 0), 1)
            self.assertEqual(int(counts.get("reingested") or 0), 1)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                after = {
                    str(serial or ""): int(epoch or 0)
                    for serial, epoch in conn.execute(
                        "SELECT serial, last_ingested_epoch_ns FROM td_sources ORDER BY serial"
                    ).fetchall()
                }
            self.assertGreater(after["SN1"], before["SN1"])
            self.assertEqual(after["SN2"], before["SN2"])

    def test_sync_cache_routes_added_serials_through_incremental_rebuild(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db_1 = root / "src1.sqlite3"
            src_db_2 = root / "src2.sqlite3"
            self._make_source_sqlite(src_db_1)
            self._make_source_sqlite(src_db_2)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db_1)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            wb = load_workbook(str(wb_path))
            try:
                ws = wb["Sources"]
                ws.append(["SN2", "", "", "", "", str(src_db_2)])
                wb.save(str(wb_path))
            finally:
                wb.close()

            with mock.patch.object(be, "rebuild_test_data_project_cache", return_value={"db_path": str(root / "implementation_trending.sqlite3")}) as rebuild_mock:
                payload = be.sync_test_data_project_cache(root, wb_path)

            self.assertEqual(str(payload.get("mode") or ""), "incremental_raw")
            self.assertEqual(int((payload.get("counts") or {}).get("added") or 0), 1)
            kwargs = rebuild_mock.call_args.kwargs
            self.assertEqual(kwargs.get("_full_reset"), False)
            entries_override = kwargs.get("_entries_override") or []
            self.assertEqual([str(item.get("serial") or "") for item in entries_override], ["SN2"])

    def test_sync_cache_prunes_removed_serial_from_raw_and_impl(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db_1 = root / "src1.sqlite3"
            src_db_2 = root / "src2.sqlite3"
            self._make_source_sqlite(src_db_1)
            self._make_source_sqlite(src_db_2)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1", "SN2"],
                docs=[
                    {"serial_number": "SN1", "excel_sqlite_rel": str(src_db_1)},
                    {"serial_number": "SN2", "excel_sqlite_rel": str(src_db_2)},
                ],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            wb = load_workbook(str(wb_path))
            try:
                ws = wb["Sources"]
                ws.delete_rows(3, 1)
                wb.save(str(wb_path))
            finally:
                wb.close()

            payload = be.sync_test_data_project_cache(root, wb_path)
            self.assertEqual(str(payload.get("mode") or ""), "incremental_raw")
            counts = payload.get("counts") or {}
            self.assertEqual(int(counts.get("removed") or 0), 1)

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                serials = [
                    str(row[0] or "")
                    for row in conn.execute("SELECT serial FROM td_sources ORDER BY serial").fetchall()
                ]
            self.assertEqual(serials, ["SN1"])

            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                raw_serials = [
                    str(row[0] or "")
                    for row in conn.execute(
                        "SELECT DISTINCT serial FROM td_curves_raw ORDER BY serial"
                    ).fetchall()
                ]
            self.assertEqual(raw_serials, ["SN1"])

    def test_validate_existing_cache_hard_fails_when_source_changes(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            time.sleep(0.05)
            src_db.unlink()
            self._make_source_sqlite_with_rows(
                src_db,
                [
                    (1, 0.0, 100.0, 5.0, 9.0),
                    (2, 1.0, 100.0, 5.0, 19.0),
                    (3, 2.0, 100.0, 5.0, 29.0),
                    (4, 3.0, 100.0, 5.0, 39.0),
                    (5, 4.0, 100.0, 5.0, 49.0),
                    (6, 5.0, 100.0, 5.0, 59.0),
                ],
            )

            with self.assertRaisesRegex(RuntimeError, "Project cache is stale"):
                be.validate_existing_test_data_project_cache(root, wb_path)

    def test_update_workbook_builds_missing_cache_db(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))
            self.assertTrue((root / "implementation_trending.sqlite3").exists())

            wb = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertIn("RunA.thrust.mean", values)
            finally:
                wb.close()

    def test_update_workbook_skips_performance_sheets_by_default_and_reports_stage_timings(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            progress: list[str] = []
            result = be.update_test_data_trending_project_workbook(
                root,
                wb_path,
                overwrite=True,
                progress_cb=progress.append,
            )

            timings = result.get("timings") or {}
            for key in (
                "data_calc_build_s",
                "metrics_long_sheet_s",
                "raw_cache_long_sheet_s",
                "perf_candidates_main_s",
                "perf_candidates_cp_total_s",
                "perf_candidates_cp_count",
                "metadata_sync_s",
                "post_cache_workbook_build_s",
            ):
                self.assertIn(key, timings)
            self.assertEqual(timings.get("perf_candidates_main_s"), 0.0)
            self.assertEqual(timings.get("perf_candidates_cp_total_s"), 0.0)
            self.assertEqual(timings.get("perf_candidates_cp_count"), 0)

            wb = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                self.assertNotIn("Performance_candidates", wb.sheetnames)
                self.assertFalse(any(str(name).startswith("Performance_candidates_CP_") for name in wb.sheetnames))
            finally:
                wb.close()

            expected_stages = [
                "Refreshing support workbook",
                "Saving workbook before cache validation",
                "Ensuring project cache",
                "Reading project cache",
                "Rebuilding Data_calc",
                "Writing Metrics_long",
                "Writing RawCache_long",
                "Syncing workbook metadata",
                "Saving updated workbook",
            ]
            indices = [next(i for i, msg in enumerate(progress) if stage in msg) for stage in expected_stages]
            self.assertEqual(indices, sorted(indices))

    def test_generate_performance_sheets_writes_main_and_cp_sheets(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "Seq1", 3.0, 10.0),
                    ("SN1", "Seq2", 4.0, 11.0),
                    ("SN1", "Seq3", 5.0, 12.0),
                ],
            )
            wb_path = root / "project.xlsx"

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE td_condition_observations SET run_type=?, control_period=?",
                    ("pm", 120.0),
                )
                conn.commit()

            progress: list[str] = []
            result = be.generate_test_data_project_performance_sheets(
                root,
                wb_path,
                progress_cb=progress.append,
            )

            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))
            timings = result.get("timings") or {}
            self.assertGreaterEqual(int(timings.get("perf_candidates_cp_count") or 0), 1)

            wb = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                self.assertIn("Performance_candidates", wb.sheetnames)
                self.assertTrue(any(str(name).startswith("Performance_candidates_CP_120") for name in wb.sheetnames))
                ws = wb["Performance_candidates"]
                self.assertGreater(int(ws.max_row or 0), 1)
            finally:
                wb.close()

            self.assertTrue(any("Generating performance candidate sheets" in msg for msg in progress))
            self.assertTrue(any("Generating control-period performance sheets" in msg for msg in progress))
            self.assertTrue(any("Saving workbook with performance sheets" in msg for msg in progress))

    def test_update_workbook_repairs_incomplete_existing_cache_db(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_tables(conn)
                conn.commit()

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))
            validated = be.validate_existing_test_data_project_cache(root, wb_path)
            self.assertEqual(str(validated), str(db_path))

            wb = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertIn("RunA.thrust.mean", values)
            finally:
                wb.close()

    def test_update_workbook_repairs_schema_only_raw_and_impl_dbs(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            with sqlite3.connect(str(root / "implementation_trending.sqlite3")) as conn:
                be._ensure_test_data_tables(conn)
                conn.commit()
            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                be._ensure_test_data_raw_cache_tables(conn)
                conn.commit()

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))
            self.assertTrue((root / "test_data_raw_cache.sqlite3").exists())
            validated = be.validate_existing_test_data_project_cache(root, wb_path)
            self.assertEqual(str(validated), str(root / "implementation_trending.sqlite3"))

    def test_update_workbook_returns_final_cache_validation_payload_and_debug_path(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)

            self.assertTrue(bool(result.get("cache_validation_ok")))
            self.assertEqual(str(result.get("cache_validation_error") or ""), "")
            self.assertTrue(str(result.get("cache_validation_summary") or "").strip())
            self.assertEqual(str(result.get("backend_module_path") or ""), str(Path(be.__file__).resolve()))
            cache_state = result.get("cache_state") if isinstance(result.get("cache_state"), dict) else {}
            impl_counts = cache_state.get("impl_counts") if isinstance(cache_state.get("impl_counts"), dict) else {}
            raw_counts = cache_state.get("raw_counts") if isinstance(cache_state.get("raw_counts"), dict) else {}
            self.assertGreater(int(impl_counts.get("td_runs") or 0), 0)
            self.assertGreater(int(impl_counts.get("td_metrics_calc") or 0), 0)
            self.assertGreater(int(raw_counts.get("td_curves_raw") or 0), 0)

            debug_path = Path(str(result.get("cache_debug_path") or ""))
            self.assertTrue(debug_path.exists())
            debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
            post_build = debug_payload.get("post_build_validation") if isinstance(debug_payload, dict) else {}
            self.assertIsInstance(post_build, dict)
            self.assertTrue(bool(post_build.get("ok")))
            self.assertEqual(
                int((((post_build.get("cache_state") or {}).get("impl_counts") or {}).get("td_runs") or 0)),
                int(impl_counts.get("td_runs") or 0),
            )

    def test_update_workbook_raises_when_post_build_validation_fails(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            with mock.patch.object(
                be,
                "validate_existing_test_data_project_cache",
                side_effect=RuntimeError("Project cache DB is incomplete: synthetic failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "final TD cache validation failed"):
                    be.update_test_data_trending_project_workbook(
                        root,
                        wb_path,
                        overwrite=True,
                        require_existing_cache=False,
                    )

            debug_path = root / be.TD_CACHE_DEBUG_JSON
            self.assertTrue(debug_path.exists())
            debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
            post_build = debug_payload.get("post_build_validation") if isinstance(debug_payload, dict) else {}
            self.assertIsInstance(post_build, dict)
            self.assertFalse(bool(post_build.get("ok")))
            self.assertIn("synthetic failure", str(post_build.get("error") or ""))

    def test_validate_existing_cache_requires_built_raw_and_calc_sections(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )

            with self.assertRaisesRegex(RuntimeError, "Project cache DB not found|Project raw cache DB not found"):
                be.validate_existing_test_data_project_cache(root, wb_path)

            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

            validated = be.validate_existing_test_data_project_cache(root, wb_path)
            self.assertEqual(str(validated), str(root / "implementation_trending.sqlite3"))

    def test_td_list_x_columns_falls_back_to_raw_curves(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = root / "test_data_raw_cache.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_raw_cache_tables(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_raw_sequences(run_name, display_name, x_axis_kind, source_run_name, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("RunA", "", "Time", "RunA", 1),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_raw_curve_catalog
                    (run_name, parameter_name, units, x_axis_kind, table_name, display_name, source_kind, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "lbf", "Time", "td_raw__runa__thrust", "", "source_sqlite", 1),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_raw_curve_catalog
                    (run_name, parameter_name, units, x_axis_kind, table_name, display_name, source_kind, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust_pulse", "lbf", "Pulse Number", "td_raw__runa__thrust_pulse", "", "source_sqlite", 1),
                )
                conn.commit()

            xs = be.td_list_x_columns(db_path, "RunA")
            self.assertEqual(xs, ["Time", "Pulse Number"])

    def test_td_load_curves_reads_materialized_raw_curve_table(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = root / "test_data_raw_cache.sqlite3"
            table_name = "td_raw__runa__thrust"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_raw_cache_tables(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_raw_curve_catalog
                    (run_name, parameter_name, units, x_axis_kind, table_name, display_name, source_kind, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "lbf", "Time", table_name, "", "source_sqlite", 1),
                )
                conn.execute(
                    f"""
                    CREATE TABLE {be._quote_ident(table_name)} (
                        serial TEXT PRIMARY KEY,
                        x_json TEXT NOT NULL,
                        y_json TEXT NOT NULL,
                        n_points INTEGER NOT NULL,
                        source_mtime_ns INTEGER,
                        computed_epoch_ns INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {be._quote_ident(table_name)}
                    (serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("SN1", "[0,1,2]", "[10,20,30]", 3, 1, 1),
                )
                conn.commit()

            curves = be.td_load_curves(db_path, "RunA", "thrust", "Time")
            self.assertEqual(len(curves), 1)
            self.assertEqual(curves[0].get("serial"), "SN1")
            self.assertEqual(curves[0].get("x"), [0, 1, 2])
            self.assertEqual(curves[0].get("y"), [10, 20, 30])
            self.assertTrue(str(curves[0].get("observation_id") or "").strip())

    def test_debug_export_writes_excel_mirror_for_project_cache(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = be.ensure_test_data_project_cache(root, wb_path, rebuild=True)
            impl_mirror_path = db_path.with_suffix(".xlsx")
            raw_mirror_path = (root / "test_data_raw_cache.sqlite3").with_suffix(".xlsx")
            raw_points_path = root / "test_data_raw_points.xlsx"
            self.assertFalse(impl_mirror_path.exists())
            self.assertFalse(raw_mirror_path.exists())
            self.assertFalse(raw_points_path.exists())

            generated = be.export_test_data_project_debug_excels(root, wb_path, force=True)
            self.assertTrue(impl_mirror_path.exists())
            self.assertTrue(raw_mirror_path.exists())
            self.assertTrue(raw_points_path.exists())
            self.assertEqual(generated["implementation_excel"], impl_mirror_path)
            self.assertEqual(generated["raw_cache_excel"], raw_mirror_path)
            self.assertEqual(generated["raw_points_excel"], raw_points_path)

            wb = load_workbook(str(impl_mirror_path), read_only=True, data_only=True)
            try:
                self.assertIn("td_metrics_calc", wb.sheetnames)
            finally:
                wb.close()
            wb = load_workbook(str(raw_mirror_path), read_only=True, data_only=True)
            try:
                self.assertIn("td_curves_raw", wb.sheetnames)
            finally:
                wb.close()
            wb = load_workbook(str(raw_points_path), read_only=True, data_only=True)
            try:
                self.assertIn("RunA__thrust__Time", wb.sheetnames)
                ws = wb["RunA__thrust__Time"]
                self.assertEqual(ws.cell(1, 1).value, "Time")
                self.assertEqual(ws.cell(1, 2).value, "SN1")
                self.assertEqual(ws.cell(2, 1).value, 0)
                self.assertEqual(ws.cell(2, 2).value, 10)
            finally:
                wb.close()

    def test_calc_cache_can_be_rebuilt_from_raw_db_without_source_sqlite(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": "missing.sqlite3"}],
                config=self._make_config(),
            )
            wb = load_workbook(str(wb_path))
            try:
                ws_cfg = wb["Config"]
                for row in range(2, (ws_cfg.max_row or 0) + 1):
                    if str(ws_cfg.cell(row, 1).value or "").strip().lower() == "statistics":
                        ws_cfg.cell(row, 2).value = "mean, max"
                        break
                wb.save(str(wb_path))
            finally:
                wb.close()
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = root / "implementation_trending.sqlite3"
            raw_db_path = root / "test_data_raw_cache.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_impl_tables(conn)
                conn.commit()
            with sqlite3.connect(str(raw_db_path)) as conn:
                be._ensure_test_data_raw_cache_tables(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_raw_sequences(run_name, display_name, x_axis_kind, source_run_name, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("RunA", "", "Time", "RunA", 123),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("RunA", "Time", "", "x"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("RunA", "thrust", "lbf", "y"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_curves_raw
                    (run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1__RunA", "SN1", "[0,1,2,3]", "[10,20,30,40]", 4, 123, 123, "", "RunA"),
                )
                conn.commit()

            with mock.patch.object(
                be,
                "_load_runtime_td_trend_config",
                return_value={"config": {}, "columns": [{"name": "thrust", "units": "lbf"}], "statistics": ["mean", "max"]},
            ):
                payload = be._rebuild_test_data_project_calc_cache_from_raw(db_path, wb_path)

            self.assertEqual(payload.get("mode"), "calc_from_raw")
            with sqlite3.connect(str(db_path)) as conn:
                stats_meta = conn.execute("SELECT value FROM td_meta WHERE key='statistics' LIMIT 1").fetchone()
                rows = conn.execute(
                    """
                    SELECT stat, value_num
                    FROM td_metrics_calc
                    WHERE serial='SN1' AND run_name='RunA' AND column_name='thrust'
                    ORDER BY stat
                    """
                ).fetchall()
            self.assertIn("std", str((stats_meta[0] if stats_meta else "") or ""))
            self.assertEqual([str(r[0] or "") for r in rows], ["max", "mean", "std"])
            self.assertEqual(float(rows[0][1]), 30.0)
            self.assertEqual(float(rows[1][1]), 20.0)
            self.assertTrue(isinstance(rows[2][1], (int, float)))
            self.assertGreater(float(rows[2][1]), 0.0)

    def test_calc_cache_still_writes_mean_when_config_omits_it(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            cfg = self._make_config()
            cfg["statistics"] = ["max"]
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": ""}],
                config=cfg,
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_tables(conn)
                conn.commit()
            raw_db_path = root / "test_data_raw_cache.sqlite3"
            with sqlite3.connect(str(raw_db_path)) as conn:
                be._ensure_test_data_raw_cache_tables(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_raw_sequences(run_name, display_name, x_axis_kind, source_run_name, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("RunA", "", "Time", "RunA", 123),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("RunA", "Time", "", "x"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("RunA", "thrust", "lbf", "y"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_curves_raw
                    (run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1__RunA", "SN1", "[0,1,2,3]", "[10,20,30,40]", 4, 123, 123, "", "RunA"),
                )
                conn.commit()

            with mock.patch.object(
                be,
                "_load_runtime_td_trend_config",
                return_value={"config": {}, "columns": [{"name": "thrust", "units": "lbf"}], "statistics": ["max"]},
            ):
                be._rebuild_test_data_project_calc_cache_from_raw(db_path, wb_path)

            with sqlite3.connect(str(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT stat, value_num
                    FROM td_metrics_calc
                    WHERE serial='SN1' AND run_name='RunA' AND column_name='thrust'
                    ORDER BY stat
                    """
                ).fetchall()
            self.assertEqual([str(r[0] or "") for r in rows], ["max", "mean", "std"])
            self.assertEqual(float(rows[0][1]), 30.0)
            self.assertEqual(float(rows[1][1]), 20.0)
            self.assertTrue(isinstance(rows[2][1], (int, float)))
            self.assertGreater(float(rows[2][1]), 0.0)

    def test_metric_plot_values_leave_mean_as_per_serial_values(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        series_rows = [
            {"serial": "SN1", "value_num": 10.0},
            {"serial": "SN2", "value_num": 20.0},
            {"serial": "SN3", "value_num": 40.0},
        ]
        serials = ["SN1", "SN2", "SN3", "SN4"]

        values_mean = be.td_metric_plot_values(series_rows, serials, "mean")
        self.assertEqual(values_mean[:3], [10.0, 20.0, 40.0])
        self.assertTrue(math.isnan(values_mean[3]))
        values_max = be.td_metric_plot_values(series_rows, serials, "max")
        self.assertEqual(values_max[:3], [10.0, 20.0, 40.0])
        self.assertTrue(math.isnan(values_max[3]))

    def test_metric_average_plot_values_use_overall_average_of_mean_points(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        series_rows = [
            {"serial": "SN1", "value_num": 10.0},
            {"serial": "SN2", "value_num": 20.0},
            {"serial": "SN3", "value_num": 40.0},
        ]
        serials = ["SN1", "SN2", "SN3", "SN4"]

        self.assertEqual(
            be.td_metric_average_plot_values(series_rows, serials),
            [23.333333333333332, 23.333333333333332, 23.333333333333332, 23.333333333333332],
        )

    def test_calc_cache_hard_fails_when_raw_only_exists_in_implementation_db(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": "missing.sqlite3"}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name) VALUES (?, ?, ?)",
                    ("RunA", "Time", ""),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("RunA", "Time", "", "x"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("RunA", "thrust", "lbf", "y"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_curves_raw
                    (run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1__RunA", "SN1", "[0,1,2,3]", "[10,20,30,40]", 4, 123, 123, "", "RunA"),
                )
                conn.commit()

            with self.assertRaisesRegex(RuntimeError, "Project raw cache DB not found"):
                be._rebuild_test_data_project_calc_cache_from_raw(db_path, wb_path)


    def test_cache_rebuild_tracks_source_metadata_updates(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            support_dir = root / "EIDAT Support"
            art_dir = support_dir / "debug" / "ocr" / "SN1"
            art_dir.mkdir(parents=True, exist_ok=True)
            meta_path = art_dir / "sn1_metadata.json"

            def _write_meta(vendor: str) -> None:
                meta_path.write_text(
                    json.dumps(
                        {
                            "serial_number": "SN1",
                            "program_title": "Program Alpha",
                            "asset_type": "Thruster",
                            "vendor": vendor,
                            "part_number": "PN-001",
                            "revision": "B",
                            "document_type": "TD",
                            "document_type_acronym": "TD",
                            "similarity_group": "SG-1",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            _write_meta("Vendor A")

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[
                    {
                        "serial_number": "SN1",
                        "excel_sqlite_rel": str(src_db),
                        "metadata_rel": str(meta_path.relative_to(support_dir)),
                        "artifacts_rel": str(art_dir.relative_to(support_dir)),
                    }
                ],
                config=self._make_config(),
            )

            out_db = root / "implementation_trending.sqlite3"
            be.rebuild_test_data_project_cache(out_db, wb_path)

            with sqlite3.connect(str(out_db)) as conn:
                row = conn.execute(
                    """
                    SELECT vendor, program_title, part_number, metadata_rel, artifacts_rel, metadata_mtime_ns
                    FROM td_source_metadata
                    WHERE serial=?
                    """,
                    ("SN1",),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "Vendor A")
                self.assertEqual(row[1], "Program Alpha")
                self.assertEqual(row[2], "PN-001")
                self.assertEqual(row[3], str(meta_path.relative_to(support_dir)))
                self.assertEqual(row[4], str(art_dir.relative_to(support_dir)))
                self.assertGreater(int(row[5] or 0), 0)

            time.sleep(0.05)
            _write_meta("Vendor B")
            be.ensure_test_data_project_cache(root, wb_path)

            with sqlite3.connect(str(out_db)) as conn:
                row = conn.execute("SELECT vendor FROM td_source_metadata WHERE serial=?", ("SN1",)).fetchone()
                self.assertEqual(row[0], "Vendor B")

    def test_cache_rebuild_prefers_active_node_sources_and_heals_stale_links(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            current_support = root / "EIDAT" / "EIDAT Support"
            art_dir = current_support / "debug" / "ocr" / "SN1"
            art_dir.mkdir(parents=True, exist_ok=True)
            current_src = art_dir / "source_current.sqlite3"
            self._make_source_sqlite(current_src)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as old_td:
                old_root = Path(old_td)
                old_art_dir = old_root / "EIDAT Support" / "debug" / "ocr" / "SN1"
                old_art_dir.mkdir(parents=True, exist_ok=True)
                old_src = old_art_dir / "source_old.sqlite3"
                self._make_source_sqlite(old_src)

                wb_path = root / "project.xlsx"
                be._write_test_data_trending_workbook(
                    wb_path,
                    global_repo=root,
                    serials=["SN1"],
                    docs=[
                        {
                            "serial_number": "SN1",
                            "excel_sqlite_rel": "",
                            "artifacts_rel": str(art_dir.relative_to(current_support)),
                            "document_type": "TD",
                        }
                    ],
                    config=self._make_config(),
                )
                wb = load_workbook(str(wb_path))
                try:
                    ws = wb["Sources"]
                    headers = {
                        str(ws.cell(1, c).value or "").strip().lower(): c
                        for c in range(1, (ws.max_column or 0) + 1)
                    }
                    ws.cell(2, headers["excel_sqlite_rel"]).value = str(old_src)
                    wb.save(str(wb_path))
                finally:
                    wb.close()

                support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
                be._write_td_support_workbook(
                    support_path,
                    sequence_names=["RunA"],
                    param_defs=[{"name": "thrust", "units": "lbf"}],
                )

                db_path = be.ensure_test_data_project_cache(root, wb_path, rebuild=True)
                expected_rel = "EIDAT\\EIDAT Support\\debug\\ocr\\SN1\\source_current.sqlite3"

                with sqlite3.connect(str(db_path)) as conn:
                    row = conn.execute("SELECT sqlite_path FROM td_sources WHERE serial=?", ("SN1",)).fetchone()
                    self.assertEqual(str(row[0] or ""), str(current_src))
                    meta = conn.execute(
                        "SELECT excel_sqlite_rel FROM td_source_metadata WHERE serial=?",
                        ("SN1",),
                    ).fetchone()
                    self.assertEqual(str(meta[0] or ""), expected_rel)

                wb = load_workbook(str(wb_path), read_only=True, data_only=True)
                try:
                    ws = wb["Sources"]
                    headers = {
                        str(ws.cell(1, c).value or "").strip().lower(): c
                        for c in range(1, (ws.max_column or 0) + 1)
                    }
                    self.assertEqual(str(ws.cell(2, headers["excel_sqlite_rel"]).value or ""), expected_rel)
                finally:
                    wb.close()

    def test_cache_rebuild_forces_refresh_when_cached_workbook_context_mismatches(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            old_src = root / "old_source.sqlite3"
            new_src = root / "new_source.sqlite3"
            self._make_source_sqlite(old_src)
            self._make_source_sqlite(new_src)

            old_wb = root / "old_project.xlsx"
            be._write_test_data_trending_workbook(
                old_wb,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(old_src)}],
                config=self._make_config(),
            )
            old_support = be.td_support_workbook_path_for(old_wb, project_dir=root)
            be._write_td_support_workbook(
                old_support,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )
            be.ensure_test_data_project_cache(root, old_wb, rebuild=True)

            new_wb = root / "new_project.xlsx"
            be._write_test_data_trending_workbook(
                new_wb,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(new_src)}],
                config=self._make_config(),
            )
            new_support = be.td_support_workbook_path_for(new_wb, project_dir=root)
            be._write_td_support_workbook(
                new_support,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = be.ensure_test_data_project_cache(root, new_wb)
            with sqlite3.connect(str(db_path)) as conn:
                meta = {
                    str(k or ""): str(v or "")
                    for k, v in conn.execute(
                        "SELECT key, value FROM td_meta WHERE key IN ('workbook_path', 'node_root')"
                    ).fetchall()
                }
                self.assertEqual(meta.get("workbook_path"), str(new_wb))
                self.assertEqual(meta.get("node_root"), str(root))
                row = conn.execute("SELECT sqlite_path FROM td_sources WHERE serial=?", ("SN1",)).fetchone()
                self.assertEqual(str(row[0] or ""), str(new_src))

    def test_cache_rebuild_recovers_blank_source_link_from_artifacts_and_stays_stable(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            current_support = root / "EIDAT" / "EIDAT Support"
            art_dir = current_support / "debug" / "ocr" / "SN1"
            art_dir.mkdir(parents=True, exist_ok=True)
            current_src = art_dir / "source_current.sqlite3"
            self._make_source_sqlite(current_src)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[
                    {
                        "serial_number": "SN1",
                        "excel_sqlite_rel": "",
                        "artifacts_rel": str(art_dir.relative_to(current_support)),
                        "document_type": "TD",
                    }
                ],
                config=self._make_config(),
            )
            wb = load_workbook(str(wb_path))
            try:
                ws = wb["Sources"]
                headers = {
                    str(ws.cell(1, c).value or "").strip().lower(): c
                    for c in range(1, (ws.max_column or 0) + 1)
                }
                ws.cell(2, headers["excel_sqlite_rel"]).value = ""
                wb.save(str(wb_path))
            finally:
                wb.close()

            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            db_path = be.ensure_test_data_project_cache(root, wb_path, rebuild=True)
            db_path_2 = be.ensure_test_data_project_cache(root, wb_path)
            self.assertEqual(str(db_path), str(db_path_2))

            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute("SELECT sqlite_path FROM td_sources WHERE serial=?", ("SN1",)).fetchone()
                self.assertEqual(str(row[0] or ""), str(current_src))

            wb = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws = wb["Sources"]
                headers = {
                    str(ws.cell(1, c).value or "").strip().lower(): c
                    for c in range(1, (ws.max_column or 0) + 1)
                }
                healed = str(ws.cell(2, headers["excel_sqlite_rel"]).value or "")
                self.assertEqual(healed, "EIDAT\\EIDAT Support\\debug\\ocr\\SN1\\source_current.sqlite3")
            finally:
                wb.close()

    def test_cache_rebuild_fails_clearly_when_no_valid_sources_resolve(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as old_td:
                old_root = Path(old_td)
                old_src = old_root / "source_old.sqlite3"
                self._make_source_sqlite(old_src)

                current_support = root / "EIDAT" / "EIDAT Support"
                current_support.mkdir(parents=True, exist_ok=True)

                wb_path = root / "project.xlsx"
                be._write_test_data_trending_workbook(
                    wb_path,
                    global_repo=root,
                    serials=["SN1"],
                    docs=[
                        {
                            "serial_number": "SN1",
                            "excel_sqlite_rel": str(old_src),
                            "artifacts_rel": "debug/ocr/SN1",
                            "document_type": "TD",
                        }
                    ],
                    config=self._make_config(),
                )
                support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
                be._write_td_support_workbook(
                    support_path,
                    sequence_names=["RunA"],
                    param_defs=[{"name": "thrust", "units": "lbf"}],
                )

                with self.assertRaisesRegex(RuntimeError, "Sources sheet"):
                    be.ensure_test_data_project_cache(root, wb_path, rebuild=True)

                debug_path = root / be.TD_CACHE_DEBUG_JSON
                self.assertTrue(debug_path.exists())
                payload = json.loads(debug_path.read_text(encoding="utf-8"))
                self.assertEqual(int((payload.get("counts") or {}).get("valid_sources") or 0), 0)
                reason = str(((payload.get("sources") or [{}])[0]).get("reason") or "")
                self.assertIn("outside the active node root", reason)

    def test_rebuild_purges_legacy_raw_tables_from_implementation_db(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            self._make_source_sqlite(src_db)

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=self._make_config(),
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            impl_db = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(impl_db)) as conn:
                be._ensure_test_data_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    ("LegacyRun", "thrust", "lbf", "y"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_curves_raw
                    (run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns, program_title, source_run_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("LegacyRun", "thrust", "Time", "SN1__LegacyRun", "SN1", "[0,1]", "[2,3]", 2, 1, 1, "", "LegacyRun"),
                )
                conn.commit()

            be.rebuild_test_data_project_cache(impl_db, wb_path)

            with sqlite3.connect(str(impl_db)) as conn:
                tables = {
                    str(r[0] or "").strip()
                    for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                self.assertNotIn("td_columns_raw", tables)
                self.assertNotIn("td_curves_raw", tables)
                self.assertIn("td_metrics_calc", tables)

    def test_rebuild_matches_y_columns_via_aliases(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            src_db = root / "src_alias.sqlite3"
            with sqlite3.connect(str(src_db)) as conn:
                conn.execute(
                    """
                    CREATE TABLE "sheet__RunA" (
                        excel_row INTEGER NOT NULL,
                        "Time" REAL,
                        "Thrust_lbf" REAL
                    )
                    """
                )
                rows = [(i + 1, float(i), float(i + 1) * 10.0) for i in range(6)]
                conn.executemany(
                    'INSERT INTO "sheet__RunA"(excel_row,"Time","Thrust_lbf") VALUES(?,?,?)',
                    rows,
                )
                conn.commit()

            cfg = self._make_config()
            cfg["columns"] = [
                {"name": "thrust", "units": "lbf", "range_min": None, "range_max": None, "aliases": ["Thrust lbf"]}
            ]

            wb_path = root / "project.xlsx"
            be._write_test_data_trending_workbook(
                wb_path,
                global_repo=root,
                serials=["SN1"],
                docs=[{"serial_number": "SN1", "excel_sqlite_rel": str(src_db)}],
                config=cfg,
            )
            support_path = be.td_support_workbook_path_for(wb_path, project_dir=root)
            be._write_td_support_workbook(
                support_path,
                sequence_names=["RunA"],
                param_defs=[{"name": "thrust", "units": "lbf"}],
            )

            impl_db = root / "implementation_trending.sqlite3"
            payload = be.rebuild_test_data_project_cache(impl_db, wb_path)
            self.assertEqual(int(payload.get("curves_written") or 0), 1)
            timings = payload.get("timings") or {}
            for key in (
                "source_resolution_s",
                "source_sqlite_read_s",
                "run_discovery_and_matching_s",
                "raw_curve_extraction_s",
                "raw_db_write_s",
                "calc_rebuild_s",
                "total_s",
            ):
                self.assertIn(key, timings)

            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                row = conn.execute(
                    "SELECT y_name, x_name FROM td_curves_raw WHERE run_name=? AND serial=?",
                    ("RunA", "SN1"),
                ).fetchone()
                self.assertEqual(row, ("thrust", "Time"))
                raw_curve_count = int(conn.execute("SELECT COUNT(*) FROM td_curves_raw").fetchone()[0] or 0)
                legacy_curve_count = int(conn.execute("SELECT COUNT(*) FROM td_curves").fetchone()[0] or 0)
                self.assertGreater(raw_curve_count, 0)
                self.assertEqual(legacy_curve_count, 0)

            debug_payload = json.loads((root / be.TD_CACHE_DEBUG_JSON).read_text(encoding="utf-8"))
            debug_timings = debug_payload.get("timings") or {}
            for key in (
                "source_resolution_s",
                "source_sqlite_read_s",
                "run_discovery_and_matching_s",
                "raw_curve_extraction_s",
                "raw_db_write_s",
                "calc_rebuild_s",
                "total_s",
            ):
                self.assertIn(key, debug_timings)

    def test_perf_candidate_discovery_clusters_near_equal_x_by_default(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "Seq1", 3.0, 10.0),
                    ("SN1", "Seq2", 3.1, 11.0),
                    ("SN1", "Seq3", 3.14, 12.0),
                ],
            )

            candidates = be.td_discover_performance_candidates(db_path)
            self.assertEqual(candidates, [])

    def test_perf_display_value_prefers_median_plus_minus_3sigma_by_default(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        values = {"median": 100.0, "std": 2.0, "min": 95.0, "max": 105.0}
        self.assertEqual(be.td_perf_display_value(values, "min"), 94.0)
        self.assertEqual(be.td_perf_display_value(values, "max"), 106.0)
        self.assertEqual(be.td_perf_display_value(values, "min", bounds_mode="actual"), 95.0)
        self.assertEqual(be.td_perf_display_value(values, "max", bounds_mode="actual"), 105.0)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_condition_dropdown_prefers_run_condition_label(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "sequence",
                "run_condition": "350 psia, PM, 60 Sec ON / 120 Sec OFF",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            }
        ]
        harness.cb_run_mode.setCurrentIndex(harness.cb_run_mode.findData("condition"))

        harness._refresh_run_dropdown()

        self.assertEqual(harness.cb_run.itemText(0), "350 psia, PM, 60 Sec ON / 120 Sec OFF")

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_condition_title_is_condition_first_and_not_sequence(self) -> None:
        harness = _RunSelectionHarness()
        selection = {
            "mode": "condition",
            "id": "condition:seq1",
            "run_name": "Seq1",
            "display_text": "sequence",
            "run_condition": "350 psia, PM, 60 Sec ON / 120 Sec OFF",
            "member_runs": ["Seq1"],
            "member_sequences": ["Seq1", "Seq2"],
        }

        title = harness._compose_run_title(selection, "mean")

        self.assertEqual(
            title,
            "Run Condition: 350 psia, PM, 60 Sec ON / 120 Sec OFF | Sequences: Seq1, Seq2 | mean",
        )

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_condition_auto_plot_name_uses_run_condition_label(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "sequence",
                "run_condition": "350 psia, PM, 60 Sec ON / 120 Sec OFF",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            }
        ]
        harness._auto_plots = [
            {
                "mode": "metrics",
                "selector_mode": "condition",
                "selection_id": "condition:seq1",
                "run": "Seq1",
                "run_condition": "350 psia, PM, 60 Sec ON / 120 Sec OFF",
                "display_text": "sequence",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "stats": ["mean"],
                "y": ["thrust"],
            }
        ]

        harness._refresh_auto_plots_list()

        self.assertEqual(
            harness.list_auto_plots.item(0).text(),
            "Metrics: 350 psia, PM, 60 Sec ON / 120 Sec OFF mean (thrust)",
        )

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_metrics_condition_checklist_populates_and_defaults_all_checked(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "350 psia, SS",
                "run_condition": "350 psia, SS",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            },
            {
                "mode": "condition",
                "id": "condition:seq3",
                "run_name": "Seq3",
                "display_text": "410 psia, PM",
                "run_condition": "410 psia, PM",
                "member_runs": ["Seq3"],
                "member_sequences": ["Seq3"],
                "details_text": "Source Sequences: Seq3",
            },
        ]
        harness._set_mode("metrics")
        harness.cb_run_mode.setCurrentIndex(harness.cb_run_mode.findData("condition"))

        self.assertEqual(harness.list_metric_run_conditions.count(), 2)
        self.assertEqual(harness.checked_condition_ids(), ["condition:seq1", "condition:seq3"])

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_run_dropdown_hides_unchecked_program_items(self) -> None:
        harness = _RunSelectionHarness()
        harness._checked_program_filters = ["Program A"]
        harness._run_selection_views["sequence"] = [
            {
                "mode": "sequence",
                "id": "sequence:a",
                "run_name": "RunA",
                "sequence_name": "SeqA",
                "display_text": "Program A - SeqA",
                "program_title": "Program A",
                "member_programs": ["Program A"],
                "member_runs": ["RunA"],
                "member_sequences": ["SeqA"],
                "details_text": "Program: Program A | Source Sequence: SeqA",
            },
            {
                "mode": "sequence",
                "id": "sequence:b",
                "run_name": "RunB",
                "sequence_name": "SeqB",
                "display_text": "Program B - SeqB",
                "program_title": "Program B",
                "member_programs": ["Program B"],
                "member_runs": ["RunB"],
                "member_sequences": ["SeqB"],
                "details_text": "Program: Program B | Source Sequence: SeqB",
            },
        ]
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:a",
                "run_name": "RunA",
                "display_text": "Condition A",
                "run_condition": "Condition A",
                "member_programs": ["Program A"],
                "member_runs": ["RunA"],
                "member_sequences": ["SeqA"],
                "details_text": "Source Sequences: SeqA",
            },
            {
                "mode": "condition",
                "id": "condition:b",
                "run_name": "RunB",
                "display_text": "Condition B",
                "run_condition": "Condition B",
                "member_programs": ["Program B"],
                "member_runs": ["RunB"],
                "member_sequences": ["SeqB"],
                "details_text": "Source Sequences: SeqB",
            },
        ]

        harness._sync_run_mode_availability()
        harness._refresh_run_dropdown()

        self.assertEqual(harness.cb_run.count(), 1)
        self.assertEqual(harness.cb_run.itemText(0), "Program A - SeqA")
        self.assertEqual(harness.list_metric_run_conditions.count(), 1)
        self.assertEqual(harness.list_metric_run_conditions.item(0).text(), "Condition A")

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_current_run_selections_and_member_runs_follow_checked_conditions(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:a",
                "run_name": "RunA",
                "display_text": "350 psia, SS",
                "run_condition": "350 psia, SS",
                "member_runs": ["RunA"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            },
            {
                "mode": "condition",
                "id": "condition:b",
                "run_name": "RunB",
                "display_text": "410 psia, PM",
                "run_condition": "410 psia, PM",
                "member_runs": ["RunA", "RunB"],
                "member_sequences": ["Seq3"],
                "details_text": "Source Sequences: Seq3",
            },
        ]
        harness._set_mode("metrics")
        harness.cb_run_mode.setCurrentIndex(harness.cb_run_mode.findData("condition"))
        harness._set_metric_condition_selection_ids(["condition:b", "condition:a"])

        self.assertEqual(
            [item.get("id") for item in harness._current_run_selections()],
            ["condition:a", "condition:b"],
        )
        self.assertEqual(harness._current_member_runs(), ["RunA", "RunB"])

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_multi_condition_title_uses_pluralized_condition_label(self) -> None:
        harness = _RunSelectionHarness()
        selection = harness._combine_run_selections(
            [
                {
                    "mode": "condition",
                    "id": "condition:seq1",
                    "run_name": "Seq1",
                    "display_text": "350 psia, SS",
                    "run_condition": "350 psia, SS",
                    "member_runs": ["Seq1"],
                    "member_sequences": ["Seq1", "Seq2"],
                    "details_text": "Source Sequences: Seq1, Seq2",
                },
                {
                    "mode": "condition",
                    "id": "condition:seq3",
                    "run_name": "Seq3",
                    "display_text": "410 psia, PM",
                    "run_condition": "410 psia, PM",
                    "member_runs": ["Seq3"],
                    "member_sequences": ["Seq3"],
                    "details_text": "Source Sequences: Seq3",
                },
            ]
        )

        title = harness._compose_run_title(selection, "mean")

        self.assertEqual(
            title,
            "Run Conditions: 350 psia, SS, 410 psia, PM | Sequences: Seq1, Seq2, Seq3 | mean",
        )

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_multi_condition_auto_plot_name_uses_combined_labels(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "350 psia, SS",
                "run_condition": "350 psia, SS",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            },
            {
                "mode": "condition",
                "id": "condition:seq3",
                "run_name": "Seq3",
                "display_text": "410 psia, PM",
                "run_condition": "410 psia, PM",
                "member_runs": ["Seq3"],
                "member_sequences": ["Seq3"],
                "details_text": "Source Sequences: Seq3",
            },
        ]
        harness._auto_plots = [
            {
                "mode": "metrics",
                "selector_mode": "condition",
                "selection_id": "condition:multi:condition:seq1|condition:seq3",
                "selection_ids": ["condition:seq1", "condition:seq3"],
                "selection_labels": ["350 psia, SS", "410 psia, PM"],
                "run_conditions": ["350 psia, SS", "410 psia, PM"],
                "run_condition": "350 psia, SS, 410 psia, PM",
                "member_runs": ["Seq1", "Seq3"],
                "member_sequences": ["Seq1", "Seq2", "Seq3"],
                "stats": ["mean"],
                "y": ["thrust"],
            }
        ]

        harness._refresh_auto_plots_list()

        self.assertEqual(
            harness.list_auto_plots.item(0).text(),
            "Metrics: 350 psia, SS, 410 psia, PM mean (thrust)",
        )

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_open_selected_auto_plot_restores_multi_condition_checks(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "350 psia, SS",
                "run_condition": "350 psia, SS",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            },
            {
                "mode": "condition",
                "id": "condition:seq3",
                "run_name": "Seq3",
                "display_text": "410 psia, PM",
                "run_condition": "410 psia, PM",
                "member_runs": ["Seq3"],
                "member_sequences": ["Seq3"],
                "details_text": "Source Sequences: Seq3",
            },
        ]
        harness._auto_plots = [
            {
                "mode": "metrics",
                "selector_mode": "condition",
                "selection_id": "condition:multi:condition:seq1|condition:seq3",
                "selection_ids": ["condition:seq1", "condition:seq3"],
                "selection_labels": ["350 psia, SS", "410 psia, PM"],
                "run_conditions": ["350 psia, SS", "410 psia, PM"],
                "member_runs": ["Seq1", "Seq3"],
                "member_sequences": ["Seq1", "Seq2", "Seq3"],
                "stats": ["mean"],
                "y": ["thrust"],
            }
        ]
        harness._refresh_auto_plots_list()
        harness.list_auto_plots.item(0).setSelected(True)

        harness._open_selected_auto_plot()

        self.assertEqual(harness.checked_condition_ids(), ["condition:seq1", "condition:seq3"])
        self.assertTrue(harness._plot_metrics_called)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_open_selected_auto_plot_keeps_single_condition_for_legacy_payload(self) -> None:
        harness = _RunSelectionHarness()
        harness._run_selection_views["condition"] = [
            {
                "mode": "condition",
                "id": "condition:seq1",
                "run_name": "Seq1",
                "display_text": "350 psia, SS",
                "run_condition": "350 psia, SS",
                "member_runs": ["Seq1"],
                "member_sequences": ["Seq1", "Seq2"],
                "details_text": "Source Sequences: Seq1, Seq2",
            },
            {
                "mode": "condition",
                "id": "condition:seq3",
                "run_name": "Seq3",
                "display_text": "410 psia, PM",
                "run_condition": "410 psia, PM",
                "member_runs": ["Seq3"],
                "member_sequences": ["Seq3"],
                "details_text": "Source Sequences: Seq3",
            },
        ]
        harness._auto_plots = [
            {
                "mode": "metrics",
                "selector_mode": "condition",
                "selection_id": "condition:seq3",
                "run": "Seq3",
                "run_condition": "410 psia, PM",
                "display_text": "410 psia, PM",
                "member_runs": ["Seq3"],
                "member_sequences": ["Seq3"],
                "stats": ["mean"],
                "y": ["thrust"],
            }
        ]
        harness._refresh_auto_plots_list()
        harness.list_auto_plots.item(0).setSelected(True)

        harness._open_selected_auto_plot()

        self.assertEqual(harness.checked_condition_ids(), ["condition:seq3"])
        self.assertTrue(harness._plot_metrics_called)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_perf_axis_selecting_third_parameter_does_not_recurse(self) -> None:
        harness = _PerfAxisHarness()

        harness.clear_calls = 0
        harness.set_user_value(harness.cb_perf_z_col, "C")

        self.assertEqual(harness._perf_var_names(), ("B", "A", "C"))
        self.assertEqual(harness.clear_calls, 1)
        self.assertIn("Mode: surface", harness.lbl_perf_common_runs.text())

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_perf_axis_blank_z_remains_optional(self) -> None:
        harness = _PerfAxisHarness()

        harness.clear_calls = 0
        harness.cb_perf_z_col.setCurrentIndex(0)

        self.assertEqual(harness._perf_current_col_name(harness.cb_perf_z_col), "")
        self.assertEqual(harness._perf_var_names(), ("B", "A", ""))
        self.assertEqual(harness.clear_calls, 0)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_perf_axis_invalidated_peer_falls_back_once(self) -> None:
        harness = _PerfAxisHarness()
        harness.clear_calls = 0
        self.assertTrue(harness._set_combo_to_value(harness.cb_perf_x_col, "B"))
        self.assertTrue(harness._set_combo_to_value(harness.cb_perf_y_col, "B"))
        self.assertTrue(harness._set_combo_to_value(harness.cb_perf_z_col, "C"))
        harness._on_perf_axis_changed("x")

        output_name, input1_name, input2_name = harness._perf_var_names()
        self.assertEqual(input1_name, "B")
        self.assertEqual(input2_name, "C")
        self.assertEqual(output_name, "A")
        self.assertEqual(len({output_name, input1_name, input2_name}), 3)
        self.assertEqual(harness.clear_calls, 1)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_perf_axis_preset_restore_with_three_parameters_is_safe(self) -> None:
        harness = _PerfAxisHarness()

        harness.clear_calls = 0
        self.assertTrue(harness._set_combo_to_value(harness.cb_perf_y_col, "C"))
        self.assertTrue(harness._set_combo_to_value(harness.cb_perf_x_col, "A"))
        self.assertTrue(harness._set_combo_to_value(harness.cb_perf_z_col, "B"))
        harness._on_perf_axis_changed("z")

        self.assertEqual(harness._perf_var_names(), ("C", "A", "B"))
        self.assertEqual(harness.clear_calls, 1)

    def test_perf_mean_3sigma_value_uses_mean_plus_minus_three_sigma(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        values = {"mean": 100.0, "std": 2.0, "min": 95.0, "max": 105.0}
        self.assertEqual(be.td_perf_mean_3sigma_value(values, "min_3sigma"), 94.0)
        self.assertEqual(be.td_perf_mean_3sigma_value(values, "max_3sigma"), 106.0)

    def test_perf_mean_3sigma_value_requires_mean_and_std(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        self.assertIsNone(be.td_perf_mean_3sigma_value({"std": 2.0}, "min_3sigma"))
        self.assertIsNone(be.td_perf_mean_3sigma_value({"mean": 100.0}, "max_3sigma"))

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_selects_logarithmic_for_log_data(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [1.0, 2.0, 4.0, 8.0, 16.0]
        ys = [5.0 + (3.0 * math.log(x)) for x in xs]
        model = be.td_perf_fit_model(xs, ys, fit_mode="auto", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_LOGARITHMIC)
        self.assertEqual(str(model.get("fit_mode") or ""), be.TD_PERF_FIT_MODE_AUTO)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_selects_saturating_exponential_for_ceiling_data(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        ys = [100.0 - (40.0 * math.exp(-0.45 * x)) for x in xs]
        model = be.td_perf_fit_model(xs, ys, fit_mode="auto", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_SATURATING_EXPONENTIAL)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_selects_polynomial_for_quadratic_data(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ys = [2.0 + (3.0 * x) + (0.5 * x * x) for x in xs]
        model = be.td_perf_fit_model(xs, ys, fit_mode="auto", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_POLYNOMIAL)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_manual_logarithmic_rejects_nonpositive_x(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        model = be.td_perf_fit_model([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], fit_mode="logarithmic")
        self.assertIsNone(model)

    def test_perf_fit_model_manual_piecewise_2_recovers_late_rise(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ys = [1.0, 1.2, 1.4, 1.6, 1.8, 4.8, 7.8, 10.8]
        model = be.td_perf_fit_model(xs, ys, fit_mode="piecewise_2", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_PIECEWISE_2)
        params = model.get("params") or {}
        self.assertEqual(int(params.get("segment_count") or 0), 2)
        self.assertEqual(len(params.get("breakpoints") or []), 1)
        preds = be.td_perf_predict_model(model, xs)
        for actual, pred in zip(ys, preds):
            self.assertAlmostEqual(actual, pred, places=6)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_auto_selects_piecewise_2_for_late_rise(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ys = [1.0, 1.2, 1.4, 1.6, 1.8, 4.8, 7.8, 10.8]
        model = be.td_perf_fit_model(xs, ys, fit_mode="auto", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_PIECEWISE_2)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_auto_selects_piecewise_for_reversed_drop(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ys = [9.0, 8.5, 8.0, 7.5, 7.0, 3.0, -1.0, -5.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="auto", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertIn(
            str(model.get("fit_family") or ""),
            {be.TD_PERF_FIT_MODE_PIECEWISE_2, be.TD_PERF_FIT_MODE_PIECEWISE_3},
        )

    def test_perf_fit_model_piecewise_auto_prefers_three_segments_when_needed(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = list(range(12))
        ys = [1.0, 1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="piecewise_auto", polynomial_degree=2, normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_PIECEWISE_3)

    def test_perf_fit_model_manual_piecewise_requires_enough_points(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        model = be.td_perf_fit_model([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], fit_mode="piecewise_2")
        self.assertIsNone(model)

    def test_perf_fit_model_manual_piecewise_rejects_degenerate_splits(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0 + (1.5 * x) for x in xs]
        model = be.td_perf_fit_model(xs, ys, fit_mode="piecewise_2")
        self.assertIsNone(model)

    def test_perf_fit_model_manual_piecewise_3_rejects_breakpoints_too_close(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 0.001, 0.002, 0.003, 1.0, 2.0, 3.0]
        ys = [1.0, 1.1, 1.2, 1.3, 4.0, 5.0, 6.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="piecewise_3")
        self.assertIsNone(model)

    def test_perf_predict_model_piecewise_hits_breakpoint_exactly(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="piecewise_2")
        self.assertIsNotNone(model)
        params = model.get("params") or {}
        breakpoint = float((params.get("breakpoints") or [0.0])[0])
        pred_at_break = be.td_perf_predict_model(model, [breakpoint])[0]
        pred_left = be.td_perf_predict_model(model, [breakpoint - 1e-6])[0]
        pred_right = be.td_perf_predict_model(model, [breakpoint + 1e-6])[0]
        self.assertAlmostEqual(pred_at_break, pred_left, places=4)
        self.assertAlmostEqual(pred_at_break, pred_right, places=4)

    def test_perf_build_aggregate_curve_uses_serial_medians_not_raw_density(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        curves = {
            "dense_high": [(0.0, 100.0, "r1"), (0.1, 100.0, "r2"), (0.2, 100.0, "r3"), (0.3, 100.0, "r4")],
            "sparse_low": [(0.2, 0.0, "r5")],
        }
        agg = be.td_perf_build_aggregate_curve(curves, max_bins=4, min_serials_per_bin=1)
        self.assertTrue(agg.get("x"))
        self.assertTrue(agg.get("y"))
        self.assertLess(float(min(agg.get("y") or [0.0])), 100.0)
        self.assertGreater(float(max(agg.get("y") or [0.0])), 0.0)

    def test_perf_build_aggregate_curve_returns_weighting_meta(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        curves = {
            "a": [(0.0, 10.0, "r1"), (1.0, 11.0, "r2"), (2.0, 12.0, "r3")],
            "b": [(0.0, 20.0, "r4"), (2.0, 24.0, "r5")],
        }
        agg = be.td_perf_build_aggregate_curve(curves, max_bins=3, min_serials_per_bin=1, return_meta=True)
        xs = agg.get("x") or []
        ys = agg.get("y") or []
        support = agg.get("serial_support") or []
        edge_weight = agg.get("edge_weight") or []
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(xs), len(support))
        self.assertEqual(len(xs), len(edge_weight))
        self.assertTrue(all(float(v) >= 1.0 for v in edge_weight))

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_weighted_polynomial_improves_sparse_endpoint_residual(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0002, 0.00035, 0.0005, 0.0007, 0.0010, 0.0015, 0.0030, 0.0060, 0.0120]
        ys = [170.0, 208.0, 220.0, 223.0, 224.0, 224.2, 224.8, 228.5, 233.5]
        weighted = be.td_perf_fit_model(xs, ys, fit_mode="polynomial", polynomial_degree=2, sample_weights=None)
        unweighted = be.td_perf_fit_model(xs, ys, fit_mode="polynomial", polynomial_degree=2, sample_weights=[1.0] * len(xs))
        self.assertIsNotNone(weighted)
        self.assertIsNotNone(unweighted)
        weighted_preds = be.td_perf_predict_model(weighted, [xs[0], xs[-1]])
        unweighted_preds = be.td_perf_predict_model(unweighted, [xs[0], xs[-1]])
        weighted_edge_error = abs(float(weighted_preds[0]) - ys[0]) + abs(float(weighted_preds[1]) - ys[-1])
        unweighted_edge_error = abs(float(unweighted_preds[0]) - ys[0]) + abs(float(unweighted_preds[1]) - ys[-1])
        self.assertLess(weighted_edge_error, unweighted_edge_error)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_auto_prefers_new_family_for_sharp_left_knee_and_tail_drift(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0002, 0.00035, 0.0005, 0.0007, 0.0010, 0.0015, 0.0030, 0.0060, 0.0120]
        ys = [165.0, 205.0, 220.0, 223.0, 224.0, 224.5, 226.0, 229.5, 234.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="auto", polynomial_degree=2, normalize_x=True)
        sat = be.td_perf_fit_model(xs, ys, fit_mode="saturating_exponential")
        self.assertIsNotNone(model)
        self.assertIsNotNone(sat)
        self.assertIn(
            str(model.get("fit_family") or ""),
            {be.TD_PERF_FIT_MODE_MONOTONE_PCHIP, be.TD_PERF_FIT_MODE_HYBRID_SATURATING_LINEAR},
        )
        self.assertLess(float(model.get("edge_rmse_norm") or 0.0), float(sat.get("edge_rmse_norm") or float("inf")))

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_predict_model_monotone_pchip_hits_knots_and_clamps_outside_domain(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.5, 1.0, 2.0, 4.0, 8.0]
        ys = [10.0, 12.0, 15.0, 17.0, 18.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="monotone_pchip")
        self.assertIsNotNone(model)
        preds = be.td_perf_predict_model(model, xs)
        for actual, pred in zip(ys, preds):
            self.assertAlmostEqual(actual, pred, places=6)
        clamped = be.td_perf_predict_model(model, [0.1, 10.0])
        self.assertAlmostEqual(clamped[0], ys[0], places=6)
        self.assertAlmostEqual(clamped[1], ys[-1], places=6)

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_hybrid_saturating_linear_is_monotone_and_bounded(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0002, 0.00035, 0.0005, 0.0007, 0.0010, 0.0015, 0.0030, 0.0060, 0.0120]
        ys = [165.0, 205.0, 220.0, 223.0, 224.0, 224.5, 226.0, 229.5, 234.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="hybrid_saturating_linear")
        self.assertIsNotNone(model)
        params = model.get("params") or {}
        self.assertGreaterEqual(float(params.get("m") or 0.0), 0.0)
        self.assertGreaterEqual(float(params.get("A") or 0.0), 0.0)
        self.assertGreaterEqual(float(params.get("k") or 0.0), 0.0)
        preds = be.td_perf_predict_model(model, xs)
        self.assertEqual(preds, sorted(preds))

    def test_perf_fit_model_piecewise_allows_breakpoint_near_left_edge_when_supported(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0, 0.001, 0.002, 0.003, 0.02, 0.04, 0.06, 1.0, 2.0, 3.0]
        ys = [1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 4.0, 5.0, 6.0]
        model = be.td_perf_fit_model(xs, ys, fit_mode="piecewise_2")
        self.assertIsNotNone(model)
        breakpoint = float(((model.get("params") or {}).get("breakpoints") or [999.0])[0])
        self.assertLess(breakpoint, 0.1)

    def test_perf_fit_family_label_includes_piecewise_modes(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        self.assertEqual(be.td_perf_fit_family_label("piecewise_auto"), "Piecewise Auto")
        self.assertEqual(be.td_perf_fit_family_label("piecewise_2"), "Piecewise 2-Segment")
        self.assertEqual(be.td_perf_fit_family_label("piecewise_3"), "Piecewise 3-Segment")
        self.assertEqual(be.td_perf_fit_family_label("hybrid_saturating_linear"), "Hybrid Saturating + Linear")
        self.assertEqual(be.td_perf_fit_family_label("hybrid_quadratic_residual"), "Hybrid + Quadratic Residual")
        self.assertEqual(be.td_perf_fit_family_label("monotone_pchip"), "Monotone PCHIP")
        self.assertEqual(be.td_perf_fit_family_label("quadratic_surface_control_period"), "Quadratic Surface + Control Period")

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_fit_model_hybrid_quadratic_residual_competes_and_predicts(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        xs = [0.0002, 0.00035, 0.0005, 0.0007, 0.0010, 0.0015, 0.0030, 0.0060, 0.0120]
        ys = [165.0, 205.0, 220.0, 223.0, 224.0, 224.5, 226.2, 230.2, 235.8]
        model = be.td_perf_fit_model(xs, ys, fit_mode="hybrid_quadratic_residual", normalize_x=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_MODE_HYBRID_QUADRATIC_RESIDUAL)
        preds = be.td_perf_predict_model(model, xs)
        self.assertEqual(len(preds), len(xs))
        self.assertTrue(all(math.isfinite(float(v)) for v in preds))

    def test_perf_fit_surface_model_manual_plane_override(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        for x1 in (1.0, 2.0, 3.0):
            for x2 in (5.0, 7.0, 9.0):
                x1s.append(x1)
                x2s.append(x2)
                ys.append(3.0 + (2.0 * x1) - (0.75 * x2))

        model = be.td_perf_fit_surface_model(x1s, x2s, ys, surface_family="plane")
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_FAMILY_PLANE)

    def test_perf_fit_surface_model_manual_quadratic_override(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        for x1 in (1.0, 2.0, 3.0):
            for x2 in (4.0, 6.0, 8.0):
                x1s.append(x1)
                x2s.append(x2)
                ys.append(10.0 + (2.0 * x1) - (1.5 * x2) + (0.25 * x1 * x2) + (0.5 * x1 * x1) - (0.2 * x2 * x2))

        model = be.td_perf_fit_surface_model(x1s, x2s, ys, surface_family="quadratic_surface")
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE)

    def test_perf_fit_surface_model_prefers_quadratic_surface_for_quadratic_data(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        for x1 in (1.0, 2.0, 3.0):
            for x2 in (4.0, 6.0, 8.0):
                x1s.append(x1)
                x2s.append(x2)
                ys.append(10.0 + (2.0 * x1) - (1.5 * x2) + (0.25 * x1 * x2) + (0.5 * x1 * x1) - (0.2 * x2 * x2))

        model = be.td_perf_fit_surface_model(x1s, x2s, ys, auto_surface_families=False)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE)
        preds = be.td_perf_predict_surface(model, x1s, x2s)
        for actual, pred in zip(ys, preds):
            self.assertAlmostEqual(actual, pred, places=6)
        self.assertTrue(str(model.get("x_norm_equation") or "").strip())
        self.assertIsInstance(model.get("x1_center"), float)
        self.assertIsInstance(model.get("x2_scale"), float)

    def test_perf_fit_surface_model_auto_prefers_plane_for_planar_data(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        for x1 in (1.0, 2.0, 3.0):
            for x2 in (5.0, 7.0, 9.0):
                x1s.append(x1)
                x2s.append(x2)
                ys.append(3.0 + (2.0 * x1) - (0.75 * x2))

        model = be.td_perf_fit_surface_model(x1s, x2s, ys, auto_surface_families=True)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_FAMILY_PLANE)

    def test_perf_fit_surface_model_rejects_degenerate_input_coverage(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        model = be.td_perf_fit_surface_model(
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 5.0, 5.0, 5.0],
            [10.0, 12.0, 14.0, 16.0],
            auto_surface_families=False,
        )
        self.assertIsNone(model)

    def test_perf_fit_surface_model_uses_normalized_solver_for_disparate_scales(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        for x1 in (0.0003, 0.0006, 0.0010, 0.0020, 0.0040, 0.0080):
            for x2 in (50.0, 100.0, 150.0, 200.0):
                x1s.append(x1)
                x2s.append(x2)
                ys.append(220.0 + (5000.0 * x1) - (0.05 * x2) + (100000.0 * x1 * x1) + (0.002 * x1 * x2) + (0.0002 * x2 * x2))

        model = be.td_perf_fit_surface_model(x1s, x2s, ys, auto_surface_families=False)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE)
        self.assertEqual(str(model.get("solver") or ""), "irls_lstsq")
        self.assertLess(float(model.get("condition_number") or 999.0), 100.0)
        self.assertEqual(float(model.get("ridge_alpha") or 0.0), 0.0)
        self.assertGreaterEqual(int(model.get("iterations") or 0), 2)
        self.assertIs(bool(model.get("converged")), True)
        self.assertTrue(math.isfinite(float(model.get("y_center") or 0.0)))
        self.assertGreater(float(model.get("y_scale") or 0.0), 0.0)
        preds = be.td_perf_predict_surface(model, x1s, x2s)
        for actual, pred in zip(ys, preds):
            self.assertAlmostEqual(actual, pred, places=5)

    def test_perf_fit_surface_model_switches_to_ridge_when_normalized_design_is_ill_conditioned(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x2s = [2.0, 4.000001, 6.000001, 8.000002, 10.000002, 12.000003]
        ys = [10.0 + (3.0 * x1) - (2.0 * x2) + (0.5 * x1 * x1) for x1, x2 in zip(x1s, x2s)]
        model = be.td_perf_fit_surface_model(x1s, x2s, ys, auto_surface_families=False)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("solver") or ""), "irls_ridge")
        self.assertGreater(float(model.get("condition_number") or 0.0), 1e5)
        self.assertIn(float(model.get("ridge_alpha") or 0.0), {1e-6, 1e-4})
        self.assertGreaterEqual(int(model.get("iterations") or 0), 1)
        self.assertIsInstance(model.get("converged"), bool)
        self.assertGreater(float(model.get("y_scale") or 0.0), 0.0)
        preds = be.td_perf_predict_surface(model, x1s, x2s)
        self.assertEqual(len(preds), len(ys))
        self.assertTrue(all(math.isfinite(float(v)) for v in preds))

    def test_perf_fit_surface_model_reduces_low_end_overshoot_with_iterative_weights(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        np = be._td_perf_import_numpy()
        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        for x1 in (0.0, 1.0, 2.0, 3.0, 4.0):
            for x2 in (0.0, 1.0, 2.0, 3.0, 4.0):
                y_val = 42.0 + (2.5 * x1) + (1.75 * x2) + (0.35 * x1 * x1) + (0.2 * x1 * x2) + (0.28 * x2 * x2)
                if x1 <= 1.0 and x2 <= 1.0:
                    y_val -= 6.0
                x1s.append(x1)
                x2s.append(x2)
                ys.append(y_val)

        model = be.td_perf_fit_surface_model(x1s, x2s, ys, auto_surface_families=False)
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("solver") or ""), "irls_lstsq")

        x1n, x2n, _x1_center, _x1_scale, _x2_center, _x2_scale = be._td_perf_surface_normalize_axes(x1s, x2s)
        design = be._td_perf_surface_design_matrix(x1n, x2n, be.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE)
        y_arr = np.asarray(ys, dtype=float)
        baseline_coeffs, _residuals, rank, _singular = np.linalg.lstsq(design, y_arr, rcond=None)
        self.assertEqual(int(rank), 6)
        baseline_preds = design.dot(np.asarray(baseline_coeffs, dtype=float))
        iterative_preds = np.asarray(be.td_perf_predict_surface(model, x1s, x2s), dtype=float)

        low_mask = np.asarray([(x1 <= 1.0 and x2 <= 1.0) for x1, x2 in zip(x1s, x2s)], dtype=bool)
        baseline_low_overshoot = float(np.mean(np.maximum(baseline_preds[low_mask] - y_arr[low_mask], 0.0)))
        iterative_low_overshoot = float(np.mean(np.maximum(iterative_preds[low_mask] - y_arr[low_mask], 0.0)))
        baseline_rmse = float(np.sqrt(np.mean((baseline_preds - y_arr) ** 2)))
        iterative_rmse = float(np.sqrt(np.mean((iterative_preds - y_arr) ** 2)))

        self.assertGreater(baseline_low_overshoot, 0.0)
        self.assertLess(iterative_low_overshoot, baseline_low_overshoot)
        self.assertLessEqual(iterative_rmse, (baseline_rmse * 1.10) + 1e-9)

    def test_perf_fit_surface_model_control_period_matches_three_trained_slices(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        cps = [60.0, 120.0, 180.0]
        x1_axis = [1.0, 2.0, 3.0]
        x2_axis = [10.0, 20.0, 30.0]
        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        control_periods: list[float] = []
        expected_by_cp: dict[float, tuple[list[float], list[float], list[float]]] = {}
        for cp in cps:
            cp_x1: list[float] = []
            cp_x2: list[float] = []
            cp_y: list[float] = []
            c0 = 15.0 + (0.2 * cp) + (0.0015 * cp * cp)
            c1 = 0.8 + (0.006 * cp) + (0.00003 * cp * cp)
            c2 = -0.12 + (0.001 * cp) + (0.000004 * cp * cp)
            c3 = 0.08 + (0.00025 * cp)
            c4 = 0.01 + (0.00006 * cp)
            c5 = 0.002 + (0.000015 * cp)
            for x1 in x1_axis:
                for x2 in x2_axis:
                    y_val = c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2)
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(y_val)
                    control_periods.append(cp)
                    cp_x1.append(x1)
                    cp_x2.append(x2)
                    cp_y.append(y_val)
            expected_by_cp[cp] = (cp_x1, cp_x2, cp_y)

        model = be.td_perf_fit_surface_model(
            x1s,
            x2s,
            ys,
            surface_family="quadratic_surface_control_period",
            control_periods=control_periods,
        )
        self.assertIsNotNone(model)
        self.assertEqual(str(model.get("fit_family") or ""), be.TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD)
        self.assertEqual(int(model.get("control_period_degree") or -1), 2)
        self.assertEqual([float(v) for v in (model.get("control_period_values") or [])], cps)
        for cp in cps:
            cp_x1, cp_x2, cp_y = expected_by_cp[cp]
            preds = be.td_perf_predict_surface(model, cp_x1, cp_x2, control_period=cp)
            for actual, pred in zip(cp_y, preds):
                self.assertAlmostEqual(actual, pred, places=5)

    def test_perf_fit_surface_model_control_period_uses_linear_interpolation_for_two_slices(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        cps = [50.0, 150.0]
        x1_axis = [1.0, 2.0, 3.0]
        x2_axis = [5.0, 10.0, 15.0]
        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        control_periods: list[float] = []
        expected_by_cp: dict[float, tuple[list[float], list[float], list[float]]] = {}
        for cp in cps:
            cp_x1: list[float] = []
            cp_x2: list[float] = []
            cp_y: list[float] = []
            for x1 in x1_axis:
                for x2 in x2_axis:
                    y_val = (
                        (12.0 + (0.08 * cp))
                        + ((1.2 + (0.004 * cp)) * x1)
                        + ((-0.05 + (0.0008 * cp)) * x2)
                        + (0.12 * x1 * x1)
                        + ((0.015 + (0.00004 * cp)) * x1 * x2)
                        + (0.003 * x2 * x2)
                    )
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(y_val)
                    control_periods.append(cp)
                    cp_x1.append(x1)
                    cp_x2.append(x2)
                    cp_y.append(y_val)
            expected_by_cp[cp] = (cp_x1, cp_x2, cp_y)

        model = be.td_perf_fit_surface_model(
            x1s,
            x2s,
            ys,
            surface_family="quadratic_surface_control_period",
            control_periods=control_periods,
        )
        self.assertIsNotNone(model)
        self.assertEqual(int(model.get("control_period_degree") or -1), 1)
        for cp in cps:
            cp_x1, cp_x2, cp_y = expected_by_cp[cp]
            preds = be.td_perf_predict_surface(model, cp_x1, cp_x2, control_period=cp)
            for actual, pred in zip(cp_y, preds):
                self.assertAlmostEqual(actual, pred, places=5)

    def test_perf_fit_surface_model_control_period_interpolates_new_slice_stably(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        cps = [40.0, 80.0, 120.0, 160.0]
        x1_axis = [1.0, 2.0, 3.0]
        x2_axis = [8.0, 16.0, 24.0]
        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        control_periods: list[float] = []
        for cp in cps:
            c0 = 25.0 + (0.18 * cp) + (0.0008 * cp * cp)
            c1 = 0.6 + (0.004 * cp) + (0.00002 * cp * cp)
            c2 = -0.08 + (0.0006 * cp) + (0.000002 * cp * cp)
            c3 = 0.05 + (0.00015 * cp)
            c4 = 0.02 + (0.00003 * cp)
            c5 = 0.0015 + (0.00001 * cp)
            for x1 in x1_axis:
                for x2 in x2_axis:
                    y_val = c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2)
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(y_val)
                    control_periods.append(cp)

        model = be.td_perf_fit_surface_model(
            x1s,
            x2s,
            ys,
            surface_family="quadratic_surface_control_period",
            control_periods=control_periods,
        )
        self.assertIsNotNone(model)
        self.assertEqual(int(model.get("control_period_degree") or -1), 2)
        interp_x1 = [1.0, 2.0, 3.0]
        interp_x2 = [8.0, 16.0, 24.0]
        preds = be.td_perf_predict_surface(model, interp_x1, interp_x2, control_period=100.0)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(math.isfinite(float(v)) for v in preds))
        expected = []
        cp = 100.0
        c0 = 25.0 + (0.18 * cp) + (0.0008 * cp * cp)
        c1 = 0.6 + (0.004 * cp) + (0.00002 * cp * cp)
        c2 = -0.08 + (0.0006 * cp) + (0.000002 * cp * cp)
        c3 = 0.05 + (0.00015 * cp)
        c4 = 0.02 + (0.00003 * cp)
        c5 = 0.0015 + (0.00001 * cp)
        for x1, x2 in zip(interp_x1, interp_x2):
            expected.append(c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2))
        for actual, pred in zip(expected, preds):
            self.assertAlmostEqual(actual, pred, places=5)

    def test_perf_fit_surface_model_control_period_requires_multiple_distinct_periods(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        model = be.td_perf_fit_surface_model(
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            surface_family="quadratic_surface_control_period",
            control_periods=[120.0] * 6,
        )
        self.assertIsNone(model)

    def test_perf_fit_surface_model_control_period_skips_sparse_slices_and_records_ignored_periods(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        control_periods: list[float] = []
        for cp in (60.0, 120.0):
            c0 = 10.0 + (0.05 * cp)
            c1 = 0.8 + (0.002 * cp)
            c2 = -0.08 + (0.0005 * cp)
            c3 = 0.04
            c4 = 0.01 + (0.00002 * cp)
            c5 = 0.002
            for x1 in (1.0, 2.0, 3.0):
                for x2 in (10.0, 20.0, 30.0):
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2))
                    control_periods.append(cp)
        for x1, x2 in ((1.0, 10.0), (1.0, 20.0), (2.0, 10.0), (2.0, 20.0)):
            cp = 180.0
            x1s.append(x1)
            x2s.append(x2)
            ys.append(12.0 + (0.05 * cp) + (0.9 * x1) - (0.02 * x2) + (0.03 * x1 * x1) + (0.01 * x1 * x2))
            control_periods.append(cp)

        model = be.td_perf_fit_surface_model(
            x1s,
            x2s,
            ys,
            surface_family="quadratic_surface_control_period",
            control_periods=control_periods,
        )
        self.assertIsNotNone(model)
        self.assertEqual([float(v) for v in (model.get("eligible_control_period_values") or [])], [60.0, 120.0])
        self.assertEqual([float(v) for v in (model.get("fit_domain_control_period") or [])], [60.0, 120.0])
        ignored = list(model.get("ignored_control_periods") or [])
        self.assertEqual(len(ignored), 1)
        self.assertEqual(float(ignored[0].get("control_period") or 0.0), 180.0)
        self.assertIn("points (<6)", str(ignored[0].get("reason") or ""))
        self.assertIn("180", str(model.get("fit_warning_text") or ""))

    def test_perf_fit_surface_model_control_period_interpolates_inside_domain_when_middle_slice_is_ignored(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        control_periods: list[float] = []
        for cp in (60.0, 180.0):
            c0 = 18.0 + (0.04 * cp)
            c1 = 0.7 + (0.003 * cp)
            c2 = -0.03 + (0.0004 * cp)
            c3 = 0.05
            c4 = 0.012 + (0.000015 * cp)
            c5 = 0.0022
            for x1 in (1.0, 2.0, 3.0):
                for x2 in (8.0, 16.0, 24.0):
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2))
                    control_periods.append(cp)
        for x1, x2 in ((1.0, 8.0), (1.0, 16.0), (2.0, 8.0), (2.0, 16.0)):
            cp = 120.0
            x1s.append(x1)
            x2s.append(x2)
            ys.append(14.0 + (0.04 * cp) + (0.8 * x1) - (0.01 * x2) + (0.04 * x1 * x1) + (0.008 * x1 * x2))
            control_periods.append(cp)

        model = be.td_perf_fit_surface_model(
            x1s,
            x2s,
            ys,
            surface_family="quadratic_surface_control_period",
            control_periods=control_periods,
        )
        self.assertIsNotNone(model)
        preds = be.td_perf_predict_surface(model, [1.0, 2.0, 3.0], [8.0, 16.0, 24.0], control_period=120.0)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(math.isfinite(float(v)) for v in preds))
        grid = be.td_perf_build_surface_grid(model, 1.0, 3.0, 8.0, 24.0, points=4, control_period=120.0)
        self.assertTrue(grid.get("z_grid"))

    def test_perf_build_surface_grid_control_period_outside_fit_domain_returns_empty_grid(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        x1s: list[float] = []
        x2s: list[float] = []
        ys: list[float] = []
        control_periods: list[float] = []
        for cp in (60.0, 120.0):
            for x1 in (1.0, 2.0, 3.0):
                for x2 in (10.0, 20.0, 30.0):
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(20.0 + (0.06 * cp) + x1 - (0.03 * x2) + (0.02 * x1 * x2))
                    control_periods.append(cp)
        for x1, x2 in ((1.0, 10.0), (1.0, 20.0), (2.0, 10.0), (2.0, 20.0)):
            cp = 180.0
            x1s.append(x1)
            x2s.append(x2)
            ys.append(15.0 + (0.03 * cp) + x1 - (0.01 * x2))
            control_periods.append(cp)

        model = be.td_perf_fit_surface_model(
            x1s,
            x2s,
            ys,
            surface_family="quadratic_surface_control_period",
            control_periods=control_periods,
        )
        self.assertIsNotNone(model)
        grid = be.td_perf_build_surface_grid(model, 1.0, 3.0, 10.0, 30.0, points=4, control_period=180.0)
        self.assertEqual(grid.get("x1_grid"), [])
        self.assertEqual(grid.get("x2_grid"), [])
        self.assertEqual(grid.get("z_grid"), [])

    @unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
    def test_perf_collect_saved_equation_snapshot_reports_descriptive_cp_coverage_failure(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            rows: list[tuple[str, str, float, float, float, float | None]] = []
            cp = 60.0
            for x1 in (1.0, 2.0, 3.0):
                for x2 in (10.0, 20.0, 30.0):
                    rows.append(("SN1", "RunCP", x1, x2, 18.0 + (0.04 * cp) + x1 - (0.03 * x2) + (0.01 * x1 * x2), cp))
            cp = 120.0
            for x1, x2 in ((1.0, 10.0), (1.0, 20.0), (2.0, 10.0), (2.0, 20.0)):
                rows.append(("SN1", "RunCP", x1, x2, 18.0 + (0.04 * cp) + x1 - (0.03 * x2), cp))
            db_path = self._seed_perf_export_db_3d(root, rows=rows)
            self._seed_perf_source_metadata(db_path, [("SN1", "Thruster", "Hall_A")])

            snapshot = be.td_perf_collect_saved_equation_snapshot(
                db_path,
                {
                    "output": "thrust",
                    "input1": "impulse bit",
                    "input2": "feed pressure",
                    "member_runs": ["RunCP"],
                    "stats": ["mean"],
                    "fit_enabled": True,
                    "surface_fit_family": "quadratic_surface_control_period",
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "all_conditions",
                    "require_min_points": 2,
                },
            )
            fit_error_text = str(((snapshot.get("plot_metadata") or {}).get("fit_error_text") or "")).strip()
            self.assertIn("Eligible periods: 0", fit_error_text)
            self.assertIn("CP 60", fit_error_text)
            self.assertIn("CP 120", fit_error_text)
            self.assertIn("points (<6)", fit_error_text)

    @unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
    def test_perf_collect_saved_equation_snapshot_keeps_invalid_control_period_error(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            rows = [
                ("SN1", "RunCP", 1.0, 10.0, 20.0, 60.0),
                ("SN2", "RunCP", 1.0, 20.0, 21.0, 60.0),
                ("SN1", "RunCP", 2.0, 10.0, 22.0, 60.0),
                ("SN2", "RunCP", 2.0, 20.0, 23.0, 60.0),
                ("SN1", "RunCP", 3.0, 20.0, 24.0, None),
                ("SN2", "RunCP", 3.0, 30.0, 25.0, 120.0),
            ]
            db_path = self._seed_perf_export_db_3d(root, rows=rows)
            self._seed_perf_source_metadata(db_path, [("SN1", "Thruster", "Hall_A"), ("SN2", "Thruster", "Hall_A")])

            snapshot = be.td_perf_collect_saved_equation_snapshot(
                db_path,
                {
                    "output": "thrust",
                    "input1": "impulse bit",
                    "input2": "feed pressure",
                    "member_runs": ["RunCP"],
                    "stats": ["mean"],
                    "fit_enabled": True,
                    "surface_fit_family": "quadratic_surface_control_period",
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "all_conditions",
                    "require_min_points": 2,
                },
            )
            fit_error_text = str(((snapshot.get("plot_metadata") or {}).get("fit_error_text") or "")).strip()
            self.assertEqual(
                fit_error_text,
                "Quadratic Surface + Control Period requires usable control-period values for all fitted pulsed-mode points.",
            )

    @unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_perf_export_equation_workbook_writes_2d_formulas_actual_mean_and_support(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_export_db_2d(
                root,
                rows=[
                    ("SN1", "RunA", 1.000, 10.0, 1.5),
                    ("SN2", "RunA", 1.020, 14.0, 1.5),
                    ("SN1", "RunA", 2.000, 18.0, 1.5),
                    ("SN2", "RunA", 2.010, 22.0, 1.5),
                ],
            )
            xs = [1.0, 1.5, 2.0, 2.5, 3.0]
            ys = [10.0, 13.0, 18.0, 23.0, 28.0]
            piecewise_xs = [0.0, 0.001, 0.002, 0.003, 0.02, 0.04, 0.06, 1.0, 2.0, 3.0]
            piecewise_ys = [1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 4.0, 5.0, 6.0]
            results = {
                "mean": {"master_model": be.td_perf_fit_model(xs, ys, fit_mode="hybrid_quadratic_residual", normalize_x=True)},
                "min": {"master_model": be.td_perf_fit_model(piecewise_xs, piecewise_ys, fit_mode="piecewise_2")},
                "max": {"master_model": be.td_perf_fit_model(xs, ys, fit_mode="monotone_pchip")},
                "std": {"master_model": be.td_perf_fit_model(xs, [1.0, 1.1, 1.3, 1.6, 2.0], fit_mode="polynomial", polynomial_degree=2)},
            }
            out_path = root / "perf_export.xlsx"
            exported = be.td_perf_export_equation_workbook(
                db_path,
                out_path,
                plot_metadata={
                    "plot_dimension": "2d",
                    "output_target": "thrust",
                    "output_units": "lbf",
                    "input1_target": "impulse bit",
                    "input1_units": "mN-s",
                    "run_selection_label": "RunA",
                    "member_runs": ["RunA"],
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "match_control_period",
                },
                results_by_stat=results,
                run_specs=[
                    {
                        "run_name": "RunA",
                        "display_name": "RunA",
                        "output_column": "thrust",
                        "output_units": "lbf",
                        "input1_column": "impulse bit",
                        "input1_units": "mN-s",
                        "input2_column": "",
                        "input2_units": "",
                    }
                ],
                control_period_filter=1.5,
                run_type_filter="pulsed_mode",
            )
            self.assertEqual(Path(exported), out_path)

            wb = load_workbook(str(out_path), read_only=False, data_only=False)
            try:
                ws = wb["Equation Export"]
                labels = {
                    str(ws.cell(r, 1).value or "").strip(): ws.cell(r, 2).value
                    for r in range(1, (ws.max_row or 0) + 1)
                    if str(ws.cell(r, 1).value or "").strip()
                }
                self.assertEqual(labels.get("Condition Family"), "Pulsed mode")
                self.assertEqual(labels.get("PM Filter Mode"), "match_control_period")
                self.assertEqual(str(labels.get("Selected Control Period")), "1.5")

                header_row = next(
                    r for r in range(1, (ws.max_row or 0) + 1) if str(ws.cell(r, 1).value or "").strip() == "run_name"
                )
                headers = [str(ws.cell(header_row, c).value or "").strip() for c in range(1, (ws.max_column or 0) + 1)]
                self.assertIn("input_1_norm", headers)
                self.assertIn("pred_mean", headers)
                self.assertIn("pred_min", headers)
                self.assertIn("pred_max", headers)
                self.assertIn("actual_mean", headers)
                self.assertIn("pct_delta_mean", headers)

                row1 = header_row + 1
                mean_formula = str(ws.cell(row1, headers.index("pred_mean") + 1).value or "")
                min_formula = str(ws.cell(row1, headers.index("pred_min") + 1).value or "")
                max_formula = str(ws.cell(row1, headers.index("pred_max") + 1).value or "")
                min_3sigma_formula = str(ws.cell(row1, headers.index("pred_min_3sigma") + 1).value or "")
                max_3sigma_formula = str(ws.cell(row1, headers.index("pred_max_3sigma") + 1).value or "")
                pct_formula = str(ws.cell(row1, headers.index("pct_delta_mean") + 1).value or "")
                actual_mean = float(ws.cell(row1, headers.index("actual_mean") + 1).value or 0.0)
                self.assertIn("EXP(", mean_formula)
                self.assertIn("^2", mean_formula)
                self.assertIn("MAX(0", min_formula)
                self.assertIn("IF(", max_formula)
                self.assertIn("^3", max_formula)
                self.assertIn("-(", min_3sigma_formula)
                self.assertIn("3*", min_3sigma_formula)
                self.assertIn("+(", max_3sigma_formula)
                self.assertIn("3*", max_3sigma_formula)
                self.assertIn("/", pct_formula)
                self.assertAlmostEqual(actual_mean, 12.0, places=6)

                self.assertIn("Model Support", wb.sheetnames)
                self.assertEqual(wb["Model Support"].sheet_state, "hidden")
                self.assertGreater(wb["Model Support"].max_row, 1)
            finally:
                wb.close()

    @unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
    def test_perf_export_equation_workbook_writes_3d_surface_formulas_and_norm_columns(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_export_db_3d(
                root,
                rows=[
                    ("SN1", "Run3D", 1.0, 10.0, 33.0, 2.0),
                    ("SN2", "Run3D", 1.02, 10.01, 35.0, 2.0),
                    ("SN1", "Run3D", 2.0, 20.0, 55.0, 2.0),
                    ("SN2", "Run3D", 2.01, 20.02, 57.0, 2.0),
                    ("SN1", "Run3D", 3.0, 30.0, 81.0, 2.0),
                    ("SN2", "Run3D", 3.01, 30.03, 83.0, 2.0),
                ],
            )
            x1s = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
            x2s = [10.0, 20.0, 10.0, 20.0, 20.0, 30.0]
            ys = [33.0, 43.0, 45.0, 55.0, 69.0, 83.0]
            surface = be.td_perf_fit_surface_model(x1s, x2s, ys, surface_family="quadratic_surface")
            self.assertIsNotNone(surface)
            out_path = root / "perf_export_3d.xlsx"
            be.td_perf_export_equation_workbook(
                db_path,
                out_path,
                plot_metadata={
                    "plot_dimension": "3d",
                    "output_target": "thrust",
                    "output_units": "lbf",
                    "input1_target": "impulse bit",
                    "input1_units": "mN-s",
                    "input2_target": "feed pressure",
                    "input2_units": "psia",
                    "run_selection_label": "Run3D",
                    "member_runs": ["Run3D"],
                    "performance_filter_mode": "match_control_period",
                },
                results_by_stat={"mean": {"master_model": surface}},
                run_specs=[
                    {
                        "run_name": "Run3D",
                        "display_name": "Run3D",
                        "output_column": "thrust",
                        "output_units": "lbf",
                        "input1_column": "impulse bit",
                        "input1_units": "mN-s",
                        "input2_column": "feed pressure",
                        "input2_units": "psia",
                    }
                ],
                control_period_filter=2.0,
            )

            wb = load_workbook(str(out_path), read_only=False, data_only=False)
            try:
                ws = wb["Equation Export"]
                header_row = next(
                    r for r in range(1, (ws.max_row or 0) + 1) if str(ws.cell(r, 1).value or "").strip() == "run_name"
                )
                headers = [str(ws.cell(header_row, c).value or "").strip() for c in range(1, (ws.max_column or 0) + 1)]
                self.assertIn("input_2", headers)
                self.assertIn("input_2_norm", headers)
                self.assertIn("pred_mean", headers)
                self.assertIn("pct_delta_mean", headers)
                formula = str(ws.cell(header_row + 1, headers.index("pred_mean") + 1).value or "")
                pct_formula = str(ws.cell(header_row + 1, headers.index("pct_delta_mean") + 1).value or "")
                self.assertIn("^2", formula)
                self.assertIn("*", formula)
                self.assertTrue(formula.startswith("="))
                self.assertTrue(pct_formula.startswith("="))
                self.assertIn("/", pct_formula)
                self.assertTrue(str(ws.cell(header_row + 1, headers.index("input_1_norm") + 1).value or "").startswith("="))
                self.assertTrue(str(ws.cell(header_row + 1, headers.index("input_2_norm") + 1).value or "").startswith("="))
                actual_mean = float(ws.cell(header_row + 1, headers.index("actual_mean") + 1).value or 0.0)
                self.assertAlmostEqual(actual_mean, 34.0, places=6)
            finally:
                wb.close()

    @unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
    def test_perf_export_equation_workbook_writes_control_period_aware_surface_formulas(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            rows: list[tuple[str, str, float, float, float, float | None]] = []
            x1s: list[float] = []
            x2s: list[float] = []
            ys: list[float] = []
            cps: list[float] = []
            for cp in (60.0, 120.0, 180.0):
                c0 = 20.0 + (0.16 * cp) + (0.001 * cp * cp)
                c1 = 0.9 + (0.003 * cp)
                c2 = -0.04 + (0.0004 * cp)
                c3 = 0.08
                c4 = 0.012 + (0.00002 * cp)
                c5 = 0.0025
                for serial, x1, x2 in (
                    ("SN1", 1.0, 10.0),
                    ("SN2", 1.0, 20.0),
                    ("SN1", 2.0, 10.0),
                    ("SN2", 2.0, 20.0),
                    ("SN1", 3.0, 20.0),
                    ("SN2", 3.0, 30.0),
                ):
                    y_val = c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2)
                    rows.append((serial, "RunCP", x1, x2, y_val, cp))
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(y_val)
                    cps.append(cp)

            db_path = self._seed_perf_export_db_3d(root, rows=rows)
            surface = be.td_perf_fit_surface_model(
                x1s,
                x2s,
                ys,
                surface_family="quadratic_surface_control_period",
                control_periods=cps,
            )
            self.assertIsNotNone(surface)
            out_path = root / "perf_export_cp_3d.xlsx"
            be.td_perf_export_equation_workbook(
                db_path,
                out_path,
                plot_metadata={
                    "plot_dimension": "3d",
                    "output_target": "thrust",
                    "output_units": "lbf",
                    "input1_target": "impulse bit",
                    "input1_units": "mN-s",
                    "input2_target": "feed pressure",
                    "input2_units": "psia",
                    "run_selection_label": "RunCP",
                    "member_runs": ["RunCP"],
                    "performance_filter_mode": "all_conditions",
                },
                results_by_stat={"mean": {"master_model": surface}},
                run_specs=[
                    {
                        "run_name": "RunCP",
                        "display_name": "RunCP",
                        "output_column": "thrust",
                        "output_units": "lbf",
                        "input1_column": "impulse bit",
                        "input1_units": "mN-s",
                        "input2_column": "feed pressure",
                        "input2_units": "psia",
                    }
                ],
                run_type_filter="pulsed_mode",
            )

            wb = load_workbook(str(out_path), read_only=False, data_only=False)
            try:
                ws = wb["Equation Export"]
                header_row = next(
                    r for r in range(1, (ws.max_row or 0) + 1) if str(ws.cell(r, 1).value or "").strip() == "run_name"
                )
                headers = [str(ws.cell(header_row, c).value or "").strip() for c in range(1, (ws.max_column or 0) + 1)]
                self.assertIn("control_period", headers)
                self.assertIn("control_period_norm", headers)
                self.assertIn("pred_mean", headers)
                self.assertIn("pct_delta_mean", headers)
                row1 = header_row + 1
                cp_norm_formula = str(ws.cell(row1, headers.index("control_period_norm") + 1).value or "")
                pred_formula = str(ws.cell(row1, headers.index("pred_mean") + 1).value or "")
                pct_formula = str(ws.cell(row1, headers.index("pct_delta_mean") + 1).value or "")
                self.assertTrue(cp_norm_formula.startswith("="))
                self.assertTrue(pred_formula.startswith("="))
                self.assertIn("^2", pred_formula)
                self.assertIn("/", pct_formula)
                self.assertIn("Model Parameters", wb.sheetnames)
                self.assertIn("Model Support", wb.sheetnames)
                params_ws = wb["Model Parameters"]
                support_ws = wb["Model Support"]
                params_values = [str(params_ws.cell(r, 3).value or "").strip() for r in range(2, (params_ws.max_row or 0) + 1)]
                self.assertIn("cp_center", params_values)
                self.assertIn("coeff_cp_models", params_values)
                self.assertGreater(support_ws.max_row, 1)
            finally:
                wb.close()

    def test_saved_perf_equation_store_round_trip_preserves_asset_metadata(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_export_db_2d(
                root,
                rows=[
                    ("SN1", "RunA", 1.0, 10.0, 1.5),
                    ("SN2", "RunA", 2.0, 20.0, 1.5),
                ],
            )
            self._seed_perf_source_metadata(
                db_path,
                [
                    ("SN1", "Thruster", "Hall_A"),
                    ("SN2", "Thruster", "Hall_A"),
                ],
            )
            model = be.td_perf_fit_model([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], fit_mode="polynomial", polynomial_degree=2)
            self.assertIsNotNone(model)
            entry = be.td_perf_build_saved_equation_entry(
                db_path,
                name="Thrust vs Impulse",
                plot_definition={"output": "thrust", "input1": "impulse bit", "member_runs": ["RunA"], "stats": ["mean"]},
                plot_metadata={"plot_dimension": "2d", "output_target": "thrust", "input1_target": "impulse bit"},
                results_by_stat={
                    "mean": {
                        "plot_dimension": "2d",
                        "curves": {"SN1": [(1.0, 10.0, "RunA")], "SN2": [(2.0, 20.0, "RunA")]},
                        "master_model": model,
                    }
                },
                run_specs=[
                    {
                        "run_name": "RunA",
                        "display_name": "RunA",
                        "output_column": "thrust",
                        "output_units": "lbf",
                        "input1_column": "impulse bit",
                        "input1_units": "mN-s",
                        "input2_column": "",
                        "input2_units": "",
                    }
                ],
            )
            be.td_perf_upsert_saved_equation(root, entry)
            store = be.load_td_saved_performance_equations(root)
            entries = store.get("entries") or []
            self.assertEqual(len(entries), 1)
            loaded = dict(entries[0])
            asset_metadata = dict(loaded.get("asset_metadata") or {})
            self.assertEqual(asset_metadata.get("primary_asset_type"), "Thruster")
            self.assertEqual(asset_metadata.get("primary_asset_specific_type"), "Hall_A")

    def test_saved_perf_snapshot_derives_mixed_asset_metadata(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_export_db_2d(
                root,
                rows=[
                    ("SN1", "RunA", 1.0, 10.0, 1.5),
                    ("SN2", "RunA", 2.0, 20.0, 1.5),
                    ("SN1", "RunA", 3.0, 30.0, 1.5),
                    ("SN2", "RunA", 4.0, 40.0, 1.5),
                ],
            )
            self._seed_perf_source_metadata(
                db_path,
                [
                    ("SN1", "Thruster", "Hall_A"),
                    ("SN2", "Valve", "Cathode_B"),
                ],
            )
            snapshot = be.td_perf_collect_saved_equation_snapshot(
                db_path,
                {
                    "output": "thrust",
                    "input1": "impulse bit",
                    "member_runs": ["RunA"],
                    "stats": ["mean"],
                    "fit_enabled": True,
                    "fit_mode": "polynomial",
                    "polynomial_degree": 2,
                    "normalize_x": True,
                    "require_min_points": 2,
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "match_control_period",
                    "selected_control_period": 1.5,
                },
            )
            asset_metadata = dict(snapshot.get("asset_metadata") or {})
            self.assertEqual(asset_metadata.get("primary_asset_type"), "mixed_asset_type")
            self.assertEqual(asset_metadata.get("primary_asset_specific_type"), "mixed_asset_specific")

    @unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
    def test_saved_perf_export_workbook_includes_asset_metadata_rows(self) -> None:
        from openpyxl import load_workbook  # type: ignore
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_export_db_2d(
                root,
                rows=[
                    ("SN1", "RunA", 1.0, 10.0, 1.5),
                    ("SN2", "RunA", 2.0, 20.0, 1.5),
                    ("SN1", "RunA", 3.0, 30.0, 1.5),
                    ("SN2", "RunA", 4.0, 40.0, 1.5),
                ],
            )
            self._seed_perf_source_metadata(
                db_path,
                [
                    ("SN1", "Thruster", "Hall_A"),
                    ("SN2", "Thruster", "Hall_A"),
                ],
            )
            snapshot = be.td_perf_collect_saved_equation_snapshot(
                db_path,
                {
                    "output": "thrust",
                    "input1": "impulse bit",
                    "member_runs": ["RunA"],
                    "stats": ["mean"],
                    "fit_enabled": True,
                    "fit_mode": "polynomial",
                    "polynomial_degree": 2,
                    "normalize_x": True,
                    "require_min_points": 2,
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "match_control_period",
                    "selected_control_period": 1.5,
                },
            )
            entry = {
                "id": "entry1",
                "name": "Thrust vs Impulse",
                "slug": "thrust_vs_impulse",
                "saved_at": "2026-01-01 00:00:00",
                "updated_at": "2026-01-01 00:00:00",
                "plot_definition": {
                    "output": "thrust",
                    "input1": "impulse bit",
                    "member_runs": ["RunA"],
                    "stats": ["mean"],
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "match_control_period",
                    "selected_control_period": 1.5,
                },
                **snapshot,
            }
            out_path = root / "saved_perf.xlsx"
            be.td_perf_export_saved_equations_workbook(db_path, out_path, entries=[entry])
            wb = load_workbook(str(out_path), read_only=False, data_only=False)
            try:
                ws = wb[wb.sheetnames[0]]
                labels = {
                    str(ws.cell(r, 1).value or "").strip(): ws.cell(r, 2).value
                    for r in range(1, min(20, (ws.max_row or 0) + 1))
                    if str(ws.cell(r, 1).value or "").strip()
                }
                self.assertEqual(labels.get("Asset Type"), "Thruster")
                self.assertEqual(labels.get("Asset Specific Type"), "Hall_A")
            finally:
                wb.close()

    def test_saved_perf_export_matlab_groups_by_asset_type_and_subtype(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            model = be.td_perf_fit_model([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], fit_mode="polynomial", polynomial_degree=2)
            self.assertIsNotNone(model)
            out_path = root / "saved_perf_equations.m"
            be.td_perf_export_saved_equations_matlab(
                out_path,
                entries=[
                    {
                        "id": "entry1",
                        "name": "Thrust vs Impulse",
                        "slug": "thrust_vs_impulse",
                        "plot_metadata": {"plot_dimension": "2d", "output_target": "thrust", "input1_target": "impulse bit"},
                        "results_by_stat": {"mean": {"master_model": model}},
                        "equation_rows": [{"stat": "mean"}],
                        "asset_metadata": {
                            "primary_asset_type": "Thruster",
                            "primary_asset_specific_type": "Hall_A",
                            "asset_types": ["Thruster"],
                            "asset_specific_types": ["Hall_A"],
                        },
                    }
                ],
            )
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("% Usage: out = saved_perf_equations();", text)
            self.assertIn("out.Thruster.Hall_A.thrust_vs_impulse", text)
            self.assertIn("% Usage: y = out.Thruster.Hall_A.thrust_vs_impulse.mean(impulse_bit)", text)
            self.assertIn("% Inputs: impulse bit (impulse_bit)", text)
            self.assertIn("% Output: y is predicted thrust; scalar and array inputs are evaluated element-wise.", text)
            self.assertIn("equation_text_mean", text)
            self.assertIn("asset_specific_types", text)

    def test_saved_perf_export_matlab_documents_control_period_surface_signature(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            x1s: list[float] = []
            x2s: list[float] = []
            ys: list[float] = []
            cps: list[float] = []
            for cp in (60.0, 120.0, 180.0):
                c0 = 20.0 + (0.16 * cp) + (0.001 * cp * cp)
                c1 = 0.9 + (0.003 * cp)
                c2 = -0.04 + (0.0004 * cp)
                c3 = 0.08
                c4 = 0.012 + (0.00002 * cp)
                c5 = 0.0025
                for x1, x2 in (
                    (1.0, 10.0),
                    (1.0, 20.0),
                    (2.0, 10.0),
                    (2.0, 20.0),
                    (3.0, 20.0),
                    (3.0, 30.0),
                ):
                    x1s.append(x1)
                    x2s.append(x2)
                    ys.append(c0 + (c1 * x1) + (c2 * x2) + (c3 * x1 * x1) + (c4 * x1 * x2) + (c5 * x2 * x2))
                    cps.append(cp)
            model = be.td_perf_fit_surface_model(
                x1s,
                x2s,
                ys,
                surface_family="quadratic_surface_control_period",
                control_periods=cps,
            )
            self.assertIsNotNone(model)
            out_path = root / "saved_perf_equations_cp.m"
            be.td_perf_export_saved_equations_matlab(
                out_path,
                entries=[
                    {
                        "id": "entry1",
                        "name": "Thrust vs Inputs",
                        "slug": "thrust_vs_inputs",
                        "plot_metadata": {
                            "plot_dimension": "3d",
                            "output_target": "thrust",
                            "input1_target": "impulse bit",
                            "input2_target": "feed pressure",
                        },
                        "results_by_stat": {"mean": {"master_model": model}},
                        "equation_rows": [{"stat": "mean"}],
                        "asset_metadata": {
                            "primary_asset_type": "Thruster",
                            "primary_asset_specific_type": "Hall_A",
                            "asset_types": ["Thruster"],
                            "asset_specific_types": ["Hall_A"],
                        },
                    }
                ],
            )
            text = out_path.read_text(encoding="utf-8")
            self.assertIn(
                "% Usage: y = out.Thruster.Hall_A.thrust_vs_inputs.mean(impulse_bit, feed_pressure, control_period)",
                text,
            )
            self.assertIn(
                "% Inputs: impulse bit (impulse_bit), feed pressure (feed_pressure), control period (control_period)",
                text,
            )
            self.assertIn("% Output: y is predicted thrust; scalar and array inputs are evaluated element-wise.", text)
            self.assertIn("% Control-period-aware quadratic surface predictor for exported 3D equations.", text)
            self.assertIn("equation_text_mean", text)

    def test_saved_perf_refresh_store_recomputes_asset_metadata_from_cache(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_export_db_2d(
                root,
                rows=[
                    ("SN1", "RunA", 1.0, 10.0, 1.5),
                    ("SN2", "RunA", 2.0, 20.0, 1.5),
                    ("SN1", "RunA", 3.0, 30.0, 1.5),
                    ("SN2", "RunA", 4.0, 40.0, 1.5),
                ],
            )
            self._seed_perf_source_metadata(
                db_path,
                [
                    ("SN1", "Thruster", "Hall_A"),
                    ("SN2", "Thruster", "Hall_A"),
                ],
            )
            snapshot = be.td_perf_collect_saved_equation_snapshot(
                db_path,
                {
                    "output": "thrust",
                    "input1": "impulse bit",
                    "member_runs": ["RunA"],
                    "stats": ["mean"],
                    "fit_enabled": True,
                    "fit_mode": "polynomial",
                    "polynomial_degree": 2,
                    "normalize_x": True,
                    "require_min_points": 2,
                    "performance_run_type_mode": "pulsed_mode",
                    "performance_filter_mode": "match_control_period",
                    "selected_control_period": 1.5,
                },
            )
            be.save_td_saved_performance_equations(
                root,
                {
                    "entries": [
                        {
                            "id": "entry1",
                            "name": "Thrust vs Impulse",
                            "slug": "thrust_vs_impulse",
                            "saved_at": "2026-01-01 00:00:00",
                            "updated_at": "2026-01-01 00:00:00",
                            "plot_definition": {
                                "output": "thrust",
                                "input1": "impulse bit",
                                "member_runs": ["RunA"],
                                "stats": ["mean"],
                                "fit_enabled": True,
                                "fit_mode": "polynomial",
                                "polynomial_degree": 2,
                                "normalize_x": True,
                                "require_min_points": 2,
                                "performance_run_type_mode": "pulsed_mode",
                                "performance_filter_mode": "match_control_period",
                                "selected_control_period": 1.5,
                            },
                            **snapshot,
                        }
                    ]
                },
            )
            self._seed_perf_source_metadata(
                db_path,
                [
                    ("SN1", "Valve", "Cathode_B"),
                    ("SN2", "Valve", "Cathode_B"),
                ],
            )
            refresh = be.td_perf_refresh_saved_equation_store(root, db_path)
            self.assertEqual(int(refresh.get("refreshed_count") or 0), 1)
            store = be.load_td_saved_performance_equations(root)
            entry = dict((store.get("entries") or [])[0])
            asset_metadata = dict(entry.get("asset_metadata") or {})
            self.assertEqual(asset_metadata.get("primary_asset_type"), "Valve")
            self.assertEqual(asset_metadata.get("primary_asset_specific_type"), "Cathode_B")

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_saved_perf_save_action_upserts_entry_and_toasts(self) -> None:
        harness = _SavedPerfActionHarness(Path(tempfile.mkdtemp()))
        with mock.patch("PySide6.QtWidgets.QInputDialog.getText", return_value=("Saved Equation", True)):
            with mock.patch("PySide6.QtWidgets.QMessageBox.question", return_value=0):
                with mock.patch("EIDAT_App_Files.ui_next.backend.load_td_saved_performance_equations", return_value={"entries": []}):
                    with mock.patch("EIDAT_App_Files.ui_next.backend.td_perf_upsert_saved_equation") as mocked_upsert:
                        harness._save_current_performance_equation()
        mocked_upsert.assert_called_once()
        saved_entry = mocked_upsert.call_args.args[1]
        self.assertEqual(saved_entry.get("name"), "Saved Equation")
        self.assertEqual(harness._toast_message, "Saved performance equation")

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_perf_control_period_combo_enables_for_control_period_surface_slice(self) -> None:
        harness = _PerfControlPeriodHarness()
        harness.cb_perf_filter_mode.setCurrentIndex(harness.cb_perf_filter_mode.findData("all_conditions"))
        harness.cb_perf_surface_model.setCurrentIndex(harness.cb_perf_surface_model.findData("quadratic_surface_control_period"))
        harness._update_perf_control_period_state()
        self.assertTrue(harness.cb_perf_control_period.isEnabled())
        self.assertEqual(harness._selected_perf_control_period(), 60.0)
        harness.cb_perf_surface_model.setCurrentIndex(harness.cb_perf_surface_model.findData("quadratic_surface"))
        harness._update_perf_control_period_state()
        self.assertFalse(harness.cb_perf_control_period.isEnabled())

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_plot_view_band_parse_accepts_blank_and_open_ranges(self) -> None:
        harness = _PlotViewBandHarness()
        self.assertIsNone(harness._parse_view_band_value("", "Y", "min"))
        self.assertEqual(harness._normalize_view_band("Y", "", 30.0), (None, 30.0))
        self.assertEqual(harness._normalize_view_band("X", 1.5, None), (1.5, None))

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_plot_view_band_parse_rejects_invalid_values(self) -> None:
        harness = _PlotViewBandHarness()
        with self.assertRaisesRegex(ValueError, "must be a number"):
            harness._parse_view_band_value("abc", "X", "min")
        with self.assertRaisesRegex(ValueError, "cannot be greater than max"):
            harness._normalize_view_band("Y", 5.0, 4.0)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_plot_view_band_controls_follow_active_mode(self) -> None:
        harness = _PlotViewBandHarness()

        harness._mode = "metrics"
        harness._refresh_plot_view_band_controls()
        self.assertFalse(harness.plot_band_frame.isHidden())
        self.assertTrue(harness.ed_plot_x_band_min.isHidden())
        self.assertFalse(harness.ed_plot_y_band_min.isHidden())

        harness._mode = "performance"
        harness._refresh_plot_view_band_controls()
        self.assertFalse(harness.ed_plot_x_band_min.isHidden())
        self.assertFalse(harness.ed_plot_y_band_min.isHidden())

        harness._mode = "curves"
        harness._refresh_plot_view_band_controls()
        self.assertTrue(harness.plot_band_frame.isHidden())

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_metrics_view_band_changes_axis_limits_only(self) -> None:
        harness = _PlotViewBandHarness()
        axis = _MockPlotAxis(xlim=(0.0, 9.0), ylim=(0.0, 100.0))
        harness.ed_plot_y_band_min.setText("10")
        harness.ed_plot_y_band_max.setText("20")
        harness._apply_plot_view_bands_to_axes(axis, mode="metrics")
        self.assertEqual(axis.get_xlim(), (0.0, 9.0))
        self.assertEqual(axis.get_ylim(), (10.0, 20.0))

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_performance_view_band_changes_axis_limits_only(self) -> None:
        harness = _PlotViewBandHarness()
        axis = _MockPlotAxis(xlim=(0.0, 9.0), ylim=(0.0, 100.0))
        harness.ed_plot_x_band_min.setText("2")
        harness.ed_plot_x_band_max.setText("4")
        harness.ed_plot_y_band_min.setText("10")
        harness.ed_plot_y_band_max.setText("20")
        harness._apply_plot_view_bands_to_axes(axis, mode="performance")
        self.assertEqual(axis.get_xlim(), (2.0, 4.0))
        self.assertEqual(axis.get_ylim(), (10.0, 20.0))

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_view_band_edits_do_not_mutate_performance_results(self) -> None:
        harness = _PlotViewBandHarness()
        results = {"mean": {"curves": {"SN1": [(1.0, 2.0, "RunA")]}}}
        harness._mode = "performance"
        harness._last_plot_def = {"mode": "performance"}
        harness._axes = _MockPlotAxis()
        harness._canvas = _MockPlotCanvas()
        harness._perf_results_by_stat = results
        harness.ed_plot_x_band_min.setText("1")
        harness.ed_plot_y_band_max.setText("20")
        harness._apply_current_plot_view_bands()
        self.assertIs(harness._perf_results_by_stat, results)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_plot_performance_does_not_store_band_keys_in_last_plot_def(self) -> None:
        harness = _PlotPerformanceHarness()
        harness._plot_performance()
        self.assertIsInstance(harness._last_plot_def, dict)
        for key in ("x_band_min", "x_band_max", "y_band_min", "y_band_max"):
            self.assertNotIn(key, harness._last_plot_def)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_plot_performance_surfaces_cp_fit_warning_as_plot_note(self) -> None:
        harness = _PlotPerformanceHarness()
        harness._perf_collect_results = lambda *args, **kwargs: (
            {
                "mean": {
                    "plot_dimension": "2d",
                    "master_model": {"fit_warning_text": "Ignored control periods for CP-surface fit: CP 180: 4 points, 2 distinct x1, 2 distinct x2 (4 points (<6))"},
                    "curves": {"SN1": [(1.0, 2.0, "RunA"), (2.0, 3.0, "RunA")]},
                    "fit_warning_text": "Ignored control periods for CP-surface fit: CP 180: 4 points, 2 distinct x1, 2 distinct x2 (4 points (<6))",
                }
            },
            ["mean"],
            "",
        )
        from PySide6 import QtWidgets

        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
            harness._plot_performance()
        warning_mock.assert_not_called()
        self.assertIn("Ignored control periods", harness._plot_note)

    @unittest.skipUnless(_have_pyside6(), "PySide6 not installed")
    def test_plot_performance_blocks_fit_error_only_when_no_master_model(self) -> None:
        harness = _PlotPerformanceHarness()
        message = "Quadratic Surface + Control Period requires at least two distinct control periods with valid surface coverage. Eligible periods: 1. CP 120: 4 points, 2 distinct x1, 2 distinct x2 (4 points (<6))"
        harness._perf_collect_results = lambda *args, **kwargs: (
            {
                "mean": {
                    "plot_dimension": "2d",
                    "master_model": {},
                    "curves": {"SN1": [(1.0, 2.0, "RunA"), (2.0, 3.0, "RunA")]},
                }
            },
            ["mean"],
            message,
        )
        from PySide6 import QtWidgets

        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning_mock:
            harness._plot_performance()
        warning_mock.assert_called_once()
        self.assertIn("Eligible periods: 1", str(warning_mock.call_args.args[2]))

    @unittest.skipUnless(_have_matplotlib(), "matplotlib not installed")
    def test_render_plot_def_ignores_legacy_band_keys(self) -> None:
        harness = _RenderPlotDefHarness()
        fig = harness._render_plot_def_to_figure(
            {
                "mode": "performance",
                "output": "output",
                "input1": "input1",
                "input2": "",
                "stats": ["mean"],
                "view_stat": "mean",
                "fit_enabled": False,
                "x_band_min": 1.0,
                "x_band_max": 2.0,
                "y_band_min": 3.0,
                "y_band_max": 4.0,
            }
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(harness.collect_calls), 1)
        self.assertEqual(harness.collect_calls[0]["plot_stats"], ["mean"])

    def test_metric_series_loader_filters_programs_and_serials(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        harness = _GlobalFilterHarness()
        rows = [
            {"observation_id": "obs_a", "serial": "SN1", "value_num": 1.0, "program_title": "Program A"},
            {"observation_id": "obs_b", "serial": "SN2", "value_num": 2.0, "program_title": "Program B"},
        ]

        with mock.patch.object(be, "td_load_metric_series", return_value=rows) as load_mock:
            result = harness._load_metric_series_for_selection("RunA", "thrust", "mean")

        load_mock.assert_called_once()
        self.assertEqual([row.get("serial") for row in result], ["SN1"])

    def test_curve_loader_filters_programs_and_serials(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        harness = _GlobalFilterHarness()
        rows = [
            {"serial": "SN1", "program_title": "Program A", "x": [0.0, 1.0], "y": [1.0, 2.0]},
            {"serial": "SN2", "program_title": "Program B", "x": [0.0, 1.0], "y": [2.0, 3.0]},
        ]

        with mock.patch.object(be, "td_load_curves", return_value=rows) as load_mock:
            result = harness._load_curves_for_selection("RunA", "thrust", "Time")

        load_mock.assert_called_once()
        self.assertEqual(load_mock.call_args.kwargs.get("serials"), ["SN1"])
        self.assertEqual([row.get("serial") for row in result], ["SN1"])

    def test_selected_perf_serials_and_highlight_use_active_filter_intersection(self) -> None:
        harness = _GlobalFilterHarness()

        self.assertEqual(harness._selected_perf_serials(), ["SN1"])
        harness._set_highlight_serials(["SN1", "SN2"])

        self.assertEqual(harness._highlight_sns, ["SN1"])
        self.assertEqual(harness._highlight_sn, "SN1")

    def test_perf_candidate_discovery_accepts_separated_x(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "Seq1", 3.0, 10.0),
                    ("SN1", "Seq2", 4.0, 11.0),
                    ("SN1", "Seq3", 5.0, 12.0),
                ],
            )

            candidates = be.td_discover_performance_candidates(db_path)
            match = next(
                (
                    item
                    for item in candidates
                    if str(item.get("display_name") or "") == "thrust vs impulse bit"
                ),
                None,
            )
            self.assertIsNotNone(match)
            self.assertEqual(int(match.get("qualifying_serial_count") or 0), 1)
            self.assertEqual(int(match.get("distinct_x_point_count") or 0), 3)
            self.assertEqual(int(match.get("min_distinct_x_points_per_serial") or 0), 3)

    def test_perf_candidate_discovery_honors_support_setting_overrides(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "Seq1", 3.0, 10.0),
                    ("SN1", "Seq2", 3.1, 11.0),
                    ("SN1", "Seq3", 3.14, 12.0),
                ],
                support_settings={
                    "perf_eq_strictness": "loose",
                    "perf_eq_point_count": "loose",
                },
            )

            candidates = be.td_discover_performance_candidates(db_path)
            match = next(
                (
                    item
                    for item in candidates
                    if str(item.get("display_name") or "") == "thrust vs impulse bit"
                ),
                None,
            )
            self.assertIsNotNone(match)
            self.assertEqual(int(match.get("qualifying_serial_count") or 0), 1)
            self.assertEqual(int(match.get("distinct_x_point_count") or 0), 2)

    def test_perf_candidate_discovery_run_type_filter_separates_steady_state_and_pm(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "SS1", 1.0, 10.0),
                    ("SN1", "SS2", 2.0, 20.0),
                    ("SN1", "PM1", 3.0, 30.0),
                    ("SN1", "PM2", 4.0, 40.0),
                ],
                support_settings={
                    "perf_eq_strictness": "loose",
                    "perf_eq_point_count": "loose",
                },
            )

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("UPDATE td_runs SET run_type=?, control_period=? WHERE run_name IN (?, ?)", ("steady state", None, "SS1", "SS2"))
                conn.execute("UPDATE td_runs SET run_type=?, control_period=? WHERE run_name IN (?, ?)", ("pulsed mode", 120.0, "PM1", "PM2"))
                conn.execute(
                    "UPDATE td_condition_observations SET run_type=?, control_period=? WHERE run_name IN (?, ?)",
                    ("steady state", None, "SS1", "SS2"),
                )
                conn.execute(
                    "UPDATE td_condition_observations SET run_type=?, control_period=? WHERE run_name IN (?, ?)",
                    ("pulsed mode", 120.0, "PM1", "PM2"),
                )
                conn.commit()

            all_candidates = be.td_discover_performance_candidates(db_path)
            ss_candidates = be.td_discover_performance_candidates(db_path, run_type_filter="steady_state")
            pm_candidates = be.td_discover_performance_candidates(db_path, run_type_filter="pulsed_mode")

            def _match(items: list[dict]) -> dict | None:
                return next(
                    (
                        item
                        for item in items
                        if str(item.get("display_name") or "") == "thrust vs impulse bit"
                    ),
                    None,
                )

            all_match = _match(all_candidates)
            ss_match = _match(ss_candidates)
            pm_match = _match(pm_candidates)
            self.assertIsNotNone(all_match)
            self.assertIsNotNone(ss_match)
            self.assertIsNotNone(pm_match)
            self.assertEqual(int((all_match or {}).get("source_point_count") or 0), 4)
            self.assertEqual(int((ss_match or {}).get("source_point_count") or 0), 2)
            self.assertEqual(int((pm_match or {}).get("source_point_count") or 0), 2)

    def test_perf_collect_equation_export_rows_honor_run_type_filter(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "SS1", 1.0, 10.0),
                    ("SN1", "SS2", 2.0, 20.0),
                    ("SN1", "PM1", 3.0, 30.0),
                    ("SN1", "PM2", 4.0, 40.0),
                ],
            )

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE td_condition_observations SET run_type=?, control_period=? WHERE run_name IN (?, ?)",
                    ("steady state", None, "SS1", "SS2"),
                )
                conn.execute(
                    "UPDATE td_condition_observations SET run_type=?, control_period=? WHERE run_name IN (?, ?)",
                    ("pulsed mode", 120.0, "PM1", "PM2"),
                )
                conn.commit()

            run_specs = [
                {
                    "run_name": run_name,
                    "display_name": run_name,
                    "output_column": "thrust",
                    "output_units": "lbf",
                    "input1_column": "impulse bit",
                    "input1_units": "mN-s",
                    "input2_column": "",
                    "input2_units": "",
                }
                for run_name in ("SS1", "SS2", "PM1", "PM2")
            ]

            ss_rows = be.td_perf_collect_equation_export_rows(
                db_path,
                run_specs=run_specs,
                run_type_filter="steady_state",
            )
            pm_rows = be.td_perf_collect_equation_export_rows(
                db_path,
                run_specs=run_specs,
                run_type_filter="pulsed_mode",
            )

            self.assertEqual({str(row.get("run_name") or "") for row in ss_rows}, {"SS1", "SS2"})
            self.assertEqual({str(row.get("run_name") or "") for row in pm_rows}, {"PM1", "PM2"})

    def test_perf_candidate_discovery_legacy_support_workbook_uses_defaults(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db_path = self._seed_perf_candidate_db(
                root,
                rows=[
                    ("SN1", "Seq1", 3.0, 10.0),
                    ("SN1", "Seq2", 3.1, 11.0),
                    ("SN1", "Seq3", 3.14, 12.0),
                ],
                legacy_support_only=True,
            )

            support_cfg = be._read_td_support_workbook(root / "project.xlsx", project_dir=root)
            self.assertNotIn("perf_eq_strictness", support_cfg.get("settings") or {})

            candidates = be.td_discover_performance_candidates(db_path)
            self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
