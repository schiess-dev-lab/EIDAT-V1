from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

try:
    from ui_next import trend_auto_report as tar
except ModuleNotFoundError:
    from EIDAT_App_Files.ui_next import trend_auto_report as tar


class _FakeParagraph:
    def __init__(self, text: str, style: object) -> None:
        self.text = text
        self.style = style


class _FakeTable:
    def __init__(self, cell_text: list[list[object]], colWidths: object = None, repeatRows: int = 0, hAlign: str = "LEFT") -> None:
        self._cellvalues = cell_text
        self.colWidths = colWidths
        self.repeatRows = repeatRows
        self.hAlign = hAlign
        self.table_styles: list[object] = []

    def setStyle(self, table_style: object) -> None:
        self.table_styles.append(table_style)


class _FakeTableStyle:
    def __init__(self, commands: list[tuple]) -> None:
        self.commands = list(commands)


class _FakeSpacer:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height


class _FakePageBreak:
    pass


class _FakeColors:
    white = "#ffffff"

    @staticmethod
    def HexColor(value: str) -> str:
        return value


def _fake_reportlab() -> dict[str, object]:
    return {
        "colors": _FakeColors,
        "inch": 72.0,
        "letter": (8.5 * 72.0, 11.0 * 72.0),
        "PageBreak": _FakePageBreak,
        "Paragraph": _FakeParagraph,
        "Spacer": _FakeSpacer,
        "Table": _FakeTable,
        "TableStyle": _FakeTableStyle,
    }


def _fake_styles() -> dict[str, str]:
    return {
        "cover_title": "cover_title",
        "cover_subtitle": "cover_subtitle",
        "body": "body",
        "small": "small",
        "section": "section",
        "card_title": "card_title",
    }


def _table_text(table: _FakeTable) -> list[list[str]]:
    out: list[list[str]] = []
    for row in table._cellvalues:
        out.append([str(getattr(cell, "text", cell)) for cell in row])
    return out


def _iter_fake_tables(value: object):
    if isinstance(value, _FakeTable):
        yield value
        for row in value._cellvalues:
            for cell in row:
                yield from _iter_fake_tables(cell)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_fake_tables(item)


class _FakePlotHandle:
    def __init__(self, kind: str, kwargs: dict[str, object]) -> None:
        self.kind = kind
        self.kwargs = dict(kwargs)


class _FakeRectangle:
    def __init__(self, xy: tuple[float, float], width: float, height: float, **kwargs: object) -> None:
        self.xy = xy
        self.width = width
        self.height = height
        self.kwargs = dict(kwargs)


class _FakePlotAxes:
    def __init__(self) -> None:
        self.transData = "transData"
        self.transAxes = "transAxes"
        self.position: list[float] | None = None
        self.xlabel: tuple[object, dict[str, object]] | None = None
        self.ylabel: tuple[object, dict[str, object]] | None = None
        self.title: tuple[tuple[object, ...], dict[str, object]] | None = None
        self.scatter_calls: list[tuple[list[float], list[float], dict[str, object]]] = []
        self.plot_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.fill_between_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.axhline_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.axvline_calls: list[tuple[float, dict[str, object]]] = []
        self.xticks: list[float] = []
        self.xticklabels: list[str] = []
        self.xticklabel_kwargs: dict[str, object] = {}
        self.xlim: tuple[float, float] | None = None
        self.grid_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.patches: list[object] = []
        self.text_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.legend_call: tuple[list[object], list[str], dict[str, object]] | None = None
        self._legend_handles: list[object] = []
        self._legend_labels: list[str] = []

    def _record_legend(self, handle: object, kwargs: dict[str, object]) -> None:
        label = str(kwargs.get("label") or "")
        if label and not label.startswith("_"):
            self._legend_handles.append(handle)
            self._legend_labels.append(label)

    def set_position(self, position: list[float]) -> None:
        self.position = list(position)

    def set_title(self, *args: object, **kwargs: object) -> None:
        self.title = (args, dict(kwargs))

    def set_xlabel(self, label: object, **kwargs: object) -> None:
        self.xlabel = (label, dict(kwargs))

    def set_ylabel(self, label: object, **kwargs: object) -> None:
        self.ylabel = (label, dict(kwargs))

    def scatter(self, xs: list[float], ys: list[float], **kwargs: object) -> _FakePlotHandle:
        xs_list = [float(value) for value in xs]
        ys_list = [float(value) for value in ys]
        payload = dict(kwargs)
        self.scatter_calls.append((xs_list, ys_list, payload))
        handle = _FakePlotHandle("scatter", payload)
        self._record_legend(handle, payload)
        return handle

    def plot(self, *args: object, **kwargs: object) -> list[_FakePlotHandle]:
        payload = dict(kwargs)
        self.plot_calls.append((args, payload))
        handle = _FakePlotHandle("plot", payload)
        self._record_legend(handle, payload)
        return [handle]

    def fill_between(self, *args: object, **kwargs: object) -> _FakePlotHandle:
        payload = dict(kwargs)
        self.fill_between_calls.append((args, payload))
        handle = _FakePlotHandle("fill_between", payload)
        self._record_legend(handle, payload)
        return handle

    def axhline(self, *args: object, **kwargs: object) -> _FakePlotHandle:
        payload = dict(kwargs)
        self.axhline_calls.append((args, payload))
        handle = _FakePlotHandle("axhline", payload)
        self._record_legend(handle, payload)
        return handle

    def axvline(self, x: float, **kwargs: object) -> _FakePlotHandle:
        payload = dict(kwargs)
        self.axvline_calls.append((float(x), payload))
        return _FakePlotHandle("axvline", payload)

    def set_xticks(self, ticks: list[float]) -> None:
        self.xticks = [float(value) for value in ticks]

    def set_xticklabels(self, labels: list[str], **kwargs: object) -> None:
        self.xticklabels = [str(value) for value in labels]
        self.xticklabel_kwargs = dict(kwargs)

    def set_xlim(self, left: float, right: float) -> None:
        self.xlim = (float(left), float(right))

    def grid(self, *args: object, **kwargs: object) -> None:
        self.grid_calls.append((args, dict(kwargs)))

    def axis(self, *args: object, **kwargs: object) -> None:
        return None

    def add_patch(self, patch: object) -> None:
        self.patches.append(patch)

    def text(self, *args: object, **kwargs: object) -> None:
        self.text_calls.append((args, dict(kwargs)))

    def get_legend_handles_labels(self) -> tuple[list[object], list[str]]:
        return list(self._legend_handles), list(self._legend_labels)

    def legend(self, handles: list[object], labels: list[str], **kwargs: object) -> None:
        self.legend_call = (list(handles), list(labels), dict(kwargs))


class _FakePlotFigure:
    def __init__(self) -> None:
        self.transFigure = "transFigure"
        self.add_axes_calls: list[list[float]] = []
        self.added_axes: list[_FakePlotAxes] = []

    def add_axes(self, bounds: list[float]) -> _FakePlotAxes:
        self.add_axes_calls.append(list(bounds))
        axes = _FakePlotAxes()
        self.added_axes.append(axes)
        return axes


class _FakeHeaderPatch:
    def __init__(self) -> None:
        self.facecolors: list[str] = []

    def set_facecolor(self, value: str) -> None:
        self.facecolors.append(str(value))


class _FakeHeaderFigure:
    def __init__(self) -> None:
        self.transFigure = "transFigure"
        self.patch = _FakeHeaderPatch()
        self.artists: list[object] = []
        self.text_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def add_artist(self, artist: object) -> None:
        self.artists.append(artist)

    def text(self, *args: object, **kwargs: object) -> None:
        self.text_calls.append((args, dict(kwargs)))


class _FakePlotPdf:
    def __init__(self) -> None:
        self.saved_figures: list[object] = []

    def savefig(self, fig: object) -> None:
        self.saved_figures.append(fig)


class _FakePyplot:
    def __init__(self) -> None:
        self.closed: list[object] = []

    def close(self, fig: object) -> None:
        self.closed.append(fig)


class TestTrendAutoReportFilters(unittest.TestCase):
    def _prepare_row_specs_for_condition_rows(self, filter_rows: list[dict[str, object]]) -> list[dict]:
        selection = {
            "id": "sel-1",
            "mode": "condition",
            "run_name": "Run A",
            "member_runs": ["Run A"],
            "display_text": "Condition A",
            "run_condition": "Condition A",
        }
        run_by_name = {"Run A": {"display_name": "Run A", "default_x": "Time"}}
        base_series = [
            tar.CurveSeries(
                serial=str(row.get("serial") or ""),
                x=[0.0, 1.0],
                y=[1.0, 2.0],
                observation_id=str(row.get("observation_id") or ""),
                source_run_name=str(row.get("source_run_name") or ""),
                run_name=str(row.get("run_name") or "Run A"),
            )
            for row in filter_rows
        ]

        def _load_curves(
            _be: object,
            _db_path: Path,
            _run_name: str,
            _y_name: str,
            _x_name: str,
            *,
            selection: dict | None = None,
            serials: list[str] | None = None,
            filter_state: object = None,
            parameter_context: object = None,
            raw_names: object = None,
        ) -> list[tar.CurveSeries]:
            if isinstance(filter_state, dict) and ("suppression_voltages" in filter_state or "valve_voltages" in filter_state):
                return []
            return list(base_series)

        conn = sqlite3.connect(":memory:")
        try:
            with mock.patch.object(tar, "_resolve_curve_x_key", return_value="Time"), mock.patch.object(
                tar,
                "_tar_curve_y_columns_for_run",
                return_value=[{"name": "Pressure", "units": "psi"}],
            ), mock.patch.object(tar, "_load_curves_for_selection", side_effect=_load_curves), mock.patch.object(
                tar,
                "_load_metric_map_for_selection",
                return_value={str(row.get("serial") or ""): float(index + 1) for index, row in enumerate(filter_rows)},
            ):
                return tar._tar_prepare_row_specs(
                    be=object(),
                    db_path=Path("fake.sqlite3"),
                    conn=conn,
                    run_by_name=run_by_name,
                    selections=[selection],
                    params=["Pressure"],
                    filter_rows=filter_rows,
                    filter_state={},
                )
        finally:
            conn.close()

    def _prepare_specs_for_run_type_mode_regression(
        self,
        selections: list[dict[str, object]] | None = None,
        params: list[object] | None = None,
        params_by_family: Mapping[str, Sequence[object]] | None = None,
    ) -> dict[str, object]:
        run_by_name = {"Run A": {"display_name": "Run A", "default_x": "Time"}}
        filter_rows = [
            {
                "observation_id": "obs-ss",
                "serial": "SN-SS",
                "run_name": "Run A",
                "program_title": "Program A",
                "source_run_name": "Seq SS",
                "run_type": "steady state",
                "feed_pressure": 275.0,
                "feed_pressure_units": "psia",
                "suppression_voltage": 5.0,
                "valve_voltage": 28.0,
            },
            {
                "observation_id": "obs-pm",
                "serial": "SN-PM",
                "run_name": "Run A",
                "program_title": "Program A",
                "source_run_name": "Seq PM",
                "run_type": "pulsed mode",
                "control_period": 10.0,
                "pulse_width_on": 2.0,
                "off_time": 8.0,
                "pulse_width_units": "ms",
                "off_time_units": "ms",
                "feed_pressure": 320.0,
                "feed_pressure_units": "psia",
                "suppression_voltage": 5.0,
                "valve_voltage": 28.0,
            },
        ]
        steady_selection = {
            "id": "condition:ss",
            "mode": "condition",
            "run_name": "Run A",
            "member_runs": ["Run A"],
            "member_sequences": ["Seq SS"],
            "member_programs": ["Program A"],
            "display_text": "Steady State Condition",
            "run_condition": "Steady State Condition",
            "member_run_type_modes": ["steady_state"],
            "run_type_mode": "steady_state",
        }
        pulsed_selection = {
            "id": "condition:pm",
            "mode": "condition",
            "run_name": "Run A",
            "member_runs": ["Run A"],
            "member_sequences": ["Seq PM"],
            "member_programs": ["Program A"],
            "display_text": "Pulse Mode Condition",
            "run_condition": "Pulse Mode Condition",
            "member_run_type_modes": ["pulsed_mode"],
            "run_type_mode": "pulsed_mode",
            "member_control_periods": ["10"],
        }
        selected = [dict(item) for item in (selections or [steady_selection, pulsed_selection])]
        metric_calls: list[dict[str, object]] = []
        curve_calls: list[dict[str, object]] = []

        def _metric_rows_for_mode(mode: str) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            if mode in {"", "steady_state"}:
                rows.append(
                    {
                        "observation_id": "obs-ss",
                        "serial": "SN-SS",
                        "value_num": 11.0,
                        "program_title": "Program A",
                        "source_run_name": "Seq SS",
                        "run_type": "steady state",
                    }
                )
            if mode in {"", "pulsed_mode"}:
                rows.append(
                    {
                        "observation_id": "obs-pm",
                        "serial": "SN-PM",
                        "value_num": 29.5,
                        "program_title": "Program A",
                        "source_run_name": "Seq PM",
                        "run_type": "pulsed mode",
                        "control_period": 10.0,
                    }
                )
            return rows

        def _curve_rows_for_mode(mode: str) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            if mode in {"", "steady_state"}:
                rows.append(
                    {
                        "observation_id": "obs-ss",
                        "serial": "SN-SS",
                        "x": [0.0, 1.0, 2.0],
                        "y": [10.0, 11.0, 12.0],
                        "program_title": "Program A",
                        "source_run_name": "Seq SS",
                        "run_type": "steady state",
                    }
                )
            if mode in {"", "pulsed_mode"}:
                rows.append(
                    {
                        "observation_id": "obs-pm",
                        "serial": "SN-PM",
                        "x": [0.0, 1.0, 2.0],
                        "y": [20.0, 21.0, 22.0],
                        "program_title": "Program A",
                        "source_run_name": "Seq PM",
                        "run_type": "pulsed mode",
                    }
                )
            return rows

        def _td_load_metric_series(
            _db_path: Path,
            _run_name: str,
            _column_name: str,
            _stat: str,
            **kwargs: object,
        ) -> list[dict[str, object]]:
            metric_source = str(kwargs.get("metric_source") or "")
            run_type_filter = str(kwargs.get("run_type_filter") or "")
            metric_calls.append(
                {
                    "metric_source": metric_source,
                    "run_type_filter": run_type_filter,
                    "source_run_name": str(kwargs.get("source_run_name") or ""),
                }
            )
            if metric_source == "aggregate":
                if run_type_filter == "pulsed_mode":
                    return []
                return [
                    {
                        "observation_id": "agg-ss",
                        "serial": "SN-SS",
                        "value_num": 11.5,
                        "program_title": "Program A",
                        "source_run_name": "",
                        "run_type": "",
                    }
                ]
            return _metric_rows_for_mode(run_type_filter)

        def _td_load_curves(
            _db_path: Path,
            _run_name: str,
            _column_name: str,
            _x_name: str,
            **kwargs: object,
        ) -> list[dict[str, object]]:
            run_type_filter = str(kwargs.get("run_type_filter") or "")
            curve_calls.append(
                {
                    "run_type_filter": run_type_filter,
                    "source_run_name": str(kwargs.get("source_run_name") or ""),
                }
            )
            if run_type_filter:
                return _curve_rows_for_mode(run_type_filter)
            return [
                {
                    "observation_id": "agg-curve-ss",
                    "serial": "SN-SS",
                    "x": [0.0, 1.0, 2.0],
                    "y": [10.0, 11.0, 12.0],
                    "program_title": "Program A",
                    "source_run_name": "",
                    "run_type": "",
                }
            ]

        fake_be = SimpleNamespace(
            TD_METRIC_PLOT_SOURCE_AGGREGATE="aggregate",
            TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES="all_sequences",
            td_load_metric_series=_td_load_metric_series,
            td_load_curves=_td_load_curves,
        )

        conn = sqlite3.connect(":memory:")
        try:
            with mock.patch.object(tar, "_resolve_curve_x_key", return_value="Time"), mock.patch.object(
                tar,
                "_tar_curve_y_columns_for_run",
                return_value=[{"name": "Pressure", "units": "psi"}],
            ):
                specs = tar._tar_prepare_row_specs(
                    be=fake_be,
                    db_path=Path("fake.sqlite3"),
                    conn=conn,
                    run_by_name=run_by_name,
                    selections=selected,
                    params=list(params or ["Pressure"]),
                    params_by_family=params_by_family,
                    filter_rows=[dict(row) for row in filter_rows],
                    filter_state={},
                    parameter_context={},
                )
        finally:
            conn.close()

        return {
            "specs": specs,
            "selections": selected,
            "filter_rows": filter_rows,
            "run_by_name": run_by_name,
            "fake_be": fake_be,
            "metric_calls": metric_calls,
            "curve_calls": curve_calls,
        }

    def test_filter_rows_respects_explicit_empty_filter_lists(self) -> None:
        rows = [
            {
                "serial": "SN-001",
                "program_title": "Program A",
                "control_period": 10.0,
                "suppression_voltage": 5.0,
            }
        ]
        self.assertEqual(tar._filter_rows_for_filter_state(rows, {"serials": []}), [])
        self.assertEqual(tar._filter_rows_for_filter_state(rows, {"programs": []}), [])

    def test_resolve_filtered_serials_honors_filter_state_and_run_scope(self) -> None:
        fake_be = SimpleNamespace(
            td_read_observation_filter_rows_from_cache=lambda _db_path: [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq A",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-002",
                    "program_title": "Program B",
                    "source_run_name": "Seq B",
                    "control_period": 20.0,
                    "suppression_voltage": 7.0,
                },
            ]
        )
        options = {
            "filter_state": {
                "programs": ["Program A"],
                "serials": ["SN-001"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            },
            "run_selections": [
                {
                    "member_sequences": ["Seq A"],
                    "member_programs": ["Program A"],
                }
            ],
        }

        resolved = tar._resolve_filtered_serials(
            fake_be,
            Path("cache.sqlite3"),
            ["SN-001", "SN-002"],
            options,
        )
        self.assertEqual(resolved, ["SN-001"])

    def test_initial_analysis_options_strip_suppression_and_ignore_filtered_serials(self) -> None:
        fake_be = SimpleNamespace(
            td_read_observation_filter_rows_from_cache=lambda _db_path: [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq A",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-002",
                    "program_title": "Program A",
                    "source_run_name": "Seq A",
                    "control_period": 10.0,
                    "suppression_voltage": 7.0,
                },
            ]
        )
        options = {
            "filter_state": {
                "programs": ["Program A"],
                "serials": ["SN-001", "SN-002"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
                "valve_voltages": ["28"],
            },
            "filtered_serials": ["SN-001"],
            "run_selections": [
                {
                    "member_sequences": ["Seq A"],
                    "member_programs": ["Program A"],
                    "program_title": "Program A",
                    "member_valve_voltages": ["28"],
                    "valve_voltage": "28",
                }
            ],
        }

        initial_options = tar._tar_initial_analysis_options(options)
        self.assertEqual(
            initial_options["filter_state"],
            {
                "control_periods": ["10"],
            },
        )
        self.assertEqual(initial_options["run_selections"], [{"member_sequences": ["Seq A"]}])
        self.assertNotIn("filtered_serials", initial_options)

        resolved = tar._resolve_filtered_serials(
            fake_be,
            Path("cache.sqlite3"),
            ["SN-001", "SN-002"],
            initial_options,
        )
        self.assertEqual(resolved, ["SN-001", "SN-002"])

    def test_build_quick_summary_includes_program_serial_watch_and_comparison_programs(self) -> None:
        ctx = {
            "hi": ["SN-001"],
            "all_serials": ["SN-001", "SN-002", "SN-003"],
            "pair_specs": [
                {"base_condition_label": "Condition A"},
                {"base_condition_label": "Condition B"},
            ],
            "meta_by_sn": {
                "SN-001": {"program_title": "Program A"},
                "SN-002": {"program_title": "Program B"},
                "SN-003": {"program_title": "Program C"},
            },
            "nonpass_findings": [{"param": "Flow"}, {"param": "Pressure"}],
            "comparison_rows": [
                {
                    "final_suppression_voltage_label": "5",
                    "final_valve_voltage_label": "28",
                    "prepass_gate_mode": "noise_normalized_rms_to_certifying_program",
                    "prepass_included_programs": ["Program A", "Program B"],
                    "prepass_excluded_programs": ["Program C"],
                }
            ],
            "filter_state": {},
        }

        summary = tar._tar_build_quick_summary(ctx)

        self.assertEqual(summary["certifying_programs"], ["Program A"])
        self.assertEqual(summary["certified_serials"], ["SN-001"])
        self.assertEqual(summary["selected_run_conditions"], ["Condition A", "Condition B"])
        self.assertEqual(summary["watch_parameters"], ["Flow", "Pressure"])
        self.assertEqual(summary["comparison_programs"], ["Program B", "Program C"])
        self.assertEqual(summary["prepass_gate_modes"], ["Noise-aware RMS gate"])
        self.assertEqual(summary["prepass_admitted_programs"], ["Program A", "Program B"])
        self.assertEqual(summary["prepass_excluded_programs"], ["Program C"])
        self.assertEqual(summary["initial_suppression_voltage"], "All")
        self.assertEqual(summary["p8_suppression_voltage"], "5")
        self.assertEqual(summary["p8_valve_voltage"], "28")
        self.assertTrue(any("Programs Compared: Program B, Program C" in line for line in summary["lines"]))
        self.assertTrue(any("Pre-pass Gate: Noise-aware RMS gate" in line for line in summary["lines"]))
        self.assertIn("Suppression Voltage: 5", summary["lines"])
        self.assertIn("Valve Voltage: 28", summary["lines"])
        self.assertFalse(any(line.startswith("P8 ") for line in summary["lines"]))

    def test_build_quick_summary_lines_show_display_serial_for_composite_source_keys(self) -> None:
        composite_serial = "Program A / Valve / Injector / SN-001 / source_a"
        other_serial = "Program B / Valve / Pilot / SN-002 / source_b"
        ctx = {
            "hi": [composite_serial],
            "all_serials": [composite_serial, other_serial],
            "pair_specs": [],
            "meta_by_sn": {
                composite_serial: {"program_title": "Program A", "serial_number": "SN-001"},
                other_serial: {"program_title": "Program B", "serial_number": "SN-002"},
            },
            "comparison_rows": [],
            "filter_state": {},
        }

        summary = tar._tar_build_quick_summary(ctx)

        self.assertEqual(summary["certified_serials"], [composite_serial])
        self.assertIn("Certified Serial(s): SN-001", summary["lines"])
        self.assertFalse(any(composite_serial in line for line in summary["lines"]))

    def test_build_quick_summary_lines_show_display_serial_for_prefixed_composite_source_keys(self) -> None:
        composite_serial = r"C:\repo\EIDAT Support\doc_a / Program A / Valve / Injector / SN-001 / source_a"
        other_serial = r"C:\repo\EIDAT Support\doc_b / Program B / Valve / Pilot / SN-002 / source_b"
        ctx = {
            "hi": [composite_serial],
            "all_serials": [composite_serial, other_serial],
            "pair_specs": [],
            "meta_by_sn": {
                composite_serial: {"program_title": "Program A", "serial_number": "SN-001"},
                other_serial: {"program_title": "Program B", "serial_number": "SN-002"},
            },
            "comparison_rows": [],
            "filter_state": {},
        }

        summary = tar._tar_build_quick_summary(ctx)

        self.assertIn("Certified Serial(s): SN-001", summary["lines"])
        self.assertFalse(any(composite_serial in line for line in summary["lines"]))

    def test_metadata_snapshot_lines_include_each_certified_serial(self) -> None:
        ctx = {
            "hi": ["SN-001", "SN-002"],
            "meta_by_sn": {
                "SN-001": {
                    "program_title": "Program A",
                    "similarity_group": "SG-1",
                    "acceptance_test_plan_number": "ATP-10",
                    "asset_type": "Valve",
                    "asset_specific_type": "Injector",
                    "vendor": "Vendor A",
                    "part_number": "PN-1",
                    "revision": "A",
                    "test_date": "2026-03-01",
                    "report_date": "2026-03-05",
                    "document_type": "Acceptance Test Plan",
                    "document_type_acronym": "ATP",
                },
                "SN-002": {
                    "program_title": "Program A",
                    "similarity_group": "SG-2",
                    "acceptance_test_plan_number": "ATP-11",
                    "asset_type": "Valve",
                    "asset_specific_type": "Pilot",
                    "vendor": "Vendor B",
                    "part_number": "PN-2",
                    "revision": "B",
                    "test_date": "2026-03-02",
                    "report_date": "2026-03-06",
                    "document_type": "Certification Report",
                    "document_type_acronym": "CR",
                },
            },
            "meta_note": "Workbook metadata unavailable.",
        }

        lines = tar._tar_metadata_snapshot_lines(ctx)

        self.assertTrue(any("SN-001 | Program: Program A" in line for line in lines))
        self.assertTrue(any("SN-002 | Program: Program A" in line for line in lines))
        self.assertTrue(any("Document: Acceptance Test Plan (ATP)" in line for line in lines))
        self.assertTrue(any("Document: Certification Report (CR)" in line for line in lines))
        self.assertEqual(lines[-1], "Metadata note: Workbook metadata unavailable.")

    def test_metadata_snapshot_lines_use_display_serial_for_composite_source_keys(self) -> None:
        composite_serial = "Program A / Valve / Injector / SN-001 / source_a"
        ctx = {
            "hi": [composite_serial],
            "meta_by_sn": {
                composite_serial: {
                    "serial_number": "SN-001",
                    "program_title": "Program A",
                    "similarity_group": "SG-1",
                    "acceptance_test_plan_number": "ATP-10",
                    "asset_type": "Valve",
                    "asset_specific_type": "Injector",
                    "vendor": "Vendor A",
                    "part_number": "PN-1",
                    "revision": "A",
                    "test_date": "2026-03-01",
                    "report_date": "2026-03-05",
                    "document_type": "Acceptance Test Plan",
                    "document_type_acronym": "ATP",
                }
            },
        }

        lines = tar._tar_metadata_snapshot_lines(ctx)

        self.assertTrue(any(line.startswith("SN-001 | Program: Program A") for line in lines))
        self.assertFalse(any(composite_serial in line for line in lines))

    def test_show_pooled_family_overlay_only_for_single_set(self) -> None:
        self.assertFalse(
            tar._tar_show_pooled_family_overlay(
                {
                    "member_pair_ids": ["pair-1", "pair-2"],
                    "selection_labels": ["Condition A", "Condition B"],
                }
            )
        )
        self.assertTrue(
            tar._tar_show_pooled_family_overlay(
                {
                    "member_pair_ids": ["pair-1"],
                    "selection_labels": ["Condition A"],
                }
            )
        )

    def test_grade_basis_selected_pool_uses_counts_without_program_names(self) -> None:
        text = tar._tar_grade_basis_text(
            {
                "official_pass_type": "selected_program_pool",
                "grading_basis_status": "selected_program_pool",
                "selected_programs": ["Program A", "Program B"],
                "selected_program_count": 2,
                "selected_pool_series_count": 3,
                "comparison_programs": ["Program B"],
                "comparison_program_count": 1,
                "target_excluded_comparison_series_count": 2,
                "comparison_pool_text": "2 programs (Program A, Program B), 3 series used",
                "target_comparison_text": "HI graded against: 1 program (Program B), 2 comparison series",
            }
        )

        self.assertEqual(text, "Programs Used in Comparison Series\nPrograms used: 1 | Comparison series: 2")
        self.assertNotIn("Program A", text)
        self.assertNotIn("Program B", text)

    def test_sequence_bullet_text_uses_only_sequence_nomenclature(self) -> None:
        text = tar._tar_sequence_bullet_text(
            {
                "selection_member_programs": ["Unknown Program", "Program A"],
                "selection_member_sequences": ["Seq 1", "Seq 2"],
                "selection_member_runs": ["Run 1", "Run 2"],
                "sequence_text": "Fallback Seq",
            }
        )

        self.assertEqual(text, "- Seq 1\n- Seq 2")
        self.assertNotIn("Unknown Program", text)
        self.assertNotIn("Program A", text)

    def test_prepare_row_specs_resolves_singleton_suppression_for_blank_rows(self) -> None:
        specs = self._prepare_row_specs_for_condition_rows(
            [
                {"observation_id": "obs-1", "run_name": "Run A", "serial": "SN-001", "suppression_voltage": 5.0, "valve_voltage": 28.0},
                {"observation_id": "obs-2", "run_name": "Run A", "serial": "SN-002", "suppression_voltage": "", "valve_voltage": 28.0},
            ]
        )

        self.assertEqual(len(specs), 1)
        pair_key = tar._tar_condition_combo_key("5", "28")
        self.assertEqual([pair["key"] for pair in specs[0]["condition_pairs"]], [pair_key])
        self.assertEqual(len(specs[0]["series_by_condition_key"][pair_key]), 2)
        self.assertEqual(
            [tar._td_suppression_voltage_filter_value(row) for row in specs[0]["condition_context_rows"]],
            ["5"],
        )

    def test_prepare_row_specs_resolves_singleton_valve_for_blank_rows(self) -> None:
        specs = self._prepare_row_specs_for_condition_rows(
            [
                {"observation_id": "obs-1", "run_name": "Run A", "serial": "SN-001", "suppression_voltage": 5.0, "valve_voltage": 28.0},
                {"observation_id": "obs-2", "run_name": "Run A", "serial": "SN-002", "suppression_voltage": 5.0, "valve_voltage": ""},
            ]
        )

        self.assertEqual(len(specs), 1)
        pair_key = tar._tar_condition_combo_key("5", "28")
        self.assertEqual([pair["key"] for pair in specs[0]["condition_pairs"]], [pair_key])
        self.assertEqual(len(specs[0]["series_by_condition_key"][pair_key]), 2)
        self.assertEqual(
            [tar._td_valve_voltage_filter_value(row) for row in specs[0]["condition_context_rows"]],
            ["28"],
        )

    def test_prepare_row_specs_resolves_both_voltage_fields_for_mixed_blank_rows(self) -> None:
        specs = self._prepare_row_specs_for_condition_rows(
            [
                {"observation_id": "obs-1", "run_name": "Run A", "serial": "SN-001", "suppression_voltage": 5.0, "valve_voltage": ""},
                {"observation_id": "obs-2", "run_name": "Run A", "serial": "SN-002", "suppression_voltage": "", "valve_voltage": 28.0},
            ]
        )

        self.assertEqual(len(specs), 1)
        pair_key = tar._tar_condition_combo_key("5", "28")
        self.assertEqual([pair["key"] for pair in specs[0]["condition_pairs"]], [pair_key])
        self.assertEqual(len(specs[0]["series_by_condition_key"][pair_key]), 2)

    def test_prepare_row_specs_keeps_blank_suppression_separate_when_multiple_real_values_exist(self) -> None:
        specs = self._prepare_row_specs_for_condition_rows(
            [
                {"observation_id": "obs-1", "run_name": "Run A", "serial": "SN-001", "suppression_voltage": 5.0, "valve_voltage": 28.0},
                {"observation_id": "obs-2", "run_name": "Run A", "serial": "SN-002", "suppression_voltage": 7.0, "valve_voltage": 28.0},
                {"observation_id": "obs-3", "run_name": "Run A", "serial": "SN-003", "suppression_voltage": "", "valve_voltage": 28.0},
            ]
        )

        pair_keys = [pair["key"] for pair in specs[0]["condition_pairs"]]
        self.assertEqual(
            pair_keys,
            [
                tar._tar_condition_combo_key("5", "28"),
                tar._tar_condition_combo_key("7", "28"),
                tar._tar_condition_combo_key("", "28"),
            ],
        )

    def test_prepare_row_specs_keeps_blank_valve_separate_when_multiple_real_values_exist(self) -> None:
        specs = self._prepare_row_specs_for_condition_rows(
            [
                {"observation_id": "obs-1", "run_name": "Run A", "serial": "SN-001", "suppression_voltage": 5.0, "valve_voltage": 28.0},
                {"observation_id": "obs-2", "run_name": "Run A", "serial": "SN-002", "suppression_voltage": 5.0, "valve_voltage": 30.0},
                {"observation_id": "obs-3", "run_name": "Run A", "serial": "SN-003", "suppression_voltage": 5.0, "valve_voltage": ""},
            ]
        )

        pair_keys = [pair["key"] for pair in specs[0]["condition_pairs"]]
        self.assertEqual(
            pair_keys,
            [
                tar._tar_condition_combo_key("5", "28"),
                tar._tar_condition_combo_key("5", "30"),
                tar._tar_condition_combo_key("5", ""),
            ],
        )

    def test_resolve_params_for_report_groups_raw_names_under_normalized_selection(self) -> None:
        fake_be = SimpleNamespace(
            td_build_parameter_selector_options=lambda _ctx, **_kwargs: [
                {
                    "value": "td_param:thrust_nominal",
                    "display_name": "Thrust nominal",
                    "preferred_units": "N",
                    "raw_names": ["thrust-end", "thrust-normalized"],
                }
            ],
            td_parameter_selection_raw_names=lambda _ctx, _value, **_kwargs: ["thrust-end", "thrust-normalized"],
            td_parameter_value_display_name=lambda _ctx, _value, **_kwargs: "Thrust nominal",
        )
        conn = sqlite3.connect(":memory:")
        try:
            with mock.patch.object(
                tar,
                "_tar_y_column_catalog",
                return_value={
                    tar._norm_key("thrust-end"): {"name": "thrust-end", "units": "N"},
                    tar._norm_key("thrust-normalized"): {"name": "thrust-normalized", "units": "N"},
                },
            ):
                groups, by_raw = tar._tar_resolve_params_for_report(
                    fake_be,
                    Path("fake.sqlite3"),
                    conn,
                    runs=["Run A"],
                    options={"params": ["td_param:thrust_nominal"]},
                    parameter_context={},
                )
        finally:
            conn.close()

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["param_key"], "td_param:thrust_nominal")
        self.assertEqual(groups[0]["display_name"], "Thrust nominal")
        self.assertEqual(groups[0]["display_units"], "N")
        self.assertEqual(groups[0]["raw_params"], ["thrust-end", "thrust-normalized"])
        self.assertEqual(by_raw[tar._norm_key("thrust-end")]["selection_value"], "td_param:thrust_nominal")
        self.assertEqual(by_raw[tar._norm_key("thrust-normalized")]["display_name"], "Thrust nominal")

    def test_resolve_selected_params_prefers_family_specific_lists_with_legacy_fallback(self) -> None:
        options = {
            "params": ["legacy_param"],
            "pm_params": ["pm_param"],
            "ss_params": ["ss_param"],
        }
        self.assertEqual(
            tar._resolve_selected_params(object(), runs=["Run A"], options=options, run_type_mode="pulsed_mode"),
            ["pm_param"],
        )
        self.assertEqual(
            tar._resolve_selected_params(object(), runs=["Run A"], options=options, run_type_mode="steady_state"),
            ["ss_param"],
        )
        self.assertEqual(
            tar._resolve_selected_params(object(), runs=["Run A"], options={"params": ["legacy_param"]}, run_type_mode="pulsed_mode"),
            ["legacy_param"],
        )

    def test_load_metric_map_for_selection_unions_mapped_raw_names(self) -> None:
        fake_be = SimpleNamespace(
            td_parameter_selection_raw_names=lambda _ctx, _value, **_kwargs: ["thrust-end", "thrust-normalized"],
            td_parameter_selection_matches=lambda _ctx, selection_value, raw_name, *_args: (
                str(selection_value) == "td_param:thrust_nominal"
                and str(raw_name) in {"thrust-end", "thrust-normalized"}
            ),
            td_load_metric_series=lambda _db_path, _run_name, column_name, _stat, **_kwargs: (
                [
                    {
                        "observation_id": "obs-1",
                        "serial": "SN-CERT",
                        "value_num": 10.0,
                        "program_title": "Program A",
                        "source_run_name": "Seq A",
                    }
                ]
                if str(column_name) == "thrust-end"
                else [
                    {
                        "observation_id": "obs-2",
                        "serial": "SN-COMP",
                        "value_num": 12.5,
                        "program_title": "Program A",
                        "source_run_name": "Seq A",
                    }
                ]
            ),
        )

        metric_map = tar._load_metric_map_for_selection(
            fake_be,
            Path("fake.sqlite3"),
            "Run A",
            "td_param:thrust_nominal",
            "mean",
            selection={"run_name": "Run A", "member_runs": ["Run A"]},
            filter_state={},
            parameter_context={},
            raw_names=["thrust-end", "thrust-normalized"],
        )

        self.assertEqual(metric_map, {"SN-CERT": 10.0, "SN-COMP": 12.5})

    def test_load_metric_map_for_selection_falls_back_to_sequence_rows_for_pulsed_mode(self) -> None:
        fixture = self._prepare_specs_for_run_type_mode_regression(
            selections=[
                {
                    "id": "condition:pm",
                    "mode": "condition",
                    "run_name": "Run A",
                    "member_runs": ["Run A"],
                    "member_sequences": ["Seq PM"],
                    "member_programs": ["Program A"],
                    "display_text": "Pulse Mode Condition",
                    "run_condition": "Pulse Mode Condition",
                    "member_run_type_modes": ["pulsed_mode"],
                    "run_type_mode": "pulsed_mode",
                }
            ]
        )
        fake_be = fixture["fake_be"]
        metric_calls = fixture["metric_calls"]
        metric_calls.clear()

        metric_map = tar._load_metric_map_for_selection(
            fake_be,
            Path("fake.sqlite3"),
            "Run A",
            "Pressure",
            "mean",
            selection=dict((fixture["selections"] or [])[0]),
            filter_state={},
            parameter_context={},
        )

        self.assertEqual(metric_map, {"SN-PM": 29.5})
        self.assertEqual(
            [(call["metric_source"], call["run_type_filter"]) for call in metric_calls],
            [("aggregate", "pulsed_mode"), ("all_sequences", "pulsed_mode")],
        )

    def test_load_curves_for_selection_unions_mapped_raw_names(self) -> None:
        fake_be = SimpleNamespace(
            td_parameter_selection_raw_names=lambda _ctx, _value, **_kwargs: ["thrust-end", "thrust-normalized"],
            td_parameter_selection_matches=lambda _ctx, selection_value, raw_name, *_args: (
                str(selection_value) == "td_param:thrust_nominal"
                and str(raw_name) in {"thrust-end", "thrust-normalized"}
            ),
            td_load_curves=lambda _db_path, _run_name, column_name, _x_name, **_kwargs: (
                [
                    {
                        "observation_id": "obs-1",
                        "serial": "SN-CERT",
                        "x": [0.0, 1.0],
                        "y": [10.0, 11.0],
                        "program_title": "Program A",
                        "source_run_name": "Seq A",
                    }
                ]
                if str(column_name) == "thrust-end"
                else [
                    {
                        "observation_id": "obs-2",
                        "serial": "SN-COMP",
                        "x": [0.0, 1.0],
                        "y": [12.0, 13.0],
                        "program_title": "Program A",
                        "source_run_name": "Seq A",
                    }
                ]
            ),
        )

        series = tar._load_curves_for_selection(
            fake_be,
            Path("fake.sqlite3"),
            "Run A",
            "td_param:thrust_nominal",
            "Time",
            selection={"run_name": "Run A", "member_runs": ["Run A"]},
            filter_state={},
            parameter_context={},
            raw_names=["thrust-end", "thrust-normalized"],
        )

        self.assertEqual([curve.serial for curve in series], ["SN-CERT", "SN-COMP"])

    def test_loaders_do_not_force_run_type_filter_for_mixed_mode_selection(self) -> None:
        metric_calls: list[str] = []
        curve_calls: list[str] = []

        def _td_load_metric_series(
            _db_path: Path,
            _run_name: str,
            _column_name: str,
            _stat: str,
            **kwargs: object,
        ) -> list[dict[str, object]]:
            metric_calls.append(str(kwargs.get("run_type_filter") or ""))
            return [
                {
                    "observation_id": "obs-ss",
                    "serial": "SN-SS",
                    "value_num": 11.0,
                    "program_title": "Program A",
                    "source_run_name": "Seq SS",
                    "run_type": "steady state",
                },
                {
                    "observation_id": "obs-pm",
                    "serial": "SN-PM",
                    "value_num": 29.5,
                    "program_title": "Program A",
                    "source_run_name": "Seq PM",
                    "run_type": "pulsed mode",
                },
            ]

        def _td_load_curves(
            _db_path: Path,
            _run_name: str,
            _column_name: str,
            _x_name: str,
            **kwargs: object,
        ) -> list[dict[str, object]]:
            curve_calls.append(str(kwargs.get("run_type_filter") or ""))
            return [
                {
                    "observation_id": "obs-ss",
                    "serial": "SN-SS",
                    "x": [0.0, 1.0],
                    "y": [10.0, 11.0],
                    "program_title": "Program A",
                    "source_run_name": "Seq SS",
                    "run_type": "steady state",
                },
                {
                    "observation_id": "obs-pm",
                    "serial": "SN-PM",
                    "x": [0.0, 1.0],
                    "y": [20.0, 21.0],
                    "program_title": "Program A",
                    "source_run_name": "Seq PM",
                    "run_type": "pulsed mode",
                },
            ]

        selection = {
            "id": "condition:mixed",
            "mode": "condition",
            "run_name": "Run A",
            "member_runs": ["Run A"],
            "member_sequences": ["Seq SS", "Seq PM"],
            "member_programs": ["Program A"],
            "display_text": "Combined Conditions",
            "run_condition": "Combined Conditions",
            "member_run_type_modes": ["steady_state", "pulsed_mode"],
        }
        fake_be = SimpleNamespace(
            TD_METRIC_PLOT_SOURCE_AGGREGATE="aggregate",
            TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES="all_sequences",
            td_load_metric_series=_td_load_metric_series,
            td_load_curves=_td_load_curves,
        )

        metric_map = tar._load_metric_map_for_selection(
            fake_be,
            Path("fake.sqlite3"),
            "Run A",
            "Pressure",
            "mean",
            selection=selection,
            filter_state={},
            parameter_context={},
        )
        series = tar._load_curves_for_selection(
            fake_be,
            Path("fake.sqlite3"),
            "Run A",
            "Pressure",
            "Time",
            selection=selection,
            filter_state={},
            parameter_context={},
        )

        self.assertEqual(metric_map, {"SN-SS": 11.0, "SN-PM": 29.5})
        self.assertEqual([curve.serial for curve in series], ["SN-SS", "SN-PM"])
        self.assertTrue(metric_calls)
        self.assertTrue(curve_calls)
        self.assertTrue(all(not value for value in metric_calls))
        self.assertTrue(all(not value for value in curve_calls))

    def test_prepare_row_specs_builds_single_normalized_parameter_spec(self) -> None:
        selection = {
            "id": "sel-1",
            "mode": "sequence",
            "run_name": "Run A",
            "member_runs": ["Run A"],
            "display_text": "Sequence A",
        }
        run_by_name = {"Run A": {"display_name": "Run A", "default_x": "Time"}}
        conn = sqlite3.connect(":memory:")
        fake_be = SimpleNamespace()
        try:
            with mock.patch.object(tar, "_resolve_curve_x_key", return_value="Time"), mock.patch.object(
                tar,
                "_tar_curve_y_columns_for_run",
                return_value=[
                    {"name": "thrust-end", "units": "N"},
                    {"name": "thrust-normalized", "units": "N"},
                ],
            ), mock.patch.object(
                tar,
                "_load_curves_for_selection",
                return_value=[
                    tar.CurveSeries(serial="SN-CERT", x=[0.0, 1.0], y=[10.0, 11.0], observation_id="obs-1", run_name="Run A"),
                    tar.CurveSeries(serial="SN-COMP", x=[0.0, 1.0], y=[12.0, 13.0], observation_id="obs-2", run_name="Run A"),
                ],
            ), mock.patch.object(
                tar,
                "_load_metric_map_for_selection",
                return_value={"SN-CERT": 10.5, "SN-COMP": 12.5},
            ):
                specs = tar._tar_prepare_row_specs(
                    be=fake_be,
                    db_path=Path("fake.sqlite3"),
                    conn=conn,
                    run_by_name=run_by_name,
                    selections=[selection],
                    params=[
                        {
                            "param_key": "td_param:thrust_nominal",
                            "selection_value": "td_param:thrust_nominal",
                            "display_name": "Thrust nominal",
                            "display_units": "N",
                            "raw_params": ["thrust-end", "thrust-normalized"],
                        }
                    ],
                    filter_rows=[],
                    filter_state={},
                    parameter_context={},
                )
        finally:
            conn.close()

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["param"], "td_param:thrust_nominal")
        self.assertEqual(specs[0]["param_display"], "Thrust nominal")
        self.assertEqual(specs[0]["raw_params"], ["thrust-end", "thrust-normalized"])

    def test_pulsed_selection_survives_into_comparison_rows_and_plot_specs(self) -> None:
        fixture = self._prepare_specs_for_run_type_mode_regression(
            selections=[
                {
                    "id": "condition:pm",
                    "mode": "condition",
                    "run_name": "Run A",
                    "member_runs": ["Run A"],
                    "member_sequences": ["Seq PM"],
                    "member_programs": ["Program A"],
                    "display_text": "Pulse Mode Condition",
                    "run_condition": "Pulse Mode Condition",
                    "member_run_type_modes": ["pulsed_mode"],
                    "run_type_mode": "pulsed_mode",
                    "member_control_periods": ["10"],
                }
            ]
        )
        specs = fixture["specs"] or []
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["selection_id"], "condition:pm")
        self.assertEqual([curve.serial for curve in specs[0]["series"]], ["SN-PM"])
        self.assertEqual(specs[0]["metric_mean_by_serial"], {"SN-PM": 29.5})
        self.assertEqual([str(row.get("run_type") or "") for row in specs[0]["condition_context_rows"]], ["pulsed mode"])

        analysis = tar._tar_analyze_curve_groups(
            specs,
            hi=["SN-PM"],
            program_by_serial={"SN-PM": "Program A"},
            certifying_program="Program A",
            prepass_cfg={"enabled": False},
            grid_points=5,
            degree=2,
            normalize_x=True,
            z_pass=2.0,
            z_watch=3.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )
        ctx = {
            "filter_state": {},
            "be": fixture["fake_be"],
            "db_path": Path("fake.sqlite3"),
            "options": {"run_selections": list(fixture["selections"] or [])},
            "parameter_context": {},
        }
        comparison_rows = tar._tar_build_per_serial_comparison_rows(
            ctx,
            pair_specs=list(analysis.get("pair_specs") or []),
            all_serials=["SN-PM"],
            hi=["SN-PM"],
            initial_grade_map_by_pair_serial=dict(analysis.get("initial_grade_map_by_pair_serial") or {}),
            final_grade_map_by_pair_serial=dict(analysis.get("final_grade_map_by_pair_serial") or {}),
            finding_by_pair_serial=dict(analysis.get("finding_by_pair_serial") or {}),
        )
        self.assertEqual([row["selection_id"] for row in comparison_rows], ["condition:pm"])

        pair_specs = [dict(spec) for spec in (analysis.get("pair_specs") or []) if isinstance(spec, dict)]
        plot_ctx = {
            **ctx,
            "pair_by_id": {str(spec.get("pair_id") or ""): dict(spec) for spec in pair_specs},
            "all_serials": ["SN-PM"],
            "initial_cohort_specs": list(analysis.get("initial_cohort_specs") or []),
            "regrade_cohort_specs": list(analysis.get("regrade_cohort_specs") or []),
            "performance_plot_specs": [],
            "watch_pair_ids": [],
            "metric_stats": ["mean"],
            "include_metrics": True,
            "run_by_name": dict(fixture["run_by_name"] or {}),
            "final_grade_map_by_pair_serial": dict(analysis.get("final_grade_map_by_pair_serial") or {}),
        }
        plot_specs = tar._tar_plan_plot_specs(plot_ctx, intro_pages=0)
        self.assertEqual(
            [spec["base_condition_label"] for spec in plot_specs if spec.get("section") == "run_condition_plot_metrics"],
            ["Pulse Mode Condition"],
        )
        self.assertEqual(
            [spec["base_condition_label"] for spec in plot_specs if spec.get("section") == "run_condition_curve_overlays"],
            ["Pulse Mode Condition"],
        )

    def test_analyze_curve_groups_prefers_admitted_initial_curve_overlay_but_keeps_visual_traces(self) -> None:
        row_specs = [
            {
                "pair_id": "pair-1",
                "selection_id": "sel-1",
                "run": "Run A",
                "selection_label": "Condition A",
                "condition_label": "Condition A",
                "base_condition_label": "Condition A",
                "param": "Pressure",
                "units": "psi",
                "x_name": "Time",
                "series": [
                    tar.CurveSeries(serial="SN-001", x=[0.0, 1.0], y=[2.0, 2.0], observation_id="obs-1", run_name="Run A"),
                    tar.CurveSeries(serial="SN-010", x=[0.0, 1.0], y=[10.0, 10.0], observation_id="obs-2", run_name="Run A"),
                ],
                "metric_mean_by_serial": {"SN-001": 2.0, "SN-010": 10.0},
                "condition_context_rows": [{"condition_label": "Condition A"}],
            }
        ]

        def _fake_program_model(
            *,
            x_name: str,
            units: str,
            x_grid: list[float],
            traces_by_program: dict[str, list[float]],
            degree: int,
            normalize_x: bool,
        ) -> dict[str, object]:
            if set(traces_by_program.keys()) == {"Program A"}:
                return {
                    "x_name": x_name,
                    "units": units,
                    "domain": [0.0, 1.0],
                    "grid_points": len(x_grid),
                    "poly": {},
                    "equation": "admitted",
                    "master_y": [2.0, 2.0],
                    "std_y": [0.25, 0.25],
                    "denom": 1.0,
                }
            return {
                "x_name": x_name,
                "units": units,
                "domain": [0.0, 1.0],
                "grid_points": len(x_grid),
                "poly": {},
                "equation": "visual",
                "master_y": [10.0, 10.0],
                "std_y": [1.0, 1.0],
                "denom": 1.0,
            }

        with mock.patch.object(
            tar,
            "_tar_build_curve_model_for_series",
            return_value={"x_grid": [0.0, 1.0], "x_name": "Time", "units": "psi"},
        ), mock.patch.object(
            tar,
            "_tar_prepass_gate_details_for_program_traces",
            return_value=(["Program A"], ["Program B"], [], "patched"),
        ), mock.patch.object(
            tar,
            "_tar_build_curve_model_for_program_traces",
            side_effect=_fake_program_model,
        ), mock.patch.object(
            tar,
            "_tar_build_target_excluded_pool_model",
            return_value={
                "selected_programs": ["Program A"],
                "comparison_programs": ["Program A"],
                "selected_program_count": 1,
                "selected_pool_series_count": 1,
                "target_excluded_comparison_series_count": 2,
                "master_y": [2.0, 2.0],
                "std_y": [0.25, 0.25],
                "denom": 1.0,
                "poly": {},
            },
        ), mock.patch.object(
            tar,
            "_tar_compute_band_deviation",
            return_value={"deviation_score": 0.5, "max_band_deviation": 0.5, "max_pct": 1.0},
        ):
            analysis = tar._tar_analyze_curve_groups(
                row_specs,
                hi=["SN-001"],
                program_by_serial={"SN-001": "Program A", "SN-010": "Program B"},
                certifying_program="Program A",
                prepass_cfg={"enabled": True},
                grid_points=2,
                degree=2,
                normalize_x=True,
                z_pass=2.0,
                z_watch=3.0,
                max_abs_thr=None,
                max_pct_thr=None,
                rms_pct_thr=None,
            )

        cohort = list(analysis.get("initial_cohort_specs") or [])[0]
        self.assertEqual(cohort["master_y"], [2.0, 2.0])
        self.assertEqual(cohort["std_y"], [0.25, 0.25])
        self.assertEqual(cohort["visual_program_scope"], "admitted_programs")
        self.assertEqual({trace["serial"] for trace in (cohort.get("trace_curves") or [])}, {"SN-001", "SN-010"})

    def test_prepare_row_specs_and_outputs_preserve_steady_and_pulsed_selections(self) -> None:
        fixture = self._prepare_specs_for_run_type_mode_regression()
        specs = [dict(spec) for spec in (fixture["specs"] or []) if isinstance(spec, dict)]
        self.assertEqual({spec["selection_id"] for spec in specs}, {"condition:ss", "condition:pm"})
        spec_by_id = {str(spec.get("selection_id") or ""): spec for spec in specs}
        self.assertEqual([curve.serial for curve in spec_by_id["condition:ss"]["series"]], ["SN-SS"])
        self.assertEqual([curve.serial for curve in spec_by_id["condition:pm"]["series"]], ["SN-PM"])
        self.assertEqual(spec_by_id["condition:pm"]["metric_mean_by_serial"], {"SN-PM": 29.5})
        self.assertEqual(
            [(call["metric_source"], call["run_type_filter"]) for call in fixture["metric_calls"]],
            [
                ("aggregate", "steady_state"),
                ("all_sequences", "steady_state"),
                ("aggregate", "pulsed_mode"),
                ("all_sequences", "pulsed_mode"),
            ],
        )

        analysis = tar._tar_analyze_curve_groups(
            specs,
            hi=["SN-SS", "SN-PM"],
            program_by_serial={"SN-SS": "Program A", "SN-PM": "Program A"},
            certifying_program="Program A",
            prepass_cfg={"enabled": False},
            grid_points=5,
            degree=2,
            normalize_x=True,
            z_pass=2.0,
            z_watch=3.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )
        ctx = {
            "filter_state": {},
            "be": fixture["fake_be"],
            "db_path": Path("fake.sqlite3"),
            "options": {"run_selections": list(fixture["selections"] or [])},
            "parameter_context": {},
        }
        comparison_rows = tar._tar_build_per_serial_comparison_rows(
            ctx,
            pair_specs=list(analysis.get("pair_specs") or []),
            all_serials=["SN-SS", "SN-PM"],
            hi=["SN-SS", "SN-PM"],
            initial_grade_map_by_pair_serial=dict(analysis.get("initial_grade_map_by_pair_serial") or {}),
            final_grade_map_by_pair_serial=dict(analysis.get("final_grade_map_by_pair_serial") or {}),
            finding_by_pair_serial=dict(analysis.get("finding_by_pair_serial") or {}),
        )
        self.assertEqual({row["selection_id"] for row in comparison_rows}, {"condition:ss", "condition:pm"})

        pair_specs = [dict(spec) for spec in (analysis.get("pair_specs") or []) if isinstance(spec, dict)]
        plot_ctx = {
            **ctx,
            "pair_by_id": {str(spec.get("pair_id") or ""): dict(spec) for spec in pair_specs},
            "all_serials": ["SN-SS", "SN-PM"],
            "initial_cohort_specs": list(analysis.get("initial_cohort_specs") or []),
            "regrade_cohort_specs": list(analysis.get("regrade_cohort_specs") or []),
            "performance_plot_specs": [],
            "watch_pair_ids": [],
            "metric_stats": ["mean"],
            "include_metrics": True,
            "run_by_name": dict(fixture["run_by_name"] or {}),
            "final_grade_map_by_pair_serial": dict(analysis.get("final_grade_map_by_pair_serial") or {}),
        }
        plot_specs = tar._tar_plan_plot_specs(plot_ctx, intro_pages=0)
        self.assertEqual(
            {spec["base_condition_label"] for spec in plot_specs if spec.get("section") == "run_condition_plot_metrics"},
            {"Steady State Condition", "Pulse Mode Condition"},
        )
        self.assertEqual(
            {spec["base_condition_label"] for spec in plot_specs if spec.get("section") == "run_condition_curve_overlays"},
            {"Steady State Condition", "Pulse Mode Condition"},
        )

    def test_prepare_row_specs_uses_family_specific_params(self) -> None:
        fixture = self._prepare_specs_for_run_type_mode_regression(
            params=["Legacy Pressure"],
            params_by_family={
                "steady_state": ["Steady Pressure"],
                "pulsed_mode": ["Pulse Pressure"],
            },
        )
        specs = [dict(spec) for spec in (fixture["specs"] or []) if isinstance(spec, dict)]
        self.assertEqual({spec["selection_id"] for spec in specs}, {"condition:ss", "condition:pm"})
        spec_by_id = {str(spec.get("selection_id") or ""): spec for spec in specs}
        self.assertEqual(spec_by_id["condition:ss"]["param"], "Steady Pressure")
        self.assertEqual(spec_by_id["condition:pm"]["param"], "Pulse Pressure")

    def test_expand_report_selections_by_family_splits_mixed_conditions(self) -> None:
        expanded = tar._tar_expand_report_selections_by_family(
            [
                {
                    "id": "condition:mixed",
                    "mode": "condition",
                    "run_name": "Run A",
                    "member_runs": ["Run A"],
                    "member_sequences": ["Seq SS", "Seq PM"],
                    "member_programs": ["Program A"],
                    "display_text": "Combined Conditions",
                    "run_condition": "Combined Conditions",
                    "member_run_type_modes": ["steady_state", "pulsed_mode"],
                }
            ]
        )

        self.assertEqual({str(selection.get("id") or "") for selection in expanded}, {"condition:mixed:ss", "condition:mixed:pm"})
        self.assertEqual(
            {str(selection.get("run_type_mode") or "") for selection in expanded},
            {"steady_state", "pulsed_mode"},
        )
        self.assertEqual(
            {tuple(selection.get("member_run_type_modes") or []) for selection in expanded},
            {("steady_state",), ("pulsed_mode",)},
        )

    def test_mixed_condition_branches_survive_into_comparison_rows_and_plot_specs(self) -> None:
        mixed_selection = {
            "id": "condition:mixed",
            "mode": "condition",
            "run_name": "Run A",
            "member_runs": ["Run A"],
            "member_sequences": ["Seq SS", "Seq PM"],
            "member_programs": ["Program A"],
            "display_text": "Combined Conditions",
            "run_condition": "Combined Conditions",
            "member_run_type_modes": ["steady_state", "pulsed_mode"],
        }
        expanded = tar._tar_expand_report_selections_by_family([mixed_selection])
        fixture = self._prepare_specs_for_run_type_mode_regression(
            selections=expanded,
            params=["Legacy Pressure"],
            params_by_family={
                "steady_state": ["Steady Pressure"],
                "pulsed_mode": ["Pulse Pressure"],
            },
        )

        specs = [dict(spec) for spec in (fixture["specs"] or []) if isinstance(spec, dict)]
        self.assertEqual({spec["selection_id"] for spec in specs}, {"condition:mixed:ss", "condition:mixed:pm"})
        spec_by_id = {str(spec.get("selection_id") or ""): spec for spec in specs}
        self.assertEqual([curve.serial for curve in spec_by_id["condition:mixed:ss"]["series"]], ["SN-SS"])
        self.assertEqual([curve.serial for curve in spec_by_id["condition:mixed:pm"]["series"]], ["SN-PM"])
        self.assertEqual(spec_by_id["condition:mixed:ss"]["param"], "Steady Pressure")
        self.assertEqual(spec_by_id["condition:mixed:pm"]["param"], "Pulse Pressure")

        analysis = tar._tar_analyze_curve_groups(
            specs,
            hi=["SN-SS", "SN-PM"],
            program_by_serial={"SN-SS": "Program A", "SN-PM": "Program A"},
            certifying_program="Program A",
            prepass_cfg={"enabled": False},
            grid_points=5,
            degree=2,
            normalize_x=True,
            z_pass=2.0,
            z_watch=3.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )
        ctx = {
            "filter_state": {},
            "be": fixture["fake_be"],
            "db_path": Path("fake.sqlite3"),
            "options": {"run_selections": list(expanded)},
            "parameter_context": {},
        }
        comparison_rows = tar._tar_build_per_serial_comparison_rows(
            ctx,
            pair_specs=list(analysis.get("pair_specs") or []),
            all_serials=["SN-SS", "SN-PM"],
            hi=["SN-SS", "SN-PM"],
            initial_grade_map_by_pair_serial=dict(analysis.get("initial_grade_map_by_pair_serial") or {}),
            final_grade_map_by_pair_serial=dict(analysis.get("final_grade_map_by_pair_serial") or {}),
            finding_by_pair_serial=dict(analysis.get("finding_by_pair_serial") or {}),
        )
        self.assertEqual({row["selection_id"] for row in comparison_rows}, {"condition:mixed:ss", "condition:mixed:pm"})

        pair_specs = [dict(spec) for spec in (analysis.get("pair_specs") or []) if isinstance(spec, dict)]
        plot_ctx = {
            **ctx,
            "pair_by_id": {str(spec.get("pair_id") or ""): dict(spec) for spec in pair_specs},
            "all_serials": ["SN-SS", "SN-PM"],
            "initial_cohort_specs": list(analysis.get("initial_cohort_specs") or []),
            "regrade_cohort_specs": list(analysis.get("regrade_cohort_specs") or []),
            "performance_plot_specs": [],
            "watch_pair_ids": [],
            "metric_stats": ["mean"],
            "include_metrics": True,
            "run_by_name": dict(fixture["run_by_name"] or {}),
            "final_grade_map_by_pair_serial": dict(analysis.get("final_grade_map_by_pair_serial") or {}),
        }
        plot_specs = tar._tar_plan_plot_specs(plot_ctx, intro_pages=0)
        self.assertEqual(
            {
                pair_id
                for spec in plot_specs
                if spec.get("section") == "run_condition_plot_metrics"
                for pair_id in (spec.get("member_pair_ids") or [])
            },
            {
                "condition:mixed:ss::Run A::Steady Pressure",
                "condition:mixed:pm::Run A::Pulse Pressure",
            },
        )
        self.assertEqual(
            {
                pair_id
                for spec in plot_specs
                if spec.get("section") == "run_condition_curve_overlays"
                for pair_id in (spec.get("member_pair_ids") or [])
            },
            {
                "condition:mixed:ss::Run A::Steady Pressure",
                "condition:mixed:pm::Run A::Pulse Pressure",
            },
        )

    def test_build_per_serial_comparison_rows_tracks_initial_and_final_values(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-1",
                "selection_id": "sel-1",
                "run": "Run A",
                "run_title": "Sequence A",
                "base_condition_label": "Condition A",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence A",
                    "condition_text": "Condition A",
                },
                "param": "Pressure",
                "units": "psi",
                "initial_plot_payload": {
                    "master_y": [10.0, 20.0],
                    "y_resampled_by_sn": {
                        "SN-001": [8.0, 12.0],
                        "SN-002": [18.0, 42.0],
                    },
                },
                "regrade_plot_payloads": {
                    tar._tar_condition_combo_key("5", "28"): {
                        "master_y": [14.0, 22.0],
                        "y_resampled_by_sn": {
                            "SN-001": [12.0, 16.0],
                            "SN-002": [20.0, 32.0],
                        },
                    }
                },
                "filter_state_override": {"suppression_voltages": ["5"], "valve_voltages": ["28"]},
                "suppression_voltage_label": "5",
                "valve_voltage_label": "28",
            }
        ]
        def _metric_map_side_effect(_ctx, pair_spec, stat, *, filter_state_override=None):
            self.assertEqual(stat, "mean")
            self.assertEqual(str(pair_spec.get("pair_id") or ""), "pair-1")
            suppression = tuple((filter_state_override or {}).get("suppression_voltages") or [])
            valve = tuple((filter_state_override or {}).get("valve_voltages") or [])
            if suppression == ("5",) and valve == ("28",):
                return {"SN-001": 14.0, "SN-002": 26.0}
            return {"SN-001": 10.0, "SN-002": 30.0}

        with mock.patch.object(tar, "_tar_metric_map_for_pair", side_effect=_metric_map_side_effect):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-001", "SN-002"],
                hi=["SN-001", "SN-002"],
                initial_grade_map_by_pair_serial={
                    ("pair-1", "SN-001"): "PASS",
                    ("pair-1", "SN-002"): "PASS",
                },
                final_grade_map_by_pair_serial={
                    ("pair-1", "SN-001"): "WATCH",
                    ("pair-1", "SN-002"): "PASS",
                },
                finding_by_pair_serial={
                    ("pair-1", "SN-001"): {
                        "regrade_applied": True,
                        "final_pass_requested": True,
                        "final_pass_available": True,
                        "final_pass_applied": True,
                        "official_pass_type": "final_exact_condition",
                        "official_grade": "WATCH",
                        "official_suppression_voltage_label": "5",
                        "official_valve_voltage_label": "28",
                        "initial_status": "PASS",
                        "prepass_gate_mode": "noise_normalized_rms_to_certifying_program",
                        "prepass_included_programs": ["Program A", "Program B"],
                        "prepass_excluded_programs": [],
                        "regrade_suppression_voltage_label": "5",
                        "regrade_valve_voltage_label": "28",
                        "regrade_condition_key": tar._tar_condition_combo_key("5", "28"),
                        "initial_z": -0.75,
                        "final_z": 1.25,
                    }
                },
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_condition"], "Condition A")
        self.assertEqual(rows[0]["initial_family_mean"], 20.0)
        self.assertEqual(rows[0]["final_family_mean"], 20.0)
        self.assertEqual(rows[0]["initial_serial_mean"], 10.0)
        self.assertEqual(rows[0]["final_serial_mean"], 14.0)
        self.assertEqual(rows[0]["initial_zscore"], -0.75)
        self.assertEqual(rows[0]["final_zscore"], 1.25)
        self.assertEqual(rows[0]["initial_grade"], "PASS")
        self.assertEqual(rows[0]["final_grade"], "WATCH")
        self.assertEqual(rows[0]["grade_text"], "Initial: PASS\nFinal: WATCH")
        self.assertEqual(rows[0]["official_pass_type"], "final_exact_condition")
        self.assertEqual(rows[0]["official_baseline_mean"], 20.0)
        self.assertEqual(rows[0]["official_serial_mean"], 14.0)
        self.assertEqual(rows[0]["official_zscore"], 1.25)
        self.assertEqual(rows[0]["official_grade"], "WATCH")
        self.assertEqual(rows[0]["initial_status"], "PASS")
        self.assertEqual(rows[0]["grade_basis_text"], "Program-synced exact-condition final\nSupp: 5 | Valve: 28")
        self.assertEqual(rows[0]["initial_suppression_voltage_label"], "All")
        self.assertEqual(rows[0]["final_suppression_voltage_label"], "5")
        self.assertEqual(rows[0]["initial_valve_voltage_label"], "All")
        self.assertEqual(rows[0]["final_valve_voltage_label"], "28")
        self.assertTrue(rows[0]["regrade_applied"])

    def test_build_per_serial_comparison_rows_uses_only_normalized_parameter_label(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-1",
                "selection_id": "sel-1",
                "run": "Run A",
                "run_title": "Sequence A",
                "base_condition_label": "Condition A",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence A",
                    "condition_text": "Condition A",
                },
                "param": "td_param:thrust_nominal",
                "param_display": "Thrust nominal",
                "raw_params": ["thrust-end", "thrust-normalized"],
                "units": "N",
                "display_units": "N",
                "initial_plot_payload": {
                    "master_y": [10.0, 20.0],
                    "y_resampled_by_sn": {
                        "SN-001": [8.0, 12.0],
                    },
                },
                "regrade_plot_payloads": {},
                "suppression_voltage_label": "5",
                "valve_voltage_label": "28",
            }
        ]

        with mock.patch.object(tar, "_tar_metric_map_for_pair", return_value={"SN-001": 10.0}):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-001"],
                hi=["SN-001"],
                initial_grade_map_by_pair_serial={("pair-1", "SN-001"): "PASS"},
                final_grade_map_by_pair_serial={("pair-1", "SN-001"): "PASS"},
                finding_by_pair_serial={("pair-1", "SN-001"): {"initial_status": "PASS"}},
            )

        self.assertEqual(rows[0]["parameter"], "Thrust nominal")
        self.assertNotIn("[", rows[0]["parameter"])
        self.assertEqual(rows[0]["param"], "td_param:thrust_nominal")
        self.assertEqual(rows[0]["parameter_key"], "tdparamthrustnominal")
        self.assertEqual(rows[0]["raw_parameters"], ["thrust-end", "thrust-normalized"])

    def test_build_per_serial_comparison_rows_falls_back_when_no_regrade_override_exists(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-2",
                "selection_id": "sel-2",
                "run": "Run B",
                "run_title": "Sequence B",
                "base_condition_label": "Condition B",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence B",
                    "condition_text": "Condition B",
                },
                "param": "Flow",
                "units": "lpm",
                "initial_plot_payload": {
                    "master_y": [8.0, 12.0],
                    "y_resampled_by_sn": {
                        "SN-001": [7.0, 9.0],
                        "SN-002": [11.0, 21.0],
                    },
                },
            }
        ]
        with mock.patch.object(tar, "_tar_metric_map_for_pair", return_value={"SN-001": 8.0, "SN-002": 16.0}):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-001", "SN-002"],
                hi=["SN-001"],
                initial_grade_map_by_pair_serial={("pair-2", "SN-001"): "PASS"},
                final_grade_map_by_pair_serial={("pair-2", "SN-001"): "PASS"},
                finding_by_pair_serial={
                    ("pair-2", "SN-001"): {
                        "initial_z": -0.5,
                        "final_z": -0.5,
                        "official_pass_type": "initial_prepass",
                        "initial_status": "PASS",
                        "official_grade": "PASS",
                        "prepass_gate_mode": "noise_normalized_rms_to_certifying_program",
                        "prepass_included_programs": ["Program A", "Program B"],
                        "prepass_excluded_programs": [],
                    }
                },
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["initial_family_mean"], 12.0)
        self.assertEqual(rows[0]["final_family_mean"], 12.0)
        self.assertEqual(rows[0]["initial_serial_mean"], 8.0)
        self.assertEqual(rows[0]["final_serial_mean"], 8.0)
        self.assertEqual(rows[0]["initial_zscore"], -0.5)
        self.assertEqual(rows[0]["final_zscore"], -0.5)
        self.assertEqual(rows[0]["grade_text"], "PASS")
        self.assertEqual(rows[0]["official_pass_type"], "initial_prepass")
        self.assertEqual(rows[0]["official_baseline_mean"], 12.0)
        self.assertEqual(rows[0]["official_serial_mean"], 8.0)
        self.assertEqual(rows[0]["official_grade"], "PASS")
        self.assertEqual(rows[0]["initial_status"], "PASS")
        self.assertEqual(rows[0]["grade_basis_text"], "Initial admitted-program cohort")
        self.assertEqual(rows[0]["final_suppression_voltage_label"], "All")
        self.assertFalse(rows[0]["regrade_applied"])

    def test_build_per_serial_comparison_rows_labels_per_serial_exact_condition_final(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-2b",
                "selection_id": "sel-2b",
                "run": "Run B",
                "run_title": "Sequence B",
                "base_condition_label": "Condition B",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence B",
                    "condition_text": "Condition B",
                },
                "param": "Flow",
                "units": "lpm",
                "initial_plot_payload": {
                    "master_y": [8.0, 12.0],
                    "y_resampled_by_sn": {
                        "SN-001": [7.0, 9.0],
                        "SN-002": [11.0, 21.0],
                    },
                },
                "regrade_plot_payloads": {
                    tar._tar_condition_combo_key("7", "30"): {
                        "master_y": [9.0, 13.0],
                        "y_resampled_by_sn": {
                            "SN-001": [8.0, 10.0],
                            "SN-002": [10.0, 14.0],
                        },
                    }
                },
            }
        ]
        def _metric_map_side_effect(_ctx, pair_spec, stat, *, filter_state_override=None):
            self.assertEqual(stat, "mean")
            self.assertEqual(str(pair_spec.get("pair_id") or ""), "pair-2b")
            suppression = tuple((filter_state_override or {}).get("suppression_voltages") or [])
            valve = tuple((filter_state_override or {}).get("valve_voltages") or [])
            if suppression == ("7",) and valve == ("30",):
                return {"SN-001": 9.0, "SN-002": 12.0}
            return {"SN-001": 8.0, "SN-002": 16.0}

        with mock.patch.object(tar, "_tar_metric_map_for_pair", side_effect=_metric_map_side_effect):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-001", "SN-002"],
                hi=["SN-001"],
                initial_grade_map_by_pair_serial={("pair-2b", "SN-001"): "PASS"},
                final_grade_map_by_pair_serial={("pair-2b", "SN-001"): "WATCH"},
                finding_by_pair_serial={
                    ("pair-2b", "SN-001"): {
                        "regrade_applied": True,
                        "final_pass_requested": True,
                        "final_pass_available": True,
                        "final_pass_applied": True,
                        "official_pass_type": "final_exact_condition",
                        "official_grade": "WATCH",
                        "official_suppression_voltage_label": "7",
                        "official_valve_voltage_label": "30",
                        "initial_status": "PASS",
                        "final_selection_mode": "per_serial_exact_condition",
                        "regrade_suppression_voltage_label": "7",
                        "regrade_valve_voltage_label": "30",
                        "regrade_condition_key": tar._tar_condition_combo_key("7", "30"),
                        "initial_z": -0.5,
                        "final_z": 1.0,
                    }
                },
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["grade_basis_text"],
            "Program-synced exact-condition final\nPer-serial exact condition\nSupp: 7 | Valve: 30",
        )

    def test_build_per_serial_comparison_rows_labels_final_unavailable_fallback(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-2c",
                "selection_id": "sel-2c",
                "run": "Run C",
                "run_title": "Sequence C",
                "base_condition_label": "Condition C",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence C",
                    "condition_text": "Condition C",
                },
                "param": "Flow",
                "units": "lpm",
                "initial_plot_payload": {
                    "master_y": [8.0, 12.0],
                    "y_resampled_by_sn": {
                        "SN-001": [7.0, 9.0],
                        "SN-002": [11.0, 21.0],
                    },
                },
            }
        ]
        with mock.patch.object(tar, "_tar_metric_map_for_pair", return_value={"SN-001": 8.0, "SN-002": 16.0}):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-001", "SN-002"],
                hi=["SN-001"],
                initial_grade_map_by_pair_serial={("pair-2c", "SN-001"): "NO_DATA"},
                final_grade_map_by_pair_serial={("pair-2c", "SN-001"): "NO_DATA"},
                finding_by_pair_serial={
                    ("pair-2c", "SN-001"): {
                        "initial_skipped": True,
                        "initial_skip_reason": "no_compatible_programs",
                        "initial_status": "SKIPPED",
                        "final_pass_requested": True,
                        "final_pass_available": False,
                        "final_unavailable_reason": "missing_final_candidates_for_some_serials",
                        "official_pass_type": "initial_prepass",
                        "official_grade": "NO_DATA",
                    }
                },
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["grade_basis_text"],
            "Initial admitted-program cohort\nFinal unavailable: missing final candidates for some serials",
        )

    def test_build_per_serial_comparison_rows_omits_cert_serial_without_selected_parameter_data(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-3",
                "selection_id": "sel-3",
                "run": "Run C",
                "run_title": "Sequence C",
                "base_condition_label": "Condition C",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence C",
                    "condition_text": "Condition C",
                },
                "param": "Temperature",
                "units": "F",
                "initial_plot_payload": {
                    "master_y": [100.0, 102.0],
                    "y_resampled_by_sn": {
                        "SN-001": [99.0, 101.0],
                    },
                },
            }
        ]
        with mock.patch.object(tar, "_tar_metric_map_for_pair", return_value={"SN-001": 100.0}):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-001", "SN-002"],
                hi=["SN-001", "SN-002"],
                initial_grade_map_by_pair_serial={
                    ("pair-3", "SN-001"): "PASS",
                },
                final_grade_map_by_pair_serial={
                    ("pair-3", "SN-001"): "PASS",
                },
                finding_by_pair_serial={
                    ("pair-3", "SN-001"): {"initial_z": -0.25, "final_z": -0.25},
                },
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["serial"], "SN-001")
        self.assertEqual(rows[0]["parameter"], "Temperature")

    def test_build_per_serial_comparison_rows_emits_not_graded_row_for_selected_serial_data(self) -> None:
        ctx = {"filter_state": {}, "be": object(), "db_path": Path("fake.sqlite3"), "options": {}}
        pair_specs = [
            {
                "pair_id": "pair-ng",
                "selection_id": "sel-ng",
                "run": "Run NG",
                "run_title": "Sequence NG",
                "base_condition_label": "Condition NG",
                "selection_fields": {
                    "mode": "condition",
                    "sequence_text": "Sequence NG",
                    "condition_text": "Condition NG",
                },
                "param": "Temperature",
                "units": "F",
                "metric_mean_by_serial": {"SN-002": 100.5},
                "series": [
                    tar.CurveSeries(
                        serial="SN-002",
                        x=[0.0, 1.0],
                        y=[99.0, 102.0],
                        observation_id="obs-ng",
                        run_name="Run NG",
                    )
                ],
                "initial_plot_payload": {},
            }
        ]

        with mock.patch.object(tar, "_tar_metric_map_for_pair", return_value={}):
            rows = tar._tar_build_per_serial_comparison_rows(
                ctx,
                pair_specs=pair_specs,
                all_serials=["SN-002"],
                hi=["SN-002"],
                initial_grade_map_by_pair_serial={},
                final_grade_map_by_pair_serial={},
                finding_by_pair_serial={},
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["serial"], "SN-002")
        self.assertEqual(rows[0]["initial_status"], "NOT_GRADED")
        self.assertEqual(rows[0]["official_grade"], "NOT_GRADED")
        self.assertEqual(rows[0]["official_pass_type"], "comparison_unavailable")
        self.assertEqual(rows[0]["official_baseline_mean"], None)
        self.assertEqual(rows[0]["official_serial_mean"], 100.5)
        self.assertEqual(
            rows[0]["grade_basis_text"],
            "Informational only\nComparison unavailable: raw curve data was present but no valid comparison cohort was available",
        )

    def test_curve_plot_payload_for_pair_falls_back_to_base_series_when_cached_payload_is_empty(self) -> None:
        pair_spec = {
            "pair_id": "pair-plot",
            "run": "Run Plot",
            "param": "Pressure",
            "units": "psi",
            "x_name": "Time",
            "selection": {"run_name": "Run Plot", "member_runs": ["Run Plot"]},
            "series": [
                tar.CurveSeries(serial="SN-001", x=[0.0, 1.0], y=[10.0, 12.0], observation_id="obs-plot", run_name="Run Plot"),
            ],
            "plot_payload": {},
            "model": {},
            "filter_state_override": {},
        }
        ctx = {
            "curve_plot_cache": {"pair-plot": {}},
            "grid_points": 5,
            "report_cfg": {"model": {"degree": 3, "normalize_x": True}},
            "program_by_serial": {},
            "options": {},
            "run_by_name": {"Run Plot": {"default_x": "Time"}},
            "be": object(),
            "db_path": Path("fake.sqlite3"),
            "filter_state": {},
        }

        payload = tar._tar_curve_plot_payload_for_pair(ctx, "Run Plot", "Pressure", pair_spec=pair_spec)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["x_name"], "Time")
        self.assertIn("SN-001", payload["y_resampled_by_sn"])

    def test_not_graded_rows_are_excluded_from_rollups(self) -> None:
        rows = [
            {"initial_status": "NOT_GRADED", "official_grade": "NOT_GRADED"},
            {"initial_status": "PASS", "official_grade": "PASS"},
        ]

        counts = tar._tar_grade_counts_from_rows(rows)

        self.assertEqual(counts["PASS"], 1)
        self.assertEqual(counts["WATCH"], 0)
        self.assertEqual(counts["FAIL"], 0)
        self.assertEqual(counts["NOT_GRADED"], 1)
        self.assertEqual(tar._tar_initial_overall_status_from_rows(rows), "CERTIFIED")
        self.assertEqual(tar._tar_final_overall_status_from_rows(rows), "CERTIFIED")
        self.assertEqual(
            tar._tar_final_overall_status_from_rows([{"initial_status": "NOT_GRADED", "official_grade": "NOT_GRADED"}]),
            "",
        )

    def test_metric_program_segments_group_contiguous_programs(self) -> None:
        segments = tar._tar_metric_program_segments(
            ["SN-001", "SN-002", "SN-003", "SN-004"],
            {
                "SN-001": {"program_title": "Program A"},
                "SN-002": {"program_title": "Program A"},
                "SN-003": {"program_title": "Program B"},
                "SN-004": {"program_title": "Program A"},
            },
        )

        self.assertEqual(
            segments,
            [
                {"program": "Program A", "start": 0, "end": 1, "serials": ["SN-001", "SN-002"]},
                {"program": "Program B", "start": 2, "end": 2, "serials": ["SN-003"]},
                {"program": "Program A", "start": 3, "end": 3, "serials": ["SN-004"]},
            ],
        )

    def test_apply_metric_axis_format_shows_all_serial_ticks_guides_and_program_boxes(self) -> None:
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        fake_patches = SimpleNamespace(Rectangle=_FakeRectangle)
        fake_transforms = SimpleNamespace(blended_transform_factory=lambda *args: ("blend", args))

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
                "matplotlib.patches": fake_patches,
                "matplotlib.transforms": fake_transforms,
            },
        ):
            tar._tar_apply_metric_axis_format(
                fig,
                axes,
                serials=["SN-001", "SN-002", "SN-003"],
                meta_by_sn={
                    "SN-001": {"program_title": "Program A"},
                    "SN-002": {"program_title": "Program A"},
                    "SN-003": {"program_title": "Program B"},
                },
                metric_rows=[
                    {
                        "serial": "SN-001",
                        "program_title": "Program A",
                        "suppression_voltage": 5.0,
                        "valve_voltage": 24.0,
                    },
                    {
                        "serial": "SN-002",
                        "program_title": "Program A",
                        "suppression_voltage": 5.0,
                        "valve_voltage": 24.0,
                    },
                    {
                        "serial": "SN-003",
                        "program_title": "Program B",
                        "suppression_voltage": 7.0,
                        "valve_voltage": 28.0,
                    },
                ],
            )

        self.assertEqual(axes.position, [0.06, 0.20, 0.90, 0.60])
        self.assertEqual(axes.xlabel, ("Serial Number", {}))
        self.assertEqual(axes.xticks, [0.0, 1.0, 2.0])
        self.assertEqual(axes.xticklabels, ["SN-001", "SN-002", "SN-003"])
        self.assertEqual(axes.xticklabel_kwargs["rotation"], 90)
        self.assertEqual(axes.xticklabel_kwargs["ha"], "center")
        self.assertEqual(axes.xticklabel_kwargs["fontsize"], 6)
        self.assertEqual(axes.xlim, (-0.5, 2.5))
        self.assertEqual(len(axes.axvline_calls), 3)
        self.assertTrue(all(call[1]["color"] == tar._TAR_METRIC_GUIDE_COLOR for call in axes.axvline_calls))
        self.assertEqual(len(axes.patches), 2)
        self.assertEqual(
            [call[0][2] for call in axes.text_calls],
            [
                "Program A\nSupp: 5 | Valve: 24",
                "Program B\nSupp: 7 | Valve: 28",
            ],
        )

    def test_apply_metric_axis_format_lists_multiple_program_voltage_variants(self) -> None:
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        fake_patches = SimpleNamespace(Rectangle=_FakeRectangle)
        fake_transforms = SimpleNamespace(blended_transform_factory=lambda *args: ("blend", args))

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
                "matplotlib.patches": fake_patches,
                "matplotlib.transforms": fake_transforms,
            },
        ):
            tar._tar_apply_metric_axis_format(
                fig,
                axes,
                serials=["SN-001", "SN-002"],
                meta_by_sn={
                    "SN-001": {"program_title": "Program A"},
                    "SN-002": {"program_title": "Program A"},
                },
                metric_rows=[
                    {
                        "serial": "SN-001",
                        "program_title": "Program A",
                        "suppression_voltage": 5.0,
                        "valve_voltage": 24.0,
                    },
                    {
                        "serial": "SN-002",
                        "program_title": "Program A",
                        "suppression_voltage": 7.0,
                        "valve_voltage": 28.0,
                    },
                ],
            )

        self.assertEqual(
            [call[0][2] for call in axes.text_calls],
            ["Program A\nSupp: 5, 7 | Valve: 24, 28"],
        )

    def test_apply_metric_axis_format_keeps_program_only_label_without_voltage_data(self) -> None:
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        fake_patches = SimpleNamespace(Rectangle=_FakeRectangle)
        fake_transforms = SimpleNamespace(blended_transform_factory=lambda *args: ("blend", args))

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
                "matplotlib.patches": fake_patches,
                "matplotlib.transforms": fake_transforms,
            },
        ):
            tar._tar_apply_metric_axis_format(
                fig,
                axes,
                serials=["SN-001"],
                meta_by_sn={"SN-001": {"program_title": "Program A"}},
                metric_rows=[{"serial": "SN-001", "program_title": "Program A"}],
            )

        self.assertEqual([call[0][2] for call in axes.text_calls], ["Program A"])

    def test_apply_metric_axis_format_uses_display_serial_ticks_for_composite_source_keys(self) -> None:
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        fake_patches = SimpleNamespace(Rectangle=_FakeRectangle)
        fake_transforms = SimpleNamespace(blended_transform_factory=lambda *args: ("blend", args))
        composite_a = "Program A / Valve / Injector / SN-001 / source_a"
        composite_b = "Program B / Valve / Pilot / SN-002 / source_b"

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
                "matplotlib.patches": fake_patches,
                "matplotlib.transforms": fake_transforms,
            },
        ):
            tar._tar_apply_metric_axis_format(
                fig,
                axes,
                serials=[composite_a, composite_b],
                meta_by_sn={
                    composite_a: {"program_title": "Program A", "serial_number": "SN-001"},
                    composite_b: {"program_title": "Program B", "serial_number": "SN-002"},
                },
            )

        self.assertEqual(axes.xticklabels, ["SN-001", "SN-002"])

    def test_render_metric_cohort_page_uses_scatter_only_and_supports_both_metric_sections(self) -> None:
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        fake_patches = SimpleNamespace(Rectangle=_FakeRectangle)
        fake_transforms = SimpleNamespace(blended_transform_factory=lambda *args: ("blend", args))
        cohort_spec = {
            "cohort_id": "cohort-1",
            "param": "Pressure",
            "units": "psi",
            "x_name": "Time",
            "selection_labels": ["Condition A"],
            "member_pair_ids": ["pair-1", "pair-2"],
            "suppression_voltage_label": "5",
        }
        ctx = {
            "print_ctx": tar.PrintContext(
                printed_at="2026-04-12 09:00 MDT",
                printed_timezone="MDT",
                report_title="EIDAT Test Trend Data Analyze Auto Report",
                report_subtitle="Certification",
            ),
            "all_serials": ["SN-001", "SN-002", "SN-003"],
            "hi": ["SN-001"],
            "colors": ["#ef4444", "#2563eb"],
            "meta_by_sn": {
                "SN-001": {"program_title": "Program A"},
                "SN-002": {"program_title": "Program A"},
                "SN-003": {"program_title": "Program B"},
            },
            "pair_by_id": {
                "pair-1": {
                    "pair_id": "pair-1",
                    "selection_label": "Run A",
                    "run_title": "Run A",
                    "base_condition_label": "Condition A",
                    "selection_fields": {"sequence_text": "Run A", "condition_text": "Condition A"},
                    "condition_context_rows": [
                        {
                            "condition_label": "Steady State Condition",
                            "run_type": "pulsed mode",
                            "feed_pressure": 275.0,
                            "feed_pressure_units": "psia",
                            "suppression_voltage": 5.0,
                        }
                    ],
                },
                "pair-2": {
                    "pair_id": "pair-2",
                    "selection_label": "Run B",
                    "run_title": "Run B",
                    "base_condition_label": "Condition A",
                    "selection_fields": {"sequence_text": "Run B", "condition_text": "Condition A"},
                    "condition_context_rows": [
                        {
                            "condition_label": "Pulse Mode Condition",
                            "run_type": "steady state",
                            "feed_pressure": 320.0,
                            "feed_pressure_units": "psia",
                            "pulse_width_on": 0.02,
                            "pulse_width_units": "s",
                            "off_time": 0.08,
                            "off_time_units": "s",
                            "suppression_voltage": 5.0,
                        }
                    ],
                },
            },
        }
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        pdf = _FakePlotPdf()

        def _fake_metric_map(_ctx: object, pair_spec: object, _stat: str, *, filter_state_override: object = None) -> dict[str, float]:
            pair_id = str((pair_spec or {}).get("pair_id") or "")
            if pair_id == "pair-1":
                return {"SN-001": 10.0, "SN-002": 12.0, "SN-003": 14.0}
            return {"SN-001": 11.0, "SN-002": 13.0, "SN-003": 15.0}

        def _fake_metric_rows(_ctx: object, pair_spec: object, _stat: str, *, filter_state_override: object = None) -> list[dict[str, object]]:
            pair_id = str((pair_spec or {}).get("pair_id") or "")
            values = (
                [
                    ("SN-001", 10.0, "Program A", 5.0, 24.0),
                    ("SN-002", 12.0, "Program A", 5.0, 24.0),
                    ("SN-003", 14.0, "Program B", 7.0, 28.0),
                ]
                if pair_id == "pair-1"
                else [
                    ("SN-001", 11.0, "Program A", 5.0, 24.0),
                    ("SN-002", 13.0, "Program A", 5.0, 24.0),
                    ("SN-003", 15.0, "Program B", 7.0, 28.0),
                ]
            )
            return [
                {
                    "serial": serial,
                    "value_num": value,
                    "program_title": program,
                    "suppression_voltage": suppression,
                    "valve_voltage": valve,
                }
                for serial, value, program, suppression, valve in values
            ]

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
                "matplotlib.patches": fake_patches,
                "matplotlib.transforms": fake_transforms,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(fig, axes)) as create_page_mock, mock.patch.object(
            tar,
            "_tar_metric_map_for_pair",
            side_effect=_fake_metric_map,
        ), mock.patch.object(
            tar,
            "_tar_metric_series_for_pair",
            side_effect=_fake_metric_rows,
        ):
            run_condition_spec = tar._tar_render_metric_cohort_page(
                ctx,
                pdf,
                cohort_spec=cohort_spec,
                metric_stat="mean",
                page_number=4,
                section_title="Run Condition Metrics",
                section_key="run_condition_plot_metrics",
                grade_map_by_pair_serial={
                    ("pair-1", "SN-001"): "WATCH",
                    ("pair-2", "SN-001"): "PASS",
                },
                family_mean_label="Pooled family mean",
            )

        self.assertEqual(run_condition_spec["section"], "run_condition_plot_metrics")
        self.assertEqual(run_condition_spec["page_number"], 4)
        self.assertEqual(create_page_mock.call_args.kwargs["section_title"], "")
        self.assertEqual(create_page_mock.call_args.kwargs["section_subtitle"], "")
        self.assertEqual(
            create_page_mock.call_args.kwargs["plot_context_lines"],
            [
                "Steady State Condition | Feed Pressure: 275 psia",
                "Pulse Mode Condition | Feed Pressure: 320 psia | On Time: 0.02 s | Off Time: 0.08 s",
            ],
        )
        self.assertFalse(create_page_mock.call_args.kwargs["show_plot_backlink"])
        self.assertEqual(len(axes.plot_calls), 0)
        self.assertEqual(len(axes.scatter_calls), 4)
        self.assertEqual(axes.title[0][0], "Pressure (mean)")
        self.assertIn("bbox", axes.title[1])
        series_scatter_labels = [str(call[2].get("label") or "") for call in axes.scatter_calls if call[2].get("label")]
        highlight_calls = [call for call in axes.scatter_calls if not call[2].get("label")]
        highlight_colors = [call[2]["color"] for call in highlight_calls]
        highlight_markers = [call[2].get("marker") for call in highlight_calls]
        self.assertIn([0.0, 1.0, 2.0], [call[0] for call in axes.scatter_calls if call[2].get("label")])
        self.assertEqual(series_scatter_labels, ["Run A | Condition A", "Run B | Condition A"])
        self.assertEqual(set(highlight_colors), {"#ef4444", "#2563eb"})
        self.assertEqual(highlight_markers, ["x", "x"])
        self.assertEqual(len(axes.axvline_calls), 3)
        self.assertTrue(all(call[1]["color"] == tar._TAR_METRIC_GUIDE_COLOR for call in axes.axvline_calls))
        self.assertEqual(axes.xticklabels, ["SN-001", "SN-002", "SN-003"])
        self.assertEqual(len(axes.patches), 2)
        self.assertEqual(
            [call[0][2] for call in axes.text_calls],
            [
                "Program A\nSupp: 5 | Valve: 24",
                "Program B\nSupp: 7 | Valve: 28",
            ],
        )
        self.assertEqual(len(pdf.saved_figures), 1)
        self.assertEqual(fake_plt.closed, [fig])
        self.assertIsNotNone(axes.legend_call)
        self.assertNotIn("Pooled family mean", axes.legend_call[1])

        axes_regrade = _FakePlotAxes()
        fig_regrade = _FakePlotFigure()
        pdf_regrade = _FakePlotPdf()
        single_regrade_spec = dict(cohort_spec)
        single_regrade_spec["member_pair_ids"] = ["pair-1"]
        single_regrade_spec["selection_labels"] = ["Condition A"]
        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
                "matplotlib.patches": fake_patches,
                "matplotlib.transforms": fake_transforms,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(fig_regrade, axes_regrade)) as create_regrade_page_mock, mock.patch.object(
            tar,
            "_tar_metric_map_for_pair",
            side_effect=_fake_metric_map,
        ), mock.patch.object(
            tar,
            "_tar_metric_series_for_pair",
            side_effect=_fake_metric_rows,
        ):
            regrade_spec = tar._tar_render_metric_cohort_page(
                ctx,
                pdf_regrade,
                cohort_spec=single_regrade_spec,
                metric_stat="mean",
                page_number=7,
                section_title="Regrade Pass Metrics",
                section_key="regrade_pass_plot_metrics",
                grade_map_by_pair_serial={
                    ("pair-1", "SN-001"): "WATCH",
                    ("pair-2", "SN-001"): "PASS",
                },
                filter_state_override={"suppression_voltages": ["5"]},
                family_mean_label="Regrade family mean",
            )

        self.assertEqual(regrade_spec["section"], "regrade_pass_plot_metrics")
        self.assertEqual(regrade_spec["page_number"], 7)
        self.assertEqual(create_regrade_page_mock.call_args.kwargs["section_title"], "Regrade Pass Metrics")
        self.assertEqual(
            create_regrade_page_mock.call_args.kwargs["plot_context_lines"],
            ["Steady State Condition | Feed Pressure: 275 psia"],
        )
        self.assertEqual(axes_regrade.xticklabels, ["SN-001", "SN-002", "SN-003"])
        self.assertEqual(len(axes_regrade.scatter_calls), 2)
        self.assertEqual(len(axes_regrade.axhline_calls), 1)
        self.assertEqual(float(axes_regrade.axhline_calls[0][0][0]), 12.0)
        self.assertIsNotNone(axes_regrade.legend_call)
        self.assertIn("Regrade family mean", axes_regrade.legend_call[1])

    def test_render_run_condition_curve_cohort_page_uses_full_width_without_side_panel(self) -> None:
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        cohort_spec = {
            "cohort_id": "curve-cohort-1",
            "param": "Pressure",
            "units": "psi",
            "x_name": "Time",
            "selection_labels": ["Condition A"],
            "member_pair_ids": ["pair-1", "pair-2"],
            "trace_curves": [
                {
                    "serial": "SN-010",
                    "pair_id": "pair-1",
                    "selection_label": "Condition A | Suppression Voltage: 5 | Valve Voltage: 28",
                    "y_curve": [1.0, 1.1, 1.2],
                },
                {
                    "serial": "SN-001",
                    "pair_id": "pair-1",
                    "selection_label": "Condition A | Suppression Voltage: 5 | Valve Voltage: 28",
                    "y_curve": [1.3, 1.4, 1.5],
                },
            ],
            "x_grid": [0.0, 1.0, 2.0],
            "master_y": [1.1, 1.2, 1.3],
            "std_y": [0.1, 0.1, 0.1],
            "model": {},
        }
        ctx = {
            "print_ctx": tar.PrintContext(
                printed_at="2026-04-12 09:00 MDT",
                printed_timezone="MDT",
                report_title="EIDAT Test Trend Data Analyze Auto Report",
                report_subtitle="Certification",
            ),
            "hi": ["SN-001"],
            "colors": ["#ef4444", "#2563eb"],
            "pair_by_id": {
                "pair-1": {
                    "pair_id": "pair-1",
                    "condition_context_rows": [
                        {
                            "condition_label": "Steady State Condition",
                            "run_type": "pulsed mode",
                            "feed_pressure": 275.0,
                            "feed_pressure_units": "psia",
                        }
                    ],
                },
                "pair-2": {
                    "pair_id": "pair-2",
                    "condition_context_rows": [
                        {
                            "condition_label": "Pulse Mode Condition",
                            "run_type": "steady state",
                            "feed_pressure": 320.0,
                            "feed_pressure_units": "psia",
                            "pulse_width_on": 0.02,
                            "pulse_width_units": "s",
                            "off_time": 0.08,
                            "off_time_units": "s",
                        }
                    ],
                },
            },
            "finding_by_pair_serial": {
                ("pair-1", "SN-001"): {
                    "initial_max_pct": 2.0,
                    "initial_z": 1.5,
                }
            },
        }
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        pdf = _FakePlotPdf()

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(fig, axes)) as create_page_mock:
            plot_spec = tar._tar_render_curve_cohort_page(
                ctx,
                pdf,
                cohort_spec=cohort_spec,
                page_number=9,
                section_title="Run Condition Curve Overlay",
                section_key="run_condition_curve_overlays",
                subtitle="Pressure | Time | Condition A",
                grade_map_by_pair_serial={("pair-1", "SN-001"): "WATCH"},
                metric_prefix="initial",
                family_label="In-family, graded mean",
                band_label="In-family, graded +/-2 sigma",
                equation_label="In-family, graded equation",
            )

        self.assertEqual(plot_spec["section"], "run_condition_curve_overlays")
        self.assertEqual(plot_spec["page_number"], 9)
        self.assertEqual(create_page_mock.call_args.kwargs["section_title"], "")
        self.assertEqual(create_page_mock.call_args.kwargs["section_subtitle"], "")
        self.assertEqual(
            create_page_mock.call_args.kwargs["plot_context_lines"],
            [
                "Steady State Condition | Feed Pressure: 275 psia",
                "Pulse Mode Condition | Feed Pressure: 320 psia | On Time: 0.02 s | Off Time: 0.08 s",
            ],
        )
        self.assertFalse(create_page_mock.call_args.kwargs["show_plot_backlink"])
        self.assertIsNone(axes.position)
        self.assertEqual(axes.title[0][0], "Pressure")
        self.assertIn("bbox", axes.title[1])
        self.assertEqual(len(pdf.saved_figures), 1)
        self.assertEqual(fake_plt.closed, [fig])
        self.assertEqual(len(axes.plot_calls), 2)
        self.assertEqual(len(axes.fill_between_calls), 0)
        self.assertIsNotNone(axes.legend_call)
        self.assertIn("SN-001 | Condition A (WATCH)", axes.legend_call[1])
        self.assertFalse(any("In-family, graded" in label for label in axes.legend_call[1]))
        self.assertFalse(any("Suppression Voltage" in label for label in axes.legend_call[1] if label.startswith("SN-001")))

    def test_render_exact_condition_curve_cohort_page_omits_family_overlay_and_equation(self) -> None:
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        cohort_spec = {
            "cohort_id": "curve-cohort-2",
            "param": "Pressure",
            "units": "psi",
            "x_name": "Time",
            "selection_labels": ["Condition A"],
            "member_pair_ids": ["pair-1"],
            "suppression_voltage_label": "5",
            "valve_voltage_label": "28",
            "trace_curves": [
                {
                    "serial": "SN-010",
                    "pair_id": "pair-1",
                    "selection_label": "Condition A",
                    "y_curve": [1.0, 1.1, 1.2],
                },
                {
                    "serial": "SN-001",
                    "pair_id": "pair-1",
                    "selection_label": "Condition A",
                    "y_curve": [1.3, 1.4, 1.5],
                },
            ],
            "x_grid": [0.0, 1.0, 2.0],
            "master_y": [1.1, 1.2, 1.3],
            "std_y": [0.1, 0.1, 0.1],
            "model": {"equation": "y = 1.23x + 4.56", "poly": {"rmse": 0.07}},
        }
        ctx = {
            "print_ctx": tar.PrintContext(
                printed_at="2026-04-12 09:00 MDT",
                printed_timezone="MDT",
                report_title="EIDAT Test Trend Data Analyze Auto Report",
                report_subtitle="Certification",
            ),
            "hi": ["SN-001"],
            "colors": ["#ef4444", "#2563eb"],
            "pair_by_id": {"pair-1": {"pair_id": "pair-1", "condition_context_rows": [{"condition_label": "Condition A"}]}},
            "finding_by_pair_serial": {("pair-1", "SN-001"): {"final_max_pct": 2.0, "final_z": 1.5}},
        }
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        pdf = _FakePlotPdf()

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(fig, axes)):
            tar._tar_render_curve_cohort_page(
                ctx,
                pdf,
                cohort_spec=cohort_spec,
                page_number=10,
                section_title="Final Exact-Condition Pass",
                section_key="regrade_pass_curve_overlays",
                subtitle="Pressure | Time | Condition A",
                grade_map_by_pair_serial={("pair-1", "SN-001"): "WATCH"},
                metric_prefix="final",
                family_label="In-family, graded mean",
                band_label="In-family, graded +/-2 sigma",
                equation_label="In-family, graded equation",
            )

        self.assertIsNotNone(axes.legend_call)
        self.assertFalse(any("In-family, graded" in label for label in axes.legend_call[1]))
        self.assertEqual(len(axes.fill_between_calls), 0)
        self.assertEqual(len(fig.added_axes), 1)
        self.assertEqual(len(fig.added_axes[0].text_calls), 1)
        side_text = str(fig.added_axes[0].text_calls[0][0][2])
        self.assertNotIn("In-family, graded equation", side_text)
        self.assertNotIn("RMSE", side_text)
        self.assertIn("SN-001 | Condition A | WATCH", side_text)

    def test_render_watch_curve_page_uses_updated_family_legend_and_strips_serial_voltages(self) -> None:
        fake_plt = _FakePyplot()
        fake_matplotlib = SimpleNamespace(pyplot=fake_plt)
        pair_spec = {
            "pair_id": "pair-1",
            "run": "Run A",
            "run_title": "Sequence A",
            "param": "Pressure",
            "units": "psi",
            "selection_label": "Condition A | Suppression Voltage: 5 | Valve Voltage: 28",
            "selection": {"display_text": "Condition A"},
            "filter_state_override": {"suppression_voltages": ["5"], "valve_voltages": ["28"]},
            "condition_context_rows": [
                {
                    "condition_label": "Condition A",
                    "feed_pressure": 275.0,
                    "feed_pressure_units": "psia",
                    "suppression_voltage": 5.0,
                    "valve_voltage": 28.0,
                }
            ],
            "plot_payload": {
                "x_name": "Time",
                "x_grid": [0.0, 1.0],
                "y_resampled_by_sn": {
                    "SN-001": [1.3, 1.4],
                    "SN-010": [1.0, 1.1],
                },
                "master_y": [1.1, 1.2],
                "std_y": [0.1, 0.2],
            },
        }
        ctx = {
            "print_ctx": tar.PrintContext(
                printed_at="2026-04-12 09:00 MDT",
                printed_timezone="MDT",
                report_title="EIDAT Test Trend Data Analyze Auto Report",
                report_subtitle="Certification",
            ),
            "hi": ["SN-001"],
            "colors": ["#ef4444", "#2563eb"],
            "run_by_name": {},
            "final_grade_map_by_pair_serial": {("pair-1", "SN-001"): "WATCH"},
            "finding_by_pair_serial": {
                ("pair-1", "SN-001"): {
                    "final_max_pct": 2.0,
                    "final_rms_pct": 1.5,
                    "final_z": 1.5,
                }
            },
        }
        axes = _FakePlotAxes()
        fig = _FakePlotFigure()
        pdf = _FakePlotPdf()

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_plt,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(fig, axes)):
            plot_spec = tar._tar_render_watch_curve_page(
                ctx,
                pdf,
                pair_spec=pair_spec,
                page_number=11,
            )

        self.assertEqual(plot_spec["section"], "watch_nonpass_curves")
        self.assertEqual(axes.position, [0.06, 0.09, 0.88, 0.70])
        self.assertEqual(fig.add_axes_calls, [])
        self.assertEqual(len(axes.fill_between_calls), 3)
        for actual, expected in zip(list(axes.fill_between_calls[0][0][1]), [0.8, 0.6]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(list(axes.fill_between_calls[0][0][2]), [1.4, 1.8]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(list(axes.fill_between_calls[1][0][1]), [0.9, 0.8]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(list(axes.fill_between_calls[1][0][2]), [1.3, 1.6]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(list(axes.fill_between_calls[2][0][1]), [1.0, 1.0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(list(axes.fill_between_calls[2][0][2]), [1.2, 1.4]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(axes.fill_between_calls[0][1]["color"], "#dc2626")
        self.assertEqual(axes.fill_between_calls[1][1]["color"], "#ca8a04")
        self.assertEqual(axes.fill_between_calls[2][1]["color"], "#15803d")
        self.assertIsNotNone(axes.legend_call)
        self.assertIn(
            "In-family, graded mean | Suppression Voltage: 5 | Valve Voltage: 28",
            axes.legend_call[1],
        )
        self.assertIn(
            "In-family, graded +/-1 sigma | Suppression Voltage: 5 | Valve Voltage: 28",
            axes.legend_call[1],
        )
        self.assertIn(
            "In-family, graded +/-2 sigma | Suppression Voltage: 5 | Valve Voltage: 28",
            axes.legend_call[1],
        )
        self.assertIn(
            "In-family, graded +/-3 sigma | Suppression Voltage: 5 | Valve Voltage: 28",
            axes.legend_call[1],
        )
        self.assertIn("SN-001 | Condition A (WATCH)", axes.legend_call[1])
        self.assertFalse(any("Suppression Voltage" in label for label in axes.legend_call[1] if label.startswith("SN-001")))

    def test_build_plot_navigation_creates_compact_labels_and_ignores_non_plot_sections(self) -> None:
        navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "run_condition_plot_metrics",
                    "param": "Pressure",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": 4,
                },
                {
                    "section": "performance_plots",
                    "name": "ATP Fit",
                    "x": "Bus Voltage",
                    "y": "Flow",
                    "stat": "mean",
                    "page_number": 8,
                },
                {
                    "section": "watch_nonpass_curves",
                    "run": "Run A",
                    "param": "Pressure",
                    "serials": ["SN-001", "SN-002"],
                    "page_number": 10,
                },
                {
                    "section": "performance_equations",
                    "title": "Should Be Ignored",
                    "page_number": 12,
                },
            ]
        )

        self.assertEqual(len(navigation), 3)
        self.assertEqual(navigation[0]["section_label"], "Run Condition Metrics")
        self.assertEqual(navigation[0]["navigator_label"], "Run Metrics")
        self.assertEqual(navigation[0]["plot_label"], "Parameter: Pressure | X: Time | Stat: mean")
        self.assertEqual(navigation[0]["destination_page_index"], 3)
        self.assertEqual(navigation[1]["plot_label"], "ATP Fit | Flow vs Bus Voltage | Stat: mean")
        self.assertEqual(navigation[2]["plot_label"], "Parameter: Pressure | Serials: SN-001, SN-002")

    def test_paginate_plot_navigation_allows_continuation_and_repeats_section_header(self) -> None:
        navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "run_condition_plot_metrics",
                    "param": f"Pressure {idx}",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": idx + 1,
                }
                for idx in range(140)
            ]
        )

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            pages = tar._tar_paginate_plot_navigation(navigation)

        self.assertGreater(len(pages), 1)
        self.assertEqual(pages[0]["column_count"], 3)
        self.assertEqual(pages[0]["navigator_sections"], [])
        self.assertEqual(pages[0]["rows"][0]["text"], "Run Condition Metrics")
        self.assertEqual(pages[1]["rows"][0]["text"], "Run Condition Metrics")
        first_page_columns = [column for column in pages[0]["columns"] if column.get("rows")]
        self.assertEqual(len(first_page_columns), 3)
        self.assertTrue(all(column["rows"][0]["text"] == "Run Condition Metrics" for column in first_page_columns))

    def test_paginate_plot_navigation_uses_two_columns_before_continuing(self) -> None:
        navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "run_condition_plot_metrics",
                    "param": f"Pressure {idx}",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": idx + 1,
                }
                for idx in range(45)
            ]
        )

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            pages = tar._tar_paginate_plot_navigation(navigation)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["column_count"], 2)
        used_columns = [column for column in pages[0]["columns"] if column.get("rows")]
        self.assertEqual(len(used_columns), 2)
        self.assertEqual(used_columns[0]["rows"][0]["text"], "Run Condition Metrics")
        self.assertEqual(used_columns[1]["rows"][0]["text"], "Run Condition Metrics")

    def test_paginate_plot_navigation_embeds_run_condition_headers(self) -> None:
        navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "run_condition_plot_metrics",
                    "base_condition_label": "Condition A",
                    "suppression_voltage_label": "5",
                    "valve_voltage_label": "28",
                    "param": "Pressure",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": 4,
                },
                {
                    "section": "run_condition_plot_metrics",
                    "base_condition_label": "Condition A",
                    "suppression_voltage_label": "5",
                    "valve_voltage_label": "28",
                    "param": "Flow",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": 5,
                },
                {
                    "section": "run_condition_plot_metrics",
                    "base_condition_label": "Condition B",
                    "param": "Pressure",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": 6,
                },
            ]
        )

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            pages = tar._tar_paginate_plot_navigation(navigation)

        rows = pages[0]["rows"]
        self.assertEqual(rows[0]["text"], "Run Condition Metrics")
        self.assertEqual(rows[1]["kind"], "condition")
        self.assertEqual(rows[1]["text"], "Run Condition: Condition A | Supp 5 | Valve 28")
        self.assertEqual(rows[1]["page_text"], "4")
        self.assertEqual(rows[2]["text"], "Parameter: Pressure | X: Time | Stat: mean")
        self.assertEqual(rows[3]["text"], "Parameter: Flow | X: Time | Stat: mean")
        self.assertEqual(rows[4]["kind"], "condition")
        self.assertEqual(rows[4]["text"], "Run Condition: Condition B")
        self.assertEqual(rows[5]["text"], "Parameter: Pressure | X: Time | Stat: mean")

    def test_prepare_intro_story_with_actual_plot_specs_rebases_toc_pages(self) -> None:
        ctx = {
            "comparison_page_specs": [{"serial": "SN1"}, {"serial": "SN2"}],
        }
        actual_plot_specs = [
            {
                "section": "performance_plots",
                "name": "Actual Fit",
                "x": "Bus Voltage",
                "y": "Flow",
                "stat": "mean",
                "page_number": 999,
            }
        ]

        with mock.patch.object(tar, "_tar_plan_plot_specs") as plan_mock, mock.patch.object(
            tar, "_tar_paginate_plot_navigation", return_value=[{"rows": []}]
        ), mock.patch.object(tar, "_tar_build_intro_story", return_value=["intro"]):
            story = tar._tar_prepare_intro_story_with_navigation(
                ctx,
                intro_pages=4,
                plot_specs_override=actual_plot_specs,
                comparison_page_count=2,
            )

        plan_mock.assert_not_called()
        self.assertEqual(story, ["intro"])
        self.assertEqual(ctx["planned_plot_specs"][0]["page_number"], 7)
        self.assertEqual(ctx["plot_navigation"][0]["page_text"], "7")
        self.assertEqual(ctx["plot_navigation"][0]["plot_label"], "Actual Fit | Flow vs Bus Voltage | Stat: mean")

    def test_apply_pdf_navigation_adds_bookmarks_and_comparison_links(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "comparison_navigation_links.pdf"
            doc = fitz.open()
            comparison_page = doc.new_page()
            comparison_page.insert_text((72, 72), "Metric Chart")
            comparison_page.insert_text((220, 72), "Metric Chart")
            comparison_page.insert_text((72, 104), "Plot Curves")
            comparison_page.insert_text((220, 104), "Plot Curves")
            doc.new_page()
            doc.new_page()
            doc.save(str(pdf_path))
            doc.close()

            tar._tar_apply_pdf_navigation(
                pdf_path,
                plot_navigation=[
                    {
                        "section_key": "run_condition_plot_metrics",
                        "section_label": "Run Condition Metrics",
                        "navigator_label": "Run Metrics",
                        "plot_label": "Pressure | Time | mean",
                        "page_number": 2,
                        "destination_page_index": 1,
                    }
                ],
                comparison_chart_links=[
                    {"source_page_index": 0, "link_label": "Metric Chart", "occurrence_index": 0, "destination_page_index": 1},
                    {"source_page_index": 0, "link_label": "Metric Chart", "occurrence_index": 1, "destination_page_index": 2},
                    {"source_page_index": 0, "link_label": "Plot Curves", "occurrence_index": 0, "destination_page_index": 1},
                    {"source_page_index": 0, "link_label": "Plot Curves", "occurrence_index": 1, "destination_page_index": 2},
                ],
            )

            result = fitz.open(str(pdf_path))
            try:
                toc_rows = result.get_toc()
                self.assertEqual(toc_rows[0][1], "Run Condition Metrics")
                self.assertEqual(toc_rows[0][2], 2)
                self.assertEqual(toc_rows[1][1], "Pressure | Time | mean")
                links = result.load_page(0).get_links()
                self.assertEqual(sorted(link.get("page") for link in links), [1, 1, 2, 2])
            finally:
                result.close()

    def test_apply_pdf_navigation_uses_occurrence_order_for_repeated_comparison_labels(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "comparison_navigation_occurrence.pdf"
            doc = fitz.open()
            comparison_page = doc.new_page()
            comparison_page.insert_text((72, 72), "Metric Chart")
            comparison_page.insert_text((180, 72), "Metric Chart")
            comparison_page.insert_text((288, 72), "Metric Chart")
            doc.new_page()
            doc.new_page()
            doc.new_page()
            doc.save(str(pdf_path))
            doc.close()

            tar._tar_apply_pdf_navigation(
                pdf_path,
                plot_navigation=[],
                comparison_chart_links=[
                    {"source_page_index": 0, "link_label": "Metric Chart", "occurrence_index": 0, "destination_page_index": 1},
                    {"source_page_index": 0, "link_label": "Metric Chart", "occurrence_index": 1, "destination_page_index": 2},
                    {"source_page_index": 0, "link_label": "Metric Chart", "occurrence_index": 2, "destination_page_index": 3},
                ],
            )

            result = fitz.open(str(pdf_path))
            try:
                links = result.load_page(0).get_links()
                self.assertEqual([link.get("page") for link in links], [1, 2, 3])
                self.assertTrue(links[0]["from"].x0 < links[1]["from"].x0 < links[2]["from"].x0)
            finally:
                result.close()

    def test_apply_pdf_navigation_links_repeated_watch_grades_by_row_context(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "intro_exception_grade_links.pdf"
            doc = fitz.open()
            intro_page = doc.new_page()
            intro_page.insert_text((72, 72), "WATCH")
            intro_page.insert_text((72, 120), "SN-001")
            intro_page.insert_text((140, 120), "Condition A")
            intro_page.insert_text((300, 120), "Seq A")
            intro_page.insert_text((390, 120), "Pressure")
            intro_page.insert_text((520, 120), "WATCH")
            intro_page.insert_text((72, 150), "SN-001")
            intro_page.insert_text((140, 150), "Condition B")
            intro_page.insert_text((300, 150), "Seq B")
            intro_page.insert_text((390, 150), "Pressure")
            intro_page.insert_text((470, 150), "REQUIRES EVAL")
            doc.new_page()
            doc.new_page()
            doc.save(str(pdf_path))
            doc.close()

            tar._tar_apply_pdf_navigation(
                pdf_path,
                plot_navigation=[],
                exception_grade_links=[
                    {
                        "serial_text": "SN-001",
                        "run_condition_text": "Condition A",
                        "sequence_text": "Seq A",
                        "parameter_text": "Pressure",
                        "grade_text": "WATCH",
                        "destination_page_index": 1,
                    },
                    {
                        "serial_text": "SN-001",
                        "run_condition_text": "Condition B",
                        "sequence_text": "Seq B",
                        "parameter_text": "Pressure",
                        "grade_token": "FAIL",
                        "grade_text": "REQUIRES EVAL",
                        "destination_page_index": 2,
                    },
                ],
                intro_page_count=1,
            )

            result = fitz.open(str(pdf_path))
            try:
                links = result.load_page(0).get_links()
                self.assertEqual(sorted(link.get("page") for link in links), [1, 2])
                self.assertTrue(all(link.get("from").x0 > 450 for link in links))
            finally:
                result.close()

    def test_apply_pdf_navigation_links_repeated_watch_grades_to_matching_rows(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "intro_exception_watch_row_links.pdf"
            doc = fitz.open()
            intro_page = doc.new_page()
            intro_page.insert_text((72, 120), "SN-001")
            intro_page.insert_text((140, 120), "Condition A")
            intro_page.insert_text((300, 120), "Seq A")
            intro_page.insert_text((390, 120), "Pressure")
            intro_page.insert_text((520, 120), "WATCH")
            intro_page.insert_text((72, 150), "SN-001")
            intro_page.insert_text((140, 150), "Condition B")
            intro_page.insert_text((300, 150), "Seq B")
            intro_page.insert_text((390, 150), "Pressure")
            intro_page.insert_text((520, 150), "WATCH")
            doc.new_page()
            doc.new_page()
            doc.save(str(pdf_path))
            doc.close()

            tar._tar_apply_pdf_navigation(
                pdf_path,
                plot_navigation=[],
                exception_grade_links=[
                    {
                        "serial_text": "SN-001",
                        "run_condition_text": "Condition A",
                        "sequence_text": "Seq A",
                        "parameter_text": "Pressure",
                        "grade_text": "WATCH",
                        "destination_page_index": 1,
                    },
                    {
                        "serial_text": "SN-001",
                        "run_condition_text": "Condition B",
                        "sequence_text": "Seq B",
                        "parameter_text": "Pressure",
                        "grade_text": "WATCH",
                        "destination_page_index": 2,
                    },
                ],
                intro_page_count=1,
            )

            result = fitz.open(str(pdf_path))
            try:
                links = sorted(result.load_page(0).get_links(), key=lambda link: float(link.get("from").y0))
                self.assertEqual([link.get("page") for link in links], [1, 2])
                self.assertTrue(float(links[0].get("from").y0) < float(links[1].get("from").y0))
            finally:
                result.close()

    def test_build_exec_exception_rows_prefers_exact_condition_curves_for_requires_eval(self) -> None:
        rows = tar._tar_build_exec_exception_rows(
            {
                "plot_navigation": tar._tar_build_plot_navigation(
                    [
                        {
                            "section": "watch_nonpass_curves",
                            "pair_id": "pair-1",
                            "param": "Pressure",
                            "page_number": 4,
                        },
                        {
                            "section": "regrade_pass_curve_overlays",
                            "cohort_id": "cohort-1",
                            "param": "Pressure",
                            "x_name": "Time",
                            "page_number": 7,
                        },
                    ]
                ),
                "comparison_rows": [
                    {
                        "pair_id": "pair-1",
                        "regrade_cohort_id": "cohort-1",
                        "serial": "SN-001",
                        "run_condition": "Condition A",
                        "sequence_text": "Seq A",
                        "parameter": "Pressure",
                        "official_grade": "WATCH",
                    },
                    {
                        "pair_id": "pair-1",
                        "regrade_cohort_id": "cohort-1",
                        "serial": "SN-002",
                        "run_condition": "Condition A",
                        "sequence_text": "Seq A",
                        "parameter": "Pressure",
                        "official_grade": "FAIL",
                    },
                ],
            }
        )

        row_by_serial = {str(row.get("serial") or ""): row for row in rows}
        self.assertEqual(row_by_serial["SN-001"]["chart_target_section"], "watch_nonpass_curves")
        self.assertEqual(row_by_serial["SN-001"]["chart_target_page_index"], 3)
        self.assertEqual(row_by_serial["SN-002"]["chart_target_section"], "regrade_pass_curve_overlays")
        self.assertEqual(row_by_serial["SN-002"]["chart_target_page_index"], 6)

    def test_apply_pdf_navigation_adds_backlink_to_first_comparison_page(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "comparison_navigation_backlink.pdf"
            doc = fitz.open()
            doc.new_page()
            plot_page = doc.new_page()
            plot_page.insert_text((430, 82), tar._TAR_PLOT_BACKLINK_TEXT)
            doc.save(str(pdf_path))
            doc.close()

            tar._tar_apply_pdf_navigation(
                pdf_path,
                plot_navigation=[
                    {
                        "section_key": "run_condition_plot_metrics",
                        "section_label": "Run Condition Metrics",
                        "navigator_label": "Run Metrics",
                        "plot_label": "Pressure | Time | mean",
                        "page_number": 2,
                        "destination_page_index": 1,
                    }
                ],
                comparison_section_start_page_index=0,
            )

            result = fitz.open(str(pdf_path))
            try:
                links = result.load_page(1).get_links()
                self.assertTrue(any(link.get("page") == 0 for link in links))
            finally:
                result.close()

    def test_apply_pdf_navigation_draws_backlink_text_when_plot_page_label_is_missing(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "comparison_navigation_backlink_fallback.pdf"
            doc = fitz.open()
            doc.new_page()
            doc.new_page()
            doc.save(str(pdf_path))
            doc.close()

            tar._tar_apply_pdf_navigation(
                pdf_path,
                plot_navigation=[
                    {
                        "section_key": "run_condition_plot_metrics",
                        "section_label": "Run Condition Metrics",
                        "navigator_label": "Run Metrics",
                        "plot_label": "Pressure | Time | mean",
                        "page_number": 2,
                        "destination_page_index": 1,
                    }
                ],
                comparison_section_start_page_index=0,
            )

            result = fitz.open(str(pdf_path))
            try:
                plot_page = result.load_page(1)
                links = plot_page.get_links()
                self.assertTrue(any(link.get("page") == 0 for link in links))
                backlink_rects = list(plot_page.search_for(tar._TAR_PLOT_BACKLINK_TEXT))
                self.assertTrue(backlink_rects)
            finally:
                result.close()

    def test_apply_plot_page_header_renders_plot_backlink_text(self) -> None:
        fig = _FakeHeaderFigure()
        print_ctx = tar.PrintContext(
            printed_at="2026-04-14 09:00 MDT",
            printed_timezone="MDT",
            report_title="Auto Report",
            report_subtitle="Certification",
        )
        rectangle_mock = mock.Mock(side_effect=lambda *args, **kwargs: _FakeRectangle(*args, **kwargs))

        with mock.patch.dict("sys.modules", {"matplotlib.patches": SimpleNamespace(Rectangle=rectangle_mock)}):
            tar._apply_plot_page_header(
                fig,
                print_ctx=print_ctx,
                page_number=7,
                section_title="Run Condition Metrics | Condition A",
                section_subtitle="Pressure | Time | mean",
                plot_context_lines=["Steady State Condition | Feed Pressure: 275 psia"],
            )

        rendered_text = [str(args[2]) for args, _kwargs in fig.text_calls if len(args) >= 3]
        self.assertIn(tar._TAR_PLOT_BACKLINK_TEXT, rendered_text)
        self.assertIn("Steady State Condition | Feed Pressure: 275 psia", rendered_text)

    def test_plot_condition_header_line_formats_pulse_and_steady_state_metadata(self) -> None:
        steady = tar._tar_plot_condition_header_line(
            {
                "run_type": "steady state",
                "feed_pressure": 275.0,
                "feed_pressure_units": "psia",
            }
        )
        pulse = tar._tar_plot_condition_header_line(
            {
                "run_type": "pulsed mode",
                "feed_pressure": 320.0,
                "feed_pressure_units": "psia",
                "pulse_width_on": 0.02,
                "pulse_width_units": "s",
                "off_time": 0.08,
                "off_time_units": "s",
            }
        )

        self.assertEqual(steady, "Steady State Condition | Feed Pressure: 275 psia")
        self.assertEqual(
            pulse,
            "Pulse Mode Condition | Feed Pressure: 320 psia | On Time: 0.02 s | Off Time: 0.08 s",
        )

    def test_plot_condition_header_line_prefers_explicit_condition_label_over_run_type(self) -> None:
        steady = tar._tar_plot_condition_header_line(
            {
                "condition_label": "Steady State Condition",
                "run_type": "pulsed mode",
                "feed_pressure": 275.0,
                "feed_pressure_units": "psia",
                "pulse_width_on": 0.02,
                "pulse_width_units": "s",
                "off_time": 0.08,
                "off_time_units": "s",
            }
        )
        pulse = tar._tar_plot_condition_header_line(
            {
                "condition_label": "Pulse Mode Condition",
                "run_type": "steady state",
                "feed_pressure": 320.0,
                "feed_pressure_units": "psia",
                "pulse_width_on": 0.02,
                "pulse_width_units": "s",
                "off_time": 0.08,
                "off_time_units": "s",
            }
        )

        self.assertEqual(steady, "Steady State Condition | Feed Pressure: 275 psia")
        self.assertEqual(
            pulse,
            "Pulse Mode Condition | Feed Pressure: 320 psia | On Time: 0.02 s | Off Time: 0.08 s",
        )

    def test_build_intro_story_places_quick_summary_before_counts_and_groups_run_tables(self) -> None:
        plot_navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "run_condition_plot_metrics",
                    "param": "Pressure",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": 3,
                },
                {
                    "section": "performance_plots",
                    "name": "ATP Fit",
                    "x": "Bus Voltage",
                    "y": "Flow",
                    "stat": "mean",
                    "page_number": 7,
                },
            ]
        )
        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            ctx = {
                "print_ctx": tar.PrintContext(
                    printed_at="2026-04-12 09:00 MDT",
                    printed_timezone="MDT",
                    report_title="EIDAT Test Trend Data Analyze Auto Report",
                    report_subtitle="Certification",
                ),
                "pair_specs": [
                    {
                        "pair_id": "pair-1",
                        "param": "Pressure",
                        "selection_fields": {"mode": "condition", "display_text": "Condition A"},
                    }
                ],
                "options": {},
                "overall_by_sn": {"SN-001": "CERTIFIED"},
                "nonpass_findings": [],
                "pair_by_id": {},
                "hi": ["SN-001"],
                "params": ["Pressure"],
                "metric_stats": ["mean"],
                "include_metrics": False,
                "meta_note": "",
                "change_summary": "",
                "performance_plot_specs": [],
                "initial_overall_by_sn": {"SN-001": "CERTIFIED"},
                "final_overall_by_sn": {"SN-001": "CERTIFIED"},
                "comparison_rows": [
                    {
                        "run_condition": "Condition A",
                        "serial": "SN-001",
                        "sequence_text": "Sequence A",
                        "parameter": "Pressure",
                        "units": "psi",
                        "initial_atp_mean": 10.0,
                        "final_atp_mean": 12.0,
                        "initial_actual_mean": 9.0,
                        "final_actual_mean": 11.0,
                        "initial_delta": -1.0,
                        "final_delta": -1.0,
                        "initial_grade": "PASS",
                        "final_grade": "WATCH",
                        "initial_suppression_voltage_label": "All",
                        "final_suppression_voltage_label": "5",
                        "initial_valve_voltage_label": "All",
                        "final_valve_voltage_label": "28",
                        "regrade_applied": False,
                    }
                ],
                "meta_by_sn": {
                    "SN-001": {
                        "program_title": "Program A",
                        "part_number": "PN-1",
                        "revision": "A",
                        "acceptance_test_plan_number": "ATP-1",
                        "similarity_group": "SG-1",
                        "asset_type": "Valve",
                        "asset_specific_type": "Injector",
                        "vendor": "Vendor A",
                        "test_date": "2026-03-01",
                        "report_date": "2026-03-04",
                        "document_type": "Acceptance Test Plan",
                        "document_type_acronym": "ATP",
                    },
                    "SN-010": {"program_title": "Program B"},
                },
                "watch_pair_ids": ["pair-1"],
                "runs": ["Run A"],
                "all_serials": ["SN-001", "SN-010"],
                "plot_navigation": plot_navigation,
                "quick_summary": {
                    "lines": [
                        "Certifying Program(s): Program A",
                        "Certified Serial(s): SN-001",
                        "Selected Run Condition(s): Condition A",
                        "Watch Parameter(s): Pressure",
                        "Programs Compared: Program B",
                        "Suppression Voltage: 5",
                        "Valve Voltage: 28",
                    ],
                    "initial_suppression_voltage": "All",
                    "final_suppression_voltage": "5",
                    "initial_valve_voltage": "All",
                    "final_valve_voltage": "28",
                    "p8_suppression_voltage": "5",
                    "p8_valve_voltage": "28",
                },
            }
            story = tar._tar_build_intro_story(ctx)

        self.assertFalse(
            any(
                isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Quick Executive Summary"
                for item in story
            )
        )
        self.assertFalse(
            any(
                "Certification Scope" in cell
                for item in story
                if isinstance(item, _FakeTable)
                for row in _table_text(item)
                for cell in row
            )
        )

        exec_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeParagraph) and item.text == "Executive Summary"
        )
        scope_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Scope Item"
        )
        grading_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Grade Item"
        )
        serial_table_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "SN" and _table_text(item)[0][1] == "Initial / Final"
        )
        serial_table = story[serial_table_idx]
        exception_table_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "SN" and _table_text(item)[0][1] == "Run Condition"
        )
        exception_table = story[exception_table_idx]
        serial_break_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakePageBreak)
        )
        self.assertFalse(
            any(
                isinstance(item, _FakeParagraph) and "Plot Table of Contents" in item.text
                for item in story
            )
        )
        self.assertLess(scope_idx, grading_idx)
        self.assertLess(grading_idx, serial_break_idx)
        self.assertLess(serial_break_idx, serial_table_idx)
        self.assertLess(serial_table_idx, exception_table_idx)
        self.assertLess(exec_idx, scope_idx)
        self.assertIsInstance(serial_table, _FakeTable)
        self.assertIsInstance(exception_table, _FakeTable)
        self.assertAlmostEqual(sum(exception_table.colWidths or []), 6.9 * 72.0)
        self.assertAlmostEqual((exception_table.colWidths or [])[0], (serial_table.colWidths or [])[0])
        self.assertEqual(_table_text(exception_table)[0][-1], "Grading")
        self.assertEqual(len(_table_text(exception_table)[0]), 8)
        exception_style_cmds = [
            command
            for style in exception_table.table_styles
            for command in getattr(style, "commands", [])
        ]
        self.assertIn(("BACKGROUND", (7, 1), (7, 1), "#fef3c7"), exception_style_cmds)

        self.assertFalse(
            any(
                isinstance(item, _FakeTable) and _table_text(item) and "Run Condition: Condition A" in _table_text(item)[0][0]
                for item in story
            )
        )

        all_tables = list(_iter_fake_tables(story))
        self.assertFalse(
            any(
                "Run Metrics" in cell
                for table in all_tables
                for row in _table_text(table)
                for cell in row
            )
        )
        self.assertFalse(
            any(
                _table_text(table) and _table_text(table)[0][0] == "Plot / Section"
                for table in all_tables
            )
        )

    def test_build_intro_story_exception_table_colors_and_links_grade_cells(self) -> None:
        plot_navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "watch_nonpass_curves",
                    "pair_id": "pair-1",
                    "param": "Pressure",
                    "page_number": 3,
                }
            ]
        )

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            story = tar._tar_build_intro_story(
                {
                    "print_ctx": tar.PrintContext(
                        printed_at="2026-04-12 09:00 MDT",
                        printed_timezone="MDT",
                        report_title="EIDAT Test Trend Data Analyze Auto Report",
                        report_subtitle="Certification",
                    ),
                    "pair_specs": [{"pair_id": "pair-1", "param": "Pressure", "selection_fields": {"mode": "condition", "display_text": "Condition A"}}],
                    "options": {},
                    "overall_by_sn": {"SN-001": "WATCH"},
                    "nonpass_findings": [],
                    "pair_by_id": {},
                    "hi": ["SN-001"],
                    "params": ["Pressure"],
                    "metric_stats": ["mean"],
                    "include_metrics": False,
                    "meta_note": "",
                    "change_summary": "",
                    "performance_plot_specs": [],
                    "initial_overall_by_sn": {"SN-001": "PASS"},
                    "final_overall_by_sn": {"SN-001": "WATCH"},
                    "comparison_rows": [
                        {
                            "pair_id": "pair-1",
                            "run_condition": "Condition A",
                            "serial": "SN-001",
                            "sequence_text": "Sequence A",
                            "parameter": "Pressure",
                            "units": "psi",
                            "initial_atp_mean": 10.0,
                            "final_atp_mean": 12.0,
                            "initial_actual_mean": 9.0,
                            "final_actual_mean": 11.0,
                            "initial_delta": -1.0,
                            "final_delta": -1.0,
                            "initial_grade": "PASS",
                            "final_grade": "FAIL",
                            "initial_suppression_voltage_label": "All",
                            "final_suppression_voltage_label": "5",
                            "initial_valve_voltage_label": "All",
                            "final_valve_voltage_label": "28",
                            "regrade_applied": False,
                        }
                    ],
                    "meta_by_sn": {"SN-001": {}},
                    "watch_pair_ids": ["pair-1"],
                    "runs": ["Run A"],
                    "all_serials": ["SN-001"],
                    "plot_navigation": plot_navigation,
                    "quick_summary": {
                        "lines": [],
                        "initial_suppression_voltage": "All",
                        "final_suppression_voltage": "5",
                        "initial_valve_voltage": "All",
                        "final_valve_voltage": "28",
                        "p8_suppression_voltage": "5",
                        "p8_valve_voltage": "28",
                    },
                }
            )

        exception_table = next(
            item
            for item in story
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "SN" and _table_text(item)[0][1] == "Run Condition"
        )
        exception_style_cmds = [
            command
            for style in exception_table.table_styles
            for command in getattr(style, "commands", [])
        ]
        self.assertIn(("BACKGROUND", (7, 1), (7, 1), "#fee2e2"), exception_style_cmds)
        self.assertIn(("TEXTCOLOR", (7, 1), (7, 1), tar._TAR_EXEC_EXCEPTION_GRADE_TEXT_COLOR), exception_style_cmds)
        self.assertEqual(_table_text(exception_table)[1][-1], "REQUIRES EVAL")

    def test_build_intro_story_renames_watch_detail_heading(self) -> None:
        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            story = tar._tar_build_intro_story(
                {
                    "print_ctx": tar.PrintContext(
                        printed_at="2026-04-12 09:00 MDT",
                        printed_timezone="MDT",
                        report_title="EIDAT Test Trend Data Analyze Auto Report",
                        report_subtitle="Certification",
                    ),
                    "pair_specs": [],
                    "options": {},
                    "overall_by_sn": {},
                    "nonpass_findings": [],
                    "pair_by_id": {},
                    "hi": [],
                    "params": [],
                    "metric_stats": ["mean"],
                    "include_metrics": False,
                    "comparison_rows": [],
                    "meta_by_sn": {},
                    "runs": [],
                    "all_serials": [],
                    "plot_navigation": [],
                    "quick_summary": {"lines": []},
                }
            )

        headings = [item.text for item in story if isinstance(item, _FakeParagraph)]
        self.assertIn("Watch Items Table", headings)

    def test_pass_fail_synopsis_lines_uses_requires_eval_wording(self) -> None:
        lines = tar._tar_pass_fail_synopsis_lines(
            {
                "comparison_rows": [
                    {"serial": "SN-001", "run_condition": "Condition A", "parameter": "Pressure", "final_grade": "FAIL"},
                    {"serial": "SN-002", "run_condition": "Condition A", "parameter": "Flow", "final_grade": "WATCH"},
                ],
                "overall_by_sn": {"SN-001": "FAILED", "SN-002": "WATCH", "SN-003": "CERTIFIED"},
            }
        )

        self.assertTrue(any("REQUIRES EVAL 1" in line for line in lines))
        self.assertFalse(any("FAILED " in line for line in lines))

    def test_build_plot_toc_story_uses_side_by_side_columns_without_navigator(self) -> None:
        plot_navigation = tar._tar_build_plot_navigation(
            [
                {
                    "section": "run_condition_plot_metrics",
                    "param": f"Pressure {idx}",
                    "x_name": "Time",
                    "stat": "mean",
                    "page_number": idx + 1,
                }
                for idx in range(45)
            ]
        )

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            layout = tar._tar_paginate_plot_navigation(plot_navigation)
            story = tar._tar_build_plot_toc_story(
                {"plot_navigation": plot_navigation, "plot_toc_layout": layout},
                styles=_fake_styles(),
                rl=_fake_reportlab(),
            )

        self.assertEqual(len(layout), 1)
        self.assertEqual(layout[0]["column_count"], 2)
        outer_toc_table = next(
            item
            for item in story
            if isinstance(item, _FakeTable) and any(isinstance(cell, _FakeTable) for row in item._cellvalues for cell in row)
        )
        inner_toc_tables = [cell for cell in outer_toc_table._cellvalues[0] if isinstance(cell, _FakeTable)]
        self.assertEqual(len(inner_toc_tables), 2)
        self.assertEqual(_table_text(inner_toc_tables[0])[1][0], "Run Condition Metrics")
        self.assertEqual(_table_text(inner_toc_tables[1])[1][0], "Run Condition Metrics")
        self.assertFalse(
            any(
                "Run Metrics" in cell
                for table in _iter_fake_tables(story)
                for row in _table_text(table)
                for cell in row
            )
        )

    def test_build_equation_story_adds_overall_and_serial_equation_table(self) -> None:
        ctx = {
            "hi": ["SN-002", "SN-001"],
            "performance_models": [
                {
                    "name": "Run Fit",
                    "x": {"column": "Pressure"},
                    "y": {"column": "Flow"},
                    "stat": "mean",
                    "master": {"equation": "y = 2*x + 1", "rmse": 0.125},
                    "highlighted": {
                        "SN-001": {"equation": "y = 2.1*x + 0.9", "rmse": 0.2},
                        "SN-002": {"equation": "y = 1.9*x + 1.1", "rmse": 0.3},
                    },
                }
            ],
            "equation_cards": [],
        }

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            story = tar._tar_build_equation_story(ctx)

        table = next(
            item
            for item in story
            if isinstance(item, _FakeTable) and _table_text(item)[0] == ["Run Equation", "Serial", "Equation", "RMSE"]
        )
        rows = _table_text(table)
        self.assertEqual(rows[1], ["Run Fit | Flow vs Pressure | Stat: mean", "Overall", "y = 2*x + 1", "0.125"])
        self.assertEqual(rows[2], ["Run Fit | Flow vs Pressure | Stat: mean", "SN-002", "y = 1.9*x + 1.1", "0.3"])
        self.assertEqual(rows[3], ["Run Fit | Flow vs Pressure | Stat: mean", "SN-001", "y = 2.1*x + 0.9", "0.2"])

    def test_build_intro_story_excludes_regraded_run_comparison_tables(self) -> None:
        ctx = {
            "print_ctx": tar.PrintContext(
                printed_at="2026-04-12 09:00 MDT",
                printed_timezone="MDT",
                report_title="EIDAT Test Trend Data Analyze Auto Report",
                report_subtitle="Certification",
            ),
            "pair_specs": [{"pair_id": "pair-1", "param": "Pressure", "selection_fields": {"mode": "condition", "display_text": "Condition A"}}],
            "options": {},
            "overall_by_sn": {"SN-001": "WATCH"},
            "nonpass_findings": [],
            "pair_by_id": {},
            "hi": ["SN-001"],
            "params": ["Pressure"],
            "metric_stats": ["mean"],
            "include_metrics": False,
            "meta_note": "",
            "change_summary": "",
            "performance_plot_specs": [],
            "initial_overall_by_sn": {"SN-001": "PASS"},
            "final_overall_by_sn": {"SN-001": "WATCH"},
            "comparison_rows": [
                {
                    "run_condition": "Condition A",
                    "serial": "SN-001",
                    "sequence_text": "Sequence A",
                    "parameter": "Pressure",
                    "units": "psi",
                    "initial_atp_mean": 10.0,
                    "final_atp_mean": 12.0,
                    "initial_actual_mean": 9.0,
                    "final_actual_mean": 11.0,
                    "initial_delta": -1.0,
                    "final_delta": -1.0,
                    "initial_grade": "PASS",
                    "final_grade": "WATCH",
                    "initial_suppression_voltage_label": "All",
                    "final_suppression_voltage_label": "5",
                    "initial_valve_voltage_label": "All",
                    "final_valve_voltage_label": "28",
                    "regrade_applied": True,
                }
            ],
            "meta_by_sn": {"SN-001": {}},
            "watch_pair_ids": ["pair-1"],
            "runs": ["Run A"],
            "all_serials": ["SN-001"],
            "plot_navigation": [],
            "plot_toc_layout": [],
            "quick_summary": {
                "lines": [],
                "initial_suppression_voltage": "All",
                "final_suppression_voltage": "5",
                "initial_valve_voltage": "All",
                "final_valve_voltage": "28",
                "p8_suppression_voltage": "5",
                "p8_valve_voltage": "28",
            },
        }

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            story = tar._tar_build_intro_story(ctx)

        self.assertFalse(
            any(
                isinstance(item, _FakeTable) and _table_text(item) and "Run Condition: Condition A" in _table_text(item)[0][0]
                for item in story
            )
        )

    def test_reportlab_imports_reports_active_python_when_package_is_missing(self) -> None:
        def _raise_missing(_name: str):
            exc = ModuleNotFoundError("No module named 'reportlab'")
            exc.name = "reportlab"  # type: ignore[attr-defined]
            raise exc

        with mock.patch.object(tar.importlib, "import_module", side_effect=_raise_missing):
            with self.assertRaises(RuntimeError) as ctx:
                tar._reportlab_imports()

        msg = str(ctx.exception)
        self.assertIn("reportlab is required to build formatted portrait report pages.", msg)
        self.assertIn(sys.executable, msg)

    def test_reportlab_imports_accepts_uppercase_tabloid_pagesize(self) -> None:
        fake_modules = {
            "reportlab": SimpleNamespace(__file__="C:/fake/site-packages/reportlab/__init__.py"),
            "reportlab.lib.colors": _FakeColors,
            "reportlab.lib.enums": SimpleNamespace(TA_CENTER="CENTER", TA_LEFT="LEFT"),
            "reportlab.lib.pagesizes": SimpleNamespace(
                landscape=lambda value: ("landscape", value),
                letter=(1, 2),
                TABLOID=(3, 4),
            ),
            "reportlab.lib.styles": SimpleNamespace(ParagraphStyle=object, getSampleStyleSheet=lambda: {}),
            "reportlab.lib.units": SimpleNamespace(inch=1.0),
            "reportlab.platypus": SimpleNamespace(
                KeepTogether=object,
                PageBreak=object,
                Paragraph=object,
                SimpleDocTemplate=object,
                Spacer=object,
                Table=object,
                TableStyle=object,
            ),
        }

        def _fake_import(name: str):
            if name in fake_modules:
                return fake_modules[name]
            raise AssertionError(f"Unexpected module import: {name}")

        with mock.patch.object(tar.importlib, "import_module", side_effect=_fake_import):
            rl = tar._reportlab_imports()

        self.assertEqual((1, 2), rl["letter"])
        self.assertEqual((3, 4), rl["tabloid"])

    def test_reportlab_imports_surfaces_nested_dependency_failure(self) -> None:
        fake_modules = {
            "reportlab": SimpleNamespace(__file__="C:/fake/site-packages/reportlab/__init__.py"),
            "reportlab.lib.colors": _FakeColors,
            "reportlab.lib.enums": SimpleNamespace(TA_CENTER="CENTER", TA_LEFT="LEFT"),
            "reportlab.lib.pagesizes": SimpleNamespace(landscape="landscape", letter=(1, 2), tabloid=(3, 4)),
            "reportlab.lib.styles": SimpleNamespace(ParagraphStyle=object, getSampleStyleSheet=lambda: {}),
            "reportlab.lib.units": SimpleNamespace(inch=1.0),
        }

        def _fake_import(name: str):
            if name == "reportlab.platypus":
                exc = ModuleNotFoundError("No module named 'PIL'")
                exc.name = "PIL"  # type: ignore[attr-defined]
                raise exc
            if name in fake_modules:
                return fake_modules[name]
            raise AssertionError(f"Unexpected module import: {name}")

        with mock.patch.object(tar.importlib, "import_module", side_effect=_fake_import):
            with self.assertRaises(RuntimeError) as ctx:
                tar._reportlab_imports()

        msg = str(ctx.exception)
        self.assertIn("reportlab is installed at 'C:/fake/site-packages/reportlab/__init__.py'", msg)
        self.assertIn("PIL", msg)
        self.assertIn(sys.executable, msg)


if __name__ == "__main__":
    unittest.main()
