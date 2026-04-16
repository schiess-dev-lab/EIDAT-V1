from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

from ui_next import trend_auto_report as tar


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
        "inch": 1.0,
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
            },
            "filtered_serials": ["SN-001"],
            "run_selections": [
                {
                    "member_sequences": ["Seq A"],
                    "member_programs": ["Program A"],
                }
            ],
        }

        initial_options = tar._tar_initial_analysis_options(options)
        self.assertEqual(
            initial_options["filter_state"],
            {
                "programs": ["Program A"],
                "serials": ["SN-001", "SN-002"],
                "control_periods": ["10"],
            },
        )
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
            "comparison_rows": [{"final_suppression_voltage_label": "5", "final_valve_voltage_label": "28"}],
            "filter_state": {},
        }

        summary = tar._tar_build_quick_summary(ctx)

        self.assertEqual(summary["certifying_programs"], ["Program A"])
        self.assertEqual(summary["certified_serials"], ["SN-001"])
        self.assertEqual(summary["selected_run_conditions"], ["Condition A", "Condition B"])
        self.assertEqual(summary["watch_parameters"], ["Flow", "Pressure"])
        self.assertEqual(summary["comparison_programs"], ["Program B", "Program C"])
        self.assertEqual(summary["initial_suppression_voltage"], "All")
        self.assertEqual(summary["p8_suppression_voltage"], "5")
        self.assertEqual(summary["p8_valve_voltage"], "28")
        self.assertTrue(any("Programs Compared: Program B, Program C" in line for line in summary["lines"]))
        self.assertIn("Suppression Voltage: 5", summary["lines"])
        self.assertIn("Valve Voltage: 28", summary["lines"])
        self.assertFalse(any(line.startswith("P8 ") for line in summary["lines"]))

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

    def test_build_per_serial_comparison_rows_tracks_initial_and_final_values(self) -> None:
        ctx = {"filter_state": {}}
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
                        "SN-002": [18.0, 22.0],
                    },
                },
                "regrade_plot_payloads": {
                    tar._tar_condition_combo_key("5", "28"): {
                        "master_y": [14.0, 22.0],
                        "y_resampled_by_sn": {
                            "SN-001": [12.0, 16.0],
                            "SN-002": [20.0, 24.0],
                        },
                    }
                },
                "filter_state_override": {"suppression_voltages": ["5"], "valve_voltages": ["28"]},
                "suppression_voltage_label": "5",
                "valve_voltage_label": "28",
            }
        ]
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
                    "regrade_suppression_voltage_label": "5",
                    "regrade_valve_voltage_label": "28",
                    "regrade_condition_key": tar._tar_condition_combo_key("5", "28"),
                }
            },
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_condition"], "Condition A")
        self.assertEqual(rows[0]["initial_atp_mean"], 15.0)
        self.assertEqual(rows[0]["final_atp_mean"], 18.0)
        self.assertEqual(rows[0]["initial_actual_mean"], 10.0)
        self.assertEqual(rows[0]["final_actual_mean"], 14.0)
        self.assertEqual(rows[0]["initial_delta"], -5.0)
        self.assertEqual(rows[0]["final_delta"], -4.0)
        self.assertEqual(rows[0]["initial_grade"], "PASS")
        self.assertEqual(rows[0]["final_grade"], "WATCH")
        self.assertEqual(rows[0]["grade_text"], "Initial: PASS\nFinal: WATCH")
        self.assertEqual(rows[0]["initial_suppression_voltage_label"], "All")
        self.assertEqual(rows[0]["final_suppression_voltage_label"], "5")
        self.assertEqual(rows[0]["initial_valve_voltage_label"], "All")
        self.assertEqual(rows[0]["final_valve_voltage_label"], "28")
        self.assertTrue(rows[0]["regrade_applied"])

    def test_build_per_serial_comparison_rows_falls_back_when_no_regrade_override_exists(self) -> None:
        ctx = {"filter_state": {}}
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
                        "SN-002": [11.0, 13.0],
                    },
                },
            }
        ]
        rows = tar._tar_build_per_serial_comparison_rows(
            ctx,
            pair_specs=pair_specs,
            all_serials=["SN-001", "SN-002"],
            hi=["SN-001"],
            initial_grade_map_by_pair_serial={("pair-2", "SN-001"): "PASS"},
            final_grade_map_by_pair_serial={("pair-2", "SN-001"): "PASS"},
            finding_by_pair_serial={},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["initial_atp_mean"], 10.0)
        self.assertEqual(rows[0]["final_atp_mean"], 10.0)
        self.assertEqual(rows[0]["initial_actual_mean"], 8.0)
        self.assertEqual(rows[0]["final_actual_mean"], 8.0)
        self.assertEqual(rows[0]["initial_delta"], -2.0)
        self.assertEqual(rows[0]["final_delta"], -2.0)
        self.assertEqual(rows[0]["grade_text"], "PASS")
        self.assertEqual(rows[0]["final_suppression_voltage_label"], "All")
        self.assertFalse(rows[0]["regrade_applied"])

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
        self.assertEqual([call[0][2] for call in axes.text_calls], ["Program A", "Program B"])

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
                "pair-1": {"pair_id": "pair-1", "selection_label": "Run A"},
                "pair-2": {"pair_id": "pair-2", "selection_label": "Run B"},
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
        self.assertFalse(create_page_mock.call_args.kwargs["show_plot_toc_backlink"])
        self.assertEqual(len(axes.plot_calls), 0)
        self.assertEqual(len(axes.scatter_calls), 4)
        series_scatter_labels = [str(call[2].get("label") or "") for call in axes.scatter_calls if call[2].get("label")]
        highlight_calls = [call for call in axes.scatter_calls if not call[2].get("label")]
        highlight_colors = [call[2]["color"] for call in highlight_calls]
        highlight_markers = [call[2].get("marker") for call in highlight_calls]
        self.assertIn([0.0, 1.0, 2.0], [call[0] for call in axes.scatter_calls if call[2].get("label")])
        self.assertEqual(series_scatter_labels, ["Run A", "Run B"])
        self.assertEqual(set(highlight_colors), {"#ef4444", "#2563eb"})
        self.assertEqual(highlight_markers, ["x", "x"])
        self.assertEqual(len(axes.axvline_calls), 3)
        self.assertTrue(all(call[1]["color"] == tar._TAR_METRIC_GUIDE_COLOR for call in axes.axvline_calls))
        self.assertEqual(axes.xticklabels, ["SN-001", "SN-002", "SN-003"])
        self.assertEqual(len(axes.patches), 2)
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
        self.assertEqual(create_regrade_page_mock.call_args.kwargs["section_title"], "Regrade Pass Metrics | Condition A")
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
                family_label="Pooled family median",
                band_label="Pooled family +/-1 sigma",
                equation_label="Pooled family equation",
            )

        self.assertEqual(plot_spec["section"], "run_condition_curve_overlays")
        self.assertEqual(plot_spec["page_number"], 9)
        self.assertEqual(create_page_mock.call_args.kwargs["section_title"], "")
        self.assertEqual(create_page_mock.call_args.kwargs["section_subtitle"], "")
        self.assertFalse(create_page_mock.call_args.kwargs["show_plot_toc_backlink"])
        self.assertIsNone(axes.position)
        self.assertEqual(len(pdf.saved_figures), 1)
        self.assertEqual(fake_plt.closed, [fig])
        self.assertGreaterEqual(len(axes.plot_calls), 2)
        self.assertIsNotNone(axes.legend_call)

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
        self.assertEqual(navigation[0]["plot_label"], "Pressure | Time | mean")
        self.assertEqual(navigation[0]["destination_page_index"], 3)
        self.assertEqual(navigation[1]["plot_label"], "ATP Fit | Flow vs Bus Voltage | mean")
        self.assertEqual(navigation[2]["plot_label"], "Run A | Pressure | 2 Serials")

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
                for idx in range(30)
            ]
        )

        pages = tar._tar_paginate_plot_navigation(navigation)

        self.assertGreater(len(pages), 1)
        self.assertTrue(pages[0]["show_navigator"])
        self.assertFalse(pages[1]["show_navigator"])
        self.assertEqual(pages[0]["rows"][0]["text"], "Run Condition Metrics")
        self.assertEqual(pages[1]["rows"][0]["text"], "Run Condition Metrics")

    def test_apply_pdf_navigation_adds_links_and_bookmarks(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "toc_navigation.pdf"
            doc = fitz.open()
            toc_page = doc.new_page()
            toc_page.insert_text((72, 72), "Run Metrics")
            toc_page.insert_text((72, 104), "Run Condition Metrics")
            toc_page.insert_text((72, 136), "Pressure | Time | mean")
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
                plot_toc_layout=[
                    {
                        "toc_page_number": 1,
                        "navigator_sections": [{"label": "Run Metrics", "target_page_index": 1}],
                        "rows": [
                            {"kind": "section", "text": "Run Condition Metrics", "target_page_index": 1},
                            {"kind": "plot", "text": "Pressure | Time | mean", "target_page_index": 1, "page_text": "2"},
                        ],
                    }
                ],
            )

            result = fitz.open(str(pdf_path))
            try:
                toc_rows = result.get_toc()
                self.assertEqual(toc_rows[0][1], "Run Condition Metrics")
                self.assertEqual(toc_rows[0][2], 2)
                self.assertEqual(toc_rows[1][1], "Pressure | Time | mean")
                links = result.load_page(0).get_links()
                self.assertGreaterEqual(len(links), 3)
                self.assertTrue(all(link.get("page") == 1 for link in links))
            finally:
                result.close()

    def test_apply_pdf_navigation_resolves_shifted_toc_pages_from_headings(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "toc_navigation_shifted.pdf"
            doc = fitz.open()
            cover_page = doc.new_page()
            cover_page.insert_text((72, 72), "Cover Page")
            toc_page_1 = doc.new_page()
            toc_page_1.insert_text((72, 72), "Plot Table of Contents")
            toc_page_1.insert_text((72, 104), "Run Condition Metrics")
            toc_page_1.insert_text((72, 136), "Pressure | Time | mean")
            toc_page_2 = doc.new_page()
            toc_page_2.insert_text((72, 72), "Plot Table of Contents (Continued)")
            toc_page_2.insert_text((72, 104), "Regrade Pass Metrics")
            toc_page_2.insert_text((72, 136), "Flow | Time | mean")
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
                        "page_number": 4,
                        "destination_page_index": 3,
                    },
                    {
                        "section_key": "regrade_pass_plot_metrics",
                        "section_label": "Regrade Pass Metrics",
                        "navigator_label": "Regrade Metrics",
                        "plot_label": "Flow | Time | mean",
                        "page_number": 5,
                        "destination_page_index": 4,
                    },
                ],
                plot_toc_layout=[
                    {
                        "toc_page_number": 1,
                        "navigator_sections": [{"label": "Run Metrics", "target_page_index": 3}],
                        "rows": [
                            {"kind": "section", "text": "Run Condition Metrics", "target_page_index": 3},
                            {"kind": "plot", "text": "Pressure | Time | mean", "target_page_index": 3, "page_text": "4"},
                        ],
                    },
                    {
                        "toc_page_number": 2,
                        "navigator_sections": [],
                        "rows": [
                            {"kind": "section", "text": "Regrade Pass Metrics", "target_page_index": 4},
                            {"kind": "plot", "text": "Flow | Time | mean", "target_page_index": 4, "page_text": "5"},
                        ],
                    },
                ],
            )

            result = fitz.open(str(pdf_path))
            try:
                self.assertEqual(result.load_page(0).get_links(), [])
                page_2_links = result.load_page(1).get_links()
                page_3_links = result.load_page(2).get_links()
                self.assertGreaterEqual(len(page_2_links), 2)
                self.assertGreaterEqual(len(page_3_links), 2)
                self.assertTrue(all(link.get("page") == 3 for link in page_2_links))
                self.assertTrue(all(link.get("page") == 4 for link in page_3_links))
            finally:
                result.close()

    def test_apply_pdf_navigation_adds_backlink_to_first_plot_toc_page(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "toc_navigation_backlink.pdf"
            doc = fitz.open()
            toc_page = doc.new_page()
            toc_page.insert_text((72, 72), "Plot Table of Contents")
            toc_page.insert_text((72, 104), "Run Condition Metrics")
            plot_page = doc.new_page()
            plot_page.insert_text((430, 82), tar._TAR_PLOT_TOC_BACKLINK_TEXT)
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
                plot_toc_layout=[
                    {
                        "toc_page_number": 1,
                        "navigator_sections": [],
                        "rows": [
                            {"kind": "section", "text": "Run Condition Metrics", "target_page_index": 1},
                        ],
                    }
                ],
            )

            result = fitz.open(str(pdf_path))
            try:
                links = result.load_page(1).get_links()
                self.assertTrue(any(link.get("page") == 0 for link in links))
            finally:
                result.close()

    def test_apply_plot_page_header_renders_plot_toc_backlink_text(self) -> None:
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
            )

        rendered_text = [str(args[2]) for args, _kwargs in fig.text_calls if len(args) >= 3]
        self.assertIn(tar._TAR_PLOT_TOC_BACKLINK_TEXT, rendered_text)

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
            "plot_toc_layout": tar._tar_paginate_plot_navigation(plot_navigation),
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

        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
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

        page_break_idx = next(idx for idx, item in enumerate(story) if isinstance(item, _FakePageBreak))
        exec_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeParagraph) and item.text == "Executive Summary"
        )
        toc_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeParagraph) and item.text == "Plot Table of Contents"
        )
        counts_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Serials Under Certification"
        )
        exec_table_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Serial" and _table_text(item)[0][1] == "Overall"
        )
        exception_table_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Serial" and _table_text(item)[0][1] == "Run Condition"
        )
        disposition_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Disposition"
        )
        initial_vs_final_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Initial vs Final Run"
        )
        watch_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Watch / Review Highlights"
        )
        self.assertLess(exec_idx, page_break_idx)
        self.assertGreater(toc_idx, page_break_idx)
        self.assertLess(exec_table_idx, disposition_idx)
        self.assertLess(exception_table_idx, disposition_idx)
        self.assertLess(disposition_idx, initial_vs_final_idx)
        self.assertLess(initial_vs_final_idx, watch_idx)
        self.assertLess(watch_idx, counts_idx)
        self.assertLess(counts_idx, toc_idx)

        self.assertFalse(
            any(
                isinstance(item, _FakeTable) and _table_text(item) and "Run Condition: Condition A" in _table_text(item)[0][0]
                for item in story
            )
        )

        toc_table = next(
            item
            for item in story
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Plot / Section"
        )
        toc_rows = _table_text(toc_table)
        self.assertEqual(toc_rows[1][0], "Run Condition Metrics")
        self.assertEqual(toc_rows[2][0], "Pressure | Time | mean")
        self.assertEqual(toc_rows[3][0], "Performance Plots")

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
