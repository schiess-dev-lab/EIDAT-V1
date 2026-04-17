from __future__ import annotations

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
                "pair-1": {
                    "pair_id": "pair-1",
                    "selection_label": "Run A",
                    "run_title": "Run A",
                    "base_condition_label": "Condition A",
                    "selection_fields": {"sequence_text": "Run A", "condition_text": "Condition A"},
                },
                "pair-2": {
                    "pair_id": "pair-2",
                    "selection_label": "Run B",
                    "run_title": "Run B",
                    "base_condition_label": "Condition A",
                    "selection_fields": {"sequence_text": "Run B", "condition_text": "Condition A"},
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
        self.assertEqual(series_scatter_labels, ["Run A | Condition A", "Run B | Condition A"])
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

    def test_apply_pdf_navigation_adds_links_and_bookmarks(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "toc_navigation.pdf"
            doc = fitz.open()
            toc_page = doc.new_page()
            toc_page.insert_text((72, 72), "Run Condition Metrics")
            toc_page.insert_text((72, 104), "Pressure | Time | mean")
            toc_page.insert_text((220, 104), "2")
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
                        "navigator_sections": [],
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
                self.assertGreaterEqual(len(links), 2)
                self.assertTrue(all(link.get("page") == 1 for link in links))
            finally:
                result.close()

    def test_apply_pdf_navigation_links_rows_in_multiple_toc_columns(self) -> None:
        fitz = __import__("fitz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "toc_navigation_multicolumn.pdf"
            doc = fitz.open()
            toc_page = doc.new_page()
            toc_page.insert_text((72, 72), "Plot Table of Contents")
            toc_page.insert_text((72, 104), "Run Condition Metrics")
            toc_page.insert_text((72, 136), "Pressure | Time | mean")
            toc_page.insert_text((220, 136), "2")
            toc_page.insert_text((250, 104), "Performance Plots")
            toc_page.insert_text((250, 136), "ATP Fit | Flow vs Bus Voltage | mean")
            toc_page.insert_text((398, 136), "3")
            toc_page.insert_text((428, 104), "Watch / Non-PASS Curves")
            toc_page.insert_text((428, 136), "Run A | Pressure | 2 Serials")
            toc_page.insert_text((576, 136), "4")
            doc.new_page()
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
                    },
                    {
                        "section_key": "performance_plots",
                        "section_label": "Performance Plots",
                        "navigator_label": "Performance",
                        "plot_label": "ATP Fit | Flow vs Bus Voltage | mean",
                        "page_number": 3,
                        "destination_page_index": 2,
                    },
                    {
                        "section_key": "watch_nonpass_curves",
                        "section_label": "Watch / Non-PASS Curves",
                        "navigator_label": "Watch / Fail",
                        "plot_label": "Run A | Pressure | 2 Serials",
                        "page_number": 4,
                        "destination_page_index": 3,
                    },
                ],
                plot_toc_layout=[
                    {
                        "toc_page_number": 1,
                        "navigator_sections": [],
                        "column_count": 3,
                        "columns": [
                            {
                                "column_index": 1,
                                "rows": [
                                    {"kind": "section", "text": "Run Condition Metrics", "target_page_index": 1},
                                    {"kind": "plot", "text": "Pressure | Time | mean", "target_page_index": 1, "page_text": "2"},
                                ],
                            },
                            {
                                "column_index": 2,
                                "rows": [
                                    {"kind": "section", "text": "Performance Plots", "target_page_index": 2},
                                    {
                                        "kind": "plot",
                                        "text": "ATP Fit | Flow vs Bus Voltage | mean",
                                        "target_page_index": 2,
                                        "page_text": "3",
                                    },
                                ],
                            },
                            {
                                "column_index": 3,
                                "rows": [
                                    {"kind": "section", "text": "Watch / Non-PASS Curves", "target_page_index": 3},
                                    {"kind": "plot", "text": "Run A | Pressure | 2 Serials", "target_page_index": 3, "page_text": "4"},
                                ],
                            },
                        ],
                        "rows": [
                            {"kind": "section", "text": "Run Condition Metrics", "target_page_index": 1},
                            {"kind": "plot", "text": "Pressure | Time | mean", "target_page_index": 1, "page_text": "2"},
                            {"kind": "section", "text": "Performance Plots", "target_page_index": 2},
                            {"kind": "plot", "text": "ATP Fit | Flow vs Bus Voltage | mean", "target_page_index": 2, "page_text": "3"},
                            {"kind": "section", "text": "Watch / Non-PASS Curves", "target_page_index": 3},
                            {"kind": "plot", "text": "Run A | Pressure | 2 Serials", "target_page_index": 3, "page_text": "4"},
                        ],
                    }
                ],
            )

            result = fitz.open(str(pdf_path))
            try:
                links = result.load_page(0).get_links()
                self.assertEqual(sorted(link.get("page") for link in links), [1, 1, 2, 2, 3, 3])
                page_2_links = [link for link in links if link.get("page") == 1]
                page_3_links = [link for link in links if link.get("page") == 2]
                page_4_links = [link for link in links if link.get("page") == 3]
                self.assertTrue(all(link["from"].x1 < 240 for link in page_2_links))
                self.assertTrue(all(link["from"].x0 > 200 for link in page_3_links))
                self.assertTrue(all(link["from"].x0 > 380 for link in page_4_links))
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
            toc_page_1.insert_text((220, 136), "4")
            toc_page_2 = doc.new_page()
            toc_page_2.insert_text((72, 72), "Plot Table of Contents (Continued)")
            toc_page_2.insert_text((72, 104), "Regrade Pass Metrics")
            toc_page_2.insert_text((72, 136), "Flow | Time | mean")
            toc_page_2.insert_text((220, 136), "5")
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
                        "navigator_sections": [],
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
        with mock.patch.object(tar, "_reportlab_imports", return_value=_fake_reportlab()), mock.patch.object(
            tar,
            "_build_portrait_styles",
            return_value=_fake_styles(),
        ):
            plot_toc_layout = tar._tar_paginate_plot_navigation(plot_navigation)
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
                "plot_toc_layout": plot_toc_layout,
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
        program_pool_idx = next(
            idx
            for idx, item in enumerate(story)
            if isinstance(item, _FakeTable) and _table_text(item)[0][0] == "Program Pool Grading"
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
        self.assertLess(disposition_idx, program_pool_idx)
        self.assertLess(program_pool_idx, watch_idx)
        self.assertLess(watch_idx, counts_idx)
        self.assertLess(counts_idx, toc_idx)

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
        toc_tables = [table for table in all_tables if _table_text(table) and _table_text(table)[0][0] == "Plot / Section"]
        toc_rows = [row for table in toc_tables for row in _table_text(table)[1:]]
        self.assertEqual(toc_rows[0][0], "Run Condition Metrics")
        self.assertEqual(toc_rows[1][0], "Pressure | Time | mean")
        self.assertEqual(toc_rows[2][0], "Performance Plots")

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
