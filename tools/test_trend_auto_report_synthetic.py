import json
import math
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import sqlite3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _have_numpy() -> bool:
    try:
        import numpy as _np  # noqa: F401
    except Exception:
        return False
    return True


def _have_matplotlib() -> bool:
    try:
        from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401
    except Exception:
        return False
    return True


class TestTrendAutoReportSynthetic(unittest.TestCase):
    def _make_td_db(self, root: Path) -> Path:
        db = root / "td_test.sqlite3"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                CREATE TABLE td_sources (
                    serial TEXT PRIMARY KEY,
                    sqlite_path TEXT,
                    mtime_ns INTEGER,
                    size_bytes INTEGER,
                    status TEXT,
                    last_ingested_epoch_ns INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_source_metadata (
                    serial TEXT PRIMARY KEY,
                    program_title TEXT,
                    asset_type TEXT,
                    asset_specific_type TEXT,
                    vendor TEXT,
                    acceptance_test_plan_number TEXT,
                    part_number TEXT,
                    revision TEXT,
                    test_date TEXT,
                    report_date TEXT,
                    document_type TEXT,
                    document_type_acronym TEXT,
                    similarity_group TEXT,
                    metadata_rel TEXT,
                    artifacts_rel TEXT,
                    excel_sqlite_rel TEXT,
                    metadata_mtime_ns INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_runs (
                    run_name TEXT PRIMARY KEY,
                    default_x TEXT,
                    display_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_columns (
                    run_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    units TEXT,
                    kind TEXT NOT NULL,
                    PRIMARY KEY (run_name, name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_curves (
                    run_name TEXT NOT NULL,
                    y_name TEXT NOT NULL,
                    x_name TEXT NOT NULL,
                    serial TEXT NOT NULL,
                    x_json TEXT NOT NULL,
                    y_json TEXT NOT NULL,
                    n_points INTEGER NOT NULL,
                    source_mtime_ns INTEGER,
                    computed_epoch_ns INTEGER NOT NULL,
                    PRIMARY KEY (run_name, y_name, x_name, serial)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_metrics (
                    serial TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    value_num REAL,
                    computed_epoch_ns INTEGER NOT NULL,
                    source_mtime_ns INTEGER,
                    PRIMARY KEY (serial, run_name, column_name, stat)
                )
                """
            )
            conn.execute("INSERT INTO td_sources(serial,status) VALUES ('SN1','ok'),('SN2','ok')")
            conn.execute(
                """
                INSERT INTO td_source_metadata(
                    serial, program_title, asset_type, vendor, part_number, revision, document_type, similarity_group, metadata_mtime_ns
                ) VALUES
                    ('SN1','Program A','Thruster','Vendor 1','PN-1','A','TD','SG-1',1),
                    ('SN2','Program A','Thruster','Vendor 2','PN-2','B','TD','SG-1',1)
                """
            )
            conn.execute("INSERT INTO td_runs(run_name,default_x,display_name) VALUES ('Run1','Time','Run 1')")
            conn.execute(
                "INSERT INTO td_columns(run_name,name,units,kind) VALUES ('Run1','thrust','lbf','y')"
            )

            x = [0.0, 1.0, 2.0, 3.0]
            y1 = [0.0, 1.0, 2.0, 3.0]
            y2 = [0.0, 2.0, 4.0, 6.0]
            conn.execute(
                """
                INSERT INTO td_curves(run_name,y_name,x_name,serial,x_json,y_json,n_points,computed_epoch_ns)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                ("Run1", "thrust", "Time", "SN1", json.dumps(x), json.dumps(y1), len(x), 1),
            )
            conn.execute(
                """
                INSERT INTO td_curves(run_name,y_name,x_name,serial,x_json,y_json,n_points,computed_epoch_ns)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                ("Run1", "thrust", "Time", "SN2", json.dumps(x), json.dumps(y2), len(x), 1),
            )

            # Metric min/max for autofill ranges.
            conn.execute(
                """
                INSERT INTO td_metrics(serial,run_name,column_name,stat,value_num,computed_epoch_ns)
                VALUES
                    ('SN1','Run1','thrust','min',0.0,1),
                    ('SN1','Run1','thrust','max',3.0,1),
                    ('SN2','Run1','thrust','min',0.0,1),
                    ('SN2','Run1','thrust','max',6.0,1)
                """
            )
        return db

    def _make_perf_db(self, root: Path) -> Path:
        db = root / "td_perf.sqlite3"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                CREATE TABLE td_sources (
                    serial TEXT PRIMARY KEY,
                    sqlite_path TEXT,
                    mtime_ns INTEGER,
                    size_bytes INTEGER,
                    status TEXT,
                    last_ingested_epoch_ns INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_runs (
                    run_name TEXT PRIMARY KEY,
                    default_x TEXT,
                    display_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_columns (
                    run_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    units TEXT,
                    kind TEXT NOT NULL,
                    PRIMARY KEY (run_name, name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_metrics (
                    serial TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    value_num REAL,
                    computed_epoch_ns INTEGER NOT NULL,
                    source_mtime_ns INTEGER,
                    PRIMARY KEY (serial, run_name, column_name, stat)
                )
                """
            )
            conn.execute("INSERT INTO td_sources(serial,status) VALUES ('SN1','ok'),('SN2','ok')")
            conn.execute(
                "INSERT INTO td_runs(run_name,default_x,display_name) VALUES ('Run1','Time','Run 1'),('Run2','Time','Run 2')"
            )
            for rn in ("Run1", "Run2"):
                conn.execute("INSERT INTO td_columns(run_name,name,units,kind) VALUES (?,?,?,?)", (rn, "isp", "s", "y"))
                conn.execute(
                    "INSERT INTO td_columns(run_name,name,units,kind) VALUES (?,?,?,?)", (rn, "thrust", "lbf", "y")
                )

            # Metrics: 2 runs x 2 serials x 2 cols x 3 stats.
            rows = []
            # Run1
            rows.extend(
                [
                    ("SN1", "Run1", "isp", "mean", 200.0),
                    ("SN1", "Run1", "isp", "min", 195.0),
                    ("SN1", "Run1", "isp", "max", 205.0),
                    ("SN1", "Run1", "thrust", "mean", 10.0),
                    ("SN1", "Run1", "thrust", "min", 9.5),
                    ("SN1", "Run1", "thrust", "max", 10.5),
                    ("SN2", "Run1", "isp", "mean", 210.0),
                    ("SN2", "Run1", "isp", "min", 205.0),
                    ("SN2", "Run1", "isp", "max", 215.0),
                    ("SN2", "Run1", "thrust", "mean", 11.0),
                    ("SN2", "Run1", "thrust", "min", 10.5),
                    ("SN2", "Run1", "thrust", "max", 11.5),
                ]
            )
            # Run2
            rows.extend(
                [
                    ("SN1", "Run2", "isp", "mean", 220.0),
                    ("SN1", "Run2", "isp", "min", 215.0),
                    ("SN1", "Run2", "isp", "max", 225.0),
                    ("SN1", "Run2", "thrust", "mean", 12.0),
                    ("SN1", "Run2", "thrust", "min", 11.5),
                    ("SN1", "Run2", "thrust", "max", 12.5),
                    ("SN2", "Run2", "isp", "mean", 230.0),
                    ("SN2", "Run2", "isp", "min", 225.0),
                    ("SN2", "Run2", "isp", "max", 235.0),
                    ("SN2", "Run2", "thrust", "mean", 13.0),
                    ("SN2", "Run2", "thrust", "min", 12.5),
                    ("SN2", "Run2", "thrust", "max", 13.5),
                ]
            )
            conn.executemany(
                """
                INSERT INTO td_metrics(serial,run_name,column_name,stat,value_num,computed_epoch_ns)
                VALUES (?,?,?,?,?,?)
                """,
                [(sn, rn, col, st, v, 1) for (sn, rn, col, st, v) in rows],
            )
        return db

    def test_autofill_excel_trend_config_from_cache(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_td_db(root)
            cfg = root / "excel_trend_config.json"
            cfg.write_text(
                json.dumps(
                    {
                        "description": "x",
                        "data_group": "Excel Data",
                        "columns": [{"name": "thrust", "units": "", "range_min": None, "range_max": None}],
                        "statistics": ["mean", "min", "max"],
                        "header_row": 0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            updated, summary = tar.autofill_excel_trend_config_from_td_cache(db, cfg, fill_units=True, fill_ranges=True)
            self.assertIn("Backup created:", summary)
            cols = updated.get("columns") or []
            thrust = [c for c in cols if isinstance(c, dict) and str(c.get("name")) == "thrust"][0]
            self.assertEqual(thrust.get("units"), "lbf")
            self.assertEqual(float(thrust.get("range_min")), 0.0)
            self.assertEqual(float(thrust.get("range_max")), 6.0)

    def test_reads_cached_source_metadata(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_td_db(root)
            with sqlite3.connect(str(db)) as conn:
                meta_by_sn, note = tar._read_cached_source_metadata(conn)
            self.assertEqual(note, "")
            self.assertEqual(meta_by_sn["SN1"]["program_title"], "Program A")
            self.assertEqual(meta_by_sn["SN1"]["part_number"], "PN-1")
            self.assertEqual(meta_by_sn["SN2"]["vendor"], "Vendor 2")

    def test_options_params_limits_selected_params(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_perf_db(root)
            with sqlite3.connect(str(db)) as conn:
                run_rows = tar._td_list_runs(conn)
                runs = tar._resolve_selected_runs(run_rows, {"runs": ["Run1"]})
                self.assertEqual(runs, ["Run1"])

                params = tar._resolve_selected_params(conn, runs=runs, options={"params": ["isp"]})
                self.assertEqual(params, ["isp"])

                # If no params are passed, auto-detect returns all y columns.
                auto = tar._resolve_selected_params(conn, runs=runs, options={})
                self.assertIn("isp", [p.lower() for p in auto])
                self.assertIn("thrust", [p.lower() for p in auto])

    def test_run_selections_expand_member_runs(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_perf_db(root)
            with sqlite3.connect(str(db)) as conn:
                run_rows = tar._td_list_runs(conn)
                runs = tar._resolve_selected_runs(
                    run_rows,
                    {
                        "run_selections": [
                            {
                                "mode": "condition",
                                "id": "condition:test",
                                "display_text": "Condition A",
                                "member_runs": ["Run2", "Run1", "Run2"],
                            }
                        ]
                    },
                )
                self.assertEqual(runs, ["Run2", "Run1"])

    def test_overall_certification_status(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        self.assertEqual(tar._overall_cert_status(["PASS", "PASS"]), "CERTIFIED")
        self.assertEqual(tar._overall_cert_status(["PASS", "WATCH"]), "WATCH")
        self.assertEqual(tar._overall_cert_status(["WATCH", "FAIL"]), "FAILED")
        self.assertEqual(tar._overall_cert_status(["NO_DATA", "PASS"]), "NO_DATA")
        self.assertEqual(tar._overall_cert_status([]), "NO_DATA")

    def test_capture_print_context_contains_single_formatted_timestamp(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        ctx = tar._capture_print_context()
        self.assertEqual(ctx.report_title, tar.REPORT_TITLE)
        self.assertEqual(ctx.report_subtitle, tar.REPORT_SUBTITLE_DEFAULT)
        self.assertTrue(ctx.printed_timezone)
        self.assertRegex(ctx.printed_at, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} .+$")

    def test_selection_display_fields_condition_mode_keeps_condition_and_sequences(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        selection = {
            "mode": "condition",
            "run_name": "Run1",
            "run_conditions": ["Condition A"],
            "member_sequences": ["Seq 1", "Seq 2"],
        }
        fields = tar._selection_display_fields(selection, {"Run1": {"display_name": "Run 1"}})
        self.assertEqual(fields["condition_text"], "Condition A")
        self.assertEqual(fields["sequence_text"], "Seq 1, Seq 2")
        title = tar._selection_title_text(selection, {"Run1": {"display_name": "Run 1"}})
        self.assertIn("Run Condition: Condition A", title)
        self.assertIn("Sequences: Seq 1, Seq 2", title)

    def test_selection_display_fields_sequence_mode_backfills_run_condition(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        selection = {
            "mode": "sequence",
            "run_name": "Run1",
            "sequence_name": "Seq 9",
            "run_condition": "Condition B",
        }
        fields = tar._selection_display_fields(selection, {"Run1": {"display_name": "Run 1"}})
        self.assertEqual(fields["sequence_text"], "Seq 9")
        self.assertEqual(fields["condition_text"], "Condition B")
        title = tar._selection_title_text(selection, {"Run1": {"display_name": "Run 1"}})
        self.assertTrue(title.startswith("Sequence: Seq 9"))
        self.assertIn("Run Condition: Condition B", title)

    def test_metric_map_cache_reuses_loaded_series_within_report(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        ctx = {
            "be": object(),
            "db_path": Path("fake.sqlite3"),
            "options": {},
            "filter_state": {},
            "metric_map_cache": {},
        }
        with mock.patch.object(tar, "_load_metric_map_for_selection", return_value={"SN1": 1.23}) as loader:
            first = tar._tar_metric_map_for_run(ctx, "Run1", "thrust", "mean")
            second = tar._tar_metric_map_for_run(ctx, "Run1", "thrust", "mean")
        self.assertEqual(first, {"SN1": 1.23})
        self.assertEqual(second, {"SN1": 1.23})
        self.assertEqual(loader.call_count, 1)

    def test_curve_plot_payload_cache_reuses_loaded_curves_within_report(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        ctx = {
            "be": object(),
            "db_path": Path("fake.sqlite3"),
            "options": {},
            "filter_state": {},
            "grid_points": 5,
            "run_by_name": {"Run1": {"default_x": "Time", "display_name": "Run 1"}},
            "curve_plot_cache": {},
            "pair_by_key": {
                ("Run1", "thrust"): {
                    "run": "Run1",
                    "param": "thrust",
                    "units": "lbf",
                    "selection": {},
                    "model": {"domain": [0.0, 4.0], "x_name": "Time"},
                }
            },
        }
        curves = [
            tar.CurveSeries(serial="SN1", x=[0.0, 2.0, 4.0], y=[0.0, 1.0, 2.0]),
            tar.CurveSeries(serial="SN2", x=[0.0, 2.0, 4.0], y=[0.0, 2.0, 4.0]),
        ]
        with mock.patch.object(tar, "_load_curves_for_selection", return_value=curves) as loader:
            first = tar._tar_curve_plot_payload_for_pair(ctx, "Run1", "thrust")
            second = tar._tar_curve_plot_payload_for_pair(ctx, "Run1", "thrust")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(first["x_name"], "Time")
        self.assertEqual(first["x_grid"], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertIn("SN1", first["y_resampled_by_sn"])

    def test_render_plot_sections_curve_overlay_uses_pair_model(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        class _FakeAxis:
            def set_position(self, *_args, **_kwargs):
                return None

            def axis(self, *_args, **_kwargs):
                return None

            def set_title(self, *_args, **_kwargs):
                return None

            def set_xlabel(self, *_args, **_kwargs):
                return None

            def set_ylabel(self, *_args, **_kwargs):
                return None

            def plot(self, *_args, **_kwargs):
                return None

            def fill_between(self, *_args, **_kwargs):
                return None

            def grid(self, *_args, **_kwargs):
                return None

            def get_legend_handles_labels(self):
                return ([], [])

            def legend(self, *_args, **_kwargs):
                return None

            def text(self, *_args, **_kwargs):
                return None

        class _FakeFigure:
            def add_axes(self, *_args, **_kwargs):
                return _FakeAxis()

        class _FakePdfPages:
            def __init__(self, *_args, **_kwargs):
                self.saved = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def savefig(self, fig):
                self.saved.append(fig)

        fake_matplotlib = types.ModuleType("matplotlib")
        fake_backends = types.ModuleType("matplotlib.backends")
        fake_backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")
        fake_backend_pdf.PdfPages = _FakePdfPages
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.close = lambda _fig: None
        pair_spec = {
            "run": "Run1",
            "run_title": "Run 1",
            "param": "thrust",
            "units": "lbf",
            "selection": {},
            "model": {
                "equation": "y = 1.0x + 0.0",
                "poly": {"rmse": 0.01},
            },
        }
        ctx = {
            "print_ctx": tar._capture_print_context(),
            "include_metrics": False,
            "pair_specs": [pair_spec],
            "performance_plot_specs": [],
            "watch_pair_keys": [],
            "run_by_name": {"Run1": {"display_name": "Run 1"}},
            "hi": ["SN1"],
            "colors": ["#ef4444"],
            "grade_map": {("Run1", "thrust", "SN1"): "PASS"},
            "finding_by_key": {("Run1", "thrust", "SN1"): {"max_pct": 1.0, "z": 0.2}},
        }
        payload = {
            "selection": {},
            "x_name": "Time",
            "x_grid": [0.0, 1.0, 2.0],
            "y_resampled_by_sn": {"SN1": [0.0, 1.0, 2.0], "SN2": [0.0, 0.8, 1.6]},
            "master_y": [0.0, 0.9, 1.8],
            "std_y": [0.0, 0.1, 0.2],
        }
        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.backends": fake_backends,
                "matplotlib.backends.backend_pdf": fake_backend_pdf,
                "matplotlib.pyplot": fake_pyplot,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(_FakeFigure(), _FakeAxis())), mock.patch.object(
            tar,
            "_tar_curve_plot_payload_for_pair",
            return_value=payload,
        ):
            result = tar._tar_render_plot_sections(ctx, intro_pages=2, plots_pdf=Path("fake.pdf"))
        self.assertEqual(result["curve_plot_count"], 1)
        self.assertEqual(result["plot_specs"][0]["section"], "curve_overlays")

    def test_fmt_num_is_defined_and_safe(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        self.assertEqual(tar._fmt_num(None), "—")
        self.assertEqual(tar._fmt_num(float("nan")), "—")
        self.assertEqual(tar._fmt_num(float("inf")), "—")
        self.assertEqual(tar._fmt_num(-0.0), "0")
        self.assertEqual(tar._fmt_num(1.234567, sig=4), "1.235")

    def test_ceil_div_defined_and_correct(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        self.assertTrue(hasattr(tar, "_ceil_div"))
        self.assertTrue(hasattr(tar, "ceil_div"))

        self.assertEqual(tar._ceil_div(0, 30), 0)
        self.assertEqual(tar._ceil_div(1, 30), 1)
        self.assertEqual(tar._ceil_div(30, 30), 1)
        self.assertEqual(tar._ceil_div(31, 30), 2)
        self.assertEqual(tar._ceil_div(61, 30), 3)

        self.assertEqual(tar.ceil_div(31, 30), tar._ceil_div(31, 30))

    def test_build_chart_specs_severity_sort(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        run_param_pairs = [("Run1", "p1"), ("Run1", "p2"), ("Run1", "p3")]
        nonpass_findings = [
            {"run": "Run1", "param": "p1", "grade": "WATCH", "z": 10.0, "max_pct": 1.0},
            {"run": "Run1", "param": "p2", "grade": "FAIL", "z": 2.0, "max_pct": 50.0},
            {"run": "Run1", "param": "p3", "grade": "WATCH", "z": 3.0, "max_pct": 99.0},
        ]
        specs = tar._build_chart_specs(run_param_pairs=run_param_pairs, nonpass_findings=nonpass_findings, max_plots=None)
        ordered = [(run, param) for _key, run, param in specs]
        # FAIL always first; within grade, higher abs(z) first.
        self.assertEqual(ordered[0], ("Run1", "p2"))
        self.assertEqual(ordered[1], ("Run1", "p1"))
        self.assertEqual(ordered[2], ("Run1", "p3"))

    def test_page_planning_enforces_cap_deterministically(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        max_pages = 10
        appendix_include_grade_matrix = True
        appendix_include_pass_details = True
        include_metrics = True
        grading_rows = [{"serial": "SN1"}]  # presence toggles deviations on

        run_param_pairs = [("Run1", "p1"), ("Run1", "p2"), ("Run1", "p3"), ("Run1", "p4")]
        serials_nonpass_sorted = ["SN1", "SN2"]  # 1 by-serial page
        perf_defs_all = [{"x": {"column": "a"}, "y": {"column": "b"}}]  # 1 page
        run_details_all = ["Run1"]  # 1 page

        nonpass_findings = [
            {"run": "Run1", "param": "p1", "grade": "FAIL", "z": 5.0, "max_pct": 10.0},
            {"run": "Run1", "param": "p2", "grade": "WATCH", "z": 4.0, "max_pct": 9.0},
        ]
        chart_specs_all = tar._build_chart_specs(run_param_pairs=run_param_pairs, nonpass_findings=nonpass_findings, max_plots=None)

        metrics_pairs_all = list(run_param_pairs)
        metrics_pairs_nonpass = [("Run1", "p1"), ("Run1", "p2")]

        plan = tar._plan_page_selections(
            max_pages=max_pages,
            appendix_include_grade_matrix=appendix_include_grade_matrix,
            appendix_include_pass_details=appendix_include_pass_details,
            include_metrics=include_metrics,
            grading_rows=grading_rows,
            run_param_pairs=run_param_pairs,
            serials_nonpass_sorted=serials_nonpass_sorted,
            perf_defs_all=perf_defs_all,
            run_details_all=run_details_all,
            chart_specs_all=chart_specs_all,
            metrics_pairs_all=metrics_pairs_all,
            metrics_pairs_nonpass=metrics_pairs_nonpass,
        )

        # With a tight cap, deviations drop first, then metrics drop to non-pass only,
        # and charts are trimmed to leave room for the omitted-items note.
        self.assertFalse(plan["include_deviations"])
        self.assertTrue(all(p in metrics_pairs_nonpass for p in plan["metrics_sel"]))
        self.assertTrue(plan["include_omitted_page"])
        self.assertEqual(plan["charts_sel"], [])  # popped to make room for omitted-items page
        self.assertTrue(any("Appendix: full deviations table omitted" in x for x in plan["omitted_items"]))
        self.assertTrue(any("Metrics: PASS-only" in x for x in plan["omitted_items"]))

    @unittest.skipUnless(_have_matplotlib(), "matplotlib not installed")
    def test_generate_auto_report_fails_clearly_for_empty_cache(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = root / "empty.sqlite3"
            with sqlite3.connect(str(db)) as conn:
                conn.execute("CREATE TABLE td_sources(serial TEXT PRIMARY KEY)")
                conn.execute("CREATE TABLE td_runs(run_name TEXT PRIMARY KEY, default_x TEXT, display_name TEXT)")
                conn.commit()

            with mock.patch.object(
                tar,
                "default_trend_auto_report_config",
                return_value=tar.default_trend_auto_report_config(),
            ), mock.patch(
                "EIDAT_App_Files.ui_next.backend.ensure_test_data_project_cache",
                return_value=db,
            ), mock.patch(
                "EIDAT_App_Files.ui_next.backend.load_excel_trend_config",
                return_value={
                    "description": "x",
                    "data_group": "Excel Data",
                    "columns": [{"name": "thrust", "units": "lbf"}],
                    "statistics": ["mean"],
                    "header_row": 0,
                },
            ), mock.patch(
                "EIDAT_App_Files.ui_next.backend.load_trend_auto_report_config",
                return_value=tar.default_trend_auto_report_config(),
            ):
                with self.assertRaisesRegex(RuntimeError, "no usable Test Data sources"):
                    tar.generate_test_data_auto_report(
                        root,
                        root / "project.xlsx",
                        root / "report.pdf",
                        highlighted_serials=[],
                        options={"update_excel_trend_config": False},
                    )

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_master_curve_and_poly_fit(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_td_db(root)
            with sqlite3.connect(str(db)) as conn:
                series = tar._load_curves(conn, "Run1", "thrust", "Time")
                self.assertEqual(len(series), 2)
                x_grid = [0.0, 1.0, 2.0, 3.0]
                ys = [tar._interp_linear(s.x, s.y, x_grid) for s in series]
                master = tar._nan_median(ys)
                # With y1=x and y2=2x, median with 2 series is the midpoint: 1.5x
                self.assertTrue(math.isclose(master[2], 3.0, rel_tol=1e-9))
                poly = tar._poly_fit(x_grid, master, 3, normalize_x=True)
                self.assertEqual(int(poly.get("degree")), 3)
                self.assertEqual(len(poly.get("coeffs") or []), 4)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_performance_pooled_points_and_poly_fit(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_perf_db(root)
            with sqlite3.connect(str(db)) as conn:
                serials = [r[0] for r in conn.execute("SELECT serial FROM td_sources ORDER BY serial").fetchall()]
                runs = [r[0] for r in conn.execute("SELECT run_name FROM td_runs ORDER BY run_name").fetchall()]

                def metric_map(run: str, col: str, stat: str) -> dict[str, float]:
                    rows = conn.execute(
                        "SELECT serial, value_num FROM td_metrics WHERE run_name=? AND column_name=? AND stat=?",
                        (run, col, stat),
                    ).fetchall()
                    return {str(sn): float(v) for sn, v in rows if sn is not None and v is not None}

                pooled_x: list[float] = []
                pooled_y: list[float] = []
                curves: dict[str, list[tuple[float, float]]] = {}
                for sn in serials:
                    pts = []
                    for rn in runs:
                        xmap = metric_map(rn, "isp", "mean")
                        ymap = metric_map(rn, "thrust", "mean")
                        if sn in xmap and sn in ymap:
                            pts.append((xmap[sn], ymap[sn]))
                    pts.sort(key=lambda t: t[0])
                    curves[sn] = pts
                    pooled_x.extend([p[0] for p in pts])
                    pooled_y.extend([p[1] for p in pts])

                self.assertEqual(len(curves.get("SN1") or []), 2)
                self.assertEqual(len(curves.get("SN2") or []), 2)
                self.assertEqual(len(pooled_x), 4)
                self.assertEqual(len(pooled_y), 4)

                poly = tar._poly_fit(pooled_x, pooled_y, 2, normalize_x=True)
                self.assertEqual(int(poly.get("degree")), 2)
                self.assertEqual(len(poly.get("coeffs") or []), 3)
                eqn = tar._fmt_equation(poly)
                self.assertTrue(bool(eqn))


if __name__ == "__main__":
    unittest.main()
