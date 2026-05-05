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
        self.assertEqual(tar._overall_cert_status(["NO_DATA", "PASS"], ignore_no_data=True), "CERTIFIED")
        self.assertEqual(tar._overall_cert_status(["NO_DATA", "WATCH"], ignore_no_data=True), "WATCH")
        self.assertEqual(tar._overall_cert_status(["NO_DATA"], ignore_no_data=True), "NO_DATA")
        self.assertEqual(tar._overall_cert_status(["NO_DATA"], ignore_no_data=True, empty_status=""), "")
        self.assertEqual(tar._overall_cert_status([]), "NO_DATA")

    def test_capture_print_context_contains_single_formatted_timestamp(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        ctx = tar._capture_print_context()
        self.assertEqual(ctx.report_title, tar.REPORT_TITLE)
        self.assertEqual(ctx.report_subtitle, tar.REPORT_SUBTITLE_DEFAULT)
        self.assertTrue(ctx.printed_timezone)
        self.assertRegex(ctx.printed_at, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} .+$")

    def test_default_report_subtitle_uses_only_program_and_asset_metadata(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        subtitle = tar._tar_default_report_subtitle(
            serials=["SN-001", "SN-002"],
            meta_by_sn={
                "SN-001": {
                    "asset_type": "Valve",
                    "asset_specific_type": "Main Fuel Valve",
                    "program_title": "Program A",
                },
                "SN-002": {
                    "asset_type": "Valve",
                    "asset_specific_type": "Main Fuel Valve",
                    "program_title": "Program A",
                },
            },
        )

        self.assertEqual(subtitle, "Program A | Valve | Main Fuel Valve")
        self.assertNotIn("SN-001", subtitle)

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

    def test_td_cached_statistics_reads_existing_cache_meta(self):
        from EIDAT_App_Files.ui_next import backend as be

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "cache.sqlite3"
            with sqlite3.connect(str(db)) as conn:
                be._ensure_test_data_impl_tables(conn)
                conn.execute("INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)", ("statistics", "mean,std"))
                conn.commit()
            self.assertEqual(be.td_cached_statistics(db), ["mean", "std"])

    def test_tar_resolve_report_db_path_skips_deep_cache_status_check(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        be = types.SimpleNamespace(
            inspect_test_data_project_cache_state=mock.Mock(side_effect=AssertionError("deep cache status check should be skipped")),
            validate_test_data_project_cache_for_open=mock.Mock(return_value=Path("validated.sqlite3")),
            ensure_test_data_project_cache=mock.Mock(return_value=Path("ensured.sqlite3")),
        )
        progress: list[str] = []
        db_path = tar._tar_resolve_report_db_path(
            be,
            Path("project"),
            Path("workbook.xlsx"),
            rebuild=False,
            progress_cb=progress.append,
        )
        self.assertEqual(db_path, Path("validated.sqlite3"))
        be.inspect_test_data_project_cache_state.assert_not_called()
        be.validate_test_data_project_cache_for_open.assert_called_once()
        be.ensure_test_data_project_cache.assert_not_called()
        self.assertTrue(any("Using existing project cache" in msg for msg in progress))

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

    def test_render_plot_sections_curve_overlay_uses_cohort_model(self):
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

            def scatter(self, *_args, **_kwargs):
                return None

            def axhline(self, *_args, **_kwargs):
                return None

            def set_xticks(self, *_args, **_kwargs):
                return None

            def set_xticklabels(self, *_args, **_kwargs):
                return None

            def set_xlim(self, *_args, **_kwargs):
                return None

            def annotate(self, *_args, **_kwargs):
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
            "pair_id": "pair-1",
            "run": "Run1",
            "run_title": "Run 1",
            "param": "thrust",
            "units": "lbf",
            "selection_label": "Condition A | Supp 100",
        }
        cohort_spec = {
            "cohort_id": "initial:1:thrust:time",
            "param": "thrust",
            "units": "lbf",
            "x_name": "Time",
            "selection_labels": ["Condition A | Supp 100"],
            "member_pair_ids": ["pair-1"],
            "model": {
                "equation": "y = 1.0x + 0.0",
                "poly": {"rmse": 0.01},
            },
            "x_grid": [0.0, 1.0, 2.0],
            "master_y": [0.0, 0.9, 1.8],
            "std_y": [0.0, 0.1, 0.2],
            "trace_curves": [
                {"pair_id": "pair-1", "selection_label": "Condition A | Supp 100", "serial": "SN1", "y_curve": [0.0, 1.0, 2.0]},
                {"pair_id": "pair-1", "selection_label": "Condition A | Supp 100", "serial": "SN2", "y_curve": [0.0, 0.8, 1.6]},
            ],
        }
        ctx = {
            "print_ctx": tar._capture_print_context(),
            "include_metrics": False,
            "initial_cohort_specs": [cohort_spec],
            "regrade_cohort_specs": [],
            "performance_plot_specs": [],
            "watch_pair_ids": [],
            "hi": ["SN1"],
            "colors": ["#ef4444"],
            "pair_by_id": {"pair-1": pair_spec},
            "initial_grade_map_by_pair_serial": {("pair-1", "SN1"): "PASS"},
            "final_grade_map_by_pair_serial": {("pair-1", "SN1"): "PASS"},
            "finding_by_pair_serial": {("pair-1", "SN1"): {"initial_max_pct": 1.0, "initial_z": 0.2}},
        }
        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.backends": fake_backends,
                "matplotlib.backends.backend_pdf": fake_backend_pdf,
                "matplotlib.pyplot": fake_pyplot,
            },
        ), mock.patch.object(tar, "_create_landscape_plot_page", return_value=(_FakeFigure(), _FakeAxis())):
            result = tar._tar_render_plot_sections(ctx, intro_pages=2, plots_pdf=Path("fake.pdf"))
        self.assertEqual(result["curve_plot_count"], 1)
        self.assertEqual(result["plot_specs"][0]["section"], "run_condition_curve_overlays")

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
                "EIDAT_App_Files.ui_next.backend.validate_test_data_project_cache_for_open",
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

    def test_prepare_performance_models_resolves_selector_values_per_run(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        class FakeBackend:
            @staticmethod
            def td_parameter_selection_raw_names(_context, selection_value, *, run_names=None, surface="", raw_names=None):
                run_name = str((run_names or [""])[0] or "").strip()
                mapping = {
                    ("parameter:specific_impulse", "Run1"): ["isp_raw"],
                    ("parameter:specific_impulse", "Run2"): ["specific_impulse"],
                    ("parameter:thrust", "Run1"): ["thrust_raw"],
                    ("parameter:thrust", "Run2"): ["gross_thrust"],
                }
                return mapping.get((str(selection_value or "").strip(), run_name), [str(selection_value or "").strip()])

            @staticmethod
            def td_parameter_value_display_name(_context, selection_value, fallback=""):
                return {
                    "parameter:specific_impulse": "Specific Impulse",
                    "parameter:thrust": "Thrust",
                }.get(str(selection_value or "").strip(), str(fallback or "").strip())

            @staticmethod
            def td_list_metric_y_columns(_db_path, run_name):
                cols_by_run = {
                    "Run1": [
                        {"name": "isp_raw", "units": "s"},
                        {"name": "thrust_raw", "units": "lbf"},
                    ],
                    "Run2": [
                        {"name": "specific_impulse", "units": "s"},
                        {"name": "gross_thrust", "units": "lbf"},
                    ],
                }
                return list(cols_by_run.get(str(run_name or "").strip(), []))

            @staticmethod
            def td_load_metric_series(_db_path, run_name, column_name, stat, **_kwargs):
                series_map = {
                    ("Run1", "isp_raw", "mean"): [
                        {"observation_id": "Run1-SN1", "serial": "SN1", "value_num": 200.0},
                        {"observation_id": "Run1-SN2", "serial": "SN2", "value_num": 210.0},
                    ],
                    ("Run1", "thrust_raw", "mean"): [
                        {"observation_id": "Run1-SN1", "serial": "SN1", "value_num": 10.0},
                        {"observation_id": "Run1-SN2", "serial": "SN2", "value_num": 11.0},
                    ],
                    ("Run2", "specific_impulse", "mean"): [
                        {"observation_id": "Run2-SN1", "serial": "SN1", "value_num": 220.0},
                        {"observation_id": "Run2-SN2", "serial": "SN2", "value_num": 230.0},
                    ],
                    ("Run2", "gross_thrust", "mean"): [
                        {"observation_id": "Run2-SN1", "serial": "SN1", "value_num": 12.0},
                        {"observation_id": "Run2-SN2", "serial": "SN2", "value_num": 13.0},
                    ],
                }
                return list(series_map.get((str(run_name), str(column_name), str(stat).lower()), []))

        ctx = {
            "excel_cfg": {},
            "options": {
                "performance_plotters": [
                    {
                        "name": "Displayed Parameter Fit",
                        "x": {"selection_value": "parameter:specific_impulse"},
                        "y": {"selection_value": "parameter:thrust"},
                        "stats": ["mean"],
                        "require_min_points": 2,
                        "fit": {"degree": 0, "normalize_x": True},
                    }
                ]
            },
            "conn": sqlite3.connect(":memory:"),
            "be": FakeBackend(),
            "db_path": Path("cache.sqlite3"),
            "run_by_name": {
                "Run1": {"display_name": "Run 1"},
                "Run2": {"display_name": "Run 2"},
            },
            "runs": ["Run1", "Run2"],
            "all_serials": ["SN1", "SN2"],
            "hi": ["SN1"],
            "filter_state": {},
            "parameter_context": {"normalization": {"source": "test"}},
        }
        try:
            tar._tar_prepare_performance_models(ctx)
        finally:
            ctx["conn"].close()

        self.assertEqual(len(ctx["performance_models"]), 1)
        model = ctx["performance_models"][0]
        self.assertEqual(model["name"], "Displayed Parameter Fit")
        self.assertEqual(model["x"]["display_name"], "Specific Impulse")
        self.assertEqual(model["y"]["display_name"], "Thrust")
        self.assertEqual(model["x"]["selection_value"], "parameter:specific_impulse")
        self.assertEqual(model["y"]["selection_value"], "parameter:thrust")
        self.assertEqual(model["points_total"], 4)
        self.assertEqual(model["serials_curves"], 2)

        curves = ctx["performance_plot_specs"][0]["curves"]
        self.assertEqual(len(curves["SN1"]), 2)
        self.assertEqual(len(curves["SN2"]), 2)
        self.assertEqual(curves["SN1"][0][:2], (200.0, 10.0))
        self.assertEqual(curves["SN1"][1][:2], (220.0, 12.0))

    def test_prepare_performance_models_accepts_legacy_raw_targets(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        class FakeBackend:
            @staticmethod
            def td_list_metric_y_columns(_db_path, run_name):
                return [
                    {"name": "isp", "units": "s"},
                    {"name": "thrust", "units": "lbf"},
                ]

            @staticmethod
            def td_load_metric_series(_db_path, run_name, column_name, stat, **_kwargs):
                series_map = {
                    ("Run1", "isp", "mean"): [
                        {"observation_id": "Run1-SN1", "serial": "SN1", "value_num": 200.0},
                        {"observation_id": "Run1-SN2", "serial": "SN2", "value_num": 210.0},
                    ],
                    ("Run1", "thrust", "mean"): [
                        {"observation_id": "Run1-SN1", "serial": "SN1", "value_num": 10.0},
                        {"observation_id": "Run1-SN2", "serial": "SN2", "value_num": 11.0},
                    ],
                    ("Run2", "isp", "mean"): [
                        {"observation_id": "Run2-SN1", "serial": "SN1", "value_num": 220.0},
                        {"observation_id": "Run2-SN2", "serial": "SN2", "value_num": 230.0},
                    ],
                    ("Run2", "thrust", "mean"): [
                        {"observation_id": "Run2-SN1", "serial": "SN1", "value_num": 12.0},
                        {"observation_id": "Run2-SN2", "serial": "SN2", "value_num": 13.0},
                    ],
                }
                return list(series_map.get((str(run_name), str(column_name), str(stat).lower()), []))

        ctx = {
            "excel_cfg": {},
            "options": {
                "performance_plotters": [
                    {
                        "name": "Legacy Raw Fit",
                        "x": {"column": "isp"},
                        "y": {"column": "thrust"},
                        "stats": ["mean"],
                        "require_min_points": 2,
                        "fit": {"degree": 0, "normalize_x": True},
                    }
                ]
            },
            "conn": sqlite3.connect(":memory:"),
            "be": FakeBackend(),
            "db_path": Path("cache.sqlite3"),
            "run_by_name": {
                "Run1": {"display_name": "Run 1"},
                "Run2": {"display_name": "Run 2"},
            },
            "runs": ["Run1", "Run2"],
            "all_serials": ["SN1", "SN2"],
            "hi": ["SN1"],
            "filter_state": {},
            "parameter_context": {},
        }
        try:
            tar._tar_prepare_performance_models(ctx)
        finally:
            ctx["conn"].close()

        self.assertEqual(len(ctx["performance_models"]), 1)
        model = ctx["performance_models"][0]
        self.assertEqual(model["name"], "Legacy Raw Fit")
        self.assertEqual(model["x"]["display_name"], "isp")
        self.assertEqual(model["y"]["display_name"], "thrust")
        self.assertEqual(model["points_total"], 4)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_regrades_mixed_suppression_and_final_watch_uses_final_grade(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-100",
                run="Run100",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0])],
            ),
            self._row_spec(
                tar,
                pair_id="pair-200",
                run="Run200",
                selection_label="Condition A | Supp 200",
                base_condition_label="Condition A",
                suppression_value="200",
                series=[
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP3", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                ],
            ),
        ]
        for spec in row_specs:
            spec["base_filter_state"] = {
                "programs": ["Program A"],
                "serials": ["HI", "ATP1", "ATP2", "ATP3"],
                "control_periods": ["10"],
            }
        program_by_serial = {"HI": "Program A", "ATP1": "Program B", "ATP2": "Program B", "ATP3": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=0.5,
            z_watch=1.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        self.assertEqual(len(analysis["initial_cohort_specs"]), 1)
        self.assertEqual(analysis["initial_cohort_specs"][0]["prepass_included_programs"], ["Program A"])
        self.assertEqual(analysis["regrade_cohort_specs"], [])
        row = analysis["grading_rows"][0]
        self.assertEqual(row["pair_id"], "pair-100")
        self.assertEqual(row["initial_grade"], "LIMITED")
        self.assertFalse(row["initial_skipped"])
        self.assertEqual(row["initial_skip_reason"], "")
        self.assertEqual(row["final_grade"], "LIMITED")
        self.assertFalse(row["regrade_applied"])
        self.assertEqual(row["official_pass_type"], "selected_program_pool")
        self.assertEqual(row["grading_basis_status"], "limited_target_excluded_baseline")
        self.assertEqual(row["selected_program_count"], 1)
        self.assertEqual(row["selected_programs"], ["Program A"])
        self.assertEqual(row["selected_pool_series_count"], 1)
        self.assertEqual(row["target_excluded_comparison_series_count"], 0)
        pair_specs = {str(spec.get("pair_id") or ""): spec for spec in analysis["pair_specs"]}
        self.assertEqual(pair_specs["pair-100"]["filter_state_override"], {})
        self.assertEqual(pair_specs["pair-200"]["filter_state_override"], {})
        self.assertEqual(analysis["initial_nonpass_findings"], [])
        self.assertEqual(analysis["nonpass_findings"], [])
        self.assertEqual(analysis["watch_pair_ids"], [])

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_skips_regrade_when_initial_grade_passes_with_mixed_suppression(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-100",
                run="Run100",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0])],
            ),
            self._row_spec(
                tar,
                pair_id="pair-200",
                run="Run200",
                selection_label="Condition A | Supp 200",
                base_condition_label="Condition A",
                suppression_value="200",
                series=[
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                ],
            ),
        ]
        program_by_serial = {"HI": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=2.0,
            z_watch=3.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        row = analysis["grading_rows"][0]
        pair_spec = next(spec for spec in analysis["pair_specs"] if str(spec.get("pair_id") or "") == "pair-100")
        self.assertEqual(row["initial_grade"], "PASS")
        self.assertEqual(row["final_grade"], "PASS")
        self.assertFalse(row["regrade_applied"])
        self.assertEqual(row["official_pass_type"], "selected_program_pool")
        self.assertEqual(row["selected_programs"], ["Program A", "Program B"])
        self.assertEqual(row["selected_pool_series_count"], 3)
        self.assertEqual(row["target_excluded_comparison_series_count"], 2)
        self.assertEqual(row["target_comparison_text"], "HI graded against: 1 program, 2 comparison series")
        self.assertEqual(len(analysis["initial_cohort_specs"]), 1)
        self.assertEqual(analysis["regrade_cohort_specs"], [])
        self.assertEqual(pair_spec["filter_state_override"], {})

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_skips_regrade_for_single_suppression(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-100",
                run="Run100",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[1.0, 1.0, 1.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[1.2, 1.2, 1.2]),
                ],
            )
        ]
        program_by_serial = {"HI": "Program A", "ATP1": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=2.0,
            z_watch=3.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        self.assertEqual(analysis["regrade_cohort_specs"], [])
        row = analysis["grading_rows"][0]
        self.assertFalse(row["initial_skipped"])
        self.assertFalse(row["regrade_applied"])
        self.assertEqual(row["initial_grade"], "LIMITED")
        self.assertEqual(row["final_grade"], "LIMITED")
        self.assertEqual(row["official_pass_type"], "selected_program_pool")
        self.assertEqual(row["grading_basis_status"], "limited_target_excluded_baseline")
        self.assertEqual(row["selected_programs"], ["Program A"])
        self.assertEqual(row["selected_pool_series_count"], 1)
        self.assertEqual(row["target_excluded_comparison_series_count"], 0)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_limits_regrade_cohort_to_targeted_suppression_members(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-target",
                run="RunTarget",
                selection_label="Condition A | Supp 100 | Target",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[100.0, 100.0, 100.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                ],
            ),
            self._row_spec(
                tar,
                pair_id="pair-peer",
                run="RunPeer",
                selection_label="Condition A | Supp 100 | Peer",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="ATP3", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP4", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                ],
            ),
            self._row_spec(
                tar,
                pair_id="pair-200",
                run="Run200",
                selection_label="Condition A | Supp 200",
                base_condition_label="Condition A",
                suppression_value="200",
                series=[
                    tar.CurveSeries(serial="ATP5", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP6", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                ],
            ),
        ]
        program_by_serial = {
            "HI": "Program A",
            "ATP1": "Program B",
            "ATP2": "Program B",
            "ATP3": "Program C",
            "ATP4": "Program C",
            "ATP5": "Program D",
            "ATP6": "Program D",
        }

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=0.5,
            z_watch=1.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        self.assertEqual(analysis["regrade_cohort_specs"], [])
        row = analysis["grading_rows"][0]
        self.assertEqual(row["serial"], "HI")
        self.assertEqual(row["final_grade"], "LIMITED")
        self.assertEqual(row["official_pass_type"], "selected_program_pool")
        self.assertEqual(row["selected_programs"], ["Program A"])
        self.assertEqual(row["selected_pool_series_count"], 1)
        self.assertEqual(row["target_excluded_comparison_series_count"], 0)
        self.assertFalse(row["final_pass_applied"])

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_noise_aware_prepass_admits_noisy_candidate_program(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-noisy",
                run="RunNoise",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[90.0, 90.0, 90.0]),
                    tar.CurveSeries(serial="A2", x=[0.0, 1.0, 2.0], y=[110.0, 110.0, 110.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[96.0, 96.0, 96.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[116.0, 116.0, 116.0]),
                ],
            )
        ]
        program_by_serial = {"HI": "Program A", "A2": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=10.0,
            z_watch=20.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        row = analysis["grading_rows"][0]
        detail = next(item for item in row["prepass_gate_details"] if item["program"] == "Program B")
        self.assertFalse(row["initial_skipped"])
        self.assertEqual(row["prepass_gate_mode"], "noise_normalized_rms_to_certifying_program")
        self.assertEqual(row["prepass_included_programs"], ["Program A", "Program B"])
        self.assertTrue(detail["admitted"])
        self.assertGreater(detail["mean_delta_pct"], 5.0)
        self.assertLess(detail["noise_score"], 1.25)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_prepass_uses_cached_metric_means_for_mean_delta_guard(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        spec = self._row_spec(
            tar,
            pair_id="pair-metric-mean",
            run="RunMetricMean",
            selection_label="Condition A | Supp 100",
            base_condition_label="Condition A",
            suppression_value="100",
            series=[
                tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[80.0, 80.0, 80.0]),
                tar.CurveSeries(serial="A2", x=[0.0, 1.0, 2.0], y=[120.0, 120.0, 120.0]),
                tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[90.0, 90.0, 90.0]),
                tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[130.0, 130.0, 130.0]),
            ],
        )
        spec["metric_mean_by_serial"] = {
            "HI": 99.0,
            "A2": 101.0,
            "ATP1": 103.0,
            "ATP2": 105.0,
        }
        program_by_serial = {"HI": "Program A", "A2": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            [spec],
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=10.0,
            z_watch=20.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        row = analysis["grading_rows"][0]
        detail = next(item for item in row["prepass_gate_details"] if item["program"] == "Program B")
        self.assertEqual(detail["mean_source"], "cached_metric_mean")
        self.assertTrue(detail["admitted"])
        self.assertLess(detail["mean_delta_pct"], 8.0)
        self.assertEqual(row["prepass_included_programs"], ["Program A", "Program B"])
        self.assertEqual(row["selected_programs"], ["Program A", "Program B"])
        self.assertEqual(row["selected_pool_series_count"], 4)
        self.assertEqual(row["target_excluded_comparison_series_count"], 3)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_noise_aware_prepass_rejects_tight_shifted_candidate_program(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-tight",
                run="RunTight",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[100.0, 100.0, 100.0]),
                    tar.CurveSeries(serial="A2", x=[0.0, 1.0, 2.0], y=[100.2, 100.2, 100.2]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[102.1, 102.1, 102.1]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[102.3, 102.3, 102.3]),
                ],
            )
        ]
        program_by_serial = {"HI": "Program A", "A2": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=10.0,
            z_watch=20.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        row = analysis["grading_rows"][0]
        detail = next(item for item in row["prepass_gate_details"] if item["program"] == "Program B")
        self.assertFalse(row["initial_skipped"])
        self.assertEqual(row["initial_skip_reason"], "")
        self.assertEqual(row["initial_grade"], "LIMITED")
        self.assertEqual(row["final_grade"], "LIMITED")
        self.assertEqual(row["grading_basis_status"], "limited_target_excluded_baseline")
        self.assertEqual(row["selected_programs"], ["Program A"])
        self.assertEqual(row["selected_pool_series_count"], 2)
        self.assertEqual(row["target_excluded_comparison_series_count"], 1)
        self.assertFalse(detail["admitted"])
        self.assertLess(detail["mean_delta_pct"], 8.0)
        self.assertGreater(detail["noise_score"], 1.25)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_percent_guard_rejects_noisy_large_shift_candidate_program(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-guard",
                run="RunGuard",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[80.0, 80.0, 80.0]),
                    tar.CurveSeries(serial="A2", x=[0.0, 1.0, 2.0], y=[120.0, 120.0, 120.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[70.0, 70.0, 70.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[170.0, 170.0, 170.0]),
                ],
            )
        ]
        program_by_serial = {"HI": "Program A", "A2": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=10.0,
            z_watch=20.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        row = analysis["grading_rows"][0]
        detail = next(item for item in row["prepass_gate_details"] if item["program"] == "Program B")
        self.assertFalse(row["initial_skipped"])
        self.assertEqual(row["initial_grade"], "LIMITED")
        self.assertEqual(row["final_grade"], "LIMITED")
        self.assertEqual(row["selected_programs"], ["Program A"])
        self.assertEqual(row["selected_pool_series_count"], 2)
        self.assertEqual(row["target_excluded_comparison_series_count"], 1)
        self.assertFalse(detail["admitted"])
        self.assertGreater(detail["mean_delta_pct"], 8.0)
        self.assertLess(detail["noise_score"], 1.25)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_sparse_prepass_uses_stricter_percent_fallback(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-sparse",
                run="RunSparse",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[100.0, 100.0, 100.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[102.0, 102.0, 102.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[108.0, 108.0, 108.0]),
                ],
            )
        ]
        program_by_serial = {"HI": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=10.0,
            z_watch=20.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        row = analysis["grading_rows"][0]
        detail = next(item for item in row["prepass_gate_details"] if item["program"] == "Program B")
        self.assertFalse(row["initial_skipped"])
        self.assertEqual(row["initial_skip_reason"], "")
        self.assertEqual(row["initial_grade"], "LIMITED")
        self.assertEqual(row["final_grade"], "LIMITED")
        self.assertEqual(row["selected_programs"], ["Program A"])
        self.assertEqual(row["selected_pool_series_count"], 1)
        self.assertEqual(row["target_excluded_comparison_series_count"], 0)
        self.assertEqual(detail["gate_mode"], "sparse_percent_fallback")
        self.assertFalse(detail["admitted"])
        self.assertGreater(detail["mean_delta_pct"], 4.0)
        self.assertEqual(len(analysis["initial_cohort_specs"]), 1)
        cohort_spec = analysis["initial_cohort_specs"][0]
        self.assertTrue(cohort_spec["master_y"])
        self.assertEqual(cohort_spec["visual_program_scope"], "all_programs")
        self.assertEqual(
            {str(trace.get("serial") or "") for trace in cohort_spec["trace_curves"]},
            {"HI", "ATP1", "ATP2"},
        )
        self.assertAlmostEqual(float(cohort_spec["master_y"][0]), 102.5)
        pair_spec = analysis["pair_specs"][0]
        self.assertTrue(pair_spec["initial_plot_payload"]["master_y"])
        self.assertIn("HI", pair_spec["initial_plot_payload"]["y_resampled_by_sn"])
        self.assertNotIn("ATP1", pair_spec["initial_plot_payload"]["y_resampled_by_sn"])

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_syncs_final_pass_across_certified_serials_in_same_block(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-sync",
                run="RunSync",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                series=[
                    tar.CurveSeries(serial="HI1", x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
                    tar.CurveSeries(serial="HI2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="A3", x=[0.0, 1.0, 2.0], y=[20.0, 20.0, 20.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                ],
            )
        ]
        program_by_serial = {
            "HI1": "Program A",
            "HI2": "Program A",
            "A3": "Program A",
            "ATP1": "Program B",
            "ATP2": "Program B",
        }

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI1", "HI2"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=1.0,
            z_watch=1.5,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        self.assertEqual(analysis["regrade_cohort_specs"], [])
        rows = {row["serial"]: row for row in analysis["grading_rows"]}
        self.assertFalse(rows["HI1"]["final_pass_applied"])
        self.assertFalse(rows["HI2"]["final_pass_applied"])
        self.assertFalse(rows["HI2"]["program_sync_applied"])
        self.assertEqual(rows["HI1"]["shared_final_condition_key"], "")
        self.assertEqual(rows["HI2"]["shared_final_condition_key"], "")
        self.assertEqual(rows["HI1"]["official_pass_type"], "selected_program_pool")
        self.assertEqual(rows["HI2"]["official_pass_type"], "selected_program_pool")
        self.assertFalse(rows["HI1"]["block_final_required"])
        self.assertFalse(rows["HI2"]["block_final_available"])
        self.assertEqual(rows["HI1"]["final_grade"], "FAIL")
        self.assertEqual(rows["HI2"]["final_grade"], "PASS")
        self.assertAlmostEqual(float(rows["HI1"]["official_baseline_mean"]), 12.5)
        self.assertAlmostEqual(float(rows["HI1"]["official_serial_mean"]), 0.0)
        self.assertAlmostEqual(float(rows["HI2"]["official_baseline_mean"]), 10.0)
        self.assertAlmostEqual(float(rows["HI2"]["official_serial_mean"]), 10.0)
        self.assertEqual(rows["HI1"]["target_excluded_comparison_series_count"], 4)
        self.assertEqual(rows["HI2"]["target_excluded_comparison_series_count"], 4)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_uses_per_serial_final_when_no_shared_exact_condition_exists(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        key_100 = tar._tar_condition_combo_key("100", "")
        key_200 = tar._tar_condition_combo_key("200", "")
        row_specs = [
            {
                **self._row_spec(
                    tar,
                    pair_id="pair-sync-fallback",
                    run="RunSyncFallback",
                    selection_label="Condition A",
                    base_condition_label="Condition A",
                    suppression_value="",
                    series=[
                        tar.CurveSeries(serial="HI1", x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
                        tar.CurveSeries(serial="HI2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                        tar.CurveSeries(serial="A3", x=[0.0, 1.0, 2.0], y=[20.0, 20.0, 20.0]),
                        tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                        tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    ],
                ),
                "condition_pairs": [
                    {"key": key_100, "suppression_voltage_label": "100", "valve_voltage_label": ""},
                    {"key": key_200, "suppression_voltage_label": "200", "valve_voltage_label": ""},
                ],
                "series_by_condition_key": {
                    key_100: [
                        tar.CurveSeries(serial="HI1", x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
                        tar.CurveSeries(serial="A3", x=[0.0, 1.0, 2.0], y=[20.0, 20.0, 20.0]),
                        tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                        tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    ],
                    key_200: [
                        tar.CurveSeries(serial="HI2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                        tar.CurveSeries(serial="A3", x=[0.0, 1.0, 2.0], y=[20.0, 20.0, 20.0]),
                        tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                        tar.CurveSeries(serial="ATP2", x=[0.0, 1.0, 2.0], y=[10.0, 10.0, 10.0]),
                    ],
                },
            }
        ]
        program_by_serial = {
            "HI1": "Program A",
            "HI2": "Program A",
            "A3": "Program A",
            "ATP1": "Program B",
            "ATP2": "Program B",
        }

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI1", "HI2"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=1.0,
            z_watch=1.5,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        self.assertEqual(analysis["regrade_cohort_specs"], [])
        rows = {row["serial"]: row for row in analysis["grading_rows"]}
        self.assertFalse(rows["HI1"]["final_pass_applied"])
        self.assertFalse(rows["HI2"]["final_pass_applied"])
        self.assertFalse(rows["HI1"]["block_final_available"])
        self.assertFalse(rows["HI2"]["block_final_available"])
        self.assertEqual(rows["HI1"]["official_pass_type"], "selected_program_pool")
        self.assertEqual(rows["HI2"]["official_pass_type"], "selected_program_pool")
        self.assertEqual(rows["HI1"]["final_selection_mode"], "")
        self.assertEqual(rows["HI2"]["final_selection_mode"], "")
        self.assertEqual(rows["HI1"]["shared_final_condition_key"], "")
        self.assertEqual(rows["HI2"]["shared_final_condition_key"], "")
        self.assertEqual(rows["HI1"]["final_condition_key"], "")
        self.assertEqual(rows["HI2"]["final_condition_key"], "")

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_analyze_curve_groups_splits_incompatible_x_axes(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        row_specs = [
            self._row_spec(
                tar,
                pair_id="pair-time",
                run="RunTime",
                selection_label="Condition A | Supp 100",
                base_condition_label="Condition A",
                suppression_value="100",
                x_name="Time",
                series=[
                    tar.CurveSeries(serial="HI", x=[0.0, 1.0, 2.0], y=[1.0, 1.0, 1.0]),
                    tar.CurveSeries(serial="ATP1", x=[0.0, 1.0, 2.0], y=[1.02, 1.02, 1.02]),
                ],
            ),
            self._row_spec(
                tar,
                pair_id="pair-pulse",
                run="RunPulse",
                selection_label="Condition B | Supp 200",
                base_condition_label="Condition B",
                suppression_value="200",
                x_name="Pulse Number",
                series=[
                    tar.CurveSeries(serial="HI", x=[1.0, 2.0, 3.0], y=[2.0, 2.0, 2.0]),
                    tar.CurveSeries(serial="ATP2", x=[1.0, 2.0, 3.0], y=[2.04, 2.04, 2.04]),
                ],
            ),
        ]
        program_by_serial = {"HI": "Program A", "ATP1": "Program B", "ATP2": "Program B"}

        analysis = tar._tar_analyze_curve_groups(
            row_specs,
            hi=["HI"],
            program_by_serial=program_by_serial,
            certifying_program="Program A",
            grid_points=3,
            degree=1,
            normalize_x=False,
            z_pass=2.0,
            z_watch=3.0,
            max_abs_thr=None,
            max_pct_thr=None,
            rms_pct_thr=None,
        )

        self.assertEqual(len(analysis["initial_cohort_specs"]), 2)
        self.assertEqual(sorted(spec["x_name"] for spec in analysis["initial_cohort_specs"]), ["Pulse Number", "Time"])

    def test_build_per_serial_comparison_rows_uses_plot_payload_metrics(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        pair_spec = {
            "pair_id": "pair-1",
            "selection_id": "selection-1",
            "run": "Run1",
            "run_title": "Run 1",
            "param": "thrust",
            "units": "lbf",
            "selection_fields": {
                "mode": "condition",
                "condition_text": "Condition A",
                "sequence_text": "Seq 1",
            },
            "base_condition_label": "Condition A",
            "suppression_voltage_label": "100",
            "prepass_reference_program": "Program A",
            "prepass_included_programs": ["Program A", "Program B"],
            "prepass_excluded_programs": [],
            "initial_plot_payload": {
                "master_y": [1.0, 1.0, 1.0],
                "y_resampled_by_sn": {
                    "SN1": [1.5, 1.5, 1.5],
                    "SN2": [2.5, 2.5, 2.5],
                },
            },
            "regrade_plot_payloads": {
                tar._tar_condition_combo_key("100", ""): {
                    "master_y": [4.0, 4.0, 4.0],
                    "y_resampled_by_sn": {
                        "SN1": [5.0, 5.0, 5.0],
                        "SN2": [7.0, 7.0, 7.0],
                    },
                }
            },
            "filter_state_override": {"suppression_voltages": ["100"]},
        }

        def _metric_map_side_effect(_ctx, pair_spec_arg, stat, *, filter_state_override=None):
            self.assertEqual(stat, "mean")
            self.assertEqual(str(pair_spec_arg.get("pair_id") or ""), "pair-1")
            suppression = tuple((filter_state_override or {}).get("suppression_voltages") or [])
            if suppression == ("100",):
                return {"SN1": 5.0, "SN2": 7.0}
            return {"SN1": 1.5, "SN2": 2.5}

        with mock.patch.object(tar, "_tar_metric_map_for_pair", side_effect=_metric_map_side_effect):
            rows = tar._tar_build_per_serial_comparison_rows(
                {
                    "filter_state": {},
                    "be": object(),
                    "db_path": Path("fake.sqlite3"),
                    "options": {},
                    "program_by_serial": {"SN1": "Program A", "SN2": "Program B"},
                },
                pair_specs=[pair_spec],
                all_serials=["SN1"],
                hi=["SN1"],
                initial_grade_map_by_pair_serial={("pair-1", "SN1"): "FAIL"},
                final_grade_map_by_pair_serial={("pair-1", "SN1"): "PASS"},
                finding_by_pair_serial={
                    ("pair-1", "SN1"): {
                        "regrade_applied": True,
                        "final_pass_applied": True,
                        "regrade_suppression_voltage_label": "100",
                        "regrade_condition_key": tar._tar_condition_combo_key("100", ""),
                        "regrade_cohort_id": "regrade:1",
                        "initial_z": -0.5,
                        "final_z": 1.25,
                        "prepass_reference_program": "Program A",
                        "prepass_included_programs": ["Program A", "Program B"],
                        "prepass_excluded_programs": [],
                    }
                },
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["initial_family_mean"], 2.0)
        self.assertEqual(row["initial_serial_mean"], 1.5)
        self.assertEqual(row["initial_zscore"], -0.5)
        self.assertEqual(row["final_family_mean"], 6.0)
        self.assertEqual(row["final_serial_mean"], 5.0)
        self.assertEqual(row["final_zscore"], 1.25)
        self.assertTrue(row["regrade_applied"])
        self.assertTrue(row["final_pass_applied"])
        self.assertEqual(row["regrade_cohort_id"], "regrade:1")
        self.assertEqual(row["prepass_reference_program"], "Program A")
        self.assertEqual(row["prepass_included_programs"], ["Program A", "Program B"])
        self.assertEqual(row["prepass_excluded_programs"], [])

    def test_report_parameter_display_mapping_preserves_raw_analysis_name(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        class FakeBackend:
            @staticmethod
            def td_list_raw_y_columns(_db_path, _run_name):
                return [{"name": "feed_pressure_raw", "units": "psi"}]

            @staticmethod
            def td_list_curve_y_columns(_db_path, _run_name):
                return []

            @staticmethod
            def td_build_parameter_selector_options(_context, *, run_names=None, surface="", raw_names=None):
                return [
                    {
                        "value": "parameter:chamber_feed_pressure",
                        "label": "Chamber Feed Pressure (psia)",
                        "display_name": "Chamber Feed Pressure",
                        "preferred_units": "psia",
                        "raw_names": ["feed_pressure_raw"],
                    }
                ]

            @staticmethod
            def td_parameter_selection_raw_names(_context, selection_value, *, run_names=None, surface="", raw_names=None):
                if selection_value == "parameter:chamber_feed_pressure":
                    return ["feed_pressure_raw"]
                return [selection_value]

            @staticmethod
            def td_parameter_value_display_name(_context, selection_value, fallback=""):
                if selection_value == "parameter:chamber_feed_pressure":
                    return "Chamber Feed Pressure"
                return fallback

        with sqlite3.connect(":memory:") as conn:
            params, display_by_raw = tar._tar_resolve_params_for_report(
                FakeBackend(),
                Path("cache.sqlite3"),
                conn,
                runs=["Run1"],
                options={"params": ["parameter:chamber_feed_pressure"]},
                parameter_context={"normalization": {"source": "test"}},
            )

        raw_key = tar._norm_key("feed_pressure_raw")
        self.assertEqual(params, ["feed_pressure_raw"])
        self.assertEqual(display_by_raw[raw_key]["display_name"], "Chamber Feed Pressure")
        self.assertEqual(display_by_raw[raw_key]["display_units"], "psia")
        self.assertEqual(
            tar._tar_param_display_name(display_by_raw, "feed_pressure_raw"),
            "Chamber Feed Pressure",
        )
        self.assertEqual(tar._tar_param_display_units(display_by_raw, "feed_pressure_raw", "psi"), "psia")

    def test_report_scope_uses_display_parameters_when_available(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        rows = tar._tar_exec_scope_table_rows(
            {
                "hi": ["SN1"],
                "meta_by_sn": {},
                "params": ["feed_pressure_raw"],
                "display_params": ["Chamber Feed Pressure"],
                "execution_summary": {
                    "selected_run_conditions": ["Condition A"],
                    "comparison_programs": ["Program A"],
                },
                "overall_by_sn": {"SN1": "CERTIFIED"},
                "comparison_rows": [],
            },
            quick_summary={
                "selected_run_conditions": ["Condition A"],
                "comparison_programs": ["Program A"],
            },
        )

        by_name = {str(row[0]): str(row[1]) for row in rows[1:]}
        self.assertEqual(by_name["Parameters analyzed"], "Chamber Feed Pressure")

    def test_comparison_rows_render_display_parameter_and_keep_raw_parameter(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        pair_spec = {
            "pair_id": "pair-display",
            "selection_id": "selection-display",
            "run": "Run1",
            "run_title": "Run 1",
            "param": "feed_pressure_raw",
            "param_display": "Chamber Feed Pressure",
            "units": "psi",
            "display_units": "psia",
            "selection_fields": {
                "mode": "condition",
                "condition_text": "Condition A",
                "sequence_text": "Seq 1",
            },
            "base_condition_label": "Condition A",
            "initial_plot_payload": {
                "master_y": [1.0, 1.0],
                "y_resampled_by_sn": {"SN1": [1.5, 1.5], "SN2": [2.5, 2.5]},
            },
        }

        with mock.patch.object(tar, "_tar_metric_map_for_pair", return_value={"SN1": 1.5, "SN2": 2.5}):
            rows = tar._tar_build_per_serial_comparison_rows(
                {
                    "filter_state": {},
                    "be": object(),
                    "db_path": Path("fake.sqlite3"),
                    "options": {},
                    "program_by_serial": {"SN1": "Program A", "SN2": "Program B"},
                },
                pair_specs=[pair_spec],
                all_serials=["SN1", "SN2"],
                hi=["SN1"],
                initial_grade_map_by_pair_serial={("pair-display", "SN1"): "PASS"},
                final_grade_map_by_pair_serial={("pair-display", "SN1"): "PASS"},
                finding_by_pair_serial={
                    ("pair-display", "SN1"): {
                        "initial_z": -0.25,
                        "final_z": -0.25,
                        "official_pass_type": "initial_prepass",
                        "initial_status": "PASS",
                        "official_grade": "PASS",
                    }
                },
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["parameter"], "Chamber Feed Pressure")
        self.assertEqual(rows[0]["param"], "feed_pressure_raw")
        self.assertEqual(rows[0]["raw_parameter"], "feed_pressure_raw")
        self.assertEqual(rows[0]["units"], "psia")
        self.assertEqual(rows[0]["raw_units"], "psi")

    def test_build_intro_story_renders_outcome_mix_and_exception_rows(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        fake_rl = {
            "Spacer": lambda *args, **kwargs: ("Spacer", args),
            "PageBreak": lambda *args, **kwargs: ("PageBreak",),
            "inch": 1.0,
            "colors": types.SimpleNamespace(HexColor=lambda value: value, white="#ffffff"),
        }
        fake_styles = {name: name for name in ("cover_title", "cover_subtitle", "body", "section", "small", "card_title")}
        ctx = {
            "print_ctx": tar._capture_print_context(),
            "pair_specs": [{"selection_fields": {"mode": "condition", "display_text": "Condition A | Supp 100"}}],
            "options": {"run_selection_labels": ["Condition A | Supp 100"]},
            "overall_by_sn": {"SN1": "WATCH"},
            "initial_overall_by_sn": {"SN1": "FAILED"},
            "final_overall_by_sn": {"SN1": "WATCH"},
            "nonpass_findings": [],
            "pair_by_id": {},
            "hi": ["SN1"],
            "runs": ["Run1"],
            "params": ["thrust"],
            "watch_pair_ids": [],
            "metric_stats": ["mean"],
            "include_metrics": True,
            "filter_state": {},
            "meta_by_sn": {"SN1": {}},
            "meta_note": "",
            "change_summary": "",
            "performance_plot_specs": [],
            "plot_navigation": [
                {
                    "section_key": "regrade_pass_curve_overlays",
                    "cohort_id": "regrade:1",
                    "destination_page_index": 6,
                    "page_number": 7,
                }
            ],
            "comparison_rows": [
                {
                    "run_condition": "Condition A | Supp 100",
                    "serial": "SN1",
                    "sequence_text": "Seq 1",
                    "parameter": "thrust",
                    "units": "lbf",
                    "initial_atp_mean": 1.0,
                    "final_atp_mean": 1.2,
                    "initial_actual_mean": 1.1,
                    "final_actual_mean": 1.0,
                    "initial_delta": 0.1,
                    "final_delta": -0.2,
                    "initial_grade": "FAIL",
                    "final_grade": "PASS",
                    "initial_suppression_voltage_label": "All",
                    "final_suppression_voltage_label": "100",
                    "initial_bus_voltage_label": "",
                    "final_bus_voltage_label": "",
                    "regrade_applied": True,
                    "regrade_cohort_id": "regrade:1",
                },
                {
                    "run_condition": "Condition A | Supp 100",
                    "serial": "SN1",
                    "sequence_text": "Seq 2",
                    "parameter": "torque",
                    "units": "Nm",
                    "initial_atp_mean": 2.0,
                    "final_atp_mean": 2.1,
                    "initial_actual_mean": 2.4,
                    "final_actual_mean": 2.3,
                    "initial_delta": 0.4,
                    "final_delta": 0.2,
                    "initial_grade": "FAIL",
                    "final_grade": "WATCH",
                    "initial_suppression_voltage_label": "All",
                    "final_suppression_voltage_label": "100",
                    "initial_bus_voltage_label": "",
                    "final_bus_voltage_label": "",
                    "regrade_applied": True,
                    "regrade_cohort_id": "regrade:1",
                },
                {
                    "run_condition": "Condition A | Supp 100",
                    "serial": "SN1",
                    "sequence_text": "Seq 3",
                    "parameter": "flow",
                    "units": "kg/s",
                    "initial_atp_mean": None,
                    "final_atp_mean": None,
                    "initial_actual_mean": None,
                    "final_actual_mean": None,
                    "initial_delta": None,
                    "final_delta": None,
                    "initial_grade": "NO_DATA",
                    "final_grade": "NO_DATA",
                    "initial_suppression_voltage_label": "All",
                    "final_suppression_voltage_label": "All",
                    "initial_bus_voltage_label": "",
                    "final_bus_voltage_label": "",
                    "regrade_applied": False,
                    "regrade_cohort_id": "",
                },
            ],
        }

        with mock.patch.object(tar, "_reportlab_imports", return_value=fake_rl), mock.patch.object(
            tar, "_build_portrait_styles", return_value=fake_styles
        ), mock.patch.object(tar, "_portrait_paragraph", side_effect=lambda text, style, _rl: ("Paragraph", text)), mock.patch.object(
            tar, "_portrait_card", side_effect=lambda title, lines, **_kwargs: ("Card", title, list(lines))
        ), mock.patch.object(
            tar, "_portrait_box_table", side_effect=lambda rows, **_kwargs: ("Table", rows)
        ):
            story = tar._tar_build_intro_story(ctx)

        tables = [item[1] for item in story if isinstance(item, tuple) and item and item[0] == "Table"]
        scope_table = next(rows for rows in tables if rows and rows[0][0] == "Scope Item")
        grading_table = next(rows for rows in tables if rows and rows[0][0] == "Grade Item")
        serial_table = next(rows for rows in tables if rows and rows[0][0] == "SN" and rows[0][1] == "Initial / Final")
        exception_table = next(rows for rows in tables if rows and rows[0][0] == "SN" and rows[0][1] == "Run Condition")
        self.assertEqual(scope_table[1][0], "SNs analyzed")
        self.assertEqual(grading_table[1][0], "Official graded mean")
        self.assertEqual(serial_table[1][1], "Initial: FAILED\nFinal: WATCH")
        self.assertEqual(serial_table[0], ["SN", "Initial / Final", "Program", "Part #", "Rev", "P/W/F items"])
        self.assertEqual(serial_table[1][5], "PASS 1/2 (50%)\nWATCH 1/2 (50%)\nFAIL 0/2 (0%)")
        self.assertEqual(exception_table[1][7], "WATCH")
        self.assertTrue(str(exception_table[1][8]).startswith("Chart "))
        self.assertIn("Suppression Voltage: 100", str(exception_table[1][1]))
        self.assertIn("Seq 2", str(exception_table[1][2]))
        self.assertFalse(any(rows and str(rows[0][0]).startswith("Run Condition:") for rows in tables))

    def test_plan_comparison_pages_pivots_by_serial_and_splits_by_run_blocks(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        comparison_rows: list[dict[str, object]] = []
        for block_index in range(18):
            selection_id = f"sel-{block_index:02d}"
            for param_name, units, initial_atp, final_atp, initial_actual, final_actual, initial_delta, final_delta, initial_grade, final_grade, regrade_applied in (
                ("Pressure", "psi", 10.0 + block_index, 12.0 + block_index, 9.0 + block_index, 11.0 + block_index, -1.0, -1.0, "PASS", "WATCH", block_index == 0),
                ("Flow", "kg/s", 20.0 + block_index, 20.0 + block_index, 19.0 + block_index, 19.0 + block_index, -1.0, -1.0, "PASS", "PASS", False),
            ):
                official_pass_type = "final_exact_condition" if regrade_applied else "initial_prepass"
                official_baseline = final_atp if regrade_applied else initial_atp
                official_actual = final_actual if regrade_applied else initial_actual
                official_grade = final_grade if regrade_applied else initial_grade
                comparison_rows.append(
                    {
                        "selection_id": selection_id,
                        "run_condition": f"Condition {block_index:02d}",
                        "sequence_text": f"Seq {block_index:02d}",
                        "serial": "SN1",
                        "parameter": param_name,
                        "units": units,
                        "initial_atp_mean": initial_atp,
                        "final_atp_mean": final_atp,
                        "initial_actual_mean": initial_actual,
                        "final_actual_mean": final_actual,
                        "initial_delta": initial_delta,
                        "final_delta": final_delta,
                        "initial_grade": initial_grade,
                        "final_grade": final_grade,
                        "initial_status": initial_grade,
                        "regrade_applied": regrade_applied,
                        "official_pass_type": official_pass_type,
                        "official_baseline_mean": official_baseline,
                        "official_serial_mean": official_actual,
                        "official_zscore": final_delta if regrade_applied else initial_delta,
                        "official_grade": official_grade,
                        "grade_basis_text": (
                            "Program-synced exact-condition final\nSupp: 5 | Valve: 28"
                            if regrade_applied
                            else "Initial admitted-program cohort"
                        ),
                    }
                )
        comparison_rows.extend(
            [
                {
                    "selection_id": "sel-sn2",
                    "run_condition": "Condition Z",
                    "sequence_text": "Seq Z",
                    "serial": "SN2",
                    "parameter": "Pressure",
                    "units": "psi",
                    "initial_atp_mean": 33.0,
                    "final_atp_mean": 33.0,
                    "initial_actual_mean": 31.0,
                    "final_actual_mean": 31.0,
                    "initial_delta": -2.0,
                    "final_delta": -2.0,
                    "initial_grade": "WATCH",
                    "final_grade": "WATCH",
                    "initial_status": "WATCH",
                    "regrade_applied": False,
                    "official_pass_type": "initial_prepass",
                    "official_baseline_mean": 33.0,
                    "official_serial_mean": 31.0,
                    "official_zscore": -2.0,
                    "official_grade": "WATCH",
                    "grade_basis_text": "Initial admitted-program cohort",
                },
                {
                    "selection_id": "sel-sn2",
                    "run_condition": "Condition Z",
                    "sequence_text": "Seq Z",
                    "serial": "SN2",
                    "parameter": "Flow",
                    "units": "kg/s",
                    "initial_atp_mean": 44.0,
                    "final_atp_mean": 44.0,
                    "initial_actual_mean": 42.0,
                    "final_actual_mean": 42.0,
                    "initial_delta": -2.0,
                    "final_delta": -2.0,
                    "initial_grade": "PASS",
                    "final_grade": "PASS",
                    "initial_status": "PASS",
                    "regrade_applied": False,
                    "official_pass_type": "initial_prepass",
                    "official_baseline_mean": 44.0,
                    "official_serial_mean": 42.0,
                    "official_zscore": -2.0,
                    "official_grade": "PASS",
                    "grade_basis_text": "Initial admitted-program cohort",
                },
            ]
        )

        with mock.patch.object(tar, "_reportlab_imports", return_value={}), mock.patch.object(
            tar,
            "_tar_measure_comparison_table_height",
            side_effect=lambda page_spec, rl=None: 140.0 + 80.0 * len(page_spec.get("blocks") or []),
        ):
            page_specs = tar._tar_plan_comparison_pages({"hi": ["SN1", "SN2"], "comparison_rows": comparison_rows})

        self.assertGreaterEqual(len([page for page in page_specs if page["serial"] == "SN1"]), 2)
        sn1_page_count = len([page for page in page_specs if page["serial"] == "SN1"])
        self.assertTrue(all(page["serial"] == "SN1" for page in page_specs[:sn1_page_count]))
        self.assertEqual(page_specs[sn1_page_count]["serial"], "SN2")

        first_page = page_specs[0]
        matrix_rows, style_cmds = tar._tar_build_comparison_page_matrix(first_page)
        self.assertEqual(matrix_rows[0][:5], ["Run Condition", "Sequence(s)", "Metric", "Pressure", "Flow"])
        self.assertEqual(matrix_rows[1][:5], ["", "", "", "psi", "kg/s"])
        self.assertEqual(matrix_rows[2][2:5], ["Initial Status", "PASS", "PASS"])
        self.assertIn("Condition 00", matrix_rows[2][0])
        self.assertIn("Seq 00", matrix_rows[2][1])
        self.assertEqual(matrix_rows[3][:5], ["", "", "Graded Mean", "12", "20"])
        self.assertEqual(matrix_rows[4][:5], ["", "", "Certified Serial Mean", "11", "19"])
        self.assertEqual(matrix_rows[5][:5], ["", "", "Deviation Score", "-1", "-1"])
        self.assertEqual(matrix_rows[6][:5], ["", "", "Official Grade", "WATCH", "PASS"])
        self.assertEqual(matrix_rows[7][:5], ["", "", "Grade Basis", "Program-synced exact-condition final\nSupp: 5 | Valve: 28", "Initial admitted-program cohort"])
        self.assertIn(("SPAN", (0, 2), (0, 7)), style_cmds)
        self.assertIn(("SPAN", (1, 2), (1, 7)), style_cmds)
        self.assertIn(("BACKGROUND", (3, 2), (3, 7), "#fef3c7"), style_cmds)

    def test_generate_auto_report_merges_tabloid_comparison_pages_before_plots(self):
        fitz = __import__("fitz")
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        def _write_pdf(path: Path, sizes: list[tuple[float, float]]) -> None:
            doc = fitz.open()
            try:
                for index, (width, height) in enumerate(sizes, start=1):
                    page = doc.new_page(width=width, height=height)
                    page.insert_text((72, 72), f"{path.stem}-{index}")
                doc.save(str(path))
            finally:
                doc.close()

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            out_pdf = root / "report.pdf"
            ctx = {
                "out_pdf": out_pdf,
                "print_ctx": tar._capture_print_context(),
                "proj": root,
                "wb": root / "project.xlsx",
                "db_path": root / "cache.sqlite3",
                "report_cfg": {},
                "options": {},
                "hi": ["SN1"],
                "meta_by_sn": {"SN1": {}},
                "overall_by_sn": {"SN1": "WATCH"},
                "initial_overall_by_sn": {"SN1": "FAILED"},
                "final_overall_by_sn": {"SN1": "WATCH"},
                "nonpass_findings": [],
                "initial_nonpass_findings": [],
                "runs": ["Run1"],
                "params": ["Pressure", "Flow"],
                "metric_stats": ["mean"],
                "curves_summary": {},
                "initial_watch_items": [],
                "watch_items": [],
                "grading_rows": [],
                "comparison_rows": [
                    {
                        "selection_id": "sel-1",
                        "run_condition": "Condition A",
                        "sequence_text": "Seq A",
                        "serial": "SN1",
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
                        "regrade_applied": True,
                    },
                    {
                        "selection_id": "sel-1",
                        "run_condition": "Condition A",
                        "sequence_text": "Seq A",
                        "serial": "SN1",
                        "parameter": "Flow",
                        "units": "kg/s",
                        "initial_atp_mean": 20.0,
                        "final_atp_mean": 20.0,
                        "initial_actual_mean": 19.0,
                        "final_actual_mean": 19.0,
                        "initial_delta": -1.0,
                        "final_delta": -1.0,
                        "initial_grade": "PASS",
                        "final_grade": "PASS",
                        "regrade_applied": False,
                    },
                ],
                "initial_cohort_specs": [{"cohort_id": "initial:1"}],
                "regrade_cohort_specs": [],
                "equation_cards": [],
                "performance_models": [],
                "report_opts": {},
                "conn": sqlite3.connect(":memory:"),
            }
            captured: dict[str, object] = {}

            def _render_intro(
                intro_pdf: Path,
                *,
                ctx: dict[str, object],
                plot_specs_override: object = None,
                comparison_page_count: object = None,
                progress_cb: object = None,
            ) -> tuple[int, list[object]]:
                _write_pdf(intro_pdf, [(612.0, 792.0), (612.0, 792.0)])
                return 2, ["intro"]

            def _render_plot_sections(_ctx: dict[str, object], *, intro_pages: int, plots_pdf: Path, progress_cb: object = None) -> dict[str, object]:
                captured["plot_intro_pages"] = intro_pages
                _write_pdf(plots_pdf, [(612.0, 792.0)])
                return {
                    "plot_page_count": 1,
                    "metric_plot_count": 1,
                    "curve_plot_count": 0,
                    "run_condition_metric_plot_count": 1,
                    "run_condition_curve_plot_count": 0,
                    "regrade_metric_plot_count": 0,
                    "regrade_curve_plot_count": 0,
                    "performance_plot_count": 0,
                    "watch_plot_count": 0,
                    "plot_specs": [],
                }

            def _render_portrait(out_path: Path, *, story: list[object], print_ctx: object, page_number_offset: int = 0) -> int:
                _write_pdf(out_path, [(612.0, 792.0)])
                return 1

            def _render_comparison(out_path: Path, *, story: list[object], print_ctx: object, page_number_offset: int = 0) -> int:
                _write_pdf(out_path, [(1224.0, 792.0)])
                return 1

            with mock.patch.object(tar, "_reportlab_imports", return_value={}), mock.patch.object(
                tar, "_tar_prepare_base", return_value=ctx
            ), mock.patch.object(
                tar, "_tar_prepare_performance_models", side_effect=lambda _ctx: None
            ), mock.patch.object(
                tar, "_tar_plan_comparison_pages", return_value=[{"serial": "SN1", "serial_page_index": 1, "serial_page_count": 1}]
            ), mock.patch.object(
                tar, "_tar_build_comparison_story", return_value=["comparison"]
            ), mock.patch.object(
                tar, "_tar_prepare_intro_story_with_navigation", return_value=["intro"]
            ), mock.patch.object(
                tar, "_tar_render_stabilized_intro_pdf", side_effect=_render_intro
            ), mock.patch.object(
                tar, "_tar_build_equation_story", return_value=["equations"]
            ), mock.patch.object(
                tar, "_tar_render_plot_sections", side_effect=_render_plot_sections
            ), mock.patch.object(
                tar, "_tar_build_closing_story", return_value=["closing"]
            ), mock.patch.object(
                tar, "_render_portrait_story_pdf", side_effect=_render_portrait
            ), mock.patch.object(
                tar, "_render_tabloid_landscape_story_pdf", side_effect=_render_comparison
            ), mock.patch.object(
                tar, "_tar_apply_pdf_navigation", side_effect=lambda *args, **kwargs: None
            ), mock.patch.object(
                tar, "_write_json", side_effect=lambda path, payload: captured.update({"path": path, "payload": payload})
            ):
                tar.generate_test_data_auto_report(
                    root,
                    root / "project.xlsx",
                    out_pdf,
                    highlighted_serials=["SN1"],
                    options={},
                )

            result = fitz.open(str(out_pdf))
            try:
                self.assertEqual(result.page_count, 6)
                self.assertEqual(int(round(result.load_page(0).rect.width)), 612)
                self.assertEqual(int(round(result.load_page(1).rect.width)), 612)
                self.assertEqual(int(round(result.load_page(2).rect.width)), 1224)
                self.assertEqual(int(round(result.load_page(2).rect.height)), 792)
                self.assertEqual(int(round(result.load_page(3).rect.width)), 612)
            finally:
                result.close()

        self.assertEqual(captured["plot_intro_pages"], 3)

    def test_build_exception_chart_links_targets_regrade_curve_pages(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        ctx = {
            "plot_navigation": [
                {
                    "section_key": "regrade_pass_curve_overlays",
                    "cohort_id": "regrade:cohort-1",
                    "destination_page_index": 8,
                }
            ],
            "comparison_rows": [
                {
                    "serial": "SN1",
                    "run_condition": "Condition A",
                    "sequence_text": "Seq 1",
                    "parameter": "thrust",
                    "final_grade": "WATCH",
                    "regrade_cohort_id": "regrade:cohort-1",
                },
                {
                    "serial": "SN1",
                    "run_condition": "Condition A",
                    "sequence_text": "Seq 2",
                    "parameter": "torque",
                    "final_grade": "PASS",
                    "regrade_cohort_id": "regrade:cohort-1",
                },
            ],
        }

        exception_rows = tar._tar_build_exec_exception_rows(ctx)
        self.assertEqual(len(exception_rows), 1)
        self.assertEqual(exception_rows[0]["chart_target_page_index"], 8)
        self.assertEqual(exception_rows[0]["regrade_cohort_id"], "regrade:cohort-1")
        self.assertTrue(str(exception_rows[0]["chart_label"]).startswith("Chart "))

        chart_links = tar._tar_build_exception_chart_links(ctx)
        self.assertEqual(len(chart_links), 1)
        self.assertEqual(chart_links[0]["destination_page_index"], 8)
        self.assertEqual(chart_links[0]["regrade_cohort_id"], "regrade:cohort-1")

    def test_build_exception_chart_links_prefers_watch_curve_pages(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        ctx = {
            "plot_navigation": [
                {
                    "section_key": "regrade_pass_curve_overlays",
                    "cohort_id": "regrade:cohort-1",
                    "destination_page_index": 8,
                },
                {
                    "section_key": "watch_nonpass_curves",
                    "pair_id": "pair-1",
                    "destination_page_index": 12,
                },
            ],
            "comparison_rows": [
                {
                    "pair_id": "pair-1",
                    "serial": "SN1",
                    "run_condition": "Condition A",
                    "sequence_text": "Seq 1",
                    "parameter": "thrust",
                    "final_grade": "FAIL",
                    "regrade_cohort_id": "regrade:cohort-1",
                }
            ],
        }

        exception_rows = tar._tar_build_exec_exception_rows(ctx)
        self.assertEqual(exception_rows[0]["chart_target_page_index"], 12)
        self.assertEqual(exception_rows[0]["chart_target_section"], "watch_nonpass_curves")

        chart_links = tar._tar_build_exception_chart_links(ctx)
        self.assertEqual(chart_links[0]["destination_page_index"], 12)
        self.assertEqual(chart_links[0]["target_section"], "watch_nonpass_curves")
        self.assertEqual(chart_links[0]["pair_id"], "pair-1")

    def test_generate_auto_report_sidecar_uses_regrade_section_order_and_split_overall_results(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            out_pdf = root / "report.pdf"
            ctx = {
                "out_pdf": out_pdf,
                "print_ctx": tar._capture_print_context(),
                "proj": root,
                "wb": root / "project.xlsx",
                "db_path": root / "cache.sqlite3",
                "report_cfg": {},
                "options": {},
                "hi": ["SN1"],
                "meta_by_sn": {"SN1": {}},
                "overall_by_sn": {"SN1": "WATCH"},
                "initial_overall_by_sn": {"SN1": "FAILED"},
                "final_overall_by_sn": {"SN1": "WATCH"},
                "nonpass_findings": [],
                "initial_nonpass_findings": [],
                "runs": ["Run1"],
                "params": ["thrust"],
                "metric_stats": ["mean"],
                "curves_summary": {},
                "initial_watch_items": [],
                "watch_items": [],
                "grading_rows": [],
                "comparison_rows": [],
                "initial_cohort_specs": [{"cohort_id": "initial:1"}],
                "regrade_cohort_specs": [{"cohort_id": "regrade:1"}],
                "equation_cards": [],
                "performance_models": [],
                "report_opts": {},
                "conn": sqlite3.connect(":memory:"),
            }

            with mock.patch.object(tar, "_reportlab_imports", return_value={}), mock.patch.object(
                tar, "_tar_prepare_base", return_value=ctx
            ), mock.patch.object(
                tar, "_tar_prepare_performance_models", side_effect=lambda _ctx: None
            ), mock.patch.object(
                tar, "_tar_plan_comparison_pages", return_value=[]
            ), mock.patch.object(
                tar, "_tar_build_comparison_story", return_value=[]
            ), mock.patch.object(
                tar, "_tar_prepare_intro_story_with_navigation", return_value=["intro"]
            ), mock.patch.object(tar, "_tar_render_stabilized_intro_pdf", return_value=(2, ["intro"])), mock.patch.object(
                tar, "_tar_build_equation_story", return_value=["equations"]
            ), mock.patch.object(
                tar,
                "_tar_render_plot_sections",
                return_value={
                    "plot_page_count": 5,
                    "metric_plot_count": 2,
                    "curve_plot_count": 2,
                    "run_condition_metric_plot_count": 1,
                    "run_condition_curve_plot_count": 1,
                    "regrade_metric_plot_count": 1,
                    "regrade_curve_plot_count": 1,
                    "performance_plot_count": 1,
                    "watch_plot_count": 1,
                    "plot_specs": [],
                },
            ), mock.patch.object(tar, "_tar_build_closing_story", return_value=["closing"]), mock.patch.object(
                tar, "_render_portrait_story_pdf", side_effect=[1, 1]
            ), mock.patch.object(tar, "_merge_report_pdfs", side_effect=lambda *_args, **_kwargs: None), mock.patch.object(
                tar, "_write_json", side_effect=lambda path, payload: captured.update({"path": path, "payload": payload})
            ):
                tar.generate_test_data_auto_report(
                    root,
                    root / "project.xlsx",
                    out_pdf,
                    highlighted_serials=["SN1"],
                    options={},
                )

        payload = captured["payload"]
        self.assertEqual(payload["version"], 6)
        self.assertEqual(
            payload["section_order"],
            [
                "cover",
                "executive_summary",
                "comparison_table",
                "run_condition_plot_metrics",
                "run_condition_curve_overlays",
                "regrade_pass_plot_metrics",
                "regrade_pass_curve_overlays",
                "performance_equations",
                "performance_plots",
                "watch_nonpass_curves",
                "closing_summary",
            ],
        )
        self.assertEqual(payload["initial_overall_results_by_serial"], {"SN1": "FAILED"})
        self.assertEqual(payload["final_overall_results_by_serial"], {"SN1": "WATCH"})
        self.assertEqual(payload["comparison_exception_rows"], [])
        self.assertEqual(payload["exception_chart_links"], [])

    def _row_spec(
        self,
        tar,
        *,
        pair_id: str,
        run: str,
        selection_label: str,
        base_condition_label: str,
        suppression_value: str,
        series: list,
        x_name: str = "Time",
        param: str = "thrust",
        units: str = "lbf",
    ) -> dict:
        return {
            "pair_id": pair_id,
            "selection_id": pair_id,
            "run": run,
            "run_title": run,
            "selection": {"mode": "condition", "run_name": run},
            "selection_fields": {
                "mode": "condition",
                "run": run,
                "sequence_text": run,
                "condition_text": base_condition_label,
                "display_text": selection_label,
            },
            "selection_label": selection_label,
            "base_condition_label": base_condition_label,
            "suppression_values": [suppression_value] if suppression_value else [],
            "suppression_voltage_label": suppression_value,
            "param": param,
            "units": units,
            "x_name": x_name,
            "series": list(series),
            "series_by_suppression": {suppression_value: list(series)} if suppression_value else {},
            "initial_model": {},
            "initial_plot_payload": {},
            "regrade_models": {},
            "regrade_plot_payloads": {},
        }


if __name__ == "__main__":
    unittest.main()
