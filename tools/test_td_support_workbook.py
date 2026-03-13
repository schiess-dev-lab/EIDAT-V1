import json
import math
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


@unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
class TestTDSupportWorkbook(unittest.TestCase):
    def _make_source_sqlite(self, path: Path) -> None:
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
            rows = [
                (1, 0.0, 100.0, 5.0, 10.0),
                (2, 1.0, 100.0, 5.0, 20.0),
                (3, 2.0, 100.0, 5.0, 30.0),
                (4, 3.0, 100.0, 5.0, 40.0),
                (5, 4.0, 100.0, 5.0, 50.0),
                (6, 5.0, 100.0, 5.0, 60.0),
            ]
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
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_metrics_calc
                    (serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (serial, run, "impulse bit", "mean", float(x_val), 0, 0),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_metrics_calc
                    (serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (serial, run, "thrust", "mean", float(y_val), 0, 0),
                )
            conn.commit()
        return db_path

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
                self.assertEqual(wb.sheetnames, ["Settings", "Sequences", "ParameterBounds"])
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

                ws_seq = wb["Sequences"]
                self.assertEqual(ws_seq.cell(2, 1).value, "RunA")
                self.assertEqual(ws_seq.cell(2, 2).value, "RunA")
                self.assertIsNone(ws_seq.cell(2, 3).value)
                self.assertEqual(ws_seq.cell(1, 4).value, "feed_pressure_units")
                self.assertEqual(ws_seq.cell(1, 5).value, "run_type")
                self.assertEqual(ws_seq.cell(1, 6).value, "pulse_width")
                self.assertEqual(ws_seq.cell(1, 7).value, "control_period")
                self.assertTrue(bool(ws_seq.cell(2, 10).value))

                ws_bounds = wb["ParameterBounds"]
                self.assertEqual(ws_bounds.cell(2, 1).value, "RunA")
                self.assertEqual(ws_bounds.cell(2, 2).value, "thrust")
                self.assertEqual(ws_bounds.cell(2, 3).value, "lbf")
                self.assertIsNone(ws_bounds.cell(2, 4).value)
            finally:
                wb.close()

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
                ws_seq = wb["Sequences"]
                ws_seq.cell(2, 1).value = "Seq1"
                ws_seq.cell(2, 2).value = "RunA"
                ws_seq.cell(2, 3).value = 100
                ws_seq.cell(2, 6).value = 5
                ws_bounds = wb["ParameterBounds"]
                ws_bounds.cell(2, 1).value = "Seq1"
                ws_bounds.cell(2, 2).value = "thrust"
                ws_bounds.cell(2, 4).value = 15
                ws_bounds.cell(2, 5).value = 45
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
                self.assertAlmostEqual(float(mean_row[0]), 55.0)

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertEqual(values.get("Seq1.thrust.mean"), 55.0)
            finally:
                wb2.close()

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
                ws_seq = wb["Sequences"]
                rows = {
                    str(ws_seq.cell(r, 1).value or "").strip(): r
                    for r in range(2, (ws_seq.max_row or 0) + 1)
                }
                for seq_name in ("Seq1", "Seq2"):
                    row = rows[seq_name]
                    ws_seq.cell(row, 3).value = 350
                    ws_seq.cell(row, 4).value = "psia"
                    ws_seq.cell(row, 5).value = "steady state"
                row = rows["Seq3"]
                ws_seq.cell(row, 3).value = 350
                ws_seq.cell(row, 4).value = "psia"
                ws_seq.cell(row, 5).value = "pulsed mode"
                ws_seq.cell(row, 6).value = 60
                ws_seq.cell(row, 7).value = 120
                wb.save(str(support_path))
            finally:
                wb.close()

            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_impl_tables(conn)
                for run in ("Seq1", "Seq2", "Seq3"):
                    conn.execute(
                        "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name) VALUES (?, ?, ?)",
                        (run, "Time", ""),
                    )
                conn.commit()

            views = be.td_list_run_selection_views(db_path, wb_path, project_dir=root)
            seq_items = views.get("sequence") or []
            cond_items = views.get("condition") or []

            self.assertEqual([item.get("display_text") for item in seq_items], ["Seq1", "Seq2", "Seq3"])
            self.assertEqual([item.get("display_text") for item in cond_items], ["350 psia, PM, 60 Sec ON / 120 Sec OFF", "350 psia, SS"])
            ss_group = next(item for item in cond_items if item.get("display_text") == "350 psia, SS")
            self.assertEqual(ss_group.get("member_runs"), ["Seq1", "Seq2"])
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
                ws_seq = wb["Sequences"]
                ws_seq.cell(2, 1).value = "Seq1"
                ws_seq.cell(2, 2).value = "RunA"
                ws_seq.cell(2, 3).value = 100
                ws_seq.cell(2, 6).value = 5
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
                ws_seq = wb["Sequences"]
                ws_seq.cell(2, 1).value = "Seq1"
                ws_seq.cell(2, 2).value = "RunA"
                ws_seq.cell(2, 6).value = 5
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
                ws_seq = wb["Sequences"]
                ws_seq.cell(1, 6).value = "pulse_width_on"
                ws_seq.cell(2, 6).value = 7
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
            self.assertAlmostEqual(float(mean_row[0]), 55.0)

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                values = {
                    str(ws_calc.cell(r, 1).value or "").strip(): ws_calc.cell(r, 2).value
                    for r in range(1, (ws_calc.max_row or 0) + 1)
                }
                self.assertEqual(values.get("RunA.thrust.mean"), 55.0)
            finally:
                wb2.close()

    def test_update_workbook_fails_when_cache_db_missing(self) -> None:
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

            with self.assertRaisesRegex(RuntimeError, "Build / Refresh Cache"):
                be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)

    def test_update_workbook_fails_when_cache_db_incomplete(self) -> None:
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

            db_path = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_tables(conn)
                conn.commit()

            with self.assertRaisesRegex(RuntimeError, "Project cache DB is incomplete"):
                be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)

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

            with self.assertRaisesRegex(RuntimeError, "Build / Refresh Cache"):
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
            self.assertEqual(curves, [{"serial": "SN1", "x": [0, 1, 2], "y": [10, 20, 30]}])

    def test_rebuild_writes_excel_mirror_for_project_cache(self) -> None:
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
            self.assertTrue(impl_mirror_path.exists())
            self.assertTrue(raw_mirror_path.exists())
            self.assertTrue(raw_points_path.exists())

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
                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1", "[0,1,2,3]", "[10,20,30,40]", 4, 123, 123),
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
                rows = conn.execute(
                    """
                    SELECT stat, value_num
                    FROM td_metrics_calc
                    WHERE serial='SN1' AND run_name='RunA' AND column_name='thrust'
                    ORDER BY stat
                    """
                ).fetchall()
            self.assertEqual(rows, [("max", 40.0), ("mean", 25.0)])

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
                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1", "[0,1,2,3]", "[10,20,30,40]", 4, 123, 123),
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
            self.assertEqual(rows, [("max", 40.0), ("mean", 25.0)])

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
                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1", "[0,1,2,3]", "[10,20,30,40]", 4, 123, 123),
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
                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("LegacyRun", "thrust", "Time", "SN1", "[0,1]", "[2,3]", 2, 1, 1),
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

            with sqlite3.connect(str(root / "test_data_raw_cache.sqlite3")) as conn:
                row = conn.execute(
                    "SELECT y_name, x_name FROM td_curves_raw WHERE run_name=? AND serial=?",
                    ("RunA", "SN1"),
                ).fetchone()
                self.assertEqual(row, ("thrust", "Time"))

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

        xs = list(range(10))
        ys = [1.0, 1.4, 1.8, 5.0, 8.2, 11.4, 10.8, 10.2, 9.6, 9.0]
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
        self.assertAlmostEqual(pred_at_break, 4.0, places=6)

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

    def test_perf_fit_family_label_includes_piecewise_modes(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        self.assertEqual(be.td_perf_fit_family_label("piecewise_auto"), "Piecewise Auto")
        self.assertEqual(be.td_perf_fit_family_label("piecewise_2"), "Piecewise 2-Segment")
        self.assertEqual(be.td_perf_fit_family_label("piecewise_3"), "Piecewise 3-Segment")

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
