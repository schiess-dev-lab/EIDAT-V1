import json
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

            wb = load_workbook(str(support), read_only=True, data_only=True)
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

                ws_seq = wb["Sequences"]
                self.assertEqual(ws_seq.cell(2, 1).value, "RunA")
                self.assertEqual(ws_seq.cell(2, 2).value, "RunA")
                self.assertIsNone(ws_seq.cell(2, 3).value)
                self.assertTrue(bool(ws_seq.cell(2, 7).value))

                ws_bounds = wb["ParameterBounds"]
                self.assertEqual(ws_bounds.cell(2, 1).value, "RunA")
                self.assertEqual(ws_bounds.cell(2, 2).value, "thrust")
                self.assertEqual(ws_bounds.cell(2, 3).value, "lbf")
                self.assertIsNone(ws_bounds.cell(2, 4).value)
            finally:
                wb.close()

    def test_rebuild_uses_support_sequence_name_and_filters(self) -> None:
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

            from openpyxl import load_workbook  # type: ignore

            wb = load_workbook(str(support_path))
            try:
                ws_settings = wb["Settings"]
                ws_settings.cell(3, 2).value = 2
                ws_seq = wb["Sequences"]
                ws_seq.cell(2, 1).value = "Seq1"
                ws_seq.cell(2, 2).value = "RunA"
                ws_seq.cell(2, 3).value = 100
                ws_seq.cell(2, 4).value = 5
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

            with sqlite3.connect(str(out_db)) as conn:
                row = conn.execute(
                    "SELECT x_json, y_json FROM td_curves_raw WHERE run_name=? AND y_name=? AND x_name=? AND serial=?",
                    ("Seq1", "thrust", "Time", "SN1"),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(json.loads(row[0]), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                self.assertEqual(json.loads(row[1]), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics_calc WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
                    ("Seq1", "thrust", "mean", "SN1"),
                ).fetchone()
                self.assertIsNotNone(mean_row)
                self.assertAlmostEqual(float(mean_row[0]), 35.0)

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
                ws_seq.cell(2, 4).value = 5
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
            db_path = root / "implementation_trending.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                be._ensure_test_data_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name) VALUES (?, ?, ?)",
                    ("RunA", "Time", ""),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_curves_raw
                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Time", "SN1", "[0,1,2]", "[1,2,3]", 3, 1, 1),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_curves_raw
                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("RunA", "thrust", "Pulse Number", "SN1", "[1,2,3]", "[1,2,3]", 3, 1, 1),
                )
                conn.commit()

            xs = be.td_list_x_columns(db_path, "RunA")
            self.assertEqual(xs, ["Time", "Pulse Number"])

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
            mirror_path = db_path.with_suffix(".xlsx")
            self.assertTrue(mirror_path.exists())

            wb = load_workbook(str(mirror_path), read_only=True, data_only=True)
            try:
                self.assertIn("td_curves_raw", wb.sheetnames)
                self.assertIn("td_metrics_calc", wb.sheetnames)
            finally:
                wb.close()

    def test_calc_cache_can_be_rebuilt_from_raw_without_source_sqlite(self) -> None:
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
