import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


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
                    "SELECT x_json, y_json FROM td_curves WHERE run_name=? AND y_name=? AND x_name=? AND serial=?",
                    ("Seq1", "thrust", "Time", "SN1"),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(json.loads(row[0]), [2.0, 3.0])
                self.assertEqual(json.loads(row[1]), [30.0, 40.0])

                mean_row = conn.execute(
                    "SELECT value_num FROM td_metrics WHERE run_name=? AND column_name=? AND stat=? AND serial=?",
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

            result = be.update_test_data_trending_project_workbook(root, wb_path, overwrite=True)
            self.assertEqual(str(result.get("workbook") or ""), str(wb_path))

            wb2 = load_workbook(str(wb_path), read_only=True, data_only=True)
            try:
                ws_calc = wb2["Data_calc"]
                metrics = [str(ws_calc.cell(r, 1).value or "").strip() for r in range(1, (ws_calc.max_row or 0) + 1)]
                self.assertIn("Seq1.thrust.mean", metrics)

                ws_data = wb2["Data"]
                data_groups = [str(ws_data.cell(r, 6).value or "").strip() for r in range(2, (ws_data.max_row or 0) + 1)]
                self.assertIn("Seq1", data_groups)
            finally:
                wb2.close()


if __name__ == "__main__":
    unittest.main()
