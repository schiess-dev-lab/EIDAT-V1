import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "EIDAT_App_Files" / "Application"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _have_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
class TestTDExcelHeaderMapping(unittest.TestCase):
    def _build_workbook(
        self,
        path: Path,
        *,
        headers: list[tuple[int, int, object]],
        data_rows: list[list[object]],
        title: str = "Seq1",
    ) -> None:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        ws = wb.active
        ws.title = str(title)
        for row, col, value in headers:
            ws.cell(row=row, column=col).value = value
        for idx, row_vals in enumerate(data_rows, start=3):
            for col, value in enumerate(row_vals, start=1):
                ws.cell(row=idx, column=col).value = value
        wb.save(str(path))
        try:
            wb.close()
        except Exception:
            pass

    def _build_multi_sheet_workbook(
        self,
        path: Path,
        *,
        sheets: list[dict[str, object]],
    ) -> None:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        first = True
        for spec in sheets:
            if first:
                ws = wb.active
                first = False
            else:
                ws = wb.create_sheet()
            ws.title = str(spec.get("title") or "Sheet1")
            for row, col, value in list(spec.get("headers") or []):
                ws.cell(row=int(row), column=int(col)).value = value
            for row_idx, row_vals in enumerate(list(spec.get("data_rows") or []), start=3):
                for col_idx, value in enumerate(list(row_vals), start=1):
                    ws.cell(row=row_idx, column=col_idx).value = value
        wb.save(str(path))
        try:
            wb.close()
        except Exception:
            pass

    def _import_workbook(
        self,
        workbook_path: Path,
        sqlite_path: Path,
        *,
        synthesize_td_seq_aliases: bool = False,
    ) -> dict:
        import eidat_manager_excel_to_sqlite as etm  # type: ignore

        return etm._write_workbook_sqlite(  # type: ignore[attr-defined]
            excel_path=workbook_path,
            sqlite_path=sqlite_path,
            overwrite=True,
            synthesize_td_seq_aliases=bool(synthesize_td_seq_aliases),
            max_scan_rows=200,
            max_cols=200,
            lookahead_rows=60,
            min_numeric_count=8,
            min_numeric_ratio=0.60,
            min_data_cols=1,
        )

    def _sheet_info_row(
        self,
        sqlite_path: Path,
        *,
        sheet_name: str,
    ) -> tuple[list[str], list[str], list[str], str]:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            row = conn.execute(
                "SELECT headers_json, mapped_headers_json, table_name, COALESCE(source_sheet_name, '') FROM __sheet_info WHERE sheet_name=? LIMIT 1",
                (sheet_name,),
            ).fetchone()
            self.assertIsNotNone(row, f"expected __sheet_info row for {sheet_name}")
            headers = json.loads(row[0])
            mapped = json.loads(row[1])
            table_name = str(row[2])
            source_sheet_name = str(row[3] or "")
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()]
        finally:
            conn.close()
        return headers, mapped, cols, source_sheet_name

    def _sheet_info_names(self, sqlite_path: Path) -> list[tuple[str, str]]:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            rows = conn.execute(
                "SELECT sheet_name, COALESCE(source_sheet_name, '') FROM __sheet_info ORDER BY rowid"
            ).fetchall()
        finally:
            conn.close()
        return [(str(row[0] or ""), str(row[1] or "")) for row in rows]

    def test_importer_reconstructs_split_and_merged_headers(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            workbook_path = root / "td.xlsx"
            sqlite_path = root / "td.sqlite3"
            headers = [
                (1, 1, "Seq Time"),
                (2, 1, "(sec)"),
                (1, 2, "Ti 10% Pc"),
                (2, 2, "(msec)"),
                (1, 3, "Thrust-N Calc"),
                (2, 3, "(lbf)"),
                (1, 4, "Rough"),
                (2, 4, "+/- P-to-P (+/- %)"),
                (2, 5, "2 sigma (%)"),
            ]
            data_rows = [
                [float(i), float(i) * 0.1, float(i) * 10.0, float(i) * 0.2, float(i) * 0.05]
                for i in range(20)
            ]
            self._build_workbook(workbook_path, headers=headers, data_rows=data_rows)

            from openpyxl import load_workbook  # type: ignore

            wb = load_workbook(str(workbook_path))
            try:
                wb["Seq1"].merge_cells(start_row=1, start_column=4, end_row=1, end_column=5)
                wb.save(str(workbook_path))
            finally:
                try:
                    wb.close()
                except Exception:
                    pass

            self._import_workbook(workbook_path, sqlite_path)
            raw_headers, mapped_headers, sqlite_cols, source_sheet_name = self._sheet_info_row(
                sqlite_path,
                sheet_name="Seq1",
            )

            self.assertEqual(
                raw_headers[:5],
                [
                    "Seq Time (sec)",
                    "Ti 10% Pc (msec)",
                    "Thrust-N Calc (lbf)",
                    "Rough +/- P-to-P (+/- %)",
                    "Rough 2 sigma (%)",
                ],
            )
            self.assertEqual(
                mapped_headers[:5],
                ["Time", "Ti_10_Pc_msec", "Thrust", "Rough", "Rough_2_sigma"],
            )
            self.assertEqual(
                sqlite_cols[:6],
                ["excel_row", "Time", "Ti_10_Pc_msec", "Thrust", "Rough", "Rough_2_sigma"],
            )
            self.assertEqual(source_sheet_name, "Seq1")

    def test_importer_strips_units_and_protects_short_tokens(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            workbook_path = root / "td_units.xlsx"
            sqlite_path = root / "td_units.sqlite3"
            headers = [
                (1, 1, "Time"),
                (2, 1, "sec"),
                (1, 2, "Ti"),
                (2, 2, "(msec)"),
                (1, 3, "Pf"),
                (2, 3, "(psia)"),
                (1, 4, "Throat Area, Hot"),
                (2, 4, "(in^2)"),
            ]
            data_rows = [
                [float(i), float(i) * 0.5, 300.0 + float(i), 0.02 + (float(i) * 0.001)]
                for i in range(20)
            ]
            self._build_workbook(workbook_path, headers=headers, data_rows=data_rows)

            self._import_workbook(workbook_path, sqlite_path)
            raw_headers, mapped_headers, sqlite_cols, source_sheet_name = self._sheet_info_row(
                sqlite_path,
                sheet_name="Seq1",
            )

            self.assertEqual(raw_headers[:4], ["Time sec", "Ti (msec)", "Pf (psia)", "Throat Area, Hot (in^2)"])
            self.assertEqual(mapped_headers[:4], ["Time", "Ti_10_Pc_msec", "Pf", "Throat_Area_Hot"])
            self.assertIn("Time", sqlite_cols)
            self.assertIn("Ti_10_Pc_msec", sqlite_cols)
            self.assertNotEqual(mapped_headers[1], "Time")
            self.assertEqual(source_sheet_name, "Seq1")

    def test_importer_synthesizes_seq_name_for_single_non_seq_td_sheet(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            workbook_path = root / "td_datapages.xlsx"
            sqlite_path = root / "td_datapages.sqlite3"
            headers = [
                (1, 1, "Time"),
                (1, 2, "Thrust"),
            ]
            data_rows = [[float(i), float(i) * 2.0] for i in range(20)]
            self._build_workbook(workbook_path, headers=headers, data_rows=data_rows, title="DataPages")

            payload = self._import_workbook(
                workbook_path,
                sqlite_path,
                synthesize_td_seq_aliases=True,
            )
            raw_headers, mapped_headers, sqlite_cols, source_sheet_name = self._sheet_info_row(
                sqlite_path,
                sheet_name="seq_1",
            )

            self.assertEqual(source_sheet_name, "DataPages")
            self.assertEqual(raw_headers[:2], ["Time", "Thrust"])
            self.assertEqual(mapped_headers[:2], ["Time", "Thrust"])
            self.assertIn("Time", sqlite_cols)
            self.assertEqual(
                self._sheet_info_names(sqlite_path),
                [("seq_1", "DataPages")],
            )
            self.assertEqual(str((payload.get("sheets") or [])[0].get("sheet") or ""), "seq_1")
            self.assertEqual(str((payload.get("sheets") or [])[0].get("source_sheet_name") or ""), "DataPages")

    def test_importer_synthesizes_seq_names_in_detected_sheet_order(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            workbook_path = root / "td_multi.xlsx"
            sqlite_path = root / "td_multi.sqlite3"
            sheets = [
                {
                    "title": "Notes",
                    "headers": [(1, 1, "memo")],
                    "data_rows": [],
                },
                {
                    "title": "DataPages",
                    "headers": [(1, 1, "Time"), (1, 2, "Thrust")],
                    "data_rows": [[float(i), float(i) * 2.0] for i in range(20)],
                },
                {
                    "title": "PulseSummary",
                    "headers": [(1, 1, "Time"), (1, 2, "Pf")],
                    "data_rows": [[float(i), 250.0 + float(i)] for i in range(20)],
                },
            ]
            self._build_multi_sheet_workbook(workbook_path, sheets=sheets)

            self._import_workbook(workbook_path, sqlite_path, synthesize_td_seq_aliases=True)

            self.assertEqual(
                self._sheet_info_names(sqlite_path),
                [("seq_1", "DataPages"), ("seq_2", "PulseSummary")],
            )

    def test_importer_skips_synthetic_seq_aliasing_when_real_seq_tab_exists(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            workbook_path = root / "td_existing_seq.xlsx"
            sqlite_path = root / "td_existing_seq.sqlite3"
            sheets = [
                {
                    "title": "Seq1",
                    "headers": [(1, 1, "Time"), (1, 2, "Thrust")],
                    "data_rows": [[float(i), float(i) * 2.0] for i in range(20)],
                },
                {
                    "title": "DataPages",
                    "headers": [(1, 1, "Time"), (1, 2, "Pf")],
                    "data_rows": [[float(i), 250.0 + float(i)] for i in range(20)],
                },
            ]
            self._build_multi_sheet_workbook(workbook_path, sheets=sheets)

            self._import_workbook(workbook_path, sqlite_path, synthesize_td_seq_aliases=True)

            self.assertEqual(
                self._sheet_info_names(sqlite_path),
                [("Seq1", "Seq1"), ("DataPages", "DataPages")],
            )

    def test_importer_preserves_original_sheet_names_when_synthetic_flag_disabled(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            workbook_path = root / "td_no_synth.xlsx"
            sqlite_path = root / "td_no_synth.sqlite3"
            headers = [(1, 1, "Time"), (1, 2, "Thrust")]
            data_rows = [[float(i), float(i) * 2.0] for i in range(20)]
            self._build_workbook(workbook_path, headers=headers, data_rows=data_rows, title="DataPages")

            self._import_workbook(workbook_path, sqlite_path, synthesize_td_seq_aliases=False)

            self.assertEqual(
                self._sheet_info_names(sqlite_path),
                [("DataPages", "DataPages")],
            )


if __name__ == "__main__":
    unittest.main()
