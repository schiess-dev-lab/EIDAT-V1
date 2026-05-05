from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class TestEidatManagerExcelToSqlite(unittest.TestCase):
    def _write_td_workbook_with_sparkline_ext(self, path: Path) -> None:
        try:
            from openpyxl import Workbook  # type: ignore
        except Exception as exc:
            self.skipTest(f"openpyxl not available: {exc}")

        wb = Workbook()
        ws = wb.active
        ws.title = "Seq 1"
        ws.append(["Time", "Thrust"])
        for idx in range(1, 12):
            ws.append([idx, idx * 2])
        wb.save(str(path))
        try:
            wb.close()
        except Exception:
            pass

        sparkline_ext = '<extLst><ext uri="{05C60535-1F16-4FD2-B633-F4F36F0B64E0}"/></extLst>'
        tmp_path = path.with_name(path.stem + ".tmp.xlsx")
        with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(
            tmp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "xl/worksheets/sheet1.xml":
                    text = data.decode("utf-8")
                    text = text.replace("</worksheet>", sparkline_ext + "</worksheet>")
                    data = text.encode("utf-8")
                zout.writestr(info, data)
        tmp_path.replace(path)

    def test_sqlite_writer_ignores_sparkline_extension_parse_failure(self) -> None:
        try:
            from openpyxl.worksheet import _reader as worksheet_reader  # type: ignore
        except Exception as exc:
            self.skipTest(f"openpyxl not available: {exc}")

        import eidat_manager_excel_to_sqlite as mod

        parser_cls = worksheet_reader.WorkSheetParser
        original_parse_extensions = parser_cls.parse_extensions

        def _fail_on_extension(self, element):  # type: ignore[no-untyped-def]
            raise RuntimeError("Sparkline Group extension parse failure")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xlsx_path = root / "td_with_sparkline.xlsx"
            sqlite_path = root / "td_with_sparkline.sqlite3"
            self._write_td_workbook_with_sparkline_ext(xlsx_path)

            parser_cls.parse_extensions = _fail_on_extension
            try:
                result = mod._write_workbook_sqlite(
                    excel_path=xlsx_path,
                    sqlite_path=sqlite_path,
                    overwrite=True,
                    max_scan_rows=30,
                    max_cols=20,
                    lookahead_rows=20,
                    min_numeric_count=4,
                    min_numeric_ratio=0.5,
                    min_data_cols=1,
                    synthesize_td_seq_aliases=True,
                )
            finally:
                parser_cls.parse_extensions = original_parse_extensions

            self.assertTrue(sqlite_path.exists(), "SQLite output should still be created")
            self.assertEqual(str(result.get("sqlite_path")), str(sqlite_path))

            conn = sqlite3.connect(str(sqlite_path))
            try:
                rows = conn.execute("SELECT sheet_name, rows_inserted FROM __sheet_info").fetchall()
            finally:
                conn.close()
            self.assertEqual(rows, [("Seq 1", 11)])

    def test_load_mat_payload_reports_scipy_import_context(self) -> None:
        import builtins
        import eidat_manager_excel_to_sqlite as mod

        original_import = builtins.__import__

        def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "scipy.io" or str(name).startswith("scipy"):
                raise ImportError("DLL load failed while importing scipy.io")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=_blocked_import):
            with self.assertRaises(RuntimeError) as ctx:
                mod._load_mat_payload(Path("sample.mat"))

        message = str(ctx.exception)
        self.assertIn("Failed to import `scipy.io.loadmat`", message)
        self.assertIn("DLL load failed while importing scipy.io", message)
        self.assertIn(sys.executable, message)
        self.assertIn(sys.version.split()[0], message)


if __name__ == "__main__":
    unittest.main()
