import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction.labeled_tables_exporter import export_labeled_tables_db  # noqa: E402


class TestLabeledTablesExporter(unittest.TestCase):
    def test_exports_labeled_tables(self) -> None:
        content = "\n".join(
            [
                "=== Page 1 ===",
                "[Table/Chart Title]",
                "FAKE_EIDP_REPEAT_TABLE - Acceptance Test Data",
                "",
                "[TABLE_LABEL]",
                "Acceptance Test Data",
                "",
                "+----------------------+--------------+-----------+---------------+--------+",
                "| Parameter            | Resistance   | Voltage   | Valve Voltage | Notes  |",
                "+======================+==============+===========+===============+========+",
                "| Test Temp            | 10 kohm      | 28 V      | 5.00 V        | PASS   |",
                "+----------------------+--------------+-----------+---------------+--------+",
                "",
                "[STRING]",
                "Acceptance Test Data (Run B)",
                "",
                "[TABLE_LABEL]",
                "Acceptance Test Data (2)",
                "",
                "+----------------------+--------------+-----------+---------------+--------+",
                "| Parameter            | Resistance   | Voltage   | Valve Voltage | Notes  |",
                "+======================+==============+===========+===============+========+",
                "| Test Temp            | 12 kohm      | 28 V      | 5.20 V        | PASS   |",
                "+----------------------+--------------+-----------+---------------+--------+",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            combined = art / "combined.txt"
            combined.write_text(content + "\n", encoding="utf-8")

            db_path = export_labeled_tables_db(artifacts_dir=art, combined_txt_path=combined, marker="TABLE_LABEL")
            self.assertIsNotNone(db_path)
            self.assertTrue((art / "labeled_tables.db").exists())

            conn = sqlite3.connect(str(art / "labeled_tables.db"))
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT table_index, page, heading, label, label_base, label_instance, n_rows, n_cols FROM tables ORDER BY table_index"
                ).fetchall()
                self.assertEqual(len(rows), 2)

                r1 = dict(rows[0])
                self.assertEqual(r1["table_index"], 1)
                self.assertEqual(r1["page"], 1)
                self.assertIn("Acceptance Test Data", str(r1["heading"]))
                self.assertEqual(r1["label"], "Acceptance Test Data")
                self.assertEqual(r1["label_base"], "Acceptance Test Data")
                self.assertIsNone(r1["label_instance"])
                self.assertGreaterEqual(int(r1["n_rows"]), 2)
                self.assertGreaterEqual(int(r1["n_cols"]), 5)

                r2 = dict(rows[1])
                self.assertEqual(r2["table_index"], 2)
                self.assertEqual(r2["label"], "Acceptance Test Data (2)")
                self.assertEqual(r2["label_base"], "Acceptance Test Data")
                self.assertEqual(r2["label_instance"], 2)

                # Verify a few cells were written.
                t1_id = conn.execute("SELECT id FROM tables WHERE table_index = 1").fetchone()["id"]
                header_param = conn.execute(
                    "SELECT text FROM cells WHERE table_id = ? AND row_index = 0 AND col_index = 0",
                    (t1_id,),
                ).fetchone()["text"]
                self.assertEqual(header_param, "Parameter")

                data_res = conn.execute(
                    "SELECT text FROM cells WHERE table_id = ? AND row_index = 1 AND col_index = 1",
                    (t1_id,),
                ).fetchone()["text"]
                self.assertEqual(data_res, "10 kohm")
            finally:
                conn.close()

    def test_no_labels_deletes_db(self) -> None:
        content = "\n".join(
            [
                "=== Page 1 ===",
                "+-----+-----+",
                "| A   | B   |",
                "+=====+=====+",
                "| 1   | 2   |",
                "+-----+-----+",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            art = Path(td)
            combined = art / "combined.txt"
            combined.write_text(content + "\n", encoding="utf-8")

            # Pre-create a dummy DB to ensure exporter removes it when no labels exist.
            dummy = art / "labeled_tables.db"
            conn = sqlite3.connect(str(dummy))
            try:
                conn.execute("CREATE TABLE t(x INTEGER)")
                conn.commit()
            finally:
                conn.close()
            self.assertTrue(dummy.exists())

            db_path = export_labeled_tables_db(artifacts_dir=art, combined_txt_path=combined, marker="TABLE_LABEL")
            self.assertIsNone(db_path)
            self.assertFalse(dummy.exists())


if __name__ == "__main__":
    unittest.main()
