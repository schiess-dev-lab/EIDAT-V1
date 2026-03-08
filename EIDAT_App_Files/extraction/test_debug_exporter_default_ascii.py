import sys
import tempfile
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction import debug_exporter  # noqa: E402


def _pipe_rows(ascii_table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for ln in str(ascii_table or "").splitlines():
        if not ln.startswith("|"):
            continue
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        rows.append(parts)
    return rows


class TestDebugExporterDefaultAscii(unittest.TestCase):
    def test_default_ascii_respects_true_column_indices(self) -> None:
        table = {
            "cells": [
                {"row": 0, "col": 0, "text": "A"},
                {"row": 0, "col": 2, "text": "C"},
                {"row": 1, "col": 0, "text": "1"},
                {"row": 1, "col": 1, "text": "2"},
                {"row": 1, "col": 2, "text": "3"},
            ],
            "borderless": False,
        }
        ascii_table = debug_exporter._render_table_ascii(table, mode="default")
        rows = _pipe_rows(ascii_table)
        self.assertEqual(len(rows), 2, msg=f"Unexpected row count:\n{ascii_table}")
        self.assertEqual(rows[0], ["A", "", "C"], msg=f"Column shift detected:\n{ascii_table}")
        self.assertEqual(rows[1], ["1", "2", "3"], msg=f"Unexpected row content:\n{ascii_table}")

    def test_default_ascii_preserves_fully_empty_column(self) -> None:
        table = {
            "cells": [
                {"row": 0, "col": 0, "text": "A"},
                {"row": 0, "col": 2, "text": "C"},
                {"row": 1, "col": 0, "text": "1"},
                {"row": 1, "col": 2, "text": "3"},
            ],
            "borderless": False,
        }
        ascii_table = debug_exporter._render_table_ascii(table, mode="default")
        rows = _pipe_rows(ascii_table)
        self.assertEqual(len(rows), 2, msg=f"Unexpected row count:\n{ascii_table}")
        self.assertEqual(rows[0], ["A", "", "C"], msg=f"Blank structural column was dropped:\n{ascii_table}")
        self.assertEqual(rows[1], ["1", "", "3"], msg=f"Blank structural column was dropped:\n{ascii_table}")

    def test_default_ascii_preserves_fully_empty_row(self) -> None:
        table = {
            "cells": [
                {"row": 0, "col": 0, "text": "A"},
                {"row": 2, "col": 0, "text": "B"},
            ],
            "borderless": False,
        }
        ascii_table = debug_exporter._render_table_ascii(table, mode="default")
        rows = _pipe_rows(ascii_table)
        self.assertEqual(len(rows), 3, msg=f"Blank structural row was dropped:\n{ascii_table}")
        self.assertEqual(rows[0], ["A"], msg=f"Unexpected row content:\n{ascii_table}")
        self.assertEqual(rows[1], [""], msg=f"Blank structural row missing:\n{ascii_table}")
        self.assertEqual(rows[2], ["B"], msg=f"Unexpected row content:\n{ascii_table}")

    def test_combined_text_salvages_fused_majority_for_timed_out_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            page_dir = out_dir / "pages" / "page_1"
            page_dir.mkdir(parents=True, exist_ok=True)
            fused_txt = "[Table 1]\n+----+\n| A |\n+----+\n"
            (page_dir / "variant_fused_majority.txt").write_text(fused_txt, encoding="utf-8")

            output_path = debug_exporter.export_combined_text(
                Path("dummy.pdf"),
                [
                    {
                        "page": 1,
                        "timeout": True,
                        "error": "Timeout: exceeded 3s",
                        "tokens": [],
                        "tables": [],
                        "charts": [],
                        "flow": {},
                        "dpi": 900,
                        "ocr_dpi": 450,
                    }
                ],
                out_dir,
            )

            combined = output_path.read_text(encoding="utf-8")

        self.assertIn("--- PAGE 1 SKIPPED: Timeout: exceeded 3s ---", combined)
        self.assertIn("[Timed-Out Table Salvage]", combined)
        self.assertIn("[Table 1]", combined)
        self.assertIn("| A |", combined)


if __name__ == "__main__":
    unittest.main()
