import sys
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

    def test_default_ascii_prunes_fully_empty_column(self) -> None:
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
        self.assertEqual(rows[0], ["A", "C"], msg=f"Empty column not pruned:\n{ascii_table}")
        self.assertEqual(rows[1], ["1", "3"], msg=f"Unexpected row content:\n{ascii_table}")

    def test_default_ascii_prunes_fully_empty_row(self) -> None:
        table = {
            "cells": [
                {"row": 0, "col": 0, "text": "A"},
                {"row": 2, "col": 0, "text": "B"},
            ],
            "borderless": False,
        }
        ascii_table = debug_exporter._render_table_ascii(table, mode="default")
        rows = _pipe_rows(ascii_table)
        self.assertEqual(len(rows), 2, msg=f"Unexpected row count:\n{ascii_table}")
        self.assertEqual(rows[0], ["A"], msg=f"Unexpected row content:\n{ascii_table}")
        self.assertEqual(rows[1], ["B"], msg=f"Unexpected row content:\n{ascii_table}")


if __name__ == "__main__":
    unittest.main()

