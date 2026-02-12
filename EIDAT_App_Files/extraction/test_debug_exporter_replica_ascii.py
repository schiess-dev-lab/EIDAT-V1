import os
import sys
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction import debug_exporter  # noqa: E402


class TestDebugExporterReplicaAscii(unittest.TestCase):
    def test_replica_ascii_drops_fully_empty_rows(self) -> None:
        old_env = {
            "EIDAT_TABLE_ASCII_SNAP_TOL_PX": os.environ.get("EIDAT_TABLE_ASCII_SNAP_TOL_PX"),
            "EIDAT_TABLE_ASCII_SNAP_TOL_RATIO": os.environ.get("EIDAT_TABLE_ASCII_SNAP_TOL_RATIO"),
            "EIDAT_TABLE_ASCII_SNAP_MAX_PX": os.environ.get("EIDAT_TABLE_ASCII_SNAP_MAX_PX"),
        }
        try:
            # Force deterministic snapping behavior for this synthetic table.
            os.environ["EIDAT_TABLE_ASCII_SNAP_TOL_PX"] = "0"
            os.environ["EIDAT_TABLE_ASCII_SNAP_TOL_RATIO"] = "0.006"
            os.environ["EIDAT_TABLE_ASCII_SNAP_MAX_PX"] = "25"

            # Synthetic 3-column table with an intentional vertical gap band (no cells)
            # between two content rows.
            cells = []
            col_x = [(0, 100), (100, 200), (200, 300)]

            def add_row(y0: float, y1: float, texts: list[str]) -> None:
                for i, (x0, x1) in enumerate(col_x):
                    cells.append({"bbox_px": [x0, y0, x1, y1], "text": texts[i]})

            add_row(0, 20, ["Parameter", "Spec", "Result"])
            add_row(20, 40, ["Flow Rate", "10-12", "14.4"])
            # Gap from 40..55 should not become a blank ASCII row.
            add_row(55, 75, ["Pressure", "45.55", "52"])

            table = {"cells": cells, "borderless": False}
            ascii_table = debug_exporter._render_table_ascii(table, mode="replica")

            row_lines = [ln for ln in ascii_table.splitlines() if ln.startswith("|")]
            self.assertEqual(
                len(row_lines),
                3,
                msg=f"Unexpected row count (blank row likely inserted):\n{ascii_table}",
            )
            self.assertTrue(
                all(ln.strip().strip("|").strip() != "" for ln in row_lines),
                msg=f"Found a fully empty ASCII row:\n{ascii_table}",
            )
            self.assertIn("Flow Rate", ascii_table)
            self.assertIn("Pressure", ascii_table)
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_replica_ascii_merges_outer_frame_sliver_column(self) -> None:
        old_env = {
            "EIDAT_TABLE_ASCII_SNAP_TOL_PX": os.environ.get("EIDAT_TABLE_ASCII_SNAP_TOL_PX"),
            "EIDAT_TABLE_ASCII_SNAP_TOL_RATIO": os.environ.get("EIDAT_TABLE_ASCII_SNAP_TOL_RATIO"),
            "EIDAT_TABLE_ASCII_SNAP_MAX_PX": os.environ.get("EIDAT_TABLE_ASCII_SNAP_MAX_PX"),
        }
        try:
            # Auto tolerance (snap_tol_px=0) should still collapse small repeated edge gaps.
            os.environ["EIDAT_TABLE_ASCII_SNAP_TOL_PX"] = "0"
            os.environ["EIDAT_TABLE_ASCII_SNAP_TOL_RATIO"] = "0.006"
            os.environ["EIDAT_TABLE_ASCII_SNAP_MAX_PX"] = "25"

            # Header row uses full outer frame, data rows inset by ~9px on each side.
            cells = []
            header_x = [(0, 100), (100, 200), (200, 300), (300, 400)]
            inset_x = [(9, 91), (109, 191), (209, 291), (309, 391)]

            def add_row(y0: float, y1: float, x_pairs, texts: list[str]) -> None:
                for i, (x0, x1) in enumerate(x_pairs):
                    cells.append({"bbox_px": [x0, y0, x1, y1], "text": texts[i]})

            add_row(0, 20, header_x, ["B1", "B2", "B3", "B4"])
            add_row(20, 40, inset_x, ["\"", "2.2", "3.3", "44"])
            add_row(40, 60, inset_x, ["5.5", "6.6", "7.7", "8.8"])
            add_row(60, 80, inset_x, ["9.9", "10.0", "11.1", "12.2"])

            table = {"cells": cells, "borderless": False}
            ascii_table = debug_exporter._render_table_ascii(table, mode="replica")

            # Ensure we didn't produce a blank leading sliver column like: |   |5.5...
            for needle in ("\"", "5.5", "9.9"):
                line = next((ln for ln in ascii_table.splitlines() if ln.startswith("|") and needle in ln), "")
                self.assertTrue(line.startswith(f"|{needle}"), msg=f"Unexpected sliver column for {needle}:\n{ascii_table}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
