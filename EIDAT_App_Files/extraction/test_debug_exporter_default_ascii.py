import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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


def _table_blocks(combined_text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] | None = None
    for ln in str(combined_text or "").splitlines():
        if ln.startswith("[Table ") and ln.endswith("]"):
            if current:
                blocks.append("\n".join(current))
            current = []
            continue
        if current is None:
            continue
        if not ln.strip():
            if current:
                blocks.append("\n".join(current))
                current = None
            continue
        current.append(ln)
    if current:
        blocks.append("\n".join(current))
    return blocks


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

    def test_combined_text_emits_skip_marker_for_graph_item_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            output_path = debug_exporter.export_combined_text(
                Path("dummy.pdf"),
                [
                    {
                        "page": 1,
                        "tokens": [],
                        "tables": [],
                        "charts": [],
                        "flow": {},
                        "dpi": 900,
                        "ocr_dpi": 450,
                        "skip_reason": "dense_raster_graph_page",
                        "skip_marker": "{ITEM SKIPPED}",
                    }
                ],
                out_dir,
            )

            combined = output_path.read_text(encoding="utf-8")

        self.assertIn("=== Page 1 ===", combined)
        self.assertIn("{ITEM SKIPPED}", combined)
        self.assertNotIn("--- PAGE 1 SKIPPED:", combined)

    def test_summary_report_includes_graph_item_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            output_path = debug_exporter.create_summary_report(
                Path("dummy.pdf"),
                [
                    {
                        "page": 1,
                        "tokens": [],
                        "tables": [],
                        "charts": [],
                        "flow": {},
                        "warnings": ["graph_page_item_skipped"],
                        "skip_reason": "dense_raster_graph_page",
                        "skip_marker": "{ITEM SKIPPED}",
                    }
                ],
                out_dir,
            )

            summary = output_path.read_text(encoding="utf-8")

        self.assertIn("dense_raster_graph_page", summary)
        self.assertIn("graph_page_item_skipped", summary)
        self.assertIn("{ITEM SKIPPED}", summary)

    def test_combined_text_uses_default_ascii_after_vline_split(self) -> None:
        old_env = {
            "EIDAT_TABLE_SPLIT_MODE": os.environ.get("EIDAT_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_MODE": os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE"),
        }
        try:
            os.environ["EIDAT_TABLE_SPLIT_MODE"] = "replica"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_MODE"] = "vline"

            table = {
                "bbox_px": [0.0, 10.0, 200.0, 80.0],
                "cells": [
                    {"row": 0, "col": 0, "text": "A", "bbox_px": [0.0, 10.0, 100.0, 30.0]},
                    {"row": 0, "col": 2, "text": "C", "bbox_px": [100.0, 10.0, 200.0, 30.0]},
                    {"row": 2, "col": 0, "text": "1", "bbox_px": [0.0, 50.0, 100.0, 80.0]},
                    {"row": 2, "col": 2, "text": "3", "bbox_px": [100.0, 50.0, 200.0, 80.0]},
                ],
                "borderless": False,
            }

            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                with mock.patch(
                    "extraction.debug_exporter._split_table_on_vline_mismatch_for_display",
                    return_value=[table],
                ):
                    output_path = debug_exporter.export_combined_text(
                        Path("dummy.pdf"),
                        [
                            {
                                "page": 1,
                                "tokens": [],
                                "tables": [table],
                                "charts": [],
                                "flow": {"body_tokens": [], "table_titles": []},
                                "dpi": 900,
                                "ocr_dpi": 450,
                            }
                        ],
                        out_dir,
                    )

                combined = output_path.read_text(encoding="utf-8")

            rows = _pipe_rows(combined)
            self.assertEqual(len(rows), 3, msg=f"Blank structural row was dropped:\n{combined}")
            self.assertEqual(rows[0], ["A", "", "C"], msg=f"Blank structural column was dropped:\n{combined}")
            self.assertEqual(rows[1], ["", "", ""], msg=f"Blank structural row missing:\n{combined}")
            self.assertEqual(rows[2], ["1", "", "3"], msg=f"Blank structural column was dropped:\n{combined}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_combined_text_rebases_split_parts_to_local_indices(self) -> None:
        old_env = {
            "EIDAT_TABLE_SPLIT_MODE": os.environ.get("EIDAT_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_MODE": os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE"),
        }
        try:
            os.environ["EIDAT_TABLE_SPLIT_MODE"] = "replica"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_MODE"] = "vline"

            split_parts = [
                {
                    "bbox_px": [0.0, 10.0, 100.0, 40.0],
                    "cells": [
                        {"row": 0, "col": 0, "text": "A", "bbox_px": [0.0, 10.0, 50.0, 25.0]},
                        {"row": 0, "col": 1, "text": "B", "bbox_px": [50.0, 10.0, 100.0, 25.0]},
                        {"row": 1, "col": 0, "text": "1", "bbox_px": [0.0, 25.0, 50.0, 40.0]},
                        {"row": 1, "col": 1, "text": "2", "bbox_px": [50.0, 25.0, 100.0, 40.0]},
                    ],
                    "borderless": False,
                },
                {
                    "bbox_px": [0.0, 60.0, 150.0, 105.0],
                    "cells": [
                        {"row": 5, "col": 3, "text": "X", "bbox_px": [0.0, 60.0, 75.0, 75.0]},
                        {"row": 5, "col": 4, "text": "Y", "bbox_px": [75.0, 60.0, 150.0, 75.0]},
                        {"row": 6, "col": 3, "text": "7", "bbox_px": [0.0, 75.0, 75.0, 90.0]},
                        {"row": 6, "col": 4, "text": "8", "bbox_px": [75.0, 75.0, 150.0, 90.0]},
                    ],
                    "borderless": False,
                },
            ]

            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                with mock.patch(
                    "extraction.debug_exporter._split_table_on_vline_mismatch_for_display",
                    return_value=split_parts,
                ):
                    output_path = debug_exporter.export_combined_text(
                        Path("dummy.pdf"),
                        [
                            {
                                "page": 1,
                                "tokens": [],
                                "tables": [split_parts[0]],
                                "charts": [],
                                "flow": {"body_tokens": [], "table_titles": []},
                                "dpi": 900,
                                "ocr_dpi": 450,
                            }
                        ],
                        out_dir,
                    )

                combined = output_path.read_text(encoding="utf-8")

            blocks = _table_blocks(combined)
            self.assertEqual(len(blocks), 2, msg=f"Expected two distinct table blocks:\n{combined}")
            self.assertEqual(_pipe_rows(blocks[0]), [["A", "B"], ["1", "2"]], msg=f"Unexpected first split table:\n{blocks[0]}")
            self.assertEqual(_pipe_rows(blocks[1]), [["X", "Y"], ["7", "8"]], msg=f"Second split table was not rebased locally:\n{blocks[1]}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_combined_text_collapses_multiple_artifact_empty_columns(self) -> None:
        old_env = {
            "EIDAT_TABLE_SPLIT_MODE": os.environ.get("EIDAT_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_MODE": os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE"),
        }
        try:
            os.environ["EIDAT_TABLE_SPLIT_MODE"] = "replica"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_MODE"] = "vline"

            table = {
                "bbox_px": [0.0, 0.0, 250.0, 60.0],
                "cells": [
                    {"row": 0, "col": 0, "text": "Item", "bbox_px": [0.0, 0.0, 80.0, 20.0]},
                    {"row": 0, "col": 2, "text": "Value", "bbox_px": [80.0, 0.0, 160.0, 20.0]},
                    {"row": 0, "col": 4, "text": "Date", "bbox_px": [160.0, 0.0, 250.0, 20.0]},
                    {"row": 1, "col": 0, "text": "Gage", "bbox_px": [0.0, 20.0, 80.0, 40.0]},
                    {"row": 1, "col": 2, "text": "1804461", "bbox_px": [80.0, 20.0, 160.0, 40.0]},
                    {"row": 1, "col": 4, "text": "7/30/25", "bbox_px": [160.0, 20.0, 250.0, 40.0]},
                ],
                "borderless": False,
            }

            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                output_path = debug_exporter.export_combined_text(
                    Path("dummy.pdf"),
                    [
                        {
                            "page": 1,
                            "tokens": [],
                            "tables": [table],
                            "charts": [],
                            "flow": {"body_tokens": [], "table_titles": []},
                            "dpi": 900,
                            "ocr_dpi": 450,
                        }
                    ],
                    out_dir,
                )
                combined = output_path.read_text(encoding="utf-8")

            blocks = _table_blocks(combined)
            self.assertEqual(len(blocks), 1, msg=f"Unexpected combined table count:\n{combined}")
            self.assertEqual(
                _pipe_rows(blocks[0]),
                [["Item", "Value", "Date"], ["Gage", "1804461", "7/30/25"]],
                msg=f"Artifact overlap columns were not collapsed:\n{blocks[0]}",
            )
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_combined_text_splits_leading_single_cell_row(self) -> None:
        old_env = {
            "EIDAT_TABLE_SPLIT_MODE": os.environ.get("EIDAT_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_MODE": os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR": os.environ.get(
                "EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR"
            ),
        }
        try:
            os.environ["EIDAT_TABLE_SPLIT_MODE"] = "replica"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_MODE"] = "vline"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR"] = "1"

            table = {
                "bbox_px": [0.0, 0.0, 180.0, 60.0],
                "cells": [
                    {"row": 0, "col": 0, "text": "Header row", "bbox_px": [0.0, 0.0, 180.0, 20.0]},
                    {"row": 1, "col": 0, "text": "A", "bbox_px": [0.0, 20.0, 90.0, 40.0]},
                    {"row": 1, "col": 1, "text": "B", "bbox_px": [90.0, 20.0, 180.0, 40.0]},
                    {"row": 2, "col": 0, "text": "1", "bbox_px": [0.0, 40.0, 90.0, 60.0]},
                    {"row": 2, "col": 1, "text": "2", "bbox_px": [90.0, 40.0, 180.0, 60.0]},
                ],
                "borderless": False,
            }

            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                output_path = debug_exporter.export_combined_text(
                    Path("dummy.pdf"),
                    [
                        {
                            "page": 1,
                            "tokens": [],
                            "tables": [table],
                            "charts": [],
                            "flow": {"body_tokens": [], "table_titles": []},
                            "dpi": 900,
                            "ocr_dpi": 450,
                        }
                    ],
                    out_dir,
                )
                combined = output_path.read_text(encoding="utf-8")

            blocks = _table_blocks(combined)
            self.assertEqual(len(blocks), 2, msg=f"Leading single-cell row was not split:\n{combined}")
            self.assertEqual(_pipe_rows(blocks[0]), [["Header row"]], msg=f"Unexpected leading single-cell block:\n{blocks[0]}")
            self.assertEqual(_pipe_rows(blocks[1]), [["A", "B"], ["1", "2"]], msg=f"Unexpected trailing table block:\n{blocks[1]}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_combined_text_splits_trailing_single_cell_row(self) -> None:
        old_env = {
            "EIDAT_TABLE_SPLIT_MODE": os.environ.get("EIDAT_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_MODE": os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE"),
            "EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR": os.environ.get(
                "EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR"
            ),
        }
        try:
            os.environ["EIDAT_TABLE_SPLIT_MODE"] = "replica"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_MODE"] = "vline"
            os.environ["EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR"] = "1"

            table = {
                "bbox_px": [0.0, 0.0, 180.0, 60.0],
                "cells": [
                    {"row": 0, "col": 0, "text": "A", "bbox_px": [0.0, 0.0, 90.0, 20.0]},
                    {"row": 0, "col": 1, "text": "B", "bbox_px": [90.0, 0.0, 180.0, 20.0]},
                    {"row": 1, "col": 0, "text": "1", "bbox_px": [0.0, 20.0, 90.0, 40.0]},
                    {"row": 1, "col": 1, "text": "2", "bbox_px": [90.0, 20.0, 180.0, 40.0]},
                    {"row": 2, "col": 0, "text": "Footer row", "bbox_px": [0.0, 40.0, 180.0, 60.0]},
                ],
                "borderless": False,
            }

            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                output_path = debug_exporter.export_combined_text(
                    Path("dummy.pdf"),
                    [
                        {
                            "page": 1,
                            "tokens": [],
                            "tables": [table],
                            "charts": [],
                            "flow": {"body_tokens": [], "table_titles": []},
                            "dpi": 900,
                            "ocr_dpi": 450,
                        }
                    ],
                    out_dir,
                )
                combined = output_path.read_text(encoding="utf-8")

            blocks = _table_blocks(combined)
            self.assertEqual(len(blocks), 2, msg=f"Trailing single-cell row was not split:\n{combined}")
            self.assertEqual(_pipe_rows(blocks[0]), [["A", "B"], ["1", "2"]], msg=f"Unexpected leading table block:\n{blocks[0]}")
            self.assertEqual(_pipe_rows(blocks[1]), [["Footer row"]], msg=f"Unexpected trailing single-cell block:\n{blocks[1]}")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
