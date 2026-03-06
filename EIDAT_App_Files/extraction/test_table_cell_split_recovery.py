import copy
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction import table_cell_split_recovery  # noqa: E402


def _tok(text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "text": str(text),
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "cx": (float(x0) + float(x1)) / 2.0,
        "cy": (float(y0) + float(y1)) / 2.0,
        "conf": 0.99,
    }


def _cell(bbox: list[float], *, tokens: list[dict] | None = None, text: str = "") -> dict:
    toks = list(tokens or [])
    return {
        "bbox_px": [float(v) for v in bbox],
        "tokens": toks,
        "text": str(text),
        "token_count": int(len([t for t in toks if str(t.get("text", "")).strip()])),
    }


@contextmanager
def _patch_env(values: dict[str, str]):
    prev: dict[str, str | None] = {}
    for k, v in values.items():
        prev[k] = os.environ.get(k)
        os.environ[str(k)] = str(v)
    try:
        yield
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


class TestTableCellSplitRecovery(unittest.TestCase):
    def test_multirow_gate_blocks_one_off_boundary(self) -> None:
        # 2 columns, 5 rows. Target merged row has only 1 supporting neighbor row,
        # so MIN_RUN=2 should block splitting when evidence is enabled.
        table = {
            "bbox_px": [0, 0, 200, 250],
            "cells": [
                _cell([0, 0, 200, 50]),  # row 0 merged (no boundary)
                _cell([0, 50, 100, 100]),  # row 1 split (boundary present)
                _cell([100, 50, 200, 100]),
                _cell(  # row 2 merged target
                    [0, 100, 200, 150],
                    tokens=[
                        _tok("L2", 10, 112, 35, 132),
                        _tok("R2", 160, 112, 190, 132),
                    ],
                    text="L2 R2",
                ),
                _cell([0, 150, 200, 200]),  # row 3 merged (no boundary)
                _cell([0, 200, 200, 250]),  # row 4 merged (no boundary)
            ],
        }

        cfg_on = {
            "EIDAT_TABLE_CELL_SPLIT_RECOVERY": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_MAX_LOOK": "3",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_MIN_RUN": "2",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_REQUIRE_ADJACENT": "1",
            "EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TEXT_CHARS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS": "1",
            "EIDAT_TABLE_CELL_SPLIT_DEBUG": "0",
        }
        with _patch_env(cfg_on):
            t0 = copy.deepcopy(table)
            stats = table_cell_split_recovery.split_merged_cells_post_projection(
                [t0], img_gray_det=None, img_w=200, img_h=250, debug_dir=None
            )
            self.assertEqual(int(stats.get("cells_split") or 0), 0)
            self.assertTrue(any(c.get("bbox_px") == [0.0, 100.0, 200.0, 150.0] for c in (t0.get("cells") or [])))
            self.assertFalse(any(c.get("bbox_px") == [0.0, 100.0, 100.0, 150.0] for c in (t0.get("cells") or [])))

        cfg_off = dict(cfg_on)
        cfg_off["EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW"] = "0"
        with _patch_env(cfg_off):
            t1 = copy.deepcopy(table)
            stats = table_cell_split_recovery.split_merged_cells_post_projection(
                [t1], img_gray_det=None, img_w=200, img_h=250, debug_dir=None
            )
            self.assertEqual(int(stats.get("cells_split") or 0), 1)
            self.assertFalse(any(c.get("bbox_px") == [0.0, 100.0, 200.0, 150.0] for c in (t1.get("cells") or [])))
            self.assertTrue(any(c.get("bbox_px") == [0.0, 100.0, 100.0, 150.0] for c in (t1.get("cells") or [])))
            self.assertTrue(any(c.get("bbox_px") == [100.0, 100.0, 200.0, 150.0] for c in (t1.get("cells") or [])))

    def test_whitespace_corridor_gate_rejects_ink_crossing(self) -> None:
        if not bool(getattr(table_cell_split_recovery, "HAVE_CV2", False)):
            self.skipTest("cv2/numpy unavailable")

        import numpy as np  # type: ignore

        table = {
            "bbox_px": [0, 0, 200, 250],
            "cells": [
                _cell([0, 0, 100, 50]),
                _cell([100, 0, 200, 50]),
                _cell([0, 50, 100, 100]),
                _cell([100, 50, 200, 100]),
                _cell(
                    [0, 100, 200, 150],
                    tokens=[
                        _tok("L", 10, 112, 35, 132),
                        _tok("R", 160, 112, 190, 132),
                    ],
                    text="L R",
                ),
                _cell([0, 150, 200, 200]),
                _cell([0, 200, 200, 250]),
            ],
        }

        cfg = {
            "EIDAT_TABLE_CELL_SPLIT_RECOVERY": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_MIN_RUN": "2",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_REQUIRE_ADJACENT": "1",
            "EIDAT_TABLE_CELL_SPLIT_WS_STRIP_HALF_WIDTH_PX": "3",
            "EIDAT_TABLE_CELL_SPLIT_WS_TB_PAD_PX": "6",
            "EIDAT_TABLE_CELL_SPLIT_WS_ROW_INK_PX_MAX": "1",
            "EIDAT_TABLE_CELL_SPLIT_WS_MIN_CLEAR_FRAC": "0.85",
            "EIDAT_TABLE_CELL_SPLIT_WS_REQUIRED": "1",
            "EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS": "1",
            "EIDAT_TABLE_CELL_SPLIT_DEBUG": "0",
        }

        img_block = np.full((250, 200), 255, dtype=np.uint8)
        # Draw a thick vertical "ink" stroke across the boundary inside the merged cell.
        x = 100
        y0 = 100 + 6
        y1 = 150 - 6
        img_block[y0:y1, (x - 1):(x + 2)] = 0

        with _patch_env(cfg):
            t0 = copy.deepcopy(table)
            stats0 = table_cell_split_recovery.split_merged_cells_post_projection(
                [t0], img_gray_det=img_block, img_w=200, img_h=250, debug_dir=None
            )
            self.assertEqual(int(stats0.get("cells_split") or 0), 0)

            img_clear = np.full((250, 200), 255, dtype=np.uint8)
            t1 = copy.deepcopy(table)
            stats1 = table_cell_split_recovery.split_merged_cells_post_projection(
                [t1], img_gray_det=img_clear, img_w=200, img_h=250, debug_dir=None
            )
            self.assertEqual(int(stats1.get("cells_split") or 0), 1)

    def test_end_to_end_split_success_three_cols(self) -> None:
        if not bool(getattr(table_cell_split_recovery, "HAVE_CV2", False)):
            self.skipTest("cv2/numpy unavailable")

        import numpy as np  # type: ignore

        table = {
            "bbox_px": [0, 0, 300, 250],
            "cells": [
                # row 0 (support)
                _cell([0, 0, 100, 50]),
                _cell([100, 0, 200, 50]),
                _cell([200, 0, 300, 50]),
                # row 1 (support)
                _cell([0, 50, 100, 100]),
                _cell([100, 50, 200, 100]),
                _cell([200, 50, 300, 100]),
                # row 2 merged target (3 cols)
                _cell(
                    [0, 100, 300, 150],
                    tokens=[
                        _tok("A", 10, 112, 35, 132),
                        _tok("B", 130, 112, 155, 132),
                        _tok("C", 240, 112, 265, 132),
                    ],
                    text="A B C",
                ),
                # rows 3-4 merged (no support needed)
                _cell([0, 150, 300, 200]),
                _cell([0, 200, 300, 250]),
            ],
        }

        cfg = {
            "EIDAT_TABLE_CELL_SPLIT_RECOVERY": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_MIN_RUN": "2",
            "EIDAT_TABLE_CELL_SPLIT_MULTIROW_REQUIRE_ADJACENT": "1",
            "EIDAT_TABLE_CELL_SPLIT_WS_REQUIRED": "1",
            "EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS": "1",
            "EIDAT_TABLE_CELL_SPLIT_DEBUG": "0",
        }

        img = np.full((250, 300), 255, dtype=np.uint8)
        with _patch_env(cfg):
            t0 = copy.deepcopy(table)
            stats = table_cell_split_recovery.split_merged_cells_post_projection(
                [t0], img_gray_det=img, img_w=300, img_h=250, debug_dir=None
            )
            self.assertEqual(int(stats.get("cells_split") or 0), 1)

            row2_cells = [
                c
                for c in (t0.get("cells") or [])
                if c.get("bbox_px") == [0.0, 100.0, 100.0, 150.0]
                or c.get("bbox_px") == [100.0, 100.0, 200.0, 150.0]
                or c.get("bbox_px") == [200.0, 100.0, 300.0, 150.0]
            ]
            self.assertEqual(len(row2_cells), 3)

            cols = sorted(int(c.get("col", -1)) for c in row2_cells)
            self.assertEqual(cols, [0, 1, 2])
            texts = sorted(str(c.get("text", "")).strip() for c in row2_cells)
            self.assertEqual(texts, ["A", "B", "C"])

    def test_allows_leading_blank_structural_subcell(self) -> None:
        # 2 columns: merged target spans both but has content only in the RIGHT segment.
        # The LEFT blank is structural and should be allowed (no blank cells to the right of content).
        table = {
            "bbox_px": [0, 0, 200, 150],
            "cells": [
                # row 0 (support)
                _cell([0, 0, 100, 50]),
                _cell([100, 0, 200, 50]),
                # row 1 (support)
                _cell([0, 50, 100, 100]),
                _cell([100, 50, 200, 100]),
                # row 2 merged target (2 cols; left is intentionally blank)
                _cell(
                    [0, 100, 200, 150],
                    tokens=[
                        _tok("X", 160, 112, 190, 132),
                    ],
                    text="X",
                ),
            ],
        }

        cfg = {
            "EIDAT_TABLE_CELL_SPLIT_RECOVERY": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS": "1",
            "EIDAT_TABLE_CELL_SPLIT_DEBUG": "0",
        }

        with _patch_env(cfg):
            t0 = copy.deepcopy(table)
            stats = table_cell_split_recovery.split_merged_cells_post_projection(
                [t0], img_gray_det=None, img_w=200, img_h=150, debug_dir=None
            )
            self.assertEqual(int(stats.get("cells_split") or 0), 1)
            self.assertFalse(any(c.get("bbox_px") == [0.0, 100.0, 200.0, 150.0] for c in (t0.get("cells") or [])))
            self.assertTrue(any(c.get("bbox_px") == [0.0, 100.0, 100.0, 150.0] for c in (t0.get("cells") or [])))
            self.assertTrue(any(c.get("bbox_px") == [100.0, 100.0, 200.0, 150.0] for c in (t0.get("cells") or [])))

    def test_rejects_split_that_creates_trailing_blank_subcell(self) -> None:
        # 2 columns: merged target spans both but has content only in the LEFT segment.
        # Creating a blank cell to the RIGHT of content is disallowed.
        table = {
            "bbox_px": [0, 0, 200, 150],
            "cells": [
                # row 0 (support)
                _cell([0, 0, 100, 50]),
                _cell([100, 0, 200, 50]),
                # row 1 (support)
                _cell([0, 50, 100, 100]),
                _cell([100, 50, 200, 100]),
                # row 2 merged target (2 cols; right is blank)
                _cell(
                    [0, 100, 200, 150],
                    tokens=[
                        _tok("X", 10, 112, 35, 132),
                    ],
                    text="X",
                ),
            ],
        }

        cfg = {
            "EIDAT_TABLE_CELL_SPLIT_RECOVERY": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS": "1",
            "EIDAT_TABLE_CELL_SPLIT_DEBUG": "0",
        }

        with _patch_env(cfg):
            t0 = copy.deepcopy(table)
            stats = table_cell_split_recovery.split_merged_cells_post_projection(
                [t0], img_gray_det=None, img_w=200, img_h=150, debug_dir=None
            )
            self.assertEqual(int(stats.get("cells_split") or 0), 0)
            self.assertTrue(any(c.get("bbox_px") == [0.0, 100.0, 200.0, 150.0] for c in (t0.get("cells") or [])))
            self.assertFalse(any(c.get("bbox_px") == [100.0, 100.0, 200.0, 150.0] for c in (t0.get("cells") or [])))

    def test_rejects_split_with_internal_blank_subcell(self) -> None:
        # 3 columns: merged target spans all 3 but only has tokens in 2 segments.
        # Reject to avoid introducing a blank middle subcell (blank to the right of content).
        table = {
            "bbox_px": [0, 0, 300, 250],
            "cells": [
                # row 0 (support)
                _cell([0, 0, 100, 50]),
                _cell([100, 0, 200, 50]),
                _cell([200, 0, 300, 50]),
                # row 1 (support)
                _cell([0, 50, 100, 100]),
                _cell([100, 50, 200, 100]),
                _cell([200, 50, 300, 100]),
                # row 2 merged target (3 cols; missing middle token)
                _cell(
                    [0, 100, 300, 150],
                    tokens=[
                        _tok("A", 10, 112, 35, 132),
                        _tok("C", 240, 112, 265, 132),
                    ],
                    text="A C",
                ),
            ],
        }

        cfg = {
            "EIDAT_TABLE_CELL_SPLIT_RECOVERY": "1",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP": "0",
            "EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS": "2",
            "EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS": "1",
            "EIDAT_TABLE_CELL_SPLIT_DEBUG": "0",
        }

        with _patch_env(cfg):
            t0 = copy.deepcopy(table)
            stats = table_cell_split_recovery.split_merged_cells_post_projection(
                [t0], img_gray_det=None, img_w=300, img_h=250, debug_dir=None
            )
            self.assertEqual(int(stats.get("cells_split") or 0), 0)
            self.assertTrue(any(c.get("bbox_px") == [0.0, 100.0, 300.0, 150.0] for c in (t0.get("cells") or [])))
            self.assertFalse(any(c.get("bbox_px") == [100.0, 100.0, 200.0, 150.0] for c in (t0.get("cells") or [])))


if __name__ == "__main__":
    unittest.main()
