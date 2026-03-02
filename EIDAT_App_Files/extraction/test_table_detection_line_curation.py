import os
import sys
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction import table_detection  # noqa: E402


class TestTableDetectionLineCuration(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_weak_density_cap_skips_weak(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("cv2/numpy not available")

        h, w = 60, 200
        strong = np.zeros((h, w), dtype=np.uint8)
        strong[10, 10:190] = 255

        bin_otsu = np.zeros_like(strong)
        bin_adapt = strong.copy()
        # Flooded weak evidence (density=1.0)
        bin_high = np.full_like(strong, 255)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

        prepared = {
            "h": h,
            "w": w,
            "bin_otsu": bin_otsu,
            "bin_high": bin_high,
            "bin_adapt": bin_adapt,
            "h_kernel": h_kernel,
            "v_kernel": v_kernel,
        }

        os.environ["EIDAT_TABLE_DETECT_LINE_CURATION"] = "1"
        os.environ["EIDAT_TABLE_DETECT_PRE_CLOSE"] = "0"
        os.environ["EIDAT_TABLE_DETECT_WEAK_DENSITY_CAP"] = "0.2"
        os.environ["EIDAT_TABLE_DETECT_POST_CLOSE_GAP_PX"] = "0"
        os.environ["EIDAT_TABLE_DETECT_DOUBLE_LINE_COLLAPSE"] = "0"

        horiz, vert = table_detection._extract_open_line_masks(prepared, verbose=False)  # type: ignore[attr-defined]

        expected_h = cv2.morphologyEx(strong, cv2.MORPH_OPEN, h_kernel)
        self.assertTrue(np.array_equal(horiz, expected_h), msg="Weak flood should be skipped entirely")
        self.assertEqual(int(np.count_nonzero(vert)), 0)

    def test_overlap_keep_rule_keeps_thick_weak_component(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("cv2/numpy not available")

        h, w = 60, 200
        bin_otsu = np.zeros((h, w), dtype=np.uint8)
        bin_adapt = np.zeros((h, w), dtype=np.uint8)
        bin_high = np.zeros((h, w), dtype=np.uint8)

        # Strong evidence: a short 1px line (one side "attached").
        bin_adapt[25, 10:35] = 255
        # Weak evidence: a thicker faint band that should NOT pass the "thin" gate,
        # but should be kept because it overlaps the strong line.
        bin_high[20:30, 0:130] = 255

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

        prepared = {
            "h": h,
            "w": w,
            "bin_otsu": bin_otsu,
            "bin_high": bin_high,
            "bin_adapt": bin_adapt,
            "h_kernel": h_kernel,
            "v_kernel": v_kernel,
        }

        os.environ["EIDAT_TABLE_DETECT_LINE_CURATION"] = "1"
        os.environ["EIDAT_TABLE_DETECT_PRE_CLOSE"] = "0"
        os.environ["EIDAT_TABLE_DETECT_WEAK_DENSITY_CAP"] = "0.35"
        os.environ["EIDAT_TABLE_DETECT_WEAK_KEEP_OVERLAP_PX"] = "10"
        # Make the weak component fail the "line-like" branch unless overlap saves it.
        os.environ["EIDAT_TABLE_DETECT_WEAK_MAX_THICK_PX"] = "5"
        os.environ["EIDAT_TABLE_DETECT_WEAK_MIN_LEN_RATIO"] = "0.9"
        os.environ["EIDAT_TABLE_DETECT_POST_CLOSE_GAP_PX"] = "0"
        os.environ["EIDAT_TABLE_DETECT_DOUBLE_LINE_COLLAPSE"] = "0"

        horiz, _vert = table_detection._extract_open_line_masks(prepared, verbose=False)  # type: ignore[attr-defined]

        # Pixel far from the overlap region should still be present if the whole weak component was kept.
        self.assertEqual(int(horiz[22, 100]), 255)

    def test_double_line_collapse_merges_parallel_lines(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            self.skipTest("cv2/numpy not available")

        h, w = 40, 120
        horiz = np.zeros((h, w), dtype=np.uint8)
        vert = np.zeros((h, w), dtype=np.uint8)
        horiz[10, 10:110] = 255
        horiz[12, 10:110] = 255  # 2px gap between lines

        os.environ["EIDAT_TABLE_DETECT_POST_CLOSE_GAP_PX"] = "0"
        os.environ["EIDAT_TABLE_DETECT_DOUBLE_LINE_COLLAPSE"] = "1"
        os.environ["EIDAT_TABLE_DETECT_DOUBLE_LINE_GAP_PX"] = "3"

        before = cv2.connectedComponents((horiz > 0).astype(np.uint8), connectivity=8)[0] - 1
        h2, _v2 = table_detection._apply_line_postproc(horiz, vert, curation_enabled=True)  # type: ignore[attr-defined]
        after = cv2.connectedComponents((h2 > 0).astype(np.uint8), connectivity=8)[0] - 1

        self.assertEqual(int(before), 2)
        self.assertEqual(int(after), 1)


if __name__ == "__main__":
    unittest.main()

