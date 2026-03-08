import sys
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class TestGraphPageGuard(unittest.TestCase):
    def test_dense_regular_grid_region_is_detected(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy not available")

        from extraction.graph_page_guard import find_chart_like_regions

        img = np.full((1000, 800), 255, dtype=np.uint8)
        x0, y0, x1, y1 = 420, 120, 720, 420
        for x in range(x0, x1 + 1, 40):
            cv2.line(img, (x, y0), (x, y1), 0, 1)
        for y in range(y0, y1 + 1, 40):
            cv2.line(img, (x0, y), (x1, y), 0, 1)

        result = find_chart_like_regions(img)
        regions = list(result.get("regions") or [])
        self.assertTrue(regions)
        self.assertGreaterEqual(int(regions[0].get("h_lines") or 0), 7)
        self.assertGreaterEqual(int(regions[0].get("v_lines") or 0), 7)

    def test_smaller_embedded_chart_region_is_detected(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy not available")

        from extraction.graph_page_guard import find_chart_like_regions

        img = np.full((1200, 1600), 255, dtype=np.uint8)
        x0, y0, x1, y1 = 980, 120, 1220, 360
        for x in range(x0, x1 + 1, 30):
            cv2.line(img, (x, y0), (x, y1), 0, 1)
        for y in range(y0, y1 + 1, 30):
            cv2.line(img, (x0, y), (x1, y), 0, 1)

        result = find_chart_like_regions(img)
        regions = list(result.get("regions") or [])
        self.assertTrue(regions)
        self.assertGreaterEqual(float(regions[0].get("grid_area_ratio") or 0.0), 0.02)

    def test_normal_table_does_not_emit_chart_regions(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy not available")

        from extraction.graph_page_guard import find_chart_like_regions

        img = np.full((1000, 800), 255, dtype=np.uint8)
        x0, y0, x1, y1 = 100, 160, 620, 520
        for x in range(x0, x1 + 1, 104):
            cv2.line(img, (x, y0), (x, y1), 0, 2)
        for y in range(y0, y1 + 1, 60):
            cv2.line(img, (x0, y), (x1, y), 0, 2)

        result = find_chart_like_regions(img)
        self.assertFalse(result.get("regions"))


if __name__ == "__main__":
    unittest.main()
