import sys
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class TestGraphPageGuard(unittest.TestCase):
    def test_dense_regular_grid_skips_table_work(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy not available")

        from extraction.graph_page_guard import inspect_page_for_graph_grid

        img = np.full((1000, 800), 255, dtype=np.uint8)
        x0, y0, x1, y1 = 120, 120, 620, 620
        for x in range(x0, x1 + 1, 40):
            cv2.line(img, (x, y0), (x, y1), 0, 1)
        for y in range(y0, y1 + 1, 40):
            cv2.line(img, (x0, y), (x1, y), 0, 1)

        result = inspect_page_for_graph_grid(img)
        self.assertTrue(result.get("skip_table_work"))
        self.assertEqual(result.get("reason"), "dense_regular_orthogonal_grid")

    def test_normal_table_does_not_skip_table_work(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy not available")

        from extraction.graph_page_guard import inspect_page_for_graph_grid

        img = np.full((1000, 800), 255, dtype=np.uint8)
        x0, y0, x1, y1 = 100, 160, 620, 520
        for x in range(x0, x1 + 1, 104):
            cv2.line(img, (x, y0), (x, y1), 0, 2)
        for y in range(y0, y1 + 1, 60):
            cv2.line(img, (x0, y), (x1, y), 0, 2)

        result = inspect_page_for_graph_grid(img)
        self.assertFalse(result.get("skip_table_work"))


if __name__ == "__main__":
    unittest.main()
