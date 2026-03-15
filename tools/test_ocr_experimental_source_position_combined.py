import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OCR_EXPERIMENTAL_ROOT = REPO_ROOT / "OCR Experimental"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOCRExperimentalSourcePositionCombined(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(OCR_EXPERIMENTAL_ROOT) not in sys.path:
            sys.path.insert(0, str(OCR_EXPERIMENTAL_ROOT))
        cls.text_outputs = _load_module("ocr_experimental_text_outputs", OCR_EXPERIMENTAL_ROOT / "text_outputs.py")
        cls.main_module = _load_module("ocr_experimental_main", OCR_EXPERIMENTAL_ROOT / "main.py")

    def test_build_combined_page_renders_source_positions_without_markers(self) -> None:
        tokens = [
            {"text": "HELLO", "x0": 10, "y0": 10, "x1": 60, "y1": 20},
            {"text": "END", "x0": 40, "y0": 30, "x1": 70, "y1": 40},
        ]

        combined = self.text_outputs.build_combined_page(2, tokens, 100, 80)

        self.assertTrue(combined.startswith("=== Page 2 ===\n\n"))
        self.assertNotIn("[TABLE]", combined)
        self.assertNotIn("[STRING]", combined)
        body_lines = combined.rstrip("\n").split("\n")[2:]
        self.assertGreaterEqual(len(body_lines), 4, msg=combined)
        self.assertEqual(body_lines[1].index("HELLO"), 1, msg=combined)
        self.assertEqual(body_lines[3].index("END"), 4, msg=combined)

    def test_main_writes_source_position_combined_only(self) -> None:
        main = self.main_module

        class DummyGlyphBank:
            def __init__(self, _font_bank: Path) -> None:
                pass

            def render_text_to_box(self, _text: str, width: int, height: int) -> dict[str, np.ndarray]:
                return {"mask": np.ones((max(1, height), max(1, width)), dtype=np.uint8)}

        source_tokens = [
            {"text": "Alpha", "x0": 10, "y0": 10, "x1": 60, "y1": 20, "width": 50, "height": 10, "conf": 0.9},
            {"text": "Bravo", "x0": 20, "y0": 30, "x1": 70, "y1": 40, "width": 50, "height": 10, "conf": 0.9},
        ]
        ocr_calls = {"count": 0}

        def fake_ocr_page_tokens(_image, *, tesseract_cmd: str, psm: int):
            self.assertEqual(tesseract_cmd, "fake-tesseract")
            self.assertEqual(psm, 11)
            ocr_calls["count"] += 1
            if ocr_calls["count"] > 1:
                raise AssertionError("combined.txt should not require a second OCR pass")
            return list(source_tokens)

        def fake_reinforce_token(*, token_region, token, glyph_bank, tesseract_cmd, max_iterations):
            self.assertIsInstance(glyph_bank, DummyGlyphBank)
            self.assertEqual(tesseract_cmd, "fake-tesseract")
            self.assertEqual(max_iterations, main.MAX_REINFORCE_ITERATIONS)
            return {
                "summary": {
                    "decision": "accepted",
                    "final_text": "ALPHA" if token["text"] == "Alpha" else "",
                }
            }

        def fake_prepare_token_region(_image, token, padding):
            self.assertEqual(padding, main.TOKEN_PADDING)
            return {"bbox": [token["x0"], token["y0"], token["x1"], token["y1"]]}

        page = types.SimpleNamespace(
            page_number=1,
            image=np.full((60, 120), 255, dtype=np.uint8),
            width=120,
            height=60,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "dummy.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            args = types.SimpleNamespace(
                pdf=str(pdf_path),
                output_dir=str(tmp_dir / "output"),
                dpi=300,
                font_bank=str(tmp_dir / "fonts"),
                tesseract_cmd="",
                page_psm=11,
            )

            with mock.patch.object(main, "parse_args", return_value=args), \
                mock.patch.object(main, "resolve_tesseract_cmd", return_value="fake-tesseract"), \
                mock.patch.object(main, "GlyphBank", DummyGlyphBank), \
                mock.patch.object(main, "render_pdf", return_value=[page]), \
                mock.patch.object(main, "ocr_page_tokens", side_effect=fake_ocr_page_tokens), \
                mock.patch.object(main, "detect_page_structure", return_value={"horizontal": [], "vertical": [], "tables": [], "boxes": []}), \
                mock.patch.object(main, "prepare_token_region", side_effect=fake_prepare_token_region), \
                mock.patch.object(main, "reinforce_token", side_effect=fake_reinforce_token), \
                mock.patch.object(main, "write_image", return_value=None):
                exit_code = main.main()

            self.assertEqual(exit_code, 0)
            run_dirs = sorted((tmp_dir / "output").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            combined_path = run_dir / "combined.txt"
            replica_path = run_dir / "replica.txt"
            self.assertTrue(combined_path.exists())
            self.assertFalse(replica_path.exists())
            combined = combined_path.read_text(encoding="utf-8")
            self.assertIn("=== Page 1 ===", combined)
            self.assertIn("ALPHA", combined)
            self.assertIn("Bravo", combined)
            self.assertNotIn("[TABLE]", combined)
            self.assertEqual(ocr_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
