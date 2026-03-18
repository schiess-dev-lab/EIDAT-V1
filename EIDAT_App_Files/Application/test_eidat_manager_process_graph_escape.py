import os
import sys
import tempfile
import unittest
import json
import importlib
import types
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


APP_DIR = Path(__file__).resolve().parent
APP_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class _FakeQueue:
    def __init__(self) -> None:
        self._items = []

    def put(self, item) -> None:
        self._items.append(item)

    def get_nowait(self):
        if not self._items:
            raise RuntimeError("empty")
        return self._items.pop(0)


class _FakeProcess:
    def __init__(self, plan, target, args, daemon) -> None:
        self._plan = dict(plan)
        self._target = target
        self._args = args
        self.daemon = daemon
        self._alive = False

    def start(self) -> None:
        if self._plan.get("alive"):
            self._alive = True
            return
        self._alive = False
        if "payload" in self._plan:
            q = self._args[-1]
            q.put(self._plan["payload"])
            return
        if self._plan.get("call_target"):
            self._target(*self._args)

    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False


class _FakeContext:
    def __init__(self, plans) -> None:
        self._plans = iter(plans)

    def Queue(self, maxsize=1):
        return _FakeQueue()

    def Process(self, target, args, daemon=True):
        return _FakeProcess(next(self._plans), target, args, daemon)


class TestEidatManagerProcessGraphEscape(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_import_uses_pointer_token_fallback_when_embed_module_is_stale(self) -> None:
        saved_proc = sys.modules.pop("eidat_manager_process", None)
        saved_embed = sys.modules.get("eidat_manager_embed")
        fake_embed = types.ModuleType("eidat_manager_embed")
        fake_embed.embed_pointer_token = lambda *_args, **_kwargs: True
        fake_embed.has_pointer_token = lambda *_args, **_kwargs: False
        sys.modules["eidat_manager_embed"] = fake_embed
        try:
            proc = importlib.import_module("eidat_manager_process")
            token = proc.build_pointer_token({"rel_path": "Programs/test.pdf"})
            self.assertTrue(token.startswith("EIDAT_PTR:"))
        finally:
            sys.modules.pop("eidat_manager_process", None)
            if saved_proc is not None:
                sys.modules["eidat_manager_process"] = saved_proc
            if saved_embed is not None:
                sys.modules["eidat_manager_embed"] = saved_embed
            else:
                sys.modules.pop("eidat_manager_embed", None)

    def test_import_uses_noop_pointer_helpers_when_embed_module_has_no_symbols(self) -> None:
        saved_proc = sys.modules.pop("eidat_manager_process", None)
        saved_embed = sys.modules.get("eidat_manager_embed")
        sys.modules["eidat_manager_embed"] = types.ModuleType("eidat_manager_embed")
        try:
            proc = importlib.import_module("eidat_manager_process")
            token = proc.build_pointer_token({"rel_path": "Programs/test.pdf"})
            self.assertTrue(token.startswith("EIDAT_PTR:"))
            self.assertFalse(proc.has_pointer_token(Path("Programs/test.pdf")))
            self.assertFalse(proc.embed_pointer_token(Path("Programs/test.pdf"), token))
        finally:
            sys.modules.pop("eidat_manager_process", None)
            if saved_proc is not None:
                sys.modules["eidat_manager_process"] = saved_proc
            if saved_embed is not None:
                sys.modules["eidat_manager_embed"] = saved_embed
            else:
                sys.modules.pop("eidat_manager_embed", None)

    def _write_page_png(self, path: Path) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np
        except Exception:
            self.skipTest("cv2/numpy not available")
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.full((60, 60), 255, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def _base_settings(self) -> dict:
        return {
            "project_root": str(APP_ROOT),
            "render_dpi": 450,
            "ocr_dpi": 450,
            "table_ocr_dpi": 450,
            "detection_dpi": 900,
            "table_variants_lang": "eng",
            "table_grid_enable_prepass": False,
            "tg_draw_overlay_lines": False,
            "tg_draw_hlines": False,
            "tg_draw_seps_in_tables": False,
            "tg_draw_separators": False,
            "tg_border_thickness": 4,
            "enable_borderless": False,
            "page_timeout_sec": 0,
        }

    def test_chart_regions_are_masked_for_table_work_but_keep_flow(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        import eidat_manager_process as proc

        with tempfile.TemporaryDirectory() as tmp:
            page_dir = Path(tmp) / "page_1"
            self._write_page_png(page_dir / "page_1.png")
            settings = self._base_settings()
            tokens = [{"text": "Voltage", "x0": 1.0, "y0": 1.0, "x1": 10.0, "y1": 8.0}]
            flow = {"body_tokens": tokens, "table_titles": []}
            masked_regions = [{"bbox": [10.0, 10.0, 30.0, 30.0], "h_lines": 8, "v_lines": 8}]
            called = {"input_path": None}

            def _run_for_input(input_path, **kwargs):
                called["input_path"] = Path(input_path)
                return {
                    "tables": [],
                    "warnings": [],
                    "detection_dpi": 900,
                    "detection_size": {"w": 60, "h": 60},
                    "variants": [],
                    "variant_rows_all": [],
                }

            with (
                mock.patch(
                    "extraction.graph_page_guard.find_chart_like_regions",
                    return_value={"regions": masked_regions, "stats": {"region_count": 1}},
                ),
                mock.patch("extraction.graph_page_guard.mask_regions", return_value=np.full((60, 60), 255, dtype=np.uint8)),
                mock.patch("debug_method.run_table_variants._run_for_input", side_effect=_run_for_input),
                mock.patch("debug_method.table_grid_debug.run_for_image", side_effect=AssertionError("table_grid prepass called")),
                mock.patch("extraction.ocr_engine.get_tesseract_lang", return_value="eng"),
                mock.patch("extraction.ocr_engine.ocr_page", return_value=(tokens, 60, 60, None)),
                mock.patch("extraction.ocr_engine.render_pdf_page", return_value=(np.zeros((60, 60), dtype=np.uint8), 60, 60)),
                mock.patch("extraction.ocr_engine.reocr_low_confidence_tokens", side_effect=lambda *args, **kwargs: tokens),
                mock.patch("extraction.token_projector.scale_tokens_to_dpi", side_effect=lambda toks, *_args, **_kwargs: toks),
                mock.patch("extraction.page_analyzer.extract_flow_text", return_value=flow),
                mock.patch("extraction.chart_detection.detect_charts", side_effect=AssertionError("chart detection called")),
                mock.patch("extraction.debug_exporter.export_page_debug"),
            ):
                page_data = proc._process_debug_method_page(Path("dummy.pdf"), 1, page_dir, settings)

        self.assertEqual(page_data.get("tables"), [])
        self.assertEqual(page_data.get("flow"), flow)
        self.assertIn("graph_like_chart_region_masked", page_data.get("warnings", []))
        self.assertEqual(page_data.get("masked_chart_regions"), masked_regions)
        self.assertEqual((called["input_path"] or Path("")).name, "page_1_table_masked.png")
        self.assertFalse(bool(page_data.get("timeout")))
        self.assertIsNone(page_data.get("error"))

    def test_graph_page_skip_bypasses_tables_ocr_and_charts(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        import eidat_manager_process as proc

        with tempfile.TemporaryDirectory() as tmp:
            page_dir = Path(tmp) / "page_1"
            self._write_page_png(page_dir / "page_1.png")
            settings = self._base_settings()
            settings["graph_page_bypass"] = True

            with (
                mock.patch(
                    "extraction.graph_page_guard.inspect_page_for_graph_item_skip",
                    return_value={
                        "skip_page": True,
                        "reason": "dense_raster_graph_page",
                        "stats": {"dominant_raster_gate": True, "microgrid_gate": True, "no_useful_text_gate": True},
                    },
                ),
                mock.patch("extraction.graph_page_guard.find_chart_like_regions", side_effect=AssertionError("region guard called")),
                mock.patch("debug_method.run_table_variants._run_for_input", side_effect=AssertionError("table variants called")),
                mock.patch("extraction.ocr_engine.get_tesseract_lang", return_value="eng"),
                mock.patch("extraction.ocr_engine.ocr_page", side_effect=AssertionError("page OCR called")),
                mock.patch("extraction.chart_detection.detect_charts", side_effect=AssertionError("chart detection called")),
                mock.patch("extraction.debug_exporter.export_page_debug"),
            ):
                page_data = proc._process_debug_method_page(Path("dummy.pdf"), 1, page_dir, settings)

        self.assertEqual(page_data.get("tokens"), [])
        self.assertEqual(page_data.get("tables"), [])
        self.assertEqual(page_data.get("charts"), [])
        self.assertEqual(page_data.get("flow"), {})
        self.assertEqual(page_data.get("skip_reason"), "dense_raster_graph_page")
        self.assertEqual(page_data.get("skip_marker"), "{ITEM SKIPPED}")
        self.assertIn("graph_page_item_skipped", page_data.get("warnings", []))
        self.assertTrue(page_data.get("graph_page_skip_stats"))

    def test_bordered_table_timeout_preserves_ocr_flow(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        import eidat_manager_process as proc

        with tempfile.TemporaryDirectory() as tmp:
            page_dir = Path(tmp) / "page_1"
            self._write_page_png(page_dir / "page_1.png")
            settings = self._base_settings()
            tokens = [{"text": "Current", "x0": 1.0, "y0": 1.0, "x1": 10.0, "y1": 8.0}]
            flow = {"body_tokens": tokens, "table_titles": []}

            with (
                mock.patch("extraction.graph_page_guard.find_chart_like_regions", return_value={"regions": [], "stats": {}}),
                mock.patch("debug_method.run_table_variants._run_for_input", return_value={
                    "tables": [],
                    "warnings": ["bordered_table_detection_timeout"],
                    "detection_dpi": 900,
                    "detection_size": {"w": 60, "h": 60},
                    "variants": [],
                    "variant_rows_all": [],
                }),
                mock.patch("extraction.ocr_engine.get_tesseract_lang", return_value="eng"),
                mock.patch("extraction.ocr_engine.ocr_page", return_value=(tokens, 60, 60, None)),
                mock.patch("extraction.ocr_engine.render_pdf_page", return_value=(np.zeros((60, 60), dtype=np.uint8), 60, 60)),
                mock.patch("extraction.ocr_engine.reocr_low_confidence_tokens", side_effect=lambda *args, **kwargs: tokens),
                mock.patch("extraction.token_projector.scale_tokens_to_dpi", side_effect=lambda toks, *_args, **_kwargs: toks),
                mock.patch("extraction.page_analyzer.extract_flow_text", return_value=flow),
                mock.patch("extraction.chart_detection.detect_charts", return_value=[]),
                mock.patch("extraction.debug_exporter.export_page_debug"),
            ):
                page_data = proc._process_debug_method_page(Path("dummy.pdf"), 1, page_dir, settings)

        self.assertEqual(page_data.get("tables"), [])
        self.assertEqual(page_data.get("flow"), flow)
        self.assertIn("bordered_table_detection_timeout", page_data.get("warnings", []))
        self.assertFalse(bool(page_data.get("timeout")))
        self.assertIsNone(page_data.get("error"))

    def test_table_grid_prepass_requires_opt_in(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        import eidat_manager_process as proc

        with tempfile.TemporaryDirectory() as tmp:
            page_dir = Path(tmp) / "page_1"
            self._write_page_png(page_dir / "page_1.png")
            tokens = [{"text": "Pin", "x0": 1.0, "y0": 1.0, "x1": 10.0, "y1": 8.0}]

            settings = self._base_settings()
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "extraction.graph_page_guard.find_chart_like_regions",
                        return_value={"regions": [], "stats": {}},
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "debug_method.run_table_variants._run_for_input",
                        return_value={
                            "tables": [],
                            "warnings": [],
                            "detection_dpi": 900,
                            "detection_size": {"w": 60, "h": 60},
                            "variants": [],
                            "variant_rows_all": [],
                        },
                    )
                )
                stack.enter_context(mock.patch("extraction.ocr_engine.get_tesseract_lang", return_value="eng"))
                stack.enter_context(mock.patch("extraction.ocr_engine.ocr_page", return_value=(tokens, 60, 60, None)))
                stack.enter_context(
                    mock.patch(
                        "extraction.ocr_engine.render_pdf_page",
                        return_value=(np.zeros((60, 60), dtype=np.uint8), 60, 60),
                    )
                )
                stack.enter_context(
                    mock.patch("extraction.ocr_engine.reocr_low_confidence_tokens", side_effect=lambda *args, **kwargs: tokens)
                )
                stack.enter_context(
                    mock.patch("extraction.token_projector.scale_tokens_to_dpi", side_effect=lambda toks, *_args, **_kwargs: toks)
                )
                stack.enter_context(
                    mock.patch("extraction.page_analyzer.extract_flow_text", return_value={"body_tokens": tokens, "table_titles": []})
                )
                stack.enter_context(mock.patch("extraction.chart_detection.detect_charts", return_value=[]))
                stack.enter_context(mock.patch("extraction.debug_exporter.export_page_debug"))
                stack.enter_context(
                    mock.patch("debug_method.table_grid_debug.run_for_image", side_effect=AssertionError("prepass called"))
                )
                proc._process_debug_method_page(Path("dummy.pdf"), 1, page_dir, settings)

            settings_with_prepass = dict(settings)
            settings_with_prepass["table_grid_enable_prepass"] = True
            called = {"count": 0}

            def _grid(*args, **kwargs):
                called["count"] += 1
                return {"tables": []}

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "extraction.graph_page_guard.find_chart_like_regions",
                        return_value={"regions": [], "stats": {}},
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "debug_method.run_table_variants._run_for_input",
                        return_value={
                            "tables": [],
                            "warnings": [],
                            "detection_dpi": 900,
                            "detection_size": {"w": 60, "h": 60},
                            "variants": [],
                            "variant_rows_all": [],
                        },
                    )
                )
                stack.enter_context(mock.patch("extraction.ocr_engine.get_tesseract_lang", return_value="eng"))
                stack.enter_context(mock.patch("extraction.ocr_engine.ocr_page", return_value=(tokens, 60, 60, None)))
                stack.enter_context(
                    mock.patch(
                        "extraction.ocr_engine.render_pdf_page",
                        return_value=(np.zeros((60, 60), dtype=np.uint8), 60, 60),
                    )
                )
                stack.enter_context(
                    mock.patch("extraction.ocr_engine.reocr_low_confidence_tokens", side_effect=lambda *args, **kwargs: tokens)
                )
                stack.enter_context(
                    mock.patch("extraction.token_projector.scale_tokens_to_dpi", side_effect=lambda toks, *_args, **_kwargs: toks)
                )
                stack.enter_context(
                    mock.patch("extraction.page_analyzer.extract_flow_text", return_value={"body_tokens": tokens, "table_titles": []})
                )
                stack.enter_context(mock.patch("extraction.chart_detection.detect_charts", return_value=[]))
                stack.enter_context(mock.patch("extraction.debug_exporter.export_page_debug"))
                stack.enter_context(mock.patch("debug_method.table_grid_debug.run_for_image", side_effect=_grid))
                proc._process_debug_method_page(Path("dummy.pdf"), 1, page_dir, settings_with_prepass)

        self.assertEqual(called["count"], 1)

    def test_main_processor_page_timeout_continues_to_next_page(self) -> None:
        import eidat_manager_process as proc

        settings = self._base_settings()
        settings["page_timeout_sec"] = 3

        def _render(pdf_path: Path, pages_root: Path, dpi: int) -> int:
            for page_num in (1, 2):
                page_dir = pages_root / f"page_{page_num}"
                page_dir.mkdir(parents=True, exist_ok=True)
                (page_dir / f"page_{page_num}.png").write_bytes(b"png")
            return 2

        def _write_combined(_pdf_path, pages_data, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "combined.txt"
            path.write_text(str(len(pages_data)), encoding="utf-8")
            return path

        def _write_summary(_pdf_path, pages_data, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "summary.json"
            path.write_text(str(len(pages_data)), encoding="utf-8")
            return path

        result_page_2 = {
            "page": 2,
            "tokens": [{"text": "ok"}],
            "tables": [],
            "charts": [],
            "img_w": 10,
            "img_h": 10,
            "dpi": 900,
            "ocr_dpi": 450,
            "flow": {"body_tokens": [{"text": "ok"}]},
        }
        fake_ctx = _FakeContext(
            [
                {"alive": True},
                {"payload": ("ok", result_page_2)},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with (
                mock.patch.object(proc, "_build_debug_method_settings", return_value=settings),
                mock.patch.object(proc, "_render_pdf_pages_to_dirs", side_effect=_render),
                mock.patch("eidat_manager_process.multiprocessing.get_context", return_value=fake_ctx),
                mock.patch("extraction.debug_exporter.export_combined_text", side_effect=_write_combined),
                mock.patch("extraction.debug_exporter.create_summary_report", side_effect=_write_summary),
            ):
                result = proc._run_debug_method_extraction(Path("dummy.pdf"), dpi=None, output_dir=out_dir)

        pages = list(result.get("results") or [])
        self.assertEqual(len(pages), 2)
        self.assertTrue(pages[0].get("timeout"))
        self.assertEqual(pages[1].get("page"), 2)
        self.assertEqual(pages[1].get("flow"), {"body_tokens": [{"text": "ok"}]})

    def test_variant_candidate_sidecar_uses_default_ascii_for_combined_mapping(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        import eidat_manager_process as proc

        os.environ["EIDAT_TABLE_SPLIT_MODE"] = "replica"
        os.environ["EIDAT_COMBINED_TABLE_SPLIT_MODE"] = "vline"

        table = {
            "bbox_px": [0.0, 0.0, 120.0, 60.0],
            "cells": [
                {"row": 0, "col": 0, "text": "A", "bbox_px": [0.0, 0.0, 60.0, 20.0]},
                {"row": 0, "col": 2, "text": "C", "bbox_px": [60.0, 0.0, 120.0, 20.0]},
                {"row": 2, "col": 0, "text": "1", "bbox_px": [0.0, 40.0, 60.0, 60.0]},
                {"row": 2, "col": 2, "text": "3", "bbox_px": [60.0, 40.0, 120.0, 60.0]},
            ],
            "borderless": False,
        }
        variant_table = {
            "rows": [
                ["A", "", "C"],
                ["", "", ""],
                ["1", "", "3"],
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            page_dir = Path(tmp) / "page_1"
            self._write_page_png(page_dir / "page_1.png")
            settings = self._base_settings()
            tokens = [{"text": "Voltage", "x0": 1.0, "y0": 1.0, "x1": 10.0, "y1": 8.0}]
            flow = {"body_tokens": tokens, "table_titles": []}

            with (
                mock.patch("extraction.graph_page_guard.find_chart_like_regions", return_value={"regions": [], "stats": {}}),
                mock.patch(
                    "debug_method.run_table_variants._run_for_input",
                    return_value={
                        "tables": [table],
                        "warnings": [],
                        "detection_dpi": 900,
                        "detection_size": {"w": 60, "h": 60},
                        "variants": [{"name": "majority"}],
                        "variant_rows_all": [[variant_table]],
                    },
                ),
                mock.patch(
                    "extraction.debug_exporter._split_table_on_vline_mismatch_for_display",
                    return_value=[table],
                ),
                mock.patch("extraction.ocr_engine.get_tesseract_lang", return_value="eng"),
                mock.patch("extraction.ocr_engine.ocr_page", return_value=(tokens, 60, 60, None)),
                mock.patch("extraction.ocr_engine.render_pdf_page", return_value=(np.zeros((60, 60), dtype=np.uint8), 60, 60)),
                mock.patch("extraction.ocr_engine.reocr_low_confidence_tokens", side_effect=lambda *args, **kwargs: tokens),
                mock.patch("extraction.token_projector.scale_tokens_to_dpi", side_effect=lambda toks, *_args, **_kwargs: toks),
                mock.patch("extraction.page_analyzer.extract_flow_text", return_value=flow),
                mock.patch("extraction.chart_detection.detect_charts", return_value=[]),
                mock.patch("extraction.debug_exporter.export_page_debug"),
            ):
                proc._process_debug_method_page(Path("dummy.pdf"), 1, page_dir, settings)

            sidecar = json.loads((page_dir / "table_variant_candidates.json").read_text(encoding="utf-8"))

        self.assertEqual(sidecar.get("ascii_mode"), "default")


if __name__ == "__main__":
    unittest.main()
