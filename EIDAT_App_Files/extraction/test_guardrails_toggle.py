import os
import unittest


class TestGuardrailsToggle(unittest.TestCase):
    def setUp(self) -> None:
        from . import batch_processor

        self.batch_processor = batch_processor
        self._saved_env = os.environ.copy()

        # Save originals for restore.
        self._orig_render = batch_processor.ocr_engine.render_pdf_page
        self._orig_ocr = batch_processor.ocr_engine.ocr_page
        self._orig_detect_tables = batch_processor.table_detection.detect_tables
        self._orig_detect_tables_with_timeout = batch_processor._detect_tables_with_timeout

        # Avoid running the full pipeline: return no OCR tokens so process_page exits early.
        batch_processor.ocr_engine.render_pdf_page = lambda *args, **kwargs: ("img", 100, 100)
        batch_processor.ocr_engine.ocr_page = lambda *args, **kwargs: ([], 0, 0, None)

    def tearDown(self) -> None:
        bp = self.batch_processor
        bp.ocr_engine.render_pdf_page = self._orig_render
        bp.ocr_engine.ocr_page = self._orig_ocr
        bp.table_detection.detect_tables = self._orig_detect_tables
        bp._detect_tables_with_timeout = self._orig_detect_tables_with_timeout

        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_guardrails_off_skips_timeout_wrapper(self) -> None:
        bp = self.batch_processor
        called = {"direct": 0, "timeout": 0}

        def _direct(*args, **kwargs):
            called["direct"] += 1
            return {"tables": [], "cells": []}

        def _timeout(*args, **kwargs):
            called["timeout"] += 1
            return {"tables": [], "cells": []}

        bp.table_detection.detect_tables = _direct
        bp._detect_tables_with_timeout = _timeout

        os.environ["EIDAT_TABLE_DETECT_GUARDRAILS"] = "0"
        os.environ["EIDAT_TABLE_DETECT_TIMEOUT_SEC"] = "1"

        pipeline = bp.ExtractionPipeline()
        res = pipeline.process_page(pdf_path=bp.Path("dummy.pdf"), page_num=0, debug_dir=None, verbose=False)
        self.assertIsInstance(res, dict)
        self.assertEqual(called["timeout"], 0)
        self.assertEqual(called["direct"], 1)

    def test_guardrails_on_uses_timeout_wrapper(self) -> None:
        bp = self.batch_processor
        called = {"direct": 0, "timeout": 0}

        def _direct(*args, **kwargs):
            called["direct"] += 1
            return {"tables": [], "cells": []}

        def _timeout(*args, **kwargs):
            called["timeout"] += 1
            return {"tables": [], "cells": []}

        bp.table_detection.detect_tables = _direct
        bp._detect_tables_with_timeout = _timeout

        os.environ["EIDAT_TABLE_DETECT_GUARDRAILS"] = "1"
        os.environ["EIDAT_TABLE_DETECT_TIMEOUT_SEC"] = "1"

        pipeline = bp.ExtractionPipeline()
        res = pipeline.process_page(pdf_path=bp.Path("dummy.pdf"), page_num=0, debug_dir=None, verbose=False)
        self.assertIsInstance(res, dict)
        self.assertEqual(called["timeout"], 1)
        self.assertEqual(called["direct"], 0)


if __name__ == "__main__":
    unittest.main()

