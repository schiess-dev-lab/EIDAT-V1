import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestTableDetectTimeoutWiring(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.copy()
        self._saved_sys_path = list(sys.path)
        app_root = Path(__file__).resolve().parents[1]
        if str(app_root) not in sys.path:
            sys.path.insert(0, str(app_root))

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)
        sys.path[:] = self._saved_sys_path

    def test_guardrails_on_uses_timeout_wrapper(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        from debug_method import run_table_variants

        os.environ["EIDAT_TABLE_DETECT_GUARDRAILS"] = "1"
        os.environ["EIDAT_TABLE_DETECT_TIMEOUT_SEC"] = "1"

        called = {"timeout": 0}

        def _timeout(*args, **kwargs):
            called["timeout"] += 1
            return {"tables": [], "cells": [], "timed_out": False}

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with (
                mock.patch("extraction.table_detect_timeout.detect_tables_hard_timeout", side_effect=_timeout),
                mock.patch.object(run_table_variants, "_load_gray_image", return_value=np.zeros((20, 20), dtype=np.uint8)),
                mock.patch("extraction.table_detection.detect_tables", side_effect=AssertionError("direct detect_tables called")),
            ):
                payload = run_table_variants._run_for_input(
                    Path("dummy.png"),
                    out_dir=out_dir,
                    page=1,
                    ocr_dpi_base=450,
                    detection_dpi=900,
                    lang="eng",
                    clean=False,
                    fuse=True,
                    emit_variants=False,
                    emit_fused=False,
                    allow_no_tables=True,
                    enable_borderless=False,
                    return_fused=True,
                    return_variant_rows=False,
                )
        self.assertIsInstance(payload, dict)
        self.assertEqual(called["timeout"], 1)

    def test_guardrails_off_skips_timeout_wrapper(self) -> None:
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")

        from debug_method import run_table_variants

        os.environ["EIDAT_TABLE_DETECT_GUARDRAILS"] = "0"
        os.environ["EIDAT_TABLE_DETECT_TIMEOUT_SEC"] = "1"

        called = {"direct": 0, "timeout": 0}

        def _direct(*args, **kwargs):
            called["direct"] += 1
            return {"tables": [], "cells": []}

        def _timeout(*args, **kwargs):
            called["timeout"] += 1
            return {"tables": [], "cells": []}

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with (
                mock.patch("extraction.table_detection.detect_tables", side_effect=_direct),
                mock.patch("extraction.table_detect_timeout.detect_tables_hard_timeout", side_effect=_timeout),
                mock.patch.object(run_table_variants, "_load_gray_image", return_value=np.zeros((20, 20), dtype=np.uint8)),
            ):
                payload = run_table_variants._run_for_input(
                    Path("dummy.png"),
                    out_dir=out_dir,
                    page=1,
                    ocr_dpi_base=450,
                    detection_dpi=900,
                    lang="eng",
                    clean=False,
                    fuse=True,
                    emit_variants=False,
                    emit_fused=False,
                    allow_no_tables=True,
                    enable_borderless=False,
                    return_fused=True,
                    return_variant_rows=False,
                )
        self.assertIsInstance(payload, dict)
        self.assertEqual(called["timeout"], 0)
        self.assertEqual(called["direct"], 1)


if __name__ == "__main__":
    unittest.main()
