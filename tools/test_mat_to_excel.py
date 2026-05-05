from __future__ import annotations

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "mat_to_excel.py"


def _load_mat_to_excel_module():
    spec = importlib.util.spec_from_file_location("eidat_test_mat_to_excel", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMatToExcelRuntime(unittest.TestCase):
    def test_load_mat_reports_scipy_import_context(self) -> None:
        mod = _load_mat_to_excel_module()
        original_import = builtins.__import__

        def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "scipy.io" or str(name).startswith("scipy"):
                raise ImportError("DLL load failed while importing scipy.io")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=_blocked_import):
            with self.assertRaises(RuntimeError) as ctx:
                mod._load_mat(Path("sample.mat"))

        message = str(ctx.exception)
        self.assertIn("Failed to import `scipy.io.loadmat`", message)
        self.assertIn("DLL load failed while importing scipy.io", message)
        self.assertIn(sys.executable, message)
        self.assertIn(sys.version.split()[0], message)


if __name__ == "__main__":
    unittest.main()
