from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestBackendRuntimeEnv(unittest.TestCase):
    def test_resolve_project_python_ignores_non_runnable_repo_venv(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            temp_app_root = Path(tmp)
            venv_python = temp_app_root / ".venv" / "Scripts" / "python.exe"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

            with patch.object(backend, "APP_ROOT", temp_app_root), patch.object(
                backend, "load_scanner_env", return_value={}
            ), patch.object(
                backend, "_python_executable_is_usable", return_value=False
            ), patch.object(
                backend, "_emit_runtime_warning_once"
            ) as warn_mock:
                resolved = backend.resolve_project_python()

            self.assertEqual(resolved, sys.executable)
            warn_mock.assert_called_once()
            self.assertIn(str(venv_python), str(warn_mock.call_args[0][0]))

    def test_base_env_skips_vendored_site_packages_on_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            vendored = temp_root / "Lib" / "site-packages"
            vendored.mkdir(parents=True, exist_ok=True)
            marker = vendored / ".eidat_vendor_python.txt"
            marker.write_text("major_minor=3.13\nversion=3.13.9\n", encoding="utf-8")

            existing_path = f"{vendored}{os.pathsep}C:\\keep"
            with patch.dict(os.environ, {"PYTHONPATH": existing_path}, clear=True), patch.object(
                backend, "parse_scanner_env", return_value={}
            ), patch.object(
                backend, "VENDORED_SITE_PACKAGES", vendored
            ), patch.object(
                backend, "VENDORED_SITE_PACKAGES_MARKER", marker
            ), patch.object(
                backend, "_python_major_minor", return_value="3.11"
            ), patch.object(
                backend, "_emit_runtime_warning_once"
            ) as warn_mock:
                env = backend._base_env("C:\\Python311\\python.exe")

            self.assertEqual(env.get("PYTHONPATH"), "C:\\keep")
            self.assertIn("Python 3.13", str(env.get("EIDAT_VENDORED_SITE_PACKAGES_WARNING") or ""))
            warn_mock.assert_called_once()

    def test_base_env_uses_vendored_site_packages_when_marker_matches(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            vendored = temp_root / "Lib" / "site-packages"
            vendored.mkdir(parents=True, exist_ok=True)
            marker = vendored / ".eidat_vendor_python.txt"
            marker.write_text("major_minor=3.11\nversion=3.11.9\n", encoding="utf-8")

            with patch.dict(os.environ, {"PYTHONPATH": "C:\\keep"}, clear=True), patch.object(
                backend, "parse_scanner_env", return_value={}
            ), patch.object(
                backend, "VENDORED_SITE_PACKAGES", vendored
            ), patch.object(
                backend, "VENDORED_SITE_PACKAGES_MARKER", marker
            ), patch.object(
                backend, "_python_major_minor", return_value="3.11"
            ), patch.object(
                backend, "_emit_runtime_warning_once"
            ) as warn_mock:
                env = backend._base_env("C:\\Python311\\python.exe")

            self.assertEqual(env.get("PYTHONPATH"), f"{vendored}{os.pathsep}C:\\keep")
            self.assertNotIn("EIDAT_VENDORED_SITE_PACKAGES_WARNING", env)
            warn_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
