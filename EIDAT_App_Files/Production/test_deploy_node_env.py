import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from Production import bootstrap_env, deploy  # noqa: E402


class TestDeployNodeEnv(unittest.TestCase):
    def test_deploy_strips_repo_root_from_node_env_copy(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            (runtime_root / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)
            user_inputs = runtime_root / "user_inputs"
            user_inputs.mkdir(parents=True, exist_ok=True)
            (user_inputs / "scanner.env").write_text(
                "# Shared scanner config\n"
                "QUIET=1\n"
                "REPO_ROOT=C:\\wrong\\shared\\repo\n"
                "OCR_MODE=fallback\n",
                encoding="utf-8",
            )

            rc = deploy.main(
                [
                    "--node-root",
                    str(node_root),
                    "--runtime-root",
                    str(runtime_root),
                    "--no-bootstrap-node-ui",
                    "--no-mirror-node",
                ]
            )

            self.assertEqual(rc, 0)
            node_env = node_root / "EIDAT" / "UserData" / ".env"
            self.assertTrue(node_env.exists())
            text = node_env.read_text(encoding="utf-8")
            self.assertIn("QUIET=1", text)
            self.assertIn("OCR_MODE=fallback", text)
            self.assertNotIn("REPO_ROOT=", text)

    def test_bootstrap_env_reinstalls_when_reportlab_import_is_missing(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            node_root = root / "node"
            venv_dir = root / "node-ui-venv"
            vpy = venv_dir / "Scripts" / "python.exe"
            vpy.parent.mkdir(parents=True, exist_ok=True)
            vpy.write_text("", encoding="utf-8")

            req_path = root / "requirements.txt"
            req_path.write_text("reportlab\n", encoding="utf-8")
            want_hash = bootstrap_env._requirements_hash(req_path)
            (venv_dir / "eidat_installed_hash.txt").write_text(want_hash + "\n", encoding="utf-8")

            seen_imports: list[str] = []

            def _fake_can_import(_vpy: Path, module: str) -> bool:
                seen_imports.append(str(module))
                return str(module) != "reportlab"

            with mock.patch.object(bootstrap_env, "_ensure_venv", return_value=None), mock.patch.object(
                bootstrap_env, "_ensure_pip", return_value=None
            ), mock.patch.object(
                bootstrap_env, "_venv_can_import", side_effect=_fake_can_import
            ), mock.patch.object(
                bootstrap_env, "_install_requirements", return_value=None
            ) as install_mock:
                rc = bootstrap_env.main(
                    [
                        "--yes",
                        "--profile",
                        "ui",
                        "--node-root",
                        str(node_root),
                        "--venv-dir",
                        str(venv_dir),
                        "--requirements",
                        str(req_path),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertIn("reportlab", bootstrap_env.PROFILE_REQUIRED_IMPORTS["ui"])
            self.assertIn("reportlab", bootstrap_env.PROFILE_REQUIRED_IMPORTS["full"])
            self.assertIn("reportlab", seen_imports)
            install_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
