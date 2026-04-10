import sys
import tempfile
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from Production import deploy  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
