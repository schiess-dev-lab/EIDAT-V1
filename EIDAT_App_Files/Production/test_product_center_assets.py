import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from Production import node_product_center_root, runtime_product_center_root, sync_product_center_assets  # noqa: E402
from Production.admin_runner import run_update_processor  # noqa: E402


class TestProductCenterAssetSync(unittest.TestCase):
    def test_sync_product_center_assets_copies_runtime_assets_into_node_userdata(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            src = runtime_product_center_root(runtime_root) / "images"
            src.mkdir(parents=True, exist_ok=True)
            (src / "pump_model.png").write_bytes(b"png")

            payload = sync_product_center_assets(runtime_root, node_root)

            dst = node_product_center_root(node_root) / "images" / "pump_model.png"
            self.assertTrue(dst.exists())
            self.assertEqual(int(payload.get("copied") or 0), 1)
            self.assertTrue(bool(payload.get("present")))

    def test_run_update_processor_syncs_product_center_assets(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            app_root = runtime_root / "EIDAT_App_Files" / "Production"
            app_root.mkdir(parents=True, exist_ok=True)
            (app_root / "requirements-node.txt").write_text("", encoding="utf-8")
            (app_root / "requirements-node-ui.txt").write_text("", encoding="utf-8")
            src = runtime_product_center_root(runtime_root) / "images"
            src.mkdir(parents=True, exist_ok=True)
            (src / "pump_model.png").write_bytes(b"png")
            node_root.mkdir(parents=True, exist_ok=True)

            with patch("Production.bootstrap_env.main", return_value=0):
                result = run_update_processor(node_root=node_root, runtime_root=runtime_root)

            self.assertTrue(result.ok)
            dst = node_product_center_root(node_root) / "images" / "pump_model.png"
            self.assertTrue(dst.exists())
            self.assertEqual(
                int(((result.outputs.get("product_center_assets") or {}) if isinstance(result.outputs, dict) else {}).get("copied") or 0),
                1,
            )


if __name__ == "__main__":
    unittest.main()
