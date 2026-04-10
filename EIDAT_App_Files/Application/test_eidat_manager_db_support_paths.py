import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APP_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


import eidat_manager_db  # noqa: E402


class TestSupportPaths(unittest.TestCase):
    def test_support_paths_prefers_deepest_populated_support_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            shallow = repo / "EIDAT" / "EIDAT Support"
            deep = repo / "EIDAT" / "EIDAT" / "EIDAT Support"
            shallow.mkdir(parents=True, exist_ok=True)
            deep.mkdir(parents=True, exist_ok=True)
            (deep / "eidat_support.sqlite3").write_text("", encoding="utf-8")

            paths = eidat_manager_db.support_paths(repo)

            self.assertEqual(paths.support_dir, deep)


if __name__ == "__main__":
    unittest.main()
