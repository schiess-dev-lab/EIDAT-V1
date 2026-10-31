import json
import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from eidat_manager_artifacts import (  # noqa: E402
    excel_artifacts_dir,
    folder_asset_hints,
    hashed_excel_artifacts_dir,
    legacy_excel_artifacts_dir,
)


class TestExcelArtifactsDir(unittest.TestCase):
    def _layout(self, root: Path) -> tuple[Path, Path, Path]:
        repo = root / "node"
        support = repo / "EIDAT Support"
        a = repo / "Data Packages" / "Program" / "ValveA" / "SN123.xlsx"
        b = repo / "Data Packages" / "Program" / "ValveB" / "SN123.xlsx"
        for p in (a, b):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PK")
        support.mkdir(parents=True, exist_ok=True)
        return repo, a, b

    def _write_meta(self, folder: Path, stem: str, repo: Path, owner: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        rel = owner.relative_to(repo).as_posix()
        (folder / f"{stem}_metadata.json").write_text(
            json.dumps({"serial_number": stem, "source_rel_path": rel, "source_rel_paths": [rel]}),
            encoding="utf-8",
        )

    def test_first_claimant_keeps_legacy_second_same_name_gets_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, a, b = self._layout(Path(td))
            support = repo / "EIDAT Support"
            legacy = legacy_excel_artifacts_dir(support, a)
            # Nothing processed yet: both would claim the legacy folder; A processes first.
            self.assertEqual(excel_artifacts_dir(support, a, repo_root=repo), legacy)
            self._write_meta(legacy, "SN123", repo, a)
            # A keeps the legacy folder; B (same stem, other asset folder) is moved aside.
            self.assertEqual(excel_artifacts_dir(support, a, repo_root=repo), legacy)
            hashed_b = hashed_excel_artifacts_dir(support, b)
            self.assertEqual(excel_artifacts_dir(support, b, repo_root=repo), hashed_b)
            self.assertNotEqual(hashed_b, legacy)
            self.assertTrue(hashed_b.name.startswith("SN123__") and hashed_b.name.endswith("__excel"))
            # Once B's hashed folder exists it is always preferred, even if the legacy owner changes.
            self._write_meta(hashed_b, "SN123", repo, b)
            self.assertEqual(excel_artifacts_dir(support, b, repo_root=repo), hashed_b)
            self.assertNotEqual(hashed_excel_artifacts_dir(support, a), hashed_b)

    def test_legacy_folder_without_source_tracking_stays_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, a, b = self._layout(Path(td))
            support = repo / "EIDAT Support"
            legacy = legacy_excel_artifacts_dir(support, a)
            legacy.mkdir(parents=True)
            (legacy / "SN123_metadata.json").write_text(json.dumps({"serial_number": "SN123"}), encoding="utf-8")
            # Pre-existing artifacts with no recorded source: historical behaviour, no split.
            self.assertEqual(excel_artifacts_dir(support, a, repo_root=repo), legacy)
            self.assertEqual(excel_artifacts_dir(support, b, repo_root=repo), legacy)

    def test_owner_recorded_as_absolute_path_still_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, a, b = self._layout(Path(td))
            support = repo / "EIDAT Support"
            legacy = legacy_excel_artifacts_dir(support, a)
            legacy.mkdir(parents=True)
            (legacy / "SN123_metadata.json").write_text(
                json.dumps({"serial_number": "SN123", "source_file": str(a)}), encoding="utf-8"
            )
            self.assertEqual(excel_artifacts_dir(support, a, repo_root=repo), legacy)
            self.assertEqual(excel_artifacts_dir(support, b, repo_root=repo), hashed_excel_artifacts_dir(support, b))

    def test_folder_asset_hints(self) -> None:
        hints = folder_asset_hints(Path("/x/Program/ValveA/Model7/SN1.xlsx"))
        self.assertEqual(hints, {"asset_type": "ValveA", "asset_specific_type": "Model7"})


if __name__ == "__main__":
    unittest.main()
