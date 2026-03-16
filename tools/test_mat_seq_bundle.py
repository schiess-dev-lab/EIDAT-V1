import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "EIDAT_App_Files" / "Application"))
sys.path.insert(0, str(ROOT / "EIDAT_App_Files"))

from eidat_manager_db import support_paths  # type: ignore
from eidat_manager_index import build_index  # type: ignore
from eidat_manager_mat_bundle import detect_mat_bundle_member, list_mat_bundle_members  # type: ignore
from eidat_manager_process import process_candidates  # type: ignore
from eidat_manager_scan import scan_global_repo  # type: ignore
from ui_next.backend import get_file_artifacts_path, read_eidat_index_documents  # type: ignore

try:
    import numpy as np  # type: ignore
    from scipy.io import savemat  # type: ignore
except Exception:
    np = None  # type: ignore
    savemat = None  # type: ignore


@unittest.skipUnless(np is not None and savemat is not None, "numpy/scipy not installed")
class TestMatSeqBundle(unittest.TestCase):
    def _write_valid_mat(self, path: Path, offset: float) -> None:
        savemat(
            str(path),
            {
                "time": np.array([0.0, 1.0, 2.0], dtype=float),
                "thrust": np.array([10.0 + offset, 11.0 + offset, 12.0 + offset], dtype=float),
                "pressure": np.array([[100.0 + offset], [101.0 + offset], [102.0 + offset]], dtype=float),
                "junkTelemetry": np.array([5000.0 + offset, 5001.0 + offset, 5002.0 + offset], dtype=float),
                "ignored_matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            },
        )

    def _write_invalid_mat(self, path: Path) -> None:
        savemat(
            str(path),
            {
                "bad_matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            },
        )

    def test_bundle_detector_matches_seq_files_only(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            data_dir = repo / "td"
            data_dir.mkdir(parents=True, exist_ok=True)
            good = data_dir / "SN123_seq01.mat"
            other = data_dir / "capture.mat"
            self._write_valid_mat(good, 0.0)
            self._write_valid_mat(other, 1.0)

            info = detect_mat_bundle_member(good, repo_root=repo)
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info.serial_number, "SN123")
            self.assertEqual(info.sequence_name, "seq01")
            self.assertIsNone(detect_mat_bundle_member(other, repo_root=repo))

    def test_processes_bundle_into_single_indexed_td_source(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            src_dir = repo / "Valve" / "ModelX"
            src_dir.mkdir(parents=True, exist_ok=True)
            seq1 = src_dir / "SN123_seq1.mat"
            seq2 = src_dir / "SN123_seq2.mat"
            seq3 = src_dir / "SN123_seq3.mat"
            self._write_valid_mat(seq1, 0.0)
            self._write_valid_mat(seq2, 10.0)
            self._write_valid_mat(seq3, 20.0)

            paths = support_paths(repo)
            scan_summary = scan_global_repo(paths)
            self.assertEqual(len(scan_summary.candidates), 3)

            results = process_candidates(paths)
            self.assertEqual(sum(1 for item in results if item.ok), 3)

            members = list_mat_bundle_members(seq1, repo_root=repo)
            self.assertEqual([m.sequence_name for m in members], ["seq1", "seq2", "seq3"])
            bundle = detect_mat_bundle_member(seq1, repo_root=repo)
            assert bundle is not None
            artifacts_dir = paths.support_dir / "debug" / "ocr" / f"{bundle.bundle_stem}__excel"
            sqlite_path = artifacts_dir / f"{bundle.bundle_stem}.sqlite3"
            metadata_path = artifacts_dir / f"{bundle.bundle_stem}_metadata.json"
            manifest_path = artifacts_dir / "mat_seq_bundle.json"

            self.assertTrue(artifacts_dir.exists())
            self.assertTrue(sqlite_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertFalse((paths.support_dir / "debug" / "ocr" / "SN123_seq1__excel").exists())
            meta = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
            manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(str(meta.get("asset_type") or "").strip(), "Valve")
            self.assertEqual(str(meta.get("asset_specific_type") or "").strip(), "ModelX")
            self.assertEqual(str(manifest.get("asset_type") or "").strip(), "Valve")
            self.assertEqual(str(manifest.get("asset_specific_type") or "").strip(), "ModelX")

            with sqlite3.connect(str(sqlite_path)) as conn:
                runs = [str(row[0] or "") for row in conn.execute("SELECT sheet_name FROM __sheet_info ORDER BY sheet_name").fetchall()]
                member_count = int(conn.execute("SELECT COUNT(*) FROM __mat_bundle_members").fetchone()[0] or 0)
                seq1_cols = [str(row[1] or "") for row in conn.execute('PRAGMA table_info("sheet__seq1")').fetchall()]
            self.assertEqual(runs, ["seq1", "seq2", "seq3"])
            self.assertEqual(member_count, 3)
            self.assertIn("Time", seq1_cols)
            self.assertIn("Thrust", seq1_cols)
            self.assertIn("pressure", seq1_cols)
            self.assertNotIn("junkTelemetry", seq1_cols)

            build_index(paths)
            docs = read_eidat_index_documents(repo)
            td_docs = [doc for doc in docs if str(doc.get("serial_number") or "").strip() == "SN123"]
            self.assertEqual(len(td_docs), 1)
            self.assertIn(bundle.bundle_stem, str(td_docs[0].get("excel_sqlite_rel") or ""))

            resolved_artifacts = get_file_artifacts_path(repo, "Valve/ModelX/SN123_seq2.mat")
            self.assertIsNotNone(resolved_artifacts)
            self.assertEqual(resolved_artifacts, artifacts_dir)

            scan_after = scan_global_repo(paths)
            self.assertEqual(len(scan_after.candidates), 0)

    def test_invalid_bundle_member_fails_bundle(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            src_dir = repo / "Data" / "TD"
            src_dir.mkdir(parents=True, exist_ok=True)
            good = src_dir / "SN777_seq1.mat"
            bad = src_dir / "SN777_seq2.mat"
            self._write_valid_mat(good, 0.0)
            self._write_invalid_mat(bad)

            paths = support_paths(repo)
            scan_global_repo(paths)
            results = process_candidates(paths)
            self.assertTrue(any((not item.ok) and "did not contain any numeric 1-D series" in str(item.error or "") for item in results))

    def test_non_bundle_mat_still_uses_standalone_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            src_dir = repo / "Data"
            src_dir.mkdir(parents=True, exist_ok=True)
            mat_path = src_dir / "capture.mat"
            self._write_valid_mat(mat_path, 0.0)

            paths = support_paths(repo)
            scan_global_repo(paths)
            results = process_candidates(paths)
            self.assertTrue(any(item.ok for item in results))
            standalone_dir = paths.support_dir / "debug" / "ocr" / "capture__excel"
            self.assertTrue(standalone_dir.exists())


if __name__ == "__main__":
    unittest.main()
