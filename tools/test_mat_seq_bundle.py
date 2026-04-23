import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "EIDAT_App_Files" / "Application"))
sys.path.insert(0, str(ROOT / "EIDAT_App_Files"))

from eidat_manager_db import support_paths  # type: ignore
from eidat_manager import _cmd_process  # type: ignore
from eidat_manager_index import build_index  # type: ignore
from eidat_manager_mat_bundle import detect_mat_bundle_member, list_mat_bundle_members  # type: ignore
from eidat_manager_process import process_candidates, rebuild_td_serial_aggregates  # type: ignore
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

    def _write_td_source_sqlite(self, path: Path, *, source_sheet_name: str, serial_number: str, thrust_offset: float = 0.0) -> None:
        table_name = f"sheet__{source_sheet_name}"
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                f'''
                CREATE TABLE "{table_name}" (
                    excel_row INTEGER NOT NULL,
                    "Time" REAL,
                    thrust REAL
                )
                '''
            )
            conn.executemany(
                f'INSERT INTO "{table_name}"(excel_row,"Time",thrust) VALUES(?,?,?)',
                [
                    (1, 0.0, 10.0 + thrust_offset),
                    (2, 1.0, 20.0 + thrust_offset),
                    (3, 2.0, 30.0 + thrust_offset),
                ],
            )
            conn.execute(
                """
                CREATE TABLE __sheet_info (
                    sheet_name TEXT PRIMARY KEY,
                    source_sheet_name TEXT,
                    table_name TEXT NOT NULL,
                    header_row INTEGER NOT NULL,
                    import_order INTEGER,
                    excel_col_indices_json TEXT NOT NULL,
                    headers_json TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    mapped_headers_json TEXT,
                    rows_inserted INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO __sheet_info(
                    sheet_name, source_sheet_name, table_name, header_row, import_order,
                    excel_col_indices_json, headers_json, columns_json, mapped_headers_json, rows_inserted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_sheet_name,
                    source_sheet_name,
                    table_name,
                    1,
                    1,
                    "[1,2,3]",
                    '["excel_row","Time","thrust"]',
                    '{"excel_row":"INTEGER","Time":"REAL","thrust":"REAL"}',
                    "[]",
                    3,
                ),
            )
            conn.execute(
                """
                CREATE TABLE __meta_cells (
                    sheet_name TEXT NOT NULL,
                    excel_row INTEGER NOT NULL,
                    excel_col INTEGER NOT NULL,
                    value TEXT NOT NULL
                )
                """
            )
            conn.executemany(
                "INSERT INTO __meta_cells(sheet_name, excel_row, excel_col, value) VALUES(?, ?, ?, ?)",
                [
                    (source_sheet_name, 1, 1, "Serial Number"),
                    (source_sheet_name, 1, 2, serial_number),
                ],
            )
            conn.commit()

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
            artifact_matches = list(
                (paths.support_dir / "Test Data File Extractions").glob(
                    f"*/*/*/SN123/sources/{bundle.bundle_stem}__*__excel"
                )
            )
            self.assertEqual(len(artifact_matches), 1)
            artifacts_dir = artifact_matches[0]
            sqlite_path = artifacts_dir / f"{bundle.bundle_stem}.sqlite3"
            metadata_path = artifacts_dir / f"{bundle.bundle_stem}_metadata.json"
            manifest_path = artifacts_dir / "mat_seq_bundle.json"
            aggregate_dir = artifacts_dir.parent.parent
            aggregate_sqlite = aggregate_dir / "SN123.sqlite3"
            aggregate_metadata = aggregate_dir / "SN123_metadata.json"
            aggregate_manifest = aggregate_dir / "td_serial_aggregate.json"

            self.assertTrue(artifacts_dir.exists())
            self.assertTrue(sqlite_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(aggregate_sqlite.exists())
            self.assertTrue(aggregate_metadata.exists())
            self.assertTrue(aggregate_manifest.exists())
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
            self.assertEqual(len(td_docs), 2)
            source_docs = [
                doc
                for doc in td_docs
                if "/sources/" in str(doc.get("artifacts_rel") or "").replace("\\", "/")
            ]
            aggregate_docs = [
                doc
                for doc in td_docs
                if "/sources/" not in str(doc.get("artifacts_rel") or "").replace("\\", "/")
            ]
            self.assertEqual(len(source_docs), 1)
            self.assertEqual(len(aggregate_docs), 1)
            self.assertIn(bundle.bundle_stem, str(source_docs[0].get("excel_sqlite_rel") or ""))
            self.assertTrue(str(aggregate_docs[0].get("excel_sqlite_rel") or "").replace("\\", "/").endswith("/SN123/SN123.sqlite3"))

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
            matches = list(
                (paths.support_dir / "Test Data File Extractions").glob(
                    "*/*/*/*/sources/capture__*__excel"
                )
            )
            self.assertEqual(len(matches), 1)

    def test_rebuild_td_serial_aggregates_prefers_newest_metadata_for_header(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            paths = support_paths(repo)
            support_dir = paths.support_dir
            src_a_rel = Path("Test Data File Extractions/Program_A/Thruster/Valve/SN123/sources/source_a__1111111111__excel")
            src_b_rel = Path("Test Data File Extractions/Program_A/Thruster/Valve/SN123/sources/source_b__2222222222__excel")
            src_a_dir = support_dir / src_a_rel
            src_b_dir = support_dir / src_b_rel
            src_a_dir.mkdir(parents=True, exist_ok=True)
            src_b_dir.mkdir(parents=True, exist_ok=True)
            src_a_sqlite = src_a_dir / "source_a.sqlite3"
            src_b_sqlite = src_b_dir / "source_b.sqlite3"
            self._write_td_source_sqlite(src_a_sqlite, source_sheet_name="seq1", serial_number="SN123", thrust_offset=0.0)
            self._write_td_source_sqlite(src_b_sqlite, source_sheet_name="seq2", serial_number="SN123", thrust_offset=100.0)

            meta_a = src_a_dir / "source_a_metadata.json"
            meta_b = src_b_dir / "source_b_metadata.json"
            src_a_rel_text = str(src_a_rel).replace("/", "\\")
            src_b_rel_text = str(src_b_rel).replace("/", "\\")
            meta_a.write_text(
                __import__("json").dumps(
                    {
                        "program_title": "Program A",
                        "asset_type": "Thruster",
                        "asset_specific_type": "Valve",
                        "serial_number": "SN123",
                        "document_type": "TD",
                        "document_type_acronym": "TD",
                        "document_type_status": "confirmed",
                        "document_type_review_required": False,
                        "vendor": "Vendor Old",
                        "excel_sqlite_rel": f"EIDAT Support\\{src_a_rel_text}\\source_a.sqlite3",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            time.sleep(0.05)
            meta_b.write_text(
                __import__("json").dumps(
                    {
                        "program_title": "Program A",
                        "asset_type": "Thruster",
                        "asset_specific_type": "Valve",
                        "serial_number": "SN123",
                        "document_type": "TD",
                        "document_type_acronym": "TD",
                        "document_type_status": "confirmed",
                        "document_type_review_required": False,
                        "vendor": "Vendor New",
                        "excel_sqlite_rel": f"EIDAT Support\\{src_b_rel_text}\\source_b.sqlite3",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            payload = rebuild_td_serial_aggregates(paths)
            self.assertEqual(int(payload.get("aggregate_count") or 0), 1)

            aggregate_dir = support_dir / "Test Data File Extractions" / "Program_A" / "Thruster" / "Valve" / "SN123"
            aggregate_meta = __import__("json").loads((aggregate_dir / "SN123_metadata.json").read_text(encoding="utf-8"))
            aggregate_manifest = __import__("json").loads((aggregate_dir / "td_serial_aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(str(aggregate_meta.get("vendor") or "").strip(), "Vendor New")
            self.assertEqual(
                str(aggregate_meta.get("aggregate_header_source_metadata_rel") or "").replace("\\", "/"),
                str((src_b_rel / "source_b_metadata.json")).replace("\\", "/"),
            )
            self.assertEqual(
                str(aggregate_manifest.get("header_metadata_source_metadata_rel") or "").replace("\\", "/"),
                str((src_b_rel / "source_b_metadata.json")).replace("\\", "/"),
            )

    def test_cmd_process_returns_td_aggregate_and_index_payload(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            src_dir = repo / "Valve" / "ModelX"
            src_dir.mkdir(parents=True, exist_ok=True)
            seq1 = src_dir / "SN123_seq1.mat"
            seq2 = src_dir / "SN123_seq2.mat"
            self._write_valid_mat(seq1, 0.0)
            self._write_valid_mat(seq2, 10.0)

            paths = support_paths(repo)
            scan_global_repo(paths)
            payload = _cmd_process(
                paths,
                limit=None,
                dpi=None,
                force=False,
                only_candidates=False,
                file_paths=None,
            )

            self.assertEqual(int(payload.get("processed_ok") or 0), 2)
            self.assertIn("td_serial_aggregates", payload)
            self.assertIn("index", payload)
            td_payload = dict(payload.get("td_serial_aggregates") or {})
            self.assertTrue(bool(td_payload.get("triggered")))
            self.assertEqual(int(td_payload.get("aggregate_count") or 0), 1)
            self.assertEqual(int((payload.get("index") or {}).get("metadata_count") or 0), 2)


if __name__ == "__main__":
    unittest.main()
