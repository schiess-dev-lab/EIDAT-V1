import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402
from Application import eidat_manager_index  # noqa: E402
from Application.eidat_manager_db import support_paths  # noqa: E402


def _write_candidates(data_root: Path, payload: dict) -> None:
    ui = data_root / "user_inputs"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "metadata_candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_doc_type_strategies(data_root: Path) -> None:
    ui = data_root / "user_inputs"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "document_type_strategies.json").write_text(
        json.dumps(
            {
                "version": 1,
                "document_types": [
                    {"name": "End Item Data Package", "acronym": "EIDP"},
                    {"name": "Test Data", "acronym": "TD"},
                ],
                "filename_aliases": {
                    "EIDP": ["EIDP", "End Item Data Package"],
                    "TD": ["TD", "Test Data"],
                },
                "content_aliases": {
                    "EIDP": ["EIDP", "End Item Data Package"],
                    "TD": ["TD", "Test Data"],
                },
                "extension_rules": {"EIDP": [".pdf"], "TD": [".xlsx", ".xls", ".xlsm", ".mat"]},
                "folder_rules": {"levels": 3, "aliases": {"EIDP": ["EIDP"], "TD": ["TD", "Test Data"]}},
                "serial_patterns": ["(?i)\\bSN[-_ ]?[A-Z0-9]+(?:[-_][A-Z0-9]+)*\\b"],
                "ranker": {
                    "weights": {"content": 5, "folder": 3, "extension_compatible": 1, "serial_bonus": 2},
                    "min_score": 4,
                    "conflict_gap": 2,
                },
                "special_cases": {
                    "td_folder_serial_rule": {
                        "enabled": True,
                        "compatible_extensions": [".xlsx", ".xls", ".xlsm", ".mat"],
                        "require_serial_in_filename": True,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class TestBackendMetadataOverrides(unittest.TestCase):
    def setUp(self) -> None:
        self._old_data_root = os.environ.get("EIDAT_DATA_ROOT")

    def tearDown(self) -> None:
        if self._old_data_root is None:
            os.environ.pop("EIDAT_DATA_ROOT", None)
        else:
            os.environ["EIDAT_DATA_ROOT"] = self._old_data_root

    def _prepare_repo(self, td: str) -> tuple[Path, Path, Path]:
        repo = Path(td) / "repo"
        source = repo / "source"
        source.mkdir(parents=True, exist_ok=True)
        pdf_path = source / "doc1.pdf"
        pdf_path.write_text("pdf", encoding="utf-8")
        artifacts_dir = repo / "EIDAT Support" / "debug" / "ocr" / "doc1"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = artifacts_dir / "doc1_metadata.json"
        return repo, artifacts_dir, metadata_path

    def test_edit_metadata_for_files_writes_manual_override_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, artifacts_dir, metadata_path = self._prepare_repo(td)
            metadata_path.write_text(
                json.dumps(
                    {
                        "serial_number": "SN-001",
                        "program_title": "Program Alpha",
                        "asset_type": "Valve",
                        "asset_specific_type": "Valve1",
                        "vendor": "Vendor A",
                        "document_type": "EIDP",
                        "document_type_acronym": "EIDP",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            with patch.object(backend, "eidat_manager_index", return_value={"ok": True}), patch.object(
                backend, "_sync_projects_for_metadata_rels", return_value=([], [])
            ):
                result = backend.edit_metadata_for_files(
                    repo,
                    ["source/doc1.pdf"],
                    {"vendor": "Manual Vendor", "asset_type": "Manual Asset"},
                )

            self.assertEqual(result["updated"], 1)
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("vendor"), "Manual Vendor")
            self.assertEqual(payload.get("asset_type"), "Manual Asset")
            self.assertEqual(
                sorted(payload.get("manual_override_fields") or []),
                ["asset_type", "vendor"],
            )
            self.assertEqual(payload.get("metadata_source"), "mixed")
            self.assertTrue(str(payload.get("manual_override_updated_at") or "").strip())

    def test_refresh_metadata_only_preserves_then_overwrites_manual_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td) / "data_root"
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "vendors": ["MOOG", "ACME"],
                    "asset_types": ["Valve"],
                    "asset_specific_types": ["Valve1"],
                },
            )
            _write_doc_type_strategies(data_root)
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            repo, artifacts_dir, metadata_path = self._prepare_repo(td)
            (artifacts_dir / "combined.txt").write_text(
                "\n".join(
                    [
                        "=== Page 1 ===",
                        "Program: Starlink",
                        "Vendor: ACME",
                        "Valve model: Valve1",
                    ]
                ),
                encoding="utf-8",
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "serial_number": "SN-001",
                        "program_title": "Starlink",
                        "asset_type": "Valve",
                        "asset_specific_type": "Valve1",
                        "vendor": "MOOG",
                        "document_type": "EIDP",
                        "document_type_acronym": "EIDP",
                        "manual_override_fields": ["vendor"],
                        "manual_override_updated_at": "2026-03-24T00:00:00+00:00",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with patch.object(backend, "eidat_manager_index", return_value={"ok": True}), patch.object(
                backend, "_sync_projects_for_metadata_rels", return_value=([], [])
            ):
                backend.refresh_metadata_only(repo, ["source/doc1.pdf"], overwrite_manual_fields=False)
            preserved = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(preserved.get("vendor"), "MOOG")
            self.assertEqual(preserved.get("manual_override_fields"), ["vendor"])

            with patch.object(backend, "eidat_manager_index", return_value={"ok": True}), patch.object(
                backend, "_sync_projects_for_metadata_rels", return_value=([], [])
            ):
                backend.refresh_metadata_only(repo, ["source/doc1.pdf"], overwrite_manual_fields=True)
            overwritten = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(overwritten.get("vendor"), "ACME")
            self.assertEqual(overwritten.get("manual_override_fields"), [])
            self.assertEqual(overwritten.get("manual_override_updated_at"), "")

    def test_index_and_files_reader_expose_metadata_override_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo, _artifacts_dir, metadata_path = self._prepare_repo(td)
            metadata_path.write_text(
                json.dumps(
                    {
                        "serial_number": "SN-001",
                        "program_title": "Program Alpha",
                        "asset_type": "Valve",
                        "asset_specific_type": "Valve1",
                        "vendor": "Vendor A",
                        "document_type": "EIDP",
                        "document_type_acronym": "EIDP",
                        "metadata_source": "mixed",
                        "manual_override_fields": ["vendor"],
                        "manual_override_updated_at": "2026-03-24T00:00:00+00:00",
                        "applied_asset_specific_type_rule": "Valve1",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            support = repo / "EIDAT Support"
            support.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(support / "eidat_support.sqlite3"))
            try:
                conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY AUTOINCREMENT, rel_path TEXT NOT NULL)")
                conn.execute("INSERT INTO files(rel_path) VALUES (?)", ("source/doc1.pdf",))
                conn.commit()
            finally:
                conn.close()

            eidat_manager_index.build_index(support_paths(repo))
            docs = backend.read_eidat_index_documents(repo)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].get("metadata_source"), "mixed")
            self.assertEqual(docs[0].get("manual_override_fields"), ["vendor"])
            self.assertEqual(docs[0].get("applied_asset_specific_type_rule"), "Valve1")

            files = backend.read_files_with_index_metadata(repo)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].get("metadata_source"), "mixed")
            self.assertEqual(files[0].get("manual_override_fields"), "vendor")
            self.assertTrue(bool(files[0].get("has_manual_override")))


if __name__ == "__main__":
    unittest.main()
