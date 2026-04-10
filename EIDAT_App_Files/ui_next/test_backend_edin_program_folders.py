import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestBackendEdinProgramFolders(unittest.TestCase):
    def test_sync_edin_program_folders_creates_program_and_report_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            docs = [
                {"program_title": "Program Alpha"},
                {"program_title": "Program Beta/One"},
                {"program_title": "Program Alpha"},
            ]
            with patch.object(backend, "read_eidat_index_documents", return_value=docs):
                result = backend.sync_edin_program_folders(repo)

            root = repo / backend.EDIN_PROGRAM_FOLDERS_DIRNAME
            self.assertEqual(Path(result["root"]), root)
            self.assertTrue((root / "Program Alpha" / backend.EDIN_PROGRAM_REPORTS_DIRNAME).is_dir())
            self.assertTrue((root / "Program Beta One" / backend.EDIN_PROGRAM_REPORTS_DIRNAME).is_dir())
            self.assertEqual(sorted(entry["program_title"] for entry in result["programs"]), ["Program Alpha", "Program Beta/One"])

    def test_edin_program_folder_name_is_windows_safe_and_stable(self) -> None:
        self.assertEqual(backend.edin_program_folder_name('  CON<>:"/\\\\|?*  '), "CON_")
        self.assertEqual(backend.edin_program_folder_name("Program  Alpha"), "Program Alpha")
        self.assertEqual(backend.edin_program_folder_name("Program/Alpha"), "Program Alpha")

    def test_list_eidat_program_titles_falls_back_to_metadata_files_when_index_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            artifacts = repo / "EIDAT Support" / "debug" / "ocr" / "doc_a"
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "doc_a_metadata.json").write_text(
                json.dumps({"program_title": "Program Gamma"}, indent=2),
                encoding="utf-8",
            )

            titles = backend.list_eidat_program_titles(repo)

            self.assertEqual(titles, ["Program Gamma"])

    def test_td_auto_report_default_filename_uses_program_and_certification_serials(self) -> None:
        name = backend.td_auto_report_default_filename(
            "Program Alpha",
            ["SN-001", "SN-002", "SN-003", "SN-004"],
            when=backend.datetime(2026, 4, 9, 12, 0, 0),
        )
        self.assertEqual(
            name,
            "Program Alpha__SN-001__SN-002__SN-003__plus_1_more__TD_Auto_Report__2026-04-09.pdf",
        )


if __name__ == "__main__":
    unittest.main()
