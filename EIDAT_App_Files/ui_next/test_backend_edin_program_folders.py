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


class TestBackendEdinProgramFolders(unittest.TestCase):
    def test_eidat_support_dir_prefers_deepest_populated_support_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            shallow = repo / "EIDAT" / "EIDAT Support"
            deep = repo / "EIDAT" / "EIDAT" / "EIDAT Support"
            shallow.mkdir(parents=True, exist_ok=True)
            deep.mkdir(parents=True, exist_ok=True)
            (shallow / "logs").mkdir(parents=True, exist_ok=True)
            (deep / "eidat_support.sqlite3").write_text("", encoding="utf-8")
            (deep / "projects").mkdir(parents=True, exist_ok=True)
            (deep / "projects" / "projects_registry.sqlite3").write_text("", encoding="utf-8")

            resolved = backend.eidat_support_dir(repo)

            self.assertEqual(resolved, deep)

    def test_eidat_support_dir_defaults_to_lowest_existing_eidat_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            (repo / "EIDAT" / "EIDAT").mkdir(parents=True, exist_ok=True)

            resolved = backend.eidat_support_dir(repo)

            self.assertEqual(resolved, repo / "EIDAT" / "EIDAT" / "EIDAT Support")

    def test_sync_edin_program_folders_creates_program_and_report_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir(parents=True, exist_ok=True)
            titles = ["Program Alpha", "Program Beta/One"]
            with patch.object(backend, "list_eidat_program_titles", return_value=titles):
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

    def test_list_eidat_program_titles_reads_distinct_titles_from_index_db(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            support = repo / "EIDAT" / "EIDAT" / "EIDAT Support"
            support.mkdir(parents=True, exist_ok=True)
            db_path = support / "eidat_index.sqlite3"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute("CREATE TABLE documents (program_title TEXT)")
                conn.executemany(
                    "INSERT INTO documents(program_title) VALUES(?)",
                    [
                        ("Program Beta",),
                        ("Program Alpha",),
                        ("Program Beta",),
                        (" ",),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            titles = backend.list_eidat_program_titles(repo)

            self.assertEqual(titles, ["Program Alpha", "Program Beta"])

    def test_td_auto_report_default_filename_uses_certification_serials(self) -> None:
        name = backend.td_auto_report_default_filename(
            "Program Alpha",
            ["SN-001", "SN-002", "SN-003", "SN-004"],
            when=backend.datetime(2026, 4, 9, 12, 0, 0),
        )
        self.assertEqual(
            name,
            "SN-001__SN-002__SN-003__plus_1_more_Test Data Report.pdf",
        )

    def test_get_repo_root_prefers_active_node_root_over_configured_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_root = root / "data"
            user_inputs = data_root / "user_inputs"
            node_root = root / "node_root"
            node_root.mkdir(parents=True, exist_ok=True)
            user_inputs.mkdir(parents=True, exist_ok=True)
            scanner_env = user_inputs / "scanner.env"
            scanner_local = user_inputs / "scanner.local.env"
            dot_env = data_root / ".env"
            scanner_env.write_text("REPO_ROOT=C:\\shared\\runtime\\repo\n", encoding="utf-8")
            scanner_local.write_text("REPO_ROOT=C:\\node\\override\\repo\n", encoding="utf-8")
            dot_env.write_text("REPO_ROOT=C:\\stale\\dotenv\\repo\n", encoding="utf-8")

            with patch.dict(os.environ, {"EIDAT_NODE_ROOT": str(node_root)}, clear=False), patch.object(
                backend, "SCANNER_ENV", scanner_env
            ), patch.object(backend, "SCANNER_ENV_LOCAL", scanner_local), patch.object(
                backend, "DOTENV_FILES", [dot_env]
            ), patch.object(
                backend, "ensure_repo_root_name_file", return_value=""
            ):
                self.assertEqual(backend.get_repo_root(), node_root.resolve())


if __name__ == "__main__":
    unittest.main()
