import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from Production import node_backend  # noqa: E402


class TestNodeBackendUpdateProject(unittest.TestCase):
    def test_update_project_routes_all_project_types_and_td_flags(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            repo = root / "node"
            repo.mkdir(parents=True, exist_ok=True)
            wb = repo / "projects" / "proj.xlsx"
            wb.parent.mkdir(parents=True, exist_ok=True)
            wb.write_text("", encoding="utf-8")

            trending = Mock(return_value={"kind": "trending"})
            raw = Mock(return_value={"kind": "raw"})
            td_update = Mock(return_value={"kind": "td"})
            fake_be = SimpleNamespace(
                EIDAT_PROJECT_TYPE_TRENDING="EIDP Trending",
                EIDAT_PROJECT_TYPE_RAW_TRENDING="EIDP Raw File Trending",
                EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING="Test Data Trending",
                eidat_projects_root=lambda _repo: repo / "EIDAT Support" / "projects",
                update_eidp_trending_project_workbook=trending,
                update_eidp_raw_trending_project_workbook=raw,
                update_test_data_trending_project_workbook=td_update,
            )
            progress_cb = lambda _line: None

            with patch.object(node_backend, "_be", return_value=fake_be), patch.object(node_backend, "global_repo", return_value=repo):
                trending_payload = node_backend.update_project(
                    repo,
                    workbook_path=wb,
                    project_type="EIDP Trending",
                    overwrite=True,
                    force_project_rebuild=True,
                    progress_cb=progress_cb,
                )
                raw_payload = node_backend.update_project(
                    repo,
                    workbook_path=wb,
                    project_type="EIDP Raw File Trending",
                    overwrite=False,
                    force_project_rebuild=True,
                    progress_cb=progress_cb,
                )
                td_payload = node_backend.update_project(
                    repo,
                    workbook_path=wb,
                    project_type="Test Data Trending",
                    overwrite=True,
                    force_project_rebuild=True,
                    progress_cb=progress_cb,
                )

            self.assertEqual(trending_payload["kind"], "trending")
            self.assertEqual(raw_payload["kind"], "raw")
            self.assertEqual(td_payload["kind"], "td")
            trending.assert_called_once_with(repo, wb, overwrite=True)
            raw.assert_called_once_with(repo, wb, overwrite=False)
            td_update.assert_called_once_with(
                repo,
                wb,
                overwrite=True,
                include_performance_sheets=True,
                source_refresh_mode="smart",
                force_project_rebuild=True,
                progress_cb=progress_cb,
            )

    def test_update_project_rewrites_stale_project_meta_to_node_root(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            repo = root / "node"
            runtime_root = root / "runtime"
            project_dir = repo / "EIDAT" / "EIDAT Support" / "projects" / "ProjA"
            workbook = project_dir / "ProjA.xlsx"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook.write_text("", encoding="utf-8")
            meta_path = project_dir / "project.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "name": "ProjA",
                        "global_repo": str(runtime_root),
                        "project_dir": str(runtime_root / "EIDAT Support" / "projects" / "ProjA"),
                        "workbook": str(runtime_root / "EIDAT Support" / "projects" / "ProjA" / "ProjA.xlsx"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            trending = Mock(return_value={"kind": "trending"})
            fake_be = SimpleNamespace(
                EIDAT_PROJECT_TYPE_TRENDING="EIDP Trending",
                EIDAT_PROJECT_TYPE_RAW_TRENDING="EIDP Raw File Trending",
                EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING="Test Data Trending",
                eidat_projects_root=lambda _repo: repo / "EIDAT" / "EIDAT Support" / "projects",
                update_eidp_trending_project_workbook=trending,
                update_eidp_raw_trending_project_workbook=Mock(),
                update_test_data_trending_project_workbook=Mock(),
            )

            with patch.object(node_backend, "_be", return_value=fake_be), patch.object(node_backend, "global_repo", return_value=repo):
                node_backend.update_project(
                    repo,
                    workbook_path=runtime_root / "EIDAT Support" / "projects" / "ProjA" / "ProjA.xlsx",
                    project_type="EIDP Trending",
                )

            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(str(payload.get("global_repo") or ""), str(repo))
            self.assertEqual(str(payload.get("project_dir") or ""), str(project_dir))
            self.assertEqual(str(payload.get("workbook") or ""), str(workbook))
            trending.assert_called_once_with(repo, workbook, overwrite=False)


if __name__ == "__main__":
    unittest.main()
