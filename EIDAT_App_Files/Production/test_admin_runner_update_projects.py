import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from Production.admin_runner import run_update_projects  # noqa: E402


class TestAdminRunnerUpdateProjects(unittest.TestCase):
    def test_run_update_projects_batches_successful_updates(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            (runtime_root / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)
            wb1 = node_root / "projects" / "p1.xlsx"
            wb2 = node_root / "projects" / "p2.xlsx"
            wb1.parent.mkdir(parents=True, exist_ok=True)
            wb1.write_text("", encoding="utf-8")
            wb2.write_text("", encoding="utf-8")

            calls: list[tuple[str, str, bool, bool]] = []
            progress_lines: list[str] = []

            def _fake_update(node_root_arg, *, workbook_path, project_type, overwrite=False, force_project_rebuild=False, progress_cb=None):
                calls.append((str(node_root_arg), str(workbook_path), bool(overwrite), bool(force_project_rebuild)))
                if callable(progress_cb):
                    progress_cb(f"updating {Path(workbook_path).name}")
                return {"updated_cells": 7, "project_type": project_type}

            logs: list[str] = []
            # The placeholder workbooks are 0 bytes; stub the file-health probe so the
            # runner does not (correctly) divert the TD project to a rebuild.
            with patch("Production.node_backend.update_project", side_effect=_fake_update), patch(
                "Production.node_backend.inspect_project_files", return_value={}
            ):
                result = run_update_projects(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    projects=[
                        {"name": "Proj 1", "type": "EIDP Trending", "workbook": str(wb1)},
                        {"name": "Proj 2", "type": "Test Data Trending", "workbook": str(wb2)},
                    ],
                    overwrite=True,
                    force_project_rebuild=True,
                    on_log=lambda line: logs.append(str(line)),
                )

            self.assertTrue(result.ok)
            summary = dict(result.outputs.get("update_projects") or {})
            self.assertEqual(int(summary.get("selected_count") or 0), 2)
            self.assertEqual(int(summary.get("succeeded_count") or 0), 2)
            self.assertEqual(int(summary.get("failed_count") or 0), 0)
            self.assertTrue(bool(summary.get("overwrite")))
            self.assertTrue(bool(summary.get("force_project_rebuild")))
            self.assertEqual(len(calls), 2)
            self.assertTrue(any("updating p1.xlsx" in line for line in logs))
            self.assertTrue(any("updating p2.xlsx" in line for line in logs))

    def test_run_update_projects_continues_after_failure(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            (runtime_root / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)
            wb1 = node_root / "projects" / "bad.xlsx"
            wb2 = node_root / "projects" / "good.xlsx"
            wb1.parent.mkdir(parents=True, exist_ok=True)
            wb1.write_text("", encoding="utf-8")
            wb2.write_text("", encoding="utf-8")

            seen: list[str] = []

            def _fake_update(_node_root, *, workbook_path, **_kwargs):
                seen.append(Path(workbook_path).name)
                if Path(workbook_path).name == "bad.xlsx":
                    raise RuntimeError("boom")
                return {"updated_cells": 2}

            with patch("Production.node_backend.update_project", side_effect=_fake_update):
                result = run_update_projects(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    projects=[
                        {"name": "Bad", "type": "EIDP Trending", "workbook": str(wb1)},
                        {"name": "Good", "type": "EIDP Trending", "workbook": str(wb2)},
                    ],
                )

            self.assertFalse(result.ok)
            self.assertEqual(seen, ["bad.xlsx", "good.xlsx"])
            summary = dict(result.outputs.get("update_projects") or {})
            self.assertEqual(int(summary.get("succeeded_count") or 0), 1)
            self.assertEqual(int(summary.get("failed_count") or 0), 1)
            projects = list(summary.get("projects") or [])
            self.assertEqual(len(projects), 2)
            self.assertFalse(bool(projects[0].get("ok")))
            self.assertTrue(bool(projects[1].get("ok")))

    def test_run_update_projects_auto_rebuilds_td_project_on_corrupt_workbook(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            (runtime_root / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)
            wb = node_root / "projects" / "td.xlsx"
            wb.parent.mkdir(parents=True, exist_ok=True)
            wb.write_text("", encoding="utf-8")

            update_calls: list[str] = []
            rebuild_calls: list[dict] = []

            def _fake_update(_node_root, *, workbook_path, **_kwargs):
                update_calls.append(Path(workbook_path).name)
                raise RuntimeError("zipfile.BadZipFile: File is not a zip file")

            def _fake_rebuild(_node_root, *, workbook_path, project_type, overwrite=False, regenerate_workbook=False, progress_cb=None):
                rebuild_calls.append({"workbook": Path(workbook_path).name, "regenerate_workbook": bool(regenerate_workbook)})
                return {"updated_cells": 3, "rebuild": {"purged_cache_files": ["a", "b"], "workbook_regenerated": True}}

            logs: list[str] = []
            with patch("Production.node_backend.update_project", side_effect=_fake_update), patch(
                "Production.node_backend.rebuild_project", side_effect=_fake_rebuild
            ), patch("Production.node_backend.inspect_project_files", return_value={}), patch(
                "Production.node_backend.list_project_serial_issues", return_value=[]
            ):
                result = run_update_projects(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    projects=[{"name": "TD", "type": "Test Data Trending", "workbook": str(wb)}],
                    on_log=lambda line: logs.append(str(line)),
                )

            self.assertTrue(result.ok)
            self.assertEqual(update_calls, ["td.xlsx"])
            self.assertEqual(rebuild_calls, [{"workbook": "td.xlsx", "regenerate_workbook": True}])
            summary = dict(result.outputs.get("update_projects") or {})
            self.assertEqual(int(summary.get("succeeded_count") or 0), 1)
            self.assertEqual(int(summary.get("rebuilt_count") or 0), 1)
            project = list(summary.get("projects") or [])[0]
            self.assertTrue(bool(project.get("rebuilt")))
            self.assertIn("update_error_before_rebuild", dict(project.get("payload") or {}))
            self.assertTrue(any("[REBUILD] TD" in line for line in logs))

    def test_run_update_projects_pre_diverts_td_project_with_corrupt_cache_db(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            (runtime_root / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)
            wb = node_root / "projects" / "td.xlsx"
            wb.parent.mkdir(parents=True, exist_ok=True)
            wb.write_text("", encoding="utf-8")

            update_calls: list[str] = []
            rebuild_calls: list[str] = []

            with patch(
                "Production.node_backend.update_project",
                side_effect=lambda *_a, **k: update_calls.append(Path(k["workbook_path"]).name) or {"updated_cells": 1},
            ), patch(
                "Production.node_backend.rebuild_project",
                side_effect=lambda *_a, **k: rebuild_calls.append(Path(k["workbook_path"]).name) or {"updated_cells": 1, "rebuild": {}},
            ), patch(
                "Production.node_backend.inspect_project_files",
                return_value={"workbook_problem": "", "cache_db_problem": "corrupt (file is not a database)"},
            ):
                result = run_update_projects(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    projects=[{"name": "TD", "type": "Test Data Trending", "workbook": str(wb)}],
                )
            self.assertTrue(result.ok)
            self.assertEqual(update_calls, [])
            self.assertEqual(rebuild_calls, ["td.xlsx"])

            # auto_rebuild_on_failure=False keeps the old behaviour (plain update).
            update_calls.clear()
            rebuild_calls.clear()
            with patch(
                "Production.node_backend.update_project",
                side_effect=lambda *_a, **k: update_calls.append(Path(k["workbook_path"]).name) or {"updated_cells": 1},
            ), patch(
                "Production.node_backend.rebuild_project",
                side_effect=lambda *_a, **k: rebuild_calls.append(Path(k["workbook_path"]).name) or {},
            ), patch(
                "Production.node_backend.inspect_project_files",
                return_value={"cache_db_problem": "corrupt (file is not a database)"},
            ), patch("Production.node_backend.list_project_serial_issues", return_value=[]):
                result = run_update_projects(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    projects=[{"name": "TD", "type": "Test Data Trending", "workbook": str(wb)}],
                    auto_rebuild_on_failure=False,
                )
            self.assertTrue(result.ok)
            self.assertEqual(update_calls, ["td.xlsx"])
            self.assertEqual(rebuild_calls, [])

    def test_run_update_projects_reports_invalid_workbook_path(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            node_root = root / "node"
            other_root = root / "other"
            (runtime_root / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)
            node_root.mkdir(parents=True, exist_ok=True)
            other_root.mkdir(parents=True, exist_ok=True)
            wb = other_root / "outside.xlsx"
            wb.write_text("", encoding="utf-8")

            result = run_update_projects(
                node_root=node_root,
                runtime_root=runtime_root,
                projects=[{"name": "Outside", "type": "EIDP Trending", "workbook": str(wb)}],
            )

            self.assertFalse(result.ok)
            summary = dict(result.outputs.get("update_projects") or {})
            self.assertEqual(int(summary.get("selected_count") or 0), 1)
            self.assertEqual(int(summary.get("failed_count") or 0), 1)
            project = list(summary.get("projects") or [])[0]
            self.assertIn("inside the node root", str(project.get("error") or ""))


if __name__ == "__main__":
    unittest.main()
