import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtCore, QtWidgets
    from Production import admin_db
    from Production.admin_gui import AdminWindow, _ProjectUpdatePickerDialog
except Exception:  # pragma: no cover - optional dependency guard
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    admin_db = None  # type: ignore[assignment]
    AdminWindow = None  # type: ignore[assignment]
    _ProjectUpdatePickerDialog = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or AdminWindow is None or admin_db is None, "PySide6 is required")
class TestAdminGuiUpdateProjects(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_project_picker_loads_projects_and_exposes_force_rebuild(self) -> None:
        dlg = _ProjectUpdatePickerDialog(
            node_name="NodeA",
            projects=[
                {"name": "Proj 1", "type": "EIDP Trending", "workbook": "C:/tmp/p1.xlsx"},
                {"name": "Proj 2", "type": "Test Data Trending", "workbook": "C:/tmp/p2.xlsx"},
            ],
        )
        try:
            self.assertEqual(dlg.tbl.rowCount(), 2)
            self.assertFalse(dlg.cb_force_rebuild.isChecked())
            dlg.tbl.selectRow(1)
            selected = dlg.selected_projects()
            self.assertEqual(len(selected), 1)
            self.assertEqual(str(selected[0].get("name") or ""), "Proj 2")
            self.assertIn("Test Data Trending", dlg.cb_force_rebuild.toolTip())
        finally:
            dlg.close()
            dlg.deleteLater()

    def test_act_update_projects_passes_batch_payload_to_worker(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            registry = root / "registry.sqlite3"
            node_root = root / "node"
            node_root.mkdir(parents=True, exist_ok=True)

            window = AdminWindow(runtime_root=runtime_root, registry_path=registry)
            try:
                node_id = admin_db.upsert_node(window._conn, node_root=str(node_root), runtime_root=str(runtime_root), enabled=True)
                admin_db.set_node_env_enabled(window._conn, node_id=node_id, enabled=True)
                window.refresh()
                captured: list[tuple] = []

                with patch.object(
                    window,
                    "_choose_projects_for_update",
                    return_value=([{"name": "Proj 1", "type": "EIDP Trending", "workbook": str(node_root / "projects" / "p1.xlsx")}], True, True),
                ), patch.object(window.worker, "set_tasks", side_effect=lambda tasks: captured.extend(tasks)), patch.object(
                    window.worker,
                    "start",
                    return_value=None,
                ), patch.object(window._run_status, "begin", return_value=None):
                    window._act_update_projects(node_id)

                self.assertEqual(len(captured), 1)
                task = captured[0]
                self.assertEqual(task[0], node_id)
                self.assertEqual(task[4], "update_projects")
                self.assertTrue(bool(task[5]["overwrite"]))
                self.assertTrue(bool(task[5]["force_project_rebuild"]))
                self.assertEqual(len(list(task[5]["projects"])), 1)
            finally:
                window.close()

    def test_on_finished_one_records_update_projects_run_summary(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            runtime_root = root / "runtime"
            registry = root / "registry.sqlite3"
            node_root = root / "node"
            node_root.mkdir(parents=True, exist_ok=True)

            window = AdminWindow(runtime_root=runtime_root, registry_path=registry)
            try:
                node_id = admin_db.upsert_node(window._conn, node_root=str(node_root), runtime_root=str(runtime_root), enabled=True)
                summary = {
                    "selected_count": 2,
                    "succeeded_count": 1,
                    "failed_count": 1,
                    "overwrite": True,
                    "force_project_rebuild": False,
                    "projects_sample": [{"name": "Proj A", "ok": True}, {"name": "Proj B", "ok": False}],
                }

                window._on_finished_one(node_id, "update_projects", 10, 20, False, "1 project update(s) failed out of 2.", summary)

                row = window._conn.execute(
                    "SELECT status, summary_json, error FROM runs WHERE node_id = ? ORDER BY started_epoch_ns DESC LIMIT 1",
                    (node_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(str(row["status"]), "failed")
                payload = json.loads(str(row["summary_json"]))
                self.assertEqual(int(payload.get("selected_count") or 0), 2)
                self.assertEqual(int(payload.get("failed_count") or 0), 1)
                self.assertEqual(str(payload.get("action") or ""), "update_projects")
                self.assertFalse(bool(payload.get("ok")))
                self.assertIn("failed out of 2", str(row["error"] or ""))
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
