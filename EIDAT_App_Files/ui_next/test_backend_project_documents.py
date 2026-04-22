import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ui_next import backend as be  # type: ignore


def _write_index(repo: Path, rows: list[dict[str, object]]) -> None:
    support = be.eidat_support_dir(repo)
    support.mkdir(parents=True, exist_ok=True)
    db_path = support / "eidat_index.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                program_title TEXT,
                asset_type TEXT,
                asset_specific_type TEXT,
                serial_number TEXT,
                part_number TEXT,
                revision TEXT,
                test_date TEXT,
                report_date TEXT,
                document_type TEXT,
                metadata_rel TEXT,
                artifacts_rel TEXT,
                similarity_group TEXT
            )
            """
        )
        for idx, row in enumerate(rows, start=1):
            conn.execute(
                """
                INSERT INTO documents (
                    id, program_title, asset_type, asset_specific_type, serial_number,
                    part_number, revision, test_date, report_date, document_type,
                    metadata_rel, artifacts_rel, similarity_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idx,
                    row.get("program_title") or "",
                    row.get("asset_type") or "",
                    row.get("asset_specific_type") or "",
                    row.get("serial_number") or "",
                    row.get("part_number") or "",
                    row.get("revision") or "",
                    row.get("test_date") or "",
                    row.get("report_date") or "",
                    row.get("document_type") or "",
                    row.get("metadata_rel") or "",
                    row.get("artifacts_rel") or "",
                    row.get("similarity_group") or "",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _write_project(repo: Path, *, selected: list[str] | None = None, serials: list[str] | None = None) -> tuple[Path, Path]:
    project_dir = be.eidat_projects_root(repo) / "Proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    workbook = project_dir / "Proj.xlsx"
    workbook.write_text("placeholder", encoding="utf-8")
    meta = {
        "name": "Proj",
        "type": be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
        "global_repo": str(repo),
        "project_dir": str(project_dir),
        "workbook": str(workbook),
        "selected_metadata_rel": list(selected or ["docs/SN1.json"]),
        "serials": list(serials or ["SN1"]),
        "continued_population": {"program_title": ["Program A"]},
    }
    (project_dir / be.EIDAT_PROJECT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return project_dir, workbook


class TestProjectDocumentsBackend(unittest.TestCase):
    def test_report_discovery_matches_summary_sidecar_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_index(
                repo,
                [{"program_title": "Program A", "serial_number": "SN1", "metadata_rel": "docs/SN1.json"}],
            )
            project_dir, workbook = _write_project(repo)
            report_dir = be.edin_program_report_dir(repo, "Program A", create=True)
            good = report_dir / "good.pdf"
            bad = report_dir / "bad.pdf"
            good.write_bytes(b"%PDF good")
            bad.write_bytes(b"%PDF bad")
            good.with_suffix(".summary.json").write_text(
                json.dumps({"project_dir": str(project_dir), "workbook_path": str(workbook)}),
                encoding="utf-8",
            )
            bad.with_suffix(".summary.json").write_text(
                json.dumps({"project_dir": str(repo / "other"), "workbook_path": str(repo / "other.xlsx")}),
                encoding="utf-8",
            )

            items = be.list_project_report_items(repo, project_dir, workbook)

            names = {str(item.get("name")) for item in items}
            self.assertIn("good", names)
            self.assertNotIn("bad", names)

    def test_legacy_report_discovery_matches_selected_serial_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_index(
                repo,
                [{"program_title": "Program A", "serial_number": "SN1", "metadata_rel": "docs/SN1.json"}],
            )
            project_dir, workbook = _write_project(repo)
            report_dir = be.edin_program_report_dir(repo, "Program A", create=True)
            legacy = report_dir / "SN1_Test Data Report.pdf"
            unrelated = report_dir / "SN9_Test Data Report.pdf"
            legacy.write_bytes(b"%PDF legacy")
            unrelated.write_bytes(b"%PDF unrelated")

            items = be.list_project_report_items(repo, project_dir, workbook)

            names = {str(item.get("name")) for item in items}
            self.assertIn("SN1_Test Data Report", names)
            self.assertNotIn("SN9_Test Data Report", names)

    def test_report_rename_and_delete_move_summary_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            report_dir.mkdir()
            pdf = report_dir / "old.pdf"
            sidecar = pdf.with_suffix(".summary.json")
            pdf.write_bytes(b"%PDF old")
            sidecar.write_text("{}", encoding="utf-8")

            renamed = be.rename_project_managed_file(
                pdf,
                "new",
                allowed_dir=report_dir,
                rename_summary_sidecar=True,
            )

            new_pdf = report_dir / "new.pdf"
            new_sidecar = new_pdf.with_suffix(".summary.json")
            self.assertEqual(Path(str(renamed.get("path"))), new_pdf)
            self.assertTrue(new_pdf.exists())
            self.assertTrue(new_sidecar.exists())
            self.assertFalse(pdf.exists())
            self.assertFalse(sidecar.exists())

            deleted = be.delete_project_managed_file(
                new_pdf,
                allowed_dir=report_dir,
                delete_summary_sidecar=True,
            )

            self.assertTrue(deleted.get("deleted"))
            self.assertTrue(deleted.get("sidecar_deleted"))
            self.assertFalse(new_pdf.exists())
            self.assertFalse(new_sidecar.exists())

    def test_td_graph_definition_mutations_preserve_graph_files_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            store = project_dir / "auto_plots_test_data.json"
            store.write_text(
                json.dumps({"version": 3, "graph_files": [{"id": "g1", "name": "Old", "plots": []}]}),
                encoding="utf-8",
            )

            items = be.list_project_graph_items(project_dir, be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING)
            self.assertEqual(items[0].get("graph_key"), "id:g1")
            be.rename_project_graph_definition(
                project_dir,
                be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
                "id:g1",
                "New",
            )
            payload = json.loads(store.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("version"), 3)
            self.assertEqual(payload.get("graph_files")[0].get("name"), "New")
            self.assertTrue(payload.get("graph_files")[0].get("updated_at"))

            be.delete_project_graph_definition(project_dir, be.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING, "id:g1")
            payload = json.loads(store.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("version"), 3)
            self.assertEqual(payload.get("graph_files"), [])

    def test_eidp_graph_definition_mutations_preserve_list_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            store = project_dir / "auto_plots.json"
            store.write_text(json.dumps([{"id": "g1", "name": "Old"}]), encoding="utf-8")

            be.rename_project_graph_definition(project_dir, be.EIDAT_PROJECT_TYPE_TRENDING, "id:g1", "New")
            payload = json.loads(store.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertEqual(payload[0].get("name"), "New")

            be.delete_project_graph_definition(project_dir, be.EIDAT_PROJECT_TYPE_TRENDING, "id:g1")
            payload = json.loads(store.read_text(encoding="utf-8"))
            self.assertEqual(payload, [])

    def test_graph_pdf_discovery_excludes_report_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            graph_pdf = project_dir / "trend_plot.pdf"
            report_pdf = project_dir / "report.pdf"
            graph_pdf.write_bytes(b"%PDF graph")
            report_pdf.write_bytes(b"%PDF report")
            report_pdf.with_suffix(".summary.json").write_text("{}", encoding="utf-8")

            items = be.list_project_graph_items(project_dir, be.EIDAT_PROJECT_TYPE_TRENDING)

            names = {str(item.get("name")) for item in items}
            self.assertIn("trend_plot", names)
            self.assertNotIn("report", names)


if __name__ == "__main__":
    unittest.main()
