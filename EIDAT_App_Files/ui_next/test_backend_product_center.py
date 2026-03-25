import json
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


class TestBackendProductCenter(unittest.TestCase):
    def _write_index_db(self, support_dir: Path) -> None:
        db_path = support_dir / "eidat_index.sqlite3"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(
                """
                CREATE TABLE documents (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  metadata_rel TEXT NOT NULL UNIQUE,
                  artifacts_rel TEXT,
                  program_title TEXT,
                  asset_type TEXT,
                  asset_specific_type TEXT,
                  serial_number TEXT,
                  part_number TEXT,
                  revision TEXT,
                  test_date TEXT,
                  report_date TEXT,
                  document_type TEXT,
                  document_type_acronym TEXT,
                  document_type_status TEXT,
                  document_type_source TEXT,
                  document_type_reason TEXT,
                  document_type_evidence_json TEXT,
                  document_type_review_required INTEGER,
                  vendor TEXT,
                  acceptance_test_plan_number TEXT,
                  excel_sqlite_rel TEXT,
                  tables_sqlite_rel TEXT,
                  file_extension TEXT,
                  title_norm TEXT,
                  similarity_group TEXT,
                  indexed_epoch_ns INTEGER NOT NULL,
                  certification_status TEXT,
                  certification_pass_rate TEXT
                );
                """
            )
            rows = [
                (
                    "docs/doc_a.json",
                    "debug/ocr/doc_a",
                    "Program Alpha",
                    "Pump",
                    "Pump Model",
                    "SN-001",
                    "PN-001",
                    "A",
                    "",
                    "",
                    "EIDP",
                    "EIDP",
                    "confirmed",
                    "",
                    "",
                    "",
                    0,
                    "Vendor A",
                    "ATP-001",
                    "",
                    "",
                    ".pdf",
                    "doc a",
                    "group-1",
                    1,
                    "",
                    "",
                ),
                (
                    "docs/doc_b.json",
                    "debug/ocr/doc_b",
                    "Program Alpha",
                    "Pump",
                    "Pump Model",
                    "SN-002",
                    "PN-002",
                    "A",
                    "",
                    "",
                    "TD",
                    "TD",
                    "confirmed",
                    "",
                    "",
                    "",
                    0,
                    "Vendor A",
                    "ATP-002",
                    "",
                    "",
                    ".pdf",
                    "doc b",
                    "group-1",
                    2,
                    "",
                    "",
                ),
                (
                    "docs/doc_blank.json",
                    "debug/ocr/doc_blank",
                    "Program Beta",
                    "",
                    "",
                    "SN-003",
                    "",
                    "",
                    "",
                    "",
                    "EIDP",
                    "EIDP",
                    "confirmed",
                    "",
                    "",
                    "",
                    0,
                    "",
                    "",
                    "",
                    "",
                    ".pdf",
                    "doc blank",
                    "group-2",
                    3,
                    "",
                    "",
                ),
            ]
            conn.executemany(
                """
                INSERT INTO documents(
                  metadata_rel, artifacts_rel, program_title, asset_type, asset_specific_type,
                  serial_number, part_number, revision, test_date, report_date, document_type,
                  document_type_acronym, document_type_status, document_type_source, document_type_reason,
                  document_type_evidence_json, document_type_review_required, vendor, acceptance_test_plan_number,
                  excel_sqlite_rel, tables_sqlite_rel, file_extension, title_norm, similarity_group,
                  indexed_epoch_ns, certification_status, certification_pass_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def _write_support_db(self, support_dir: Path) -> None:
        db_path = support_dir / "eidat_support.sqlite3"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(
                """
                CREATE TABLE files (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  rel_path TEXT NOT NULL,
                  last_processed_epoch_ns INTEGER,
                  needs_processing INTEGER
                );
                """
            )
            conn.executemany(
                "INSERT INTO files(rel_path, last_processed_epoch_ns, needs_processing) VALUES (?, ?, ?)",
                [
                    ("source/doc_a.pdf", 1, 0),
                    ("source/doc_b.pdf", 1, 0),
                    ("source/doc_blank.pdf", 1, 0),
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def _write_projects(self, repo: Path, support_dir: Path) -> None:
        projects_root = support_dir / "projects"
        project_dir = projects_root / "TD Project"
        project_dir.mkdir(parents=True, exist_ok=True)
        workbook = project_dir / "TD Project.xlsx"
        workbook.write_text("", encoding="utf-8")
        (project_dir / backend.EIDAT_PROJECT_META).write_text(
            json.dumps(
                {
                    "name": "TD Project",
                    "type": backend.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
                    "selected_metadata_rel": ["docs/doc_a.json"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (project_dir / backend.TD_SAVED_PERFORMANCE_EQUATIONS_JSON).write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "id": "flow_fit",
                            "name": "Flow Fit",
                            "saved_at": "2026-03-01 10:00:00",
                            "updated_at": "2026-03-02 10:00:00",
                            "plot_metadata": {
                                "output_target": "Pressure",
                                "input1_target": "Flow",
                                "asset_type": "Pump",
                                "asset_specific_type": "Pump Model",
                            },
                            "equation_rows": [{"stat": "mean", "equation": "y=mx+b"}],
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        db_path = support_dir / "projects" / backend.EIDAT_PROJECTS_REGISTRY_DB
        conn = backend._connect_projects_registry(db_path)
        try:
            backend._ensure_projects_registry_schema(conn)
            conn.execute(
                """
                INSERT INTO projects(name, type, project_dir, workbook, created_by, created_epoch_ns, updated_epoch_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "TD Project",
                    backend.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
                    str(project_dir.relative_to(repo)),
                    str(workbook.relative_to(repo)),
                    "tester",
                    1,
                    1,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_product_center_aggregates_products_projects_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            support_dir = repo / "EIDAT Support"
            ocr_root = support_dir / "debug" / "ocr"
            data_root = Path(tmpdir) / "data_root"
            ocr_root.mkdir(parents=True, exist_ok=True)
            for name in ("doc_a", "doc_b", "doc_blank"):
                (ocr_root / name).mkdir(parents=True, exist_ok=True)
            self._write_index_db(support_dir)
            self._write_support_db(support_dir)
            self._write_projects(repo, support_dir)
            with patch.object(backend, "DATA_ROOT", data_root):
                image_dir = backend.product_center_images_dir()
                (image_dir / "pump_model.png").write_bytes(b"png")

                products = backend.list_product_center_products(repo)

                pump = next(item for item in products if item["asset_specific_type"] == "Pump Model")
                blank = next(item for item in products if item["asset_specific_type"] == "(No Asset Specific Type)")

                self.assertEqual(pump["asset_type"], "Pump")
                self.assertEqual(pump["vendor"], "Vendor A")
                self.assertEqual(pump["counts"]["documents"], 2)
                self.assertEqual(pump["counts"]["eidp_documents"], 1)
                self.assertEqual(pump["counts"]["td_documents"], 1)
                self.assertEqual(pump["counts"]["projects"], 1)
                self.assertEqual(pump["part_numbers"], ["PN-001", "PN-002"])
                self.assertEqual(pump["acceptance_test_plan_numbers"], ["ATP-001", "ATP-002"])
                self.assertEqual(sorted(pump["serial_numbers"]), ["SN-001", "SN-002"])
                self.assertTrue(str(pump["image_path"]).endswith("pump_model.png"))
                self.assertEqual(pump["projects"][0]["name"], "TD Project")
                self.assertEqual(pump["saved_performance_equations"][0]["name"], "Flow Fit")
                self.assertEqual(pump["saved_performance_equations"][0]["summary"], "Pressure vs Flow")
                self.assertEqual(sorted(doc["rel_path"] for doc in pump["documents"]), ["source/doc_a.pdf", "source/doc_b.pdf"])

                self.assertEqual(blank["asset_type"], "(No Asset Type)")
                self.assertEqual(blank["vendor"], "(No Vendor)")
                self.assertEqual(blank["counts"]["documents"], 1)

    def test_resolve_product_center_image_uses_asset_specific_slug(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data_root"
            with patch.object(backend, "DATA_ROOT", data_root):
                image_dir = backend.product_center_images_dir()
                target = image_dir / "pump_model.jpeg"
                target.write_bytes(b"jpeg")
                resolved = backend.resolve_product_center_image("Pump Model")
                self.assertEqual(resolved, target)

    def test_resolve_product_center_image_matches_nested_asset_specific_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data_root"
            with patch.object(backend, "DATA_ROOT", data_root):
                image_dir = backend.product_center_images_dir()
                nested = image_dir / "Pump Model"
                nested.mkdir(parents=True, exist_ok=True)
                target = nested / "Hero Image.PNG"
                target.write_bytes(b"png")
                resolved = backend.resolve_product_center_image("Pump Model")
                self.assertEqual(resolved, target)
