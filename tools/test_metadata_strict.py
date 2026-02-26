import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "EIDAT_App_Files" / "Application"
sys.path.insert(0, str(APP_DIR))

import eidat_manager_metadata as emd  # noqa: E402


def _write_candidates(data_root: Path, payload: dict) -> None:
    ui = data_root / "user_inputs"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "metadata_candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestStrictMetadataAllowlists(unittest.TestCase):
    def test_program_title_strict_unknown_when_not_allowlisted(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(
                [
                    "=== Page 1 ===",
                    "Program Title: pulse scheme",
                ]
            )
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "docs" / "x.pdf")
            self.assertEqual(meta.get("program_title"), "Unknown")

    def test_program_title_directory_fallback(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as repo_td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(repo_td) / "Starlink" / "sub" / "file.pdf"
            meta = emd.extract_metadata_from_text("=== Page 1 ===\n(no title)\n", pdf_path=p)
            self.assertEqual(meta.get("program_title"), "Starlink")

    def test_program_title_alias_object_resolves_to_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {"program_titles": [{"name": "Starlink", "aliases": ["Star Link", "STAR-LINK"]}]},
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Program: Star Link"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "docs" / "x.pdf")
            self.assertEqual(meta.get("program_title"), "Starlink")

    def test_part_number_allowlist_match_normalized(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"], "part_numbers": ["VLV-42A"]})
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Valve model: VLV 42A"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("part_number"), "VLV-42A")

    def test_part_number_unknown_when_allowlist_missing(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Valve model: VLV-42A"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("part_number"), "Unknown")

    def test_acceptance_test_plan_allowlist_match_normalized(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "acceptance_test_plan_numbers": ["ATP-1234"],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Acceptance Test Plan: ATP 1234"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("acceptance_test_plan_number"), "ATP-1234")

    def test_vendor_alias_object_resolves_to_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "vendors": [{"name": "MOOG", "aliases": ["MOOG INC", "MOOG, INC."]}],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Manufacturer: MOOG INC"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("vendor"), "MOOG")

    def test_asset_specific_type_resolves_to_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "asset_specific_types": [{"name": "VLV-42A", "aliases": ["VLV 42A", "VLV_42A"]}],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Valve model: VLV 42A"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("asset_specific_type"), "VLV-42A")

    def test_report_date_proximity_pages_1_to_3(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(
                [
                    "=== Page 1 ===",
                    "Some header",
                    "=== Page 2 ===",
                    "Report Date: 01/16/2026",
                    "=== Page 4 ===",
                    "Report Date: 01/01/2025",
                ]
            )
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("report_date"), "2026-01-16")

    def test_report_date_fallback_top_of_page_1(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(
                [
                    "=== Page 1 ===",
                    "SN4001 | 2026-01-16",
                    "Other text",
                    "=== Page 2 ===",
                    "More stuff 01/01/2025",
                ]
            )
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("report_date"), "2026-01-16")

    def test_document_type_directory_fallback_td(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "document_types": [
                        {"name": "Test Data", "acronym": "TD", "aliases": ["Test Data", "TD"]},
                        {"name": "End Item Data Package", "acronym": "EIDP", "aliases": ["EIDP"]},
                    ],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Test Data" / "x.pdf"
            meta = emd.extract_metadata_from_text("=== Page 1 ===\n(no doc type)\n", pdf_path=p)
            self.assertEqual(meta.get("document_type"), "TD")
            self.assertEqual(meta.get("document_type_acronym"), "TD")

    def test_document_type_directory_fallback_eidp(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "document_types": [
                        {"name": "End Item Data Package", "acronym": "EIDP", "aliases": ["EIDP"]},
                    ],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "EIDP" / "x.pdf"
            meta = emd.extract_metadata_from_text("=== Page 1 ===\n(no doc type)\n", pdf_path=p)
            self.assertEqual(meta.get("document_type"), "EIDP")

    def test_canonicalize_document_type_from_path(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "document_types": [
                        {"name": "Test Data", "acronym": "TD", "aliases": ["TD"]},
                        {"name": "End Item Data Package", "acronym": "EIDP", "aliases": ["EIDP"]},
                    ],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "TD" / "x.pdf"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="EIDP")
            self.assertEqual(meta.get("document_type"), "TD")
            self.assertEqual(meta.get("document_type_acronym"), "TD")

    def test_asset_type_directory_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "asset_types": ["Valve", "Pump"],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Valve" / "x.pdf"
            meta = emd.extract_metadata_from_text("=== Page 1 ===\n(no asset)\n", pdf_path=p)
            self.assertEqual(meta.get("asset_type"), "Valve")

    def test_canonicalize_asset_type_from_path(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "asset_types": ["Valve", "Pump"],
                },
            )
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Pump" / "x.pdf"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="EIDP")
            self.assertEqual(meta.get("asset_type"), "Pump")


if __name__ == "__main__":
    unittest.main()
