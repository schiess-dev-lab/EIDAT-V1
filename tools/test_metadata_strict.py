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


def _write_doc_type_strategies(data_root: Path, payload: dict) -> None:
    ui = data_root / "user_inputs"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "document_type_strategies.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _strategy_payload() -> dict:
    return {
        "version": 1,
        "document_types": [
            {"name": "End Item Data Package", "acronym": "EIDP"},
            {"name": "Test Data", "acronym": "TD"},
        ],
        "filename_aliases": {
            "EIDP": ["EIDP", "End Item Data Package"],
            "TD": ["TD", "Test Data", "Hot Fire Test Data", "Hot Fire Test"],
        },
        "content_aliases": {
            "EIDP": ["EIDP", "End Item Data Package"],
            "TD": ["TD", "Test Data", "Hot Fire Test Data", "Hot Fire Test"],
        },
        "extension_rules": {
            "EIDP": [".pdf"],
            "TD": [".xlsx", ".xls", ".xlsm", ".mat"],
        },
        "folder_rules": {
            "levels": 3,
            "aliases": {
                "EIDP": ["EIDP", "End Item Data Package"],
                "TD": ["TD", "Test Data", "Hot Fire Test Data", "Hot Fire Test"],
            },
        },
        "serial_patterns": ["(?i)\\bSN[-_ ]?[A-Z0-9]+(?:[-_][A-Z0-9]+)*\\b"],
        "ranker": {
            "weights": {
                "content": 5,
                "folder": 3,
                "extension_compatible": 1,
                "serial_bonus": 2,
            },
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
    }


class TestStrictMetadataAllowlists(unittest.TestCase):
    def setUp(self) -> None:
        self._old_data_root = os.environ.get("EIDAT_DATA_ROOT")

    def tearDown(self) -> None:
        if self._old_data_root is None:
            os.environ.pop("EIDAT_DATA_ROOT", None)
        else:
            os.environ["EIDAT_DATA_ROOT"] = self._old_data_root

    def test_program_title_strict_unknown_when_not_allowlisted(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Program: Star Link"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "docs" / "x.pdf")
            self.assertEqual(meta.get("program_title"), "Starlink")

    def test_part_number_allowlist_match_normalized(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"], "part_numbers": ["VLV-42A"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Valve model: VLV 42A"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("part_number"), "VLV-42A")

    def test_part_number_unknown_when_allowlist_missing(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            text = "\n".join(["=== Page 1 ===", "Valve model: VLV 42A"])
            meta = emd.extract_metadata_from_text(text, pdf_path=Path(td) / "Starlink" / "x.pdf")
            self.assertEqual(meta.get("asset_specific_type"), "VLV-42A")

    def test_report_date_proximity_pages_1_to_3(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
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

    def test_document_type_folder_only_td_is_review_required_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Test Data" / "x.pdf"
            meta = emd.extract_metadata_from_text("=== Page 1 ===\n(no doc type)\n", pdf_path=p)
            self.assertEqual(meta.get("document_type"), "Unknown")
            self.assertTrue(bool(meta.get("document_type_review_required")))

    def test_document_type_filename_pdf_confirms_eidp(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_EIDP.pdf"
            meta = emd.extract_metadata_from_text("=== Page 1 ===\n(no doc type)\n", pdf_path=p)
            self.assertEqual(meta.get("document_type"), "EIDP")
            self.assertEqual(meta.get("document_type_status"), "confirmed")

    def test_document_type_content_confirms_eidp(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_unknown.pdf"
            text = "\n".join(["=== Page 1 ===", "Valve End Item Data Package (EIDP)"])
            meta = emd.extract_metadata_from_text(text, pdf_path=p)
            self.assertEqual(meta.get("document_type"), "EIDP")
            self.assertEqual(meta.get("document_type_source"), "ranker")

    def test_document_type_conflict_requires_review(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_EIDP.pdf"
            text = "\n".join(["=== Page 1 ===", "Hot Fire Test Data"])
            meta = emd.extract_metadata_from_text(text, pdf_path=p)
            self.assertEqual(meta.get("document_type"), "Unknown")
            self.assertEqual(meta.get("document_type_status"), "ambiguous")
            self.assertTrue(bool(meta.get("document_type_review_required")))

    def test_document_type_excel_folder_and_serial_confirms_td(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Hot Fire Test Data" / "SN4001_results.xlsx"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="Unknown")
            self.assertEqual(meta.get("document_type"), "TD")
            self.assertEqual(meta.get("document_type_reason"), "folder_serial_rule")

    def test_document_type_excel_folder_without_serial_stays_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Hot Fire Test Data" / "results.xlsx"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="Unknown")
            self.assertEqual(meta.get("document_type"), "Unknown")
            self.assertTrue(bool(meta.get("document_type_review_required")))

    def test_document_type_extension_only_pdf_is_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_report.pdf"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="Unknown")
            self.assertEqual(meta.get("document_type"), "Unknown")
            self.assertEqual(meta.get("document_type_reason"), "extension_only_insufficient")

    def test_document_type_mat_is_confirmed_td(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(data_root, {"program_titles": ["Starlink"]})
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "capture.mat"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="Unknown")
            self.assertEqual(meta.get("document_type"), "TD")
            self.assertEqual(meta.get("document_type_reason"), "mat_extension_match")

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
            _write_doc_type_strategies(data_root, _strategy_payload())
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
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "Pump" / "x.pdf"
            meta = emd.canonicalize_metadata_for_file(p, existing_meta=None, extracted_meta={}, default_document_type="EIDP")
            self.assertEqual(meta.get("asset_type"), "Pump")

    def test_asset_specific_type_rule_sets_asset_type_and_vendor(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "asset_types": ["Valve"],
                    "asset_specific_types": ["Valve1"],
                    "vendors": ["MOOG"],
                    "asset_specific_type_rules": [
                        {"asset_specific_type": "Valve1", "asset_type": "Valve", "vendor": "MOOG"}
                    ],
                },
            )
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_report.pdf"
            meta = emd.canonicalize_metadata_for_file(
                p,
                existing_meta=None,
                extracted_meta={"asset_specific_type": "Valve1"},
                default_document_type="EIDP",
            )
            self.assertEqual(meta.get("asset_type"), "Valve")
            self.assertEqual(meta.get("vendor"), "MOOG")
            self.assertEqual(meta.get("metadata_source"), "heuristic")
            self.assertEqual(meta.get("applied_asset_specific_type_rule"), "Valve1")

    def test_manual_override_fields_survive_canonicalize_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "asset_types": ["Valve"],
                    "asset_specific_types": ["Valve1"],
                    "vendors": ["MOOG", "ACME"],
                },
            )
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_report.pdf"
            meta = emd.canonicalize_metadata_for_file(
                p,
                existing_meta={
                    "vendor": "MOOG",
                    "manual_override_fields": ["vendor"],
                    "manual_override_updated_at": "2026-03-24T00:00:00+00:00",
                },
                extracted_meta={"vendor": "ACME"},
                default_document_type="EIDP",
            )
            self.assertEqual(meta.get("vendor"), "MOOG")
            self.assertEqual(meta.get("manual_override_fields"), ["vendor"])
            self.assertEqual(meta.get("metadata_source"), "mixed")

    def test_canonicalize_overwrite_manual_fields_clears_locks(self):
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            _write_candidates(
                data_root,
                {
                    "program_titles": ["Starlink"],
                    "asset_types": ["Valve"],
                    "asset_specific_types": ["Valve1"],
                    "vendors": ["MOOG", "ACME"],
                },
            )
            _write_doc_type_strategies(data_root, _strategy_payload())
            os.environ["EIDAT_DATA_ROOT"] = str(data_root)

            p = Path(td) / "Starlink" / "SN4001_report.pdf"
            meta = emd.canonicalize_metadata_for_file(
                p,
                existing_meta={
                    "vendor": "MOOG",
                    "manual_override_fields": ["vendor"],
                    "manual_override_updated_at": "2026-03-24T00:00:00+00:00",
                },
                extracted_meta={"vendor": "ACME"},
                default_document_type="EIDP",
                overwrite_manual_fields=True,
            )
            self.assertEqual(meta.get("vendor"), "ACME")
            self.assertEqual(meta.get("manual_override_fields"), [])
            self.assertEqual(meta.get("manual_override_updated_at"), "")


if __name__ == "__main__":
    unittest.main()
