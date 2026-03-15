import sys
import unittest
from pathlib import Path


UI_DIR = Path(__file__).resolve().parents[1] / "EIDAT_App_Files" / "ui_next"
sys.path.insert(0, str(UI_DIR))

import backend as be  # noqa: E402


class TestDocumentTypeBackendGating(unittest.TestCase):
    def test_is_test_data_doc_requires_confirmed_td_when_status_present(self):
        doc = {
            "document_type": "TD",
            "document_type_acronym": "TD",
            "document_type_status": "confirmed",
            "document_type_review_required": 0,
        }
        self.assertTrue(be.is_test_data_doc(doc))

    def test_is_test_data_doc_rejects_review_required_td(self):
        doc = {
            "document_type": "TD",
            "document_type_acronym": "TD",
            "document_type_status": "ambiguous",
            "document_type_review_required": 1,
        }
        self.assertFalse(be.is_test_data_doc(doc))

    def test_confirmed_doc_type_helper(self):
        doc = {
            "document_type": "EIDP",
            "document_type_acronym": "EIDP",
            "document_type_status": "confirmed",
            "document_type_review_required": 0,
        }
        self.assertTrue(be._is_confirmed_doc_type(doc, "EIDP"))  # type: ignore[attr-defined]
        self.assertFalse(be._is_confirmed_doc_type(doc, "TD"))  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
