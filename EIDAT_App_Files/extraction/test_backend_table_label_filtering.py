import sys
import unittest
from pathlib import Path


# Allow `import ui_next.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestTableLabelFiltering(unittest.TestCase):
    def test_extract_from_tables_filters_by_table_label(self) -> None:
        blocks = [
            {"heading": "H", "table_label": "Acceptance Test Data", "rows": [["term", "111"]], "kv": {"term": "111"}},
            {"heading": "H", "table_label": "Acceptance Test Data (2)", "rows": [["term", "222"]], "kv": {"term": "222"}},
        ]
        val, snip = backend._extract_from_tables(  # type: ignore[attr-defined]
            blocks, term="term", table_label="Acceptance Test Data (2)"
        )
        self.assertEqual(val, "222")
        self.assertIn("Acceptance Test Data (2)", snip)

    def test_extract_from_tables_by_header_filters_by_table_label(self) -> None:
        blocks = [
            {"heading": "H", "table_label": "Acceptance Test Data", "rows": [["Tag", "Measured"], ["term", "111"]], "kv": {}},
            {"heading": "H", "table_label": "Acceptance Test Data (2)", "rows": [["Tag", "Measured"], ["term", "222"]], "kv": {}},
        ]
        val, _snip, _extra = backend._extract_from_tables_by_header(  # type: ignore[attr-defined]
            blocks, term="term", header_anchor="Measured", table_label="Acceptance Test Data (2)"
        )
        self.assertEqual(val, "222")

    def test_extract_missing_label_returns_none(self) -> None:
        blocks = [
            {"heading": "H", "table_label": "Acceptance Test Data", "rows": [["term", "111"]], "kv": {"term": "111"}},
        ]
        val, _snip = backend._extract_from_tables(  # type: ignore[attr-defined]
            blocks, term="term", table_label="Acceptance Test Data (2)"
        )
        self.assertIsNone(val)


if __name__ == "__main__":
    unittest.main()

