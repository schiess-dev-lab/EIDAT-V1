import os
import sys
import unittest
from pathlib import Path


# Allow `import ui_next.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestGroupAfterAnchors(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["EIDAT_GROUP_ANCHOR_MIN_RATIO"] = "0.65"

    def test_lines_group_after_excludes_prior_values(self) -> None:
        lines = [
            "term: 111",
            "START SECTION",
            "term: 222",
        ]
        val, _snip = backend._extract_value_from_lines(  # type: ignore[attr-defined]
            lines, term="term", group_after="START SECTION", want_type="number"
        )
        self.assertEqual(val, 222.0)

    def test_lines_group_after_missing_returns_none(self) -> None:
        lines = [
            "term: 111",
            "no anchor here",
            "term: 222",
        ]
        val, _snip = backend._extract_value_from_lines(  # type: ignore[attr-defined]
            lines, term="term", group_after="START SECTION", want_type="number"
        )
        self.assertIsNone(val)

    def test_lines_group_after_fuzzy_match(self) -> None:
        lines = [
            "term: 111",
            "START SECTION",
            "term: 222",
        ]
        # OCR-style confusion: "I" vs "l" or missing a char
        val, _snip = backend._extract_value_from_lines(  # type: ignore[attr-defined]
            lines, term="term", group_after="START SECTlON", want_type="number"
        )
        self.assertEqual(val, 222.0)

    def test_tables_group_after_excludes_prior_blocks(self) -> None:
        blocks = [
            {"heading": "Before", "rows": [["term", "111"]], "kv": {"term": "111"}},
            {"heading": "START SECTION", "rows": [["term", "222"]], "kv": {"term": "222"}},
        ]
        val, _snip = backend._extract_from_tables(  # type: ignore[attr-defined]
            blocks, term="term", group_after="START SECTION"
        )
        self.assertEqual(val, "222")

    def test_tables_group_after_missing_returns_none(self) -> None:
        blocks = [
            {"heading": "Before", "rows": [["term", "111"]], "kv": {"term": "111"}},
        ]
        val, _snip = backend._extract_from_tables(  # type: ignore[attr-defined]
            blocks, term="term", group_after="START SECTION"
        )
        self.assertIsNone(val)

    def test_tables_group_after_row_level_anchor(self) -> None:
        blocks = [
            {
                "heading": "Some Table",
                "rows": [
                    ["term", "111"],
                    ["START SECTION", ""],
                    ["term", "222"],
                ],
                "kv": {"term": "111"},
            }
        ]
        val, _snip = backend._extract_from_tables(  # type: ignore[attr-defined]
            blocks, term="term", group_after="START SECTION"
        )
        self.assertEqual(val, "222")

    def test_table_header_group_after_excludes_prior_blocks(self) -> None:
        blocks = [
            {"heading": "Before", "rows": [["Tag", "Measured"], ["term", "111"]], "kv": {}},
            {"heading": "START SECTION", "rows": [["Tag", "Measured"], ["term", "222"]], "kv": {}},
        ]
        val, _snip, _extra = backend._extract_from_tables_by_header(  # type: ignore[attr-defined]
            blocks, term="term", header_anchor="Measured", group_after="START SECTION"
        )
        self.assertEqual(val, "222")

    def test_table_header_group_after_row_level_anchor(self) -> None:
        blocks = [
            {
                "heading": "Some Table",
                "rows": [
                    ["Tag", "Measured"],
                    ["term", "111"],
                    ["START SECTION", ""],
                    ["term", "222"],
                ],
                "kv": {},
            }
        ]
        val, _snip, _extra = backend._extract_from_tables_by_header(  # type: ignore[attr-defined]
            blocks, term="term", header_anchor="Measured", group_after="START SECTION"
        )
        self.assertEqual(val, "222")

    def test_table_header_group_after_missing_returns_none(self) -> None:
        blocks = [{"heading": "Before", "rows": [["Tag", "Measured"], ["term", "111"]], "kv": {}}]
        val, _snip, _extra = backend._extract_from_tables_by_header(  # type: ignore[attr-defined]
            blocks, term="term", header_anchor="Measured", group_after="START SECTION"
        )
        self.assertIsNone(val)


if __name__ == "__main__":
    unittest.main()
