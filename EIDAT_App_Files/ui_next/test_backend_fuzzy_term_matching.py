import sys
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestBackendFuzzyTermMatching(unittest.TestCase):
    def test_extract_from_tables_matches_term_label(self) -> None:
        blocks = [
            {
                "heading": "",
                "rows": [["Parameter", "Value"], ["Seat leak @ 150 psig", "0.12"]],
                "kv": {"seatleak150psig": "0.12"},
            }
        ]
        val, _ = backend._extract_from_tables(blocks, term="LEAK150", term_label="Seat Leak @ 150 psig")
        self.assertEqual(val, "0.12")

    def test_extract_from_tables_fuzzy_label(self) -> None:
        blocks = [
            {
                "heading": "",
                "rows": [["Parameter", "Value"], ["Open Stroke Time", "1.23"]],
                "kv": {},
            }
        ]
        val, _ = backend._extract_from_tables(
            blocks,
            term="OPEN_T",
            term_label="Opening Stroke Time",
            fuzzy_term=True,
            fuzzy_min_ratio=0.80,
        )
        self.assertEqual(val, "1.23")

    def test_resolve_acceptance_term_key_fuzzy(self) -> None:
        lookup = {"openstroktime": "Open Stroke Time"}
        key = backend._resolve_acceptance_term_key(
            lookup,
            term="OPEN_T",
            term_label="Opening Stroke Time",
            fuzzy_term=True,
            fuzzy_min_ratio=0.80,
        )
        self.assertEqual(key, "Open Stroke Time")

    def test_extract_value_from_lines_term_label_numeric(self) -> None:
        lines = ["Opening Stroke Time: 1.23 s"]
        val, _ = backend._extract_value_from_lines(
            lines,
            term="OPEN_T",
            term_label="Opening Stroke Time",
            want_type="number",
        )
        self.assertEqual(val, 1.23)


if __name__ == "__main__":
    unittest.main()

