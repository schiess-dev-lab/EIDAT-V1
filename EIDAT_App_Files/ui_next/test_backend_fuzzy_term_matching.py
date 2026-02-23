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

    def test_fuzzy_term_score_does_not_overmatch_generic_subset(self) -> None:
        term = "Res, Upstream Propellant Valve"
        term_label = "Res, Upstream Prop. Valve"

        self.assertLess(backend._fuzzy_term_score(term, term_label, "Upstream Valve"), 0.78)
        self.assertLess(backend._fuzzy_term_score(term, term_label, "Drop-Out Voltage, Upstream Valve"), 0.78)
        self.assertGreaterEqual(backend._fuzzy_term_score(term, term_label, "Res, Upstream Prop. Valve"), 0.99)

    def test_extract_from_tables_prefers_termlabel_over_generic_term(self) -> None:
        blocks = [
            {
                "heading": "Example",
                "rows": [
                    ["Parameter", "Value"],
                    ["Drop-Out Voltage, Upstream Valve", "111"],
                    ["Res, Upstream Prop. Valve", "222"],
                ],
                "kv": {},
            }
        ]

        # Simulate a workbook row where `Term` is generic and collides, but `TermLabel` is specific.
        val, _ = backend._extract_from_tables(
            blocks,
            term="Upstream Valve",
            term_label="Res, Upstream Prop. Valve",
            fuzzy_term=True,
            fuzzy_min_ratio=0.78,
        )
        self.assertEqual(val, "222")

    def test_extract_value_from_lines_header_anchor_missing_returns_none(self) -> None:
        lines = ["Valve Resistance: 2.22 ohm"]
        val, _ = backend._extract_value_from_lines(
            lines,
            term="VALVE_RES",
            term_label="Valve Resistance",
            header_anchor="Measured",
            want_type="number",
        )
        self.assertIsNone(val)

    def test_extract_from_tables_by_header_ranks_best_fuzzy_term_match(self) -> None:
        blocks = [
            {
                "heading": "Example",
                "rows": [
                    ["Tag", "Measured Value", "Units"],
                    ["Valve Pressure", "111", "psi"],
                    ["Valv Resist.", "222", "ohm"],
                    ["Valve Resistivity", "333", "ohm"],
                ],
                "kv": {},
            }
        ]

        val, _snip, _extra = backend._extract_from_tables_by_header(
            blocks,
            term="VALVE_RES",
            term_label="Valve Resistance",
            header_anchor="Measured",
            fuzzy_header=False,
            fuzzy_min_ratio=0.72,
            fuzzy_term=True,
            fuzzy_term_min_ratio=0.78,
        )
        self.assertEqual(val, "222")

    def test_extract_from_tables_by_header_tie_breaks_by_earliest_row(self) -> None:
        blocks = [
            {
                "heading": "Example",
                "rows": [
                    ["Tag", "Measured Value"],
                    ["Valv Resist.", "222"],
                    ["Valv Resist.", "333"],
                ],
                "kv": {},
            }
        ]

        val, _snip, _extra = backend._extract_from_tables_by_header(
            blocks,
            term="VALVE_RES",
            term_label="Valve Resistance",
            header_anchor="Measured",
            fuzzy_header=False,
            fuzzy_min_ratio=0.72,
            fuzzy_term=True,
            fuzzy_term_min_ratio=0.78,
        )
        self.assertEqual(val, "222")


if __name__ == "__main__":
    unittest.main()
