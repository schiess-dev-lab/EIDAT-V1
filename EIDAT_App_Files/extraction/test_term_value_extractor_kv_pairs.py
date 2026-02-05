import unittest
from pathlib import Path
import sys


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction.term_value_extractor import CombinedTxtParser  # noqa: E402


class TestKvPairsExtraction(unittest.TestCase):
    def test_extracts_pipe_and_gap_pairs(self) -> None:
        content = "\n".join(
            [
                "=== Page 1 ===",
                "[Header]",
                "SN4001 | 2026-01-16",
                "",
                "[Line]",
                "Test Plan         TPL-1000",
                "",
                "[Line]",
                "Operator | A. RIVERA",
                "",
                "[Footer]",
                "Page 1/8",
                "",
            ]
        )
        parser = CombinedTxtParser(content)
        result = parser.parse()

        self.assertTrue(
            any(p.get("term") == "Test Plan" and p.get("value") == "TPL-1000" for p in result.kv_pairs),
            msg=f"kv_pairs={result.kv_pairs}",
        )
        self.assertTrue(
            any(p.get("term") == "Operator" and p.get("value") == "A. RIVERA" for p in result.kv_pairs),
            msg=f"kv_pairs={result.kv_pairs}",
        )
        self.assertFalse(
            any((p.get("section") or "").lower() in {"header", "footer"} for p in result.kv_pairs),
            msg=f"kv_pairs={result.kv_pairs}",
        )


if __name__ == "__main__":
    unittest.main()

