import sys
import unittest
from pathlib import Path


# Allow `import ui_next.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestTableLabelParsing(unittest.TestCase):
    def test_parse_ascii_tables_reads_table_label(self) -> None:
        lines = [
            "[TABLE_LABEL]",
            "Acceptance Test Data (2)",
            "",
            "[STRING]",
            "My Heading",
            "",
            "+-----+-----+",
            "| Tag | Measured |",
            "+-----+-----+",
            "| Valve Voltage | 12 |",
            "+-----+-----+",
            "",
            "+-----+-----+",
            "| Tag | Measured |",
            "+-----+-----+",
            "| Valve Voltage | 99 |",
            "+-----+-----+",
            "",
        ]
        blocks = backend._parse_ascii_tables(lines)  # type: ignore[attr-defined]
        self.assertGreaterEqual(len(blocks), 2)
        self.assertEqual(blocks[0].get("table_label"), "Acceptance Test Data (2)")
        self.assertEqual(blocks[0].get("heading"), "My Heading")
        self.assertEqual(blocks[1].get("table_label"), "")


if __name__ == "__main__":
    unittest.main()

