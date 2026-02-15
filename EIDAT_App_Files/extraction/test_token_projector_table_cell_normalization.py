import os
import sys
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction import token_projector  # noqa: E402


class TestTokenProjectorTableCellNormalization(unittest.TestCase):
    def test_strips_vertical_prefix_artifacts(self) -> None:
        self.assertEqual(token_projector.normalize_table_cell_text('| x'), "x")
        self.assertEqual(token_projector.normalize_table_cell_text('|"Pressure'), "Pressure")
        self.assertEqual(token_projector.normalize_table_cell_text("│ 10.0"), "10.0")

    def test_strips_bracket_quote_prefix_artifacts(self) -> None:
        self.assertEqual(token_projector.normalize_table_cell_text('["Flow'), "Flow")
        self.assertEqual(token_projector.normalize_table_cell_text("['Rate"), "Rate")

    def test_preserves_balanced_short_reference(self) -> None:
        self.assertEqual(token_projector.normalize_table_cell_text("[1]"), "[1]")
        self.assertEqual(token_projector.normalize_table_cell_text("(A)"), "(A)")

    def test_can_be_disabled_via_env(self) -> None:
        old = os.environ.get("EIDAT_TABLE_CELL_PREFIX_CLEAN")
        try:
            os.environ["EIDAT_TABLE_CELL_PREFIX_CLEAN"] = "0"
            self.assertEqual(token_projector.normalize_table_cell_text('| x'), "| x")
        finally:
            if old is None:
                os.environ.pop("EIDAT_TABLE_CELL_PREFIX_CLEAN", None)
            else:
                os.environ["EIDAT_TABLE_CELL_PREFIX_CLEAN"] = old


if __name__ == "__main__":
    unittest.main()

