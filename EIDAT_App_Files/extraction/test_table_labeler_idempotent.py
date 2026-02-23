import sys
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction import table_labeler  # noqa: E402


class TestTableLabelerIdempotent(unittest.TestCase):
    def test_labeler_inserts_numbered_labels_and_is_idempotent(self) -> None:
        rules_cfg = {
            "version": 1,
            "append_index_after_first": True,
            "marker": "TABLE_LABEL",
            "rules": [
                {
                    "label": "Acceptance Test Data",
                    "priority": 100,
                    "must_contain_all": ["resistance", "voltage"],
                    "must_contain_any": [],
                    "must_not_contain": [],
                    "min_rows": 2,
                    "min_cols": 2,
                    "max_rows": 0,
                    "max_cols": 0,
                    "match_scope": "any_cell",
                }
            ],
        }

        lines = [
            "=== Page 1 ===\n",
            "\n",
            "+-----+-----+\n",
            "| resistance | voltage |\n",
            "+-----+-----+\n",
            "| 1 | 2 |\n",
            "+-----+-----+\n",
            "\n",
            "+-----+-----+\n",
            "| resistance | voltage |\n",
            "+-----+-----+\n",
            "| 3 | 4 |\n",
            "+-----+-----+\n",
            "\n",
        ]

        out1 = table_labeler.label_combined_lines(lines, rules_cfg)
        txt1 = "".join(out1)
        self.assertIn("[TABLE_LABEL]\nAcceptance Test Data\n\n", txt1)
        self.assertIn("[TABLE_LABEL]\nAcceptance Test Data (2)\n\n", txt1)

        out2 = table_labeler.label_combined_lines(out1, rules_cfg)
        self.assertEqual("".join(out2), txt1)


if __name__ == "__main__":
    unittest.main()

