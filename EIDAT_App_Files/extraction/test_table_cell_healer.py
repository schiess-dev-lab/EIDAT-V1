import sys
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction.table_cell_healer import heal_combined_tables_in_lines  # noqa: E402


def _mk_table(
    *,
    cols: list[tuple[str, int]],
    header: list[str],
    rows: list[list[str]],
    use_eq_header_sep: bool = True,
    newline: str = "\n",
) -> list[str]:
    def border(ch: str) -> str:
        parts = ["+"]
        for _, w in cols:
            parts.append(ch * int(w))
            parts.append("+")
        return "".join(parts) + newline

    def pipe(values: list[str]) -> str:
        out = ["|"]
        for v, (_, w) in zip(values, cols):
            v2 = str(v)[: int(w)].ljust(int(w))
            out.append(v2)
            out.append("|")
        return "".join(out) + newline

    lines: list[str] = []
    lines.append(border("-"))
    lines.append(pipe(header))
    lines.append(border("=" if use_eq_header_sep else "-"))
    for r in rows:
        lines.append(pipe(r))
        lines.append(border("-"))
    return lines


def _first_table_rows(lines: list[str]) -> list[list[str]]:
    in_table = False
    rows: list[list[str]] = []
    for ln in lines:
        s = str(ln).rstrip("\n")
        if s.strip().startswith("+") and s.strip().endswith("+") and ("-" in s or "=" in s):
            in_table = True
            continue
        if in_table and s.strip().startswith("|"):
            parts = [p.strip() for p in s.strip().strip("|").split("|")]
            rows.append(parts)
            continue
        if in_table and not s.strip():
            break
    return rows


class TestTableCellHealer(unittest.TestCase):
    def test_prunes_empty_narrow_column_and_empty_row(self) -> None:
        cols = [("A", 6), ("ghost", 3), ("B", 6)]
        header = ["colA", "", "colB"]
        rows = [
            ["x", "", "y"],
            ["", "", ""],  # fully empty row -> should be removed
            ["u", "", "v"],
        ]
        table = _mk_table(cols=cols, header=header, rows=rows, use_eq_header_sep=True)
        lines = ["=== Page 1 ===\n", "\n"] + table + ["\n"]

        cfg = {
            "version": 1,
            "enabled": True,
            "prune": {"enabled": True, "drop_empty_cols": True, "drop_empty_rows": True, "max_empty_col_width": 5},
            "spacing": {"enabled": False},
            "numeric": {"enabled": False},
        }
        out, stats = heal_combined_tables_in_lines(lines, cfg=cfg)
        txt = "".join(out)
        self.assertIn("+------+------+\n", txt)
        self.assertEqual(stats.get("cols_dropped"), 1)
        self.assertEqual(stats.get("rows_dropped"), 1)

        rows_out = _first_table_rows(out)
        self.assertEqual(len(rows_out[0]), 2)
        self.assertEqual(rows_out[1], ["x", "y"])
        self.assertEqual(rows_out[2], ["u", "v"])

    def test_spacing_fix_to_glue_preserves_scientific(self) -> None:
        cols = [("A", 14), ("B", 10)]
        header = ["desc", "val"]
        rows = [["1 to4", "1e-3"]]
        table = _mk_table(cols=cols, header=header, rows=rows, use_eq_header_sep=True)
        lines = ["=== Page 1 ===\n", "\n"] + table + ["\n"]

        cfg = {
            "version": 1,
            "enabled": True,
            "prune": {"enabled": False},
            "spacing": {"enabled": True, "default_scope": "all_cells"},
            "numeric": {"enabled": False},
        }
        out, _ = heal_combined_tables_in_lines(lines, cfg=cfg)
        rows_out = _first_table_rows(out)
        self.assertEqual(rows_out[1][0], "1 to 4")
        self.assertEqual(rows_out[1][1], "1e-3")

    def test_numeric_heal_uses_neighbors(self) -> None:
        cols = [("Measured", 10)]
        header = ["Measured"]
        rows = [["10.4"], ["1O.5"], ["10.6"]]
        table = _mk_table(cols=cols, header=header, rows=rows, use_eq_header_sep=True)
        lines = ["=== Page 1 ===\n", "\n"] + table + ["\n"]

        cfg = {
            "version": 1,
            "enabled": True,
            "prune": {"enabled": False},
            "spacing": {"enabled": True, "default_scope": "all_cells"},
            "numeric": {
                "enabled": True,
                "numeric_ratio_min": 0.6,
                "numeric_roles": ["value"],
                "role_synonyms": {"value": ["measured", "value"]},
                "unicode_minus_to_dash": True,
                "allow_o_to_zero": True,
                "allow_i_l_to_one": True,
            },
        }
        out, stats = heal_combined_tables_in_lines(lines, cfg=cfg)
        rows_out = _first_table_rows(out)
        self.assertEqual(rows_out[2][0], "10.5")
        self.assertGreaterEqual(int(stats.get("cells_numeric_healed") or 0), 1)


if __name__ == "__main__":
    unittest.main()

