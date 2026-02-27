import sys
import unittest
from pathlib import Path


# Allow `import extraction.*` when running from repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from extraction.table_multipage_merger import merge_multipage_tables_in_combined_lines  # noqa: E402


def _mk_table(
    *,
    cols: list[tuple[str, int]],
    header: list[str],
    rows: list[list[str]],
    use_eq_header_sep: bool = True,
    newline: str = "\n",
) -> list[str]:
    # cols: [(name, width)]
    # Build a grid table with +---+ borders and |...| rows.
    def border(ch: str) -> str:
        parts = ["+"]
        for _, w in cols:
            parts.append(ch * w)
            parts.append("+")
        return "".join(parts) + newline

    def pipe(values: list[str]) -> str:
        out = ["|"]
        for v, (_, w) in zip(values, cols):
            v2 = str(v)[:w].ljust(w)
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


class TestTableMultipageMerger(unittest.TestCase):
    def test_merges_two_page_continuation_and_blanks_later_region(self) -> None:
        cols = [("A", 6), ("B", 6), ("C", 6)]
        header = ["col1", "col2", "col3"]
        rows_p1 = [[f"r{i}a", f"r{i}b", f"r{i}c"] for i in range(1, 10)]  # 9 body rows
        rows_p2 = [[f"r{i}a", f"r{i}b", f"r{i}c"] for i in range(10, 13)]  # 3 more

        table1 = _mk_table(cols=cols, header=header, rows=rows_p1, use_eq_header_sep=True)
        table2 = _mk_table(cols=cols, header=header, rows=rows_p2, use_eq_header_sep=True)

        lines = []
        lines += ["=== Page 1 ===\n", "\n", "[Table/Chart Title]\n", "My Table\n", "[Table]\n", "\n"]
        lines += table1
        lines += ["\n", "Page 1 tail text\n", "\n"]
        lines += ["=== Page 2 ===\n", "\n", "[Table/Chart Title]\n", "My Table\n", "[Table]\n", "\n"]
        lines += table2
        lines += ["\n", "After table text\n"]

        out = merge_multipage_tables_in_combined_lines(lines)
        txt = "".join(out)

        # Page 1 should now contain both r1a and r12a.
        self.assertIn("|r1a", txt)
        self.assertIn("|r12a", txt)

        # Continuation region on page 2 should be blanked out (title marker line becomes blank).
        # We keep the page marker itself.
        page2_idx = out.index("=== Page 2 ===\n")
        # Find the first non-blank after the blanked region; should be "After table text\n".
        self.assertIn("After table text", txt)
        # Between Page 2 marker and After table text, the title and table should be blank lines only.
        after_idx = out.index("After table text\n")
        region = out[page2_idx + 1 : after_idx]
        self.assertTrue(region, msg="Expected a blanked region on page 2")
        self.assertTrue(all((ln.strip() == "") for ln in region), msg="Expected blank lines in the removed continuation region")

    def test_merges_three_page_chain(self) -> None:
        cols = [("A", 5), ("B", 5)]
        header = ["h1", "h2"]
        p1 = [[f"a{i}", f"b{i}"] for i in range(1, 10)]  # 9
        p2 = [[f"a{i}", f"b{i}"] for i in range(10, 16)]  # 6
        p3 = [[f"a{i}", f"b{i}"] for i in range(16, 19)]  # 3
        t1 = _mk_table(cols=cols, header=header, rows=p1, use_eq_header_sep=True)
        t2 = _mk_table(cols=cols, header=header, rows=p2, use_eq_header_sep=True)
        t3 = _mk_table(cols=cols, header=header, rows=p3, use_eq_header_sep=True)

        lines = ["=== Page 1 ===\n", "\n"] + t1 + ["\n"]
        lines += ["=== Page 2 ===\n", "\n"] + t2 + ["\n"]
        lines += ["=== Page 3 ===\n", "\n"] + t3 + ["\n"]

        out = merge_multipage_tables_in_combined_lines(lines)
        txt = "".join(out)
        self.assertIn("|a1", txt)
        self.assertIn("|a18", txt)
        # a18 should have moved onto page 1 (before the Page 2 marker).
        self.assertLess(out.index(next(ln for ln in out if ln.startswith("|a18"))), out.index("=== Page 2 ===\n"))

        # Page 2 + 3 should have blanked continuation regions.
        p2_marker = out.index("=== Page 2 ===\n")
        p3_marker = out.index("=== Page 3 ===\n")
        self.assertTrue(
            all(ln.strip() == "" for ln in out[p2_marker + 1 : p3_marker]),
            "Page 2 region should be blanked",
        )
        self.assertTrue(
            all(ln.strip() == "" for ln in out[p3_marker + 1 :]),
            "Page 3 region should be blanked",
        )

    def test_does_not_merge_on_layout_mismatch(self) -> None:
        cols1 = [("A", 6), ("B", 6)]
        cols2 = [("A", 6), ("B", 12)]  # width mismatch beyond tolerance
        header = ["h1", "h2"]
        p1 = [[f"a{i}", f"b{i}"] for i in range(1, 10)]
        p2 = [[f"a{i}", f"b{i}"] for i in range(10, 13)]
        t1 = _mk_table(cols=cols1, header=header, rows=p1, use_eq_header_sep=True)
        t2 = _mk_table(cols=cols2, header=header, rows=p2, use_eq_header_sep=True)

        lines = ["=== Page 1 ===\n", "\n"] + t1 + ["\n", "=== Page 2 ===\n", "\n"] + t2 + ["\n"]
        out = merge_multipage_tables_in_combined_lines(lines)
        self.assertEqual(out, lines)

    def test_does_not_merge_small_tables(self) -> None:
        cols = [("A", 6), ("B", 6)]
        header = ["h1", "h2"]
        p1 = [[f"a{i}", f"b{i}"] for i in range(1, 5)]  # 4 body rows (<8)
        p2 = [[f"a{i}", f"b{i}"] for i in range(5, 7)]
        t1 = _mk_table(cols=cols, header=header, rows=p1, use_eq_header_sep=True)
        t2 = _mk_table(cols=cols, header=header, rows=p2, use_eq_header_sep=True)
        lines = ["=== Page 1 ===\n", "\n"] + t1 + ["\n", "=== Page 2 ===\n", "\n"] + t2 + ["\n"]
        out = merge_multipage_tables_in_combined_lines(lines)
        self.assertEqual(out, lines)

    def test_idempotent(self) -> None:
        cols = [("A", 6), ("B", 6), ("C", 6)]
        header = ["col1", "col2", "col3"]
        rows_p1 = [[f"r{i}a", f"r{i}b", f"r{i}c"] for i in range(1, 10)]
        rows_p2 = [[f"r{i}a", f"r{i}b", f"r{i}c"] for i in range(10, 13)]
        table1 = _mk_table(cols=cols, header=header, rows=rows_p1, use_eq_header_sep=True)
        table2 = _mk_table(cols=cols, header=header, rows=rows_p2, use_eq_header_sep=True)
        lines = ["=== Page 1 ===\n", "\n"] + table1 + ["\n", "=== Page 2 ===\n", "\n"] + table2 + ["\n"]

        out1 = merge_multipage_tables_in_combined_lines(lines)
        out2 = merge_multipage_tables_in_combined_lines(out1)
        self.assertEqual(out2, out1)


if __name__ == "__main__":
    unittest.main()
