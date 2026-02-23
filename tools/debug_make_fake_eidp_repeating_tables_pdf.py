from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _draw_table(
    page: fitz.Page,
    *,
    x0: float,
    y0: float,
    col_widths: list[float],
    row_h: float,
    rows: list[list[str]],
    font: str = "helv",
    font_size: float = 9.5,
    border_w: float = 1.0,
) -> float:
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    if n_rows <= 0 or n_cols <= 0:
        return y0

    widths = list(col_widths)
    if len(widths) < n_cols:
        widths.extend([80.0] * (n_cols - len(widths)))
    widths = widths[:n_cols]

    x_edges = [x0]
    for w in widths:
        x_edges.append(x_edges[-1] + float(w))
    y_edges = [y0 + (i * float(row_h)) for i in range(n_rows + 1)]

    # Grid lines
    for xe in x_edges:
        page.draw_line((xe, y_edges[0]), (xe, y_edges[-1]), width=border_w)
    for ye in y_edges:
        page.draw_line((x_edges[0], ye), (x_edges[-1], ye), width=border_w)

    # Cell text
    pad_x = 4.0
    pad_y = 2.0
    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            txt = str(row[c_idx] if c_idx < len(row) else "")
            rect = fitz.Rect(
                x_edges[c_idx] + pad_x,
                y_edges[r_idx] + pad_y,
                x_edges[c_idx + 1] - pad_x,
                y_edges[r_idx + 1] - pad_y,
            )
            is_header = r_idx == 0
            page.insert_textbox(
                rect,
                txt,
                fontname=font,
                fontsize=(font_size + 0.5 if is_header else font_size),
                align=fitz.TEXT_ALIGN_LEFT,
            )

    return y_edges[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a fake EIDP PDF with two repeating tables.")
    parser.add_argument(
        "--out",
        type=str,
        default="Stored PDF/Fake_EIDP_Repeating_Tables.pdf",
        help="Output PDF path (relative to repo root by default).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    try:
        # Landscape US Letter for wide, readable tables.
        page = doc.new_page(width=792, height=612)

        page.insert_text((36, 42), "FAKE EIDP - Repeating Tables Debug", fontsize=18, fontname="helv")
        page.insert_text((36, 66), "Two copies of the same sub-test table appear below.", fontsize=12, fontname="helv")
        page.insert_text((36, 86), "These are drawn with real grid lines for table detection.", fontsize=12, fontname="helv")

        # 36pt margins => usable width ~720.
        col_widths = [200, 140, 110, 140, 130]
        row_h = 34.0

        # Table A
        y = 125.0
        page.insert_text((36, y - 12), "Acceptance Test Data (Run A)", fontsize=14, fontname="helv")
        rows_a = [
            ["Parameter", "Resistance", "Voltage", "Valve Voltage", "Notes"],
            ["Test Temp", "10 kohm", "28 V", "5.00 V", "FAKE_EIDP_REPEAT_TABLE"],
            ["Test Temp", "11 kohm", "28 V", "5.10 V", "PASS"],
        ]
        y = _draw_table(
            page,
            x0=36.0,
            y0=y,
            col_widths=col_widths,
            row_h=row_h,
            rows=rows_a,
            font="helv",
            font_size=12.5,
            border_w=1.5,
        )

        # Table B (repeat)
        y += 55.0
        page.insert_text((36, y - 12), "Acceptance Test Data (Run B)", fontsize=14, fontname="helv")
        rows_b = [
            ["Parameter", "Resistance", "Voltage", "Valve Voltage", "Notes"],
            ["Test Temp", "12 kohm", "28 V", "5.20 V", "FAKE_EIDP_REPEAT_TABLE"],
            ["Test Temp", "13 kohm", "28 V", "5.30 V", "PASS"],
        ]
        y = _draw_table(
            page,
            x0=36.0,
            y0=y,
            col_widths=col_widths,
            row_h=row_h,
            rows=rows_b,
            font="helv",
            font_size=12.5,
            border_w=1.5,
        )

        # Non-matching small table
        y += 55.0
        page.insert_text((36, y - 12), "Other Table (Should Not Label)", fontsize=14, fontname="helv")
        rows_c = [
            ["Foo", "Bar"],
            ["hello", "world"],
        ]
        _draw_table(
            page,
            x0=36.0,
            y0=y,
            col_widths=[180, 180],
            row_h=row_h,
            rows=rows_c,
            font="helv",
            font_size=12.5,
            border_w=1.5,
        )

        doc.save(str(out_path))
    finally:
        doc.close()

    print(f"[DONE] Wrote PDF: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
