#!/usr/bin/env python3
"""
Generate a synthetic PDF designed to exercise the table-splitting heuristics:
- A top bordered table with 3 columns
- A single-cell separator row
- A bottom bordered table with 2 columns (different internal vline positions)

Output is a raster PDF (embedded image) created via Pillow only.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_font(size: int):
    from PIL import ImageFont  # type: ignore

    # Prefer a real TrueType font for OCR legibility; fall back to default.
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\times.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_centered(draw, box, text: str, font, *, fill=(0, 0, 0)):
    x0, y0, x1, y1 = box
    w = max(1, int(x1 - x0))
    h = max(1, int(y1 - y0))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = (len(text) * 8, 16)
    tx = int(x0 + (w - tw) / 2)
    ty = int(y0 + (h - th) / 2)
    draw.text((tx, ty), text, font=font, fill=fill)


def _draw_table_grid(draw, *, left: int, top: int, right: int, bottom: int, col_xs: list[int], row_ys: list[int], width: int):
    # Outer border
    draw.rectangle([left, top, right, bottom], outline=(0, 0, 0), width=width)
    # Internal verticals
    for x in col_xs:
        draw.line([(x, top), (x, bottom)], fill=(0, 0, 0), width=width)
    # Internal horizontals
    for y in row_ys:
        draw.line([(left, y), (right, y)], fill=(0, 0, 0), width=width)


def _make_page_single_split(*, dpi: int, bridge_gap: int, include_separators: bool):
    # Letter @ DPI
    page_w = int(round(8.5 * dpi))
    page_h = int(round(11.0 * dpi))

    from PIL import Image, ImageDraw  # type: ignore

    img = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_big = _load_font(max(14, int(dpi * 0.08)))
    font = _load_font(max(12, int(dpi * 0.06)))

    # Title
    _draw_centered(draw, (0, int(dpi * 0.2), page_w, int(dpi * 0.6)), "DEBUG: TABLE SPLIT (VLINE MISMATCH)", font_big)

    left = int(dpi * 1.0)
    right = int(page_w - dpi * 1.0)
    line_w = max(2, int(round(dpi / 150)))  # ~2px at 300dpi

    # Top table: 3 columns, 5 rows
    top1 = int(dpi * 1.3)
    bot1 = int(dpi * 4.1)
    col1_a = int(left + (right - left) * 0.34)
    col1_b = int(left + (right - left) * 0.68)
    rows1 = 5
    row_h1 = int((bot1 - top1) / rows1)
    row_ys1 = [top1 + row_h1 * i for i in range(1, rows1)]
    _draw_table_grid(
        draw,
        left=left,
        top=top1,
        right=right,
        bottom=bot1,
        col_xs=[col1_a, col1_b],
        row_ys=row_ys1,
        width=line_w,
    )

    # Optional full-width single-row "separator cell" between blocks.
    # When disabled, the blocks are simply stacked with bridge_gap.
    if include_separators:
        sep_top = bot1 + bridge_gap
        sep_bot = sep_top + int(dpi * 0.35)
        draw.rectangle([left, sep_top, right, sep_bot], outline=(0, 0, 0), width=line_w)
        _draw_centered(draw, (left, sep_top, right, sep_bot), "SECTION BREAK (SINGLE CELL ROW)", font_big)
        next_top = sep_bot + bridge_gap
    else:
        next_top = bot1 + bridge_gap

    # Bottom table: 2 columns, 4 rows with different vline positions (mismatch).
    top2 = next_top
    bot2 = int(dpi * 7.4)
    col2_a = int(left + (right - left) * 0.55)  # intentionally different from col1_a/col1_b
    rows2 = 4
    row_h2 = int((bot2 - top2) / rows2)
    row_ys2 = [top2 + row_h2 * i for i in range(1, rows2)]
    _draw_table_grid(
        draw,
        left=left,
        top=top2,
        right=right,
        bottom=bot2,
        col_xs=[col2_a],
        row_ys=row_ys2,
        width=line_w,
    )

    # Some cell text (kept simple; OCR isn't the point here but helps sanity-check).
    # Top table headers
    headers = ["Parameter", "Spec", "Result"]
    x_edges_top = [left, col1_a, col1_b, right]
    y_edges_top = [top1] + row_ys1 + [bot1]
    for j in range(3):
        _draw_centered(draw, (x_edges_top[j], y_edges_top[0], x_edges_top[j + 1], y_edges_top[1]), headers[j], font_big)
    # A couple data rows
    top_rows = [
        ("Flow Rate", "10-12", "11.1"),
        ("Pressure", "45-55", "52"),
        ("Temp", "20-30", "28"),
        ("Notes", "N/A", "OK"),
    ]
    for i, row in enumerate(top_rows, start=1):
        for j in range(3):
            _draw_centered(draw, (x_edges_top[j], y_edges_top[i], x_edges_top[j + 1], y_edges_top[i + 1]), row[j], font)

    # Bottom table headers
    x_edges_bot = [left, col2_a, right]
    y_edges_bot = [top2] + row_ys2 + [bot2]
    _draw_centered(draw, (x_edges_bot[0], y_edges_bot[0], x_edges_bot[1], y_edges_bot[1]), "Item", font_big)
    _draw_centered(draw, (x_edges_bot[1], y_edges_bot[0], x_edges_bot[2], y_edges_bot[1]), "Value", font_big)
    bot_rows = [
        ("Alpha", "123"),
        ("Beta", "456"),
        ("Gamma", "789"),
    ]
    for i, row in enumerate(bot_rows, start=1):
        _draw_centered(draw, (x_edges_bot[0], y_edges_bot[i], x_edges_bot[1], y_edges_bot[i + 1]), row[0], font)
        _draw_centered(draw, (x_edges_bot[1], y_edges_bot[i], x_edges_bot[2], y_edges_bot[i + 1]), row[1], font)

    return img


def _make_page_multi_split(*, dpi: int, bridge_gap: int, include_separators: bool):
    """
    A busier page: several stacked table blocks with different internal vlines.
    Blocks are "attached" via full-width single-cell separator rows to encourage
    the detector to cluster them as one table, then rely on the split heuristics.
    """
    page_w = int(round(8.5 * dpi))
    page_h = int(round(11.0 * dpi))

    from PIL import Image, ImageDraw  # type: ignore

    img = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_big = _load_font(max(14, int(dpi * 0.08)))
    font = _load_font(max(12, int(dpi * 0.06)))

    _draw_centered(draw, (0, int(dpi * 0.2), page_w, int(dpi * 0.6)), "DEBUG: MULTI TABLE SPLITS (ATTACHED)", font_big)

    left = int(dpi * 0.8)
    right = int(page_w - dpi * 0.8)
    line_w = max(2, int(round(dpi / 150)))

    y = int(dpi * 1.1)

    def _sep(label: str) -> None:
        nonlocal y
        top = y
        bot = top + int(dpi * 0.30)
        draw.rectangle([left, top, right, bot], outline=(0, 0, 0), width=line_w)
        _draw_centered(draw, (left, top, right, bot), label, font_big)
        y = bot + bridge_gap

    def _block(cols: list[float], rows: int, height_in: float, label_prefix: str) -> None:
        nonlocal y
        top = y
        bot = top + int(dpi * height_in)
        col_xs = [int(left + (right - left) * p) for p in cols]
        row_h = max(1, int((bot - top) / max(1, rows)))
        row_ys = [top + row_h * i for i in range(1, rows)]
        _draw_table_grid(draw, left=left, top=top, right=right, bottom=bot, col_xs=col_xs, row_ys=row_ys, width=line_w)

        x_edges = [left] + col_xs + [right]
        y_edges = [top] + row_ys + [bot]
        if len(y_edges) >= 3:
            for j in range(len(x_edges) - 1):
                _draw_centered(draw, (x_edges[j], y_edges[0], x_edges[j + 1], y_edges[1]), f"{label_prefix}H{j+1}", font_big)
            for j in range(len(x_edges) - 1):
                _draw_centered(draw, (x_edges[j], y_edges[1], x_edges[j + 1], y_edges[2]), f"{label_prefix}{j+1}", font)

        y = bot + bridge_gap

    # Several table blocks with different internal vertical lines.
    _block([0.33, 0.66], rows=4, height_in=2.0, label_prefix="A")
    if include_separators:
        _sep("BREAK 1 (SINGLE CELL)")
    _block([0.25, 0.50, 0.75], rows=4, height_in=2.0, label_prefix="B")
    if include_separators:
        _sep("BREAK 2 (SINGLE CELL)")
    _block([0.58], rows=3, height_in=1.7, label_prefix="C")
    if include_separators:
        _sep("BREAK 3 (SINGLE CELL)")
    _block([0.38, 0.70], rows=4, height_in=2.1, label_prefix="D")

    _draw_centered(draw, (0, page_h - int(dpi * 0.8), page_w, page_h - int(dpi * 0.3)), "Expected: multiple splits + separator rows isolated", font_big)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output PDF path.")
    ap.add_argument("--dpi", type=int, default=300, help="Raster DPI used to size the page (default 300).")
    ap.add_argument(
        "--bridge-gap-px",
        type=int,
        default=2,
        help="Vertical pixel gap between stacked table blocks at raster DPI (default 2). Use 0 to force clustering.",
    )
    ap.add_argument(
        "--extra-misaligned-page",
        action="store_true",
        help="Add a second page with several attached misaligned tables.",
    )
    ap.add_argument(
        "--no-separators",
        action="store_true",
        help="Do not draw the full-width single-cell separator rows between table blocks.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = int(args.dpi)
    if dpi <= 0:
        dpi = 300

    bridge_gap = int(args.bridge_gap_px)
    if bridge_gap < 0:
        bridge_gap = 0

    include_separators = not bool(args.no_separators)
    img1 = _make_page_single_split(dpi=dpi, bridge_gap=bridge_gap, include_separators=include_separators)
    if bool(args.extra_misaligned_page):
        img2 = _make_page_multi_split(dpi=dpi, bridge_gap=bridge_gap, include_separators=include_separators)
        img1.save(str(out_path), "PDF", resolution=float(dpi), save_all=True, append_images=[img2])
    else:
        # Save PDF (embedded raster). "resolution" sets the page size in PDF points.
        img1.save(str(out_path), "PDF", resolution=float(dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
