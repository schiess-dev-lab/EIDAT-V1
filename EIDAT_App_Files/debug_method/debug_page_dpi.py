#!/usr/bin/env python3
"""
Render PDF pages in debug_method/debug_file to PNG at fixed DPI.

Outputs deterministic filenames: page_1.png, page_2.png, ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _find_pdfs(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise RuntimeError(f"Input folder not found: {input_dir}")
    pdfs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {input_dir}")
    return pdfs


def _render_pdf_to_png(pdf_path: Path, out_dir: Path, dpi: int) -> int:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required to render PDF pages.") from exc

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow (PIL) is required to save PNGs with DPI metadata.") from exc

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_index in range(doc.page_count):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        out_path = out_dir / f"page_{page_index + 1}.png"
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        img.save(str(out_path), dpi=(float(dpi), float(dpi)))
    return doc.page_count


def _verify_outputs(out_dir: Path, page_count: int) -> None:
    missing = []
    for page_index in range(1, page_count + 1):
        path = out_dir / f"page_{page_index}.png"
        if not path.exists():
            missing.append(path.name)
    if missing:
        raise RuntimeError(f"Missing rendered pages: {', '.join(missing)}")

    existing = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name.startswith("page_") and p.suffix.lower() == ".png")
    if len(existing) < page_count:
        raise RuntimeError(f"Expected {page_count} PNGs, found {len(existing)}")

    print(f"Rendered {page_count} pages to {out_dir}")


def render_pdf_set(
    pdfs: List[Path],
    *,
    out_root: Path,
    dpi: int,
    clean: bool,
) -> Dict[str, Tuple[Path, int]]:
    results: Dict[str, Tuple[Path, int]] = {}
    for pdf_path in pdfs:
        per_out = out_root / pdf_path.stem
        per_out.mkdir(parents=True, exist_ok=True)
        if clean:
            for path in per_out.glob("page_*.png"):
                try:
                    path.unlink()
                except Exception:
                    pass
        page_count = _render_pdf_to_png(pdf_path, per_out, dpi)
        _verify_outputs(per_out, page_count)
        results[pdf_path.stem] = (per_out, page_count)
    return results


def main() -> int:
    debug_dir = Path(__file__).resolve().parent
    default_input_dir = debug_dir / "DebugFileLocation"
    default_out_root = debug_dir / "debug_file"

    parser = argparse.ArgumentParser(description="Render PDF pages to PNG at fixed DPI.")
    parser.add_argument("--pdf", type=str, default="", help="Optional PDF path (overrides input-dir).")
    parser.add_argument("--input-dir", type=str, default=str(default_input_dir), help="Folder containing PDFs.")
    parser.add_argument("--out-root", type=str, default=str(default_out_root), help="Output root for rendered PNGs.")
    parser.add_argument("--dpi", type=int, default=450, help="Render DPI (default: 450).")
    parser.add_argument("--clean", dest="clean", action="store_true", help="Remove existing page_*.png outputs.")
    parser.add_argument("--no-clean", dest="clean", action="store_false", help="Keep existing page_*.png outputs.")
    parser.set_defaults(clean=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise RuntimeError(f"PDF not found: {pdf_path}")
        per_out = out_root / pdf_path.stem
        per_out.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for path in per_out.glob("page_*.png"):
                try:
                    path.unlink()
                except Exception:
                    pass
        page_count = _render_pdf_to_png(pdf_path, per_out, args.dpi)
        _verify_outputs(per_out, page_count)
    else:
        input_dir = Path(args.input_dir)
        pdfs = _find_pdfs(input_dir)
        render_pdf_set(pdfs, out_root=out_root, dpi=args.dpi, clean=bool(args.clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
