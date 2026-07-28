"""PDF page rasterization + page triage.

pypdfium2 is a pip wheel (BSD/Apache). On locked-down Windows hosts the
same role can be filled by Windows.Data.Pdf (OS component, via winsdk).

Triage: pages that carry a real text layer skip OCR entirely — their words
are taken from the PDF itself (exact coordinates, exact text). Image-only
pages go down the scan path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pypdfium2 as pdfium

from .engines import Token


@dataclass
class PageRaster:
    number: int  # 1-based
    dpi: int
    gray: np.ndarray  # HxW uint8
    width_pt: float
    height_pt: float
    text_tokens: list[Token] = field(default_factory=list)  # from text layer
    source: str = "ocr"  # "text" if the text layer is usable

    @property
    def scale(self) -> float:
        return self.dpi / 72.0


def render_pages(pdf_path, dpi: int = 300, text_min_words: int = 5):
    """Yield PageRaster for each page, with text-layer triage."""
    doc = pdfium.PdfDocument(pdf_path)
    try:
        for i, page in enumerate(doc):
            bmp = page.render(scale=dpi / 72.0, grayscale=True)
            arr = bmp.to_numpy()
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            pr = PageRaster(
                number=i + 1,
                dpi=dpi,
                gray=np.ascontiguousarray(arr),
                width_pt=page.get_width(),
                height_pt=page.get_height(),
            )
            # --- text-layer triage ---------------------------------------
            # pdfplumber supplies WORD-level boxes (required for KV gap
            # detection); its coordinates are PDF points, y down from top —
            # same orientation as the raster, just scaled.
            words: list[Token] = []
            try:
                plumber_page = _plumber_pages(str(pdf_path))[i]
                s = pr.scale
                for w in plumber_page.extract_words():
                    words.append(
                        Token(
                            w["text"],
                            float(w["x0"]) * s,
                            float(w["top"]) * s,
                            float(w["x1"]) * s,
                            float(w["bottom"]) * s,
                            1.0,
                        )
                    )
            except Exception:
                words = []
            if len(words) >= text_min_words:
                pr.text_tokens = words
                pr.source = "text"
            yield pr
    finally:
        doc.close()
        _plumber_close(str(pdf_path))


_plumber_cache: dict = {}


def _plumber_pages(pdf_path: str):
    if pdf_path not in _plumber_cache:
        import pdfplumber

        _plumber_cache[pdf_path] = pdfplumber.open(pdf_path)
    return _plumber_cache[pdf_path].pages


def _plumber_close(pdf_path: str) -> None:
    obj = _plumber_cache.pop(pdf_path, None)
    if obj is not None:
        try:
            obj.close()
        except Exception:
            pass
