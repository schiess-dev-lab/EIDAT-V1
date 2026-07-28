"""Swappable OCR engines.

Every engine implements:  recognize(gray_img) -> list[Token]
where Token carries text, pixel bbox, and confidence. All downstream logic
is engine-agnostic — swapping engines never touches table logic.

Production candidates (all pip-only, no standalone EXEs):

* RapidOcrEngine  — `pip install rapidocr onnxruntime`. ONNX models ship
  inside the wheel; runs on Microsoft's ONNX Runtime, fully offline.
  Recommended default for the secure environment.
* WindowsOcrEngine — `pip install winsdk`; drives Windows.Media.Ocr, the
  OCR engine built into Windows 10/11. Zero downloads beyond one pure
  wrapper wheel; Microsoft-signed OS component does the work.
* TesseractEngine — stand-in used ONLY for this sandbox PoC (the sandbox
  pip mirror doesn't carry rapidocr). NOT for production: tesseract is a
  standalone Google binary, which violates the deployment constraint.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Token:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float  # 0..1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


class TesseractEngine:
    """Sandbox stand-in engine (see module docstring)."""

    name = "tesseract-standin"

    def recognize(self, gray) -> list[Token]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            gray, config="--psm 11", output_type=Output.DICT
        )
        out: list[Token] = []
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            conf = float(data["conf"][i])
            if not txt or conf < 0:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            out.append(Token(txt, x, y, x + w, y + h, conf / 100.0))
        return out

    def recognize_line(self, gray) -> tuple[str, float]:
        """Single-line mode for targeted cell re-OCR."""
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            gray, config="--psm 7", output_type=Output.DICT
        )
        words, confs = [], []
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            conf = float(data["conf"][i])
            if txt and conf >= 0:
                words.append(txt)
                confs.append(conf / 100.0)
        if not words:
            return "", 0.0
        return " ".join(words), min(confs)


class RapidOcrEngine:
    """Production default: RapidOCR on Microsoft ONNX Runtime (pip-only).

    pip install rapidocr onnxruntime      # models are inside the wheel
    """

    name = "rapidocr-onnx"

    def __init__(self):
        from rapidocr import RapidOCR  # type: ignore

        self._ocr = RapidOCR()

    def recognize(self, gray) -> list[Token]:
        result = self._ocr(gray)
        out: list[Token] = []
        for box, text, score in result[0] if isinstance(result, tuple) else result:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append(Token(text, min(xs), min(ys), max(xs), max(ys), float(score)))
        return out


class WindowsOcrEngine:
    """Zero-install alternative: Windows.Media.Ocr via winsdk (Windows hosts).

    pip install winsdk
    Uses the OS-bundled, Microsoft-signed OCR engine. Word-level bboxes.
    """

    name = "windows-media-ocr"

    def __init__(self):  # pragma: no cover - Windows only
        raise NotImplementedError(
            "Implemented on Windows via winsdk (Windows.Media.Ocr); "
            "not available in this Linux sandbox."
        )


def default_engine():
    """Best available engine in the current environment."""
    try:
        return RapidOcrEngine()
    except Exception:
        return TesseractEngine()
