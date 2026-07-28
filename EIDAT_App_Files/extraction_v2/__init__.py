"""EDAT Extraction v2 — proof of concept.

Scan-first EIDP extraction built exclusively on pip-installable Python
packages (no standalone executables). Pipeline:

    render (pypdfium2) -> grid detection (OpenCV) -> OCR (swappable engine)
    -> structured tables -> typed values -> document.json -> trend store

The OCR engine is pluggable (see engines.py). In production the recommended
engine is RapidOCR on onnxruntime (Microsoft); on locked-down Windows hosts,
Windows.Media.Ocr via the winsdk package needs zero downloads at all.
"""
__version__ = "2.0.0-poc"
