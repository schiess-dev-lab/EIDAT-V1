from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


def build_pointer_token(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    enc = base64.urlsafe_b64encode(raw).decode("ascii")
    return f"EIDAT_PTR:{enc}"


def embed_pointer_token(pdf_path: Path, token: str, *, overwrite: bool = False) -> bool:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to embed pointer tokens.")
    doc = fitz.open(str(pdf_path))
    try:
        md = doc.metadata or {}
        keywords = str(md.get("keywords") or "").strip()
        if ("EIDAT_PTR:" in keywords or "eidat" in keywords.lower()) and not overwrite:
            return False
        if "EIDAT_PTR:" in keywords and overwrite:
            parts = [p for p in keywords.split(";") if "EIDAT_PTR:" not in p]
            keywords = ";".join([p for p in parts if p.strip()])
        new_keywords = f"{keywords};{token}" if keywords else token
        md["keywords"] = new_keywords
        if not md.get("subject"):
            md["subject"] = "EIDAT Pointer Token"
        doc.set_metadata(md)
        doc.saveIncr()
        return True
    finally:
        try:
            doc.close()
        except Exception:
            pass


def has_pointer_token(pdf_path: Path) -> bool:
    if fitz is None:
        return False
    doc = fitz.open(str(pdf_path))
    try:
        md = doc.metadata or {}
        keywords = str(md.get("keywords") or "")
        return "EIDAT_PTR:" in keywords or "eidat" in keywords.lower()
    finally:
        try:
            doc.close()
        except Exception:
            pass


def extract_pointer_token(pdf_path: Path) -> Optional[dict]:
    if fitz is None:
        return None
    doc = fitz.open(str(pdf_path))
    try:
        md = doc.metadata or {}
        keywords = str(md.get("keywords") or "")
        if "EIDAT_PTR:" not in keywords:
            return None
        parts = keywords.split("EIDAT_PTR:")
        token = parts[-1].strip()
        if ";" in token:
            token = token.split(";", 1)[0]
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    finally:
        try:
            doc.close()
        except Exception:
            pass
