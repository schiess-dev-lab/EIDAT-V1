"""Typed value parsing for extracted cell text.

Every cell keeps its raw text; parsing adds a typed layer so trending can
consume numbers directly. Handles the patterns that dominate EIDP tables:

    "1.43"            -> number
    "73.9 psig"       -> number + unit
    "<= 5.0"          -> comparator + number     (requirement limits)
    ">= 600"          -> comparator + number
    "12.0-14.0"       -> range (lo, hi)
    "65-85"           -> range
    "600 (hold)"      -> number + qualifier
    "22 degC"         -> number + unit
    "PASS"/"FAIL"     -> verdict
"""
from __future__ import annotations

import re
from typing import Any

_NUM = r"[-+]?\d+(?:[.,]\d+)?"

_RE_VERDICT = re.compile(r"^(PASS|FAIL|N/?A)$", re.I)
_RE_COMP = re.compile(rf"^\s*(<=|>=|<|>|=)\s*({_NUM})\s*([A-Za-z%/°Ω]+.*)?$")
_RE_RANGE = re.compile(rf"^\s*({_NUM})\s*[-–—]\s*({_NUM})\s*([A-Za-z%/°Ω]+.*)?$")
_RE_NUM = re.compile(rf"^\s*({_NUM})\s*([A-Za-z%/°Ω][\w%/°Ω\.\-]*)?\s*(?:\((\w[\w\s]*)\))?\s*$")

# OCR confusions repaired ONLY in numeric context
_NUMERIC_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"})


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def parse_value(raw: str) -> dict[str, Any]:
    """Return {'kind': ..., ...} typed interpretation of a cell string."""
    text = " ".join((raw or "").split())
    if not text:
        return {"kind": "empty"}

    if _RE_VERDICT.match(text):
        return {"kind": "verdict", "value": text.upper().replace("/", "")}

    m = _RE_COMP.match(text)
    if m:
        out = {"kind": "limit", "comparator": m.group(1), "num": _to_float(m.group(2))}
        if m.group(3):
            out["unit"] = m.group(3).strip()
        return out

    m = _RE_RANGE.match(text)
    if m:
        out = {"kind": "range", "lo": _to_float(m.group(1)), "hi": _to_float(m.group(2))}
        if m.group(3):
            out["unit"] = m.group(3).strip()
        return out

    m = _RE_NUM.match(text)
    if m:
        out: dict[str, Any] = {"kind": "number", "num": _to_float(m.group(1))}
        if m.group(2):
            out["unit"] = m.group(2).strip()
        if m.group(3):
            out["qualifier"] = m.group(3).strip()
        return out

    # numeric-looking but with OCR confusions? try a repaired parse
    if re.match(r"^[\dOolISB.,\-+ ]+$", text) and any(ch.isdigit() for ch in text):
        repaired = text.translate(_NUMERIC_FIXES)
        m = _RE_NUM.match(repaired)
        if m:
            return {
                "kind": "number",
                "num": _to_float(m.group(1)),
                "repaired_from": text,
            }

    return {"kind": "text"}
