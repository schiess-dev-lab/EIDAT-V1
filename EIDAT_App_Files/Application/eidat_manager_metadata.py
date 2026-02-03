from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


REMOVED_KEYS = {"valve", "test_setup", "files", "operator", "facility", "program_code"}
ALLOWED_KEYS = {
    "program_title",
    "asset_type",
    "serial_number",
    "part_number",
    "revision",
    "test_date",
    "report_date",
    "document_type",
}

DEFAULT_CANDIDATES = {
    "program_titles": [],
    "asset_types": [
        "Valve",
        "Thruster",
        "Connector",
        "Pump",
        "Actuator",
        "Sensor",
        "Controller",
        "Regulator",
        "Manifold",
        "Motor",
        "Gearbox",
        "Battery",
        "Harness",
        "Filter",
        "Nozzle",
        "Turbine",
        "Compressor",
        "Panel",
        "Bracket",
        "Structure",
    ],
    "document_types": ["EIDP"],
}

LABEL_ALIASES = {
    "program_title": ["program title", "program name"],
    "asset_type": ["asset type", "component type", "component", "asset"],
    "serial_number": ["serial number", "serial no", "serial #", "serial"],
    "revision": ["revision", "rev"],
    "test_date": ["test date", "date of test"],
    "report_date": ["report date", "date of report"],
}

# Field name patterns that typically contain part numbers
# These are checked against table field names (keys) to extract part numbers
PART_NUMBER_FIELD_PATTERNS = [
    "part number", "part no", "part #", "p/n", "pn",
    "model", "model number", "model no", "model #",
    "item number", "item no", "item #",
    "catalog number", "catalog no", "catalog #", "cat no", "cat #",
    "product number", "product no", "product #",
    "sku", "stock number",
    "assembly number", "assy no", "assy #",
    "valve model", "pump model", "actuator model",  # asset-specific model fields
    "component model", "unit model",
]


def sanitize_metadata(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {"document_type": "EIDP"}
    cleaned = {}
    for k, v in raw.items():
        key = str(k)
        if key in REMOVED_KEYS:
            continue
        if key not in ALLOWED_KEYS:
            continue
        cleaned[key] = v
    if not cleaned.get("document_type"):
        cleaned["document_type"] = "EIDP"
    return cleaned


def load_metadata_for_pdf(pdf_path: Path) -> Optional[dict]:
    try:
        stem = pdf_path.stem
    except Exception:
        return None
    candidates = [
        pdf_path.with_name(f"{stem}_metadata.json"),
        pdf_path.with_name(f"{stem}.metadata.json"),
    ]
    for cand in candidates:
        try:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def load_metadata_from_artifacts(artifacts_dir: Path, pdf_path: Path) -> Optional[dict]:
    try:
        stem = pdf_path.stem
    except Exception:
        return None
    candidates = [
        artifacts_dir / f"{stem}_metadata.json",
        artifacts_dir / f"{stem}.metadata.json",
    ]
    for cand in candidates:
        try:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def derive_minimal_metadata(core: Any, pdf_path: Path) -> dict:
    meta = {
        "program_title": "Unknown",
        "asset_type": "Unknown",
        "serial_number": "Unknown",
        "part_number": "Unknown",
        "revision": "Unknown",
        "test_date": "Unknown",
        "report_date": "Unknown",
        "document_type": "EIDP",
    }
    return meta


def _split_table_line(line: str) -> list[str]:
    if "|" not in line:
        return []
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return [p for p in parts if p]


def _find_label_value(lines: list[str], label: str) -> Optional[str]:
    label_l = label.lower()
    label_re = re.compile(rf"\b{re.escape(label_l)}\b", flags=re.IGNORECASE)
    for idx, line in enumerate(lines):
        if not line:
            continue
        parts = _split_table_line(line)
        if parts:
            key = parts[0].lower()
            if label_l == key or label_l in key or key in label_l:
                return parts[1].strip() if len(parts) > 1 else None
            continue
        if not label_re.search(line.lower()):
            continue
        chunk = line
        if ":" in chunk:
            chunk = chunk.split(":", 1)[1]
        elif "-" in chunk:
            chunk = chunk.split("-", 1)[1]
        else:
            chunk = chunk.replace(label, "")
        value = chunk.strip()
        if value:
            return value
        if idx + 1 < len(lines):
            nxt = lines[idx + 1].strip()
            if nxt and not nxt.startswith("+") and "|" not in nxt:
                return nxt
    return None


def _find_program_title(lines: list[str]) -> Optional[str]:
    title_keys = {"title", "document title"}
    program_keys = {"program", "program title", "program name"}
    for line in lines:
        parts = _split_table_line(line)
        if parts and parts[0].lower() in title_keys:
            return parts[1].strip() if len(parts) > 1 else None
    for line in lines:
        parts = _split_table_line(line)
        if parts and parts[0].lower() in program_keys:
            return parts[1].strip() if len(parts) > 1 else None
        line_l = line.lower().strip()
        if not line_l.startswith("program"):
            continue
        if "program code" in line_l:
            continue
        if ":" in line:
            return line.split(":", 1)[1].strip()
        if "-" in line:
            return line.split("-", 1)[1].strip()
    return None


def _find_asset_type(text: str, asset_types: list[str]) -> Optional[str]:
    if not text:
        return None
    for asset in asset_types:
        if not asset:
            continue
        if re.search(rf"\b{re.escape(asset)}\b", text, flags=re.IGNORECASE):
            return str(asset)
    return None


def _find_asset_type_from_fields(lines: list[str], asset_types: list[str]) -> Optional[str]:
    for line in lines:
        parts = _split_table_line(line)
        if not parts:
            continue
        key = parts[0].lower()
        for asset in asset_types:
            asset_l = str(asset).lower()
            if asset_l and asset_l in key:
                return str(asset)
    return None


def _find_part_number_from_fields(lines: list[str], max_lines: int = 150) -> Optional[str]:
    """
    Find part number by looking for table fields with model/part-related names.
    Only searches the first max_lines to focus on header tables (pages 1-2).
    """
    for line in lines[:max_lines]:
        parts = _split_table_line(line)
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        value = parts[1].strip()
        if not value:
            continue
        # Check if the field name matches any part number pattern
        for pattern in PART_NUMBER_FIELD_PATTERNS:
            if pattern in key or key in pattern:
                # Validate: part numbers typically have letters and/or numbers, often with dashes
                # Reject values that are too short or look like other fields
                if len(value) >= 3 and re.search(r"[A-Za-z0-9]", value):
                    return value
    return None


def _normalize_revision(value: str) -> Optional[str]:
    if not value:
        return None
    val = value.strip().upper()
    if re.fullmatch(r"[A-Z]{1,2}", val):
        return val
    return None


def _normalize_date(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip()
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", v)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{int(y):04d}-{mo:02d}-{d:02d}"
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](20\d{2})", v)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def _load_candidates() -> dict:
    try:
        root = Path(__file__).resolve().parents[2]
        cand_path = root / "user_inputs" / "metadata_candidates.json"
        if cand_path.exists():
            data = json.loads(cand_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return DEFAULT_CANDIDATES


def _match_from_list(value: str, candidates: list[str]) -> Optional[str]:
    if not value or not candidates:
        return None
    val_norm = re.sub(r"\s+", " ", value).strip().lower()
    for cand in candidates:
        if val_norm == re.sub(r"\s+", " ", str(cand)).strip().lower():
            return str(cand)
    return None


def _find_asset_type_from_filename(pdf_path: Optional[Path], asset_types: list[str]) -> Optional[str]:
    """Check if any asset type appears in the PDF filename."""
    if not pdf_path:
        return None
    filename = pdf_path.stem.lower()
    for asset in asset_types:
        if not asset:
            continue
        if re.search(rf"\b{re.escape(asset.lower())}\b", filename) or asset.lower() in filename:
            return str(asset)
    return None


def _find_asset_type_by_frequency(first_page_text: str, asset_types: list[str]) -> Optional[str]:
    """Find asset type by counting mentions on the first page."""
    if not first_page_text:
        return None
    text_lower = first_page_text.lower()
    best_asset = None
    best_count = 0
    for asset in asset_types:
        if not asset:
            continue
        # Count occurrences (case insensitive, word boundary)
        count = len(re.findall(rf"\b{re.escape(asset.lower())}\b", text_lower))
        if count > best_count:
            best_count = count
            best_asset = asset
    # Require at least 2 mentions to use this heuristic
    if best_count >= 2:
        return str(best_asset)
    return None


def extract_metadata_from_text(text: str, *, asset_types: Optional[list[str]] = None, pdf_path: Optional[Path] = None) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = _load_candidates()
    asset_list = asset_types or candidates.get("asset_types") or []
    program_titles = candidates.get("program_titles") or []
    doc_types = candidates.get("document_types") or []
    meta: dict[str, Optional[str]] = {
        "program_title": None,
        "asset_type": None,
        "serial_number": None,
        "part_number": None,
        "revision": None,
        "test_date": None,
        "report_date": None,
        "document_type": None,
    }

    header_line = ""
    for line in lines[:80]:
        if "data package" in line.lower():
            header_line = line
            break
    if header_line:
        m = re.search(
            r"^(?P<title>.+?)\s+-\s+(?P<asset>.+?)\s+end item data package\s+(?P<serial>SN\d+)?\s*\|\s*(?P<date>\d{4}-\d{2}-\d{2})",
            header_line,
            flags=re.IGNORECASE,
        )
        if m:
            if not meta.get("program_title"):
                meta["program_title"] = m.group("title").strip()
            if not meta.get("asset_type"):
                asset = _find_asset_type(m.group("asset"), asset_list)
                meta["asset_type"] = asset or m.group("asset").strip().title()
            if not meta.get("serial_number"):
                meta["serial_number"] = (m.group("serial") or "").strip()
            if not meta.get("report_date"):
                meta["report_date"] = (m.group("date") or "").strip()
            if not meta.get("test_date"):
                meta["test_date"] = (m.group("date") or "").strip()

    if not meta.get("program_title"):
        meta["program_title"] = _find_program_title(lines)

    for key, aliases in LABEL_ALIASES.items():
        for label in aliases:
            val = _find_label_value(lines, label)
            if val:
                meta[key] = val.strip()
                break

    # If asset_type was found via label (e.g. "Control Valve (End Item)"),
    # try to extract the actual asset type from it
    if meta.get("asset_type"):
        extracted_asset = _find_asset_type(meta["asset_type"], asset_list)
        if extracted_asset:
            meta["asset_type"] = extracted_asset
        else:
            # Value doesn't contain a known asset type, clear it so heuristics can try
            meta["asset_type"] = None

    if not meta.get("asset_type"):
        field_asset = _find_asset_type_from_fields(lines, asset_list)
        if field_asset:
            meta["asset_type"] = field_asset

    if not meta.get("asset_type"):
        asset = _find_asset_type("\n".join(lines[:200]), asset_list)
        if asset:
            meta["asset_type"] = asset

    # Heuristic: check PDF filename for asset type
    if not meta.get("asset_type"):
        filename_asset = _find_asset_type_from_filename(pdf_path, asset_list)
        if filename_asset:
            meta["asset_type"] = filename_asset

    # Heuristic: check frequency of asset type mentions on first page
    if not meta.get("asset_type"):
        # Extract first page text (up to first "=== Page 2 ===" marker or first 200 lines)
        first_page_lines = []
        for line in lines:
            if "=== Page 2 ===" in line or "=== Page 3 ===" in line:
                break
            first_page_lines.append(line)
            if len(first_page_lines) >= 200:
                break
        first_page_text = "\n".join(first_page_lines)
        freq_asset = _find_asset_type_by_frequency(first_page_text, asset_list)
        if freq_asset:
            meta["asset_type"] = freq_asset

    if not meta.get("serial_number"):
        m = re.search(r"\bSN\d+\b", "\n".join(lines[:200]), flags=re.IGNORECASE)
        if m:
            meta["serial_number"] = m.group(0).upper()

    # Extract part number from table fields (model, part number, etc.)
    if not meta.get("part_number"):
        part_num = _find_part_number_from_fields(lines)
        if part_num:
            meta["part_number"] = part_num

    if meta.get("revision"):
        meta["revision"] = _normalize_revision(meta["revision"] or "") or "Unknown"
    if meta.get("test_date"):
        meta["test_date"] = _normalize_date(meta["test_date"] or "") or "Unknown"
    if meta.get("report_date"):
        meta["report_date"] = _normalize_date(meta["report_date"] or "") or "Unknown"

    if program_titles:
        match = _match_from_list(meta.get("program_title") or "", program_titles)
        meta["program_title"] = match or "Unknown"

    if asset_list:
        match = _match_from_list(meta.get("asset_type") or "", asset_list)
        meta["asset_type"] = match or "Unknown"

    if doc_types:
        doc_match = None
        for dt in doc_types:
            if re.search(rf"\b{re.escape(str(dt))}\b", "\n".join(lines[:120]), flags=re.IGNORECASE):
                doc_match = str(dt)
                break
        meta["document_type"] = doc_match or ("EIDP" if "EIDP" in doc_types else "Unknown")
    else:
        meta["document_type"] = "Unknown"

    for key in ["program_title", "asset_type", "serial_number", "part_number", "revision", "test_date", "report_date", "document_type"]:
        if not meta.get(key):
            meta[key] = "Unknown"

    return {k: v for k, v in meta.items() if v is not None}


def write_metadata(artifacts_dir: Path, pdf_path: Path, metadata: dict) -> Optional[Path]:
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    target = artifacts_dir / f"{pdf_path.stem}_metadata.json"
    try:
        target.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception:
        return None
    return target


def normalize_title(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", (value or "").lower())
    return s.strip()
