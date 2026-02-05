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
    "document_type_acronym",
    "vendor",
    "acceptance_test_plan_number",
}

DEFAULT_CANDIDATES = {
    "program_titles": [],
    "vendors": [],
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
    # Back-compat: this can be a list[str] or list[{"name","acronym","aliases"}]
    "document_types": ["EIDP", "Excel file data"],
}

LABEL_ALIASES = {
    "program_title": ["program title", "program name"],
    "asset_type": ["asset type", "component type", "component", "asset"],
    "serial_number": ["serial number", "serial no", "serial #", "serial"],
    "revision": ["revision", "rev"],
    "test_date": ["test date", "date of test"],
    "report_date": ["report date", "date of report"],
    "vendor": ["vendor", "supplier", "manufacturer", "mfr", "oem"],
    "acceptance_test_plan_number": ["acceptance test plan", "test plan", "acceptance plan", "atp", "plan number"],
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


def sanitize_metadata(raw: Any, *, default_document_type: str = "EIDP") -> dict:
    if not isinstance(raw, dict):
        return {"document_type": default_document_type}
    cleaned = {}
    for k, v in raw.items():
        key = str(k)
        if key in REMOVED_KEYS:
            continue
        if key not in ALLOWED_KEYS:
            continue
        cleaned[key] = v
    if not cleaned.get("document_type"):
        cleaned["document_type"] = default_document_type
    if not cleaned.get("document_type_acronym") and cleaned.get("document_type") and cleaned.get("document_type") != "Unknown":
        cleaned["document_type_acronym"] = cleaned.get("document_type")
    return cleaned


def _first_nonempty(lines: list[str], max_lines: int = 120) -> str:
    out: list[str] = []
    for ln in lines[: int(max_lines)]:
        s = str(ln or "").strip()
        if s:
            out.append(s)
    return "\n".join(out)


def _match_from_candidates(value: str, candidates: list[str]) -> Optional[str]:
    if not value:
        return None
    return _match_from_list(value, candidates)


def _iter_doc_type_entries(raw: Any) -> list[dict]:
    """
    Normalize document type candidates.

    Accepts either:
      - list[str] (treated as {"name": s, "acronym": s, "aliases":[s]})
      - list[dict] with keys name/acronym/aliases
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str) and item.strip():
            s = item.strip()
            out.append({"name": s, "acronym": s, "aliases": [s]})
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        acronym = str(item.get("acronym") or "").strip()
        aliases = item.get("aliases")
        alias_list: list[str] = []
        if isinstance(aliases, list):
            alias_list = [str(a).strip() for a in aliases if str(a).strip()]
        if not alias_list and name:
            alias_list = [name]
        if not name and alias_list:
            name = alias_list[0]
        if not acronym and name:
            acronym = name
        if name:
            out.append({"name": name, "acronym": acronym, "aliases": alias_list})
    return out


def _pick_document_type(text: str, entries: list[dict]) -> tuple[Optional[str], Optional[str]]:
    """
    Return (document_type, acronym) based on first match in text.

    document_type is standardized to acronym when present.
    """
    if not text or not entries:
        return None, None
    for e in entries:
        aliases = e.get("aliases") or []
        for a in aliases:
            if not a:
                continue
            if re.search(rf"\b{re.escape(str(a))}\b", text, flags=re.IGNORECASE):
                acr = str(e.get("acronym") or "").strip() or None
                name = str(e.get("name") or "").strip() or None
                return (acr or name), acr
    return None, None


def _extract_plan_number(text: str) -> Optional[str]:
    """
    Extract an acceptance test plan identifier from free text.

    Examples:
      - ATP-1234
      - Acceptance Test Plan: ATP 1234
      - Test Plan TP-001
    """
    if not text:
        return None
    patterns = [
        r"\b(?:ACCEPTANCE\s+TEST\s+PLAN|TEST\s+PLAN|ACCEPTANCE\s+PLAN)\s*(?:NO\.|NUMBER|#|:)?\s*([A-Z]{1,6}[-_\s]?\d{1,8}[A-Z0-9\-_\/]{0,16})\b",
        r"\b(?:ATP|TP)\s*[-_#:]?\s*(\d{1,8}[A-Z0-9\-_\/]{0,16})\b",
        r"\b(ATP[-_ ]?\d{1,8}[A-Z0-9\-_\/]{0,16})\b",
        r"\b(TP[-_ ]?\d{1,8}[A-Z0-9\-_\/]{0,16})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = (m.group(1) or "").strip()
            if val:
                return re.sub(r"\s+", "", val.upper())
    return None


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
        "document_type_acronym": "EIDP",
        "vendor": "Unknown",
        "acceptance_test_plan_number": "Unknown",
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
    vendors = candidates.get("vendors") or []
    doc_type_entries = _iter_doc_type_entries(candidates.get("document_types") or [])
    meta: dict[str, Optional[str]] = {
        "program_title": None,
        "asset_type": None,
        "serial_number": None,
        "part_number": None,
        "revision": None,
        "test_date": None,
        "report_date": None,
        "document_type": None,
        "document_type_acronym": None,
        "vendor": None,
        "acceptance_test_plan_number": None,
    }

    first_page_text = _first_nonempty(lines, max_lines=220)
    title_blob = _first_nonempty(lines, max_lines=60)
    if pdf_path is not None:
        try:
            title_blob = f"{title_blob}\n{pdf_path.stem}"
        except Exception:
            pass

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

    # Vendor: match known vendor candidates if not found by label.
    if not meta.get("vendor") and vendors:
        for v in vendors:
            sv = str(v or "").strip()
            if not sv:
                continue
            if re.search(rf"\b{re.escape(sv)}\b", title_blob, flags=re.IGNORECASE):
                meta["vendor"] = sv
                break
            if re.search(rf"\b{re.escape(sv)}\b", first_page_text, flags=re.IGNORECASE):
                meta["vendor"] = sv
                break

    # Acceptance test plan number: try labels and free-text patterns.
    if not meta.get("acceptance_test_plan_number"):
        meta["acceptance_test_plan_number"] = _extract_plan_number(title_blob) or _extract_plan_number(first_page_text)

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
        first_page_text_freq = "\n".join(first_page_lines)
        freq_asset = _find_asset_type_by_frequency(first_page_text_freq, asset_list)
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

    # Document type: match on early-page/title text, standardize to acronym when available.
    doc_match, acr = _pick_document_type(title_blob + "\n" + _first_nonempty(lines, max_lines=120), doc_type_entries)
    if doc_match:
        meta["document_type"] = doc_match
        meta["document_type_acronym"] = acr or doc_match
    else:
        meta["document_type"] = "Unknown"
        meta["document_type_acronym"] = None

    for key in [
        "program_title",
        "asset_type",
        "serial_number",
        "part_number",
        "revision",
        "test_date",
        "report_date",
        "document_type",
        "vendor",
        "acceptance_test_plan_number",
    ]:
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
