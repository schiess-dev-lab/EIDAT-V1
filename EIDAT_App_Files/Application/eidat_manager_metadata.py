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
    "excel_sqlite_rel",
    "file_extension",
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
    "part_number": ["part number", "part no", "part no.", "part #", "p/n", "pn", "par tno", "par t no", "par tno."],
    "revision": ["revision", "rev"],
    "test_date": ["test date", "date of test"],
    "report_date": ["report date", "date of report"],
    "vendor": ["vendor", "supplier", "manufacturer", "mfr", "oem"],
    "acceptance_test_plan_number": ["acceptance test plan", "test plan", "acceptance plan", "atp", "plan number"],
}

# Field name patterns that typically contain part numbers
# These are checked against table field names (keys) to extract part numbers
PART_NUMBER_FIELD_PATTERNS = [
    "part number", "part no", "part #", "p/n", "pn", "par tno", "par t no",
    "model", "model number", "model no", "model #",
    "item number", "item no", "item #",
    "catalog number", "catalog no", "catalog #", "cat no", "cat #",
    "product number", "product no", "product #",
    "sku", "stock number",
    "assembly number", "assy no", "assy #",
    "valve model", "pump model", "actuator model",  # asset-specific model fields
    "component model", "unit model",
]


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _merge_label_aliases(candidates: dict) -> dict[str, list[str]]:
    """
    Merge built-in LABEL_ALIASES with optional user overrides in user_inputs/metadata_candidates.json:
      { "label_aliases": { "<allowed_key>": ["alias1", ...] } }
    Unknown keys are ignored. Aliases are deduped, preserving order.
    """
    merged: dict[str, list[str]] = {k: list(v) for k, v in LABEL_ALIASES.items()}
    raw = candidates.get("label_aliases") if isinstance(candidates, dict) else None
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k or "").strip()
            if not key or key not in ALLOWED_KEYS:
                continue
            if not isinstance(v, list):
                continue
            merged.setdefault(key, [])
            merged[key].extend([str(x or "").strip() for x in v])
    return {k: _dedupe_strings(v) for k, v in merged.items()}


def _merge_part_number_field_patterns(candidates: dict) -> list[str]:
    """
    Extend PART_NUMBER_FIELD_PATTERNS with optional user additions in user_inputs/metadata_candidates.json:
      { "part_number_field_patterns": ["pattern1", ...] }
    """
    out = list(PART_NUMBER_FIELD_PATTERNS)
    raw = candidates.get("part_number_field_patterns") if isinstance(candidates, dict) else None
    if isinstance(raw, list):
        out.extend([str(x or "").strip() for x in raw])
    return _dedupe_strings(out)


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        if s.casefold() == "unknown":
            return True
    return False


def _as_clean_str(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _infer_serial_from_path(path: Path) -> str | None:
    try:
        stem = str(path.stem or "")
    except Exception:
        stem = str(path)
    m = re.search(r"SN\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()
    return None


def _normalize_serial(value: object) -> str | None:
    s = _as_clean_str(value)
    if not s:
        return None
    s0 = s.strip().upper().replace(" ", "")
    s0 = s0.replace("S/N", "SN")
    if re.fullmatch(r"SN\d{1,12}", s0):
        return s0
    if re.fullmatch(r"\d{1,8}", s0):
        # Best-effort: treat raw digits as SN; pad short values to 4 digits.
        digits = s0
        if len(digits) <= 4:
            digits = digits.zfill(4)
        return f"SN{digits}"
    return s.strip().upper()


def _best_candidate_match(value: str, candidates: list[str]) -> str | None:
    """
    Return best candidate that matches value.
    Preference:
      1) exact match (case/space insensitive)
      2) longest candidate contained in value (case insensitive)
    """
    if not value or not candidates:
        return None
    v_norm = re.sub(r"\s+", " ", value).strip().lower()
    best = None
    for c in candidates:
        c_s = str(c or "").strip()
        if not c_s:
            continue
        c_norm = re.sub(r"\s+", " ", c_s).strip().lower()
        if not c_norm:
            continue
        if v_norm == c_norm:
            return c_s
        if c_norm in v_norm:
            if best is None or len(c_norm) > len(re.sub(r"\s+", " ", best).strip().lower()):
                best = c_s
    return best


def canonicalize_metadata_for_file(
    abs_path: Path,
    *,
    existing_meta: Any = None,
    extracted_meta: Any = None,
    default_document_type: str | None = None,
) -> dict:
    """
    Produce a single canonical metadata dict for a file, with stable precedence:
      - existing_meta wins unless missing/Unknown
      - else extracted_meta
      - else derived-from-filename/path

    Then normalize fields (doc type pairing, dates, revision, serial format, candidates).
    """
    p = Path(abs_path).expanduser()
    ext = str(p.suffix or "").lower()
    is_excel = ext in {".xlsx", ".xls", ".xlsm"}

    existing = existing_meta if isinstance(existing_meta, dict) else {}
    extracted = extracted_meta if isinstance(extracted_meta, dict) else {}

    if default_document_type is None:
        default_document_type = "Data file" if is_excel else "EIDP"

    candidates = _load_candidates()
    program_titles = candidates.get("program_titles") or []
    vendors = candidates.get("vendors") or []
    asset_types = candidates.get("asset_types") or []

    derived: dict[str, object] = {}
    derived["file_extension"] = ext or "Unknown"

    # Serial fallback from filename.
    sn_from_name = _infer_serial_from_path(p)
    if sn_from_name:
        derived["serial_number"] = sn_from_name

    # Excel filename identity can carry serial/program.
    if is_excel:
        try:
            import importlib.util

            project_root = Path(__file__).resolve().parents[1]
            mod_path = project_root / "scripts" / "excel_extraction.py"
            spec = importlib.util.spec_from_file_location("excel_extraction", mod_path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                program, vehicle, serial = mod.derive_file_identity(p)  # type: ignore[attr-defined]
            else:
                program = vehicle = serial = ""
        except Exception:
            program = vehicle = serial = ""
        if not _is_missing(serial) and str(serial).strip():
            derived["serial_number"] = str(serial).strip()
        if (not _is_missing(program) and str(program).strip()) or (not _is_missing(vehicle) and str(vehicle).strip()):
            title = ""
            if str(program or "").strip() and str(vehicle or "").strip():
                title = f"{str(program).strip()} {str(vehicle).strip()}".strip()
            else:
                title = (str(program or "").strip() or str(vehicle or "").strip())
            if title:
                derived["program_title"] = title

    # Program/vendor/asset from filename (contains-match against candidates).
    try:
        stem_blob = re.sub(r"[_\\-]+", " ", str(p.stem or "")).strip()
    except Exception:
        stem_blob = str(p)
    if stem_blob:
        pm = _best_candidate_match(stem_blob, [str(x) for x in program_titles if str(x).strip()])
        if pm:
            derived["program_title"] = pm
        vm = _best_candidate_match(stem_blob, [str(x) for x in vendors if str(x).strip()])
        if vm:
            derived["vendor"] = vm
        am = _best_candidate_match(stem_blob, [str(x) for x in asset_types if str(x).strip()])
        if am:
            derived["asset_type"] = am

    # Doc type inference from filename/path.
    inferred_dt, inferred_acr = _infer_doc_type_from_filename(p)
    if inferred_dt:
        derived["document_type"] = inferred_dt
    if inferred_acr:
        derived["document_type_acronym"] = inferred_acr

    merged: dict[str, object] = {}
    for key in ALLOWED_KEYS:
        if key == "excel_sqlite_rel":
            # Keep optional; only set when present in some input.
            pass
        v = existing.get(key)
        if not _is_missing(v):
            merged[key] = v
            continue
        v = extracted.get(key)
        if not _is_missing(v):
            merged[key] = v
            continue
        v = derived.get(key)
        if not _is_missing(v):
            merged[key] = v

    # Always set extension deterministically.
    merged["file_extension"] = ext or "Unknown"

    # Normalize serial format.
    sn = _normalize_serial(merged.get("serial_number"))
    if sn:
        merged["serial_number"] = sn

    # Normalize revision + dates (only if present).
    rev = _normalize_revision(_as_clean_str(merged.get("revision")))
    if rev:
        merged["revision"] = rev
    td = _normalize_date(_as_clean_str(merged.get("test_date")))
    if td:
        merged["test_date"] = td
    rd = _normalize_date(_as_clean_str(merged.get("report_date")))
    if rd:
        merged["report_date"] = rd

    # Doc type pairing defaults.
    def _meta_is_td(m: dict) -> bool:
        try:
            dt = str(m.get("document_type") or "").strip().upper()
        except Exception:
            dt = ""
        try:
            acr = str(m.get("document_type_acronym") or "").strip().upper()
        except Exception:
            acr = ""
        return dt == "TD" or acr == "TD"

    is_td = _meta_is_td(existing) or _meta_is_td(extracted) or _meta_is_td(derived) or _meta_is_td(merged)
    if is_td or (str(inferred_dt or "").strip().upper() == "TD") or (str(inferred_acr or "").strip().upper() == "TD"):
        merged["document_type"] = "TD"
        merged["document_type_acronym"] = "TD"
    else:
        dt_val = _as_clean_str(merged.get("document_type"))
        acr_val = _as_clean_str(merged.get("document_type_acronym"))
        if _is_missing(dt_val):
            dt_val = str(default_document_type or "").strip() or ("Data file" if is_excel else "EIDP")
        if _is_missing(acr_val):
            if dt_val.strip().casefold() == "data file":
                acr_val = "DATA"
            else:
                acr_val = dt_val
        # Ensure Data file acronym is DATA.
        if dt_val.strip().casefold() == "data file":
            acr_val = "DATA"
        merged["document_type"] = dt_val
        merged["document_type_acronym"] = acr_val

    # Candidate-based canonicalization.
    try:
        pt = _as_clean_str(merged.get("program_title"))
        ptm = _best_candidate_match(pt, [str(x) for x in program_titles if str(x).strip()])
        if ptm:
            merged["program_title"] = ptm
    except Exception:
        pass
    try:
        vv = _as_clean_str(merged.get("vendor"))
        vvm = _best_candidate_match(vv, [str(x) for x in vendors if str(x).strip()])
        if vvm:
            merged["vendor"] = vvm
    except Exception:
        pass
    try:
        av = _as_clean_str(merged.get("asset_type"))
        avm = _best_candidate_match(av, [str(x) for x in asset_types if str(x).strip()])
        if avm:
            merged["asset_type"] = avm
    except Exception:
        pass

    # Fill common keys with Unknown so JSON and index consumers are stable.
    for k in [
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
        "file_extension",
    ]:
        if k in ALLOWED_KEYS and _is_missing(merged.get(k)):
            merged[k] = "Unknown"

    # Preserve excel_sqlite_rel if present in any input and non-missing.
    for src in (existing, extracted, derived):
        v = src.get("excel_sqlite_rel")
        if not _is_missing(v):
            merged["excel_sqlite_rel"] = v
            break

    return sanitize_metadata(merged, default_document_type=str(default_document_type or "").strip() or ("Data file" if is_excel else "EIDP"))


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


def _strip_doc_type_lines(lines: list[str]) -> list[str]:
    """Remove lines that are likely just document-type headers (to avoid asset-type false positives)."""
    out: list[str] = []
    for ln in lines:
        s = str(ln or "").strip()
        if not s:
            continue
        s_l = s.lower()
        if "end item data package" in s_l or "end-item data package" in s_l:
            continue
        if "data package" in s_l and "eidp" in s_l:
            continue
        out.append(s)
    return out


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
            try:
                chunk = re.sub(re.escape(label), "", chunk, flags=re.IGNORECASE)
            except Exception:
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


def _find_part_number_from_fields(lines: list[str], patterns: list[str], max_lines: int = 150) -> Optional[str]:
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
        for pattern in patterns:
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


def _infer_doc_type_from_filename(path: Optional[Path]) -> tuple[Optional[str], Optional[str]]:
    """
    Heuristic fallback when the document text doesn't contain a recognizable doc type.
    """
    if path is None:
        return None, None
    try:
        blob = (path.stem or "")
    except Exception:
        blob = str(path).lower()

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    blob_norm = _norm(str(blob))
    blob_tokens = set(blob_norm.split())

    # Prefer user-configured doc-type candidates + aliases (metadata_candidates.json).
    try:
        cand = _load_candidates()
        entries = _iter_doc_type_entries(cand.get("document_types") if isinstance(cand, dict) else [])
    except Exception:
        entries = []

    best: tuple[int, str | None, str | None] | None = None  # (alias_len, name, acronym)
    for e in entries:
        name = str(e.get("name") or "").strip() or None
        acr = str(e.get("acronym") or "").strip() or None
        aliases = list(e.get("aliases") or [])
        if name:
            aliases.append(name)
        if acr:
            aliases.append(acr)
        for a in aliases:
            a_s = str(a or "").strip()
            if not a_s:
                continue
            a_norm = _norm(a_s)
            if not a_norm:
                continue
            hit = False
            if len(a_norm) <= 3 and len(a_norm.split()) == 1:
                # Short token like "TD" should match whole-token only.
                hit = a_norm in blob_tokens
            else:
                hit = a_norm in blob_norm
            if not hit:
                continue
            score = len(a_norm)
            if best is None or score > best[0]:
                best = (score, (acr or name), acr)

    if best is not None:
        return best[1], best[2]

    # Back-compat hardcoded fallback (only if candidate parsing failed).
    if "test data" in blob_norm or "testdata" in blob_norm:
        return "TD", "TD"
    if "td" in blob_tokens:
        return "TD", "TD"
    return None, None


def extract_metadata_from_text(text: str, *, asset_types: Optional[list[str]] = None, pdf_path: Optional[Path] = None) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    asset_lines = _strip_doc_type_lines(lines)
    candidates = _load_candidates()
    label_aliases = _merge_label_aliases(candidates if isinstance(candidates, dict) else {})
    pn_patterns = _merge_part_number_field_patterns(candidates if isinstance(candidates, dict) else {})
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

    for key, aliases in label_aliases.items():
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
        field_asset = _find_asset_type_from_fields(asset_lines, asset_list)
        if field_asset:
            meta["asset_type"] = field_asset

    if not meta.get("asset_type"):
        asset = _find_asset_type("\n".join(asset_lines[:200]), asset_list)
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
        for line in asset_lines:
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
        part_num = _find_part_number_from_fields(lines, pn_patterns)
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

    # Document type:
    # 1) Always allow the filename to force TD (many TD files won't contain explicit "Test Data" text).
    # 2) Otherwise, match on early-page/title text and standardize to acronym when available.
    inferred_dt, inferred_acr = _infer_doc_type_from_filename(pdf_path)
    doc_match, acr = _pick_document_type(title_blob + "\n" + _first_nonempty(lines, max_lines=120), doc_type_entries)
    if inferred_dt and str(inferred_dt).strip().upper() == "TD":
        meta["document_type"] = "TD"
        meta["document_type_acronym"] = "TD"
    elif doc_match:
        meta["document_type"] = doc_match
        meta["document_type_acronym"] = acr or doc_match
    elif inferred_dt:
        meta["document_type"] = inferred_dt
        meta["document_type_acronym"] = inferred_acr or inferred_dt
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


def extract_metadata_from_excel(
    excel_path: Path,
    *,
    max_sheets: int = 6,
    max_rows: int = 120,
    max_cols: int = 30,
) -> dict:
    """
    Extract metadata from an Excel workbook by sampling early sheet cells and reusing the
    standard text-based metadata extractor.

    This builds a small "table-like" text blob with `|` separators so the existing parsing
    heuristics (label/value, table fields, doc type candidates) work consistently across PDFs and Excel.
    """
    p = Path(excel_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))

    lines: list[str] = []
    try:
        stem = str(p.stem)
        lines.append(stem)
        # Add separator-normalized variants so candidate matching can succeed on `Test_Data`, etc.
        lines.append(re.sub(r"[_\\-]+", " ", stem))
    except Exception:
        pass

    candidates = _load_candidates()
    label_aliases = _merge_label_aliases(candidates if isinstance(candidates, dict) else {})

    meta: dict
    try:
        from openpyxl import load_workbook  # type: ignore

        wb = load_workbook(str(p), read_only=True, data_only=True)
        try:
            sheetnames = list(getattr(wb, "sheetnames", []) or [])
            synthetic: list[str] = []
            seen_synth: set[tuple[str, str]] = set()

            def _norm(s: str) -> str:
                return re.sub(r"[^a-z0-9]+", "", str(s or "").strip().lower())

            preferred_label: dict[str, str] = {}
            for k, aliases in label_aliases.items():
                if not aliases:
                    continue
                preferred_label[k] = str(aliases[0]).strip() or k

            compiled_value: dict[str, list[re.Pattern]] = {}
            compiled_label_only: dict[str, list[re.Pattern]] = {}
            for k, aliases in label_aliases.items():
                value_pats: list[re.Pattern] = []
                label_pats: list[re.Pattern] = []
                for a in aliases:
                    a_s = str(a or "").strip()
                    if not a_s:
                        continue
                    try:
                        value_pats.append(
                            re.compile(
                                rf"^\s*{re.escape(a_s)}\s*(?:[:=\-#]\s*)?(?P<val>.+?)\s*$",
                                flags=re.IGNORECASE,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        label_pats.append(
                            re.compile(
                                rf"^\s*{re.escape(a_s)}\s*(?:[:=\-#]\s*)?\s*$",
                                flags=re.IGNORECASE,
                            )
                        )
                    except Exception:
                        pass
                if value_pats:
                    compiled_value[k] = value_pats
                if label_pats:
                    compiled_label_only[k] = label_pats

            all_alias_norm: set[str] = set()
            for aliases in label_aliases.values():
                for a in aliases:
                    na = _norm(str(a or ""))
                    if na:
                        all_alias_norm.add(na)

            def _looks_like_part_number(v: str) -> bool:
                s = str(v or "").strip()
                if len(s) < 3 or len(s) > 80:
                    return False
                if not re.search(r"[A-Za-z0-9]", s):
                    return False
                if len(s.split()) > 6:
                    return False
                return True

            for sheet_name in sheetnames[: int(max_sheets)]:
                try:
                    ws = wb[sheet_name]
                except Exception:
                    continue
                try:
                    lines.append(f"Sheet | {sheet_name}")
                except Exception:
                    pass

                # Excel-only: recover label/value fields from table-style layouts.
                # Example: "Part No" in one cell, value in the cell to the right.
                if compiled_value and len(synthetic) < 80:
                    try:
                        grid_rows = list(
                            ws.iter_rows(
                                min_row=1,
                                max_row=int(max_rows),
                                min_col=1,
                                max_col=int(max_cols),
                                values_only=True,
                            )
                        )
                    except Exception:
                        grid_rows = []
                    for row in grid_rows:
                        if len(synthetic) >= 80:
                            break
                        row_vals = list(row or [])
                        for ci, raw in enumerate(row_vals):
                            if len(synthetic) >= 80:
                                break
                            if raw is None:
                                continue
                            cell_text = str(raw).strip()
                            if not cell_text or len(cell_text) > 200:
                                continue

                            for key, pats in compiled_value.items():
                                label_only = compiled_label_only.get(key) or []
                                found_val: Optional[str] = None
                                label_only_hit = any(lp.match(cell_text) for lp in label_only) if label_only else False

                                # Same-cell value (e.g., "Part No: XYZ-123")
                                for pat in pats:
                                    m = pat.match(cell_text)
                                    if not m:
                                        continue
                                    cand_val = str(m.group("val") or "").strip()
                                    if not cand_val:
                                        continue
                                    if key == "part_number" and not _looks_like_part_number(cand_val):
                                        continue
                                    found_val = cand_val
                                    break

                                # Part number: allow "label <space> value" in same cell.
                                if found_val is None and key == "part_number":
                                    for a in (label_aliases.get("part_number") or []):
                                        a_s = str(a or "").strip()
                                        if not a_s:
                                            continue
                                        if cell_text.lower().startswith(a_s.lower() + " "):
                                            cand_val = cell_text[len(a_s) :].strip()
                                            if _looks_like_part_number(cand_val):
                                                found_val = cand_val
                                                break

                                # Right-cell scan (up to 3 cells) when the label cell is value-less.
                                if found_val is None and label_only_hit:
                                    for step in (1, 2, 3):
                                        if ci + step >= len(row_vals):
                                            break
                                        rv = row_vals[ci + step]
                                        if rv is None:
                                            continue
                                        rv_s = str(rv).strip()
                                        if not rv_s or len(rv_s) > 200:
                                            continue
                                        if _norm(rv_s) in all_alias_norm:
                                            continue
                                        if key == "part_number" and not _looks_like_part_number(rv_s):
                                            continue
                                        found_val = rv_s
                                        break

                                if found_val:
                                    k_norm = _norm(key)
                                    v_norm = _norm(found_val)
                                    if not k_norm or not v_norm:
                                        continue
                                    tup = (k_norm, v_norm)
                                    if tup in seen_synth:
                                        continue
                                    seen_synth.add(tup)
                                    synth_key = preferred_label.get(key) or key
                                    synthetic.append(f"{synth_key} | {found_val}")
                                    break

                try:
                    row_iter = ws.iter_rows(
                        min_row=1,
                        max_row=int(max_rows),
                        min_col=1,
                        max_col=int(max_cols),
                        values_only=True,
                    )
                except Exception:
                    continue

                for row in row_iter:
                    vals: list[str] = []
                    for v in list(row or [])[: int(max_cols)]:
                        if v is None:
                            continue
                        s = str(v).strip()
                        if not s:
                            continue
                        if len(s) > 160:
                            s = s[:160]
                        vals.append(s)
                        if len(vals) >= 6:
                            break
                    if len(vals) >= 2:
                        lines.append(" | ".join(vals))

            if synthetic:
                lines.append("Sheet | Extracted Labels")
                lines.extend(synthetic[:80])
        finally:
            try:
                wb.close()
            except Exception:
                pass
    except Exception:
        # Best-effort fallback (e.g., unsupported .xls): extract from filename/title only.
        pass

    blob = "\n".join([ln for ln in lines if str(ln).strip()])
    meta = extract_metadata_from_text(blob, pdf_path=p)

    # Fallback: derive program/title + serial from filename if missing.
    try:
        import importlib.util

        project_root = Path(__file__).resolve().parents[1]
        mod_path = project_root / "scripts" / "excel_extraction.py"
        spec = importlib.util.spec_from_file_location("excel_extraction", mod_path)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            program, vehicle, serial = mod.derive_file_identity(p)  # type: ignore[attr-defined]
        else:
            program = vehicle = serial = ""
    except Exception:
        program = vehicle = serial = ""

    try:
        prog_val = str(meta.get("program_title") or "").strip()
    except Exception:
        prog_val = ""
    if not prog_val or prog_val.lower() == "unknown":
        title = ""
        if program and vehicle:
            title = f"{program} {vehicle}".strip()
        else:
            title = (program or vehicle or "").strip()
        if title:
            meta["program_title"] = title

    try:
        sn_val = str(meta.get("serial_number") or "").strip()
    except Exception:
        sn_val = ""
    if (not sn_val or sn_val.lower() == "unknown") and str(serial or "").strip():
        meta["serial_number"] = str(serial).strip()

    return meta


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
