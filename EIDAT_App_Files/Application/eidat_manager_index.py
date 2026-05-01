from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping

from eidat_manager_db import SupportPaths
from eidat_manager_metadata import normalize_title, sanitize_metadata

TD_FILE_EXTRACTIONS_DIRNAME = "Test Data File Extractions"
UI_VISIBLE_FILES_DIRNAME = "UI Visible Files"
UI_VISIBLE_TD_DIRNAME = "Test Data File Extractions"
UI_VISIBLE_EIDP_DIRNAME = "EIDP File Extractions"
UI_VISIBLE_METADATA_FILENAME = "metadata.json"
TD_LEGACY_SERIAL_SOURCES_DIRNAME = "td_serial_sources"
TD_SERIAL_SOURCE_ARTIFACTS_DIRNAME = "sources"
TD_OFFICIAL_METADATA_SOURCES = {"td_serial_aggregate", "td_serial_official_source"}
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".mat"}


@dataclass(frozen=True)
class IndexSummary:
    index_db: Path
    indexed_count: int
    groups_count: int
    metadata_count: int


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  metadata_rel TEXT NOT NULL UNIQUE,
  artifacts_rel TEXT,
  program_title TEXT,
  asset_type TEXT,
  asset_specific_type TEXT,
  serial_number TEXT,
  part_number TEXT,
  revision TEXT,
  test_date TEXT,
  report_date TEXT,
  document_type TEXT,
  document_type_acronym TEXT,
  document_type_status TEXT,
  document_type_source TEXT,
  document_type_reason TEXT,
  document_type_evidence_json TEXT,
  document_type_review_required INTEGER,
  vendor TEXT,
  acceptance_test_plan_number TEXT,
  excel_sqlite_rel TEXT,
  tables_sqlite_rel TEXT,
  source_rel_path TEXT,
  source_rel_paths_json TEXT,
  internal_metadata_rel TEXT,
  file_extension TEXT,
  metadata_source TEXT,
  manual_override_fields_json TEXT,
  manual_override_updated_at TEXT,
  applied_asset_specific_type_rule TEXT,
  title_norm TEXT,
  similarity_group TEXT,
  indexed_epoch_ns INTEGER NOT NULL,
  certification_status TEXT,
  certification_pass_rate TEXT
);

CREATE TABLE IF NOT EXISTS groups (
  group_id TEXT PRIMARY KEY,
  title_norm TEXT,
  member_count INTEGER NOT NULL
);
"""


def _load_program_title_candidates() -> list[str]:
    """Load program title candidates from user_inputs/metadata_candidates.json."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        cand_paths: list[Path] = []
        raw_data_root = (os.environ.get("EIDAT_DATA_ROOT") or "").strip()
        if raw_data_root:
            try:
                data_root = Path(raw_data_root).expanduser()
                if not data_root.is_absolute():
                    data_root = (repo_root / data_root).resolve()
                cand_paths.append(data_root / "user_inputs" / "metadata_candidates.json")
            except Exception:
                pass
        cand_paths.append(repo_root / "user_inputs" / "metadata_candidates.json")

        for cand_path in cand_paths:
            if not cand_path.exists():
                continue
            data = json.loads(cand_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                raw = data.get("program_titles") or []
                if not isinstance(raw, list):
                    return []
                out: list[str] = []
                for item in raw:
                    if isinstance(item, str) and item.strip():
                        out.append(item.strip())
                        continue
                    if isinstance(item, dict):
                        name = str(item.get("name") or "").strip()
                        if name:
                            out.append(name)
                return [t for t in out if t]
    except Exception:
        pass
    return []


def _infer_program_title_from_filename(meta_path: Path, candidates: list[str]) -> str | None:
    """Infer program title from metadata filename (e.g., SN2222_starlink_metadata.json)."""
    if not candidates:
        return None
    try:
        stem = meta_path.stem
        stem = re.sub(r"(_metadata|\\.metadata)$", "", stem, flags=re.IGNORECASE)
        low = stem.lower()
        best = ""
        for cand in candidates:
            c = str(cand or "").strip()
            if not c:
                continue
            cl = c.lower()
            if re.search(rf"\\b{re.escape(cl)}\\b", low) or cl in low:
                if len(c) > len(best):
                    best = c
        return best or None
    except Exception:
        return None


def _infer_serial_from_filename(meta_path: Path) -> str | None:
    """Infer serial number like SN2222 from metadata filename."""
    try:
        stem = meta_path.stem
        # Filenames often use separators (e.g., SN2222_starlink_metadata),
        # so avoid strict word-boundary matching.
        m = re.search(r"SN\d+", stem, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    except Exception:
        pass
    return None


def _infer_is_test_data_from_path(meta_path: Path) -> bool:
    """
    Heuristic fallback: classify as Test Data when the *filename/folder* contains
    'test data' (including hyphen/underscore variants) or a standalone 'TD' token.
    """
    return _infer_is_test_data_from_path_with_aliases(meta_path, _load_test_data_aliases())


def _infer_is_test_data_from_path_with_aliases(meta_path: Path, td_aliases: list[str]) -> bool:
    """
    Classify as Test Data when the *filename/folder* contains any user-configured alias for TD.
    """
    try:
        parts = [
            meta_path.name,
            meta_path.stem,
            meta_path.parent.name,
        ]
        try:
            parts.append(meta_path.parent.parent.name)
        except Exception:
            pass
        blob = "\n".join([p for p in parts if str(p).strip()])
    except Exception:
        blob = str(meta_path)

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    blob_norm = _norm(blob)
    blob_tokens = set(blob_norm.split())

    # If candidates aren't available, retain a conservative fallback.
    aliases = [str(a).strip() for a in (td_aliases or []) if str(a).strip()] or ["Test Data", "Test-Data", "TestData", "TD"]
    for alias in aliases:
        a_norm = _norm(alias)
        if not a_norm:
            continue
        if len(a_norm) <= 3 and len(a_norm.split()) == 1:
            if a_norm in blob_tokens:
                return True
        else:
            if a_norm in blob_norm:
                return True
    return False


def _load_test_data_aliases() -> list[str]:
    """Load TD aliases from user_inputs/metadata_candidates.json (document_types entry with acronym TD)."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        cand_paths: list[Path] = []
        raw_data_root = (os.environ.get("EIDAT_DATA_ROOT") or "").strip()
        if raw_data_root:
            try:
                data_root = Path(raw_data_root).expanduser()
                if not data_root.is_absolute():
                    data_root = (repo_root / data_root).resolve()
                cand_paths.append(data_root / "user_inputs" / "metadata_candidates.json")
            except Exception:
                pass
        cand_paths.append(repo_root / "user_inputs" / "metadata_candidates.json")

        data: dict | None = None
        for cand_path in cand_paths:
            if not cand_path.exists():
                continue
            loaded = json.loads(cand_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
                break
        if not isinstance(data, dict):
            return []
        raw = data.get("document_types") or []
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            acr = str(entry.get("acronym") or "").strip()
            if acr.upper() != "TD":
                continue
            aliases = entry.get("aliases")
            if isinstance(aliases, list):
                out.extend([str(a).strip() for a in aliases if str(a).strip()])
            name = str(entry.get("name") or "").strip()
            if name:
                out.append(name)
            out.append("TD")
        # Dedup, preserve order-ish
        seen = set()
        result = []
        for a in out:
            k = a.casefold()
            if k in seen:
                continue
            seen.add(k)
            result.append(a)
        return result
    except Exception:
        return []


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _rel_parts(value: object) -> list[str]:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return []
    return [part for part in re.split(r"[\\/]+", raw) if part and part != "."]


def _is_confirmed_td_meta(meta: dict[str, Any]) -> bool:
    doc_type = str(meta.get("document_type_acronym") or meta.get("document_type") or "").strip().upper()
    status = str(meta.get("document_type_status") or "").strip().lower()
    review_required = bool(meta.get("document_type_review_required"))
    return doc_type == "TD" and status == "confirmed" and not review_required


def _should_index_td_metadata(meta_path: Path, support_dir: Path, meta: dict[str, Any]) -> bool:
    if not _is_confirmed_td_meta(meta):
        return True
    try:
        rel = str(meta_path.resolve().relative_to(Path(support_dir).resolve())).replace("\\", "/")
    except Exception:
        rel = str(meta_path).replace("\\", "/")
    parts = [part.casefold() for part in _rel_parts(rel)]
    td_root = TD_FILE_EXTRACTIONS_DIRNAME.casefold()
    debug_root = ["debug", "ocr"]
    if td_root in parts:
        root_idx = parts.index(td_root)
        if TD_SERIAL_SOURCE_ARTIFACTS_DIRNAME.casefold() in parts[root_idx + 1 :]:
            return False
        meta_source = str(meta.get("metadata_source") or "").strip().casefold()
        if meta_source in TD_OFFICIAL_METADATA_SOURCES:
            return True
        parent_name = str(meta_path.parent.name or "").strip()
        return str(meta_path.stem or "").strip().casefold() == f"{parent_name}_metadata".casefold()
    return parts[:2] != debug_root


def _load_metadata_files(support_dir: Path) -> list[Path]:
    files: list[Path] = []
    roots = [
        support_dir / TD_FILE_EXTRACTIONS_DIRNAME,
        support_dir / "debug" / "ocr",
    ]
    seen: set[str] = set()
    debug_ocr_root = support_dir / "debug" / "ocr"
    legacy_td_root = debug_ocr_root / TD_LEGACY_SERIAL_SOURCES_DIRNAME
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if not p.is_file():
                    continue
            except Exception:
                continue
            name = p.name.lower()
            if not (name.endswith("_metadata.json") or name.endswith(".metadata.json")):
                continue
            if root == debug_ocr_root:
                try:
                    p.resolve().relative_to(legacy_td_root.resolve())
                except Exception:
                    pass
                else:
                    # The old composite TD repository has been superseded by the
                    # top-level Test Data File Extractions root. Avoid indexing both.
                    continue
            key = str(p).casefold()
            if key in seen:
                continue
            seen.add(key)
            files.append(p)
    return files


def _ui_visible_root(support_dir: Path) -> Path:
    return Path(support_dir) / UI_VISIBLE_FILES_DIRNAME


def _ui_visible_td_root(support_dir: Path) -> Path:
    return _ui_visible_root(support_dir) / UI_VISIBLE_TD_DIRNAME


def _ui_visible_eidp_root(support_dir: Path) -> Path:
    return _ui_visible_root(support_dir) / UI_VISIBLE_EIDP_DIRNAME


def _support_rel(support_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(support_dir).resolve())).replace("\\", "/")
    except Exception:
        try:
            return str(path.relative_to(Path(support_dir))).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")


def _safe_visible_component(value: object, *, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw or raw.casefold() in {"unknown", "none", "null", "n/a", "na", "-"}:
        raw = fallback
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] or fallback


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _source_rel_from_value(global_repo: Path, value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    try:
        return str(path.resolve().relative_to(Path(global_repo).resolve())).replace("\\", "/")
    except Exception:
        try:
            return str(path.relative_to(Path(global_repo))).replace("\\", "/")
        except Exception:
            return raw.replace("\\", "/")


def _visible_override_key(payload: Mapping[str, Any]) -> str:
    for key in ("internal_metadata_rel", "excel_sqlite_rel", "artifacts_rel", "source_rel_path"):
        value = str(payload.get(key) or "").strip()
        if value:
            return f"{key}:{value.casefold()}"
    serial = str(payload.get("serial_number") or "").strip()
    doc_type = str(payload.get("document_type_acronym") or payload.get("document_type") or "").strip()
    return f"serial:{serial.casefold()}|doc:{doc_type.casefold()}"


def _load_existing_visible_overrides(support_dir: Path) -> dict[str, dict[str, Any]]:
    root = _ui_visible_root(support_dir)
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob(UI_VISIBLE_METADATA_FILENAME)):
        raw = _read_json_file(path)
        if not raw:
            continue
        out[_visible_override_key(raw)] = raw
    return out


def _preserve_visible_manual_overrides(
    generated: dict[str, Any],
    existing_visible: dict[str, Any] | None,
) -> dict[str, Any]:
    current = dict(generated)
    existing = dict(existing_visible or {})
    manual_fields_raw = existing.get("manual_override_fields") or []
    if not isinstance(manual_fields_raw, list):
        manual_fields_raw = []
    manual_fields = [str(v).strip() for v in manual_fields_raw if str(v).strip()]
    if manual_fields:
        for field in manual_fields:
            if field in existing:
                current[field] = existing.get(field)
        if {"document_type", "document_type_acronym"}.intersection(manual_fields):
            for field in ("document_type", "document_type_acronym"):
                if field in existing:
                    current[field] = existing.get(field)
        current["manual_override_fields"] = manual_fields
        current["manual_override_updated_at"] = str(existing.get("manual_override_updated_at") or "").strip()
    sanitized = sanitize_metadata(current, default_document_type=str(current.get("document_type") or "Unknown"))
    out = dict(sanitized)
    for key in (
        "artifacts_rel",
        "excel_sqlite_rel",
        "tables_sqlite_rel",
        "source_rel_path",
        "source_rel_paths",
        "internal_metadata_rel",
        "similarity_group",
        "file_extension",
    ):
        if key in current:
            out[key] = current.get(key)
    return out


def _build_source_rel_by_artifacts_rel(paths: SupportPaths) -> dict[str, str]:
    db_path = paths.support_dir / "eidat_support.sqlite3"
    out: dict[str, str] = {}
    collisions: set[str] = set()
    if not db_path.exists():
        return out
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT rel_path FROM files").fetchall()
    for row in rows:
        rel_path = str(row["rel_path"] or "").strip().replace("\\", "/")
        if not rel_path:
            continue
        path_obj = Path(rel_path)
        stem = str(path_obj.stem or "").strip()
        ext = str(path_obj.suffix or "").lower()
        if not stem:
            continue
        artifacts_rel = f"debug/ocr/{stem}{'__excel' if ext in EXCEL_EXTENSIONS else ''}"
        key = artifacts_rel.casefold()
        if key in out and out[key] != rel_path:
            collisions.add(key)
            continue
        out[key] = rel_path
    for key in collisions:
        out.pop(key, None)
    return out


def _td_internal_metadata_files(support_dir: Path) -> list[Path]:
    root = Path(support_dir) / TD_FILE_EXTRACTIONS_DIRNAME
    files: list[Path] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*_metadata.json")) + sorted(root.rglob("*.metadata.json")):
        try:
            if not path.is_file():
                continue
        except Exception:
            continue
        parts = [part.casefold() for part in path.relative_to(root).parts]
        if TD_SERIAL_SOURCE_ARTIFACTS_DIRNAME.casefold() in parts:
            continue
        files.append(path)
    seen: set[str] = set()
    out: list[Path] = []
    for path in files:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _eidp_internal_metadata_files(support_dir: Path) -> list[Path]:
    root = Path(support_dir) / "debug" / "ocr"
    files: list[Path] = []
    if not root.exists():
        return files
    legacy_td_root = root / TD_LEGACY_SERIAL_SOURCES_DIRNAME
    for path in sorted(root.rglob("*_metadata.json")) + sorted(root.rglob("*.metadata.json")):
        try:
            if not path.is_file():
                continue
        except Exception:
            continue
        try:
            path.resolve().relative_to(legacy_td_root.resolve())
        except Exception:
            pass
        else:
            continue
        files.append(path)
    seen: set[str] = set()
    out: list[Path] = []
    for path in files:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _td_source_rel_paths(paths: SupportPaths, meta_path: Path, raw: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    direct = raw.get("source_rel_paths")
    if isinstance(direct, list):
        for value in direct:
            rel = str(value or "").strip().replace("\\", "/")
            if rel:
                found.append(rel)
    direct_single = _source_rel_from_value(paths.global_repo, raw.get("source_rel_path") or raw.get("source_file"))
    if direct_single:
        found.append(direct_single)
    support_dir = paths.support_dir
    rels: list[str] = []
    manifest_path = meta_path.parent / "td_serial_aggregate.json"
    manifest = _read_json_file(manifest_path) if manifest_path.exists() else {}
    raw_meta_rels = raw.get("source_metadata_rels") or []
    if not isinstance(raw_meta_rels, list):
        raw_meta_rels = []
    rels.extend([str(v or "").strip() for v in raw_meta_rels if str(v or "").strip()])
    if isinstance(manifest, dict):
        manifest_meta_rels = manifest.get("source_metadata_rels") or []
        if isinstance(manifest_meta_rels, list):
            rels.extend([str(v or "").strip() for v in manifest_meta_rels if str(v or "").strip()])
        members = manifest.get("members") or []
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                rel = str(member.get("source_metadata_rel") or "").strip()
                if rel:
                    rels.append(rel)
    for rel in rels:
        candidate = Path(support_dir) / rel
        source_meta = _read_json_file(candidate)
        if not source_meta:
            continue
        source_rel = _source_rel_from_value(paths.global_repo, source_meta.get("source_rel_path") or source_meta.get("source_file"))
        if source_rel:
            found.append(source_rel)
    deduped: list[str] = []
    seen: set[str] = set()
    for rel in found:
        key = rel.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped


def _visible_td_payload(paths: SupportPaths, meta_path: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    meta = sanitize_metadata(dict(raw), default_document_type="TD")
    payload: dict[str, Any] = {
        "program_title": meta.get("program_title"),
        "asset_type": meta.get("asset_type"),
        "asset_specific_type": meta.get("asset_specific_type"),
        "serial_number": meta.get("serial_number"),
        "part_number": meta.get("part_number"),
        "revision": meta.get("revision"),
        "test_date": meta.get("test_date"),
        "report_date": meta.get("report_date"),
        "document_type": meta.get("document_type"),
        "document_type_acronym": meta.get("document_type_acronym"),
        "document_type_status": meta.get("document_type_status"),
        "document_type_source": meta.get("document_type_source"),
        "document_type_reason": meta.get("document_type_reason"),
        "document_type_evidence": meta.get("document_type_evidence") or [],
        "document_type_review_required": bool(meta.get("document_type_review_required")),
        "vendor": meta.get("vendor"),
        "acceptance_test_plan_number": meta.get("acceptance_test_plan_number"),
        "file_extension": meta.get("file_extension") or ".sqlite3",
        "metadata_source": meta.get("metadata_source"),
        "manual_override_fields": [],
        "manual_override_updated_at": "",
        "applied_asset_specific_type_rule": meta.get("applied_asset_specific_type_rule") or "",
        "similarity_group": str(raw.get("similarity_group") or "").strip(),
        "artifacts_rel": _support_rel(paths.support_dir, meta_path.parent),
        "excel_sqlite_rel": str(raw.get("excel_sqlite_rel") or meta.get("excel_sqlite_rel") or "").strip(),
        "tables_sqlite_rel": str(raw.get("tables_sqlite_rel") or "").strip(),
        "source_rel_path": "",
        "source_rel_paths": [],
        "internal_metadata_rel": _support_rel(paths.support_dir, meta_path),
    }
    source_paths = _td_source_rel_paths(paths, meta_path, raw)
    if source_paths:
        payload["source_rel_paths"] = list(source_paths)
        if len(source_paths) == 1:
            payload["source_rel_path"] = source_paths[0]
    return payload


def _visible_eidp_payload(
    paths: SupportPaths,
    meta_path: Path,
    raw: Mapping[str, Any],
    source_rel_by_artifacts_rel: Mapping[str, str],
) -> dict[str, Any] | None:
    meta = sanitize_metadata(dict(raw), default_document_type="Unknown")
    if _is_confirmed_td_meta(meta):
        return None
    artifacts_rel = _support_rel(paths.support_dir, meta_path.parent)
    source_rel = _source_rel_from_value(paths.global_repo, raw.get("source_rel_path") or raw.get("source_file"))
    if not source_rel:
        source_rel = str(source_rel_by_artifacts_rel.get(artifacts_rel.casefold()) or "").strip()
    tables_sqlite_rel = str(raw.get("tables_sqlite_rel") or "").strip()
    if not tables_sqlite_rel:
        tables_db = meta_path.parent / "labeled_tables.db"
        if tables_db.exists():
            tables_sqlite_rel = _support_rel(paths.support_dir, tables_db)
    return {
        "program_title": meta.get("program_title"),
        "asset_type": meta.get("asset_type"),
        "asset_specific_type": meta.get("asset_specific_type"),
        "serial_number": meta.get("serial_number"),
        "part_number": meta.get("part_number"),
        "revision": meta.get("revision"),
        "test_date": meta.get("test_date"),
        "report_date": meta.get("report_date"),
        "document_type": meta.get("document_type"),
        "document_type_acronym": meta.get("document_type_acronym"),
        "document_type_status": meta.get("document_type_status"),
        "document_type_source": meta.get("document_type_source"),
        "document_type_reason": meta.get("document_type_reason"),
        "document_type_evidence": meta.get("document_type_evidence") or [],
        "document_type_review_required": bool(meta.get("document_type_review_required")),
        "vendor": meta.get("vendor"),
        "acceptance_test_plan_number": meta.get("acceptance_test_plan_number"),
        "file_extension": meta.get("file_extension"),
        "metadata_source": meta.get("metadata_source"),
        "manual_override_fields": [],
        "manual_override_updated_at": "",
        "applied_asset_specific_type_rule": meta.get("applied_asset_specific_type_rule") or "",
        "similarity_group": str(raw.get("similarity_group") or "").strip(),
        "artifacts_rel": artifacts_rel,
        "excel_sqlite_rel": str(raw.get("excel_sqlite_rel") or meta.get("excel_sqlite_rel") or "").strip(),
        "tables_sqlite_rel": tables_sqlite_rel,
        "source_rel_path": source_rel,
        "source_rel_paths": [source_rel] if source_rel else [],
        "internal_metadata_rel": _support_rel(paths.support_dir, meta_path),
    }


def _visible_td_target_dir(root: Path, payload: Mapping[str, Any]) -> Path:
    return (
        Path(root)
        / _safe_visible_component(payload.get("program_title"), fallback="Unknown_Program")
        / _safe_visible_component(payload.get("asset_type"), fallback="Unknown_Asset")
        / _safe_visible_component(payload.get("asset_specific_type"), fallback="Unknown_Type")
        / _safe_visible_component(payload.get("serial_number"), fallback="Unknown_Serial")
    )


def _visible_eidp_target_dir(root: Path, payload: Mapping[str, Any], meta_path: Path) -> Path:
    leaf = str(payload.get("serial_number") or "").strip()
    if not leaf or leaf.casefold() == "unknown":
        source_rel = str(payload.get("source_rel_path") or "").strip()
        if source_rel:
            leaf = Path(source_rel).stem
        else:
            leaf = re.sub(r"__excel$", "", str(meta_path.parent.name or "").strip(), flags=re.IGNORECASE) or meta_path.parent.name
    return (
        Path(root)
        / _safe_visible_component(payload.get("program_title"), fallback="Unknown_Program")
        / _safe_visible_component(payload.get("asset_type"), fallback="Unknown_Asset")
        / _safe_visible_component(payload.get("asset_specific_type"), fallback="Unknown_Type")
        / _safe_visible_component(leaf, fallback="Unknown_Document")
    )


def _reserve_unique_dir(target_dir: Path, identity_key: str, claimed: dict[str, str]) -> Path:
    candidate = Path(target_dir)
    suffix = 1
    while True:
        key = str(candidate).casefold()
        owner = claimed.get(key)
        if owner in {None, identity_key}:
            claimed[key] = identity_key
            return candidate
        suffix += 1
        candidate = target_dir.with_name(f"{target_dir.name}_{suffix}")


def publish_visible_bundles(paths: SupportPaths) -> dict[str, int]:
    support_dir = paths.support_dir
    visible_root = _ui_visible_root(support_dir)
    existing_overrides = _load_existing_visible_overrides(support_dir)
    shutil.rmtree(visible_root, ignore_errors=True)
    visible_root.mkdir(parents=True, exist_ok=True)
    source_rel_by_artifacts_rel = _build_source_rel_by_artifacts_rel(paths)
    claimed: dict[str, str] = {}
    published = 0

    for meta_path in _td_internal_metadata_files(support_dir):
        raw = _read_json_file(meta_path)
        if not raw:
            continue
        payload = _visible_td_payload(paths, meta_path, raw)
        payload = _preserve_visible_manual_overrides(payload, existing_overrides.get(_visible_override_key(payload)))
        target_dir = _reserve_unique_dir(_visible_td_target_dir(_ui_visible_td_root(support_dir), payload), _visible_override_key(payload), claimed)
        _write_json_file(target_dir / UI_VISIBLE_METADATA_FILENAME, payload)
        published += 1

    for meta_path in _eidp_internal_metadata_files(support_dir):
        raw = _read_json_file(meta_path)
        if not raw:
            continue
        payload = _visible_eidp_payload(paths, meta_path, raw, source_rel_by_artifacts_rel)
        if payload is None:
            continue
        payload = _preserve_visible_manual_overrides(payload, existing_overrides.get(_visible_override_key(payload)))
        target_dir = _reserve_unique_dir(_visible_eidp_target_dir(_ui_visible_eidp_root(support_dir), payload, meta_path), _visible_override_key(payload), claimed)
        _write_json_file(target_dir / UI_VISIBLE_METADATA_FILENAME, payload)
        published += 1

    return {"published": int(published)}


def _load_visible_metadata_files(support_dir: Path) -> list[Path]:
    root = _ui_visible_root(support_dir)
    if not root.exists():
        return []
    return sorted([path for path in root.rglob(UI_VISIBLE_METADATA_FILENAME) if path.is_file()])


def _sync_similarity_group_to_visible_metadata(docs: list[dict[str, Any]], support_dir: Path) -> None:
    for doc in docs:
        metadata_rel = str(doc.get("metadata_rel") or "").strip()
        if not metadata_rel:
            continue
        meta_path = Path(support_dir) / metadata_rel
        raw = _read_json_file(meta_path)
        if not raw:
            continue
        similarity_group = str(doc.get("similarity_group") or "").strip()
        if str(raw.get("similarity_group") or "").strip() == similarity_group:
            continue
        raw["similarity_group"] = similarity_group
        try:
            _write_json_file(meta_path, raw)
        except Exception:
            continue


def _get_certification_from_db(artifacts_dir: Path) -> dict[str, Any]:
    """Read certification status from extracted_terms.db in an artifacts folder."""
    db_path = artifacts_dir / "extracted_terms.db"
    result = {"certification_status": None, "certification_pass_rate": None}
    if not db_path.exists():
        return result
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT certification_status, certification_passed_count,
                   certification_failed_count, certification_pending_count
            FROM documents
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        if row:
            status = row["certification_status"]
            passed = row["certification_passed_count"] or 0
            failed = row["certification_failed_count"] or 0
            pending = row["certification_pending_count"] or 0
            total = passed + failed + pending
            result["certification_status"] = status
            result["certification_pass_rate"] = f"{passed}/{total}" if total > 0 else None
    except Exception:
        pass
    return result


def _group_docs(docs: list[dict], similarity: float) -> None:
    groups: list[dict[str, str]] = []
    for doc in docs:
        title_norm = doc.get("title_norm") or ""
        if not title_norm:
            doc["similarity_group"] = "unknown"
            continue
        assigned = False
        for g in groups:
            ratio = SequenceMatcher(None, title_norm, g["title_norm"]).ratio()
            if ratio >= similarity:
                doc["similarity_group"] = g["group_id"]
                assigned = True
                break
        if not assigned:
            group_id = f"group_{len(groups)+1}"
            groups.append({"group_id": group_id, "title_norm": title_norm})
            doc["similarity_group"] = group_id


def build_index(paths: SupportPaths, *, similarity: float = 0.86) -> IndexSummary:
    support_dir = paths.support_dir
    index_db = support_dir / "eidat_index.sqlite3"

    publish_visible_bundles(paths)
    metadata_files = _load_visible_metadata_files(support_dir)
    docs: list[dict] = []
    for meta_path in metadata_files:
        raw = _read_json_file(meta_path)
        if not raw:
            continue
        meta = sanitize_metadata(raw, default_document_type="Unknown")
        raw_metadata_source = str(raw.get("metadata_source") or "").strip()
        if raw_metadata_source and str(meta.get("metadata_source") or "").strip() in {"", "scanned"}:
            meta["metadata_source"] = raw_metadata_source
        title = str(meta.get("program_title") or "")
        title_norm = normalize_title(title)
        artifacts_rel = str(raw.get("artifacts_rel") or "").strip()
        metadata_rel = _support_rel(support_dir, meta_path)
        artifacts_dir = Path(support_dir) / artifacts_rel if artifacts_rel else meta_path.parent
        cert_info = _get_certification_from_db(artifacts_dir)
        tables_sqlite_rel = str(raw.get("tables_sqlite_rel") or "").strip()
        docs.append(
            {
                "metadata_rel": metadata_rel,
                "artifacts_rel": artifacts_rel,
                "program_title": meta.get("program_title"),
                "asset_type": meta.get("asset_type"),
                "asset_specific_type": meta.get("asset_specific_type"),
                "serial_number": meta.get("serial_number"),
                "part_number": meta.get("part_number"),
                "revision": meta.get("revision"),
                "test_date": meta.get("test_date"),
                "report_date": meta.get("report_date"),
                "document_type": meta.get("document_type"),
                "document_type_acronym": meta.get("document_type_acronym"),
                "document_type_status": meta.get("document_type_status"),
                "document_type_source": meta.get("document_type_source"),
                "document_type_reason": meta.get("document_type_reason"),
                "document_type_evidence_json": json.dumps(meta.get("document_type_evidence") or [], ensure_ascii=True),
                "document_type_review_required": 1 if meta.get("document_type_review_required") else 0,
                "vendor": meta.get("vendor"),
                "acceptance_test_plan_number": meta.get("acceptance_test_plan_number"),
                "excel_sqlite_rel": str(raw.get("excel_sqlite_rel") or meta.get("excel_sqlite_rel") or "").strip(),
                "tables_sqlite_rel": tables_sqlite_rel,
                "source_rel_path": str(raw.get("source_rel_path") or "").strip(),
                "source_rel_paths_json": json.dumps(raw.get("source_rel_paths") or [], ensure_ascii=True),
                "internal_metadata_rel": str(raw.get("internal_metadata_rel") or "").strip(),
                "file_extension": meta.get("file_extension"),
                "metadata_source": meta.get("metadata_source"),
                "manual_override_fields_json": json.dumps(meta.get("manual_override_fields") or [], ensure_ascii=True),
                "manual_override_updated_at": meta.get("manual_override_updated_at"),
                "applied_asset_specific_type_rule": meta.get("applied_asset_specific_type_rule"),
                "title_norm": title_norm,
                "similarity_group": str(raw.get("similarity_group") or "").strip(),
                "certification_status": cert_info.get("certification_status"),
                "certification_pass_rate": cert_info.get("certification_pass_rate"),
            }
        )

    _group_docs(docs, similarity)
    _sync_similarity_group_to_visible_metadata(docs, support_dir)

    now_ns = time.time_ns()
    conn = _connect(index_db)
    try:
        conn.executescript(SCHEMA_SQL)
        # Migration: add columns if they don't exist (for existing databases)
        migration_columns = [
            ("part_number", "TEXT"),
            ("certification_status", "TEXT"),
            ("certification_pass_rate", "TEXT"),
            ("document_type_acronym", "TEXT"),
            ("document_type_status", "TEXT"),
            ("document_type_source", "TEXT"),
            ("document_type_reason", "TEXT"),
            ("document_type_evidence_json", "TEXT"),
            ("document_type_review_required", "INTEGER"),
            ("vendor", "TEXT"),
            ("acceptance_test_plan_number", "TEXT"),
            ("asset_specific_type", "TEXT"),
            ("excel_sqlite_rel", "TEXT"),
            ("tables_sqlite_rel", "TEXT"),
            ("source_rel_path", "TEXT"),
            ("source_rel_paths_json", "TEXT"),
            ("internal_metadata_rel", "TEXT"),
            ("file_extension", "TEXT"),
            ("metadata_source", "TEXT"),
            ("manual_override_fields_json", "TEXT"),
            ("manual_override_updated_at", "TEXT"),
            ("applied_asset_specific_type_rule", "TEXT"),
        ]
        for col_name, col_type in migration_columns:
            try:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM groups")
        for d in docs:
            conn.execute(
                """
                INSERT INTO documents(
                  metadata_rel, artifacts_rel, program_title, asset_type, asset_specific_type,
                  serial_number, part_number, revision, test_date, report_date, document_type,
                  document_type_acronym, document_type_status, document_type_source, document_type_reason,
                  document_type_evidence_json, document_type_review_required, vendor, acceptance_test_plan_number,
                  excel_sqlite_rel, tables_sqlite_rel, source_rel_path, source_rel_paths_json, internal_metadata_rel,
                  file_extension, metadata_source, manual_override_fields_json, manual_override_updated_at,
                  applied_asset_specific_type_rule, title_norm, similarity_group, indexed_epoch_ns,
                  certification_status, certification_pass_rate
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["metadata_rel"],
                    d["artifacts_rel"],
                    d["program_title"],
                    d["asset_type"],
                    d.get("asset_specific_type"),
                    d["serial_number"],
                    d["part_number"],
                    d["revision"],
                    d["test_date"],
                    d["report_date"],
                    d["document_type"],
                    d.get("document_type_acronym"),
                    d.get("document_type_status"),
                    d.get("document_type_source"),
                    d.get("document_type_reason"),
                    d.get("document_type_evidence_json"),
                    d.get("document_type_review_required"),
                    d.get("vendor"),
                    d.get("acceptance_test_plan_number"),
                    d.get("excel_sqlite_rel"),
                    d.get("tables_sqlite_rel"),
                    d.get("source_rel_path"),
                    d.get("source_rel_paths_json"),
                    d.get("internal_metadata_rel"),
                    d.get("file_extension"),
                    d.get("metadata_source"),
                    d.get("manual_override_fields_json"),
                    d.get("manual_override_updated_at"),
                    d.get("applied_asset_specific_type_rule"),
                    d["title_norm"],
                    d["similarity_group"],
                    now_ns,
                    d.get("certification_status"),
                    d.get("certification_pass_rate"),
                ),
            )
        group_counts: dict[str, dict[str, Any]] = {}
        for d in docs:
            gid = d.get("similarity_group") or ""
            if not gid:
                continue
            entry = group_counts.setdefault(
                gid,
                {"group_id": gid, "title_norm": d.get("title_norm") or "", "member_count": 0},
            )
            entry["member_count"] += 1
        for g in group_counts.values():
            conn.execute(
                "INSERT INTO groups(group_id, title_norm, member_count) VALUES(?, ?, ?)",
                (g["group_id"], g["title_norm"], g["member_count"]),
            )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return IndexSummary(
        index_db=index_db,
        indexed_count=len(docs),
        groups_count=len({d.get("similarity_group") for d in docs if d.get("similarity_group")}),
        metadata_count=len(metadata_files),
    )
