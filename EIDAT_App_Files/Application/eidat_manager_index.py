from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from eidat_manager_db import SupportPaths
from eidat_manager_metadata import normalize_title, sanitize_metadata


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
  serial_number TEXT,
  part_number TEXT,
  revision TEXT,
  test_date TEXT,
  report_date TEXT,
  document_type TEXT,
  document_type_acronym TEXT,
  vendor TEXT,
  acceptance_test_plan_number TEXT,
  excel_sqlite_rel TEXT,
  file_extension TEXT,
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
                titles = data.get("program_titles") or []
                return [str(t).strip() for t in titles if str(t).strip()]
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


def _load_metadata_files(support_dir: Path) -> list[Path]:
    root = support_dir / "debug" / "ocr"
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.rglob("*"):
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
        name = p.name.lower()
        if name.endswith("_metadata.json") or name.endswith(".metadata.json"):
            files.append(p)
    return files


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

    metadata_files = _load_metadata_files(support_dir)
    docs: list[dict] = []
    for meta_path in metadata_files:
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_excel_artifacts = str(meta_path.parent.name).endswith("__excel")
        default_doc_type = "Data file" if is_excel_artifacts else "EIDP"
        meta = sanitize_metadata(raw, default_document_type=default_doc_type)
        title = str(meta.get("program_title") or "")

        title_norm = normalize_title(title)
        artifacts_rel = None
        try:
            artifacts_rel = str(meta_path.parent.resolve().relative_to(support_dir.resolve()))
        except Exception:
            artifacts_rel = str(meta_path.parent)
        try:
            metadata_rel = str(meta_path.resolve().relative_to(support_dir.resolve()))
        except Exception:
            metadata_rel = str(meta_path)
        # Get certification status from extracted_terms.db
        cert_info = _get_certification_from_db(meta_path.parent)
        docs.append(
            {
                "metadata_rel": metadata_rel,
                "artifacts_rel": artifacts_rel,
                "program_title": meta.get("program_title"),
                "asset_type": meta.get("asset_type"),
                "serial_number": meta.get("serial_number"),
                "part_number": meta.get("part_number"),
                "revision": meta.get("revision"),
                "test_date": meta.get("test_date"),
                "report_date": meta.get("report_date"),
                "document_type": meta.get("document_type"),
                "document_type_acronym": meta.get("document_type_acronym"),
                "vendor": meta.get("vendor"),
                "acceptance_test_plan_number": meta.get("acceptance_test_plan_number"),
                "excel_sqlite_rel": meta.get("excel_sqlite_rel"),
                "file_extension": meta.get("file_extension"),
                "title_norm": title_norm,
                "similarity_group": "",
                "certification_status": cert_info.get("certification_status"),
                "certification_pass_rate": cert_info.get("certification_pass_rate"),
            }
        )

    _group_docs(docs, similarity)

    now_ns = time.time_ns()
    with _connect(index_db) as conn:
        conn.executescript(SCHEMA_SQL)
        # Migration: add columns if they don't exist (for existing databases)
        migration_columns = [
            ("part_number", "TEXT"),
            ("certification_status", "TEXT"),
            ("certification_pass_rate", "TEXT"),
            ("document_type_acronym", "TEXT"),
            ("vendor", "TEXT"),
            ("acceptance_test_plan_number", "TEXT"),
            ("excel_sqlite_rel", "TEXT"),
            ("file_extension", "TEXT"),
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
                  metadata_rel, artifacts_rel, program_title, asset_type,
                  serial_number, part_number, revision, test_date, report_date, document_type,
                  document_type_acronym, vendor, acceptance_test_plan_number,
                  excel_sqlite_rel, file_extension, title_norm, similarity_group, indexed_epoch_ns,
                  certification_status, certification_pass_rate
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["metadata_rel"],
                    d["artifacts_rel"],
                    d["program_title"],
                    d["asset_type"],
                    d["serial_number"],
                    d["part_number"],
                    d["revision"],
                    d["test_date"],
                    d["report_date"],
                    d["document_type"],
                    d.get("document_type_acronym"),
                    d.get("vendor"),
                    d.get("acceptance_test_plan_number"),
                    d.get("excel_sqlite_rel"),
                    d.get("file_extension"),
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

    return IndexSummary(
        index_db=index_db,
        indexed_count=len(docs),
        groups_count=len({d.get("similarity_group") for d in docs if d.get("similarity_group")}),
        metadata_count=len(metadata_files),
    )
