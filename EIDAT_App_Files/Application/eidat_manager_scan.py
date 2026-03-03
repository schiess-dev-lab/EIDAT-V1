from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from eidat_manager_db import SupportPaths, connect_db, ensure_schema, get_meta_int, set_meta
except Exception:  # pragma: no cover
    from .eidat_manager_db import SupportPaths, connect_db, ensure_schema, get_meta_int, set_meta  # type: ignore

try:
    from eidat_manager_embed import extract_pointer_token
except Exception:  # pragma: no cover
    try:
        from .eidat_manager_embed import extract_pointer_token  # type: ignore
    except Exception:  # pragma: no cover
        extract_pointer_token = None  # type: ignore[assignment]

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


# Supported file extensions for scanning
SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".xlsm"}
EXCEL_ARTIFACT_SUFFIX = "__excel"

# Ignore generated artifacts and support folders anywhere in the repo tree.
# These are not "source" PDFs/Excels and should not be tracked as inputs.
_IGNORED_REPO_DIRNAMES_CASEFOLD = {
    "eidat",
    "eidat support",
    # Common typo/legacy naming without the "I"
    "edat",
    "edat support",
    # Common heavy directories to prune while scanning node roots.
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
}

def _parse_ignore_dirs_env() -> set[str]:
    raw = str(os.environ.get("EIDAT_SCAN_IGNORE_DIRS") or "").strip()
    if not raw:
        return set()
    parts: list[str] = []
    for chunk in raw.replace(",", ";").split(";"):
        s = str(chunk or "").strip()
        if s:
            parts.append(s.casefold())
    return set(parts)


def _progress_every_env(default: int = 2000) -> int:
    raw = str(os.environ.get("EIDAT_SCAN_PROGRESS_EVERY") or "").strip()
    if not raw:
        return int(default)
    try:
        n = int(float(raw))
    except Exception:
        return int(default)
    return int(n) if int(n) > 0 else 0


def _scan_log(line: str) -> None:
    try:
        sys.stderr.write(str(line or "").rstrip("\n") + "\n")
        sys.stderr.flush()
    except Exception:
        pass


@dataclass(frozen=True)
class ScanCandidate:
    rel_path: str
    abs_path: Path
    size_bytes: int
    mtime_ns: int
    reason: str  # "new" | "changed" | "new_since_last_scan"


@dataclass(frozen=True)
class ScanSummary:
    global_repo: Path
    support_dir: Path
    db_path: Path
    pdf_count: int
    candidates: list[ScanCandidate]
    last_scan_epoch_ns_before: int
    last_scan_epoch_ns_after: int


def _iter_data_files(global_repo: Path, *, exclude_dir: Path) -> list[Path]:
    """Iterate over all supported data files (PDFs and Excel) in the repository."""
    files: list[Path] = []
    extra_ignored = _parse_ignore_dirs_env()
    progress_every = _progress_every_env()
    visited_dirs = 0
    files_seen = 0
    matched_supported = 0
    try:
        exclude_dir_res = exclude_dir.resolve()
    except Exception:
        exclude_dir_res = exclude_dir.expanduser().absolute()

    _scan_log(f"[SCAN] Walking: {global_repo}")
    if extra_ignored:
        _scan_log(f"[SCAN] Extra ignored dirs from EIDAT_SCAN_IGNORE_DIRS: {sorted(extra_ignored)}")

    # Use os.walk so we can prune ignored/support directories efficiently.
    for dirpath, dirnames, filenames in os.walk(str(global_repo), topdown=True):
        visited_dirs += 1
        base = Path(dirpath)

        # Prune directories we never want to scan.
        kept: list[str] = []
        for d in dirnames:
            try:
                if str(d).casefold() in _IGNORED_REPO_DIRNAMES_CASEFOLD:
                    continue
                if extra_ignored and str(d).casefold() in extra_ignored:
                    continue
            except Exception:
                pass
            cand = base / d
            try:
                cr = cand.resolve()
            except Exception:
                cr = cand.expanduser().absolute()
            try:
                cr.relative_to(exclude_dir_res)
                continue
            except Exception:
                pass
            kept.append(d)
        dirnames[:] = kept

        for name in filenames:
            files_seen += 1
            p = base / name
            try:
                ext = p.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                # Skip Excel temp files (start with ~$)
                if ext in {".xlsx", ".xls", ".xlsm"} and p.name.startswith("~$"):
                    continue
            except Exception:
                continue

            try:
                if not p.is_file():
                    continue
            except Exception:
                continue

            files.append(p)
            matched_supported += 1
            if progress_every and (matched_supported % int(progress_every) == 0):
                _scan_log(
                    f"[SCAN] visited_dirs={visited_dirs} files_seen={files_seen} matched_supported={matched_supported}"
                )
    return files


# Backwards compatibility alias
def _iter_pdfs(global_repo: Path, *, exclude_dir: Path) -> list[Path]:
    """Deprecated: use _iter_data_files instead."""
    return _iter_data_files(global_repo, exclude_dir=exclude_dir)


def _has_eidat_metadata(file_path: Path) -> bool:
    # Only PDFs can have embedded EIDAT metadata
    if file_path.suffix.lower() not in {".pdf"}:
        return False
    if fitz is None:
        return False
    try:
        doc = fitz.open(str(file_path))
    except Exception:
        return False
    try:
        md = doc.metadata or {}
        for k, v in md.items():
            key = str(k or "").lower()
            val = str(v or "").lower()
            if "eidat" in key or "eidat" in val:
                return True
            if key in {"eidat_uuid", "eidat_pointer", "eidat_support"}:
                return True
        try:
            xmp = doc.get_xml_metadata() or ""
            if "eidat" in xmp.lower():
                return True
        except Exception:
            pass
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return False


def _pointer_artifacts_exist(global_repo: Path, file_path: Path) -> bool:
    """
    Best-effort: if a PDF has an EIDAT pointer token, only treat it as "already processed"
    when the referenced artifacts exist under *this* node's support folder.

    This prevents a common production-node scenario:
      - a previously-processed PDF (with embedded pointer token) is copied into a new node
      - the new node does NOT yet have its EIDAT Support artifacts
      - scan must still schedule processing to regenerate artifacts/index
    """
    if extract_pointer_token is None:
        return False
    try:
        payload = extract_pointer_token(file_path)
    except Exception:
        payload = None
    if not isinstance(payload, dict) or not payload:
        return False

    repo = Path(global_repo).expanduser()
    new_support_dir = repo / "EIDAT" / "EIDAT Support"
    legacy_support_dir = repo / "EIDAT Support"

    def _support_dir() -> Path:
        try:
            if new_support_dir.is_dir():
                return new_support_dir
        except Exception:
            pass
        try:
            if legacy_support_dir.is_dir():
                return legacy_support_dir
        except Exception:
            pass
        return new_support_dir

    def _resolve(rel_or_abs: object) -> Path | None:
        try:
            raw = str(rel_or_abs or "").strip()
        except Exception:
            return None
        if not raw:
            return None
        p = Path(raw.replace("/", "\\"))
        if p.is_absolute():
            return p

        # Back-compat: older pointer tokens stored paths rooted at "EIDAT Support\..."
        # even when support is now nested under "EIDAT\EIDAT Support".
        parts = list(p.parts)
        if parts and str(parts[0]).strip().casefold() == "eidat support":
            return _support_dir() / Path(*parts[1:])
        if len(parts) >= 2 and str(parts[0]).strip().casefold() == "eidat" and str(parts[1]).strip().casefold() == "eidat support":
            return repo / p

        return repo / p

    artifacts_path = _resolve(payload.get("artifacts_rel"))
    metadata_path = _resolve(payload.get("metadata_rel"))
    support_path = _resolve(payload.get("support_rel"))

    # If token indicates a support folder, require that it exists in this node.
    if support_path is not None and not support_path.exists():
        return False

    # If token indicates a metadata file, require that it exists.
    if metadata_path is not None and metadata_path.exists():
        return True

    # If token indicates an artifacts folder, require that it exists and has at least one expected artifact.
    if artifacts_path is None or not artifacts_path.exists():
        return False
    if artifacts_path.is_file():
        return True

    try:
        if (artifacts_path / "combined.txt").exists():
            return True
        # Common metadata filename pattern used by this repo.
        for p in artifacts_path.glob("*_metadata.json"):
            if p.is_file():
                return True
        for p in artifacts_path.glob("*.metadata.json"):
            if p.is_file():
                return True
    except Exception:
        pass

    # If folder exists but we can't confirm artifacts, be conservative and re-process.
    return False


def _expected_artifacts_dir(support_dir: Path, file_path: Path) -> Path:
    """
    Where this codebase writes artifacts for a given input file.

    PDFs:  EIDAT Support/debug/ocr/<stem>/
    Excel: EIDAT Support/debug/ocr/<stem>__excel/
    """
    stem = str(file_path.stem or "").strip() or "unknown"
    ext = str(file_path.suffix or "").lower()
    name = stem + (EXCEL_ARTIFACT_SUFFIX if ext in {".xlsx", ".xls", ".xlsm"} else "")
    return Path(support_dir) / "debug" / "ocr" / name


def _expected_artifacts_exist(support_dir: Path, file_path: Path) -> bool:
    artifacts_dir = _expected_artifacts_dir(support_dir, file_path)
    if not artifacts_dir.exists():
        return False
    if artifacts_dir.is_file():
        return True
    try:
        if (artifacts_dir / "combined.txt").exists():
            return True
        for p in artifacts_dir.glob("*_metadata.json"):
            if p.is_file():
                return True
        for p in artifacts_dir.glob("*.metadata.json"):
            if p.is_file():
                return True
    except Exception:
        pass
    return False


def scan_global_repo(paths: SupportPaths) -> ScanSummary:
    global_repo = paths.global_repo
    now_ns = time.time_ns()

    with connect_db(paths.db_path) as conn:
        ensure_schema(conn)
        last_scan_before = get_meta_int(conn, "last_scan_epoch_ns", 0)

        data_files = _iter_data_files(global_repo, exclude_dir=paths.support_dir)
        candidates: list[ScanCandidate] = []

        for pdf in data_files:
            try:
                st = pdf.stat()
                size_bytes = int(getattr(st, "st_size", 0) or 0)
                mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
            except Exception:
                continue
            file_fingerprint = f"{size_bytes}:{mtime_ns}"

            try:
                rel = pdf.resolve().relative_to(global_repo.resolve())
                rel_path = rel.as_posix()
            except Exception:
                try:
                    rel_path = pdf.relative_to(global_repo).as_posix()
                except Exception:
                    rel_path = str(pdf)

            row = conn.execute(
                """
                SELECT id, mtime_ns, last_processed_mtime_ns, last_processed_epoch_ns,
                       needs_processing, last_seen_epoch_ns
                FROM files
                WHERE rel_path = ?
                """,
                (rel_path,),
            ).fetchone()

            needs_processing = 0
            reason = ""
            metadata_present = False
            metadata_ok = False
            artifacts_ok = _expected_artifacts_exist(paths.support_dir, pdf)
            if row is None:
                moved = conn.execute(
                    """
                    SELECT id, rel_path, last_processed_epoch_ns
                    FROM files
                    WHERE file_fingerprint = ?
                    ORDER BY last_seen_epoch_ns DESC
                    LIMIT 1
                    """,
                    (file_fingerprint,),
                ).fetchone()
                # Treat as "moved" only when the prior path no longer exists on disk.
                # If the prior path still exists, this is a duplicate copy and should be tracked separately.
                if moved is not None:
                    try:
                        old_rel = str(moved["rel_path"] or "").strip()
                    except Exception:
                        old_rel = ""
                    if old_rel:
                        try:
                            old_abs = (paths.global_repo / Path(old_rel)).expanduser()
                            if old_abs.exists():
                                moved = None
                        except Exception:
                            pass
                if moved is not None:
                    needs_processing = 1 if not moved["last_processed_epoch_ns"] else 0
                    reason = "moved_unprocessed" if needs_processing else "moved"
                    if needs_processing:
                        metadata_present = _has_eidat_metadata(pdf)
                        if metadata_present:
                            metadata_ok = _pointer_artifacts_exist(global_repo, pdf)
                            if metadata_ok:
                                needs_processing = 0
                                reason = "metadata_present"
                            else:
                                needs_processing = 1
                                reason = "metadata_missing_artifacts"
                    if not artifacts_ok:
                        needs_processing = 1
                        reason = reason or "missing_artifacts"
                    conn.execute(
                        """
                        UPDATE files
                        SET rel_path = ?,
                            size_bytes = ?,
                            mtime_ns = ?,
                            file_fingerprint = ?,
                            last_seen_epoch_ns = ?,
                            last_processed_epoch_ns = CASE WHEN ? = 1 AND last_processed_epoch_ns IS NULL THEN ? ELSE last_processed_epoch_ns END,
                            needs_processing = CASE WHEN ? = 1 THEN 1 ELSE needs_processing END
                        WHERE id = ?
                        """,
                        (rel_path, size_bytes, mtime_ns, file_fingerprint, now_ns, 1 if metadata_ok else 0, now_ns, needs_processing, int(moved["id"])),
                    )
                else:
                    needs_processing = 1
                    reason = "new"
                    metadata_present = _has_eidat_metadata(pdf)
                    if metadata_present:
                        metadata_ok = _pointer_artifacts_exist(global_repo, pdf)
                        if metadata_ok:
                            needs_processing = 0
                            reason = "metadata_present"
                        else:
                            needs_processing = 1
                            reason = "metadata_missing_artifacts"
                    if not artifacts_ok:
                        needs_processing = 1
                        reason = reason or "missing_artifacts"
                    conn.execute(
                        """
                        INSERT INTO files(
                          rel_path, file_fingerprint, size_bytes, mtime_ns,
                          first_seen_epoch_ns, last_seen_epoch_ns,
                          last_processed_epoch_ns, needs_processing
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rel_path,
                            file_fingerprint,
                            size_bytes,
                            mtime_ns,
                            now_ns,
                            now_ns,
                            now_ns if metadata_ok else None,
                            1 if needs_processing else 0,
                        ),
                    )
            else:
                prev_mtime_ns = int(row["mtime_ns"] or 0)
                last_processed_mtime_ns = int(row["last_processed_mtime_ns"] or 0)
                last_processed_epoch_ns = int(row["last_processed_epoch_ns"] or 0)
                prev_last_seen = int(row["last_seen_epoch_ns"] or 0)
                if int(row["needs_processing"] or 0):
                    needs_processing = 1
                    reason = "pending"
                elif not last_processed_epoch_ns:
                    needs_processing = 1
                    reason = "never_processed"
                elif last_scan_before and prev_last_seen and prev_last_seen < last_scan_before and mtime_ns > last_scan_before:
                    needs_processing = 1
                    reason = "reappeared"
                elif mtime_ns != prev_mtime_ns:
                    needs_processing = 1
                    reason = "changed"
                elif last_processed_mtime_ns and mtime_ns != last_processed_mtime_ns:
                    needs_processing = 1
                    reason = "changed"
                elif last_scan_before and mtime_ns > last_scan_before:
                    needs_processing = 1
                    reason = "new_since_last_scan"

                if needs_processing:
                    metadata_present = _has_eidat_metadata(pdf)
                    if metadata_present:
                        metadata_ok = _pointer_artifacts_exist(global_repo, pdf)
                        if metadata_ok:
                            needs_processing = 0
                            reason = "metadata_present"
                        else:
                            needs_processing = 1
                            reason = "metadata_missing_artifacts"
                else:
                    # Even if the DB says "processed", a pointer token without artifacts in this node
                    # must trigger processing (common when seeding a new node with already-processed PDFs).
                    metadata_present = _has_eidat_metadata(pdf)
                    if metadata_present:
                        metadata_ok = _pointer_artifacts_exist(global_repo, pdf)
                        if not metadata_ok:
                            needs_processing = 1
                            reason = "metadata_missing_artifacts"
                    # If the expected artifacts folder is missing, schedule regeneration regardless of fitz availability.
                    if last_processed_epoch_ns and not artifacts_ok:
                        needs_processing = 1
                        reason = reason or "missing_artifacts"

                conn.execute(
                    """
                    UPDATE files
                    SET size_bytes = ?,
                        mtime_ns = ?,
                        file_fingerprint = ?,
                        last_seen_epoch_ns = ?,
                        last_processed_epoch_ns = CASE WHEN ? = 1 AND last_processed_epoch_ns IS NULL THEN ? ELSE last_processed_epoch_ns END,
                        needs_processing = CASE WHEN ? = 1 THEN 1 ELSE needs_processing END
                    WHERE rel_path = ?
                    """,
                    (
                        size_bytes,
                        mtime_ns,
                        file_fingerprint,
                        now_ns,
                        1 if metadata_ok else 0,
                        now_ns,
                        needs_processing,
                        rel_path,
                    ),
                )

            if needs_processing:
                candidates.append(
                    ScanCandidate(
                        rel_path=rel_path,
                        abs_path=pdf,
                        size_bytes=size_bytes,
                        mtime_ns=mtime_ns,
                        reason=reason or "unknown",
                    )
                )

        set_meta(conn, "last_scan_epoch_ns", str(now_ns))
        conn.execute(
            """
            INSERT INTO scans(started_epoch_ns, finished_epoch_ns, global_repo, pdf_count, candidates_count)
            VALUES(?, ?, ?, ?, ?)
            """,
            (now_ns, time.time_ns(), str(global_repo), len(data_files), len(candidates)),
        )
        conn.commit()

    return ScanSummary(
        global_repo=global_repo,
        support_dir=paths.support_dir,
        db_path=paths.db_path,
        pdf_count=len(data_files),  # Note: includes Excel files now
        candidates=candidates,
        last_scan_epoch_ns_before=last_scan_before,
        last_scan_epoch_ns_after=now_ns,
    )
