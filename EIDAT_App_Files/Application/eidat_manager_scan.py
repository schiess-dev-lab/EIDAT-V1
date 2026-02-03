from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

try:
    from eidat_manager_db import SupportPaths, connect_db, ensure_schema, get_meta_int, set_meta
except Exception:  # pragma: no cover
    from .eidat_manager_db import SupportPaths, connect_db, ensure_schema, get_meta_int, set_meta  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


# Supported file extensions for scanning
SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".xlsm"}


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
    try:
        exclude_dir_res = exclude_dir.resolve()
    except Exception:
        exclude_dir_res = exclude_dir.expanduser().absolute()

    for p in global_repo.rglob("*"):
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
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
            pr = p.resolve()
        except Exception:
            pr = p.expanduser().absolute()
        try:
            pr.relative_to(exclude_dir_res)
            continue
        except Exception:
            pass
        files.append(p)
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
                if moved is not None:
                    needs_processing = 1 if not moved["last_processed_epoch_ns"] else 0
                    reason = "moved_unprocessed" if needs_processing else "moved"
                    if needs_processing:
                        metadata_present = _has_eidat_metadata(pdf)
                        if metadata_present:
                            needs_processing = 0
                            reason = "metadata_present"
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
                        (rel_path, size_bytes, mtime_ns, file_fingerprint, now_ns, 1 if metadata_present else 0, now_ns, needs_processing, int(moved["id"])),
                    )
                else:
                    needs_processing = 1
                    reason = "new"
                    metadata_present = _has_eidat_metadata(pdf)
                    if metadata_present:
                        needs_processing = 0
                        reason = "metadata_present"
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
                            now_ns if metadata_present else None,
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
                        needs_processing = 0
                        reason = "metadata_present"

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
                        1 if metadata_present else 0,
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
