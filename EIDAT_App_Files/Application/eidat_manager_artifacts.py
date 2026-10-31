"""
Collision-safe artifact folder naming for Excel data sources.

Historically every Excel source wrote its debug artifacts (per-file SQLite,
mirrors, `<stem>_metadata.json`) to ``EIDAT Support/debug/ocr/<stem>__excel``,
keyed only by the file *name*. Two sources with the same name in different
folders (the same serial number under two assets / asset specific types, e.g.
``Program/ValveA/SN123.xlsx`` and ``Program/ValveB/SN123.xlsx``) therefore
shared one folder: the second file loaded the first file's metadata as
"curated" (inheriting its program / asset / specific type / serial), overwrote
its SQLite, and the index — unique per ``metadata_rel`` — collapsed both into
one document and one serial object.

This module keeps the legacy folder for the first source that claimed it and
moves any *other* source with the same stem to ``<stem>__<hash10>__excel``
(the same ``sha1(resolved path)[:10]`` convention the official Test Data
``sources/`` leaf already uses), so each physical file keeps its own identity.
Scan, process and index all resolve the folder through the same function.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

EXCEL_ARTIFACT_SUFFIX = "__excel"
EXCEL_SOURCE_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}


def source_path_hash(file_path: Path) -> str:
    """Stable 10-hex hash of a source file's resolved path (case-insensitive)."""
    try:
        basis = str(Path(file_path).expanduser().resolve()).casefold()
    except Exception:
        basis = str(Path(file_path).expanduser()).casefold()
    return hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()[:10]


def legacy_excel_artifacts_dir(support_dir: Path, file_path: Path) -> Path:
    stem = str(Path(file_path).stem or "").strip() or "unknown"
    return Path(support_dir) / "debug" / "ocr" / f"{stem}{EXCEL_ARTIFACT_SUFFIX}"


def hashed_excel_artifacts_dir(support_dir: Path, file_path: Path) -> Path:
    stem = str(Path(file_path).stem or "").strip() or "unknown"
    return Path(support_dir) / "debug" / "ocr" / f"{stem}__{source_path_hash(file_path)}{EXCEL_ARTIFACT_SUFFIX}"


def _norm_rel(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/").casefold()


def _repo_rel(repo_root: Path | None, file_path: Path) -> str:
    p = Path(file_path).expanduser()
    if repo_root is None:
        return ""
    root = Path(repo_root).expanduser()
    for a, b in ((p.resolve(), root.resolve()), (p, root)):
        try:
            return _norm_rel(a.relative_to(b).as_posix())
        except Exception:
            continue
    return ""


def artifacts_dir_owner_paths(artifacts_dir: Path, file_path: Path) -> list[str]:
    """
    Source paths recorded in the `<stem>_metadata.json` inside `artifacts_dir`
    (normalized repo-relative when possible, else the raw value). Empty when
    the folder has no metadata or the metadata predates source tracking.
    """
    stem = str(Path(file_path).stem or "").strip()
    candidates = [
        Path(artifacts_dir) / f"{stem}_metadata.json",
        Path(artifacts_dir) / f"{stem}.metadata.json",
    ]
    meta: dict = {}
    for cand in candidates:
        try:
            if cand.exists():
                raw = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    meta = raw
                    break
        except Exception:
            continue
    if not meta:
        return []
    out: list[str] = []
    rels = meta.get("source_rel_paths")
    if isinstance(rels, (list, tuple)):
        out.extend(_norm_rel(v) for v in rels if str(v or "").strip())
    for key in ("source_rel_path", "source_file"):
        value = str(meta.get(key) or "").strip()
        if value:
            out.append(_norm_rel(value))
    return [v for v in dict.fromkeys(out) if v]


def _owner_matches(owners: Iterable[str], *, rel_path: str, abs_path: Path) -> bool:
    rel_key = _norm_rel(rel_path)
    abs_keys = set()
    try:
        abs_keys.add(_norm_rel(str(Path(abs_path).expanduser().resolve())))
    except Exception:
        pass
    abs_keys.add(_norm_rel(str(Path(abs_path).expanduser())))
    for owner in owners:
        if not owner:
            continue
        if rel_key and (owner == rel_key or owner.endswith("/" + rel_key)):
            return True
        if owner in abs_keys:
            return True
        # Owner recorded as an absolute path on another machine: compare the
        # repo-relative tail instead.
        if rel_key and owner.endswith(rel_key):
            return True
    return False


def excel_artifacts_dir(support_dir: Path, file_path: Path, *, repo_root: Path | None = None) -> Path:
    """
    Resolve the debug artifacts folder for an Excel source file.

    - If a hashed folder for this exact file already exists, use it.
    - Else if the legacy `<stem>__excel` folder exists and its metadata says it
      belongs to a *different* source file, use the hashed folder (collision).
    - Else use the legacy folder (first claimant keeps the historical name, and
      pre-existing artifacts with no source tracking stay valid).
    """
    support = Path(support_dir)
    legacy = legacy_excel_artifacts_dir(support, file_path)
    hashed = hashed_excel_artifacts_dir(support, file_path)
    try:
        if hashed.exists():
            return hashed
    except Exception:
        pass
    try:
        legacy_exists = legacy.exists()
    except Exception:
        legacy_exists = False
    if not legacy_exists:
        return legacy
    owners = artifacts_dir_owner_paths(legacy, file_path)
    if not owners:
        return legacy
    rel_path = _repo_rel(repo_root, file_path)
    if _owner_matches(owners, rel_path=rel_path, abs_path=Path(file_path)):
        return legacy
    return hashed


def excel_artifacts_rel_candidates(stem: str) -> tuple[str, str]:
    """Patterns (legacy name, hashed-name glob) under `debug/ocr` for a stem."""
    clean = str(stem or "").strip() or "unknown"
    return (f"{clean}{EXCEL_ARTIFACT_SUFFIX}", f"{clean}__??????????{EXCEL_ARTIFACT_SUFFIX}")


def folder_asset_hints(file_path: Path) -> dict[str, str]:
    """
    Asset hints from the source's folder structure, `<AssetType>/<AssetSpecificType>/<file>`
    (the same convention `.mat` bundles already use as a fallback).
    """
    parent = Path(file_path).expanduser().parent
    asset_specific_type = str(parent.name or "").strip()
    asset_type = str(parent.parent.name or "").strip() if parent.parent != parent else ""
    return {
        "asset_type": asset_type or "Unknown",
        "asset_specific_type": asset_specific_type or "Unknown",
    }
