from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_node_root(node_root: str | Path | None = None) -> Path:
    raw = str(node_root or "").strip()
    if not raw:
        raw = (os.environ.get("EIDAT_NODE_ROOT") or "").strip()
    if not raw:
        raise RuntimeError("EIDAT_NODE_ROOT is not set (node_root required).")
    p = Path(raw).expanduser()
    try:
        return p.resolve()
    except Exception:
        return p.absolute()


def _be():
    # Import lazily so callers can set env vars (EIDAT_DATA_ROOT, etc.) first.
    from ui_next import backend as be  # type: ignore

    return be


def global_repo(node_root: str | Path | None = None) -> Path:
    return resolve_node_root(node_root)


def support_dir(node_root: str | Path | None = None) -> Path:
    be = _be()
    return be.eidat_support_dir(global_repo(node_root))


def read_files(node_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Tracked files with join metadata (status comes from EIDAT Support DBs)."""
    be = _be()
    return list(be.read_files_with_index_metadata(global_repo(node_root)) or [])


def list_projects(node_root: str | Path | None = None) -> list[dict[str, Any]]:
    be = _be()
    return list(be.list_eidat_projects(global_repo(node_root)) or [])


def update_project(
    node_root: str | Path | None,
    *,
    workbook_path: str | Path,
    project_type: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    be = _be()
    repo = global_repo(node_root)
    wb = Path(workbook_path).expanduser()
    if project_type == getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"):
        return dict(be.update_eidp_trending_project_workbook(repo, wb, overwrite=overwrite) or {})
    if project_type == getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"):
        return dict(be.update_eidp_raw_trending_project_workbook(repo, wb, overwrite=overwrite) or {})
    raise RuntimeError(f"Unsupported project type: {project_type}")


def delete_project(node_root: str | Path | None, *, project_dir: str | Path) -> dict[str, Any]:
    be = _be()
    repo = global_repo(node_root)
    return dict(be.delete_eidat_project(repo, Path(project_dir).expanduser()) or {})


def open_path(path: str | Path) -> None:
    be = _be()
    be.open_path(Path(path).expanduser())

