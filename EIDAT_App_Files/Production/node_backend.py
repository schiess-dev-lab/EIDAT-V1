from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable


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


def _bind_node_env(node_root: str | Path | None) -> Path:
    root = resolve_node_root(node_root)
    os.environ["EIDAT_NODE_ROOT"] = str(root)
    os.environ["EIDAT_DATA_ROOT"] = str(root / "EIDAT" / "UserData")
    return root


def _be(node_root: str | Path | None = None):
    # Import lazily after binding node-local env vars so ui_next.backend
    # resolves writable paths under the selected node, not the central runtime.
    expected_data_root: Path | None = None
    if node_root is not None:
        root = _bind_node_env(node_root)
        expected_data_root = root / "EIDAT" / "UserData"
    from ui_next import backend as be  # type: ignore
    if expected_data_root is not None:
        try:
            current_data_root = Path(str(getattr(be, "DATA_ROOT", ""))).expanduser()
        except Exception:
            current_data_root = None  # type: ignore[assignment]
        if current_data_root is None or current_data_root != expected_data_root:
            be = importlib.reload(be)  # type: ignore[assignment]

    return be


def global_repo(node_root: str | Path | None = None) -> Path:
    return _bind_node_env(node_root)


def support_dir(node_root: str | Path | None = None) -> Path:
    be = _be(node_root)
    return be.eidat_support_dir(global_repo(node_root))


def read_files(node_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Tracked files with join metadata (status comes from EIDAT Support DBs)."""
    be = _be(node_root)
    return list(be.read_files_with_index_metadata(global_repo(node_root)) or [])


def list_projects(node_root: str | Path | None = None) -> list[dict[str, Any]]:
    be = _be(node_root)
    return list(be.list_eidat_projects(global_repo(node_root)) or [])


def update_project(
    node_root: str | Path | None,
    *,
    workbook_path: str | Path,
    project_type: str,
    overwrite: bool = False,
    force_project_rebuild: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    be = _be(node_root)
    repo = global_repo(node_root)
    wb = Path(workbook_path).expanduser()
    if project_type == getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"):
        return dict(be.update_eidp_trending_project_workbook(repo, wb, overwrite=overwrite) or {})
    if project_type == getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"):
        return dict(be.update_eidp_raw_trending_project_workbook(repo, wb, overwrite=overwrite) or {})
    if project_type == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
        return dict(
            be.update_test_data_trending_project_workbook(
                repo,
                wb,
                overwrite=overwrite,
                include_performance_sheets=True,
                source_refresh_mode="smart",
                force_project_rebuild=bool(force_project_rebuild),
                progress_cb=progress_cb,
            )
            or {}
        )
    raise RuntimeError(f"Unsupported project type: {project_type}")


def delete_project(node_root: str | Path | None, *, project_dir: str | Path) -> dict[str, Any]:
    be = _be(node_root)
    repo = global_repo(node_root)
    return dict(be.delete_eidat_project(repo, Path(project_dir).expanduser()) or {})


def open_path(path: str | Path) -> None:
    be = _be()
    be.open_path(Path(path).expanduser())
