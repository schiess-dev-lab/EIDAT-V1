from __future__ import annotations

import importlib
import json
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
    repo = global_repo(node_root)
    out: list[dict[str, Any]] = []
    for raw in list(be.list_eidat_projects(repo) or []):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        project_dir = _resolve_node_project_dir(repo, item.get("project_dir"))
        workbook = _resolve_node_project_workbook(repo, item.get("workbook"), project_dir=project_dir)
        item["project_dir"] = str(project_dir)
        item["workbook"] = str(workbook)
        out.append(item)
    return out


def _node_projects_root(node_root: str | Path | None) -> Path:
    be = _be(node_root)
    return Path(be.eidat_projects_root(global_repo(node_root))).expanduser()


def _try_relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return None


def _project_suffix_under_projects(path: Path) -> Path | None:
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if str(part).strip().casefold() == "projects" and idx + 1 < len(parts):
            return Path(*parts[idx + 1 :])
    return None


def _resolve_node_project_dir(node_root: str | Path | None, project_dir: object) -> Path:
    repo = global_repo(node_root)
    projects_root = _node_projects_root(repo)
    text = str(project_dir or "").strip()
    if not text:
        return projects_root
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return (repo / candidate).expanduser()
    if _try_relative_to(candidate, repo) is not None:
        return candidate
    suffix = _project_suffix_under_projects(candidate)
    if suffix is not None:
        return projects_root / suffix
    return projects_root / candidate.name


def _resolve_node_project_workbook(
    node_root: str | Path | None,
    workbook_path: object,
    *,
    project_dir: Path | None = None,
) -> Path:
    repo = global_repo(node_root)
    projects_root = _node_projects_root(repo)
    proj_dir = Path(project_dir).expanduser() if project_dir is not None else projects_root
    text = str(workbook_path or "").strip()
    if not text:
        return proj_dir
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return (repo / candidate).expanduser()
    if _try_relative_to(candidate, repo) is not None:
        return candidate
    suffix = _project_suffix_under_projects(candidate)
    if suffix is not None:
        return projects_root / suffix
    return proj_dir / candidate.name


def _rewrite_project_meta_for_node(project_dir: Path, *, node_root: Path, workbook_path: Path) -> None:
    meta_path = Path(project_dir).expanduser() / "project.json"
    if not meta_path.exists():
        return
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    payload = dict(raw)
    changed = False
    desired_repo = str(Path(node_root).expanduser())
    desired_project_dir = str(Path(project_dir).expanduser())
    desired_workbook = str(Path(workbook_path).expanduser())
    if str(payload.get("global_repo") or "").strip() != desired_repo:
        payload["global_repo"] = desired_repo
        changed = True
    if str(payload.get("project_dir") or "").strip() != desired_project_dir:
        payload["project_dir"] = desired_project_dir
        changed = True
    if str(payload.get("workbook") or "").strip() != desired_workbook:
        payload["workbook"] = desired_workbook
        changed = True
    if changed:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    project_dir = _resolve_node_project_dir(repo, Path(workbook_path).expanduser().parent)
    wb = _resolve_node_project_workbook(repo, workbook_path, project_dir=project_dir)
    _rewrite_project_meta_for_node(project_dir, node_root=repo, workbook_path=wb)
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
