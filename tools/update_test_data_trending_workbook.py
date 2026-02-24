"""
Update a "Test Data Trending" project workbook from cached `td_*` tables.

This is a thin CLI wrapper around:
  `EIDAT_App_Files/ui_next/backend.py:update_test_data_trending_project_workbook`

It:
  - migrates legacy workbooks (old `Data` -> `Data_calc` + new EIDP-style `Data`)
  - rebuilds the project cache DB (`implementation_trending.sqlite3`)
  - populates `Data_calc` and `Data` from `td_metrics`

Usage:
  python tools/update_test_data_trending_workbook.py "C:\\path\\to\\Project.xlsx"

Options:
  --global-repo "C:\\path\\to\\NodeRoot"   # folder that contains `EIDAT Support/` (auto-detected if omitted)
  --overwrite                              # overwrite existing non-empty cells
  --dry-run                                # do not save the workbook
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _find_node_root_from_workbook(workbook_path: Path) -> Path:
    """
    Best-effort: if workbook lives under `<node_root>/EIDAT Support/...`,
    return `<node_root>`. Otherwise return workbook parent.
    """
    p = Path(workbook_path).resolve()
    cur = p.parent
    for _ in range(12):
        if cur.name.lower() == "eidat support":
            return cur.parent
        cur = cur.parent
        if cur == cur.parent:
            break
    return p.parent


def _load_backend(repo_root: Path):
    backend_path = repo_root / "EIDAT_App_Files" / "ui_next" / "backend.py"
    if not backend_path.exists():
        raise FileNotFoundError(f"backend.py not found: {backend_path}")
    spec = importlib.util.spec_from_file_location("eidat_ui_backend", backend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load backend module spec: {backend_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Update a Test Data Trending workbook from td_* cache tables.")
    ap.add_argument("workbook", help="Path to the Test Data Trending .xlsx workbook.")
    ap.add_argument(
        "--global-repo",
        default="",
        help="Folder containing `EIDAT Support/` (auto-detected from workbook path if omitted).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty cells in Data/Data_calc.")
    ap.add_argument("--dry-run", action="store_true", help="Do not save the workbook (cache DB may still rebuild).")
    args = ap.parse_args(argv)

    wb_path = Path(args.workbook).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {wb_path}")

    repo_root = Path(__file__).resolve().parent.parent
    backend = _load_backend(repo_root)

    if str(args.global_repo or "").strip():
        repo = Path(args.global_repo).expanduser()
    else:
        repo = _find_node_root_from_workbook(wb_path)

    payload = backend.update_test_data_trending_project_workbook(
        repo,
        wb_path,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise

