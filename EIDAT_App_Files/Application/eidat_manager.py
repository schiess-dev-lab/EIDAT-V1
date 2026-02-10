#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_bootstrap_imports()

from eidat_manager_db import SupportPaths, support_paths  # noqa: E402
from eidat_manager_scan import scan_global_repo  # noqa: E402
from eidat_manager_process import process_candidates  # noqa: E402
from eidat_manager_index import build_index  # noqa: E402


def _cmd_init(paths: SupportPaths) -> dict:
    paths.support_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.staging_dir.mkdir(parents=True, exist_ok=True)
    # DB is created lazily by scan (and later commands).
    return {
        "global_repo": str(paths.global_repo),
        "support_dir": str(paths.support_dir),
        "db_path": str(paths.db_path),
    }


def _cmd_scan(paths: SupportPaths) -> dict:
    summary = scan_global_repo(paths)
    return {
        "global_repo": str(summary.global_repo),
        "support_dir": str(summary.support_dir),
        "db_path": str(summary.db_path),
        "pdf_count": summary.pdf_count,
        "candidates_count": len(summary.candidates),
        "candidates": [
            {
                "rel_path": c.rel_path,
                "abs_path": str(c.abs_path),
                "size_bytes": c.size_bytes,
                "mtime_ns": c.mtime_ns,
                "reason": c.reason,
            }
            for c in summary.candidates
        ],
        "last_scan_epoch_ns_before": summary.last_scan_epoch_ns_before,
        "last_scan_epoch_ns_after": summary.last_scan_epoch_ns_after,
    }


def _cmd_process(paths: SupportPaths, *, limit: int | None, dpi: int | None, force: bool, only_candidates: bool) -> dict:
    results = process_candidates(paths, limit=limit, dpi=dpi, force=force, only_candidates=only_candidates)
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    # Log failed results to stderr for debugging
    for r in failed:
        print(f"[PROCESS FAILED] {r.rel_path}: {r.error}", file=sys.stderr)
    return {
        "global_repo": str(paths.global_repo),
        "support_dir": str(paths.support_dir),
        "db_path": str(paths.db_path),
        "processed_ok": len(ok),
        "processed_failed": len(failed),
        "results": [
            {
                "rel_path": r.rel_path,
                "abs_path": r.abs_path,
                "ok": bool(r.ok),
                "artifacts_dir": r.artifacts_dir,
                "error": r.error,
            }
            for r in results
        ],
    }


def _cmd_index(paths: SupportPaths, *, similarity: float) -> dict:
    summary = build_index(paths, similarity=similarity)
    return {
        "index_db": str(summary.index_db),
        "indexed_count": summary.indexed_count,
        "groups_count": summary.groups_count,
        "metadata_count": summary.metadata_count,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EIDAT Manager: orchestrates scanning a Global Repo and managing EIDAT Support.",
    )
    p.add_argument(
        "--global-repo",
        required=True,
        help="Top-level folder where EIDPs live; EIDAT Support/ is created inside this folder.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout (default).")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="Create EIDAT Support folder scaffolding.")
    sub.add_parser("scan", help="Scan for PDFs and record change tracking in SQLite.")
    sp = sub.add_parser("process", help="Process PDFs marked as needing work into EIDAT Support (OCR merge artifacts).")
    sp.add_argument("--limit", type=int, default=0, help="Max number of PDFs to process (0 = no limit).")
    sp.add_argument("--dpi", type=int, default=0, help="OCR DPI override (0 = use existing config).")
    sp.add_argument("--force", action="store_true", help="Process all tracked PDFs, even if already processed.")
    sp.add_argument(
        "--only-candidates",
        action="store_true",
        help="Process only scan candidates (needs_processing=1). Useful with --force to overwrite only new/changed files.",
    )
    si = sub.add_parser("index", help="Build serial index + similarity grouping from EIDAT Support metadata.")
    si.add_argument("--similarity", type=float, default=0.86, help="Similarity threshold for grouping titles.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    global_repo = Path(args.global_repo).expanduser()
    paths = support_paths(global_repo)

    if args.cmd == "init":
        payload = _cmd_init(paths)
    elif args.cmd == "scan":
        payload = _cmd_scan(paths)
    elif args.cmd == "process":
        limit = int(getattr(args, "limit", 0) or 0)
        dpi = int(getattr(args, "dpi", 0) or 0)
        only_candidates = bool(getattr(args, "only_candidates", False))
        payload = _cmd_process(
            paths,
            limit=(None if limit <= 0 else limit),
            dpi=(None if dpi <= 0 else dpi),
            force=bool(getattr(args, "force", False)),
            only_candidates=only_candidates,
        )
    elif args.cmd == "index":
        sim = float(getattr(args, "similarity", 0.86) or 0.86)
        payload = _cmd_index(paths, similarity=sim)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
