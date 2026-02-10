from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import admin_db


def _as_abs(p: str | Path) -> Path:
    path = Path(p).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _default_runtime_root() -> Path:
    # .../EIDAT_App_Files/Production/node_mirror.py -> runtime root is 2 levels up
    return Path(__file__).resolve().parents[2]


def _safe_slug(text: str) -> str:
    s = (text or "").strip().replace("\\", "_").replace("/", "_")
    out: list[str] = []
    last_us = False
    for ch in s:
        ok = ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in ("-", "_", ".")
        if ok:
            out.append(ch.lower())
            last_us = False
        else:
            if not last_us:
                out.append("_")
                last_us = True
    slug = "".join(out).strip("._-")
    return slug or "node"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _is_windows() -> bool:
    return os.name == "nt"


def _is_reparse_point(path: Path) -> bool:
    try:
        st = os.stat(str(path), follow_symlinks=False)
        attrs = getattr(st, "st_file_attributes", 0)
        # FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
        return bool(int(attrs) & 0x0400)
    except FileNotFoundError:
        return False
    except Exception:
        return bool(path.is_symlink())


def _remove_link_or_dir(path: Path) -> None:
    if not path.exists():
        return
    if _is_windows():
        # Use rmdir to remove junctions/dir symlinks without touching the target.
        subprocess.run(["cmd", "/c", "rmdir", str(path)], check=False, capture_output=True, text=True)
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        return
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        path.unlink()
    except Exception:
        pass


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mklink_junction(dest: Path, target: Path) -> bool:
    if not _is_windows():
        return False
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and dest.exists()


def _mklink_symlink_dir(dest: Path, target: Path) -> bool:
    if not _is_windows():
        try:
            dest.symlink_to(target, target_is_directory=True)
            return dest.exists()
        except Exception:
            return False
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/D", str(dest), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and dest.exists()


def _robocopy_mirror(src: Path, dest: Path) -> bool:
    if not _is_windows():
        return False
    _ensure_dir(dest)
    # robocopy exit codes 0-7 are success (includes "copied" / "extra files" etc).
    proc = subprocess.run(
        [
            "robocopy",
            str(src),
            str(dest),
            "/MIR",
            "/R:1",
            "/W:1",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NP",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode <= 7


def _index_path(mirror_root: Path) -> Path:
    return mirror_root / "_node_mirror_index.json"


def _read_index(mirror_root: Path) -> dict[str, object]:
    p = _index_path(mirror_root)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updated_epoch_ns": 0, "nodes": {}}


def _write_index(mirror_root: Path, data: dict[str, object]) -> None:
    p = _index_path(mirror_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def ensure_node_mirror(
    *,
    node_root: str | Path,
    runtime_root: str | Path,
    mirror_root: str | Path | None = None,
    mode: str = "auto",
) -> dict[str, object]:
    """
    Create/refresh a repo-local mirror entry for a node root (for debugging).

    The default behavior ("auto") tries:
      1) directory junction (Windows)
      2) directory symlink
      3) robocopy /MIR (Windows) as fallback
    """
    node_root_p = _as_abs(node_root)
    runtime_root_p = _as_abs(runtime_root)
    mirror_root_p = _as_abs(mirror_root) if mirror_root else (runtime_root_p / "node_mirror")
    nodes_dir = mirror_root_p / "nodes"
    _ensure_dir(nodes_dir)

    base = node_root_p.name or "node"
    slug = f"{_safe_slug(base)}__{_short_hash(str(node_root_p))}"
    dest = nodes_dir / slug

    if dest.exists():
        if _is_reparse_point(dest) or dest.is_symlink():
            _remove_link_or_dir(dest)
        else:
            # Avoid deleting user-created data; rotate it out of the way.
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            backup = nodes_dir / f"{slug}__backup_{ts}"
            try:
                dest.rename(backup)
            except Exception:
                _remove_link_or_dir(dest)

    used = "none"
    ok = False
    err = ""
    want = str(mode or "").strip().lower() or "auto"
    try:
        if want in ("auto", "junction") and _is_windows():
            ok = _mklink_junction(dest, node_root_p)
            used = "junction" if ok else used

        if not ok and want in ("auto", "symlink"):
            ok = _mklink_symlink_dir(dest, node_root_p)
            used = "symlink" if ok else used

        if not ok and want in ("auto", "copy") and _is_windows():
            ok = _robocopy_mirror(node_root_p, dest)
            used = "copy" if ok else used

        if not ok:
            used = used if used != "none" else want
            raise RuntimeError(f"Unable to mirror node root using mode={want!r}.")
    except Exception as exc:
        err = str(exc)

    idx = _read_index(mirror_root_p)
    nodes = idx.get("nodes")
    if not isinstance(nodes, dict):
        nodes = {}
        idx["nodes"] = nodes
    nodes[slug] = {
        "node_root": str(node_root_p),
        "mirror_path": str(dest),
        "mode": used,
        "ok": bool(ok) and not bool(err),
        "error": err,
        "updated_epoch_ns": int(time.time_ns()),
    }
    idx["updated_epoch_ns"] = int(time.time_ns())
    idx["runtime_root"] = str(runtime_root_p)
    _write_index(mirror_root_p, idx)

    return {"ok": bool(ok) and not bool(err), "mode": used, "slug": slug, "dest": str(dest), "error": err}


def mirror_all_from_registry(
    *,
    runtime_root: str | Path,
    registry_path: str | Path | None = None,
    mirror_root: str | Path | None = None,
    mode: str = "auto",
) -> dict[str, object]:
    runtime_root_p = _as_abs(runtime_root)
    reg = _as_abs(registry_path) if registry_path else admin_db.default_registry_path()
    conn = admin_db.connect_registry(reg)
    try:
        nodes = admin_db.list_nodes(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    results: list[dict[str, object]] = []
    ok_count = 0
    for n in nodes:
        res = ensure_node_mirror(
            node_root=n.node_root,
            runtime_root=runtime_root_p,
            mirror_root=mirror_root,
            mode=mode,
        )
        results.append({"node_root": n.node_root, **res})
        if res.get("ok"):
            ok_count += 1

    return {"ok": ok_count == len(results), "count": len(results), "ok_count": ok_count, "results": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create a repo-local mirror of deployed nodes for debugging.")
    ap.add_argument("--runtime-root", default="", help="Central runtime root (contains EIDAT_App_Files).")
    ap.add_argument("--mirror-root", default="", help="Output root for mirrors (default: <runtime_root>\\node_mirror).")
    ap.add_argument(
        "--mode",
        choices=["auto", "junction", "symlink", "copy"],
        default="auto",
        help="Mirror mode: junction/symlink/copy (auto tries junction->symlink->copy).",
    )

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--node-root", default="", help="Mirror a single node root.")
    g.add_argument("--all-from-registry", action="store_true", help="Mirror all nodes from the admin registry.")

    ap.add_argument("--registry", default="", help="Override admin registry DB path (for --all-from-registry).")
    args = ap.parse_args(argv)

    runtime_root = _as_abs(args.runtime_root) if str(args.runtime_root or "").strip() else _default_runtime_root()
    mirror_root = _as_abs(args.mirror_root) if str(args.mirror_root or "").strip() else None
    mode = str(args.mode or "auto").strip().lower()

    if str(args.node_root or "").strip():
        res = ensure_node_mirror(node_root=args.node_root, runtime_root=runtime_root, mirror_root=mirror_root, mode=mode)
        if not res.get("ok"):
            print(str(res.get("error") or "Mirror failed"), file=sys.stderr)
            return 1
        print(json.dumps(res, indent=2))
        return 0

    reg = _as_abs(args.registry) if str(args.registry or "").strip() else None
    out = mirror_all_from_registry(runtime_root=runtime_root, registry_path=reg, mirror_root=mirror_root, mode=mode)
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

