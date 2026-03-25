"""Production deployment helpers for EIDAT nodes."""

from __future__ import annotations

from pathlib import Path
import shutil


PRODUCT_CENTER_RUNTIME_REL = Path("user_inputs") / "product_center"


def runtime_product_center_root(runtime_root: str | Path) -> Path:
    return Path(runtime_root).expanduser() / PRODUCT_CENTER_RUNTIME_REL


def node_product_center_root(node_root: str | Path) -> Path:
    from .node_layout import node_layout

    return node_layout(node_root).user_data_root / PRODUCT_CENTER_RUNTIME_REL


def sync_product_center_assets(runtime_root: str | Path, node_root: str | Path) -> dict[str, object]:
    src_root = runtime_product_center_root(runtime_root)
    dst_root = node_product_center_root(node_root)
    copied = 0
    scanned = 0

    dst_root.mkdir(parents=True, exist_ok=True)
    if not src_root.exists():
        return {"src": str(src_root), "dst": str(dst_root), "copied": 0, "scanned": 0, "present": False}

    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        scanned += 1
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        should_copy = True
        if dst.exists():
            try:
                s_stat = src.stat()
                d_stat = dst.stat()
                should_copy = not (
                    int(s_stat.st_size) == int(d_stat.st_size)
                    and int(s_stat.st_mtime_ns) <= int(d_stat.st_mtime_ns)
                )
            except Exception:
                should_copy = True
        if should_copy:
            shutil.copy2(src, dst)
            copied += 1

    return {"src": str(src_root), "dst": str(dst_root), "copied": copied, "scanned": scanned, "present": True}
