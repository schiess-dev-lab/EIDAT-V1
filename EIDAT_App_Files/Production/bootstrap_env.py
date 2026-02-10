#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import shutil
from pathlib import Path

from .node_layout import node_layout, venv_python


PROFILE_REQUIRED_IMPORTS: dict[str, list[str]] = {
    # Node end-user GUI (Files + Projects only).
    "ui": [
        "PySide6",
        "openpyxl",
        "xlsxwriter",  # XlsxWriter package
    ],
    # Full environment (admin/extraction tooling).
    "full": [
        "PySide6",
        "fitz",  # pymupdf
        "pandas",
        "openpyxl",
        "xlsxwriter",
        "matplotlib",
        "cv2",  # opencv-python-headless
    ],
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _requirements_hash(requirements_path: Path) -> str:
    content = _read_text(requirements_path).replace("\r\n", "\n")
    return _sha256_text(content.strip() + "\n")


def _prompt_yes_no(question: str, *, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        try:
            raw = input(f"{question} {suffix} ").strip().lower()
        except EOFError:
            return default_yes
        if not raw:
            return default_yes
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.", file=sys.stderr)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _ensure_venv(sys_py: str, venv_dir: Path, *, assume_yes: bool) -> None:
    vpy = venv_python(venv_dir)
    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    needs_create = not vpy.exists()
    prompt_create = needs_create

    # If the python shim exists but pyvenv.cfg is missing, the venv is broken (often due to a partial
    # delete or interrupted creation). Recreate it to avoid opaque failures like ensurepip:
    # "failed to locate pyvenv.cfg: The system cannot find the path specified."
    if vpy.exists():
        if pyvenv_cfg.exists():
            return
        if not assume_yes:
            if not _prompt_yes_no(
                f"Existing EIDAT environment at '{venv_dir}' appears broken (missing pyvenv.cfg). Recreate it now?",
                default_yes=True,
            ):
                raise RuntimeError("EIDAT environment is broken and was not recreated.")
        try:
            shutil.rmtree(venv_dir)
        except FileNotFoundError:
            pass
        vpy = venv_python(venv_dir)
        pyvenv_cfg = venv_dir / "pyvenv.cfg"
        needs_create = True
        prompt_create = False

    if not needs_create:
        return

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if prompt_create and not assume_yes:
        if not _prompt_yes_no(f"Create the EIDAT environment at '{venv_dir}' now?", default_yes=True):
            raise RuntimeError("EIDAT environment not created.")
    proc = _run([sys_py, "-m", "venv", str(venv_dir)])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip() or "Failed to create venv.")
    if not (vpy.exists() and pyvenv_cfg.exists()):
        raise RuntimeError(f"Venv creation did not produce expected files: {vpy} and {pyvenv_cfg}")


def _ensure_pip(vpy: Path) -> None:
    proc = _run([str(vpy), "-m", "pip", "--version"])
    if proc.returncode == 0:
        return
    proc = _run([str(vpy), "-m", "ensurepip", "--upgrade"])
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip() or "Failed to bootstrap pip.")


def _venv_can_import(vpy: Path, module: str) -> bool:
    code = f"import importlib; importlib.import_module({module!r}); print('ok')"
    proc = _run([str(vpy), "-c", code])
    return proc.returncode == 0


def _install_requirements(vpy: Path, requirements_path: Path, *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    parts: list[str] = []

    def run_and_log(cmd: list[str]) -> None:
        proc = _run(cmd)
        parts.append(f"$ {' '.join(cmd)}\n")
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(proc.stderr)
        parts.append("\n")
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip() or "pip failed")

    run_and_log([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
    run_and_log([str(vpy), "-m", "pip", "install", "-r", str(requirements_path)])

    duration = time.time() - start
    parts.append(f"[DONE] install_seconds={duration:.1f}\n")
    _write_text(log_path, "".join(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap a venv (node-local or shared) and required packages.")
    ap.add_argument("--node-root", required=True, help="Node root directory (contains 'EIDAT Support').")
    ap.add_argument(
        "--profile",
        choices=["ui", "full"],
        default="ui",
        help="Dependency profile: ui (end-user GUI) or full (admin/extraction).",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Run non-interactively (assume yes to prompts).",
    )
    ap.add_argument(
        "--venv-dir",
        default="",
        help="Optional venv directory override (default: <node_root>/EIDAT/Runtime/.venv).",
    )
    ap.add_argument(
        "--requirements",
        default=str(Path(__file__).resolve().parent / "requirements-node.txt"),
        help="Path to requirements file (defaults to Production/requirements-node.txt).",
    )
    ap.add_argument("--sys-python", default="", help="System python executable to use for venv creation (optional).")
    args = ap.parse_args(argv)

    layout = node_layout(Path(args.node_root))
    req_path = Path(args.requirements).expanduser()
    if not req_path.exists():
        print(f"Requirements file not found: {req_path}", file=sys.stderr)
        return 2

    sys_py = (args.sys_python or "").strip() or sys.executable
    if str(args.venv_dir or "").strip():
        venv_dir = Path(args.venv_dir).expanduser()
        if not venv_dir.is_absolute():
            venv_dir = (layout.runtime_dir / venv_dir).expanduser()
        installed_hash_path = venv_dir / "eidat_installed_hash.txt"
        last_install_log = venv_dir / "eidat_last_install_log.txt"
    else:
        venv_dir = layout.venv_dir
        installed_hash_path = layout.runtime_dir / "installed_hash.txt"
        last_install_log = layout.runtime_dir / "last_install_log.txt"

    assume_yes = bool(args.yes)
    _ensure_venv(sys_py, venv_dir, assume_yes=assume_yes)
    vpy = venv_python(venv_dir)
    if not vpy.exists():
        print(f"Venv python not found: {vpy}", file=sys.stderr)
        return 2

    _ensure_pip(vpy)

    want_hash = _requirements_hash(req_path)
    have_hash = _read_text(installed_hash_path).strip()
    required = PROFILE_REQUIRED_IMPORTS.get(str(args.profile or "").strip().lower(), PROFILE_REQUIRED_IMPORTS["ui"])
    missing = [m for m in required if not _venv_can_import(vpy, m)]
    needs_install = bool(missing) or (have_hash != want_hash)

    if not needs_install:
        return 0

    why = []
    if have_hash != want_hash:
        why.append("requirements changed")
    if missing:
        why.append("missing: " + ", ".join(missing))
    reason = "; ".join(why) if why else "dependencies not satisfied"

    if not assume_yes:
        if not _prompt_yes_no(
            f"Install/Update required packages now? ({reason}) This may take a few minutes.",
            default_yes=True,
        ):
            print("Required packages are not installed; cannot continue.", file=sys.stderr)
            return 1

    _install_requirements(vpy, req_path, log_path=last_install_log)
    _write_text(installed_hash_path, want_hash + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
