#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sqlite3
import subprocess
import sys
import time
import uuid
import shutil
from pathlib import Path

from .node_layout import node_layout


NODE_DB_NAME = "eidat_node.sqlite3"


def _as_abs(p: str | Path) -> Path:
    path = Path(p).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_rmtree_if_present(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        return
    shutil.rmtree(path, ignore_errors=True)


def _node_ui_venv_dir(layout) -> Path:
    # Keep consistent with the launcher generated in _render_eidat_bat (EIDAT_VENV_DIR default).
    return layout.extraction_node_dir / "node-ui" / ".venv"


def _render_eidat_bat(*, runtime_root: Path) -> str:
    run_line = (
        "  set \"MODE=files\"\r\n"
        "  if /i \"%~1\"==\"projects\" set \"MODE=projects\"\r\n"
        "  if /i \"%~1\"==\"files\" set \"MODE=files\"\r\n"
        "  \"%VENV_PY%\" -m Production.node_gui --node-root \"%NODE_ROOT%\" --start-tab \"%MODE%\"\r\n"
    )

    return (
        "@echo off\r\n"
        "setlocal EnableExtensions EnableDelayedExpansion\r\n"
        "\r\n"
        "rem Allow callers to override RUNTIME_ROOT (e.g. different deployment location).\r\n"
        f'if not defined RUNTIME_ROOT set "RUNTIME_ROOT={runtime_root}"\r\n'
        'set "HERE=%~dp0"\r\n'
        'for %%I in ("%HERE%..") do set "NODE_ROOT=%%~fI"\r\n'
        "\r\n"
        'set "APP_ROOT=%RUNTIME_ROOT%\\EIDAT_App_Files"\r\n'
        'set "PYTHONPATH=%APP_ROOT%;%APP_ROOT%\\Lib\\site-packages;%PYTHONPATH%"\r\n'
        'set "EIDAT_NODE_ROOT=%NODE_ROOT%"\r\n'
        'set "EIDAT_DATA_ROOT=%NODE_ROOT%\\EIDAT\\UserData"\r\n'
        'set "PYTHONDONTWRITEBYTECODE=1"\r\n'
        "\r\n"
        "set \"SYS_PY_EXE=\"\r\n"
        "set \"SYS_PY_ARGS=\"\r\n"
        "\r\n"
        "rem Prefer py launcher (does not require python.exe on PATH)\r\n"
        "where py >nul 2>nul && (py -3 -c \"import sys\" >nul 2>nul && set \"SYS_PY_EXE=py\" && set \"SYS_PY_ARGS=-3\")\r\n"
        "\r\n"
        "rem Optional node-local override: EIDAT\\Runtime\\sys_python.txt contains full path to python.exe\r\n"
        "if not defined SYS_PY_EXE (\r\n"
        "  set \"PYCFG=%NODE_ROOT%\\EIDAT\\Runtime\\sys_python.txt\"\r\n"
        "  if exist \"%PYCFG%\" (\r\n"
        "    for /f \"usebackq tokens=* delims=\" %%P in (\"%PYCFG%\") do (\r\n"
        "      set \"LINE=%%P\"\r\n"
        "      if not \"!LINE!\"==\"\" if not \"!LINE:~0,1!\"==\"#\" if not \"!LINE:~0,1!\"==\";\" (\r\n"
        "        set \"SYS_PY_EXE=!LINE!\"\r\n"
        "        goto :eidat_have_sys_py\r\n"
        "      )\r\n"
        "    )\r\n"
        "  )\r\n"
        ")\r\n"
        "\r\n"
        "rem Fallback: python on PATH (some environments allow this)\r\n"
        "if not defined SYS_PY_EXE (\r\n"
        "  where python >nul 2>nul && (python -c \"import sys\" >nul 2>nul && set \"SYS_PY_EXE=python\")\r\n"
        ")\r\n"
        "\r\n"
        ":eidat_have_sys_py\r\n"
        "if not defined SYS_PY_EXE (\r\n"
        "  echo [ERROR] Python 3 not found.\r\n"
        "  echo         Install Python 3 with the 'py' launcher OR set a node-local python path in:\r\n"
        "  echo           %NODE_ROOT%\\EIDAT\\Runtime\\sys_python.txt\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        "\r\n"
        "rem Sanity check interpreter\r\n"
        "\"%SYS_PY_EXE%\" %SYS_PY_ARGS% -c \"import sys\" >nul 2>nul\r\n"
        "if errorlevel 1 (\r\n"
        "  echo [ERROR] Python command failed: \"%SYS_PY_EXE%\" %SYS_PY_ARGS%\r\n"
        "  echo         Fix sys_python.txt or reinstall Python.\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        "\r\n"
        "rem Default: node-local venv (MVP). This avoids Microsoft Store Python path redirects under AppData.\r\n"
        "rem Override by setting EIDAT_VENV_DIR (full path) before launching.\r\n"
        "if not defined EIDAT_VENV_DIR (\r\n"
        "  set \"EIDAT_VENV_DIR=%NODE_ROOT%\\EIDAT\\ExtractionNode\\node-ui\\.venv\"\r\n"
        ")\r\n"
        "\r\n"
        "set \"VENV_PY=%EIDAT_VENV_DIR%\\Scripts\\python.exe\"\r\n"
        "\r\n"
        "\"%SYS_PY_EXE%\" %SYS_PY_ARGS% -m Production.bootstrap_env --profile ui --node-root \"%NODE_ROOT%\" --venv-dir \"%EIDAT_VENV_DIR%\" --requirements \"%APP_ROOT%\\Production\\requirements-node-ui.txt\"\r\n"
        "if errorlevel 1 exit /b 1\r\n"
        "\r\n"
        "if not exist \"%VENV_PY%\" (\r\n"
        "  echo [ERROR] EIDAT venv python not found: \"%VENV_PY%\"\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        "\r\n"
        + run_line +
        "\r\n"
        "endlocal & exit /b %ERRORLEVEL%\r\n"
    )


def _ensure_node_db(db_path: Path, *, node_root: Path, runtime_root: Path, parent_node_root: str | None) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS node (
              node_id TEXT PRIMARY KEY,
              node_root TEXT NOT NULL,
              parent_node_root TEXT,
              runtime_root TEXT NOT NULL,
              created_epoch_ns INTEGER NOT NULL,
              enabled INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        row = conn.execute("SELECT node_id FROM node LIMIT 1").fetchone()
        if row:
            conn.execute(
                "UPDATE node SET node_root=?, parent_node_root=?, runtime_root=?",
                (str(node_root), parent_node_root, str(runtime_root)),
            )
        else:
            conn.execute(
                "INSERT INTO node(node_id, node_root, parent_node_root, runtime_root, created_epoch_ns, enabled) VALUES(?, ?, ?, ?, ?, 1)",
                (str(uuid.uuid4()), str(node_root), parent_node_root, str(runtime_root), int(time.time_ns())),
            )
        conn.commit()


def _grant_writers(path: Path, principal: str) -> None:
    if not sys.platform.startswith("win"):
        return
    subprocess.run(
        ["icacls", str(path), "/grant", f"{principal}:(OI)(CI)M"],
        check=False,
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deposit EIDAT node scaffolding into a shared repository.")
    ap.add_argument("--node-root", required=True, help="Root directory to manage (the node).")
    ap.add_argument("--runtime-root", required=True, help="Central runtime root (contains EIDAT_App_Files).")
    ap.add_argument("--parent-node-root", default="", help="Optional parent node root (breadcrumbs only).")
    ap.add_argument("--ensure-node-venv", action="store_true", help="Pre-create the node venv folder location (no packages).")
    ap.add_argument(
        "--bootstrap-node-ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bootstrap the node-local UI venv and install required packages during deploy (default: true).",
    )
    ap.add_argument(
        "--node-ui-venv-dir",
        default="",
        help="Optional override venv directory for node UI bootstrap (default: <node_root>\\EIDAT\\ExtractionNode\\node-ui\\.venv).",
    )
    ap.add_argument(
        "--bootstrap-sys-python",
        default="",
        help="Optional python executable to use for venv creation (default: current interpreter).",
    )
    ap.add_argument(
        "--mirror-node",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/update a repo-local node mirror entry for debugging (default: true).",
    )
    ap.add_argument(
        "--mirror-root",
        default="",
        help="Override mirror root directory (default: <runtime_root>\\node_mirror).",
    )
    ap.add_argument("--requirements-source", default="", help="Override requirements-node-ui.txt source path.")
    ap.add_argument("--grant-writers", default="", help="Optional DOMAIN\\Group to grant Modify rights on node runtime + projects.")
    args = ap.parse_args(argv)

    node_root = _as_abs(args.node_root)
    runtime_root = _as_abs(args.runtime_root)
    if not (runtime_root / "EIDAT_App_Files").exists():
        raise SystemExit(f"runtime-root does not contain EIDAT_App_Files: {runtime_root}")

    layout = node_layout(node_root)
    # Legacy folders from older node layouts (safe to delete; they only held launcher .bat files).
    _safe_rmtree_if_present(layout.eidat_root / "FileExplorer")
    _safe_rmtree_if_present(layout.eidat_root / "Projects")
    for d in (
        layout.eidat_root,
        layout.runtime_dir,
        layout.extraction_node_dir,
        layout.support_projects_dir,
        layout.user_data_root / "user_inputs",
    ):
        d.mkdir(parents=True, exist_ok=True)

    if args.ensure_node_venv:
        layout.venv_dir.mkdir(parents=True, exist_ok=True)
        _node_ui_venv_dir(layout).mkdir(parents=True, exist_ok=True)

    # Create a node-local python path template for environments where python.exe isn't on PATH.
    sys_py_cfg = layout.runtime_dir / "sys_python.txt"
    if not sys_py_cfg.exists():
        _write_text(
            sys_py_cfg,
            "# Optional: full path to python.exe used to create the node venv.\n"
            "# Example:\n"
            "# C:\\Program Files\\Python313\\python.exe\n",
        )

    # Node-level env overrides (merged after central runtime scanner.env when enabled from the Admin Dashboard).
    node_env = layout.user_data_root / ".env"
    if not node_env.exists():
        default_env = (runtime_root / "user_inputs" / "scanner.env")
        try:
            if default_env.exists():
                node_env.parent.mkdir(parents=True, exist_ok=True)
                node_env.write_bytes(default_env.read_bytes())
            else:
                _write_text(
                    node_env,
                    "# Node-level overrides (KEY=VALUE)\n"
                    "#\n"
                    "# This file is NOT required. When enabled for this node in the Admin Dashboard,\n"
                    "# values here override the central runtime scanner.env.\n"
                    "\n",
                )
        except Exception:
            pass

    req_src = Path(args.requirements_source).expanduser() if str(args.requirements_source or "").strip() else (Path(__file__).resolve().parent / "requirements-node-ui.txt")
    if not req_src.exists():
        raise SystemExit(f"requirements source not found: {req_src}")
    req_lock = layout.runtime_dir / "requirements.lock.txt"
    _write_text(req_lock, req_src.read_text(encoding="utf-8", errors="ignore"))

    _ensure_node_db(
        layout.extraction_node_dir / NODE_DB_NAME,
        node_root=node_root,
        runtime_root=runtime_root,
        parent_node_root=(args.parent_node_root.strip() or None),
    )

    # Unified node UI launcher at the EIDAT root (single GUI for files + projects).
    ui = _render_eidat_bat(runtime_root=runtime_root)
    _write_text(layout.eidat_root / "EIDAT.bat", ui)

    if bool(args.bootstrap_node_ui):
        from .bootstrap_env import main as bootstrap_main

        if str(args.node_ui_venv_dir or "").strip():
            venv_dir = Path(args.node_ui_venv_dir).expanduser()
            if not venv_dir.is_absolute():
                venv_dir = (node_root / venv_dir).resolve()
        else:
            venv_dir = _node_ui_venv_dir(layout)

        sys_py = (args.bootstrap_sys_python or "").strip() or sys.executable
        rc = int(
            bootstrap_main(
                [
                    "--yes",
                    "--profile",
                    "ui",
                    "--node-root",
                    str(node_root),
                    "--venv-dir",
                    str(venv_dir),
                    "--requirements",
                    str(req_lock),
                    "--sys-python",
                    sys_py,
                ]
            )
        )
        if rc != 0:
            raise SystemExit(f"Failed to bootstrap node UI venv (rc={rc}). See logs under: {venv_dir}")

    if bool(args.mirror_node):
        try:
            from .node_mirror import ensure_node_mirror

            mirror_root = (args.mirror_root or "").strip() or None
            ensure_node_mirror(node_root=node_root, runtime_root=runtime_root, mirror_root=mirror_root, mode="auto")
        except Exception:
            # Debug-only feature; never fail deployment on mirror errors.
            pass

    principal = (args.grant_writers or "").strip()
    if principal and platform.system().lower().startswith("win"):
        _grant_writers(layout.runtime_dir, principal)
        _grant_writers(layout.support_projects_dir, principal)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
