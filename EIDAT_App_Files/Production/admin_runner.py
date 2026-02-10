from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineResult:
    ok: bool
    node_root: Path
    outputs: dict[str, object]
    error: str | None = None


def _as_abs(p: str | Path) -> Path:
    path = Path(p).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _eidat_manager_path(runtime_root: Path) -> Path:
    return runtime_root / "EIDAT_App_Files" / "Application" / "eidat_manager.py"


def _base_env(runtime_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    app_root = runtime_root / "EIDAT_App_Files"
    vendored = app_root / "Lib" / "site-packages"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(app_root) + os.pathsep + str(vendored) + os.pathsep + env.get("PYTHONPATH", "")

    # Ensure scanner.env config is honored during admin-driven processing (direct Python invocation).
    # run_admin_dashboard.bat also loads scanner.env, but this keeps behavior consistent when invoked another way.
    try:
        from ui_next.backend import parse_scanner_env  # type: ignore

        scanner_env = runtime_root / "user_inputs" / "scanner.env"
        env.update(parse_scanner_env(scanner_env))
    except Exception:
        pass
    return env


def _node_env_path(node_root: Path) -> Path:
    return node_root / "EIDAT" / "UserData" / ".env"


def _env_truthy(val: object) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    if not s:
        return False
    return s in {"1", "true", "yes", "on", "enable", "enabled"}


def _env_int(val: object, default: int = 0) -> int:
    s = str(val or "").strip()
    if not s:
        return int(default)
    try:
        return int(float(s))
    except Exception:
        return int(default)


def _load_node_config(runtime_root: Path, node_root: Path, *, node_env_enabled: bool) -> dict[str, str]:
    """Load scanner.env + optional node .env into a single config map (node overrides scanner)."""
    try:
        from ui_next.backend import parse_scanner_env  # type: ignore
    except Exception:
        return {}

    cfg: dict[str, str] = {}
    try:
        cfg.update(parse_scanner_env(runtime_root / "user_inputs" / "scanner.env"))
    except Exception:
        pass

    if node_env_enabled:
        try:
            p = _node_env_path(node_root)
            if p.exists():
                cfg.update(parse_scanner_env(p))
        except Exception:
            pass

    return cfg


def _run_manager(
    py: str,
    runtime_root: Path,
    node_root: Path,
    cmd: str,
    extra: list[str] | None = None,
    *,
    node_env_enabled: bool = False,
) -> dict[str, object]:
    script = _eidat_manager_path(runtime_root)
    if not script.exists():
        raise RuntimeError(f"EIDAT Manager not found: {script}")
    argv = [py, str(script), "--global-repo", str(node_root), cmd]
    if extra:
        argv.extend(extra)

    env = _base_env(runtime_root)
    if node_env_enabled:
        try:
            from ui_next.backend import parse_scanner_env  # type: ignore

            path = _node_env_path(node_root)
            if path.exists():
                env.update(parse_scanner_env(path))
        except Exception:
            pass

    proc = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err or f"{cmd} failed with exit code {proc.returncode}")
    out = (proc.stdout or "").strip()
    try:
        payload = json.loads(out) if out else {}
        if not isinstance(payload, dict):
            return {"raw": out}
        return payload
    except Exception:
        return {"raw": out}


def run_pipeline(
    *,
    node_root: str | Path,
    runtime_root: str | Path,
    py: str | None = None,
    force: bool = False,
    limit: int = 0,
    dpi: int = 0,
    similarity: float = 0.86,
    node_env_enabled: bool = False,
) -> PipelineResult:
    node = _as_abs(node_root)
    runtime = _as_abs(runtime_root)
    python_exe = py or sys.executable

    outputs: dict[str, object] = {}
    try:
        if not node.exists():
            raise RuntimeError(f"Node root does not exist: {node}")
        if not node.is_dir():
            raise RuntimeError(f"Node root is not a directory: {node}")
        if not (runtime / "EIDAT_App_Files").exists():
            raise RuntimeError(f"Runtime root does not contain EIDAT_App_Files: {runtime}")
        manager = _eidat_manager_path(runtime)
        if not manager.exists():
            raise RuntimeError(f"EIDAT Manager not found: {manager}")

        outputs["init"] = _run_manager(python_exe, runtime, node, "init", node_env_enabled=node_env_enabled)
        outputs["scan"] = _run_manager(python_exe, runtime, node, "scan", node_env_enabled=node_env_enabled)

        cfg = _load_node_config(runtime, node, node_env_enabled=node_env_enabled)
        # Dashboard inputs are now optional; allow config-driven processing via node .env/scanner.env:
        #   EIDAT_PROCESS_FORCE=1
        #   EIDAT_PROCESS_LIMIT=100
        #   EIDAT_PROCESS_DPI=900
        eff_force = bool(force) or _env_truthy(cfg.get("EIDAT_PROCESS_FORCE"))
        eff_limit = int(limit or 0)
        if eff_limit <= 0:
            eff_limit = _env_int(cfg.get("EIDAT_PROCESS_LIMIT"), 0)
        eff_dpi = int(dpi or 0)
        if eff_dpi <= 0:
            eff_dpi = _env_int(cfg.get("EIDAT_PROCESS_DPI"), 0)

        extra: list[str] = []
        if eff_limit and int(eff_limit) > 0:
            extra += ["--limit", str(int(eff_limit))]
        if eff_dpi and int(eff_dpi) > 0:
            extra += ["--dpi", str(int(eff_dpi))]
        if eff_force:
            extra += ["--force"]
        outputs["process"] = _run_manager(python_exe, runtime, node, "process", extra=extra, node_env_enabled=node_env_enabled)
        outputs["index"] = _run_manager(
            python_exe,
            runtime,
            node,
            "index",
            extra=["--similarity", str(float(similarity))],
            node_env_enabled=node_env_enabled,
        )
        return PipelineResult(ok=True, node_root=node, outputs=outputs)
    except Exception as exc:
        return PipelineResult(ok=False, node_root=node, outputs=outputs, error=str(exc))


def now_ns() -> int:
    return int(time.time_ns())


def run_scan_force_candidates(
    *,
    node_root: str | Path,
    runtime_root: str | Path,
    py: str | None = None,
    similarity: float = 0.86,
    node_env_enabled: bool = False,
) -> PipelineResult:
    """
    Scan for new/changed files, then force-process ONLY those scan candidates (needs_processing=1).

    This overwrites embedded pointer tokens (force) but does not touch already-processed unchanged files.
    """
    node = _as_abs(node_root)
    runtime = _as_abs(runtime_root)
    python_exe = py or sys.executable

    outputs: dict[str, object] = {}
    try:
        if not node.exists():
            raise RuntimeError(f"Node root does not exist: {node}")
        if not node.is_dir():
            raise RuntimeError(f"Node root is not a directory: {node}")
        if not (runtime / "EIDAT_App_Files").exists():
            raise RuntimeError(f"Runtime root does not contain EIDAT_App_Files: {runtime}")

        outputs["init"] = _run_manager(python_exe, runtime, node, "init", node_env_enabled=node_env_enabled)
        scan = _run_manager(python_exe, runtime, node, "scan", node_env_enabled=node_env_enabled)
        outputs["scan"] = scan

        candidates = int((scan or {}).get("candidates_count", 0) or 0)
        if candidates <= 0:
            outputs["process"] = {"skipped": True, "reason": "no_candidates"}
            outputs["index"] = _run_manager(
                python_exe,
                runtime,
                node,
                "index",
                extra=["--similarity", str(float(similarity))],
                node_env_enabled=node_env_enabled,
            )
            return PipelineResult(ok=True, node_root=node, outputs=outputs)

        cfg = _load_node_config(runtime, node, node_env_enabled=node_env_enabled)
        eff_limit = _env_int(cfg.get("EIDAT_PROCESS_LIMIT"), 0)
        eff_dpi = _env_int(cfg.get("EIDAT_PROCESS_DPI"), 0)

        extra: list[str] = ["--force", "--only-candidates"]
        if eff_limit and eff_limit > 0:
            extra += ["--limit", str(int(eff_limit))]
        if eff_dpi and eff_dpi > 0:
            extra += ["--dpi", str(int(eff_dpi))]

        outputs["process"] = _run_manager(
            python_exe,
            runtime,
            node,
            "process",
            extra=extra,
            node_env_enabled=node_env_enabled,
        )
        outputs["index"] = _run_manager(
            python_exe,
            runtime,
            node,
            "index",
            extra=["--similarity", str(float(similarity))],
            node_env_enabled=node_env_enabled,
        )
        return PipelineResult(ok=True, node_root=node, outputs=outputs)
    except Exception as exc:
        return PipelineResult(ok=False, node_root=node, outputs=outputs, error=str(exc))
