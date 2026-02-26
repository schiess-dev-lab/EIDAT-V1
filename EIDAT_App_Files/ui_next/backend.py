from __future__ import annotations

import json
import importlib.util
import math
import os
import sqlite3
import sys
import shutil
import subprocess
import re
from functools import lru_cache
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT.parent  # repository root that holds user data folders


def _get_data_root() -> Path:
    raw = (os.environ.get("EIDAT_DATA_ROOT") or "").strip()
    if not raw:
        return ROOT
    try:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve()
    except Exception:
        return Path(raw).expanduser().absolute()


DATA_ROOT = _get_data_root()

MASTER_DB_ROOT = DATA_ROOT / "Master_Database"
LEGACY_MASTER_ROOT = ROOT  # previous root-based location for master/registry
DEFAULT_TERMS_XLSX = DATA_ROOT / "user_inputs" / "terms.schema.simple.xlsx"
DEFAULT_PLOT_TERMS_XLSX = DATA_ROOT / "user_inputs" / "plot_terms.xlsx"
DEFAULT_PROPOSED_PLOTS_JSON = DATA_ROOT / "user_inputs" / "proposed_plots.json"
DEFAULT_EXCEL_TREND_CONFIG = DATA_ROOT / "user_inputs" / "excel_trend_config.json"
DEFAULT_ACCEPTANCE_HEURISTICS = DATA_ROOT / "user_inputs" / "acceptance_heuristics.json"
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
EXCEL_ARTIFACT_SUFFIX = "__excel"
# Default repository root where PDFs may live (user-organized, nested or flat)
DEFAULT_REPO_ROOT = ROOT / "Data Packages"
DEFAULT_PDF_DIR = DEFAULT_REPO_ROOT
SCANNER_ENV = DATA_ROOT / "user_inputs" / "scanner.env"
SCANNER_ENV_LOCAL = DATA_ROOT / "user_inputs" / "scanner.local.env"
OCR_FORCE_ENV = DATA_ROOT / "user_inputs" / "ocr_force.env"
PROJECT_SCANNER_ENV_NAME = "scanner.project.env"
DOTENV_FILES = [
    DATA_ROOT / ".env",
    DATA_ROOT / "user_inputs" / ".env",
]
EIDAT_MANAGER_ENTRY = APP_ROOT / "Application" / "eidat_manager.py"
RUNS_DIR = DATA_ROOT / "run_data_simple"
PLOTS_DIR = DATA_ROOT / "plots"
TERMS_TEMPLATE_SHEET = "Template"
TERMS_SCHEMA_COLUMNS = [
    "Data Group",
    "Term Label",
    "Term",
    "Header",
    "GroupAfter",
    "GroupBefore",
    "Units",
    "Range (min)",
    "Range (max)",
    "Report Mode",
    "Fuzzy",
]
TERMS_REPORT_CHOICES = ["value", "cell"]
MASTER_XLSX = MASTER_DB_ROOT / "master.xlsx"
MASTER_CSV = MASTER_DB_ROOT / "master.csv"
LEGACY_MASTER_XLSX = LEGACY_MASTER_ROOT / "master.xlsx"
LEGACY_MASTER_CSV = LEGACY_MASTER_ROOT / "master.csv"
REG_XLSX = MASTER_DB_ROOT / "run_registry.xlsx"
REG_CSV = MASTER_DB_ROOT / "run_registry.csv"
LEGACY_REG_XLSX = LEGACY_MASTER_ROOT / "run_registry.xlsx"
LEGACY_REG_CSV = LEGACY_MASTER_ROOT / "run_registry.csv"
MASTER_BASE_COLUMNS = {"Term Label", "Data Group", "Units", "Min", "Max"}
REPO_NAME_FILE = DATA_ROOT / "user_inputs" / "repo_root_name.txt"


def _read_repo_root_name() -> str:
    try:
        if not REPO_NAME_FILE.exists():
            return ""
        for raw in REPO_NAME_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            return line
    except Exception:
        return ""
    return ""


def ensure_repo_root_name_file() -> str:
    name = _read_repo_root_name()
    if not name:
        name = ROOT.name
        try:
            REPO_NAME_FILE.parent.mkdir(parents=True, exist_ok=True)
            REPO_NAME_FILE.write_text(f"{name}\n", encoding="utf-8")
        except Exception:
            pass
    return name


def _validate_repo_root_name() -> None:
    # In production-node mode, EIDAT_DATA_ROOT points to a node-local writable data folder.
    # The runtime folder (ROOT) will intentionally differ, so skip this guard.
    if DATA_ROOT != ROOT:
        return
    expected = ensure_repo_root_name_file()
    if expected and expected.lower() != ROOT.name.lower():
        raise RuntimeError(
            f"Repo root mismatch: expected folder '{expected}' but running under '{ROOT.name}'. "
            f"Update {REPO_NAME_FILE} to match the folder containing EIDAT_App_Files."
        )


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path_res = path.resolve()
    except Exception:
        path_res = path.expanduser().absolute()
    try:
        root_res = root.resolve()
    except Exception:
        root_res = root.expanduser().absolute()
    try:
        path_res.relative_to(root_res)
        return True
    except Exception:
        return False


def is_path_within_repo(path: Path) -> bool:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = DATA_ROOT / p
        return _path_is_within_root(p, DATA_ROOT)
    except Exception:
        return False


def resolve_repo_path(path: Path, label: str = "Path") -> Path:
    _validate_repo_root_name()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = DATA_ROOT / p
    if not _path_is_within_root(p, DATA_ROOT):
        raise RuntimeError(f"{label} must be inside data root: {DATA_ROOT}")
    return p


def resolve_terms_path(path: Optional[Path] = None) -> Path:
    target = Path(path) if path else DEFAULT_TERMS_XLSX
    return resolve_repo_path(target, "Terms schema")


def resolve_pdf_path(path: Path, label: str = "PDF") -> Path:
    _validate_repo_root_name()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    try:
        return p.resolve()
    except Exception:
        return p


def resolve_pdf_root(path: Optional[Path] = None) -> Path:
    target = Path(path) if path else DEFAULT_PDF_DIR
    return resolve_pdf_path(target, "PDF folder")


def resolve_pdf_paths(paths: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for p in paths:
        resolved.append(resolve_pdf_path(Path(p), "PDF"))
    return resolved


def parse_scanner_env(path: Path = SCANNER_ENV) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        if not path.exists():
            return env
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if "#" in v:
                v = v.split("#", 1)[0].strip()
            if ";" in v:
                v = v.split(";", 1)[0].strip()
            if not k:
                continue
            if v == "":
                continue
            env[k] = v
    except Exception:
        pass
    return env


def project_scanner_env_path(project_dir: Path) -> Path:
    return Path(project_dir).expanduser() / PROJECT_SCANNER_ENV_NAME


def load_project_scanner_env(project_dir: Path) -> Dict[str, str]:
    """Load per-project overrides from `<project>/scanner.project.env` (if present)."""
    return parse_scanner_env(project_scanner_env_path(project_dir))


def delete_project_scanner_env(project_dir: Path) -> None:
    """Delete the per-project env override file, if it exists."""
    p = project_scanner_env_path(project_dir)
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def load_scanner_env(project_dir: Optional[Path] = None) -> Dict[str, str]:
    """Load effective scanner config from all supported env files (last wins)."""
    env: Dict[str, str] = {}
    # Precedence (last wins):
    #   1) Legacy user_inputs/ocr_force.env (deprecated; fallback only)
    #   2) user_inputs/scanner.env (shared template/docs)
    #   3) user_inputs/scanner.local.env (machine-specific overrides)
    #   4) repo-local .env overrides (DATA_ROOT/.env, DATA_ROOT/user_inputs/.env)
    #   5) per-project overrides (<project>/scanner.project.env)
    env.update(parse_scanner_env(OCR_FORCE_ENV))
    env.update(parse_scanner_env(SCANNER_ENV))
    env.update(parse_scanner_env(SCANNER_ENV_LOCAL))
    for path in DOTENV_FILES:
        env.update(parse_scanner_env(path))
    if project_dir:
        env.update(load_project_scanner_env(Path(project_dir)))
    # Remove deprecated force flags (no longer used)
    for key in (
        "EIDAT_FORCE_NUMERIC_RESCUE",
        "EIDAT_FORCE_NUMERIC_STRICT",
        "EIDAT_FORCE_CELL_INTERIOR",
    ):
        env.pop(key, None)
    return env


def _env_truthy(val: object) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    if not s:
        return False
    return s in {"1", "true", "yes", "on", "enable", "enabled"}


def save_scanner_env(
    env_map: Dict[str, str],
    path: Path = SCANNER_ENV_LOCAL,
    *,
    order: Optional[list[str]] = None,
    header_lines: Optional[list[str]] = None,
) -> None:
    if order is None:
        order = [
        "QUIET",
        "force_background_processes",
        "REPO_ROOT",
        "EIDAT_TRENDING_COMBINED_ONLY",
        "OCR_MODE",
        "OCR_DPI",
        "FORCE_OCR",
        "XY_LOG",
        "OCR_ROW_EPS",
        "VENV_DIR",
        ]
    if header_lines is None:
        header_lines = [
            "# Scanner configuration (KEY=VALUE)",
            "# Edited via new GUI",
            "# EIDAT_TRENDING_COMBINED_ONLY=1 forces trending extraction to use combined.txt only",
        ]

    # Normalize input (preserve original key case for new lines)
    env_norm: dict[str, tuple[str, str]] = {}
    for k, v in (env_map or {}).items():
        k_raw = str(k or "").strip()
        v_raw = str(v or "").strip()
        if not k_raw or not v_raw:
            continue
        env_norm[k_raw.upper()] = (k_raw, v_raw)

    path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: list[str] = []
    try:
        if path.exists():
            existing_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        existing_lines = []

    if existing_lines:
        seen: set[str] = set()
        new_lines: list[str] = []
        for line in existing_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith(";") or "=" not in line:
                new_lines.append(line)
                continue
            k_raw, _ = line.split("=", 1)
            k_key = k_raw.strip()
            if not k_key:
                new_lines.append(line)
                continue
            k_norm = k_key.upper()
            if k_norm in env_norm:
                new_lines.append(f"{k_key}={env_norm[k_norm][1]}")
                seen.add(k_norm)
            else:
                new_lines.append(line)

        # Append any missing keys in preferred order, then remaining keys.
        for k in order:
            k_norm = k.upper()
            if k_norm in env_norm and k_norm not in seen:
                new_lines.append(f"{k}={env_norm[k_norm][1]}")
                seen.add(k_norm)
        for k_norm, (k_raw, v_raw) in sorted(env_norm.items(), key=lambda kv: kv[0]):
            if k_norm in seen:
                continue
            new_lines.append(f"{k_raw}={v_raw}")
            seen.add(k_norm)

        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return

    # No existing file: create a clean template with defaults ordering.
    lines = list(header_lines or [])
    written = set()
    for k in order:
        k_norm = k.upper()
        if k_norm in env_norm:
            lines.append(f"{k}={env_norm[k_norm][1]}")
            written.add(k_norm)
    for k_norm, (k_raw, v_raw) in sorted(env_norm.items(), key=lambda kv: kv[0]):
        if k_norm in written:
            continue
        lines.append(f"{k_raw}={v_raw}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_project_scanner_env(env_map: Dict[str, str], project_dir: Path) -> Path:
    """
    Save per-project overrides into `<project>/scanner.project.env`.

    This file is intended to hold only overrides that apply to that project (last-wins vs scanner.env).
    """
    proj_dir = Path(project_dir).expanduser()
    p = project_scanner_env_path(proj_dir)
    save_scanner_env(
        env_map,
        path=p,
        order=[
            "EIDAT_TRENDING_COMBINED_ONLY",
            "EIDAT_FUZZY_HEADER_STICK",
            "EIDAT_FUZZY_HEADER_MIN_RATIO",
            "EIDAT_FUZZY_TERM_STICK",
            "EIDAT_FUZZY_TERM_MIN_RATIO",
        ],
        header_lines=[
            "# Project scanner overrides (KEY=VALUE)",
            "# This file overrides user_inputs/scanner.env + user_inputs/scanner.local.env for THIS project only.",
            "# Delete this file to revert to inherited scanner.env values.",
        ],
    )
    return p


def get_repo_root() -> Path:
    """Return repository root from scanner.env or DEFAULT_REPO_ROOT."""
    ensure_repo_root_name_file()
    node_root = (os.environ.get("EIDAT_NODE_ROOT") or "").strip()
    # Repo root is a global setting, not tied to any specific workbook/project.
    env = load_scanner_env()
    val = env.get("REPO_ROOT", "").strip()
    try:
        if val:
            p = Path(val).expanduser()
            if not p.is_absolute():
                p = ROOT / p
            return p
    except Exception:
        pass
    if node_root:
        try:
            return Path(node_root).expanduser().resolve()
        except Exception:
            return Path(node_root).expanduser().absolute()
    return DEFAULT_REPO_ROOT


def set_repo_root(p: Path) -> None:
    _validate_repo_root_name()
    target = Path(p).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    env = parse_scanner_env(SCANNER_ENV_LOCAL)
    try:
        if DATA_ROOT != ROOT:
            env["REPO_ROOT"] = str(target.resolve())
        else:
            rel = target.resolve().relative_to(ROOT.resolve())
            env["REPO_ROOT"] = str(rel)
    except Exception:
        env["REPO_ROOT"] = str(target)
    save_scanner_env(env, path=SCANNER_ENV_LOCAL)


def _run_eidat_manager(global_repo: Path, cmd: str, extra_args: Optional[list[str]] = None) -> Dict[str, object]:
    target = Path(global_repo).expanduser()
    if not target.is_absolute():
        target = (ROOT / target).expanduser()
    if not target.exists():
        raise RuntimeError(f"Global repo does not exist: {target}")
    if not EIDAT_MANAGER_ENTRY.exists():
        raise RuntimeError(f"EIDAT Manager not found: {EIDAT_MANAGER_ENTRY}")
    argv = [sys.executable, str(EIDAT_MANAGER_ENTRY), "--global-repo", str(target), cmd]
    if extra_args:
        argv.extend([str(a) for a in extra_args if str(a).strip()])
    env = _base_env()
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err or f"EIDAT Manager failed with exit code {proc.returncode}")
    out = (proc.stdout or "").strip()
    try:
        payload = json.loads(out) if out else {}
        if not isinstance(payload, dict):
            raise RuntimeError("EIDAT Manager output was not an object")
        if cmd in ("scan", "process"):
            try:
                mirror = mirror_global_debug_ocr_to_local(target)
                payload["debug_ocr_mirror_root"] = mirror.get("dest")
                payload["debug_ocr_mirror_copied"] = mirror.get("copied", 0)
            except Exception:
                pass
        return payload
    except Exception as exc:
        raise RuntimeError(f"Unable to parse EIDAT Manager output as JSON: {exc}") from exc


def eidat_manager_init(global_repo: Path) -> Dict[str, object]:
    return _run_eidat_manager(global_repo, "init")


def eidat_manager_scan(global_repo: Path) -> Dict[str, object]:
    return _run_eidat_manager(global_repo, "scan")


def eidat_manager_process(global_repo: Path, *, limit: int = 0, dpi: int = 0, force: bool = False) -> Dict[str, object]:
    args: list[str] = []
    if limit and int(limit) > 0:
        args.extend(["--limit", str(int(limit))])
    if dpi and int(dpi) > 0:
        args.extend(["--dpi", str(int(dpi))])
    if force:
        args.append("--force")
    result = _run_eidat_manager(global_repo, "process", args)
    # Auto-rebuild index after processing
    if result.get("processed_ok", 0) > 0:
        try:
            index_result = eidat_manager_index(global_repo)
            result["index"] = index_result
        except Exception:
            pass  # Index failure shouldn't fail the whole process
    return result


def eidat_manager_index(global_repo: Path, *, similarity: float = 0.86) -> Dict[str, object]:
    args: list[str] = []
    if similarity:
        args.extend(["--similarity", str(similarity)])
    return _run_eidat_manager(global_repo, "index", args)


def _load_eidat_metadata_module():
    try:
        from Application import eidat_manager_metadata as emd  # type: ignore
        return emd
    except Exception:
        mod_path = APP_ROOT / "Application" / "eidat_manager_metadata.py"
        spec = importlib.util.spec_from_file_location("eidat_manager_metadata", mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load metadata module: {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod


def _merge_metadata(primary: Optional[Mapping[str, object]], fallback: Optional[Mapping[str, object]]) -> dict:
    out: dict = {}
    if isinstance(fallback, Mapping):
        out.update({k: v for k, v in fallback.items()})
    if isinstance(primary, Mapping):
        for k, v in primary.items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            out[k] = v
    return out


def refresh_metadata_only(global_repo: Path, rel_paths: list[str]) -> Dict[str, object]:
    """Refresh metadata JSONs from existing artifacts without re-OCR."""
    repo = Path(global_repo).expanduser()
    if not repo.exists():
        raise RuntimeError(f"Global repo does not exist: {repo}")

    emd = _load_eidat_metadata_module()
    results: list[dict] = []
    updated = 0
    failed = 0
    skipped = 0
    updated_metadata_rels: set[str] = set()

    clean_rel_paths = [str(p).strip() for p in (rel_paths or []) if str(p).strip()]
    for rel_path in clean_rel_paths:
        try:
            abs_path = (repo / rel_path).expanduser()
            if not abs_path.exists():
                raise FileNotFoundError(f"Missing file: {abs_path}")
            artifacts_dir = get_file_artifacts_path(repo, rel_path)
            if not artifacts_dir or not artifacts_dir.exists():
                raise FileNotFoundError(f"Artifacts folder not found for: {rel_path}")

            existing_meta = None
            try:
                existing_meta = emd.load_metadata_from_artifacts(artifacts_dir, abs_path)
            except Exception:
                existing_meta = None
            if not existing_meta:
                try:
                    existing_meta = _load_metadata_from_artifacts_dir(artifacts_dir)
                except Exception:
                    existing_meta = None

            extracted_meta = None
            is_excel = abs_path.suffix.lower() in EXCEL_EXTENSIONS
            if is_excel:
                try:
                    extracted_meta = emd.extract_metadata_from_excel(abs_path)
                except Exception:
                    extracted_meta = None
            else:
                combined_path = artifacts_dir / "combined.txt"
                combined_text = ""
                if combined_path.exists():
                    try:
                        combined_text = combined_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        combined_text = ""
                if combined_text.strip():
                    try:
                        extracted_meta = emd.extract_metadata_from_text(combined_text, pdf_path=abs_path)
                    except Exception:
                        extracted_meta = None

            default_doc_type = "Data file" if is_excel else "EIDP"
            clean_meta = emd.canonicalize_metadata_for_file(
                abs_path,
                existing_meta=existing_meta,
                extracted_meta=extracted_meta,
                default_document_type=default_doc_type,
            )
            metadata_path = emd.write_metadata(Path(artifacts_dir), abs_path, clean_meta)

            # Remove stale metadata files in this artifacts folder (keep the new one if present)
            if metadata_path:
                keep = Path(metadata_path)
                for pat in ("*_metadata.json", "*.metadata.json"):
                    for p in artifacts_dir.glob(pat):
                        try:
                            if p.resolve() == keep.resolve():
                                continue
                            p.unlink()
                        except Exception:
                            continue
                # Track updated metadata rel for project syncing.
                try:
                    rel = str(Path(metadata_path).resolve().relative_to(eidat_support_dir(repo).resolve()))
                    if rel:
                        updated_metadata_rels.add(rel)
                except Exception:
                    pass

            results.append(
                {
                    "rel_path": rel_path,
                    "ok": True,
                    "metadata_path": str(metadata_path) if metadata_path else None,
                }
            )
            updated += 1
        except Exception as exc:
            results.append(
                {
                    "rel_path": rel_path,
                    "ok": False,
                    "error": str(exc),
                }
            )
            failed += 1

    index_result = None
    index_error = None
    try:
        index_result = eidat_manager_index(repo)
    except Exception as exc:
        index_error = str(exc)

    projects_synced: list[dict] = []
    projects_sync_failed: list[dict] = []
    if updated_metadata_rels:
        try:
            projects = list_eidat_projects(repo)
        except Exception:
            projects = []
        for pr in projects:
            try:
                raw_dir = str(pr.get("project_dir") or "").strip()
                proj_dir = Path(raw_dir).expanduser()
                if not proj_dir.is_absolute():
                    proj_dir = repo / proj_dir

                selected = _project_selected_metadata_rels(proj_dir)
                if not selected:
                    continue
                if not (selected.intersection(updated_metadata_rels)):
                    continue

                raw_wb = str(pr.get("workbook") or "").strip()
                wb_path = Path(raw_wb).expanduser()
                if not wb_path.is_absolute():
                    wb_path = repo / wb_path

                res = sync_project_workbook_metadata(repo, wb_path)
                projects_synced.append(
                    {
                        "name": pr.get("name"),
                        "type": pr.get("type"),
                        "workbook": str(wb_path),
                        "result": res,
                    }
                )
            except Exception as exc:
                projects_sync_failed.append(
                    {
                        "name": pr.get("name"),
                        "type": pr.get("type"),
                        "error": str(exc),
                    }
                )

    return {
        "updated": updated,
        "failed": failed,
        "skipped": skipped,
        "results": results,
        "index": index_result,
        "index_error": index_error,
        "updated_metadata_rel": sorted(updated_metadata_rels),
        "projects_synced": projects_synced,
        "projects_sync_failed": projects_sync_failed,
    }


def _prefer_existing(*paths: Path) -> Path:
    """Return the first existing path, or the first element if none exist."""
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def master_xlsx_for_read() -> Path:
    """Prefer Master_Database/master.xlsx, fall back to root-level legacy."""
    return _prefer_existing(MASTER_XLSX, LEGACY_MASTER_XLSX)


def master_csv_for_read() -> Path:
    """Prefer Master_Database/master.csv, fall back to root-level legacy."""
    return _prefer_existing(MASTER_CSV, LEGACY_MASTER_CSV)


def registry_csv_for_read() -> Path:
    """Prefer Master_Database/run_registry.csv, fall back to root-level legacy."""
    return _prefer_existing(REG_CSV, LEGACY_REG_CSV)


def registry_xlsx_for_read() -> Path:
    """Prefer Master_Database/run_registry.xlsx, fall back to root-level legacy."""
    return _prefer_existing(REG_XLSX, LEGACY_REG_XLSX)


def run_roots() -> list[Path]:
    """Return run_data_simple root."""
    return [RUNS_DIR]


def plots_root_for_open() -> Path:
    """Return the plots folder."""
    return PLOTS_DIR


def _venv_python_from(path: Path) -> Path:
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def resolve_project_python() -> str:
    env = load_scanner_env()
    vdir = env.get("VENV_DIR", "").strip()
    if vdir:
        cand = _venv_python_from(Path(vdir))
        if cand.exists():
            return str(cand)
    cand = _venv_python_from(APP_ROOT / ".venv")
    if cand.exists():
        return str(cand)
    return sys.executable


def _base_env() -> Dict[str, str]:
    env = os.environ.copy()
    # Merge env files for direct Python invocations.
    #
    # Precedence (last wins):
    #   1) Legacy user_inputs/ocr_force.env (deprecated; fallback only)
    #   2) user_inputs/scanner.env (shared template/docs)
    #   3) user_inputs/scanner.local.env (machine-specific overrides)
    #   4) project/node .env overrides
    #
    # This lets us "move" OCR DPI settings into scanner.env without ocr_force.env continuing
    # to override them when both exist.
    env.update(parse_scanner_env(OCR_FORCE_ENV))
    env.update(parse_scanner_env(SCANNER_ENV))
    env.update(parse_scanner_env(SCANNER_ENV_LOCAL))
    for path in DOTENV_FILES:
        env.update(parse_scanner_env(path))
    # Remove deprecated force flags (no longer used)
    for key in (
        "EIDAT_FORCE_NUMERIC_RESCUE",
        "EIDAT_FORCE_NUMERIC_STRICT",
        "EIDAT_FORCE_CELL_INTERIOR",
    ):
        env.pop(key, None)
    # Ensure vendored site-packages are importable as fallback
    env["PYTHONPATH"] = str(APP_ROOT / "Lib" / "site-packages") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("QUIET", "1")
    # GUI: charts are experimental and slow; keep disabled unless explicitly enabled.
    env.setdefault("EIDAT_ENABLE_CHART_EXTRACTION", "0")
    # Force cache to always be in project root, not executable location
    # This ensures cache is consistent whether running as script or frozen exe
    if "CACHE_ROOT" not in env and "OCR_CACHE_ROOT" not in env:
        env["CACHE_ROOT"] = str(DATA_ROOT)
    return env


def spawn(cmd: Iterable[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    return subprocess.Popen(
        list(cmd),
        cwd=str(cwd or ROOT),
        env=env or _base_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def run_install_full() -> subprocess.Popen:
    if not sys.platform.startswith("win"):
        raise RuntimeError("install.bat requires Windows")
    return spawn(["cmd.exe", "/c", str(ROOT / "install.bat")])


def run_scanner(terms: Path, pdf_dir: Path) -> subprocess.Popen:
    """Run simple extraction on all PDFs under the provided directory."""
    terms_path = resolve_terms_path(terms)
    pdf_root = resolve_pdf_root(pdf_dir)
    pdfs = [p for p in pdf_root.rglob("*.pdf") if p.is_file()]
    if not pdfs:
        raise RuntimeError(f"No PDFs found under: {pdf_root}")
    return run_simple_extraction(pdfs, terms_path)


def run_script(script_rel_path: str, *args: str) -> subprocess.Popen:
    py = resolve_project_python()
    script = APP_ROOT / script_rel_path
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    return spawn([py, str(script), *args])


def generate_terms() -> subprocess.Popen:
    return run_script("scripts/generate_terms_schema_simple.py")


def compile_master() -> subprocess.Popen:
    return run_script("scripts/compile_master_simple.py")


def compile_master_from_state() -> None:
    """
    Rebuild master.xlsx from simple run outputs.
    This discards all manual edits and rebuilds from extracted data.

    Unlike compile_master(), this runs synchronously in the current process.
    """
    import sys as _sys
    if str(APP_ROOT) not in _sys.path:
        _sys.path.insert(0, str(APP_ROOT))
    from scripts.compile_master_simple import build_master_simple, write_master  # type: ignore

    # Build from simple run results
    serials, rows, prog_map, sv_map, data_map = build_master_simple()

    if not serials:
        print("[WARN] No simple run data available to compile")
        return

    # Write the master workbook
    write_master(serials, rows, program_by_sn=prog_map, sv_by_sn=sv_map, data_by_sn=data_map)


def generate_plot_terms() -> subprocess.Popen:
    return run_script("scripts/generate_plot_terms.py")


def generate_plots() -> subprocess.Popen:
    return run_script("scripts/plot_from_master.py")


def export_plots_summary() -> subprocess.Popen:
    return run_script("scripts/plots_to_excel_summary.py")


def extract_page_tables(pdf: Path, pages: str | None = None) -> subprocess.Popen:
    pdf_path = resolve_pdf_path(Path(pdf), "PDF")
    args = ["--pdf", str(pdf_path)]
    if pages:
        args += ["--pages", pages]
    return run_script("scripts/extract_page_tables.py", *args)


def extract_csv_tables(pdf: Path, pages: str, num_cols: Optional[int] = None,
                      min_cols: int = 2, min_rows: int = 3,
                      match_threshold: float = 0.5, ocr: bool = False,
                      dpi: int = 300, delimiter: Optional[str] = None,
                      output: Optional[Path] = None) -> subprocess.Popen:
    """Extract tables using line-by-line CSV detection with smart alignment."""
    pdf_path = resolve_pdf_path(Path(pdf), "PDF")
    args = ["--pdf", str(pdf_path), "--pages", pages]
    if num_cols is not None:
        args += ["--num-cols", str(num_cols)]
    if min_cols != 2:
        args += ["--min-cols", str(min_cols)]
    if min_rows != 3:
        args += ["--min-rows", str(min_rows)]
    if match_threshold != 0.5:
        args += ["--match-threshold", str(match_threshold)]
    if ocr:
        args += ["--ocr"]
    if dpi != 300:
        args += ["--dpi", str(dpi)]
    if delimiter:
        args += ["--delimiter", delimiter]
    if output:
        out_path = resolve_repo_path(Path(output), "Output path")
        args += ["--out", str(out_path)]
    return run_script("scripts/extract_table_csv_lines.py", *args)


def pre_ocr_merge_pdfs(paths: list[Path], out_root: Optional[Path] = None, dpi: Optional[int] = None) -> subprocess.Popen:
    """Pre-OCR selected PDFs and write merged text artifacts (no extraction)."""
    resolved = resolve_pdf_paths(paths)
    args: list[str] = []
    for p in resolved:
        args += ["--pdf", str(Path(p))]
    if out_root:
        out_path = resolve_repo_path(Path(out_root), "Pre-OCR output folder")
        args += ["--out", str(out_path)]
    if dpi is not None:
        args += ["--dpi", str(int(dpi))]
    return run_script("scripts/pre_ocr_merge.py", *args)


def run_simple_extraction(paths: list[Path], terms: Optional[Path] = None) -> subprocess.Popen:
    """Run the simple merged-text extraction pipeline on selected PDFs."""
    terms_path = resolve_terms_path(terms)
    pdfs = resolve_pdf_paths(paths)
    args: list[str] = []
    for p in pdfs:
        args += ["--pdf", str(Path(p))]
    args += ["--terms", str(terms_path)]
    return run_script("scripts/simple_extraction.py", *args)


def run_excel_extraction(excel_paths: list[Path], config: Optional[Path] = None, global_repo: Optional[Path] = None) -> subprocess.Popen:
    """Run the Excel data extraction pipeline on selected Excel files."""
    cfg = Path(config) if config else DEFAULT_EXCEL_TREND_CONFIG
    if not cfg.exists():
        raise FileNotFoundError(f"Excel trend config not found: {cfg}")
    repo = global_repo or DEFAULT_REPO_ROOT
    args: list[str] = ["--global-repo", str(repo)]
    for p in excel_paths:
        args += ["--excel", str(Path(p))]
    args += ["--config", str(cfg)]
    return run_script("scripts/excel_extraction.py", *args)


def run_excel_scanner(data_dir: Path, config: Optional[Path] = None) -> subprocess.Popen:
    """Run Excel extraction on all Excel files under the provided directory."""
    cfg = Path(config) if config else DEFAULT_EXCEL_TREND_CONFIG
    data_root = resolve_pdf_root(data_dir)
    excels = [
        p
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXCEL_EXTENSIONS and not p.name.startswith("~$")
    ]
    if not excels:
        raise RuntimeError(f"No Excel data files found under: {data_root}")
    return run_excel_extraction(excels, cfg, data_root)


def run_excel_to_sqlite_scanner(data_dir: Path, *, overwrite: bool = True) -> subprocess.Popen:
    """Create per-workbook SQLite outputs for all Excel files under the provided directory."""
    repo = get_repo_root()
    data_root = resolve_pdf_root(data_dir)
    args: list[str] = ["--global-repo", str(repo), "excel_to_sqlite", "--data-dir", str(data_root)]
    if overwrite:
        args.append("--overwrite")
    return run_script("Application/eidat_manager.py", *args)


def open_path(p: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(p))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])


def open_terms_file(path: Optional[Path] = None) -> None:
    tgt = resolve_terms_path(path)
    open_path(tgt)


def derive_return_value(row: Mapping[str, str], existing: Optional[str] = None) -> str:
    """Infer the return type (number|string) when the column is hidden in the UI."""
    smart = (row.get("Smart Snap Type") or "").strip().lower()
    if smart in ("number", "num", "value"):
        return "number"
    if smart in ("date", "time", "title"):
        return "string"
    hint = (existing or row.get("Return") or "").strip().lower()
    if hint in ("number", "string"):
        return hint
    return "number"


def _ensure_openpyxl_loader():
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "openpyxl is required to edit the simple schema spreadsheet. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc
    return load_workbook


def read_terms_rows(path: Optional[Path] = None) -> tuple[list[str], list[dict[str, str]]]:
    """Return (headers, rows) from the simple schema sheet starting at row 2."""
    tgt = resolve_terms_path(path)
    if not tgt.exists():
        raise FileNotFoundError(f"Terms spreadsheet not found: {tgt}")

    # Ensure all required columns exist before reading
    ensure_terms_columns(tgt)

    load_wb = _ensure_openpyxl_loader()
    wb = load_wb(tgt)
    try:
        ws = wb[TERMS_TEMPLATE_SHEET] if TERMS_TEMPLATE_SHEET in wb.sheetnames else wb.active
        if ws is None:
            raise RuntimeError(f"No worksheets found in {tgt}")
        headers: list[str] = []
        for idx, fallback in enumerate(TERMS_SCHEMA_COLUMNS, start=1):
            raw = ws.cell(row=1, column=idx).value
            name = str(raw).strip() if raw not in (None, "") else fallback
            headers.append(name or fallback)
        rows: list[dict[str, str]] = []
        for values in ws.iter_rows(min_row=2, max_col=len(headers), values_only=True):
            normalized: dict[str, str] = {}
            has_value = False
            for col_idx, header in enumerate(headers):
                cell_val = values[col_idx] if col_idx < len(values) else None
                if cell_val is None:
                    text = ""
                else:
                    text = str(cell_val)
                if text.strip():
                    has_value = True
                normalized[header] = text
            if has_value:
                rows.append(normalized)
        if not rows:
            rows.append({h: "" for h in headers})
        return headers, rows
    finally:
        wb.close()


def ensure_terms_columns(path: Optional[Path] = None) -> bool:
    """Ensure the terms spreadsheet has all required columns from TERMS_SCHEMA_COLUMNS.
    Returns True if columns were added, False if no changes needed."""
    tgt = resolve_terms_path(path)
    load_wb = _ensure_openpyxl_loader()
    if not tgt.exists():
        try:
            from openpyxl import Workbook  # type: ignore
        except Exception:
            raise
        wb = Workbook()
        ws = wb.active
        if ws is None:
            ws = wb.create_sheet()
        ws.title = TERMS_TEMPLATE_SHEET
        ws.append(TERMS_SCHEMA_COLUMNS)
        wb.save(tgt)
        wb.close()
        return True
    wb = load_wb(tgt)
    try:
        ws = wb[TERMS_TEMPLATE_SHEET] if TERMS_TEMPLATE_SHEET in wb.sheetnames else wb.active
        if ws is None:
            ws = wb.create_sheet(TERMS_TEMPLATE_SHEET)
        changed = False

        # Read existing headers
        existing_headers: list[str] = []
        max_col = max(ws.max_column or 0, len(TERMS_SCHEMA_COLUMNS))
        for idx in range(1, max_col + 1):
            raw = ws.cell(row=1, column=idx).value
            if raw is not None and str(raw).strip():
                existing_headers.append(str(raw).strip())

        # Check if we need to add any columns
        missing_columns = [col for col in TERMS_SCHEMA_COLUMNS if col not in existing_headers]
        if missing_columns:
            # Add missing column headers to row 1
            start_col = len(existing_headers) + 1
            for idx, col_name in enumerate(missing_columns, start=start_col):
                ws.cell(row=1, column=idx, value=col_name)
            changed = True

        if changed:
            wb.save(tgt)
        return changed
    finally:
        wb.close()


def write_terms_rows(
    rows: list[dict[str, str]],
    path: Optional[Path] = None,
    headers: Optional[list[str]] = None,
) -> None:
    """Persist edited simple-schema rows back into the template sheet (rows 2+)."""
    tgt = resolve_terms_path(path)
    if not tgt.exists():
        raise FileNotFoundError(f"Terms spreadsheet not found: {tgt}")

    # Ensure all required columns exist before writing
    ensure_terms_columns(tgt)

    load_wb = _ensure_openpyxl_loader()
    wb = load_wb(tgt)
    try:
        ws = wb[TERMS_TEMPLATE_SHEET] if TERMS_TEMPLATE_SHEET in wb.sheetnames else wb.active
        if ws is None:
            raise RuntimeError(f"No worksheets found in {tgt}")
        if headers is None:
            headers = []
            for idx, fallback in enumerate(TERMS_SCHEMA_COLUMNS, start=1):
                raw = ws.cell(row=1, column=idx).value
                name = str(raw).strip() if raw not in (None, "") else fallback
                headers.append(name or fallback)
        existing_rows = max(ws.max_row - 1, 0)
        if existing_rows > 0:
            ws.delete_rows(2, existing_rows)
        records = rows or [{h: "" for h in headers}]
        for record in records:
            ws.append([record.get(h, "") for h in headers])
        wb.save(tgt)
    finally:
        wb.close()

def _clean_master_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
    return str(value).strip()

def _read_master_table() -> tuple[list[str], list[dict[str, str]]]:
    header: list[str] = []
    rows: list[dict[str, str]] = []
    master_path = master_xlsx_for_read()
    if not master_path.exists():
        return header, rows

    try:
        import pandas as pd  # type: ignore
        # Preserve textual sentinels such as 'N/A' instead of coercing them
        # to NaN so downstream logic can distinguish between true blanks and
        # explicit "not applicable" markers.
        df = pd.read_excel(master_path, dtype=object, keep_default_na=False)
        header = [str(col) for col in df.columns]
        raw_rows = df.fillna("").to_dict(orient="records")
        for record in raw_rows:
            cleaned = {str(k): _clean_master_cell(v) for k, v in record.items()}
            rows.append(cleaned)
        return header, rows
    except Exception:
        try:
            from openpyxl import load_workbook  # type: ignore
            wb = load_workbook(str(master_path), data_only=True)
            ws = wb.active
            first_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
            header = [(_clean_master_cell(val) or f"column_{idx}") for idx, val in enumerate(first_row)]
            for values in ws.iter_rows(min_row=2, values_only=True):
                record: dict[str, str] = {}
                for idx, val in enumerate(values):
                    if idx >= len(header):
                        continue
                    record[header[idx]] = _clean_master_cell(val)
                rows.append(record)
            wb.close()
            return header, rows
        except Exception:
            pass
    return header, rows

def _schema_value(row: Mapping[str, str], key: str) -> str:
    for variant in (key, key.lower(), key.upper()):
        if variant in row:
            val = row.get(variant)
            if val is None:
                continue
            return str(val).strip()
    return ""

def _compute_missing_term_rows(
    serials: list[str],
    terms_path: Optional[Path] = None,
) -> tuple[list[str], list[dict[str, str]]]:
    target = resolve_terms_path(terms_path)
    headers, schema_rows = read_terms_rows(target)
    normalized_serials = [s.strip() for s in serials if s and s.strip()]
    if not normalized_serials:
        return headers, []
    master_header, master_rows = _read_master_table()
    if not master_header or not master_rows:
        # No master yet: treat all schema rows as missing for the selected serials.
        return headers, list(schema_rows)
    available_serials = {col for col in master_header if col not in MASTER_BASE_COLUMNS}
    # Lookup of rows present in master keyed by (term_label, data_group)
    term_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in master_rows:
        term_label = row.get("Term Label", "").strip()
        if not term_label:
            continue
        if term_label.lower() in ("program", "space vehicle", "data"):
            continue
        data_group = row.get("Data Group", "").strip()
        term_lookup[(term_label.lower(), data_group.lower())] = row
    missing_rows: list[dict[str, str]] = []
    for schema_row in schema_rows:
        term_label = _schema_value(schema_row, "Term Label")
        if not term_label:
            continue
        data_group = _schema_value(schema_row, "Data Group")
        key = (term_label.lower(), data_group.lower())
        master_row = term_lookup.get(key)
        needs_run = False
        for serial in normalized_serials:
            # If the serial column is absent or the row itself doesn't exist in
            # master, this schema term has never been recorded for that EIDP.
            if serial not in available_serials or master_row is None:
                needs_run = True
                break
            value = master_row.get(serial, "")
            text = str(value or "").strip()
            # Blank cells are missing; explicit 'N/A' is treated as already-attempted.
            if not text:
                needs_run = True
                break
            if text.upper() == "N/A":
                # Already attempted; do not treat as missing for this serial.
                continue
        if needs_run:
            missing_rows.append(schema_row)
    return headers, missing_rows

def count_missing_terms(serials: list[str], terms_path: Optional[Path] = None) -> int:
    _, rows = _compute_missing_term_rows(serials, terms_path)
    return len(rows)

def count_missing_terms_per_serial(
    serials: list[str],
    terms_path: Optional[Path] = None,
) -> dict[str, int]:
    """Return mapping {serial: missing_term_count} based on master.xlsx.

    A term is counted as missing for a given serial when:
      - the (Term Label, Data Group) pair doesn't exist in master at all, or
      - the row exists but the cell for that serial is blank.
    Cells containing 'N/A' are treated as already-attempted and not missing.
    """
    target = resolve_terms_path(terms_path)
    _, schema_rows = read_terms_rows(target)
    normalized_serials = [s.strip() for s in serials if s and str(s).strip()]
    if not normalized_serials:
        return {}
    counts: dict[str, int] = {s: 0 for s in normalized_serials}
    master_header, master_rows = _read_master_table()
    # If no master is present yet, treat all schema terms as missing for each serial.
    if not master_header or not master_rows:
        total_terms = 0
        for schema_row in schema_rows:
            label = _schema_value(schema_row, "Term Label")
            if label:
                total_terms += 1
        return {s: total_terms for s in normalized_serials}

    available_serials = {col for col in master_header if col not in MASTER_BASE_COLUMNS}
    # Build lookup from (term_label, data_group) -> master row (existing rows only)
    term_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in master_rows:
        term_label = str(row.get("Term Label", "") or "").strip()
        if not term_label:
            continue
        # Skip metadata rows
        if term_label.lower() in ("program", "space vehicle", "data"):
            continue
        data_group = str(row.get("Data Group", "") or "").strip()
        term_lookup[(term_label.lower(), data_group.lower())] = row

    for schema_row in schema_rows:
        term_label = _schema_value(schema_row, "Term Label")
        if not term_label:
            continue
        data_group = _schema_value(schema_row, "Data Group")
        key = (term_label.lower(), data_group.lower())
        master_row = term_lookup.get(key)
        for serial in normalized_serials:
            # If serial column is missing or row absent, this term is missing.
            if serial not in available_serials or master_row is None:
                counts[serial] = counts.get(serial, 0) + 1
                continue
            text = str(master_row.get(serial, "") or "").strip()
            # Treat explicit N/A as "attempted" (not missing).
            if not text:
                counts[serial] = counts.get(serial, 0) + 1
            elif text.upper() == "N/A":
                # Already attempted; do not treat as missing.
                continue
    return counts

def run_missing_terms_for_selected_pdfs(
    selected: list[tuple[Path, str]],
    terms: Optional[Path] = None,
) -> subprocess.Popen:
    if not selected:
        raise RuntimeError("No data packages selected.")
    terms_path = resolve_terms_path(terms)
    resolved_selected: list[tuple[Path, str]] = []
    for pdf_path, serial in selected:
        resolved_selected.append((resolve_pdf_path(Path(pdf_path), "PDF"), serial))
    serials = sorted({str(serial).strip() for _, serial in resolved_selected if str(serial).strip()})
    if not serials:
        raise RuntimeError("Selected data packages do not include serial identifiers.")
    # Quick pre-check using the master workbook so we can fail fast
    # if every selected EIDP already has values for all schema terms.
    try:
        total_missing = count_missing_terms(serials, terms_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect master workbook for missing terms: {exc}") from exc
    if total_missing <= 0:
        raise RuntimeError("No missing terms found for the selected data packages.")
    # Persist the selection for the batch helper script.
    selection_path = ROOT / "user_inputs" / "missing_terms_selection.json"
    try:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, str]] = []
        for pdf_path, serial in resolved_selected:
            try:
                p = Path(pdf_path)
            except Exception:
                continue
            s = str(serial).strip()
            if not s:
                continue
            payload.append({"pdf": str(p), "serial": s})
        if not payload:
            raise RuntimeError("Selected data packages do not include any valid PDF/serial pairs.")
        selection_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Unable to prepare missing-terms selection: {exc}") from exc
    # Delegate the per-EIDP, missing-terms-only extraction to a small helper script
    # so that the GUI can track a single process while each EIDP is processed with
    # its own tailored term list.
    return run_script(
        "scripts/run_missing_terms_per_eidp.py",
        "--terms",
        str(terms_path),
        "--selection-json",
        str(selection_path),
    )


# --- Workspace sync helpers ---

from datetime import datetime
import csv as _csv


def _parse_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _load_schema_term_keys(terms_path: Path) -> set[tuple[str, str]]:
    """Return set of (term_label, data_group) keys defined in the schema.

    Keys are lowercased to allow case-insensitive comparison. If the schema
    cannot be read, an empty set is returned and sync falls back to registry
    and file timestamps only.
    """
    keys: set[tuple[str, str]] = set()
    try:
        if not terms_path.exists():
            return keys
        suffix = terms_path.suffix.lower()
        # Excel-based schema (preferred)
        if suffix in (".xlsx", ".xlsm", ".xls"):
            try:
                import pandas as _pd  # type: ignore
                try:
                    df = _pd.read_excel(terms_path, sheet_name=TERMS_TEMPLATE_SHEET)
                except Exception:
                    df = _pd.read_excel(terms_path)
                for _, row in df.iterrows():
                    label = str(row.get("Term Label") or "").strip()
                    group = str(row.get("Data Group") or "").strip()
                    if not label and not group:
                        continue
                    keys.add((label.lower(), group.lower()))
                return keys
            except Exception:
                # Fall back to openpyxl if pandas or Excel stack isn't available
                try:
                    import openpyxl as _ox  # type: ignore
                    wb = _ox.load_workbook(str(terms_path), read_only=True, data_only=True)
                    ws = wb[TERMS_TEMPLATE_SHEET] if TERMS_TEMPLATE_SHEET in wb.sheetnames else wb.active
                    rows = list(ws.iter_rows(values_only=True))
                    if not rows:
                        return keys
                    headers = [str(v) if v is not None else "" for v in rows[0]]
                    try:
                        label_idx = headers.index("Term Label")
                    except ValueError:
                        label_idx = None
                    try:
                        group_idx = headers.index("Data Group")
                    except ValueError:
                        group_idx = None
                    if label_idx is None and group_idx is None:
                        return keys
                    for r in rows[1:]:
                        label = ""
                        group = ""
                        if label_idx is not None and label_idx < len(r) and r[label_idx] is not None:
                            label = str(r[label_idx]).strip()
                        if group_idx is not None and group_idx < len(r) and r[group_idx] is not None:
                            group = str(r[group_idx]).strip()
                        if not label and not group:
                            continue
                        keys.add((label.lower(), group.lower()))
                    return keys
                except Exception:
                    return keys
        # CSV schema (or other text-based formats)
        try:
            with terms_path.open("r", encoding="utf-8", newline="") as f:
                r = _csv.DictReader(f)
                for row in r:
                    label = str(
                        row.get("Term Label")
                        or row.get("term_label")
                        or row.get("Term")
                        or row.get("term")
                        or ""
                    ).strip()
                    group = str(row.get("Data Group") or row.get("data_group") or "").strip()
                    if not label and not group:
                        continue
                    keys.add((label.lower(), group.lower()))
        except Exception:
            return keys
    except Exception:
        return set()
    return keys


def _load_run_terms_map(reg_map: dict[str, dict[str, str]]) -> dict[str, set[tuple[str, str]]]:
    """Return mapping {serial_component: {(term_label, data_group), ...}} from scan_results.json.

    Only considers rows for the specific serial in each registry entry. Missing or
    unreadable run folders / results are silently ignored (callers can treat those
    serials as "not yet run" or out-of-sync).
    """
    out: dict[str, set[tuple[str, str]]] = {}
    for serial, info in reg_map.items():
        try:
            run_dir = Path(info.get("run_folder", ""))
        except Exception:
            continue
        if not run_dir or not run_dir.exists():
            continue
        path = run_dir / "scan_results.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        keys: set[tuple[str, str]] = set()
        for row in data:
            if not isinstance(row, dict):
                continue
            row_id = (str(row.get("serial_component") or row.get("serial_number") or "")).strip()
            if row_id and row_id != serial:
                continue
            term_label = str(row.get("term_label") or row.get("term") or "").strip()
            if not term_label:
                continue
            data_group = str(row.get("data_group") or "").strip()
            keys.add((term_label.lower(), data_group.lower()))
        if keys:
            out[serial] = keys
    return out


def _read_run_registry_map() -> dict[str, dict[str, str]]:
    """Return mapping {serial_component: {run_date, run_folder, program_name, vehicle_number}}.

    Reads run_registry.(xlsx|csv) from Master_Database, falling back to
    legacy root locations. Missing file -> empty map.
    """
    rx = registry_xlsx_for_read()
    rc = registry_csv_for_read()
    out: dict[str, dict[str, str]] = {}
    if rc.exists():
        try:
            with rc.open("r", encoding="utf-8", newline="") as f:
                r = _csv.DictReader(f)
                for row in r:
                    sc = (row.get("serial_component") or row.get("serial_number") or "").strip()
                    if not sc:
                        continue
                    out[sc] = {
                        "run_date": (row.get("run_date") or ""),
                        "run_folder": (row.get("run_folder") or ""),
                        "program_name": (row.get("program_name") or ""),
                        "vehicle_number": (row.get("vehicle_number") or ""),
                    }
        except Exception:
            pass
    elif rx.exists():
        # One-time conversion from xlsx to csv, then delete xlsx to avoid confusion
        try:
            import openpyxl as _ox  # type: ignore
            wb = _ox.load_workbook(str(rx), read_only=True, data_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True)) if ws else []
            if rows:
                headers = [str(x) if x is not None else "" for x in rows[0]]
                # DISABLED: Master_Database is no longer used
                # REG_CSV.parent.mkdir(parents=True, exist_ok=True)
                with REG_CSV.open("w", encoding="utf-8", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(headers)
                    for r in rows[1:]:
                        w.writerow([("") if c is None else str(c) for c in r])
            try:
                rx.unlink()
            except Exception:
                pass
            # Recurse to read the new CSV
            return _read_run_registry_map()
        except Exception:
            pass
    return out


def _write_run_registry_map(rows: dict[str, dict[str, str]]) -> None:
    """Write the run registry to CSV only. Remove any legacy XLSX to avoid drift."""
    reg_dir = MASTER_DB_ROOT
    # DISABLED: Master_Database is no longer used
    # reg_dir.mkdir(parents=True, exist_ok=True)
    rx = REG_XLSX
    rc = REG_CSV
    columns = ["serial_component", "program_name", "vehicle_number", "run_date", "run_folder"]
    try:
        with rc.open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for sc in sorted(rows.keys()):
                m = rows[sc]
                w.writerow({
                    "serial_component": sc,
                    "program_name": m.get("program_name", ""),
                    "vehicle_number": m.get("vehicle_number", ""),
                    "run_date": m.get("run_date", ""),
                    "run_folder": m.get("run_folder", ""),
                })
    except Exception:
        pass
    # Remove legacy xlsx to prevent confusion
    try:
        if rx.exists():
            rx.unlink()
        if LEGACY_REG_XLSX.exists():
            LEGACY_REG_XLSX.unlink()
        if LEGACY_REG_CSV.exists() and LEGACY_REG_CSV != rc:
            LEGACY_REG_CSV.unlink()
    except Exception:
        pass


def write_run_registry_rows(rows: list[dict[str, str]]) -> None:
    """Public helper to write a list of row dicts to the run registry.

    Expects keys including at least 'serial_component', 'run_date', 'run_folder'.
    """
    mapping: dict[str, dict[str, str]] = {}
    for r in rows:
        sc = str(r.get("serial_component", "")).strip()
        if not sc:
            continue
        mapping[sc] = {
            "serial_component": sc,
            "program_name": str(r.get("program_name", "")),
            "vehicle_number": str(r.get("vehicle_number", "")),
            "run_date": str(r.get("run_date", "")),
            "run_folder": str(r.get("run_folder", "")),
        }
    _write_run_registry_map(mapping)


def ensure_run_registry_consistent() -> dict[str, dict[str, str]]:
    """Prune invalid entries from run_registry.csv (no additions).

    Keeps rows whose run_folder exists and that contain at least one
    simple run artifact (scan_results_simple.*). Removes others.
    """
    reg = _read_run_registry_map()
    if not reg:
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for sc, info in reg.items():
        try:
            run_dir = _resolve_run_folder(str(info.get("run_folder", "")))
        except Exception:
            run_dir = None
        if not run_dir or not run_dir.exists():
            continue
        has_simple = (run_dir / "scan_results_simple.csv").exists() or (run_dir / "scan_results_simple.xlsx").exists()
        if has_simple:
            cleaned[sc] = info
    _write_run_registry_map(cleaned)
    return cleaned


def delete_registry_entries(serial_components: list[str]) -> None:
    """Delete given serials from run registry (run folders are preserved)."""
    reg = _read_run_registry_map()
    for sc in serial_components:
        info = reg.get(sc)
        run_dir: Optional[Path] = None
        try:
            run_dir = Path(info.get("run_folder")) if info else None
        except Exception:
            run_dir = None
        if run_dir and run_dir.exists():
            try:
                for fp in run_dir.iterdir():
                    try:
                        if fp.is_file() and sc.lower() in fp.name.lower() and fp.suffix.lower() in (".xlsx", ".json", ".csv"):
                            fp.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass
        if sc in reg:
            try:
                del reg[sc]
            except Exception:
                pass
    _write_run_registry_map(reg)


def _derive_identity_from_name(pdf_path: Path) -> tuple[str, str, str]:
    stem = pdf_path.stem
    parts = [p.strip() for p in stem.split("_") if p.strip()]
    program_name = ""
    vehicle_number = ""
    serial_component = ""
    if len(parts) >= 3:
        program_name, vehicle_number = parts[0], parts[1]
        serial_component = "_".join(parts[2:])
    elif len(parts) == 2:
        program_name = parts[0]
        serial_component = parts[1]
    elif parts:
        serial_component = parts[0]
    if not serial_component:
        # Fallback: use full stem
        serial_component = stem
    return program_name, vehicle_number, serial_component


_LAST_SYNC_SUMMARY: dict[str, str | int] | None = None
_LAST_SYNC_DETAILS: list[dict[str, str]] | None = None


def compute_workspace_sync(repo_root: Optional[Path] = None, terms_path: Optional[Path] = None) -> tuple[dict, list[dict]]:
    """Compute workspace sync status.

    Classifies PDFs in repo_root (recursively) as new/out-of-date/up-to-date by
    comparing:
      - run_registry run_date vs PDF modification time, and
      - schema-defined (Term Label, Data Group) pairs vs the values recorded
        in the master workbook for each serial.

    A PDF/serial is considered out-of-sync for "terms" when the current schema
    defines at least one term whose value is still missing (blank) in the
    master workbook for that EIDP. Merely re-saving the schema without adding
    new terms does not mark items as out-of-sync.
    """
    root = resolve_pdf_root(repo_root)
    terms = resolve_terms_path(terms_path)
    # Ensure registry reflects run_data_simple contents before comparing
    reg = ensure_run_registry_consistent()
    t_mtime = datetime.fromtimestamp(terms.stat().st_mtime) if terms.exists() else None
    # Schema term keys (used only as a guard; actual missing-term detection is
    # performed against master.xlsx via count_missing_terms_per_serial).
    schema_keys = _load_schema_term_keys(terms)

    pdfs = [p for p in root.rglob("*.pdf") if p.is_file()]
    details: list[dict[str, str]] = []
    new_count = outdated_pdf = outdated_terms = up_to_date = 0
    # Cache of per-serial missing-term counts so we only evaluate against the
    # master workbook once per serial even if multiple PDFs map to the same
    # EIDP.
    missing_counts_cache: dict[str, int] = {}

    for p in sorted(pdfs):
        try:
            prog, veh, serial = _derive_identity_from_name(p)
            info = reg.get(serial)
            run_dt = _parse_dt(info.get("run_date", "") if info else "")
            pdf_dt = datetime.fromtimestamp(p.stat().st_mtime)

            reason = "up_to_date"
            # No registry entry or unusable run date -> treat as "not yet run"
            if not info or not run_dt:
                reason = "new"
                new_count += 1
            else:
                # Registry knows about this serial; check PDF freshness first
                if pdf_dt > run_dt:
                    reason = "pdf_newer"
                    outdated_pdf += 1
                else:
                    # If we have a readable schema and master workbook, ask the
                    # master whether this EIDP still has missing values for any
                    # schema-defined terms. This lets incremental or "missing
                    # terms only" scans bring EIDPs back to an up-to-date state
                    # without requiring full re-extraction.
                    needs_terms = False
                    if schema_keys:
                        if serial not in missing_counts_cache:
                            try:
                                per_serial = count_missing_terms_per_serial([serial], terms)
                            except Exception:
                                per_serial = {}
                            missing_counts_cache[serial] = int(per_serial.get(serial, 0) or 0)
                        needs_terms = missing_counts_cache.get(serial, 0) > 0
                    if needs_terms:
                        reason = "terms_newer"
                        outdated_terms += 1
                    else:
                        up_to_date += 1

            details.append({
                "pdf": str(p),
                "serial_component": serial,
                "program_name": prog or (info.get("program_name") if info else "") or "",
                "vehicle_number": veh or (info.get("vehicle_number") if info else "") or "",
                "run_date": info.get("run_date") if info else "",
                "pdf_mtime": pdf_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "terms_mtime": t_mtime.strftime("%Y-%m-%d %H:%M:%S") if t_mtime else "",
                "reason": reason,
            })
        except Exception:
            continue
    total = len(pdfs)
    summary = {
        "total": total,
        "new": new_count,
        "pdf_newer": outdated_pdf,
        "terms_newer": outdated_terms,
        "up_to_date": up_to_date,
        "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(root),
    }
    global _LAST_SYNC_SUMMARY, _LAST_SYNC_DETAILS
    _LAST_SYNC_SUMMARY, _LAST_SYNC_DETAILS = summary, details
    return summary, details


def get_last_sync() -> tuple[dict, list[dict]]:
    return _LAST_SYNC_SUMMARY or {}, _LAST_SYNC_DETAILS or []


def run_selected_pdfs(paths: list[Path], terms: Optional[Path] = None) -> subprocess.Popen:
    """Run simple extraction on only the selected PDFs (no staging)."""
    terms = resolve_terms_path(terms)
    pdfs = [p for p in resolve_pdf_paths(paths) if p.is_file()]
    if not pdfs:
        raise RuntimeError("No valid PDF paths were found.")
    return run_simple_extraction(pdfs, terms)


def _gather_serials_from_run(run_dir: Path) -> dict[str, tuple[str, str]]:
    """Return mapping {serial: (program, vehicle)} discovered in a simple run folder."""
    found: dict[str, tuple[str, str]] = {}

    def _record(serial: str, program: str, vehicle: str) -> None:
        serial = (serial or "").strip()
        if not serial or serial in found:
            return
        found[serial] = ((program or "").strip(), (vehicle or "").strip())

    csv_path = run_dir / "scan_results_simple.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                r = _csv.DictReader(f)
                for row in r:
                    serial = str(row.get("serial_component") or row.get("serial_number") or "").strip()
                    if not serial:
                        continue
                    _record(serial, str(row.get("program_name") or ""), str(row.get("vehicle_number") or ""))
        except Exception:
            pass
    if not found:
        xlsx_path = run_dir / "scan_results_simple.xlsx"
        if xlsx_path.exists():
            try:
                from openpyxl import load_workbook  # type: ignore
                wb = load_workbook(str(xlsx_path), data_only=True)
                ws = wb.active
                rows = list(ws.iter_rows(values_only=True)) if ws else []
                if rows:
                    headers = [str(h or "").strip().lower() for h in rows[0]]
                    def _idx(name: str) -> int | None:
                        try:
                            return headers.index(name)
                        except ValueError:
                            return None
                    idx_serial = _idx("serial_component") or _idx("serial_number")
                    idx_program = _idx("program_name")
                    idx_vehicle = _idx("vehicle_number")
                    for values in rows[1:]:
                        if idx_serial is None:
                            break
                        serial = str(values[idx_serial] or "").strip()
                        if not serial:
                            continue
                        program = str(values[idx_program] or "").strip() if idx_program is not None else ""
                        vehicle = str(values[idx_vehicle] or "").strip() if idx_vehicle is not None else ""
                        _record(serial, program, vehicle)
                wb.close()
            except Exception:
                pass
    return found


def rebuild_registry_from_run_data() -> dict[str, dict[str, str]]:
    """Rebuild run_registry.csv by scanning run_data_simple folders."""
    rows: dict[str, dict[str, str]] = {}
    roots = run_roots()
    if not any(r.exists() for r in roots):
        _write_run_registry_map(rows)
        return rows
    try:
        existing = _read_run_registry_map()
    except Exception:
        existing = {}
    for root in roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                run_dt = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
            except Exception:
                run_dt = datetime.fromtimestamp(run_dir.stat().st_mtime)
            display_dt = run_dt.strftime("%Y-%m-%d %H:%M:%S")
            serial_meta = _gather_serials_from_run(run_dir)
            for serial, (program, vehicle) in serial_meta.items():
                prev = rows.get(serial) or existing.get(serial)
                if prev:
                    prev_dt = _parse_dt(prev.get("run_date", ""))
                    if prev_dt and prev_dt >= run_dt:
                        rows[serial] = prev
                        continue
                rel_run = str(run_dir.relative_to(ROOT))
                rows[serial] = {
                    "serial_component": serial,
                    "program_name": program,
                    "vehicle_number": vehicle,
                    "run_date": display_dt,
                    "run_folder": rel_run,
                }
    _write_run_registry_map(rows)
    return rows


def open_last_run_folder() -> None:
    run_dirs = []
    for root in run_roots():
        if root.exists():
            run_dirs.extend([p for p in root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run_data_simple at {RUNS_DIR}")
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime, default=None)
    if not latest:
        raise FileNotFoundError("No run folders found")
    open_path(latest)

def open_run_data_root() -> None:
    # DISABLED: run_data_simple is no longer used
    # RUNS_DIR.mkdir(parents=True, exist_ok=True)
    open_path(RUNS_DIR)


def open_run_registry() -> None:
    reg_xlsx = registry_xlsx_for_read()
    reg_csv = registry_csv_for_read()
    target = reg_xlsx if reg_xlsx.exists() else (reg_csv if reg_csv.exists() else None)
    if not target:
        raise FileNotFoundError("No run registry found (create by running a scan)")
    open_path(target)


# --- Run-data maintenance helpers ---

def _resolve_run_folder(value: str) -> Path:
    """Resolve a run_folder value from registry or cell state to an absolute path.

    - Absolute paths are returned as-is.
    - Bare folder names like ``20250115_103000`` are treated as children of RUNS_DIR
      (falling back to legacy run_data_simple if present).
    - Other relative paths are treated as ROOT-relative (e.g., ``run_data_simple/...``).
    """
    try:
        p = Path(value)
    except Exception:
        return RUNS_DIR
    if p.is_absolute():
        return p
    # Single path component -> interpret as a run_data subfolder name
    if len(p.parts) == 1:
        candidate = RUNS_DIR / p
        if candidate.exists():
            return candidate
        return candidate
    # Otherwise treat as ROOT-relative
    return ROOT / p


def clear_stale_run_data() -> tuple[int, int]:
    """Delete run_data_simple subfolders not referenced by run_registry OR master_cell_state.json.

    Returns (deleted_count, kept_count).
    """
    deleted = 0
    kept = 0

    # Get referenced folders from run_registry
    try:
        reg = _read_run_registry_map()
    except Exception:
        reg = {}
    referenced: set[Path] = set()
    for info in (reg or {}).values():
        try:
            rf = str(info.get("run_folder", ""))
        except Exception:
            rf = ""
        if not rf:
            continue
        try:
            referenced.add(_resolve_run_folder(rf).resolve())
        except Exception:
            pass

    # Also get referenced folders from master_cell_state.json
    try:
        from scripts.master_cell_state import get_referenced_run_folders
        state_folders = get_referenced_run_folders()
        for folder_name in state_folders:
            try:
                referenced.add(_resolve_run_folder(folder_name).resolve())
            except Exception:
                pass
    except Exception:
        # If cell state module isn't available, just use registry references
        pass
    try:
        any_root = False
        for root in run_roots():
            if not root.exists():
                continue
            any_root = True
            for child in root.iterdir():
                try:
                    if not child.is_dir():
                        continue
                    rchild = child.resolve()
                    if rchild in referenced:
                        kept += 1
                        continue
                    # Remove stale folder entirely
                    shutil.rmtree(str(child), ignore_errors=True)
                    deleted += 1
                except Exception:
                    # Ignore individual folder errors
                    pass
        if not any_root:
            return (0, 0)
    except Exception:
        pass
    return (deleted, kept)


def open_master_workbook() -> None:
    xlsx = master_xlsx_for_read()
    if not xlsx.exists():
        raise FileNotFoundError("master.xlsx not found (compile first)")
    open_path(xlsx)


# Deprecated: enrichment now handled during/after runs; external script removed
def enrich_run_registry() -> subprocess.Popen:  # type: ignore[dead-code]
    raise FileNotFoundError("enrich_run_registry is no longer available")


def open_plots_folder() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    open_path(plots_root_for_open())


def open_plots_summary() -> None:
    target_root = plots_root_for_open()
    target = target_root / "plots_summary.xlsx"
    if not target.exists():
        raise FileNotFoundError("No plots_summary.xlsx found (export first)")
    open_path(target)


def ensure_scaffold() -> None:
    (DATA_ROOT / "user_inputs").mkdir(parents=True, exist_ok=True)
    # Do not auto-create Data Packages on startup; only create when user selects/uses it.
    # DISABLED: run_data_simple and Master_Database are no longer used
    # RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # MASTER_DB_ROOT.mkdir(parents=True, exist_ok=True)
    if not SCANNER_ENV.exists():
        # Fallback minimal shared template (normally tracked in repo).
        save_scanner_env({"QUIET": "1"}, path=SCANNER_ENV)
    if not SCANNER_ENV_LOCAL.exists():
        save_scanner_env({"QUIET": "1"}, path=SCANNER_ENV_LOCAL)


# --- Health checks / analysis helpers ---

def check_environment() -> subprocess.Popen:
    """Spawn a short Python check that imports key packages and reports status.

    Returns a process whose stdout can be streamed for UI feedback.
    """
    py = resolve_project_python()
    code = (
        "import sys;\n"
        "print('[INFO] Checking environment for EIDAT...');\n"
        "mods=['fitz','pandas','openpyxl','matplotlib'];\n"
        "ok=True;\n"
        "from importlib import import_module;\n"
        "for m in mods:\n"
        "    try:\n"
        "        import_module(m); print('[OK]', m)\n"
        "    except Exception as e:\n"
        "        ok=False; print('[MISS]', m, type(e).__name__, str(e));\n"
        "print('[DONE] ok='+str(ok))\n"
    )
    return spawn([py, "-c", code])


ID_COLS = ["Term Label", "Data Group"]
Y_AXIS_COL = "Y Axis Label"


def read_plot_terms_table() -> list[dict]:
    """Read plot_terms from XLSX or CSV into list of dict rows.

    Falls back to CSV if Excel stack isn't available.
    """
    xlsx = DEFAULT_PLOT_TERMS_XLSX
    csvp = xlsx.with_suffix(".csv")
    if xlsx.exists():
        try:
            import pandas as pd  # type: ignore
            df = pd.read_excel(xlsx)
            return df.fillna("").to_dict(orient="records")
        except Exception:
            pass
    if csvp.exists():
        try:
            import csv as _csv
            with open(csvp, newline="", encoding="utf-8") as f:
                r = _csv.DictReader(f)
                return [dict(row) for row in r]
        except Exception:
            pass
    return []


def write_plot_terms_table(rows: list[dict]) -> None:
    """Write plot terms table to Excel only (user_inputs/plot_terms.xlsx)."""
    xlsx = DEFAULT_PLOT_TERMS_XLSX
    if not rows:
        rows = [{
            "Plot?": "",
            "Plot Name": "",
            "Tie To Plot": "",
            "X Axis": "SN",
            **{k: "" for k in ID_COLS},
            "Min": "",
            "Max": "",
            Y_AXIS_COL: "",
            "Series Label": "",
        }]
    keys: list[str] = list(rows[0].keys())
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    import pandas as _pd  # type: ignore
    df = _pd.DataFrame(rows, columns=keys)
    # Prefer xlsxwriter, fallback to openpyxl, else fail
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"
    try:
        with _pd.ExcelWriter(xlsx, engine=engine) as writer:  # type: ignore[arg-type]
            df.to_excel(writer, sheet_name="plot_terms", index=False)
        csvp = xlsx.with_suffix(".csv")
        try:
            csvp.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(f"Unable to write plot_terms.xlsx: {e}")


def list_plot_series_options() -> list[dict]:
    """Return catalog of available plot series derived from plot_terms."""
    rows = read_plot_terms_table()
    catalog: list[dict] = []
    for r in rows:
        name = str(r.get("Plot Name") or "").strip()
        if not name:
            continue
        entry = {
            "name": name,
            "term_label": str(r.get("Term Label") or "").strip(),
            "data_group": str(r.get("Data Group") or "").strip(),
            "units": str(r.get("Units") or "").strip(),
            "default_y_axis": str(r.get(Y_AXIS_COL) or "").strip(),
        }
        catalog.append(entry)
    catalog.sort(key=lambda item: item["name"].lower())
    return catalog


def read_proposed_plots() -> list[dict]:
    path = DEFAULT_PROPOSED_PLOTS_JSON
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("plots", [])
    if not isinstance(data, list):
        return []
    cleaned: list[dict] = []
    def _bool(val: object, default: bool = True) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            txt = val.strip().lower()
            if txt in ("1", "true", "yes", "y", "on"):
                return True
            if txt in ("0", "false", "no", "n", "off"):
                return False
        if isinstance(val, (int, float)):
            return val != 0
        return default

    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        y_axis = str(entry.get("y_axis") or "").strip()
        x_axis = str(entry.get("x_axis") or "SN").strip() or "SN"
        series = entry.get("series") or []
        if isinstance(series, str):
            series = [series]
        if not isinstance(series, list):
            series = []
        series_names = [str(s or "").strip() for s in series if str(s or "").strip()]
        cleaned.append({
            "name": name or "Plot",
            "series": series_names,
            "y_axis": y_axis,
            "x_axis": x_axis,
            "include_min": _bool(entry.get("include_min"), True),
            "include_max": _bool(entry.get("include_max"), True),
        })
    return cleaned


def write_proposed_plots(plots: list[dict]) -> None:
    """Persist proposed plot definitions to user_inputs."""
    DEFAULT_PROPOSED_PLOTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict] = []
    for entry in plots:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        y_axis = str(entry.get("y_axis") or "").strip()
        x_axis = str(entry.get("x_axis") or "SN").strip() or "SN"
        series = entry.get("series") or []
        if isinstance(series, str):
            series = [series]
        if not isinstance(series, list):
            series = []
        series_names = [str(s or "").strip() for s in series if str(s or "").strip()]
        payload.append({
            "name": name or "Plot",
            "series": series_names,
            "y_axis": y_axis,
            "x_axis": x_axis,
            "include_min": bool(entry.get("include_min", True)),
            "include_max": bool(entry.get("include_max", True)),
        })
    DEFAULT_PROPOSED_PLOTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- EIDAT Global Repo Projects (v2) ---

EIDAT_PROJECTS_DIRNAME = "projects"
EIDAT_PROJECTS_REGISTRY_JSON = "projects.json"
EIDAT_PROJECTS_REGISTRY_DB = "projects_registry.sqlite3"
EIDAT_PROJECT_META = "project.json"
EIDAT_PROJECT_TYPE_TRENDING = "EIDP Trending"
EIDAT_PROJECT_TYPE_RAW_TRENDING = "EIDP Raw File Trending"
EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING = "Test Data Trending"
EIDAT_PROJECT_IMPLEMENTATION_DB = "implementation_trending.sqlite3"
GLOBAL_RUN_MIRROR_DIRNAME = "global_run_mirror"
LOCAL_PROJECTS_MIRROR_DIRNAME = "projects"
PROJECT_UPDATE_DEBUG_JSON = "update_debug.json"

# Central runtime config used to define Test Data trending workbooks (independent of node-local DATA_ROOT).
CENTRAL_EXCEL_TREND_CONFIG = ROOT / "user_inputs" / "excel_trend_config.json"


def eidat_support_dir(global_repo: Path) -> Path:
    return Path(global_repo).expanduser() / "EIDAT Support"


def eidat_projects_root(global_repo: Path) -> Path:
    return eidat_support_dir(global_repo) / EIDAT_PROJECTS_DIRNAME


def ensure_trending_project_sqlite(project_dir: Path, workbook_path: Path) -> Path:
    """
    Ensure a project-local SQLite cache exists for the EIDP Trending workbook.

    The DB is stored inside the project folder and rebuilt if the workbook is newer.
    """
    proj_dir = Path(project_dir).expanduser()
    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {wb_path}")

    db_path = proj_dir / EIDAT_PROJECT_IMPLEMENTATION_DB
    rebuild = True

    if db_path.exists():
        try:
            db_mtime = db_path.stat().st_mtime
            wb_mtime = wb_path.stat().st_mtime
            rebuild = wb_mtime > db_mtime
        except Exception:
            rebuild = True

        if not rebuild:
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    tables = {
                        r["name"]
                        for r in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    }
                required = {"serials", "terms", "term_values", "meta"}
                if not required.issubset(tables):
                    rebuild = True
            except Exception:
                rebuild = True

    if rebuild:
        _build_trending_project_sqlite(db_path, wb_path)

    return db_path


def _sqlite_drop_table(conn: sqlite3.Connection, table: str) -> None:
    try:
        name = str(table or "").strip()
        if not name:
            return
        safe = name.replace('"', '""')
        conn.execute(f'DROP TABLE IF EXISTS "{safe}"')
    except Exception:
        pass


def load_trending_project_terms(db_path: Path) -> list[dict]:
    """Return available terms from the project SQLite cache."""
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    rows: list[dict] = []
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            fetched = conn.execute(
                """
                SELECT term_key, term, term_label, units, min_val, max_val, data_group, table_label, row_index
                FROM terms
                ORDER BY row_index ASC
                """
            ).fetchall()
        except Exception:
            # Backwards compat: older caches won't have the table_label column.
            fetched = conn.execute(
                """
                SELECT term_key, term, term_label, units, min_val, max_val, data_group, row_index
                FROM terms
                ORDER BY row_index ASC
                """
            ).fetchall()
        for r in fetched:
            rows.append(dict(r))
    return rows


def load_trending_project_serials(db_path: Path) -> list[str]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT serial FROM serials ORDER BY position ASC"
        ).fetchall()
    return [str(r["serial"] or "").strip() for r in rows if str(r["serial"] or "").strip()]


def load_trending_project_series(db_path: Path, term_key: str) -> list[dict]:
    """Return list of dicts with serial + value for a single term_key."""
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT v.serial, v.value_text, v.value_num
            FROM term_values v
            WHERE v.term_key = ?
            ORDER BY v.position ASC
            """,
            (str(term_key),),
        ).fetchall()
    return [dict(r) for r in rows]


def _build_trending_project_sqlite(db_path: Path, workbook_path: Path) -> None:
    """Build a SQLite cache from the EIDP Trending workbook."""
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to read the EIDP Trending workbook. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {wb_path}")

    def _to_float(value: object | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", "")
        try:
            return float(text)
        except Exception:
            pass
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    wb = load_workbook(str(wb_path), read_only=True, data_only=True)
    if "master" not in wb.sheetnames:
        raise RuntimeError("Workbook missing required 'master' sheet.")
    ws = wb["master"]

    rows = ws.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        raise RuntimeError("Workbook master sheet is empty.")

    header = list(header)
    header_norm = [_normalize_text(str(h or "")) for h in header]

    # Serial columns start after the final fixed column (Max).
    serial_start = 9  # legacy default: Term..Max (9 fixed cols)
    try:
        if "max" in header_norm:
            serial_start = int(header_norm.index("max") + 1)
    except Exception:
        serial_start = 9

    # Fixed column indices (by name; Table Label is optional).
    header_map: dict[str, int] = {}
    for idx, val in enumerate(header):
        key = _normalize_text(str(val or ""))
        if not key:
            continue
        header_map[key] = idx
        compact = key.replace(" ", "")
        if compact and compact not in header_map:
            header_map[compact] = idx

    def _h(name: str) -> int | None:
        key = _normalize_text(name)
        if key in header_map:
            return header_map[key]
        compact = key.replace(" ", "")
        return header_map.get(compact)

    idx_term = _h("Term") or 0
    idx_header = _h("Header") or 1
    idx_groupafter = _h("GroupAfter") or 2
    idx_table_label = _h("Table Label")
    idx_valuetype = _h("ValueType") or _h("Value Type") or 3
    idx_datagroup = _h("Data Group") or 4
    idx_termlabel = _h("Term Label") or 5
    idx_units = _h("Units") or 6
    idx_min = _h("Min") or 7
    idx_max = _h("Max") or 8
    serial_cols: list[tuple[int, str]] = []
    for idx in range(int(serial_start), len(header)):
        val = header[idx]
        sn = str(val).strip() if val is not None else ""
        if sn:
            serial_cols.append((idx, sn))
    serials = [sn for _, sn in serial_cols]

    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # IMPORTANT: Do not delete the entire DB file. This project cache DB may also hold
    # Test Data trending caches (td_*). Only drop/recreate the EIDP Trending tables.
    with sqlite3.connect(str(db_path)) as conn:
        for t in ("meta", "serials", "terms", "term_values"):
            _sqlite_drop_table(conn, t)

        conn.execute(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE serials (
                position INTEGER NOT NULL,
                serial TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE terms (
                term_key TEXT PRIMARY KEY,
                term TEXT,
                header TEXT,
                groupafter TEXT,
                valuetype TEXT,
                data_group TEXT,
                term_label TEXT,
                table_label TEXT,
                units TEXT,
                min_val REAL,
                max_val REAL,
                row_index INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE term_values (
                term_key TEXT NOT NULL,
                serial TEXT NOT NULL,
                position INTEGER NOT NULL,
                value_text TEXT,
                value_num REAL
            )
            """
        )
        conn.execute("CREATE INDEX idx_values_term ON term_values(term_key)")
        conn.execute("CREATE INDEX idx_values_serial ON term_values(serial)")

        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("workbook_path", str(wb_path)))
        try:
            wb_mtime = str(wb_path.stat().st_mtime)
        except Exception:
            wb_mtime = ""
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("workbook_mtime", wb_mtime))

        serial_rows = [(idx, sn) for idx, sn in enumerate(serials)]
        conn.executemany("INSERT INTO serials (position, serial) VALUES (?, ?)", serial_rows)

        row_index = 1  # header row is 1 in Excel
        term_rows: list[tuple] = []
        value_rows: list[tuple] = []

        for row in rows:
            row_index += 1
            if not row:
                continue
            row = list(row)
            term = str(row[idx_term] or "").strip() if idx_term < len(row) else ""
            if not term:
                continue
            header_val = str(row[idx_header] or "").strip() if idx_header < len(row) else ""
            group_after = str(row[idx_groupafter] or "").strip() if idx_groupafter < len(row) else ""
            table_label = str(row[idx_table_label] or "").strip() if idx_table_label is not None and idx_table_label < len(row) else ""
            value_type = str(row[idx_valuetype] or "").strip() if idx_valuetype < len(row) else ""
            data_group = str(row[idx_datagroup] or "").strip() if idx_datagroup < len(row) else ""
            term_label = str(row[idx_termlabel] or "").strip() if idx_termlabel < len(row) else ""
            units = str(row[idx_units] or "").strip() if idx_units < len(row) else ""
            min_val = _to_float(row[idx_min] if idx_min < len(row) else None)
            max_val = _to_float(row[idx_max] if idx_max < len(row) else None)

            term_key = f"{term}__row_{row_index}"
            term_rows.append(
                (
                    term_key,
                    term,
                    header_val,
                    group_after,
                    value_type,
                    data_group,
                    term_label,
                    table_label,
                    units,
                    min_val,
                    max_val,
                    row_index,
                )
            )

            for pos, (col_idx, sn) in enumerate(serial_cols):
                if col_idx >= len(row):
                    cell_val = None
                else:
                    cell_val = row[col_idx]
                val_text = "" if cell_val is None else str(cell_val)
                val_num = _to_float(cell_val)
                value_rows.append((term_key, sn, pos, val_text, val_num))

        if term_rows:
            conn.executemany(
                """
                INSERT INTO terms (
                    term_key, term, header, groupafter, valuetype,
                    data_group, term_label, table_label, units, min_val, max_val, row_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                term_rows,
            )
        if value_rows:
            conn.executemany(
                """
                INSERT INTO term_values (
                    term_key, serial, position, value_text, value_num
                ) VALUES (?, ?, ?, ?, ?)
                """,
                value_rows,
            )
        conn.commit()

    try:
        wb.close()
    except Exception:
        pass


def _infer_node_root_from_workbook_path(workbook_path: Path) -> Path:
    """
    Best-effort: if workbook lives under `<node_root>/EIDAT Support/...`, return `<node_root>`.
    Otherwise, return workbook parent.
    """
    p = Path(workbook_path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    cur = p.parent
    for _ in range(12):
        if cur.name.strip().lower() == "eidat support":
            return cur.parent
        if cur == cur.parent:
            break
        cur = cur.parent
    return p.parent


def _resolve_excel_sqlite_path_from_workbook(workbook_path: Path, excel_sqlite_rel: str) -> Path:
    """
    Resolve workbook-relative excel_sqlite_rel values into absolute paths.

    Supports:
    - absolute paths
    - paths starting with `EIDAT Support\\...` (relative to node root)
    - paths starting with `debug\\...` (relative to `<node_root>/EIDAT Support`)
    - other relative paths (relative to workbook folder)
    """
    raw = str(excel_sqlite_rel or "").strip().strip('"')
    if not raw:
        return Path()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p

    node_root = _infer_node_root_from_workbook_path(workbook_path)
    support_dir = node_root / "EIDAT Support"
    norm = raw.replace("/", "\\").lstrip("\\")
    low = norm.lower()

    if low.startswith("eidat support\\"):
        return (node_root / Path(norm)).expanduser()
    if low.startswith("debug\\") or low.startswith("projects\\") or low.startswith("cache\\"):
        return (support_dir / Path(norm)).expanduser()

    return (Path(workbook_path).expanduser().parent / p).expanduser()


def _ensure_test_data_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_sources (
            serial TEXT PRIMARY KEY,
            sqlite_path TEXT,
            mtime_ns INTEGER,
            size_bytes INTEGER,
            status TEXT,
            last_ingested_epoch_ns INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_runs (
            run_name TEXT PRIMARY KEY,
            default_x TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_columns (
            run_name TEXT NOT NULL,
            name TEXT NOT NULL,
            units TEXT,
            kind TEXT NOT NULL,
            PRIMARY KEY (run_name, name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_curves (
            run_name TEXT NOT NULL,
            y_name TEXT NOT NULL,
            x_name TEXT NOT NULL,
            serial TEXT NOT NULL,
            x_json TEXT NOT NULL,
            y_json TEXT NOT NULL,
            n_points INTEGER NOT NULL,
            source_mtime_ns INTEGER,
            computed_epoch_ns INTEGER NOT NULL,
            PRIMARY KEY (run_name, y_name, x_name, serial)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS td_curves_lookup
        ON td_curves (run_name, y_name, x_name, serial)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_metrics (
            serial TEXT NOT NULL,
            run_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            stat TEXT NOT NULL,
            value_num REAL,
            computed_epoch_ns INTEGER NOT NULL,
            source_mtime_ns INTEGER,
            PRIMARY KEY (serial, run_name, column_name, stat)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS td_terms (
            row_index INTEGER PRIMARY KEY,
            term TEXT,
            header TEXT,
            table_label TEXT,
            value_type TEXT,
            data_group TEXT,
            term_label TEXT,
            units TEXT,
            min_val TEXT,
            max_val TEXT,
            active INTEGER NOT NULL,
            normalized_stat TEXT,
            computed_epoch_ns INTEGER NOT NULL
        )
        """
    )


def _read_test_data_config_columns(workbook_path: Path) -> list[dict]:
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to read Test Data Trending workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    wb = load_workbook(str(Path(workbook_path).expanduser()), read_only=True, data_only=True)
    try:
        if "Config" not in wb.sheetnames:
            return []
        ws = wb["Config"]
        header_row_idx: int | None = None
        for r in range(1, (ws.max_row or 0) + 1):
            a = str(ws.cell(r, 1).value or "").strip().lower()
            b = str(ws.cell(r, 2).value or "").strip().lower()
            if a == "name" and b == "units":
                header_row_idx = r
                break
        if header_row_idx is None:
            return []

        cols: list[dict] = []
        for r in range(header_row_idx + 1, (ws.max_row or 0) + 1):
            name = str(ws.cell(r, 1).value or "").strip()
            if not name:
                break
            units = str(ws.cell(r, 2).value or "").strip()
            cols.append({"name": name, "units": units})
        return cols
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _read_test_data_sources(workbook_path: Path) -> list[dict]:
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to read Test Data Trending workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    wb = load_workbook(str(Path(workbook_path).expanduser()), read_only=True, data_only=True)
    try:
        if "Sources" not in wb.sheetnames:
            return []
        ws = wb["Sources"]
        headers: dict[str, int] = {}
        for col in range(1, (ws.max_column or 0) + 1):
            key = str(ws.cell(1, col).value or "").strip().lower()
            if key:
                headers[key] = col
        if "serial_number" not in headers or "excel_sqlite_rel" not in headers:
            return []

        out: list[dict] = []
        for row in range(2, (ws.max_row or 0) + 1):
            sn = str(ws.cell(row, headers["serial_number"]).value or "").strip()
            rel = str(ws.cell(row, headers["excel_sqlite_rel"]).value or "").strip()
            if not sn:
                continue
            out.append({"serial": sn, "excel_sqlite_rel": rel})
        return out
    finally:
        try:
            wb.close()
        except Exception:
            pass


def _normalize_td_stat(s: object) -> str:
    raw = str(s or "").strip().lower()
    if not raw:
        return ""
    aliases = {
        "avg": "mean",
        "average": "mean",
        "stdev": "std",
        "st.dev": "std",
        "st dev": "std",
        "stddev": "std",
        "standard deviation": "std",
    }
    return aliases.get(raw, raw)


def _read_test_data_data_definitions(workbook_path: Path) -> list[dict]:
    """
    Read the user-editable EIDP-style `Data` sheet (Test Data Trending) as metric definitions.

    Mapping:
      - Data Group: run/sheet name
      - Header: column name
      - Table Label: stat (mean/min/max/std/median/count)
    """
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to read Test Data Trending workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    wb = load_workbook(str(Path(workbook_path).expanduser()), read_only=True, data_only=True)
    try:
        if "Data" not in wb.sheetnames:
            return []
        ws = wb["Data"]
        a1 = str(ws.cell(1, 1).value or "").strip().lower()
        if a1 == "metric":
            # Legacy TD workbook layout: old Data sheet is the computed metrics list.
            return []

        headers: dict[str, int] = {}
        for col in range(1, (ws.max_column or 0) + 1):
            key = str(ws.cell(1, col).value or "").strip().lower()
            if key:
                headers[key] = col

        def _h(*names: str) -> int | None:
            for n in names:
                c = headers.get(str(n).strip().lower())
                if c:
                    return int(c)
            return None

        col_term = _h("term")
        col_header = _h("header")
        col_group_after = _h("groupafter")
        col_table_label = _h("table label", "table_label", "tablelabel")
        col_value_type = _h("valuetype", "value type")
        col_data_group = _h("data group", "data_group", "datagroup")
        col_term_label = _h("term label", "term_label", "termlabel")
        col_units = _h("units")
        col_min = _h("min", "range (min)", "range_min")
        col_max = _h("max", "range (max)", "range_max")

        if col_header is None or col_table_label is None or col_data_group is None:
            return []

        allowed = {"mean", "min", "max", "std", "median", "count"}
        out: list[dict] = []
        for row in range(2, (ws.max_row or 0) + 1):
            dg = str(ws.cell(row, int(col_data_group)).value or "").strip()
            hdr = str(ws.cell(row, int(col_header)).value or "").strip()
            tl = str(ws.cell(row, int(col_table_label)).value or "").strip()
            norm = _normalize_td_stat(tl)
            active = bool(dg and hdr and norm in allowed)
            if not (dg or hdr or tl):
                continue

            out.append(
                {
                    "row_index": int(row),
                    "term": str(ws.cell(row, int(col_term)).value or "").strip() if col_term else "",
                    "header": hdr,
                    "group_after": str(ws.cell(row, int(col_group_after)).value or "").strip() if col_group_after else "",
                    "table_label": tl,
                    "value_type": str(ws.cell(row, int(col_value_type)).value or "").strip() if col_value_type else "",
                    "data_group": dg,
                    "term_label": str(ws.cell(row, int(col_term_label)).value or "").strip() if col_term_label else "",
                    "units": str(ws.cell(row, int(col_units)).value or "").strip() if col_units else "",
                    "min": str(ws.cell(row, int(col_min)).value or "").strip() if col_min else "",
                    "max": str(ws.cell(row, int(col_max)).value or "").strip() if col_max else "",
                    "active": bool(active),
                    "normalized_stat": norm if active else "",
                }
            )
        return out
    finally:
        try:
            wb.close()
        except Exception:
            pass


def ensure_test_data_project_cache(
    project_dir: Path,
    workbook_path: Path,
    *,
    rebuild: bool = False,
) -> Path:
    """
    Ensure `td_*` cache tables exist and are up-to-date inside the project's
    `implementation_trending.sqlite3`.
    """
    proj_dir = Path(project_dir).expanduser()
    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {wb_path}")

    db_path = proj_dir / EIDAT_PROJECT_IMPLEMENTATION_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        _ensure_test_data_tables(conn)
        conn.commit()

    if rebuild:
        rebuild_test_data_project_cache(db_path, wb_path)
        return db_path

    # Quick staleness check: compare workbook Sources list + mtimes against td_sources.
    try:
        sources = _read_test_data_sources(wb_path)
        expected = {
            str(s.get("serial") or "").strip(): str(s.get("excel_sqlite_rel") or "").strip()
            for s in sources
            if str(s.get("serial") or "").strip()
        }
    except Exception:
        expected = {}

    with sqlite3.connect(str(db_path)) as conn:
        _ensure_test_data_tables(conn)
        try:
            rows = conn.execute("SELECT serial, sqlite_path, mtime_ns, status FROM td_sources").fetchall()
        except Exception:
            rows = []
        cached = {
            str(r[0] or "").strip(): {"path": str(r[1] or "").strip(), "mtime_ns": r[2], "status": str(r[3] or "").strip()}
            for r in rows
            if str(r[0] or "").strip()
        }
        try:
            curve_count = int(conn.execute("SELECT COUNT(*) FROM td_curves").fetchone()[0] or 0)
        except Exception:
            curve_count = 0

    # If we have no curves at all, treat as stale.
    if curve_count == 0 and expected:
        rebuild_test_data_project_cache(db_path, wb_path)
        return db_path

    # If serial set differs, rebuild.
    if expected and set(expected.keys()) != set(cached.keys()):
        rebuild_test_data_project_cache(db_path, wb_path)
        return db_path

    # If any path/mtime differs, rebuild.
    stale = False
    for sn, rel in expected.items():
        p = _resolve_excel_sqlite_path_from_workbook(wb_path, rel)
        c = cached.get(sn) or {}
        if str(c.get("path") or "") != str(p):
            stale = True
            break
        try:
            st = p.stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        except Exception:
            mtime_ns = None
        if mtime_ns is None:
            if str(c.get("status") or "").lower() == "ok":
                stale = True
                break
        else:
            if int(c.get("mtime_ns") or 0) != int(mtime_ns):
                stale = True
                break
    if stale:
        rebuild_test_data_project_cache(db_path, wb_path)

    return db_path


def rebuild_test_data_project_cache(db_path: Path, workbook_path: Path) -> dict:
    """
    Rebuild `td_*` cache tables inside `implementation_trending.sqlite3` from the
    workbook's `Sources` + `Config` sheets.
    """
    import time
    import statistics

    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {wb_path}")
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_cols = _read_test_data_config_columns(wb_path)
    y_cols: list[tuple[str, str]] = []
    cfg_units: dict[str, str] = {}
    for c in cfg_cols:
        name = str(c.get("name") or "").strip()
        units = str(c.get("units") or "").strip()
        if name:
            y_cols.append((name, units))
            if units:
                cfg_units[name] = units

    data_defs = _read_test_data_data_definitions(wb_path)
    data_y_by_run: dict[str, dict[str, str]] = {}
    for d in data_defs:
        if not bool(d.get("active")):
            continue
        run = str(d.get("data_group") or "").strip()
        name = str(d.get("header") or "").strip()
        units = str(d.get("units") or "").strip()
        if not run or not name:
            continue
        existing = data_y_by_run.setdefault(run, {})
        if name not in existing:
            existing[name] = units
        elif not existing.get(name) and units:
            existing[name] = units

    sources = _read_test_data_sources(wb_path)
    entries = [
        (str(s.get("serial") or "").strip(), str(s.get("excel_sqlite_rel") or "").strip())
        for s in sources
        if str(s.get("serial") or "").strip()
    ]

    computed_epoch_ns = time.time_ns()

    # For TD curve plots, only allow canonical X axes:
    # - Time
    # - Pulse Number
    #
    # Never offer excel_row as an X axis (it is still used for stable ordering when present).
    X_TIME = "Time"
    X_PULSE = "Pulse Number"
    x_priority = [X_TIME, X_PULSE]
    run_x_union: dict[str, set[str]] = {}
    run_default_x: dict[str, str] = {}

    def _norm_name(s: str) -> str:
        return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

    time_candidates = [
        "Time",
        "Time (s)",
        "Time(s)",
        "time",
        "time_s",
        "time_sec",
        "time (s)",
        "time(s)",
    ]
    pulse_candidates = [
        "Pulse Number",
        "Pulse #",
        "Pulse",
        "pulse number",
        "pulse_number",
        "pulsenumber",
        "pulse",
        "cycle",
        "Cycle",
    ]
    time_norms = {_norm_name(x) for x in time_candidates}
    pulse_norms = {_norm_name(x) for x in pulse_candidates}
    x_exclude_norms = time_norms | pulse_norms | {_norm_name("excel_row")}

    def _find_actual_col(cols: set[str], candidates: list[str]) -> str:
        # Prefer exact matches in the given priority order.
        for cand in candidates:
            if cand in cols:
                return cand
        # Fallback: normalized match (handles underscores / spaces / punctuation).
        by_norm: dict[str, str] = {}
        for c in cols:
            n = _norm_name(c)
            if n and n not in by_norm:
                by_norm[n] = c
        for cand in candidates:
            c = by_norm.get(_norm_name(cand))
            if c:
                return c
        return ""

    def _finite_float(v: object) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            f = float(v)
        else:
            t = str(v).strip().replace(",", "")
            if not t:
                return None
            try:
                f = float(t)
            except Exception:
                return None
        if math.isfinite(f):
            return f
        return None

    def _compute_stats(values: list[float]) -> dict[str, float | int | None]:
        n = len(values)
        if n == 0:
            return {"mean": None, "min": None, "max": None, "std": None, "median": None, "count": 0}
        out: dict[str, float | int | None] = {
            "mean": (sum(values) / n),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "count": n,
        }
        out["std"] = statistics.stdev(values) if n >= 2 else None
        return out

    def _quote_ident(name: str) -> str:
        return '"' + (name or "").replace('"', '""') + '"'

    # Reset td_* tables (only).
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_test_data_tables(conn)
        for t in ("td_meta", "td_sources", "td_runs", "td_columns", "td_curves", "td_metrics", "td_terms"):
            try:
                conn.execute(f"DELETE FROM {t}")
            except Exception:
                pass
        conn.execute("INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)", ("workbook_path", str(wb_path)))
        conn.execute("INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)", ("built_epoch_ns", str(computed_epoch_ns)))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)",
                ("node_root", str(_infer_node_root_from_workbook_path(wb_path))),
            )
        except Exception:
            pass

        # Store workbook-defined term rows (Data sheet) for one-to-one comparison.
        for d in data_defs:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_terms
                    (row_index, term, header, table_label, value_type, data_group, term_label, units, min_val, max_val, active, normalized_stat, computed_epoch_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(d.get("row_index") or 0),
                        str(d.get("term") or "").strip(),
                        str(d.get("header") or "").strip(),
                        str(d.get("table_label") or "").strip(),
                        str(d.get("value_type") or "").strip(),
                        str(d.get("data_group") or "").strip(),
                        str(d.get("term_label") or "").strip(),
                        str(d.get("units") or "").strip(),
                        str(d.get("min") or "").strip(),
                        str(d.get("max") or "").strip(),
                        (1 if bool(d.get("active")) else 0),
                        str(d.get("normalized_stat") or "").strip(),
                        int(computed_epoch_ns),
                    ),
                )
            except Exception:
                continue
        conn.commit()

    missing_sources = 0
    invalid_sources = 0
    curves_written = 0
    metrics_written = 0
    run_y_union: dict[str, dict[str, str]] = {}

    for sn, rel in entries:
        sqlite_path = _resolve_excel_sqlite_path_from_workbook(wb_path, rel)
        status = "ok"
        try:
            st = sqlite_path.stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            size_bytes = int(st.st_size)
        except Exception:
            mtime_ns = None
            size_bytes = None
        if not sqlite_path.exists():
            status = "missing"
            missing_sources += 1
        elif not sqlite_path.is_file():
            status = "invalid"
            invalid_sources += 1

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO td_sources(serial, sqlite_path, mtime_ns, size_bytes, status, last_ingested_epoch_ns)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sn, str(sqlite_path), mtime_ns, size_bytes, status, computed_epoch_ns),
            )
            conn.commit()

        if status != "ok":
            continue

        try:
            with sqlite3.connect(str(sqlite_path)) as src:
                # Discover runs
                try:
                    runs_rows = src.execute("SELECT sheet_name FROM __sheet_info ORDER BY sheet_name").fetchall()
                    runs = [str(r[0] or "").strip() for r in runs_rows if str(r[0] or "").strip()]
                except Exception:
                    runs = []
                if not runs:
                    try:
                        tbls = src.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sheet__%' ORDER BY name"
                        ).fetchall()
                        runs = [str(r[0] or "")[7:] for r in tbls if str(r[0] or "").startswith("sheet__")]
                    except Exception:
                        runs = []

                for run in runs:
                    # Resolve table name for the run
                    table = ""
                    try:
                        row = src.execute(
                            "SELECT table_name FROM __sheet_info WHERE sheet_name=? LIMIT 1",
                            (run,),
                        ).fetchone()
                        if row and row[0]:
                            table = str(row[0] or "").strip()
                    except Exception:
                        table = ""
                    if not table:
                        table = f"sheet__{run}"

                    # Columns present
                    try:
                        info = src.execute(f"PRAGMA table_info([{table}])").fetchall()
                        cols = {str(r[1] or "").strip() for r in info if str(r[1] or "").strip()}
                    except Exception:
                        cols = set()
                    if not cols:
                        continue

                    x_map: dict[str, str] = {}
                    actual_time = _find_actual_col(cols, time_candidates)
                    actual_pulse = _find_actual_col(cols, pulse_candidates)
                    if actual_time:
                        x_map[X_TIME] = actual_time
                    if actual_pulse:
                        x_map[X_PULSE] = actual_pulse

                    avail_x = [x for x in x_priority if x in x_map]
                    if avail_x:
                        run_x_union.setdefault(run, set()).update(avail_x)
                        existing = run_default_x.get(run, "")
                        if not existing:
                            run_default_x[run] = avail_x[0]
                        else:
                            # Prefer Time over Pulse Number.
                            if x_priority.index(avail_x[0]) < x_priority.index(existing):
                                run_default_x[run] = avail_x[0]
                    order_by = "excel_row ASC" if "excel_row" in cols else "rowid ASC"

                    # Local default_x for metrics
                    default_x = avail_x[0] if avail_x else ""

                    desired_y: list[str] = []
                    for y_name, _units in y_cols:
                        if y_name:
                            desired_y.append(y_name)
                    extra = data_y_by_run.get(str(run), {}) or {}
                    for y_name in extra.keys():
                        if y_name and y_name not in desired_y:
                            desired_y.append(y_name)

                    run_units = run_y_union.setdefault(str(run), {})
                    for y_name in desired_y:
                        if y_name not in run_units or not run_units.get(y_name):
                            u = cfg_units.get(y_name) or str((data_y_by_run.get(str(run), {}) or {}).get(y_name) or "").strip()
                            run_units[y_name] = str(u or "").strip()

                    for y_name in desired_y:
                        if y_name not in cols:
                            continue
                        if _norm_name(y_name) in x_exclude_norms:
                            continue

                        # Build curves for each available X column (supports x dropdown)
                        for x_name in avail_x:
                            actual_x = x_map.get(x_name, "")
                            if not actual_x:
                                continue
                            q_table = _quote_ident(table)
                            qx = _quote_ident(actual_x)
                            qy = _quote_ident(y_name)
                            try:
                                rows = src.execute(
                                    f"SELECT {qx}, {qy} FROM {q_table} "
                                    f"WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL "
                                    f"ORDER BY {order_by}"
                                ).fetchall()
                            except Exception:
                                rows = []
                            xs: list[float] = []
                            ys: list[float] = []
                            for rx, ry in rows:
                                fx = _finite_float(rx)
                                fy = _finite_float(ry)
                                if fx is None or fy is None:
                                    continue
                                xs.append(float(fx))
                                ys.append(float(fy))

                            with sqlite3.connect(str(db_path)) as conn:
                                conn.execute(
                                    """
                                    INSERT OR REPLACE INTO td_curves
                                    (run_name, y_name, x_name, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        run,
                                        y_name,
                                        x_name,
                                        sn,
                                        json.dumps(xs, separators=(",", ":"), ensure_ascii=False),
                                        json.dumps(ys, separators=(",", ":"), ensure_ascii=False),
                                        int(len(xs)),
                                        mtime_ns,
                                        computed_epoch_ns,
                                    ),
                                )
                                conn.commit()
                            curves_written += 1

                        # Metrics computed from the default X curve (if any)
                        y_for_stats: list[float] = []
                        if default_x:
                            actual_x = x_map.get(default_x, "")
                            if not actual_x:
                                actual_x = default_x
                            q_table = _quote_ident(table)
                            qx = _quote_ident(actual_x)
                            qy = _quote_ident(y_name)
                            try:
                                rows = src.execute(
                                    f"SELECT {qx}, {qy} FROM {q_table} "
                                    f"WHERE {qx} IS NOT NULL AND {qy} IS NOT NULL "
                                    f"ORDER BY {order_by}"
                                ).fetchall()
                            except Exception:
                                rows = []
                            for _rx, ry in rows:
                                fy = _finite_float(ry)
                                if fy is None:
                                    continue
                                y_for_stats.append(float(fy))

                        stats_map = _compute_stats(y_for_stats)
                        with sqlite3.connect(str(db_path)) as conn:
                            for stat, val in stats_map.items():
                                conn.execute(
                                    """
                                    INSERT OR REPLACE INTO td_metrics
                                    (serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (sn, run, y_name, str(stat), val, computed_epoch_ns, mtime_ns),
                                )
                                metrics_written += 1
                            conn.commit()

        except Exception:
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE td_sources SET status=?, last_ingested_epoch_ns=? WHERE serial=?",
                    ("invalid", computed_epoch_ns, sn),
                )
                conn.commit()
            invalid_sources += 1

    # Write runs + columns tables for dropdowns.
    with sqlite3.connect(str(db_path)) as conn:
        runs_all = set(run_x_union.keys()) | set(run_y_union.keys())
        cfg_order = [n for n, _u in y_cols if n]
        for run in sorted(runs_all, key=lambda s: str(s).lower()):
            xs = run_x_union.get(run, set())
            default_x = run_default_x.get(run, "")
            if not default_x:
                for x in x_priority:
                    if x in xs:
                        default_x = x
                        break
            conn.execute(
                "INSERT OR REPLACE INTO td_runs(run_name, default_x) VALUES (?, ?)",
                (run, default_x),
            )
            for x in sorted(xs, key=lambda k: x_priority.index(k) if k in x_priority else 999):
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    (run, x, "", "x"),
                )

            y_map = run_y_union.get(run, {}) or {}
            y_ordered = [n for n in cfg_order if n in y_map] + sorted([n for n in y_map.keys() if n not in cfg_order], key=lambda s: str(s).lower())
            for y_name in y_ordered:
                if _norm_name(y_name) in x_exclude_norms:
                    continue
                units = str(y_map.get(y_name) or "").strip()
                conn.execute(
                    "INSERT OR REPLACE INTO td_columns(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
                    (run, y_name, units, "y"),
                )
        conn.commit()

    return {
        "db_path": str(db_path),
        "workbook": str(wb_path),
        "serials_count": len(entries),
        "missing_sources": missing_sources,
        "invalid_sources": invalid_sources,
        "curves_written": curves_written,
        "metrics_written": metrics_written,
        "runs": sorted(run_x_union.keys()),
    }


def td_list_runs(db_path: Path) -> list[str]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(str(path)) as conn:
        _ensure_test_data_tables(conn)
        rows = conn.execute("SELECT run_name FROM td_runs ORDER BY run_name").fetchall()
    return [str(r[0] or "").strip() for r in rows if str(r[0] or "").strip()]


def td_list_serials(db_path: Path) -> list[str]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    with sqlite3.connect(str(path)) as conn:
        _ensure_test_data_tables(conn)
        rows = conn.execute("SELECT serial FROM td_sources ORDER BY serial").fetchall()
    return [str(r[0] or "").strip() for r in rows if str(r[0] or "").strip()]


def td_list_y_columns(db_path: Path, run_name: str) -> list[dict]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    run = str(run_name or "").strip()
    if not run:
        return []
    with sqlite3.connect(str(path)) as conn:
        _ensure_test_data_tables(conn)
        rows = conn.execute(
            "SELECT name, units FROM td_columns WHERE run_name=? AND kind='y' ORDER BY name",
            (run,),
        ).fetchall()
    return [
        {"name": str(r[0] or "").strip(), "units": str(r[1] or "").strip()}
        for r in rows
        if str(r[0] or "").strip()
    ]


def td_list_x_columns(db_path: Path, run_name: str) -> list[str]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    run = str(run_name or "").strip()
    if not run:
        return []
    with sqlite3.connect(str(path)) as conn:
        _ensure_test_data_tables(conn)
        row = conn.execute("SELECT default_x FROM td_runs WHERE run_name=?", (run,)).fetchone()
        default_x = str(row[0] or "").strip() if row else ""
        rows = conn.execute(
            "SELECT name FROM td_columns WHERE run_name=? AND kind='x' ORDER BY name",
            (run,),
        ).fetchall()
    xs = [str(r[0] or "").strip() for r in rows if str(r[0] or "").strip()]
    if default_x and default_x in xs:
        xs = [default_x] + [x for x in xs if x != default_x]
    return xs


def td_load_metric_series(db_path: Path, run_name: str, column_name: str, stat: str) -> list[dict]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    run = str(run_name or "").strip()
    col = str(column_name or "").strip()
    st = str(stat or "").strip().lower()
    if not run or not col or not st:
        return []
    with sqlite3.connect(str(path)) as conn:
        _ensure_test_data_tables(conn)
        rows = conn.execute(
            """
            SELECT serial, value_num
            FROM td_metrics
            WHERE run_name=? AND column_name=? AND stat=?
            ORDER BY serial
            """,
            (run, col, st),
        ).fetchall()
    return [{"serial": str(r[0] or "").strip(), "value_num": r[1]} for r in rows if str(r[0] or "").strip()]


def td_load_curves(
    db_path: Path,
    run_name: str,
    y_name: str,
    x_name: str,
    serials: list[str] | None = None,
) -> list[dict]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    run = str(run_name or "").strip()
    y = str(y_name or "").strip()
    x = str(x_name or "").strip()
    if not run or not y or not x:
        return []
    want = [str(s or "").strip() for s in (serials or []) if str(s or "").strip()]
    with sqlite3.connect(str(path)) as conn:
        _ensure_test_data_tables(conn)
        if want:
            q = ",".join(["?"] * len(want))
            rows = conn.execute(
                f"""
                SELECT serial, x_json, y_json
                FROM td_curves
                WHERE run_name=? AND y_name=? AND x_name=? AND serial IN ({q})
                ORDER BY serial
                """,
                (run, y, x, *want),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT serial, x_json, y_json
                FROM td_curves
                WHERE run_name=? AND y_name=? AND x_name=?
                ORDER BY serial
                """,
                (run, y, x),
            ).fetchall()
    out: list[dict] = []
    for sn, xj, yj in rows:
        try:
            xs = json.loads(xj or "[]")
            ys = json.loads(yj or "[]")
        except Exception:
            xs, ys = [], []
        out.append({"serial": str(sn or "").strip(), "x": xs, "y": ys})
    return out


def update_test_data_trending_project_workbook(
    global_repo: Path,
    workbook_path: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Populate a Test Data Trending workbook's `Data_calc` (computed) and `Data` (user-defined)
    sheets using cached `td_metrics`.
    """
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to update Test Data Trending workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Project workbook not found: {wb_path}")

    repo = Path(global_repo).expanduser()
    project_dir = wb_path.parent

    try:
        wb = load_workbook(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc

    if "Sources" not in wb.sheetnames:
        raise RuntimeError("Workbook missing required sheet: Sources")

    ws_src = wb["Sources"]
    src_headers: dict[str, int] = {}
    for col in range(1, (ws_src.max_column or 0) + 1):
        key = str(ws_src.cell(1, col).value or "").strip().lower()
        if key:
            src_headers[key] = col

    # Ensure modern Sources schema exists (older workbooks may have only serial_number).
    required_src_cols = [
        "serial_number",
        "program_title",
        "document_type",
        "metadata_rel",
        "artifacts_rel",
        "excel_sqlite_rel",
    ]
    for key in required_src_cols:
        if key in src_headers:
            continue
        col = int((ws_src.max_column or 0) + 1)
        ws_src.cell(1, col).value = key
        src_headers[key] = col

    project_meta: dict = {}
    project_meta_changed = False
    continued_rules: dict[str, set[str]] = {}
    try:
        pj = project_dir / EIDAT_PROJECT_META
        if pj.exists():
            raw = json.loads(pj.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                project_meta = raw
                continued_rules = _extract_continued_population_rules(project_meta)
    except Exception:
        project_meta = {}
        continued_rules = {}
    if "serial_number" not in src_headers:
        raise RuntimeError("Sources sheet missing required column: serial_number")

    serials: list[str] = []
    serials_set: set[str] = set()
    for r in range(2, (ws_src.max_row or 0) + 1):
        sn = str(ws_src.cell(r, src_headers["serial_number"]).value or "").strip()
        if sn and sn not in serials_set:
            serials.append(sn)
            serials_set.add(sn)
    if not serials:
        raise RuntimeError("No serial numbers found in Sources sheet.")

    added_serials: list[str] = []
    if continued_rules and not dry_run:
        try:
            docs = read_eidat_index_documents(repo)
            matched_docs = _docs_matching_population_rules(docs, continued_rules)

            eligible_docs: list[dict] = []
            for d in matched_docs:
                if not isinstance(d, dict) or not is_test_data_doc(d):
                    continue
                sqlite_rel = str(d.get("excel_sqlite_rel") or "").strip()
                if not sqlite_rel:
                    continue
                eligible_docs.append(d)

            docs_by_serial: dict[str, list[dict]] = {}
            for d in eligible_docs:
                sn = str(d.get("serial_number") or "").strip()
                if sn:
                    docs_by_serial.setdefault(sn, []).append(d)

            support_dir = eidat_support_dir(repo)
            for sn in sorted(docs_by_serial.keys()):
                if sn in serials_set:
                    continue
                doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn, [])) or {}
                row = int((ws_src.max_row or 0) + 1)
                ws_src.cell(row, int(src_headers["serial_number"])).value = str(sn)
                ws_src.cell(row, int(src_headers["program_title"])).value = str(doc.get("program_title") or "")
                ws_src.cell(row, int(src_headers["document_type"])).value = str(doc.get("document_type") or "")
                ws_src.cell(row, int(src_headers["metadata_rel"])).value = str(doc.get("metadata_rel") or "")
                ws_src.cell(row, int(src_headers["artifacts_rel"])).value = str(doc.get("artifacts_rel") or "")
                ws_src.cell(row, int(src_headers["excel_sqlite_rel"])).value = str(doc.get("excel_sqlite_rel") or "")
                serials.append(sn)
                serials_set.add(sn)
                added_serials.append(sn)

                if project_meta:
                    try:
                        existing_rels = [
                            str(v).strip()
                            for v in (project_meta.get("selected_metadata_rel") or [])
                            if str(v).strip()
                        ]
                    except Exception:
                        existing_rels = []
                    mr = str(doc.get("metadata_rel") or "").strip()
                    if mr and mr not in existing_rels:
                        existing_rels.append(mr)
                        project_meta["selected_metadata_rel"] = existing_rels
                        project_meta_changed = True
        except Exception:
            pass

    def _config_snapshot() -> dict:
        if "Config" not in wb.sheetnames:
            return {}
        ws_cfg = wb["Config"]
        # Parse columns block: header row contains name/units/range_min/range_max.
        header_row = None
        for r in range(1, (ws_cfg.max_row or 0) + 1):
            a = str(ws_cfg.cell(r, 1).value or "").strip().lower()
            b = str(ws_cfg.cell(r, 2).value or "").strip().lower()
            if a == "name" and b == "units":
                header_row = r
                break
        cols: list[dict] = []
        if header_row is not None:
            def _clean_jsonish(v: object) -> object:
                txt = str(v or "").strip()
                if not txt or txt.lower() in {"null", "none"}:
                    return ""
                try:
                    # Config sheet stores JSON-ish strings like "null" or numbers.
                    return json.loads(txt)
                except Exception:
                    return txt

            for r in range(header_row + 1, (ws_cfg.max_row or 0) + 1):
                name = str(ws_cfg.cell(r, 1).value or "").strip()
                if not name:
                    break
                units = str(ws_cfg.cell(r, 2).value or "").strip()
                rmin = _clean_jsonish(ws_cfg.cell(r, 3).value)
                rmax = _clean_jsonish(ws_cfg.cell(r, 4).value)
                cols.append({"name": name, "units": units, "range_min": rmin, "range_max": rmax})
        # Stats: row where key is "statistics"
        stats: list[str] = []
        for r in range(1, (ws_cfg.max_row or 0) + 1):
            k = str(ws_cfg.cell(r, 1).value or "").strip().lower()
            if k == "statistics":
                raw = str(ws_cfg.cell(r, 2).value or "").strip()
                stats = [s.strip().lower() for s in raw.split(",") if s.strip()]
                break
        return {"columns": cols, "statistics": stats}

    cfg = _config_snapshot()
    cfg_cols = list(cfg.get("columns") or [])
    cfg_stats = [str(s).strip().lower() for s in (cfg.get("statistics") or []) if str(s).strip()]
    if not cfg_stats:
        cfg_stats = ["mean", "min", "max", "std", "median", "count"]

    cfg_by_name: dict[str, dict] = {}
    cfg_order: list[str] = []
    for c in cfg_cols:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        cfg_order.append(name)
        cfg_by_name[name] = c

    # Migration: legacy workbooks used `Data` as the computed metrics sheet ("Metric" in A1).
    if "Data_calc" not in wb.sheetnames and "Data" in wb.sheetnames:
        a1 = str(wb["Data"].cell(1, 1).value or "").strip().lower()
        if a1 == "metric":
            wb["Data"].title = "Data_calc"

    # Ensure computed sheet exists.
    if "Data_calc" not in wb.sheetnames:
        ws_data_calc = wb.create_sheet("Data_calc")
        ws_data_calc.append(["Metric"] + [str(s) for s in serials])
    else:
        ws_data_calc = wb["Data_calc"]

    # Ensure user-editable Data sheet exists (EIDP-style).
    ws_data = None
    if "Data" in wb.sheetnames:
        a1 = str(wb["Data"].cell(1, 1).value or "").strip().lower()
        if a1 != "metric":
            ws_data = wb["Data"]
    if ws_data is None:
        ws_data = wb.create_sheet("Data", 0)
        headers = ["Term", "Header", "GroupAfter", "Table Label", "ValueType", "Data Group", "Term Label", "Units", "Min", "Max"]
        headers.extend([str(s) for s in serials])
        ws_data.append(headers)
        try:
            ws_data.freeze_panes = "A2"
        except Exception:
            pass

        # Seed definitions from Data_calc metric keys (if present).
        seen: set[tuple[str, str, str]] = set()
        for r in range(2, (ws_data_calc.max_row or 0) + 1):
            key = str(ws_data_calc.cell(r, 1).value or "").strip()
            if not key or "." not in key:
                continue
            parts = [p.strip() for p in key.split(".") if p.strip()]
            if len(parts) < 3:
                continue
            run, col, stat = parts[0], parts[1], parts[2].lower()
            norm = _normalize_td_stat(stat)
            if not run or not col or not norm:
                continue
            t = (run, col, norm)
            if t in seen:
                continue
            seen.add(t)
            c = cfg_by_name.get(col) or {}
            units = str(c.get("units") or "").strip()
            rmin = str(c.get("range_min") or "").strip()
            rmax = str(c.get("range_max") or "").strip()
            ws_data.append(
                [
                    col,
                    col,
                    "",
                    norm,
                    "number",
                    run,
                    f"{run}.{col}.{norm}",
                    units,
                    rmin,
                    rmax,
                ]
                + [""] * len(serials)
            )

    # Ensure serial columns exist on both sheets (by matching header cells to serial numbers).
    def _ensure_serial_headers_by_name(ws) -> dict[str, int]:
        serial_cols: dict[str, int] = {}
        for col in range(1, (ws.max_column or 0) + 1):
            sn = str(ws.cell(1, col).value or "").strip()
            if sn and sn in serials:
                serial_cols[sn] = col
        for sn in serials:
            if sn in serial_cols:
                continue
            ws.cell(row=1, column=(ws.max_column or 0) + 1).value = sn
            serial_cols[sn] = int(ws.max_column or 0)
        return serial_cols

    serial_cols_data = _ensure_serial_headers_by_name(ws_data)
    serial_cols_calc = _ensure_serial_headers_by_name(ws_data_calc)
    try:
        ws_data_calc.freeze_panes = "B2"
    except Exception:
        pass

    # Save any migration/creation before rebuilding cache (cache rebuild reads workbook from disk).
    if not dry_run:
        wb.save(str(wb_path))

    db_path = ensure_test_data_project_cache(project_dir, wb_path, rebuild=True)

    def _parse_metric(s: object) -> tuple[str, str, str] | None:
        txt = str(s or "").strip()
        if not txt or "." not in txt:
            return None
        parts = [p.strip() for p in txt.split(".") if p.strip()]
        if len(parts) < 3:
            return None
        return parts[0], parts[1], parts[2].lower()

    with sqlite3.connect(str(db_path)) as conn:
        _ensure_test_data_tables(conn)
        src_missing = int(
            conn.execute(
                "SELECT COUNT(*) FROM td_sources WHERE lower(status) <> 'ok'"
            ).fetchone()[0]
            or 0
        )
        rows = conn.execute(
            """
            SELECT serial, run_name, column_name, stat, value_num
            FROM td_metrics
            """
        ).fetchall()
        y_units_rows = conn.execute(
            "SELECT run_name, name, units FROM td_columns WHERE kind='y'"
        ).fetchall()
        runs_rows = conn.execute("SELECT run_name FROM td_runs ORDER BY run_name").fetchall()

    metric_map: dict[tuple[str, str, str, str], float | int | None] = {}
    for sn, run, col, stat, val in rows:
        metric_map[(str(sn or "").strip(), str(run or "").strip(), str(col or "").strip(), str(stat or "").strip().lower())] = val

    units_map: dict[tuple[str, str], str] = {}
    for run, name, units in y_units_rows:
        k = (str(run or "").strip(), str(name or "").strip())
        u = str(units or "").strip()
        if k[0] and k[1] and u:
            units_map[k] = u

    # Stats list for Data_calc: config stats + any stats referenced in Data definitions.
    stats = list(cfg_stats)
    try:
        defs_now = _read_test_data_data_definitions(wb_path)
    except Exception:
        defs_now = []
    for d in defs_now:
        if not bool(d.get("active")):
            continue
        s = str(d.get("normalized_stat") or "").strip().lower()
        if s and s not in stats:
            stats.append(s)

    # Rebuild Data_calc rows from cache runs + y columns union.
    runs = [str(r[0] or "").strip() for r in runs_rows if str(r[0] or "").strip()]
    y_by_run: dict[str, list[str]] = {}
    for run, name, _units in y_units_rows:
        r = str(run or "").strip()
        n = str(name or "").strip()
        if not r or not n:
            continue
        y_by_run.setdefault(r, []).append(n)

    # Clear and rebuild Data_calc sheet.
    try:
        ws_data_calc.delete_rows(1, ws_data_calc.max_row or 1)
    except Exception:
        # fallback: create a fresh sheet if deletion fails
        try:
            wb.remove(ws_data_calc)
        except Exception:
            pass
        ws_data_calc = wb.create_sheet("Data_calc")
    ws_data_calc.append(["Metric"] + [str(s) for s in serials])
    try:
        ws_data_calc.freeze_panes = "B2"
    except Exception:
        pass

    for run in runs:
        ws_data_calc.append([run] + [""] * len(serials))
        ys = y_by_run.get(run, []) or []
        # Prefer config order first.
        ys_ordered = [n for n in cfg_order if n in ys] + sorted([n for n in ys if n not in cfg_order], key=lambda s: str(s).lower())
        for col in ys_ordered:
            for stat in stats:
                ws_data_calc.append([f"{run}.{col}.{stat}"] + [""] * len(serials))
        ws_data_calc.append([""] + [""] * len(serials))

    updated_cells = 0
    missing_value = 0

    # Populate Data_calc values.
    serial_cols_calc = {str(sn): idx + 2 for idx, sn in enumerate(serials)}
    for r in range(2, (ws_data_calc.max_row or 0) + 1):
        parsed = _parse_metric(ws_data_calc.cell(r, 1).value)
        if parsed is None:
            continue
        run, col, stat = parsed
        for sn, cidx in serial_cols_calc.items():
            key = (sn, run, col, stat)
            val = metric_map.get(key)
            if val is None:
                if not overwrite and ws_data_calc.cell(r, cidx).value not in (None, ""):
                    continue
                ws_data_calc.cell(r, cidx).value = None
                missing_value += 1
                continue
            if not overwrite and ws_data_calc.cell(r, cidx).value not in (None, ""):
                continue
            if stat == "count":
                try:
                    ws_data_calc.cell(r, cidx).value = int(float(val))
                except Exception:
                    ws_data_calc.cell(r, cidx).value = val
            else:
                ws_data_calc.cell(r, cidx).value = float(val)
            updated_cells += 1

    # Populate Data values from (Data Group, Header, Table Label).
    headers: dict[str, int] = {}
    for col in range(1, (ws_data.max_column or 0) + 1):
        k = str(ws_data.cell(1, col).value or "").strip().lower()
        if k:
            headers[k] = col

    def _h(*names: str) -> int | None:
        for n in names:
            c = headers.get(str(n).strip().lower())
            if c:
                return int(c)
        return None

    col_data_group = _h("data group", "data_group", "datagroup")
    col_header = _h("header")
    col_table_label = _h("table label", "table_label", "tablelabel")
    col_units = _h("units")

    if col_data_group and col_header and col_table_label:
        for r in range(2, (ws_data.max_row or 0) + 1):
            run = str(ws_data.cell(r, int(col_data_group)).value or "").strip()
            col = str(ws_data.cell(r, int(col_header)).value or "").strip()
            raw_stat = str(ws_data.cell(r, int(col_table_label)).value or "").strip()
            stat = _normalize_td_stat(raw_stat)
            if not run or not col or not stat:
                continue
            # Fill Units from cache if blank (optional)
            if col_units:
                ucell = ws_data.cell(r, int(col_units))
                if (ucell.value is None or str(ucell.value).strip() == ""):
                    u = units_map.get((run, col))
                    if u:
                        ucell.value = u

            for sn, cidx in serial_cols_data.items():
                key = (sn, run, col, stat)
                val = metric_map.get(key)
                if val is None:
                    if not overwrite and ws_data.cell(r, cidx).value not in (None, ""):
                        continue
                    ws_data.cell(r, cidx).value = None
                    missing_value += 1
                    continue
                if not overwrite and ws_data.cell(r, cidx).value not in (None, ""):
                    continue
                if stat == "count":
                    try:
                        ws_data.cell(r, cidx).value = int(float(val))
                    except Exception:
                        ws_data.cell(r, cidx).value = val
                else:
                    ws_data.cell(r, cidx).value = float(val)
                updated_cells += 1

    # Sync workbook metadata sheets/rows to the canonical index (best-effort).
    try:
        support_dir = eidat_support_dir(repo)
        docs = read_eidat_index_documents(repo)
        selected = _project_selected_metadata_rels(project_dir)
        docs_preferred = [d for d in docs if str(d.get("metadata_rel") or "").strip() in selected] if selected else None
        _sync_project_workbook_metadata_inplace(
            wb,
            support_dir=support_dir,
            docs_all=docs,
            docs_preferred=docs_preferred,
        )
    except Exception:
        pass

    try:
        if not dry_run:
            wb.save(str(wb_path))
    finally:
        try:
            wb.close()
        except Exception:
            pass

    if project_meta and not dry_run:
        try:
            if added_serials:
                project_meta_changed = True
            if project_meta_changed:
                serials_now = sorted({str(s).strip() for s in serials if str(s).strip()})
                project_meta["serials"] = serials_now
                project_meta["serials_count"] = len(serials_now)
                if "selected_metadata_rel" in project_meta:
                    rels_now = sorted(
                        {
                            str(v).strip()
                            for v in (project_meta.get("selected_metadata_rel") or [])
                            if str(v).strip()
                        }
                    )
                    project_meta["selected_metadata_rel"] = rels_now
                    project_meta["selected_count"] = len(rels_now)
                desc = _format_continued_population_description(project_meta.get("continued_population") or {})
                if desc:
                    project_meta["description"] = desc
                (project_dir / EIDAT_PROJECT_META).write_text(json.dumps(project_meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    return {
        "workbook": str(wb_path),
        "db_path": str(db_path),
        "updated_cells": int(updated_cells),
        "missing_source": int(src_missing),
        "missing_value": int(missing_value),
        "serials_in_workbook": int(len(serials)),
        "serials_added": int(len(added_serials)),
        "added_serials": list(added_serials),
        "dry_run": bool(dry_run),
    }


def eidat_debug_ocr_root(global_repo: Path) -> Path:
    return eidat_support_dir(global_repo) / "debug" / "ocr"


def global_run_mirror_root() -> Path:
    """Writable root for mirroring Global Repo outputs (node-local when EIDAT_DATA_ROOT is set)."""
    return DATA_ROOT / GLOBAL_RUN_MIRROR_DIRNAME


def local_projects_mirror_root() -> Path:
    """Repo-local mirror for debugging projects without needing the Global Repo."""
    return global_run_mirror_root() / LOCAL_PROJECTS_MIRROR_DIRNAME


def local_debug_ocr_mirror_root() -> Path:
    """Repo-local mirror for Global Repo debug OCR artifacts."""
    return global_run_mirror_root() / "debug" / "ocr"


def _mirror_tree_incremental(src: Path, dest: Path) -> int:
    """Copy new/changed files from src to dest, preserving relative paths."""
    if not src.exists():
        return 0
    copied = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(src)
        except Exception:
            continue
        out = dest / rel
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                s_stat = p.stat()
                d_stat = out.stat()
                if s_stat.st_size == d_stat.st_size and int(s_stat.st_mtime) <= int(d_stat.st_mtime):
                    continue
            shutil.copy2(p, out)
            copied += 1
        except Exception:
            continue
    return copied


def mirror_global_debug_ocr_to_local(global_repo: Path) -> dict[str, object]:
    src = eidat_debug_ocr_root(global_repo)
    dest = local_debug_ocr_mirror_root()
    copied = _mirror_tree_incremental(src, dest)
    return {"src": str(src), "dest": str(dest), "copied": copied}


def _path_is_within(path: Path, root: Path) -> bool:
    return _path_is_within_root(path, root)


def resolve_path_within_global_repo(global_repo: Path, path: Path, label: str = "Path") -> Path:
    repo = Path(global_repo).expanduser()
    if not repo.is_absolute():
        repo = repo.absolute()
    if not repo.exists():
        raise RuntimeError(f"Global repo does not exist: {repo}")

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = repo / p
    if not _path_is_within(p, repo):
        raise RuntimeError(f"{label} must be inside Global Repo: {repo}")
    return p


def read_eidat_index_documents(global_repo: Path) -> list[dict]:
    """Return indexed document rows from `EIDAT Support/eidat_index.sqlite3`."""
    repo = Path(global_repo).expanduser()
    db_path = eidat_support_dir(repo) / "eidat_index.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"EIDAT index DB not found: {db_path}")
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        select_cols: list[str] = [
            "id",
            "program_title",
            "asset_type",
            "serial_number",
            "part_number",
            "revision",
            "test_date",
            "report_date",
            "document_type",
        ]
        for opt in ("document_type_acronym", "vendor", "acceptance_test_plan_number", "excel_sqlite_rel", "file_extension"):
            if opt in cols:
                select_cols.append(opt)
        select_cols += ["metadata_rel", "artifacts_rel", "similarity_group"]
        rows = conn.execute(
            f"""
            SELECT {", ".join(select_cols)}
            FROM documents
            ORDER BY program_title, asset_type, serial_number, metadata_rel
            """
        ).fetchall()
    docs: list[dict] = []
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        d.setdefault("document_type_acronym", None)
        d.setdefault("vendor", None)
        d.setdefault("acceptance_test_plan_number", None)
        d.setdefault("excel_sqlite_rel", None)
        d.setdefault("file_extension", None)
        docs.append(d)
    return docs


def read_eidat_index_groups(global_repo: Path) -> list[dict]:
    repo = Path(global_repo).expanduser()
    db_path = eidat_support_dir(repo) / "eidat_index.sqlite3"
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT group_id, title_norm, member_count FROM groups ORDER BY member_count DESC, group_id"
        ).fetchall()
    return [{k: r[k] for k in r.keys()} for r in rows]


def read_eidat_support_files(global_repo: Path) -> list[dict]:
    """Return all tracked files from eidat_support.sqlite3/files table."""
    repo = Path(global_repo).expanduser()
    db_path = eidat_support_dir(repo) / "eidat_support.sqlite3"
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, rel_path, file_fingerprint, content_sha1, eidat_uuid,
                   pointer_token, size_bytes, mtime_ns, first_seen_epoch_ns,
                   last_seen_epoch_ns, last_processed_epoch_ns, needs_processing
            FROM files
            ORDER BY rel_path
            """
        ).fetchall()
    return [{k: r[k] for k in r.keys()} for r in rows]


def read_files_with_index_metadata(global_repo: Path) -> list[dict]:
    """Join files from support DB with documents from index DB for unified view."""
    repo = Path(global_repo).expanduser()
    support_db = eidat_support_dir(repo) / "eidat_support.sqlite3"
    index_db = eidat_support_dir(repo) / "eidat_index.sqlite3"
    support_dir = eidat_support_dir(repo)

    def _ignore_rel_path(rel_path: str) -> bool:
        # Hide generated PDFs/Excels from the Files tab. These are outputs, not
        # original documents.
        try:
            parts = [p.casefold() for p in Path(str(rel_path or "")).parts]
        except Exception:
            parts = []
        ignored = {"eidat", "eidat support", "edat", "edat support"}
        return any(p in ignored for p in parts)

    # Read all files
    files_by_path: dict[str, dict] = {}
    if support_db.exists():
        with sqlite3.connect(str(support_db)) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT * FROM files").fetchall():
                rel = str(r["rel_path"] or "")
                if _ignore_rel_path(rel):
                    continue
                files_by_path[rel] = dict(r)

    # Read all indexed documents
    docs_by_artifacts: dict[str, dict] = {}
    if index_db.exists():
        with sqlite3.connect(str(index_db)) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT * FROM documents").fetchall():
                artifacts = r["artifacts_rel"] or ""
                docs_by_artifacts[artifacts] = dict(r)

    # Join: for each file, try to find its document metadata
    result: list[dict] = []
    for rel_path, file_info in files_by_path.items():
        # Match file to document by exact artifacts folder (avoids stem substring collisions).
        matched_doc = None
        try:
            art_dir = get_file_artifacts_path(repo, rel_path)
        except Exception:
            art_dir = None
        if art_dir:
            try:
                art_rel = str(Path(art_dir).resolve().relative_to(support_dir.resolve()))
            except Exception:
                try:
                    art_rel = str(Path(art_dir).relative_to(support_dir))
                except Exception:
                    art_rel = str(art_dir)
            matched_doc = docs_by_artifacts.get(art_rel)

        merged = {**file_info}
        if matched_doc:
            merged.update({
                "program_title": matched_doc.get("program_title"),
                "asset_type": matched_doc.get("asset_type"),
                "serial_number": matched_doc.get("serial_number"),
                "part_number": matched_doc.get("part_number"),
                "revision": matched_doc.get("revision"),
                "test_date": matched_doc.get("test_date"),
                "report_date": matched_doc.get("report_date"),
                "document_type": matched_doc.get("document_type"),
                "metadata_rel": matched_doc.get("metadata_rel"),
                "artifacts_rel": matched_doc.get("artifacts_rel"),
                "similarity_group": matched_doc.get("similarity_group"),
                "certification_status": matched_doc.get("certification_status"),
                "certification_pass_rate": matched_doc.get("certification_pass_rate"),
            })
        result.append(merged)

    return result


def get_file_artifacts_path(global_repo: Path, rel_path: str) -> Path | None:
    """Return full path to artifacts folder for a file."""
    repo = Path(global_repo).expanduser()
    path_obj = Path(rel_path)
    stem = path_obj.stem
    ext = path_obj.suffix.lower()
    root = eidat_debug_ocr_root(repo)
    candidates: list[Path] = []
    if ext in EXCEL_EXTENSIONS:
        candidates.append(root / f"{stem}{EXCEL_ARTIFACT_SUFFIX}")
    candidates.append(root / stem)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def _normalize_key(s: str) -> str:
    """Lowercase alnum-only key for matching terms/labels across OCR noise."""
    return re.sub(r"[^a-z0-9]+", "", _normalize_text(s))


def _token_overlap_score(a: str, b: str) -> float:
    """
    Return how well `b` covers the tokens in `a` (0..1).

    This is intentionally *asymmetric*: it answers "what fraction of the query tokens
    appear in the candidate?", which avoids over-scoring short/generic candidates
    (e.g., "upstream valve") against longer terms.

    Supports abbreviation-style prefix matches for longer tokens (>=4 chars):
      - "prop" ~= "propellant"
      - "press" ~= "pressure"
    """

    a_toks = [t for t in re.findall(r"[a-z0-9]+", _normalize_text(a)) if t]
    b_toks = [t for t in re.findall(r"[a-z0-9]+", _normalize_text(b)) if t]
    if not a_toks or not b_toks:
        return 0.0

    a_unique: list[str] = list(dict.fromkeys(a_toks))
    b_unique: list[str] = list(dict.fromkeys(b_toks))

    def _tok_match(q: str, c: str) -> bool:
        if q == c:
            return True
        if len(q) >= 4 and len(c) >= 4 and (q.startswith(c) or c.startswith(q)):
            return True
        return False

    matched = 0
    for qt in a_unique:
        for ct in b_unique:
            if _tok_match(qt, ct):
                matched += 1
                break
    return matched / float(len(a_unique)) if a_unique else 0.0


def _fuzzy_term_score(term: str, term_label: str, candidate: str) -> float:
    """
    Score how well `candidate` matches a requested term/label (0..1).

    Workbooks often store canonical IDs in `Term`, but documents use human
    labels in tables; when available, term_label should carry matching weight.
    """
    cand_s = str(candidate or "").strip()
    if not cand_s:
        return 0.0

    term_k = _normalize_key(term)
    label_k = _normalize_key(term_label)
    cand_k = _normalize_key(cand_s)
    cand_n = _normalize_text(cand_s)

    if term_k and cand_k and term_k == cand_k:
        return 1.0
    if label_k and cand_k and label_k == cand_k:
        return 1.0

    def _score_query(q_raw: str, q_key: str) -> float:
        if not q_raw:
            return 0.0
        seq = SequenceMatcher(None, q_key, cand_k).ratio() if (q_key and cand_k) else 0.0
        cov = _token_overlap_score(q_raw, cand_n)
        # Do not let a high SequenceMatcher score override low token coverage.
        return max(cov, (seq + cov) / 2.0)

    best = 0.0
    if term_label:
        best = max(best, _score_query(term_label, label_k))
    if term:
        best = max(best, _score_query(term, term_k))
    return float(best)


def _score_term_candidate(
    term: str,
    term_label: str,
    candidate_text: str,
    *,
    fuzzy_term: bool,
    min_ratio: float,
) -> float:
    """
    Score a candidate label/description against a requested term (0..1).

    Perfect match (1.0) is reserved for exact normalized-key equality.
    Everything else ranks below 1.0 so exact hits always win deterministically.
    """
    cand_s = str(candidate_text or "").strip()
    if not cand_s:
        return 0.0

    term_k = _normalize_key(term)
    label_k = _normalize_key(term_label)
    cand_k = _normalize_key(cand_s)
    cand_n = _normalize_text(cand_s)

    if label_k and cand_k and label_k == cand_k:
        return 1.0
    if term_k and cand_k and term_k == cand_k:
        return 1.0

    def _uniq_tokens(s: str) -> list[str]:
        toks = [t for t in re.findall(r"[a-z0-9]+", _normalize_text(s)) if t]
        return list(dict.fromkeys(toks))

    def _coverage_score(query: str) -> float:
        q = str(query or "").strip()
        if not q:
            return 0.0
        cov = _token_overlap_score(q, cand_n)
        if cov <= 0.0:
            return 0.0
        # Penalize candidates with lots of extra tokens to avoid generic over-matches.
        qt = _uniq_tokens(q)
        ct = _uniq_tokens(cand_n)
        extra = max(0, len(ct) - len(qt))
        penalty = min(0.12, 0.01 * float(extra))
        if cov >= 1.0:
            return max(0.0, 0.99 - penalty)
        return max(0.0, float(cov) - penalty)

    best = 0.0
    if term_label:
        best = max(best, _coverage_score(term_label))
    if term:
        best = max(best, 0.95 * _coverage_score(term))

    # Substring containment: helpful for short queries like "measured" vs "measured val".
    try:
        tl_n = _normalize_text(term_label)
        if tl_n and tl_n in cand_n:
            best = max(best, 0.97)
    except Exception:
        pass
    try:
        t_n = _normalize_text(term)
        if t_n and t_n in cand_n:
            best = max(best, 0.90)
    except Exception:
        pass

    if fuzzy_term:
        score = _fuzzy_term_score(term, term_label, cand_s)
        if score >= float(min_ratio or 0.0):
            # Keep strict ordering: only exact-key equality can be 1.0.
            best = max(best, min(0.99, float(score)))
    return float(best)


def _score_header_anchor(
    anchor: str,
    header_cell_text: str,
    *,
    fuzzy_header: bool,
    min_ratio: float,
) -> float:
    """
    Score how well a header cell matches a requested header anchor (0..1).

    Perfect match (1.0) means all anchor tokens are contained within header tokens
    (with abbreviation-style prefix matching for tokens >= 4 chars).
    """
    a = str(anchor or "").strip()
    h = str(header_cell_text or "").strip()
    if not a or not h:
        return 0.0

    # Perfect token coverage (asymmetric): all query tokens appear in the header.
    cov = _token_overlap_score(a, h)
    if cov >= 1.0:
        return 1.0
    if not fuzzy_header:
        return 0.0

    # Fuzzy fallback: sequence similarity on normalized keys + an abbreviation-style token score.
    def _tokens(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", _normalize_text(s))

    def _abbr_score(query: str, candidate: str) -> float:
        qt = _tokens(query)
        ct = _tokens(candidate)
        if not qt or not ct:
            return 0.0
        qi = 0
        matched = 0
        for c in ct:
            found = False
            for j in range(qi, len(qt)):
                q = qt[j]
                if q.startswith(c) or c.startswith(q):
                    matched += 1
                    qi = j + 1
                    found = True
                    break
            if not found:
                continue
        return matched / float(len(ct)) if ct else 0.0

    akey = _normalize_key(a)
    hkey = _normalize_key(h)
    seq = SequenceMatcher(None, akey, hkey).ratio() if (akey and hkey) else 0.0
    abbr = _abbr_score(a, h)
    score = float(max(seq, abbr))
    if score >= float(min_ratio or 0.0):
        return score
    return 0.0


def _resolve_acceptance_term_key(
    lookup: dict[str, str],
    *,
    term: str,
    term_label: str = "",
    fuzzy_term: bool = False,
    fuzzy_min_ratio: float = 0.78,
) -> str | None:
    """Resolve a workbook row's (term, term_label) to an acceptance_cache key."""
    if not isinstance(lookup, dict) or not lookup:
        return None

    term_k = _normalize_key(term)
    label_k = _normalize_key(term_label)

    # Direct hit (lookup keys are already normalized)
    if term_k and term_k in lookup:
        return lookup.get(term_k)
    if label_k and label_k in lookup:
        return lookup.get(label_k)

    # Containment ranking (non-fuzzy mode): pick the best "contains" candidate, not just first/longest.
    if not fuzzy_term:
        best: tuple[float, int, str] | None = None  # (score, len, orig)
        for lk, orig in lookup.items():
            if not lk:
                continue
            ok = False
            if term_k and (lk == term_k or lk in term_k or term_k in lk):
                ok = True
            elif label_k and (lk == label_k or lk in label_k or label_k in lk):
                ok = True
            if not ok:
                continue
            score = _score_term_candidate(term, term_label, orig or lk, fuzzy_term=False, min_ratio=fuzzy_min_ratio)
            key = (float(score), int(len(lk)), str(orig))
            if best is None or key > best:
                best = key
        return best[2] if best else None

    # Fuzzy fallback
    best_score = 0.0
    best_key: str | None = None
    best_len = -1
    for lk, orig in lookup.items():
        if not lk:
            continue
        score = _score_term_candidate(term, term_label, orig or lk, fuzzy_term=True, min_ratio=fuzzy_min_ratio)
        if score <= 0.0:
            continue
        if score > best_score or (score == best_score and len(lk) > best_len):
            best_score = float(score)
            best_key = orig
            best_len = len(lk)
    if best_key is not None and best_score >= float(fuzzy_min_ratio or 0.0):
        return best_key
    return None


def _split_term_instance(term: str) -> tuple[str, int | None]:
    """
    Split a term like "Torque (2)" into ("Torque", 2).

    Returns (term, None) if no numeric instance suffix is found.
    """
    raw = str(term or "").strip()
    if not raw:
        return "", None
    m = re.search(r"\s*\((\d+)\)\s*$", raw)
    if not m:
        return raw, None
    base = raw[: m.start()].strip()
    if not base:
        return raw, None
    try:
        idx = int(m.group(1))
    except Exception:
        return raw, None
    if idx <= 0:
        return raw, None
    return base, idx


def _clean_cell_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00a0", " ")
    s = s.strip()
    # Strip common table decorations.
    s = s.strip("|").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_float_loose(val: object) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    s = str(val).strip()
    if not s:
        return None
    # strip common unit decorations
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _infer_value_type(term: str, *, explicit: str = "", units: str = "", mn: float | None, mx: float | None) -> str:
    """Return 'number' or 'string'."""
    exp = _normalize_text(explicit).replace(" ", "")
    if exp in ("number", "numeric", "float", "double", "int", "integer"):
        return "number"
    if exp in ("string", "text"):
        return "string"

    term_n = _normalize_key(term)
    if term_n in ("program", "title", "vehicle", "rev", "revision", "operator", "facility", "assettype"):
        return "string"
    if term_n in ("serial", "serialnumber"):
        # serials can be string SNxxxx or numeric; if bounds are provided assume numeric is desired
        return "number" if (mn is not None or mx is not None) else "string"

    if str(units or "").strip():
        return "number"
    if mn is not None or mx is not None:
        return "number"
    return "string"


def _parse_sn_from_header(sn_header: str) -> str:
    """Try to normalize a workbook SN column heading to canonical 'SN####' when possible."""
    raw = str(sn_header or "").strip()
    if not raw:
        return ""
    m = re.search(r"\bSN\s*0*(\d{2,})\b", raw, flags=re.IGNORECASE)
    if m:
        return f"SN{int(m.group(1))}"
    m2 = re.search(r"\b(\d{3,})\b", raw)
    if m2 and "sn" in raw.lower():
        return f"SN{int(m2.group(1))}"
    return raw


def _match_serial_key(sn_header: str, known_serials: set[str]) -> str:
    """Return the best known serial key for a workbook header (supports prefixes like VALVE01_SN4001)."""
    raw = str(sn_header or "").strip()
    if not raw:
        return ""
    if raw in known_serials:
        return raw
    canonical = _parse_sn_from_header(raw)
    if canonical in known_serials:
        return canonical
    raw_l = raw.lower()
    # Pick the longest known serial that is a substring of the header.
    best = ""
    for s in known_serials:
        if not s:
            continue
        if s.lower() in raw_l and len(s) > len(best):
            best = s
    return best or raw


_CONTINUED_POPULATION_FIELDS: dict[str, tuple[str, ...]] = {
    "program_title": ("program", "programs", "program_title", "programtitle"),
    "part_number": ("part", "part_number", "partnumber", "part_numbers", "partnumbers", "pn"),
    "acceptance_test_plan_number": (
        "acceptance_test_plan_number",
        "acceptance_test_plan",
        "acceptancetestplan",
        "acceptancetestplannumber",
        "test_plan",
        "testplan",
        "atp",
    ),
    "vendor": ("vendor", "vendors", "vendor_name", "vendorname"),
    "asset_type": ("asset", "asset_type", "assettype", "asset_types", "assettypes"),
    "asset_specific_type": ("asset_specific", "asset_specific_type", "assetspecifictype", "asset_model", "model"),
}


def _normalize_meta_value(value: object) -> str:
    return str(value or "").strip().lower()


def _sanitize_continued_population(payload: Mapping[str, object] | None) -> dict[str, list[str]]:
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, list[str]] = {}
    for field, aliases in _CONTINUED_POPULATION_FIELDS.items():
        raw_values: list[object] = []
        for key in aliases:
            if key not in payload:
                continue
            val = payload.get(key)
            if isinstance(val, (list, tuple, set)):
                raw_values.extend(list(val))
            elif isinstance(val, str):
                raw_values.append(val)
        cleaned = sorted({str(v).strip() for v in raw_values if str(v).strip()})
        if cleaned:
            out[field] = cleaned
    return out


def _extract_continued_population_rules(meta: Mapping[str, object]) -> dict[str, set[str]]:
    raw = meta.get("continued_population")
    if not isinstance(raw, Mapping):
        return {}
    rules: dict[str, set[str]] = {}
    for field, aliases in _CONTINUED_POPULATION_FIELDS.items():
        raw_values: list[object] = []
        for key in aliases:
            if key not in raw:
                continue
            val = raw.get(key)
            if isinstance(val, (list, tuple, set)):
                raw_values.extend(list(val))
            elif isinstance(val, str):
                raw_values.append(val)
        cleaned = {_normalize_meta_value(v) for v in raw_values if _normalize_meta_value(v)}
        if cleaned:
            rules[field] = cleaned
    return rules


def _docs_matching_population_rules(docs: list[dict], rules: Mapping[str, set[str]]) -> list[dict]:
    if not rules:
        return []
    matched: list[dict] = []
    for d in docs:
        for field, allowed in rules.items():
            if not allowed:
                continue
            val = _normalize_meta_value(d.get(field))
            if val and val in allowed:
                matched.append(d)
                break
    return matched


def _format_continued_population_description(rules: Mapping[str, Iterable[str]]) -> str:
    if not rules:
        return ""
    labels = {
        "program_title": "Program",
        "part_number": "Part Number",
        "acceptance_test_plan_number": "Acceptance Test Plan",
        "vendor": "Vendor",
        "asset_type": "Asset Type",
        "asset_specific_type": "Asset Specific Type",
    }
    parts: list[str] = []
    for key in ("program_title", "part_number", "acceptance_test_plan_number", "vendor", "asset_type", "asset_specific_type"):
        raw_vals = rules.get(key) if isinstance(rules, Mapping) else None
        if not raw_vals:
            continue
        vals = sorted({str(v).strip() for v in raw_vals if str(v).strip()})
        if not vals:
            continue
        parts.append(f"{labels.get(key, key)}: {', '.join(vals)}")
    if not parts:
        return ""
    return "Continuous plotting enabled by " + "; ".join(parts)


def _load_metadata_from_artifacts_dir(art_dir: Path) -> dict:
    try:
        for p in sorted(art_dir.glob("*_metadata.json")):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    except Exception:
        pass
    return {}


def _maybe_load_golden_lines(global_repo: Path, meta: Mapping[str, object]) -> list[str] | None:
    """Load synthetic 'golden combined' text if referenced and available."""
    try:
        files = meta.get("files")
        if not isinstance(files, Mapping):
            return None
        name = str(files.get("golden_combined") or "").strip()
        if not name:
            return None
        # Try a few likely locations
        candidates = [
            Path(global_repo) / name,
            Path(global_repo) / "Synthetic_EIDPs" / name,
            Path(global_repo).parent / "Synthetic_EIDPs" / name,
        ]
        for c in candidates:
            try:
                if c.exists():
                    return c.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
    except Exception:
        return None
    return None


def _parse_ascii_tables(lines: list[str]) -> list[dict]:
    """Parse ASCII tables from combined.txt into table blocks with optional key/value mapping.

    Returns list of blocks: {heading, rows, kv}.
    """
    blocks: list[dict] = []
    current_heading = ""
    pending_table_label = ""

    def _looks_like_border(ln: str) -> bool:
        s = ln.strip()
        return (s.startswith("+") and s.endswith("+")) or (set(s) <= set("+-= "))

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "[STRING]":
            # next non-empty line is the content
            j = i + 1
            while j < len(lines) and not str(lines[j]).strip():
                j += 1
            if j < len(lines):
                current_heading = _clean_cell_text(lines[j])
            i = j + 1
            continue

        if ln.strip() == "[TABLE_LABEL]":
            # next non-empty line is the label (applies to the next ASCII table block)
            j = i + 1
            while j < len(lines) and not str(lines[j]).strip():
                j += 1
            if j < len(lines):
                pending_table_label = str(lines[j]).strip()
            i = j + 1
            continue

        if ln.strip().startswith("+") and "-" in ln and ln.strip().endswith("+"):
            # collect table block
            tbl_lines: list[str] = []
            j = i
            while j < len(lines):
                s = str(lines[j]).rstrip("\n")
                if not s.strip():
                    break
                if not (s.strip().startswith(("+", "|"))):
                    break
                tbl_lines.append(s)
                j += 1
            # parse table rows
            rows: list[list[str]] = []
            for tln in tbl_lines:
                if tln.strip().startswith("|"):
                    parts = [p.strip() for p in tln.strip().strip("|").split("|")]
                    parts = [_clean_cell_text(p) for p in parts]
                    # drop empty trailing columns
                    while parts and not parts[-1]:
                        parts.pop()
                    if parts:
                        rows.append(parts)
            # build kv if it looks like Field/Value or key/value table
            kv: dict[str, str] = {}
            if rows:
                # skip header row(s) that look like Field/Value etc
                start_idx = 0
                if len(rows[0]) >= 2 and _normalize_text(rows[0][0]).replace(" ", "") in ("field", "parameter", "name"):
                    start_idx = 1
                for r in rows[start_idx:]:
                    if not r:
                        continue
                    key = _normalize_key(r[0])
                    if not key:
                        continue
                    val = ""
                    if len(r) >= 2:
                        # take last non-empty cell as value
                        for c in reversed(r[1:]):
                            if c:
                                val = c
                                break
                    if val:
                        kv[key] = val
            blocks.append(
                {
                    "heading": current_heading,
                    "table_label": pending_table_label,
                    "rows": rows,
                    "kv": kv,
                }
            )
            pending_table_label = ""
            i = j + 1
            continue

        i += 1
    return blocks


def _extract_value_from_lines(
    lines: list[str],
    *,
    term: str,
    header_anchor: str = "",
    group_after: str = "",
    window_lines: int = 600,
    want_type: str = "string",
    term_label: str = "",
    fuzzy_term: bool = False,
    fuzzy_min_ratio: float = 0.78,
) -> tuple[str | float | None, str]:
    """Return (value, provenance_snippet)."""
    term_n = _normalize_text(term)
    term_label_n = _normalize_text(term_label)
    if not term_n and not term_label_n:
        return None, ""

    start = 0
    if header_anchor.strip():
        anchor_n = _normalize_text(header_anchor)
        anchor_found = False
        for i, ln in enumerate(lines):
            if anchor_n and anchor_n in _normalize_text(ln):
                start = i
                anchor_found = True
                break
        if not anchor_found:
            return None, ""

    end = min(len(lines), start + max(50, int(window_lines)))

    if group_after.strip():
        # "group_after" is a hard anchor: when provided, we must not use any values
        # that occur before its first match (within the scan window).
        ga_n = _normalize_text(group_after)
        ga_k = _normalize_key(group_after)
        try:
            min_ratio = float(
                (os.environ.get("EIDAT_GROUP_ANCHOR_MIN_RATIO") or os.environ.get("EIDAT_FUZZY_HEADER_MIN_RATIO") or "0.72").strip()
            )
        except Exception:
            min_ratio = 0.72

        def _best_anchor_line_index() -> int | None:
            if ga_n:
                for j in range(start, end):
                    if ga_n in _normalize_text(lines[j]):
                        return j
            if not ga_k or len(ga_k) < 6:
                return None
            anchor_words = [w for w in ga_n.split() if w]
            wlen = len(anchor_words)
            best_j: int | None = None
            best_score = 0.0
            for j in range(start, end):
                ln_n = _normalize_text(lines[j])
                if not ln_n:
                    continue
                words = ln_n.split()
                if not words:
                    continue
                # Slide a window so anchors embedded in longer lines can still match well.
                windows: list[str]
                if wlen >= 2 and len(words) > wlen:
                    windows = [" ".join(words[k : k + wlen]) for k in range(0, len(words) - wlen + 1)]
                else:
                    windows = [ln_n]
                for w in windows:
                    wk = _normalize_key(w)
                    if not wk:
                        continue
                    score = 1.0 if (ga_k and (ga_k in wk)) else SequenceMatcher(None, ga_k, wk).ratio()
                    if score > best_score:
                        best_score = score
                        best_j = j
            if best_j is not None and best_score >= float(min_ratio or 0.0):
                return best_j
            return None

        hit = _best_anchor_line_index()
        if hit is None:
            return None, ""
        start = min(hit + 1, end)

    for i in range(start, end):
        ln = lines[i]
        ln_n = _normalize_text(ln)
        match_n = ""
        # Prefer TermLabel when present; Term can be generic and collide across rows.
        if term_label_n and term_label_n in ln_n:
            match_n = term_label_n
        elif term_n and term_n in ln_n:
            match_n = term_n
        elif fuzzy_term:
            score = _fuzzy_term_score(term, term_label, ln_n)
            if score >= float(fuzzy_min_ratio or 0.0):
                match_n = term_label_n or term_n
        if not match_n:
            continue
        # try to parse a number from the text after the term occurrence
        idx = ln_n.find(match_n)
        if idx < 0:
            tail = str(ln)
        else:
            tail = ln[idx + len(match_n) :]
        if want_type == "number":
            m = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", tail)
            if m:
                raw = m.group(0)
                try:
                    val = float(raw.replace(",", ""))
                    return val, ln.strip()[:300]
                except Exception:
                    return raw.strip(), ln.strip()[:300]
        else:
            # Prefer trailing string content after common delimiters.
            tail_s = tail
            if ":" in tail_s:
                tail_s = tail_s.split(":", 1)[1]
            elif "-" in tail_s:
                tail_s = tail_s.split("-", 1)[1]
            tail_s = _clean_cell_text(tail_s)
            if tail_s:
                return tail_s[:200], ln.strip()[:300]
        # fallback: next line numeric
        if i + 1 < end:
            nxt = lines[i + 1].strip()
            if want_type == "number":
                m2 = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", nxt)
                if m2:
                    raw = m2.group(0)
                    try:
                        val = float(raw.replace(",", ""))
                        return val, (ln.strip() + " | " + nxt)[:300]
                    except Exception:
                        return raw.strip(), (ln.strip() + " | " + nxt)[:300]
        # If there's no obvious number, return the rest of the line as a string value.
        tail_s = re.sub(r"^[\s:\-–—]+", "", str(tail)).strip()
        if tail_s:
            return tail_s[:200], ln.strip()[:300]
    return None, ""


def _resolve_support_path(support_dir: Path, maybe_rel: str | None) -> Path | None:
    if not maybe_rel:
        return None
    p = Path(str(maybe_rel)).expanduser()
    if p.is_absolute():
        return p
    return support_dir / p


def _best_doc_for_serial(support_dir: Path, docs: list[dict]) -> dict | None:
    if not docs:
        return None
    best = None
    best_mtime = -1
    for d in docs:
        meta_rel = str(d.get("metadata_rel") or "").strip()
        meta_path = _resolve_support_path(support_dir, meta_rel)
        try:
            mtime = int(meta_path.stat().st_mtime_ns) if meta_path and meta_path.exists() else 0
        except Exception:
            mtime = 0
        if mtime >= best_mtime:
            best_mtime = mtime
            best = d
    return best or docs[0]


def _project_selected_metadata_rels(project_dir: Path) -> set[str]:
    try:
        pj = Path(project_dir).expanduser() / EIDAT_PROJECT_META
        if not pj.exists():
            return set()
        raw = json.loads(pj.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return set()
        vals = raw.get("selected_metadata_rel") or []
        if not isinstance(vals, list):
            return set()
        return {str(v).strip() for v in vals if str(v).strip()}
    except Exception:
        return set()


def _sync_project_workbook_metadata_inplace(
    wb,
    *,
    support_dir: Path,
    docs_all: list[dict],
    docs_preferred: list[dict] | None = None,
) -> dict:
    """
    Update workbook metadata cells/sheets from the index docs list.

    - Prefers docs from docs_preferred per-serial when available; falls back to docs_all.
    - Updates only metadata rows/sheets; does not touch user-entered term data.
    """
    # Build serial -> docs lists.
    docs_by_serial_all: dict[str, list[dict]] = {}
    for d in docs_all:
        sn = str(d.get("serial_number") or "").strip()
        if sn:
            docs_by_serial_all.setdefault(sn, []).append(d)
    known_serials = set(docs_by_serial_all.keys())

    docs_by_serial_pref: dict[str, list[dict]] = {}
    for d in (docs_preferred or []):
        sn = str(d.get("serial_number") or "").strip()
        if sn:
            docs_by_serial_pref.setdefault(sn, []).append(d)

    def _best_doc(sn: str) -> dict | None:
        if sn and sn in docs_by_serial_pref:
            return _best_doc_for_serial(support_dir, docs_by_serial_pref.get(sn, []))
        return _best_doc_for_serial(support_dir, docs_by_serial_all.get(sn, []))

    updated_sources_cells = 0
    if "Sources" in getattr(wb, "sheetnames", []):
        ws_src = wb["Sources"]
        hdrs: dict[str, int] = {}
        for col in range(1, (ws_src.max_column or 0) + 1):
            key = str(ws_src.cell(1, col).value or "").strip().lower()
            if key:
                hdrs[key] = col
        col_sn = hdrs.get("serial_number")
        if col_sn:
            for row in range(2, (ws_src.max_row or 0) + 1):
                sn = str(ws_src.cell(row, int(col_sn)).value or "").strip()
                if not sn:
                    continue
                sn_key = _match_serial_key(sn, known_serials) or sn
                doc = _best_doc(sn_key)
                if not doc:
                    continue
                for k in ("program_title", "document_type", "metadata_rel", "artifacts_rel", "excel_sqlite_rel"):
                    col = hdrs.get(k)
                    if not col:
                        continue
                    ws_src.cell(row, int(col)).value = str(doc.get(k) or "")
                    updated_sources_cells += 1

    ws_master = wb["master"] if "master" in getattr(wb, "sheetnames", []) else wb.active

    # Detect header columns on master sheet.
    header_row = 1
    headers: dict[str, int] = {}
    for col in range(1, (ws_master.max_column or 0) + 1):
        val = ws_master.cell(header_row, col).value
        if val is None or str(val).strip() == "":
            continue
        key = _normalize_text(str(val))
        headers[key] = col
        compact = key.replace(" ", "")
        if compact and compact not in headers:
            headers[compact] = col

    if "term" not in headers:
        # Not an EIDP-style workbook; Sources-only updates may still apply.
        return {
            "ok": True,
            "updated_sources_cells": int(updated_sources_cells),
            "updated_master_cells": 0,
            "updated_metadata_cells": 0,
        }

    col_term = headers["term"]
    col_data_group = headers.get("datagroup") or headers.get("data group")
    fixed_cols = headers.get("max") or headers.get("maximum") or 0
    if not fixed_cols:
        # Fall back to the last known fixed column in this workbook schema.
        for k in ("units", "min", "max"):
            if k in headers:
                fixed_cols = max(fixed_cols, int(headers[k] or 0))

    sn_cols: dict[str, int] = {}
    for col in range(1, (ws_master.max_column or 0) + 1):
        if fixed_cols and col <= fixed_cols:
            continue
        name = str(ws_master.cell(header_row, col).value or "").strip()
        if not name:
            continue
        sn_cols[name] = col

    meta_rows: dict[str, int] = {}
    for row in range(2, (ws_master.max_row or 0) + 1):
        term = str(ws_master.cell(row, col_term).value or "").strip()
        if not term:
            continue
        if col_data_group:
            dg = str(ws_master.cell(row, col_data_group).value or "").strip()
            if dg and _normalize_text(dg) != _normalize_text("Metadata"):
                continue
        key = _normalize_text(term)
        if key and key not in meta_rows:
            meta_rows[key] = row

    meta_field_map = {
        _normalize_text("Program"): "program_title",
        _normalize_text("Asset Type"): "asset_type",
        _normalize_text("Asset Specific Type"): "asset_specific_type",
        _normalize_text("Vendor"): "vendor",
        _normalize_text("Acceptance Test Plan"): "acceptance_test_plan_number",
        _normalize_text("Part Number"): "part_number",
        _normalize_text("Revision"): "revision",
        _normalize_text("Test Date"): "test_date",
        _normalize_text("Report Date"): "report_date",
        _normalize_text("Document Type"): "document_type",
        _normalize_text("Document Acronym"): "document_type_acronym",
        _normalize_text("Similarity Group"): "similarity_group",
    }

    updated_master_cells = 0
    for sn_header, col in sn_cols.items():
        sn = _match_serial_key(sn_header, known_serials) or str(sn_header).strip()
        doc = _best_doc(sn)
        for term_key, doc_field in meta_field_map.items():
            row = meta_rows.get(term_key)
            if not row:
                continue
            val = doc.get(doc_field) if doc else ""
            ws_master.cell(row, col).value = str(val or "")
            updated_master_cells += 1

    updated_meta_cells = 0
    if "metadata" in getattr(wb, "sheetnames", []):
        ws_meta = wb["metadata"]
        header_map: dict[str, int] = {}
        for col in range(1, (ws_meta.max_column or 0) + 1):
            val = str(ws_meta.cell(1, col).value or "").strip()
            if not val:
                continue
            header_map[_normalize_text(val)] = col
        col_sn = header_map.get("serialnumber") or header_map.get("serial")
        row_by_sn: dict[str, int] = {}
        if col_sn:
            for row in range(2, (ws_meta.max_row or 0) + 1):
                sn_val = str(ws_meta.cell(row, col_sn).value or "").strip()
                if sn_val:
                    row_by_sn[sn_val] = row

        fields = {
            "serialnumber": "serial_number",
            "program": "program_title",
            "assettype": "asset_type",
            "vendor": "vendor",
            "acceptancetestplan": "acceptance_test_plan_number",
            "partnumber": "part_number",
            "revision": "revision",
            "testdate": "test_date",
            "reportdate": "report_date",
            "documenttype": "document_type",
            "documentacronym": "document_type_acronym",
            "similaritygroup": "similarity_group",
        }

        for sn_header in sn_cols.keys():
            sn = _match_serial_key(sn_header, known_serials) or str(sn_header).strip()
            if not sn:
                continue
            row = row_by_sn.get(sn)
            if row is None:
                row = (ws_meta.max_row or 0) + 1
                if col_sn:
                    ws_meta.cell(row, col_sn).value = sn
                row_by_sn[sn] = row
            doc = _best_doc(sn)
            for key, doc_field in fields.items():
                col = header_map.get(key)
                if not col:
                    continue
                if key in ("serialnumber", "serial"):
                    val = sn
                else:
                    val = doc.get(doc_field) if doc else ""
                ws_meta.cell(row, col).value = str(val or "")
                updated_meta_cells += 1

    return {
        "ok": True,
        "serials": sorted({_match_serial_key(s, known_serials) or str(s).strip() for s in sn_cols.keys()}),
        "updated_master_cells": int(updated_master_cells),
        "updated_metadata_cells": int(updated_meta_cells),
        "updated_sources_cells": int(updated_sources_cells),
    }


def sync_project_workbook_metadata(global_repo: Path, workbook_path: Path) -> dict:
    """
    Sync a project workbook's metadata sheets/rows to match the canonical index metadata.
    """
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to sync project workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    repo = Path(global_repo).expanduser()
    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Project workbook not found: {wb_path}")

    support_dir = eidat_support_dir(repo)
    docs = read_eidat_index_documents(repo)
    selected = _project_selected_metadata_rels(wb_path.parent)
    docs_preferred = [d for d in docs if str(d.get("metadata_rel") or "").strip() in selected] if selected else None

    try:
        wb = load_workbook(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc

    try:
        res = _sync_project_workbook_metadata_inplace(
            wb,
            support_dir=support_dir,
            docs_all=docs,
            docs_preferred=docs_preferred,
        )
        wb.save(str(wb_path))
        return {"workbook": str(wb_path), **res}
    finally:
        try:
            wb.close()
        except Exception:
            pass


_META_TERM_MAP: dict[str, tuple[str, str]] = {
    # term -> (metadata_key_path, value_type)
    "program": ("program_code", "string"),
    "title": ("program_title", "string"),
    "vehicle": ("vehicle_number", "string"),
    "assettype": ("asset_type", "string"),
    "serial": ("serial_number", "serial"),
    "serialnumber": ("serial_number", "serial"),
    "vendor": ("vendor", "string"),
    "acceptancetestplan": ("acceptance_test_plan_number", "string"),
    "acceptancetestplannumber": ("acceptance_test_plan_number", "string"),
    "part": ("part_number", "string"),
    "partnumber": ("part_number", "string"),
    "model": ("part_number", "string"),
    "modelnumber": ("part_number", "string"),
    "rev": ("revision", "string"),
    "revision": ("revision", "string"),
    "testdate": ("test_date", "string"),
    "reportdate": ("report_date", "string"),
    "documenttype": ("document_type", "string"),
    "documentacronym": ("document_type_acronym", "string"),
    "similaritygroup": ("similarity_group", "string"),
    "operator": ("operator", "string"),
    "facility": ("facility", "string"),
}


EIDAT_VALIDATION_SHEET = "_eidat_validation"


def _collect_unique_nonempty(items: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        s = str(it or "").strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return sorted(out, key=lambda v: v.casefold())


def _load_metadata_candidates() -> dict:
    """
    Load user-provided metadata candidate lists.

    Primary location is under DATA_ROOT so production nodes can override it.
    Falls back to ROOT for dev/workspace convenience.
    """
    for base in (DATA_ROOT, ROOT):
        try:
            cand_path = Path(base) / "user_inputs" / "metadata_candidates.json"
            if not cand_path.exists():
                continue
            data = json.loads(cand_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def _norm_alnum_spaces(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _iter_doc_type_entries(raw: object) -> list[dict]:
    """
    Normalize doc-type entries from metadata_candidates.json.

    Accepts:
      - list[str] (treated as {"name": s, "acronym": s, "aliases":[s]})
      - list[dict] with keys name/acronym/aliases
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            s = item.strip()
            out.append({"name": s, "acronym": s, "aliases": [s]})
            continue
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        acronym = str(item.get("acronym") or "").strip()
        aliases_raw = item.get("aliases")
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            aliases = [str(a).strip() for a in aliases_raw if str(a).strip()]
        if name and name not in aliases:
            aliases.append(name)
        if acronym and acronym not in aliases:
            aliases.append(acronym)
        if not name and aliases:
            name = aliases[0]
        if not acronym and name:
            acronym = name
        if name:
            out.append({"name": name, "acronym": acronym, "aliases": aliases})
    return out


def _iter_named_alias_entries(raw: object) -> list[dict]:
    """
    Normalize allowlist entries that may be:
      - list[str]
      - list[{"name": str, "aliases": [str, ...]}]

    Output entries:
      {"name": <canonical>, "aliases": [<alias1>, ...]}
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            s = item.strip()
            out.append({"name": s, "aliases": [s]})
            continue
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        aliases_raw = item.get("aliases")
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            aliases = [str(a).strip() for a in aliases_raw if str(a).strip()]
        if name and name not in aliases:
            aliases.append(name)
        if not name and aliases:
            name = aliases[0]
        if name:
            out.append({"name": name, "aliases": aliases})
    return out


def _canonical_names(raw: object) -> list[str]:
    return [str(e.get("name") or "").strip() for e in _iter_named_alias_entries(raw) if str(e.get("name") or "").strip()]


@lru_cache(maxsize=1)
def _doc_type_entries_from_candidates() -> list[dict]:
    cand = _load_metadata_candidates()
    raw = cand.get("document_types") if isinstance(cand, dict) else []
    return _iter_doc_type_entries(raw)


@lru_cache(maxsize=1)
def _td_alias_norms() -> tuple[set[str], set[str]]:
    """
    Return (short_token_aliases, long_aliases) normalized for matching.

    short_token_aliases: things like "td" that should match whole tokens only.
    long_aliases: things like "test data" that can match as a substring in normalized blobs.
    """
    entries = _doc_type_entries_from_candidates()
    aliases: list[str] = []
    for e in entries:
        acr = str(e.get("acronym") or "").strip().upper()
        if acr != "TD":
            continue
        aliases.extend([str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()])
    # Conservative fallback if candidates are missing/malformed.
    if not aliases:
        aliases = ["Test Data", "Test-Data", "TestData", "TD"]
    short_tokens: set[str] = set()
    long_aliases: set[str] = set()
    for a in aliases:
        a_norm = _norm_alnum_spaces(a)
        if not a_norm:
            continue
        if len(a_norm) <= 3 and len(a_norm.split()) == 1:
            short_tokens.add(a_norm)
        else:
            long_aliases.add(a_norm)
    # Always treat "td" as a valid token if TD exists at all.
    short_tokens.add("td")
    return short_tokens, long_aliases


def is_test_data_doc(doc: Mapping[str, object]) -> bool:
    """
    Determine whether an index document is a Test Data (TD) report.

    Uses metadata_candidates.json doc-type aliases (acronym TD) to control acceptable variants.
    Falls back to path-based inference for older indexes.
    """
    try:
        dt = str(doc.get("document_type") or "").strip()
    except Exception:
        dt = ""
    try:
        acr = str(doc.get("document_type_acronym") or "").strip()
    except Exception:
        acr = ""

    short_tokens, long_aliases = _td_alias_norms()
    dt_norm = _norm_alnum_spaces(dt)
    acr_norm = _norm_alnum_spaces(acr)
    if dt_norm in short_tokens or dt_norm in long_aliases:
        return True
    if acr_norm in short_tokens or acr_norm in long_aliases:
        return True

    # Older/partial metadata: infer from stored paths too.
    try:
        blob = f"{doc.get('metadata_rel')}\n{doc.get('artifacts_rel')}\n{doc.get('excel_sqlite_rel')}"
    except Exception:
        blob = str(doc)
    blob_norm = _norm_alnum_spaces(blob)
    blob_tokens = set(blob_norm.split())
    if long_aliases and any(a in blob_norm for a in long_aliases):
        return True
    if short_tokens and any(t in blob_tokens for t in short_tokens):
        return True
    return False


def _build_validation_lists(docs: Iterable[Mapping[str, object]], *, extra_meta: Iterable[Mapping[str, object]] | None = None) -> dict[str, list[str]]:
    """
    Build dropdown lists for metadata-driven fields.

    These lists are used for Excel DataValidation in project workbooks.
    """
    docs_list = list(docs or [])
    meta_list = list(extra_meta or [])
    cand = _load_metadata_candidates()

    # Strict allowlists: dropdowns come only from allowlists (not extracted values).
    cand_program_titles = _canonical_names(cand.get("program_titles") if isinstance(cand, dict) else [])
    cand_asset_types = _canonical_names(cand.get("asset_types") if isinstance(cand, dict) else [])
    cand_asset_specific_types = _canonical_names(cand.get("asset_specific_types") if isinstance(cand, dict) else [])
    cand_vendors = _canonical_names(cand.get("vendors") if isinstance(cand, dict) else [])
    cand_part_numbers = _canonical_names(cand.get("part_numbers") if isinstance(cand, dict) else [])
    cand_atp_numbers = _canonical_names(cand.get("acceptance_test_plan_numbers") if isinstance(cand, dict) else [])

    # Document types: use canonical stored values (acronym when present) and canonical acronyms.
    doc_entries = _iter_doc_type_entries(cand.get("document_types") if isinstance(cand, dict) else [])
    cand_doc_types: list[str] = []
    cand_doc_acronyms: list[str] = []
    for e in doc_entries:
        name = str(e.get("name") or "").strip()
        acr = str(e.get("acronym") or "").strip()
        stored = (acr or name).strip()
        if stored:
            cand_doc_types.append(stored)
        if acr:
            cand_doc_acronyms.append(acr)
        elif stored:
            cand_doc_acronyms.append(stored)

    return {
        "program_title": _collect_unique_nonempty(cand_program_titles),
        "asset_type": _collect_unique_nonempty(cand_asset_types),
        "asset_specific_type": _collect_unique_nonempty(cand_asset_specific_types),
        "vendor": _collect_unique_nonempty(cand_vendors),
        "acceptance_test_plan_number": _collect_unique_nonempty(cand_atp_numbers),
        "part_number": _collect_unique_nonempty(cand_part_numbers),
        "document_type": _collect_unique_nonempty(cand_doc_types),
        "document_type_acronym": _collect_unique_nonempty(cand_doc_acronyms),
        "similarity_group": _collect_unique_nonempty(
            [d.get("similarity_group") for d in docs_list] + [m.get("similarity_group") for m in meta_list]
        ),
    }


def _ensure_validation_sheet(wb, *, sheet_name: str = EIDAT_VALIDATION_SHEET):
    try:
        if sheet_name in wb.sheetnames:
            wb.remove(wb[sheet_name])
    except Exception:
        pass
    ws = wb.create_sheet(sheet_name)
    try:
        ws.sheet_state = "hidden"
    except Exception:
        pass
    return ws


def _write_validation_sheet(ws, lists: dict[str, list[str]]) -> dict[str, str]:
    """
    Write lists into a hidden sheet and return dict[field -> Excel range formula].
    """
    try:
        from openpyxl.utils import get_column_letter  # type: ignore
    except Exception:
        return {}

    # Column order is stable so formulas are stable.
    cols: list[tuple[str, str]] = [
        ("Program", "program_title"),
        ("Asset Type", "asset_type"),
        ("Asset Specific Type", "asset_specific_type"),
        ("Vendor", "vendor"),
        ("Acceptance Test Plan", "acceptance_test_plan_number"),
        ("Part Number", "part_number"),
        ("Document Type", "document_type"),
        ("Document Acronym", "document_type_acronym"),
        ("Similarity Group", "similarity_group"),
    ]
    ws.append([label for label, _ in cols])
    max_len = 0
    for _, key in cols:
        max_len = max(max_len, len(lists.get(key, []) or []))
    for i in range(max_len):
        row: list[str] = []
        for _, key in cols:
            vals = lists.get(key, []) or []
            row.append(vals[i] if i < len(vals) else "")
        ws.append(row)

    ranges: dict[str, str] = {}
    for idx, (_, key) in enumerate(cols, start=1):
        vals = lists.get(key, []) or []
        if not vals:
            continue
        # Values start at row 2
        col_letter = get_column_letter(idx)  # type: ignore[name-defined]
        last_row = 1 + len(vals)
        ranges[key] = f"='{ws.title}'!${col_letter}$2:${col_letter}${int(last_row)}"
    return ranges


def _remove_eidat_data_validations(ws, *, sheet_name: str) -> None:
    try:
        dvs = getattr(ws, "data_validations", None)
        if not dvs:
            return
        dv_list = list(getattr(dvs, "dataValidation", []) or [])
        kept = []
        for dv in dv_list:
            try:
                if str(getattr(dv, "errorTitle", "") or "") == "EIDAT" and sheet_name in str(getattr(dv, "formula1", "") or ""):
                    continue
            except Exception:
                pass
            kept.append(dv)
        dvs.dataValidation = kept
    except Exception:
        return


def _apply_list_validation(ws, *, formula_range: str, sqref: str, title: str) -> None:
    try:
        from openpyxl.worksheet.datavalidation import DataValidation  # type: ignore
    except Exception:
        return
    dv = DataValidation(type="list", formula1=str(formula_range), allow_blank=True, showDropDown=True)
    dv.errorTitle = "EIDAT"
    dv.error = f"Select a value from the '{title}' list."
    ws.add_data_validation(dv)
    dv.add(str(sqref))


def _ensure_project_metadata_validations(
    wb,
    *,
    ws_master,
    ws_metadata,
    docs: Iterable[Mapping[str, object]],
    extra_meta: Iterable[Mapping[str, object]] | None = None,
    master_meta_rows: Mapping[str, int] | None = None,
    master_sn_col_range: tuple[int, int] | None = None,
) -> None:
    """
    Ensure workbook has restricted dropdowns for metadata-driven fields.
    """
    try:
        from openpyxl.utils import get_column_letter  # type: ignore
    except Exception:
        return

    lists = _build_validation_lists(docs, extra_meta=extra_meta)
    ws_val = _ensure_validation_sheet(wb, sheet_name=EIDAT_VALIDATION_SHEET)
    ranges = _write_validation_sheet(ws_val, lists)

    # Remove previously-applied validations before re-adding.
    _remove_eidat_data_validations(ws_master, sheet_name=EIDAT_VALIDATION_SHEET)
    if ws_metadata is not None:
        _remove_eidat_data_validations(ws_metadata, sheet_name=EIDAT_VALIDATION_SHEET)

    # Apply to master-sheet metadata rows across SN columns.
    if master_meta_rows and master_sn_col_range:
        sn_first, sn_last = master_sn_col_range
        if sn_first > 0 and sn_last >= sn_first:
            first_col = get_column_letter(int(sn_first))
            last_col = get_column_letter(int(sn_last))
            row_map = {k.casefold(): int(v) for k, v in dict(master_meta_rows).items() if v}

            def _row(term: str) -> int | None:
                return row_map.get(str(term).casefold())

            for term, key in (
                ("Program", "program_title"),
                ("Asset Type", "asset_type"),
                ("Asset Specific Type", "asset_specific_type"),
                ("Vendor", "vendor"),
                ("Acceptance Test Plan", "acceptance_test_plan_number"),
                ("Part Number", "part_number"),
                ("Document Type", "document_type"),
                ("Document Acronym", "document_type_acronym"),
                ("Similarity Group", "similarity_group"),
            ):
                r = _row(term)
                if not r:
                    continue
                fr = ranges.get(key)
                if not fr:
                    continue
                sqref = f"{first_col}{int(r)}:{last_col}{int(r)}"
                _apply_list_validation(ws_master, formula_range=fr, sqref=sqref, title=term)

    # Apply to metadata sheet columns across existing rows.
    if ws_metadata is not None:
        # header -> col
        header_map: dict[str, int] = {}
        try:
            for col in range(1, ws_metadata.max_column + 1):
                val = str(ws_metadata.cell(1, col).value or "").strip()
                if val:
                    header_map[_normalize_text(val)] = col
        except Exception:
            header_map = {}

        max_rows = max(int(ws_metadata.max_row or 1), 500)

        def _col(name: str) -> int | None:
            return header_map.get(_normalize_text(name))

        for header, key in (
            ("Program", "program_title"),
            ("Asset Type", "asset_type"),
            ("Asset Specific Type", "asset_specific_type"),
            ("Vendor", "vendor"),
            ("Acceptance Test Plan", "acceptance_test_plan_number"),
            ("Part Number", "part_number"),
            ("Document Type", "document_type"),
            ("Document Acronym", "document_type_acronym"),
            ("Similarity Group", "similarity_group"),
        ):
            col = _col(header)
            fr = ranges.get(key)
            if not col or not fr:
                continue
            col_letter = get_column_letter(int(col))
            sqref = f"{col_letter}2:{col_letter}{int(max_rows)}"
            _apply_list_validation(ws_metadata, formula_range=fr, sqref=sqref, title=header)


def _meta_get(meta: Mapping[str, object], path: str) -> object | None:
    cur: object = meta
    for part in str(path or "").split("."):
        if not part:
            continue
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _normalize_program_code(val: str) -> str:
    s = str(val or "").strip()
    if not s:
        return ""
    # Prefer an existing code-like token (all caps/digits/underscores).
    m = re.search(r"\b[A-Z0-9]+(?:_[A-Z0-9]+)+\b", s.upper())
    if m:
        return m.group(0)
    # Otherwise normalize title-ish text into a stable code form.
    s = s.upper()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _extract_from_tables(
    blocks: list[dict],
    *,
    term: str,
    term_label: str = "",
    header_anchor: str = "",
    group_after: str = "",
    fuzzy_term: bool = False,
    fuzzy_min_ratio: float = 0.78,
    table_label: str = "",
) -> tuple[str | None, str]:
    term_k = _normalize_key(term)
    term_label_n = _normalize_text(term_label)
    term_label_k = _normalize_key(term_label)
    if not term_k and not term_label_n:
        return None, ""

    anchor = _normalize_text(header_anchor).strip()
    ga = _normalize_text(group_after).strip()
    ga_k = _normalize_key(group_after)
    want_label = _normalize_text(table_label).strip()
    try:
        ga_min_ratio = float(
            (os.environ.get("EIDAT_GROUP_ANCHOR_MIN_RATIO") or os.environ.get("EIDAT_FUZZY_HEADER_MIN_RATIO") or "0.72").strip()
        )
    except Exception:
        ga_min_ratio = 0.72

    def _block_anchor_hit(block: dict) -> tuple[bool, int | None]:
        if not ga:
            return True, None
        heading = _normalize_text(str(block.get("heading") or ""))
        if heading and ga in heading:
            return True, None
        if ga_k and heading:
            hk = _normalize_key(heading)
            if hk and (ga_k in hk):
                return True, None
            if ga_k and hk and len(ga_k) >= 6 and SequenceMatcher(None, ga_k, hk).ratio() >= ga_min_ratio:
                return True, None
        rows = block.get("rows") or []
        if not isinstance(rows, list):
            return False, None
        if not ga_k or len(ga_k) < 6:
            # For very short anchors, avoid fuzzy matching to reduce false positives.
            for idx, r in enumerate(rows):
                try:
                    row_text = _normalize_text(" ".join(str(x or "") for x in r))
                except Exception:
                    row_text = ""
                if row_text and ga and ga in row_text:
                    return True, idx
            return False, None
        best_idx: int | None = None
        best_score = 0.0
        for idx, r in enumerate(rows):
            try:
                row_text = _normalize_text(" ".join(str(x or "") for x in r))
            except Exception:
                row_text = ""
            if not row_text:
                continue
            rk = _normalize_key(row_text)
            if not rk:
                continue
            score = 1.0 if (ga_k and (ga_k in rk)) else SequenceMatcher(None, ga_k, rk).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= ga_min_ratio:
            return True, best_idx
        return False, None

    anchor_found = not bool(ga)
    slice_rows_after: int | None = None

    def _ctx(block: dict) -> str:
        parts: list[str] = []
        tl = str(block.get("table_label") or "").strip()
        hd = str(block.get("heading") or "").strip()
        if tl:
            parts.append(tl)
        if hd:
            parts.append(hd)
        return " ".join(parts).strip()

    for b in blocks:
        if want_label:
            b_label = _normalize_text(str(b.get("table_label") or "")).strip()
            if b_label != want_label:
                continue
        if ga and not anchor_found:
            ok, row_idx = _block_anchor_hit(b)
            if not ok:
                continue
            anchor_found = True
            slice_rows_after = (row_idx + 1) if row_idx is not None else None

        heading = _normalize_text(str(b.get("heading") or ""))
        if anchor and anchor not in heading:
            continue
        # Enforce "group_after": nothing before the anchor may be used.
        if ga and not anchor_found:
            continue

        kv = b.get("kv") or {}
        # If the anchor was found inside this table block (row hit), kv may include
        # rows that occur before the anchor row; avoid kv in that case.
        if slice_rows_after is None and isinstance(kv, dict):
            v = kv.get(term_k) if term_k else None
            if not v and term_label_k:
                v = kv.get(term_label_k)
            if not v and term_k and len(term_k) >= 4:
                for kk, vv in kv.items():
                    kkn = str(kk or "")
                    if not kkn:
                        continue
                    if term_k and (term_k == kkn or term_k in kkn or kkn in term_k):
                        v = vv
                        break
                    if term_label_k and (term_label_k == kkn or term_label_k in kkn or kkn in term_label_k):
                        v = vv
                        break
            if v:
                ctx = _ctx(b)
                return str(v), f"{ctx}: {term} -> {v}".strip(": ")
        rows = b.get("rows") or []
        if not isinstance(rows, list):
            continue
        start_idx = int(slice_rows_after or 0)
        slice_rows_after = None  # only applies to the first block where the anchor is hit
        for r in rows[start_idx:]:
            if not isinstance(r, list) or not r:
                continue
            if len(r) < 2:
                continue
            left_raw = str(r[0] or "")
            left_k = _normalize_key(left_raw)
            left_n = _normalize_text(left_raw)

            direct = False
            # Prefer TermLabel when present; Term can be generic and collide across rows.
            if term_label_n and term_label_n in left_n:
                direct = True
            elif term_label_k and (
                left_k == term_label_k or (len(term_label_k) >= 6 and (term_label_k in left_k or left_k in term_label_k))
            ):
                direct = True
            elif term_k and left_k == term_k:
                direct = True
            elif (not term_label_n and not term_label_k) and term_k and (
                len(term_k) >= 4 and (term_k in left_k or left_k in term_k)
            ):
                direct = True

            if not direct:
                continue

            val = ""
            for c in reversed(r[1:]):
                if str(c).strip():
                    val = str(c).strip()
                    break
            if val:
                ctx = _ctx(b)
                return val, f"{ctx}: {r[0]} -> {val}".strip(": ")

        if fuzzy_term:
            best_score = 0.0
            best_row: list | None = None
            for r in rows[start_idx:]:
                if not isinstance(r, list) or not r or len(r) < 2:
                    continue
                score = _fuzzy_term_score(term, term_label, str(r[0] or ""))
                if score > best_score:
                    best_score = score
                    best_row = r
            if best_row is not None and best_score >= float(fuzzy_min_ratio or 0.0):
                val = ""
                for c in reversed(best_row[1:]):
                    if str(c).strip():
                        val = str(c).strip()
                        break
                if val:
                    ctx = _ctx(b)
                    return val, f"{ctx}: {best_row[0]} -> {val}".strip(": ")
    if ga and not anchor_found:
        return None, ""
    return None, ""


def _parse_requirement_range(raw: str) -> tuple[float | None, float | None]:
    s = str(raw or "").strip()
    if not s:
        return None, None
    s = s.replace("\u2264", "<=").replace("\u2265", ">=")
    # Range like "40 - 46" or "40.0 - 46.0"
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*[-–—]\s*([-+]?\d+(?:\.\d+)?)", s)
    if m:
        return _parse_float_loose(m.group(1)), _parse_float_loose(m.group(2))
    # Greater than or equal
    m = re.search(r"(>=|>|min)\s*([-+]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    if m:
        return _parse_float_loose(m.group(2)), None
    # Less than or equal
    m = re.search(r"(<=|<|max)\s*([-+]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    if m:
        return None, _parse_float_loose(m.group(2))
    # Fallback: two numbers -> treat as min/max
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        return _parse_float_loose(nums[0]), _parse_float_loose(nums[1])
    if len(nums) == 1:
        val = _parse_float_loose(nums[0])
        return val, None
    return None, None


def _extract_from_tables_by_header(
    blocks: list[dict],
    *,
    term: str,
    term_label: str = "",
    header_anchor: str = "",
    group_after: str = "",
    fuzzy_header: bool = False,
    fuzzy_min_ratio: float = 0.72,
    fuzzy_term: bool = False,
    fuzzy_term_min_ratio: float = 0.78,
    table_label: str = "",
) -> tuple[str | None, str, dict]:
    term_k = _normalize_key(term)
    term_n = _normalize_text(term)
    term_label_n = _normalize_text(term_label)
    anchor_n = _normalize_text(header_anchor)
    ga = _normalize_text(group_after).strip()
    ga_k = _normalize_key(group_after)
    want_label = _normalize_text(table_label).strip()
    try:
        ga_min_ratio = float(
            (os.environ.get("EIDAT_GROUP_ANCHOR_MIN_RATIO") or os.environ.get("EIDAT_FUZZY_HEADER_MIN_RATIO") or "0.72").strip()
        )
    except Exception:
        ga_min_ratio = 0.72
    if not term_k and not term_label_n:
        return None, "", {}

    _RANGE_LIKE_RE = re.compile(
        r"[-+]?\d+(?:\.\d+)?\s*(?:[-\u2013\u2014]|to)\s*[-+]?\d+(?:\.\d+)?",
        flags=re.IGNORECASE,
    )

    def _is_range_like(cell: str) -> bool:
        s = str(cell or "").strip()
        if not s:
            return False
        # Avoid treating negative scalar values (e.g. "-40") as ranges.
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
            return False
        return bool(_RANGE_LIKE_RE.search(s))

    def _tokens(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", _normalize_text(s))

    def _abbr_score(anchor: str, header: str) -> float:
        """Score abbreviation-style matches: 'propellant valve resistance' ~= 'prop valve res'."""
        a = _tokens(anchor)
        h = _tokens(header)
        if not a or not h:
            return 0.0
        ai = 0
        matched = 0
        for ht in h:
            found = False
            for j in range(ai, len(a)):
                at = a[j]
                if at.startswith(ht) or ht.startswith(at):
                    matched += 1
                    ai = j + 1
                    found = True
                    break
            if not found:
                continue
        return matched / float(len(h)) if h else 0.0

    def _fuzzy_header_score(anchor: str, header: str) -> float:
        akey = _normalize_key(anchor)
        hkey = _normalize_key(header)
        seq = SequenceMatcher(None, akey, hkey).ratio() if (akey and hkey) else 0.0
        abbr = _abbr_score(anchor, header)
        return max(seq, abbr)

    def _find_col(header_norm: list[str], wanted: str) -> int | None:
        w = _normalize_text(wanted)
        if not w:
            return None
        for idx, h in enumerate(header_norm):
            if not h:
                continue
            if w == h or w in h or h in w:
                return idx
        return None

    def _find_col_keywords(header_norm: list[str], keywords: list[str]) -> int | None:
        for kw in keywords:
            idx = _find_col(header_norm, kw)
            if idx is not None:
                return idx
        return None

    def _find_col_in_headers(headers_norm: list[list[str]], wanted: str) -> int | None:
        for hnorm in headers_norm:
            idx = _find_col(hnorm, wanted)
            if idx is not None:
                return idx
        return None

    def _find_col_in_headers_fuzzy(headers_norm: list[list[str]], wanted: str, min_ratio: float) -> int | None:
        best_idx: int | None = None
        best_score = 0.0
        for hnorm in headers_norm:
            for idx, h in enumerate(hnorm):
                if not h:
                    continue
                score = _fuzzy_header_score(wanted, h)
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx is not None and best_score >= float(min_ratio or 0.0):
            return best_idx
        return None

    def _find_col_keywords_in_headers(headers_norm: list[list[str]], keywords: list[str]) -> int | None:
        for hnorm in headers_norm:
            idx = _find_col_keywords(hnorm, keywords)
            if idx is not None:
                return idx
        return None

    def _find_col_keywords_in_headers_tokens(headers_norm: list[list[str]], keywords: list[str]) -> int | None:
        """Token-based header match to avoid substring false positives (e.g. 'id' in 'psid')."""
        for hnorm in headers_norm:
            for idx, h in enumerate(hnorm):
                if not h:
                    continue
                ht = set(_tokens(h))
                if not ht:
                    continue
                for kw in keywords:
                    kt = _tokens(kw)
                    if not kt:
                        continue
                    if all(t in ht for t in kt):
                        return idx
        return None

    def _looks_like_header_row(row: list[str]) -> bool:
        if not row:
            return False
        cells = [str(c or "").strip() for c in row if str(c or "").strip()]
        if not cells:
            return False
        # If any known header keywords are present, treat as header continuation.
        header_keywords = [
            "tag",
            "term",
            "parameter",
            "id",
            "field",
            "label",
            "description",
            "desc",
            "name",
            "kpi",
            "metric",
            "min",
            "minimum",
            "max",
            "maximum",
            "range",
            "measured",
            "value",
            "actual",
            "result",
            "units",
            "unit",
            "uom",
            "requirement",
            "criteria",
            "spec",
            "acceptance",
            "limit",
            "target",
            "threshold",
            "status",
            "pass",
            "fail",
            "voltage",
        ]
        keyword_hits = 0
        for c in cells:
            cn = _normalize_text(c)
            if not cn:
                continue
            if cn in header_keywords:
                keyword_hits += 1
                continue
            for kw in header_keywords:
                if kw in cn:
                    keyword_hits += 1
                    break

        # Otherwise, treat as header if it is short and mostly non-numeric.
        #
        # Important: do NOT treat a data row as a header continuation just because it contains
        # status text like "PASS"/"FAIL". Many tables have a "Pass?" column where every data row
        # includes PASS, which would otherwise cause us to incorrectly skip the first data row.
        numeric = sum(1 for c in cells if re.search(r"[-+]?\d", c))
        if numeric > 0:
            # If the row contains numbers, only treat it as a header continuation when it's strongly
            # header-like (multiple keyword hits and very few numeric cells).
            return keyword_hits >= 2 and numeric <= 1

        max_len = max(len(c) for c in cells) if cells else 0
        return (keyword_hits >= 1) or (numeric == 0 and max_len <= 24)

    anchor_found = not bool(ga)
    slice_rows_after: int | None = None
    best_key: tuple[float, float, float, int, int] | None = None
    best_val: str | None = None
    best_snip = ""
    best_extra: dict = {}

    def _ctx(block: dict) -> str:
        parts: list[str] = []
        tl = str(block.get("table_label") or "").strip()
        hd = str(block.get("heading") or "").strip()
        if tl:
            parts.append(tl)
        if hd:
            parts.append(hd)
        return " ".join(parts).strip()

    for b_idx, b in enumerate(blocks or []):
        if want_label:
            b_label = _normalize_text(str(b.get("table_label") or "")).strip()
            if b_label != want_label:
                continue
        if ga and not anchor_found:
            heading = _normalize_text(str(b.get("heading") or ""))
            if heading and ga in heading:
                anchor_found = True
                slice_rows_after = None
            elif ga_k and heading:
                hk = _normalize_key(heading)
                if hk and (ga_k in hk):
                    anchor_found = True
                    slice_rows_after = None
                elif hk and len(ga_k) >= 6 and SequenceMatcher(None, ga_k, hk).ratio() >= ga_min_ratio:
                    anchor_found = True
                    slice_rows_after = None
            if not anchor_found:
                rows0 = b.get("rows") or []
                if isinstance(rows0, list):
                    # For very short anchors, avoid fuzzy matching to reduce false positives.
                    if not ga_k or len(ga_k) < 6:
                        for idx, r in enumerate(rows0):
                            if not isinstance(r, list):
                                continue
                            row_text = _normalize_text(" ".join(str(x or "") for x in r))
                            if row_text and ga and ga in row_text:
                                anchor_found = True
                                slice_rows_after = idx + 1
                                break
                    else:
                        best_idx: int | None = None
                        best_score = 0.0
                        for idx, r in enumerate(rows0):
                            if not isinstance(r, list):
                                continue
                            row_text = _normalize_text(" ".join(str(x or "") for x in r))
                            if not row_text:
                                continue
                            rk = _normalize_key(row_text)
                            if not rk:
                                continue
                            score = 1.0 if (ga_k and (ga_k in rk)) else SequenceMatcher(None, ga_k, rk).ratio()
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                        if best_idx is not None and best_score >= ga_min_ratio:
                            anchor_found = True
                            slice_rows_after = best_idx + 1
            if not anchor_found:
                continue

        rows = b.get("rows") or []
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        header = rows[0] if isinstance(rows[0], list) else []
        header2 = rows[1] if len(rows) > 1 and isinstance(rows[1], list) else []
        header_norm = [_normalize_text(c) for c in header]
        header2_norm = [_normalize_text(c) for c in header2] if _looks_like_header_row(header2) else []

        merged_norm: list[str] = []
        if header2_norm:
            max_len = max(len(header), len(header2))
            for i in range(max_len):
                h0 = header[i] if i < len(header) else ""
                h1 = header2[i] if i < len(header2) else ""
                merged_norm.append(_normalize_text(f"{h0} {h1}".strip()))

        header_norms: list[list[str]] = [header_norm]
        if merged_norm:
            header_norms.append(merged_norm)
        if header2_norm:
            header_norms.append(header2_norm)

        term_col = _find_col_keywords_in_headers_tokens(
            header_norms, ["tag", "term", "parameter", "id", "field", "label", "test"]
        )
        if term_col is None:
            term_col = 0
        desc_col = _find_col_keywords_in_headers_tokens(
            header_norms, ["description", "desc", "name", "kpi", "metric", "test descr", "test description"]
        )

        value_col: int | None = None
        header_score = 0.0
        if anchor_n:
            # Pick the best matching column by score (not first match).
            max_cols = max((len(h) for h in header_norms if isinstance(h, list)), default=0)
            best_col: int | None = None
            best_hs = 0.0
            for ci in range(int(max_cols)):
                # Use the best scoring header variant for this column (merged/header2/etc).
                cell_best = 0.0
                for hnorm in header_norms:
                    try:
                        htxt = str(hnorm[ci] if ci < len(hnorm) else "")
                    except Exception:
                        htxt = ""
                    hs = _score_header_anchor(
                        header_anchor,
                        htxt,
                        fuzzy_header=fuzzy_header,
                        min_ratio=fuzzy_min_ratio,
                    )
                    if hs > cell_best:
                        cell_best = hs
                if cell_best > best_hs:
                    best_hs = cell_best
                    best_col = ci
            if best_col is not None and best_hs > 0.0:
                value_col = int(best_col)
                header_score = float(best_hs)
        if value_col is None and not anchor_n:
            # Auto column selection: prefer measured/value only (ignore result/voltage/etc).
            value_col = _find_col_keywords_in_headers(header_norms, ["measured", "measured value", "value", "actual"])

        units_col = _find_col_keywords_in_headers(header_norms, ["units", "unit", "uom"])
        req_col = _find_col_keywords_in_headers(header_norms, ["target", "requirement", "criteria", "spec", "acceptance", "limit"])
        min_col = _find_col_keywords_in_headers(header_norms, ["min", "minimum", "range min", "range (min)"])
        max_col = _find_col_keywords_in_headers(header_norms, ["max", "maximum", "range max", "range (max)"])

        # If a header is explicitly requested, only consider tables that actually contain that header.
        # Otherwise we can match the term in an unrelated table and return None prematurely, which
        # prevents later blocks (the correct table) from being searched.
        if anchor_n and value_col is None:
            continue

        data_start = 2 if header2_norm else 1
        if slice_rows_after is not None:
            data_start = max(data_start, int(slice_rows_after))
            slice_rows_after = None
        for row_idx, r in enumerate(rows[data_start:], start=int(data_start)):
            if not isinstance(r, list) or not r:
                continue
            tag = str(r[term_col] if term_col < len(r) else "").strip()
            desc = str(r[desc_col] if isinstance(desc_col, int) and desc_col < len(r) else "").strip()
            matched_col: int | None = None
            score_tag = _score_term_candidate(
                term,
                term_label,
                tag,
                fuzzy_term=fuzzy_term,
                min_ratio=fuzzy_term_min_ratio,
            )
            score_desc = _score_term_candidate(
                term,
                term_label,
                desc,
                fuzzy_term=fuzzy_term,
                min_ratio=fuzzy_term_min_ratio,
            ) if desc else 0.0
            term_score = float(max(score_tag, score_desc))
            if term_score > 0.0:
                if score_tag >= score_desc:
                    matched_col = int(term_col)
                else:
                    matched_col = int(desc_col) if isinstance(desc_col, int) else int(term_col)
            else:
                # Weak fallback: term appears elsewhere in the row.
                best_cell = 0.0
                best_ci: int | None = None
                for ci, cell in enumerate(r):
                    cell_s = str(cell or "").strip()
                    if not cell_s:
                        continue
                    sc = _score_term_candidate(
                        term,
                        term_label,
                        cell_s,
                        fuzzy_term=fuzzy_term,
                        min_ratio=fuzzy_term_min_ratio,
                    )
                    if sc > best_cell:
                        best_cell = float(sc)
                        best_ci = int(ci)
                if best_ci is not None and best_cell > 0.0:
                    term_score = 0.70 * float(best_cell)
                    matched_col = best_ci
            if term_score <= 0.0:
                continue

            val = None
            if value_col is not None and value_col < len(r):
                val = str(r[value_col]).strip()
            if anchor_n and not val:
                continue
            if not val and not anchor_n:
                # Header not specified: prefer a scalar numeric cell (avoid Spec/Requirement ranges like "20-30").
                if matched_col is not None:
                    numeric_scalar: list[str] = []
                    skip_cols = {c for c in (units_col, req_col, min_col, max_col) if isinstance(c, int)}
                    for ci in range(matched_col + 1, len(r)):
                        if ci in skip_cols:
                            continue
                        cs = str(r[ci] or "").strip()
                        if not cs:
                            continue
                        if _parse_float_loose(cs) is not None and not _is_range_like(cs):
                            numeric_scalar.append(cs)
                    if numeric_scalar:
                        val = numeric_scalar[-1]
                    else:
                        for c in r[matched_col + 1 :]:
                            cs = str(c or "").strip()
                            if cs:
                                val = cs
                                break
            if not val and not anchor_n:
                # Final fallback to last non-empty cell in the row.
                for c in reversed(r[1:]):
                    if str(c).strip():
                        val = str(c).strip()
                        break
            if not val:
                continue

            overall = float(term_score)
            if anchor_n:
                if header_score <= 0.0:
                    continue
                overall = 0.5 * float(term_score) + 0.5 * float(header_score)

            extra = {
                "units": str(r[units_col]).strip() if units_col is not None and units_col < len(r) else "",
                "requirement_raw": str(r[req_col]).strip() if req_col is not None and req_col < len(r) else "",
                "min_raw": str(r[min_col]).strip() if min_col is not None and min_col < len(r) else "",
                "max_raw": str(r[max_col]).strip() if max_col is not None and max_col < len(r) else "",
            }
            ctx = _ctx(b)
            snippet = f"{ctx}: {tag} -> {val}".strip(": ").strip()

            key = (overall, float(term_score), float(header_score), -int(b_idx), -int(row_idx))
            if best_key is None or key > best_key:
                best_key = key
                best_val = str(val)
                best_snip = snippet
                best_extra = dict(extra)

    if ga and not anchor_found:
        return None, "", {}
    if best_val is not None:
        return best_val, best_snip, best_extra
    return None, "", {}

def _row_has_headers(row: list[str], headers: list[str]) -> bool:
    if not row:
        return False
    got = {_normalize_key(c) for c in row if str(c or "").strip()}
    want = {_normalize_key(h) for h in headers if str(h or "").strip()}
    return bool(want) and want.issubset(got)


def _extract_from_paired_acceptance_tables(
    blocks: list[dict],
    *,
    term: str,
    term_label: str,
) -> tuple[str | None, str]:
    """Handle paired tables where criteria are listed in one table and measured values in the next.

    Example pattern (as seen in synthetic debug PDFs):
      - Table A: Tag / Description / Requirement
      - Table B: Measured / Units / Result

    The measured table often does NOT repeat the tag, so we align by row index.
    """
    term_k = _normalize_key(term)
    term_label_n = _normalize_text(term_label)
    if not term_k and not term_label:
        return None, ""

    for i, b in enumerate(blocks):
        rows_a = b.get("rows") or []
        if not isinstance(rows_a, list) or len(rows_a) < 2:
            continue
        if not isinstance(rows_a[0], list) or not _row_has_headers(rows_a[0], ["Tag", "Description", "Requirement"]):
            continue

        # Find the next measured table nearby.
        rows_b: list[list[str]] | None = None
        b_idx = -1
        for j in range(i + 1, min(i + 5, len(blocks))):
            candidate = blocks[j].get("rows") or []
            if not isinstance(candidate, list) or len(candidate) < 2:
                continue
            if isinstance(candidate[0], list) and _row_has_headers(candidate[0], ["Measured", "Units", "Result"]):
                rows_b = candidate  # type: ignore[assignment]
                b_idx = j
                break
        if rows_b is None:
            continue

        data_a = [r for r in rows_a[1:] if isinstance(r, list) and any(str(c or "").strip() for c in r)]
        data_b = [r for r in rows_b[1:] if isinstance(r, list) and any(str(c or "").strip() for c in r)]
        if not data_a or not data_b:
            continue
        # Align by index; allow extra rows in either (use min length)
        n = min(len(data_a), len(data_b))
        data_a = data_a[:n]
        data_b = data_b[:n]

        # Resolve which criteria row corresponds to the requested term.
        best_idx: int | None = None
        best_score = 0.0
        best_tag = ""
        best_desc = ""

        for idx, r in enumerate(data_a):
            tag = str(r[0] if len(r) > 0 else "").strip()
            desc = str(r[1] if len(r) > 1 else "").strip()
            tag_k = _normalize_key(tag)
            desc_n = _normalize_text(desc)

            score = 0.0
            if term_k and tag_k and (term_k == tag_k or term_k in tag_k or tag_k in term_k):
                score = 1.0
            elif term_label_n and desc_n:
                # Similarity match against description.
                score = SequenceMatcher(None, term_label_n, desc_n).ratio()
            elif term_k and desc_n:
                score = SequenceMatcher(None, _normalize_text(term), desc_n).ratio()

            if score > best_score:
                best_score = score
                best_idx = idx
                best_tag = tag
                best_desc = desc

        if best_idx is None:
            continue
        # Require a minimum confidence if we didn't match the tag directly.
        if best_score < 0.55 and not (term_k and _normalize_key(best_tag) and (term_k in _normalize_key(best_tag) or _normalize_key(best_tag) in term_k)):
            continue

        measured_row = data_b[best_idx]
        measured_val = str(measured_row[0] if len(measured_row) > 0 else "").strip()
        if not measured_val:
            continue
        heading = str(b.get("heading") or "").strip()
        return measured_val, f"paired_tables[{i}->{b_idx}] {heading}: {best_tag} ({best_desc}) -> {measured_val}"

    return None, ""


def update_eidp_trending_project_workbook(
    global_repo: Path,
    workbook_path: Path,
    *,
    overwrite: bool = False,
    window_lines: int = 600,
) -> dict:
    """Update the project workbook SN columns using `EIDAT Support/debug/ocr/*/combined.txt` artifacts.

    This is a separate pipeline from the existing simple-schema extraction; it only reads:
      - the project workbook directives
      - `EIDAT Support/eidat_index.sqlite3` (for artifacts lookup)
      - the merged OCR artifacts (`combined.txt`) created during EIDAT Manager processing.
    """
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to update project workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    repo = Path(global_repo).expanduser()
    support_dir = eidat_support_dir(repo)
    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Project workbook not found: {wb_path}")

    # scanner.env: EIDAT_TRENDING_COMBINED_ONLY=1 (or EIDAT_COMBINED_ONLY=1) skips acceptance data entirely.
    env = load_scanner_env()
    combined_only = _env_truthy(env.get("EIDAT_TRENDING_COMBINED_ONLY") or env.get("EIDAT_COMBINED_ONLY"))
    fuzzy_header_stick = _env_truthy(env.get("EIDAT_FUZZY_HEADER_STICK"))
    try:
        fuzzy_header_min_ratio = float(env.get("EIDAT_FUZZY_HEADER_MIN_RATIO", "0.72") or 0.72)
    except Exception:
        fuzzy_header_min_ratio = 0.72

    fuzzy_term_stick = _env_truthy(env.get("EIDAT_FUZZY_TERM_STICK", "1"))
    try:
        fuzzy_term_min_ratio = float(env.get("EIDAT_FUZZY_TERM_MIN_RATIO", "0.78") or 0.78)
    except Exception:
        fuzzy_term_min_ratio = 0.78

    docs = read_eidat_index_documents(repo)
    docs_by_serial: dict[str, list[dict]] = {}
    for d in docs:
        serial = str(d.get("serial_number") or "").strip()
        if not serial:
            continue
        docs_by_serial.setdefault(serial, []).append(d)
    known_serials = set(docs_by_serial.keys())

    try:
        wb = load_workbook(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc

    ws = wb["master"] if "master" in wb.sheetnames else wb.active

    header_row = 1
    headers: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        val = ws.cell(header_row, col).value
        if val is None or str(val).strip() == "":
            continue
        key = _normalize_text(str(val))
        headers[key] = col
        compact = key.replace(" ", "")
        if compact and compact not in headers:
            headers[compact] = col

    required = ["term", "header", "groupafter", "datagroup", "termlabel", "units", "min", "max"]
    missing = [h for h in required if h not in headers]
    if missing:
        raise RuntimeError(f"Project workbook is missing required columns: {', '.join(missing)}")

    col_term = headers["term"]
    col_header = headers["header"]
    col_group_after = headers["groupafter"]
    col_table_label = headers.get("tablelabel") or headers.get("table label")
    col_data_group = headers["datagroup"]
    col_term_label = headers["termlabel"]
    col_units = headers["units"]
    col_min = headers["min"]
    col_max = headers["max"]

    fixed_cols = max(
        col_term,
        col_header,
        col_group_after,
        int(col_table_label or 0),
        col_data_group,
        col_term_label,
        col_units,
        col_min,
        col_max,
    )
    sn_cols: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        if col <= fixed_cols:
            continue
        name = str(ws.cell(header_row, col).value or "").strip()
        if not name:
            continue
        sn_cols[name] = col

    if not sn_cols:
        raise RuntimeError("No SN columns found in project workbook header row.")

    added_serials: list[str] = []
    project_meta: dict = {}
    project_meta_changed = False
    try:
        pj = wb_path.parent / EIDAT_PROJECT_META
        if pj.exists():
            project_meta = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        project_meta = {}

    def _append_serial_columns(serials: list[str]) -> None:
        if not serials:
            return
        meta_rows: dict[str, int] = {}
        for row in range(2, ws.max_row + 1):
            term = str(ws.cell(row, col_term).value or "").strip()
            if not term:
                continue
            if col_data_group:
                dg = str(ws.cell(row, col_data_group).value or "").strip()
                if dg and _normalize_text(dg) != _normalize_text("Metadata"):
                    continue
            key = _normalize_text(term)
            if key and key not in meta_rows:
                meta_rows[key] = row

        meta_field_map = {
            _normalize_text("Program"): "program_title",
            _normalize_text("Asset Type"): "asset_type",
            _normalize_text("Asset Specific Type"): "asset_specific_type",
            _normalize_text("Vendor"): "vendor",
            _normalize_text("Acceptance Test Plan"): "acceptance_test_plan_number",
            _normalize_text("Part Number"): "part_number",
            _normalize_text("Revision"): "revision",
            _normalize_text("Test Date"): "test_date",
            _normalize_text("Report Date"): "report_date",
            _normalize_text("Document Type"): "document_type",
            _normalize_text("Document Acronym"): "document_type_acronym",
            _normalize_text("Similarity Group"): "similarity_group",
        }

        for sn in serials:
            col = ws.max_column + 1
            ws.cell(header_row, col).value = sn
            doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn, []))
            for term_key, doc_field in meta_field_map.items():
                row = meta_rows.get(term_key)
                if not row:
                    continue
                val = doc.get(doc_field) if doc else ""
                ws.cell(row, col).value = str(val or "")

    def _append_metadata_sheet_rows(serials: list[str]) -> None:
        if "metadata" not in wb.sheetnames:
            return
        ws_meta = wb["metadata"]
        header_map: dict[str, int] = {}
        for col in range(1, ws_meta.max_column + 1):
            val = str(ws_meta.cell(1, col).value or "").strip()
            if not val:
                continue
            header_map[_normalize_text(val)] = col
        if not header_map:
            return

        col_sn = header_map.get("serialnumber") or header_map.get("serial")
        existing_serials: set[str] = set()
        if col_sn:
            for row in range(2, ws_meta.max_row + 1):
                sn_val = str(ws_meta.cell(row, col_sn).value or "").strip()
                if sn_val:
                    existing_serials.add(sn_val)

        fields = {
            "serialnumber": "serial_number",
            "program": "program_title",
            "assettype": "asset_type",
            "vendor": "vendor",
            "acceptancetestplan": "acceptance_test_plan_number",
            "partnumber": "part_number",
            "revision": "revision",
            "testdate": "test_date",
            "reportdate": "report_date",
            "documenttype": "document_type",
            "documentacronym": "document_type_acronym",
            "similaritygroup": "similarity_group",
        }

        for sn in serials:
            if col_sn and sn in existing_serials:
                continue
            row = ws_meta.max_row + 1
            doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn, []))
            for key, doc_field in fields.items():
                col = header_map.get(key)
                if not col:
                    continue
                if key in ("serialnumber", "serial"):
                    val = sn
                else:
                    val = doc.get(doc_field) if doc else ""
                ws_meta.cell(row, col).value = str(val or "")

    continued_rules = _extract_continued_population_rules(project_meta)
    if continued_rules:
        desc = _format_continued_population_description(continued_rules)
        if desc and project_meta.get("description") != desc:
            project_meta["description"] = desc
            project_meta_changed = True
        existing_serials: set[str] = set()
        for sn_header in sn_cols.keys():
            raw = str(sn_header).strip()
            if raw:
                existing_serials.add(raw)
            mapped = _match_serial_key(raw, known_serials) if raw else ""
            if mapped:
                existing_serials.add(mapped)
        matched_docs = _docs_matching_population_rules(docs, continued_rules)
        candidate_serials = sorted(
            {
                str(d.get("serial_number") or "").strip()
                for d in matched_docs
                if str(d.get("serial_number") or "").strip()
            }
        )
        new_serials = [sn for sn in candidate_serials if sn not in existing_serials]
        if new_serials:
            _append_serial_columns(new_serials)
            _append_metadata_sheet_rows(new_serials)
            added_serials = list(new_serials)
            project_meta_changed = True
            sn_cols = {}
            for col in range(1, ws.max_column + 1):
                if col <= fixed_cols:
                    continue
                name = str(ws.cell(header_row, col).value or "").strip()
                if not name:
                    continue
                sn_cols[name] = col

    # Cache combined.txt lines per SN
    combined_cache: dict[str, list[str]] = {}
    tables_cache: dict[str, list[dict]] = {}
    meta_cache: dict[str, dict] = {}
    artifacts_used: dict[str, str] = {}
    acceptance_cache: dict[str, dict[str, list[dict]]] = {}
    acceptance_lookup: dict[str, dict[str, str]] = {}
    for sn_header in sn_cols.keys():
        sn = _match_serial_key(sn_header, known_serials)
        doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn, []))
        if not doc:
            continue
        artifacts_rel = str(doc.get("artifacts_rel") or "").strip()
        art_dir = _resolve_support_path(support_dir, artifacts_rel)
        if not art_dir or not art_dir.exists():
            continue
        meta_cache[sn_header] = _load_metadata_from_artifacts_dir(art_dir)
        artifacts_used[sn_header] = str(art_dir)
        combined = art_dir / "combined.txt"
        if not combined.exists():
            continue
        try:
            lines = combined.read_text(encoding="utf-8", errors="ignore").splitlines()
            combined_cache[sn_header] = lines
            tables_cache[sn_header] = _parse_ascii_tables(lines)
        except Exception:
            continue
        try:
            if not combined_only:
                term_data = _extract_acceptance_data_for_serial(support_dir, artifacts_rel)
                if isinstance(term_data, dict) and term_data:
                    acceptance_cache[sn_header] = term_data
                    lookup = {}
                    for k in term_data.keys():
                        kn = _normalize_key(str(k))
                        if kn:
                            lookup[kn] = str(k)
                    acceptance_lookup[sn_header] = lookup
        except Exception:
            pass

    prov_sheet = "_provenance"
    if prov_sheet in wb.sheetnames:
        ws_prov = wb[prov_sheet]
    else:
        ws_prov = wb.create_sheet(prov_sheet)
        ws_prov.append(["SN", "Term Label", "Term", "Units", "Min", "Max", "Value", "Source Artifacts Dir", "Snippet"])
        try:
            ws_prov.sheet_state = "hidden"
        except Exception:
            pass

    # Try to resolve a repo-local mirror folder for debug output.
    project_name = ""
    project_dir = wb_path.parent
    try:
        pj = project_dir / EIDAT_PROJECT_META
        if pj.exists():
            project_name = str(json.loads(pj.read_text(encoding="utf-8")).get("name") or "").strip()
    except Exception:
        project_name = ""

    mirror_dir: Path | None = None
    if project_name:
        mirror_dir = local_projects_mirror_root() / _safe_project_slug(project_name)
    else:
        mirror_dir = local_projects_mirror_root() / "_last"
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        mirror_dir = None

    debug_events: list[dict] = []
    debug_failures: list[dict] = []

    def _debug_term_hits(tables: list[dict], lines: list[str], term: str, term_label: str) -> dict:
        term_k = _normalize_key(term)
        term_label_n = _normalize_text(term_label)
        table_hits: list[dict] = []
        for b in tables or []:
            rows = b.get("rows") or []
            if not isinstance(rows, list) or not rows:
                continue
            header = rows[0] if isinstance(rows[0], list) else []
            for r in rows[1:] if len(rows) > 1 else rows:
                if not isinstance(r, list):
                    continue
                row_text = " | ".join(str(c or "") for c in r)
                row_k = _normalize_key(row_text)
                row_n = _normalize_text(row_text)
                if (term_k and term_k in row_k) or (term_label_n and term_label_n in row_n):
                    table_hits.append(
                        {
                            "heading": str(b.get("heading") or ""),
                            "header": header,
                            "row": r,
                        }
                    )
                if len(table_hits) >= 5:
                    break
            if len(table_hits) >= 5:
                break

        line_hits: list[str] = []
        term_n = _normalize_text(term)
        if term_n:
            for ln in lines or []:
                if term_n in _normalize_text(ln):
                    line_hits.append(str(ln).strip()[:300])
                if len(line_hits) >= 5:
                    break
        return {"table_hits": table_hits, "line_hits": line_hits}

    def _select_acceptance_entry(data_list: list[dict], instance: int | None) -> dict:
        if not data_list:
            return {}
        if instance is None or instance <= 0:
            return data_list[0]
        if instance <= len(data_list):
            return data_list[instance - 1]
        return {}

    updated_cells = 0
    skipped_existing = 0
    missing_source = 0
    missing_value = 0

    # Optional explicit type column
    col_value_type = headers.get("valuetype") or headers.get("value type") or headers.get("value_type")

    term_instance_counts: dict[str, int] = {}
    for row in range(2, ws.max_row + 1):
        raw_term = str(ws.cell(row, col_term).value or "").strip()
        if not raw_term:
            continue
        term, explicit_instance = _split_term_instance(raw_term)
        if not term:
            continue
        header_anchor = str(ws.cell(row, col_header).value or "").strip()
        group_after = str(ws.cell(row, col_group_after).value or "").strip()
        table_label = str(ws.cell(row, col_table_label).value or "").strip() if col_table_label else ""
        data_group = str(ws.cell(row, col_data_group).value or "").strip()
        term_label = str(ws.cell(row, col_term_label).value or "").strip()
        units = str(ws.cell(row, col_units).value or "").strip()
        mn = _parse_float_loose(ws.cell(row, col_min).value)
        mx = _parse_float_loose(ws.cell(row, col_max).value)
        explicit_type = str(ws.cell(row, col_value_type).value or "").strip() if col_value_type else ""

        label_mode = bool(str(table_label or "").strip())

        # Use Data Group as anchor if Header is generic like "Value"
        if _normalize_text(header_anchor) in ("value", "val", "result"):
            header_anchor = data_group or header_anchor

        want_type = _infer_value_type(term, explicit=explicit_type, units=units, mn=mn, mx=mx)
        term_k = _normalize_key(term)
        term_instance_counts[term_k] = term_instance_counts.get(term_k, 0) + 1
        term_instance = explicit_instance or term_instance_counts[term_k]

        for sn_header, col in sn_cols.items():
            cell = ws.cell(row, col)
            if cell.value not in (None, "") and not overwrite:
                allow_replace = False
                try:
                    if header_anchor and _parse_float_loose(cell.value) is None:
                        allow_replace = True
                    if not label_mode:
                        lookup = acceptance_lookup.get(sn_header, {})
                        term_key = _resolve_acceptance_term_key(
                            lookup,
                            term=term,
                            term_label=term_label,
                            fuzzy_term=fuzzy_term_stick,
                            fuzzy_min_ratio=fuzzy_term_min_ratio,
                        )
                        if term_key:
                            data_list = acceptance_cache.get(sn_header, {}).get(term_key, [])
                            data = _select_acceptance_entry(data_list, term_instance)
                            if data.get("value") is not None:
                                existing_num = _parse_float_loose(cell.value)
                                if existing_num is None:
                                    allow_replace = True
                except Exception:
                    allow_replace = False
                if not allow_replace:
                    skipped_existing += 1
                    continue

            meta = meta_cache.get(sn_header) or {}
            used_method = ""
            snippet = ""
            val: object | None = None
            err: str | None = None

            # 0) Acceptance-test DB/TXT extraction (preferred for min/max/units).
            #
            # IMPORTANT: If the workbook row specifies an explicit `Header`, prefer header-driven extraction
            # from the table blocks (combined.txt) so we don't accidentally fill the cell with a different
            # measured value for the same term (acceptance DB is term-instance based and not header aware).
            if (not label_mode) and sn_header in acceptance_cache:
                lookup = acceptance_lookup.get(sn_header, {})
                term_key = _resolve_acceptance_term_key(
                    lookup,
                    term=term,
                    term_label=term_label,
                    fuzzy_term=fuzzy_term_stick,
                    fuzzy_min_ratio=fuzzy_term_min_ratio,
                )
                if term_key:
                    data_list = acceptance_cache[sn_header].get(term_key, [])
                    data = _select_acceptance_entry(data_list, term_instance)
                    # Fill min/max/units if missing in workbook row.
                    if not units and data.get("units"):
                        try:
                            ws.cell(row, col_units).value = data.get("units")
                            units = str(data.get("units") or "").strip()
                        except Exception:
                            pass
                    if mn is None and data.get("requirement_min") is not None:
                        try:
                            ws.cell(row, col_min).value = data.get("requirement_min")
                            mn = _parse_float_loose(data.get("requirement_min"))
                        except Exception:
                            pass
                    if mx is None and data.get("requirement_max") is not None:
                        try:
                            ws.cell(row, col_max).value = data.get("requirement_max")
                            mx = _parse_float_loose(data.get("requirement_max"))
                        except Exception:
                            pass
                    # Only use acceptance-term values when no explicit Header is requested.
                    if not header_anchor and val in (None, ""):
                        if data.get("value") is not None:
                            val = data.get("value")
                            used_method = "acceptance_terms"
                            snippet = str(data.get("requirement_raw") or "")
                        elif data.get("raw"):
                            val = data.get("raw")
                            used_method = "acceptance_terms_raw"
                            snippet = str(data.get("requirement_raw") or "")

            # 1) Metadata-first for known document-profile fields.
            if (not label_mode) and val in (None, "") and term_k in _META_TERM_MAP and isinstance(meta, Mapping):
                meta_path, meta_kind = _META_TERM_MAP[term_k]
                raw = _meta_get(meta, meta_path)
                # Fallbacks for incomplete metadata files.
                if (raw is None or raw == "") and term_k == "program":
                    raw = _meta_get(meta, "program_title")
                if raw not in (None, ""):
                    if meta_kind == "serial" and want_type == "number":
                        digits = re.search(r"(\d{2,})", str(raw))
                        val = int(digits.group(1)) if digits else _parse_float_loose(raw)
                    else:
                        if term_k == "program":
                            val = _normalize_program_code(str(raw))
                        else:
                            val = _clean_cell_text(str(raw))
                    used_method = f"metadata:{meta_path}" if raw == _meta_get(meta, meta_path) else "metadata:fallback"

            # 1b) Manual header-driven extraction (bypass acceptance heuristics)
            if header_anchor:
                tval, tsnip, extra = _extract_from_tables_by_header(
                    tables_cache.get(sn_header) or [],
                    term=term,
                    term_label=term_label,
                    header_anchor=header_anchor,
                    group_after=group_after,
                    fuzzy_header=fuzzy_header_stick,
                    fuzzy_min_ratio=fuzzy_header_min_ratio,
                    fuzzy_term=fuzzy_term_stick,
                    fuzzy_term_min_ratio=fuzzy_term_min_ratio,
                    table_label=table_label,
                )
                if val in (None, "") and tval not in (None, ""):
                    val = _clean_cell_text(str(tval))
                    used_method = "table_header"
                    snippet = tsnip
                # Fill units/min/max from table if missing
                if not units and extra.get("units"):
                    try:
                        ws.cell(row, col_units).value = extra.get("units")
                        units = str(extra.get("units") or "").strip()
                    except Exception:
                        pass
                if mn is None or mx is None:
                    req_raw = extra.get("requirement_raw") or ""
                    min_raw = extra.get("min_raw") or ""
                    max_raw = extra.get("max_raw") or ""
                    if min_raw:
                        try:
                            mn = _parse_float_loose(min_raw)
                            if mn is not None and ws.cell(row, col_min).value in (None, ""):
                                ws.cell(row, col_min).value = mn
                        except Exception:
                            pass
                    if max_raw:
                        try:
                            mx = _parse_float_loose(max_raw)
                            if mx is not None and ws.cell(row, col_max).value in (None, ""):
                                ws.cell(row, col_max).value = mx
                        except Exception:
                            pass
                    if (mn is None or mx is None) and req_raw:
                        rmin, rmax = _parse_requirement_range(req_raw)
                        if mn is None and rmin is not None:
                            mn = rmin
                            try:
                                if ws.cell(row, col_min).value in (None, ""):
                                    ws.cell(row, col_min).value = mn
                            except Exception:
                                pass
                        if mx is None and rmax is not None:
                            mx = rmax
                            try:
                                if ws.cell(row, col_max).value in (None, ""):
                                    ws.cell(row, col_max).value = mx
                            except Exception:
                                pass

            # 1c) Auto header-driven extraction (no explicit Header set)
            if not header_anchor and val in (None, ""):
                tval, tsnip, extra = _extract_from_tables_by_header(
                    tables_cache.get(sn_header) or [],
                    term=term,
                    term_label=term_label,
                    header_anchor="",
                    group_after=group_after,
                    fuzzy_header=False,
                    fuzzy_min_ratio=fuzzy_header_min_ratio,
                    fuzzy_term=fuzzy_term_stick,
                    fuzzy_term_min_ratio=fuzzy_term_min_ratio,
                    table_label=table_label,
                )
                if tval not in (None, ""):
                    val = _clean_cell_text(str(tval))
                    used_method = "table_header_auto"
                    snippet = tsnip
                # Fill units/min/max from table if missing
                if not units and extra.get("units"):
                    try:
                        ws.cell(row, col_units).value = extra.get("units")
                        units = str(extra.get("units") or "").strip()
                    except Exception:
                        pass
                if mn is None or mx is None:
                    req_raw = extra.get("requirement_raw") or ""
                    min_raw = extra.get("min_raw") or ""
                    max_raw = extra.get("max_raw") or ""
                    if min_raw:
                        try:
                            mn = _parse_float_loose(min_raw)
                            if mn is not None and ws.cell(row, col_min).value in (None, ""):
                                ws.cell(row, col_min).value = mn
                        except Exception:
                            pass
                    if max_raw:
                        try:
                            mx = _parse_float_loose(max_raw)
                            if mx is not None and ws.cell(row, col_max).value in (None, ""):
                                ws.cell(row, col_max).value = mx
                        except Exception:
                            pass
                    if (mn is None or mx is None) and req_raw:
                        rmin, rmax = _parse_requirement_range(req_raw)
                        if mn is None and rmin is not None:
                            mn = rmin
                            try:
                                if ws.cell(row, col_min).value in (None, ""):
                                    ws.cell(row, col_min).value = mn
                            except Exception:
                                pass
                        if mx is None and rmax is not None:
                            mx = rmax
                            try:
                                if ws.cell(row, col_max).value in (None, ""):
                                    ws.cell(row, col_max).value = mx
                            except Exception:
                                pass

            # 2) ASCII table extraction from combined.txt (key/value and general tables)
            # Skip if a specific Header is provided (header overrides value selection).
            if val in (None, "") and not header_anchor:
                blocks = tables_cache.get(sn_header) or []
                tval, tsnip = _extract_from_tables(
                    blocks,
                    term=term,
                    term_label=term_label,
                    header_anchor=header_anchor,
                    group_after=group_after,
                    fuzzy_term=fuzzy_term_stick,
                    fuzzy_min_ratio=fuzzy_term_min_ratio,
                    table_label=table_label,
                )
                if tval:
                    val = _clean_cell_text(tval)
                    used_method = "table"
                    snippet = tsnip

            # 2b) Paired acceptance/measured tables (common in synthetic debug PDFs)
            # Skip if a specific Header is provided (header overrides value selection).
            if (not label_mode) and val in (None, "") and not header_anchor:
                blocks = tables_cache.get(sn_header) or []
                tval, tsnip = _extract_from_paired_acceptance_tables(blocks, term=term, term_label=term_label)
                if tval:
                    val = _clean_cell_text(tval)
                    used_method = "paired_tables"
                    snippet = tsnip

            # 3) Fallback: line-based scan
            if (not label_mode) and val in (None, ""):
                lines = combined_cache.get(sn_header)
                if not lines:
                    # Optional synthetic golden fallback when combined isn't available.
                    golden = _maybe_load_golden_lines(repo, meta) if isinstance(meta, Mapping) else None
                    if golden:
                        gblocks = _parse_ascii_tables(golden)
                        if not header_anchor:
                            tval, tsnip = _extract_from_tables(
                                gblocks,
                                term=term,
                                term_label=term_label,
                                header_anchor=header_anchor,
                                group_after=group_after,
                                fuzzy_term=fuzzy_term_stick,
                                fuzzy_min_ratio=fuzzy_term_min_ratio,
                            )
                            if tval:
                                val = _clean_cell_text(tval)
                                used_method = "golden_table"
                                snippet = tsnip
                            else:
                                tval2, tsnip2 = _extract_from_paired_acceptance_tables(
                                    gblocks, term=term, term_label=term_label
                                )
                                if tval2:
                                    val = _clean_cell_text(tval2)
                                    used_method = "golden_paired_tables"
                                    snippet = tsnip2
                        else:
                            tval, tsnip, extra = _extract_from_tables_by_header(
                                gblocks,
                                term=term,
                                term_label=term_label,
                                header_anchor=header_anchor,
                                group_after=group_after,
                                fuzzy_header=False,
                                fuzzy_min_ratio=fuzzy_header_min_ratio,
                                fuzzy_term=fuzzy_term_stick,
                                fuzzy_term_min_ratio=fuzzy_term_min_ratio,
                            )
                            if val in (None, "") and tval not in (None, ""):
                                val = _clean_cell_text(str(tval))
                                used_method = "golden_table_header"
                                snippet = tsnip
                        fallback, snip = _extract_value_from_lines(
                            golden,
                            term=term,
                            header_anchor=header_anchor,
                            group_after=group_after,
                            window_lines=window_lines,
                            want_type=want_type,
                            term_label=term_label,
                            fuzzy_term=fuzzy_term_stick,
                            fuzzy_min_ratio=fuzzy_term_min_ratio,
                        )
                        if fallback not in (None, ""):
                            val = fallback
                            used_method = "golden_lines"
                            snippet = snip
                    if val in (None, ""):
                        missing_source += 1
                        err = "missing_source"
                        try:
                            debug_failures.append(
                                {
                                    "row": int(row),
                                    "sn": sn_header,
                                    "term": term,
                                    "term_label": term_label,
                                    "data_group": data_group,
                                    "header_anchor": header_anchor,
                                    "group_after": group_after,
                                    "value_type": want_type,
                                    "error": err,
                                    "method": used_method,
                                    "artifacts_dir": artifacts_used.get(sn_header, ""),
                                }
                            )
                        except Exception:
                            pass
                        continue
                else:
                    fallback, snip = _extract_value_from_lines(
                        lines,
                        term=term,
                        header_anchor=header_anchor,
                        group_after=group_after,
                        window_lines=window_lines,
                        want_type=want_type,
                        term_label=term_label,
                        fuzzy_term=fuzzy_term_stick,
                        fuzzy_min_ratio=fuzzy_term_min_ratio,
                    )
                    if fallback not in (None, ""):
                        val = fallback
                        used_method = "lines"
                        snippet = snip

            if val in (None, ""):
                missing_value += 1
                err = "missing_value"
                try:
                    debug_hits = _debug_term_hits(
                        tables_cache.get(sn_header, []),
                        combined_cache.get(sn_header, []),
                        term,
                        term_label,
                    )
                    debug_failures.append(
                        {
                            "row": int(row),
                            "sn": sn_header,
                            "term": term,
                            "term_label": term_label,
                            "data_group": data_group,
                            "header_anchor": header_anchor,
                            "group_after": group_after,
                            "value_type": want_type,
                            "error": err,
                            "method": used_method,
                            "snippet": snippet,
                            "debug_hits": debug_hits,
                            "artifacts_dir": artifacts_used.get(sn_header, ""),
                        }
                    )
                except Exception:
                    pass
                continue

            # Coerce value based on type
            if want_type == "number":
                num = _parse_float_loose(val)
                if num is None:
                    missing_value += 1
                    err = "not_numeric"
                    try:
                        debug_hits = _debug_term_hits(
                            tables_cache.get(sn_header, []),
                            combined_cache.get(sn_header, []),
                            term,
                            term_label,
                        )
                        debug_failures.append(
                            {
                                "row": int(row),
                                "sn": sn_header,
                                "term": term,
                                "term_label": term_label,
                                "data_group": data_group,
                                "header_anchor": header_anchor,
                                "group_after": group_after,
                                "value_type": want_type,
                                "error": err,
                                "method": used_method,
                                "value_raw": str(val),
                                "snippet": snippet,
                                "debug_hits": debug_hits,
                                "artifacts_dir": artifacts_used.get(sn_header, ""),
                            }
                        )
                    except Exception:
                        pass
                    continue
                if mn is not None and num < mn:
                    snippet = (snippet + f" [below min {mn}]").strip()
                if mx is not None and num > mx:
                    snippet = (snippet + f" [above max {mx}]").strip()
                cell.value = num
            else:
                cell.value = _clean_cell_text(str(val))
            updated_cells += 1
            ws_prov.append(
                [
                    sn_header,
                    term_label,
                    term,
                    units,
                    mn,
                    mx,
                    cell.value,
                    artifacts_used.get(sn_header, ""),
                    f"{used_method} {snippet}".strip(),
                ]
            )

            # Add debug record for this cell (keep it relatively small).
            try:
                debug_events.append(
                    {
                        "row": int(row),
                        "sn": sn_header,
                        "term": term,
                        "term_label": term_label,
                        "data_group": data_group,
                        "header_anchor": header_anchor,
                        "group_after": group_after,
                        "value_type": want_type,
                        "value": cell.value,
                        "method": used_method,
                        "snippet": snippet,
                        "artifacts_dir": artifacts_used.get(sn_header, ""),
                    }
                )
            except Exception:
                pass

    # Ensure Document Type / Document Acronym reflect current metadata, and add dropdown restrictions.
    try:
        meta_rows: dict[str, int] = {}
        for row in range(2, ws.max_row + 1):
            term = str(ws.cell(row, col_term).value or "").strip()
            if not term:
                continue
            dg = str(ws.cell(row, col_data_group).value or "").strip()
            if dg and _normalize_text(dg) != _normalize_text("Metadata"):
                continue
            if _normalize_text(dg) == _normalize_text("Metadata"):
                if term not in meta_rows:
                    meta_rows[term] = row

        sn_first = min(sn_cols.values()) if sn_cols else 0
        sn_last = max(sn_cols.values()) if sn_cols else 0

        # Refresh all workbook metadata fields from canonical index docs.
        try:
            selected_rels = set()
            if isinstance(project_meta, dict):
                vals = project_meta.get("selected_metadata_rel") or []
                if isinstance(vals, list):
                    selected_rels = {str(v).strip() for v in vals if str(v).strip()}
            docs_preferred = (
                [d for d in docs if str(d.get("metadata_rel") or "").strip() in selected_rels]
                if selected_rels
                else None
            )
            _sync_project_workbook_metadata_inplace(
                wb,
                support_dir=support_dir,
                docs_all=docs,
                docs_preferred=docs_preferred,
            )
        except Exception:
            pass

        def _best_meta_value(sn_header: str, key: str) -> str:
            v = ""
            meta = meta_cache.get(sn_header) or {}
            if isinstance(meta, Mapping):
                try:
                    v = str(_meta_get(meta, key) or "").strip()
                except Exception:
                    v = ""
            if v:
                return v
            sn = _match_serial_key(sn_header, known_serials)
            doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn, []))
            try:
                return str((doc or {}).get(key) or "").strip()
            except Exception:
                return ""

        # Update master-sheet metadata rows (always overwrite for these two fields).
        def _meta_row(label: str) -> int | None:
            want = _normalize_text(label)
            for k, r in meta_rows.items():
                if _normalize_text(k) == want:
                    try:
                        return int(r)
                    except Exception:
                        return None
            return None

        row_doc_type = _meta_row("Document Type")
        row_doc_acr = _meta_row("Document Acronym")
        if row_doc_type or row_doc_acr:
            for sn_header, col in sn_cols.items():
                if row_doc_type:
                    ws.cell(int(row_doc_type), int(col)).value = _best_meta_value(sn_header, "document_type")
                if row_doc_acr:
                    ws.cell(int(row_doc_acr), int(col)).value = _best_meta_value(sn_header, "document_type_acronym")

        # Update metadata sheet values + validations.
        ws_meta = wb["metadata"] if "metadata" in wb.sheetnames else None
        if ws_meta is not None:
            header_map: dict[str, int] = {}
            for col in range(1, ws_meta.max_column + 1):
                val = str(ws_meta.cell(1, col).value or "").strip()
                if val:
                    header_map[_normalize_text(val)] = col

            col_sn = header_map.get("serialnumber")
            col_dt = header_map.get("documenttype")
            col_da = header_map.get("documentacronym")
            if col_sn and (col_dt or col_da):
                for row in range(2, ws_meta.max_row + 1):
                    sn = str(ws_meta.cell(row, int(col_sn)).value or "").strip()
                    if not sn:
                        continue
                    if col_dt:
                        ws_meta.cell(row, int(col_dt)).value = _best_meta_value(sn, "document_type")
                    if col_da:
                        ws_meta.cell(row, int(col_da)).value = _best_meta_value(sn, "document_type_acronym")

        extra_meta = [m for m in (meta_cache.values() or []) if isinstance(m, Mapping)]
        _ensure_project_metadata_validations(
            wb,
            ws_master=ws,
            ws_metadata=ws_meta,
            docs=docs,
            extra_meta=extra_meta,
            master_meta_rows=meta_rows,
            master_sn_col_range=(sn_first, sn_last) if sn_first and sn_last else None,
        )
    except Exception:
        pass

    try:
        wb.save(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc
    finally:
        try:
            wb.close()
        except Exception:
            pass

    if project_meta and project_meta_changed:
        try:
            serials_now = sorted({str(sn).strip() for sn in sn_cols.keys() if str(sn).strip()})
            project_meta["serials"] = serials_now
            project_meta["serials_count"] = len(serials_now)
            (project_dir / EIDAT_PROJECT_META).write_text(json.dumps(project_meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    # Mirror updated workbook + project.json and write debug json into repo-local mirror.
    debug_json_path = ""
    if mirror_dir:
        try:
            if project_name:
                _mirror_project_to_local(project_dir, project_name=project_name)
            else:
                mirror_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(wb_path, mirror_dir / wb_path.name)
                except Exception:
                    pass
            debug_json_path = str((mirror_dir / PROJECT_UPDATE_DEBUG_JSON).resolve())
            payload = {
                "workbook": str(wb_path),
                "global_repo": str(repo),
                "project_name": project_name,
                "project_dir": str(project_dir),
                "overwrite": bool(overwrite),
                "combined_only": bool(combined_only),
                "window_lines": int(window_lines),
                "updated_cells": int(updated_cells),
                "skipped_existing": int(skipped_existing),
                "missing_source": int(missing_source),
                "missing_value": int(missing_value),
                "serials_in_workbook": int(len(sn_cols)),
                "serials_with_source": int(len(combined_cache)),
                "serials_added": int(len(added_serials)),
                "added_serials": added_serials,
                "events_sample": debug_events[-500:],
                "failures": debug_failures[-2000:],
            }
            payload_json = json.dumps(payload, indent=2)
            # Write to local mirror
            (mirror_dir / PROJECT_UPDATE_DEBUG_JSON).write_text(payload_json, encoding="utf-8")
            # Also write to global repo project_dir
            if project_dir and project_dir.exists():
                try:
                    (project_dir / PROJECT_UPDATE_DEBUG_JSON).write_text(payload_json, encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            debug_json_path = ""

    return {
        "workbook": str(wb_path),
        "updated_cells": int(updated_cells),
        "skipped_existing": int(skipped_existing),
        "missing_source": int(missing_source),
        "missing_value": int(missing_value),
        "serials_in_workbook": int(len(sn_cols)),
        "serials_with_source": int(len(combined_cache)),
        "serials_added": int(len(added_serials)),
        "added_serials": added_serials,
        "debug_json": debug_json_path,
    }


def update_eidp_raw_trending_project_workbook(
    global_repo: Path,
    workbook_path: Path,
    *,
    overwrite: bool = False,
    window_lines: int = 600,
) -> dict:
    """Update a raw trending workbook by extracting from combined.txt only."""
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to update project workbooks. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    repo = Path(global_repo).expanduser()
    support_dir = eidat_support_dir(repo)
    wb_path = Path(workbook_path).expanduser()
    if not wb_path.exists():
        raise FileNotFoundError(f"Project workbook not found: {wb_path}")

    env = load_scanner_env()
    fuzzy_header_stick = _env_truthy(env.get("EIDAT_FUZZY_HEADER_STICK"))
    try:
        fuzzy_header_min_ratio = float(env.get("EIDAT_FUZZY_HEADER_MIN_RATIO", "0.72") or 0.72)
    except Exception:
        fuzzy_header_min_ratio = 0.72

    fuzzy_term_stick = _env_truthy(env.get("EIDAT_FUZZY_TERM_STICK", "1"))
    try:
        fuzzy_term_min_ratio = float(env.get("EIDAT_FUZZY_TERM_MIN_RATIO", "0.78") or 0.78)
    except Exception:
        fuzzy_term_min_ratio = 0.78

    docs = read_eidat_index_documents(repo)
    docs_by_serial: dict[str, list[dict]] = {}
    for d in docs:
        serial = str(d.get("serial_number") or "").strip()
        if serial:
            docs_by_serial.setdefault(serial, []).append(d)
    known_serials = set(docs_by_serial.keys())

    try:
        wb = load_workbook(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc

    ws = wb["master"] if "master" in wb.sheetnames else wb.active

    header_row = 1
    headers: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        val = ws.cell(header_row, col).value
        if val is None or str(val).strip() == "":
            continue
        key = _normalize_text(str(val))
        headers[key] = col
        compact = key.replace(" ", "")
        if compact and compact not in headers:
            headers[compact] = col

    def _col(keys: list[str]) -> int | None:
        for k in keys:
            if k in headers:
                return headers[k]
        return None

    col_term = _col(["term"])
    col_term_header = _col(["termheader", "term header"])
    col_header = _col(["header"])
    col_group_after = _col(["groupafter", "group after"])
    col_table_label = _col(["tablelabel", "table label"])
    col_value_type = _col(["valuetype", "value type", "value_type"])
    col_data_group = _col(["datagroup", "data group"])
    col_term_label = _col(["termlabel", "term label"])
    col_units = _col(["units", "unit", "uom"])
    col_min = _col(["min", "minimum", "range (min)", "range min", "range_min"])
    col_max = _col(["max", "maximum", "range (max)", "range max", "range_max"])
    col_serial = _col(["serialnumber", "serial number", "serial", "sn"])
    col_value = _col(["value", "measured", "measured value", "actual", "result"])

    if col_term is None and col_term_header is None and col_term_label is None:
        raise RuntimeError("Raw trending workbook is missing required column: Term (or Term Header / Term Label)")

    # Cache combined.txt lines/tables per serial
    combined_cache: dict[str, list[str]] = {}
    tables_cache: dict[str, list[dict]] = {}
    artifacts_used: dict[str, str] = {}

    def _split_term_header(raw: str) -> tuple[str, str]:
        s = str(raw or "").strip()
        if not s:
            return "", ""
        for delim in ("|", "::", "->", "=>"):
            if delim in s:
                left, right = s.split(delim, 1)
                return left.strip(), right.strip()
        return s, ""

    def _ensure_serial_loaded(sn_raw: str) -> tuple[str, list[str], list[dict]] | None:
        sn_key = _match_serial_key(sn_raw, known_serials) if known_serials else sn_raw
        if not sn_key:
            return None
        if sn_key in combined_cache and sn_key in tables_cache:
            return sn_key, combined_cache[sn_key], tables_cache[sn_key]
        doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn_key, []))
        if not doc:
            return None
        artifacts_rel = str(doc.get("artifacts_rel") or "").strip()
        art_dir = _resolve_support_path(support_dir, artifacts_rel)
        if not art_dir or not art_dir.exists():
            return None
        combined = art_dir / "combined.txt"
        if not combined.exists():
            return None
        try:
            lines = combined.read_text(encoding="utf-8", errors="ignore").splitlines()
            combined_cache[sn_key] = lines
            tables_cache[sn_key] = _parse_ascii_tables(lines)
            artifacts_used[sn_key] = str(art_dir)
        except Exception:
            return None
        return sn_key, combined_cache[sn_key], tables_cache[sn_key]

    # Detect layout:
    # - Row layout: has "Serial Number" + "Value" columns.
    # - Matrix layout: serial numbers are column headers (like EIDP Trending).
    if col_serial is not None and col_value is None:
        raise RuntimeError("Raw trending workbook has 'Serial Number' but is missing the 'Value' column.")
    if col_value is not None and col_serial is None:
        raise RuntimeError("Raw trending workbook has 'Value' but is missing the 'Serial Number' column.")
    use_row_layout = col_serial is not None and col_value is not None
    sn_cols: dict[str, int] = {}
    if not use_row_layout:
        fixed_cols = max(
            col_term or 0,
            col_term_header or 0,
            col_header or 0,
            col_group_after or 0,
            col_table_label or 0,
            col_value_type or 0,
            col_data_group or 0,
            col_term_label or 0,
            col_units or 0,
            col_min or 0,
            col_max or 0,
        )
        for col in range(1, ws.max_column + 1):
            if col <= fixed_cols:
                continue
            name = str(ws.cell(header_row, col).value or "").strip()
            if not name:
                continue
            sn_cols[name] = col
        if not sn_cols:
            raise RuntimeError("Raw trending workbook has no serial columns or Serial Number/Value columns.")

    # Mirror updated workbook + project.json and write debug json into repo-local mirror.
    project_name = ""
    project_dir = wb_path.parent
    try:
        pj = project_dir / EIDAT_PROJECT_META
        if pj.exists():
            project_name = str(json.loads(pj.read_text(encoding="utf-8")).get("name") or "").strip()
    except Exception:
        project_name = ""

    mirror_dir: Path | None = None
    if project_name:
        mirror_dir = local_projects_mirror_root() / _safe_project_slug(project_name)
    else:
        mirror_dir = local_projects_mirror_root() / "_last"
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        mirror_dir = None

    debug_events: list[dict] = []
    debug_failures: list[dict] = []
    updated_cells = 0
    skipped_existing = 0
    missing_source = 0
    missing_value = 0

    def _resolve_row_fields(row_idx: int) -> tuple[str, str, str, str, str, str, str, float | None, float | None, str]:
        term = str(ws.cell(row_idx, col_term).value or "").strip() if col_term else ""
        term_header_raw = str(ws.cell(row_idx, col_term_header).value or "").strip() if col_term_header else ""
        header_anchor = str(ws.cell(row_idx, col_header).value or "").strip() if col_header else ""
        group_after = str(ws.cell(row_idx, col_group_after).value or "").strip() if col_group_after else ""
        table_label = str(ws.cell(row_idx, col_table_label).value or "").strip() if col_table_label else ""
        data_group = str(ws.cell(row_idx, col_data_group).value or "").strip() if col_data_group else ""
        term_label = str(ws.cell(row_idx, col_term_label).value or "").strip() if col_term_label else ""
        units = str(ws.cell(row_idx, col_units).value or "").strip() if col_units else ""
        mn = _parse_float_loose(ws.cell(row_idx, col_min).value) if col_min else None
        mx = _parse_float_loose(ws.cell(row_idx, col_max).value) if col_max else None
        explicit_type = str(ws.cell(row_idx, col_value_type).value or "").strip() if col_value_type else ""

        if term_header_raw:
            t_part, h_part = _split_term_header(term_header_raw)
            if not term:
                term = t_part
            if not header_anchor and h_part:
                header_anchor = h_part
            if not header_anchor and not col_header and term and term_header_raw and term_header_raw != term:
                header_anchor = term_header_raw

        if not term and term_label:
            term = term_label

        # Use Data Group as anchor if Header is generic like "Value"
        if _normalize_text(header_anchor) in ("value", "val", "result"):
            header_anchor = data_group or header_anchor

        want_type = _infer_value_type(term, explicit=explicit_type, units=units, mn=mn, mx=mx)
        return term, header_anchor, group_after, table_label, data_group, term_label, units, mn, mx, want_type

    def _extract_value_for_serial(
        *,
        term: str,
        term_label: str,
        header_anchor: str,
        group_after: str,
        table_label: str,
        want_type: str,
        lines: list[str],
        tables: list[dict],
    ) -> tuple[object | None, str, str, dict]:
        used_method = ""
        snippet = ""
        val: object | None = None
        extra: dict = {}

        tval, tsnip, extra = _extract_from_tables_by_header(
            tables,
            term=term,
            term_label=term_label,
            header_anchor=header_anchor,
            group_after=group_after,
            fuzzy_header=fuzzy_header_stick,
            fuzzy_min_ratio=fuzzy_header_min_ratio,
            fuzzy_term=fuzzy_term_stick,
            fuzzy_term_min_ratio=fuzzy_term_min_ratio,
            table_label=table_label,
        )
        if tval not in (None, ""):
            val = _clean_cell_text(str(tval))
            used_method = "table_header" if header_anchor else "table_header_auto"
            snippet = tsnip

        if val in (None, "") and not header_anchor:
            tval2, tsnip2 = _extract_from_tables(
                tables,
                term=term,
                term_label=term_label,
                header_anchor=header_anchor,
                group_after=group_after,
                fuzzy_term=fuzzy_term_stick,
                fuzzy_min_ratio=fuzzy_term_min_ratio,
                table_label=table_label,
            )
            if tval2:
                val = _clean_cell_text(str(tval2))
                used_method = "table"
                snippet = tsnip2

        if val in (None, "") and not str(table_label or "").strip():
            fallback, snip = _extract_value_from_lines(
                lines,
                term=term,
                header_anchor=header_anchor,
                group_after=group_after,
                window_lines=window_lines,
                want_type=want_type,
                term_label=term_label,
                fuzzy_term=fuzzy_term_stick,
                fuzzy_min_ratio=fuzzy_term_min_ratio,
            )
            if fallback not in (None, ""):
                val = fallback
                used_method = "lines"
                snippet = snip

        return val, used_method, snippet, extra

    for row in range(2, ws.max_row + 1):
        term, header_anchor, group_after, table_label, data_group, term_label, units, mn, mx, want_type = _resolve_row_fields(row)
        if not term:
            continue
        if _normalize_text(data_group) == "metadata":
            continue

        if use_row_layout:
            sn_raw = str(ws.cell(row, col_serial).value or "").strip() if col_serial else ""
            if not sn_raw:
                missing_source += 1
                debug_failures.append({"row": int(row), "term": term, "error": "missing_serial"})
                continue
            loaded = _ensure_serial_loaded(sn_raw)
            if not loaded:
                missing_source += 1
                debug_failures.append({"row": int(row), "term": term, "serial": sn_raw, "error": "missing_source"})
                continue

            sn_key, lines, tables = loaded
            val_cell = ws.cell(row, col_value)
            if val_cell.value not in (None, "") and not overwrite:
                skipped_existing += 1
                continue

            val, used_method, snippet, extra = _extract_value_for_serial(
                term=term,
                term_label=term_label,
                header_anchor=header_anchor,
                group_after=group_after,
                table_label=table_label,
                want_type=want_type,
                lines=lines,
                tables=tables,
            )
            if val in (None, ""):
                missing_value += 1
                debug_failures.append({"row": int(row), "term": term, "serial": sn_raw, "error": "missing_value"})
                continue

            val_cell.value = val
            updated_cells += 1

            # Fill units/min/max from table if missing
            if col_units and not units and extra.get("units"):
                try:
                    ws.cell(row, col_units).value = extra.get("units")
                    units = str(extra.get("units") or "").strip()
                except Exception:
                    pass
            if (col_min or col_max) and (mn is None or mx is None):
                req_raw = extra.get("requirement_raw") or ""
                min_raw = extra.get("min_raw") or ""
                max_raw = extra.get("max_raw") or ""
                if col_min and min_raw:
                    try:
                        mn = _parse_float_loose(min_raw)
                        if mn is not None and ws.cell(row, col_min).value in (None, ""):
                            ws.cell(row, col_min).value = mn
                    except Exception:
                        pass
                if col_max and max_raw:
                    try:
                        mx = _parse_float_loose(max_raw)
                        if mx is not None and ws.cell(row, col_max).value in (None, ""):
                            ws.cell(row, col_max).value = mx
                    except Exception:
                        pass
                if (mn is None or mx is None) and req_raw:
                    rmin, rmax = _parse_requirement_range(req_raw)
                    if col_min and mn is None and rmin is not None:
                        mn = rmin
                        try:
                            if ws.cell(row, col_min).value in (None, ""):
                                ws.cell(row, col_min).value = mn
                        except Exception:
                            pass
                    if col_max and mx is None and rmax is not None:
                        mx = rmax
                        try:
                            if ws.cell(row, col_max).value in (None, ""):
                                ws.cell(row, col_max).value = mx
                        except Exception:
                            pass

            try:
                debug_events.append(
                    {
                        "row": int(row),
                        "serial": sn_key,
                        "term": term,
                        "term_label": term_label,
                        "header_anchor": header_anchor,
                        "group_after": group_after,
                        "value_type": want_type,
                        "value": val,
                        "method": used_method,
                        "snippet": snippet,
                        "artifacts_dir": artifacts_used.get(sn_key, ""),
                    }
                )
            except Exception:
                pass
            continue

        # Matrix layout: iterate serial columns
        for sn_header, col in sn_cols.items():
            cell = ws.cell(row, col)
            if cell.value not in (None, "") and not overwrite:
                skipped_existing += 1
                continue
            loaded = _ensure_serial_loaded(sn_header)
            if not loaded:
                missing_source += 1
                debug_failures.append({"row": int(row), "term": term, "serial": sn_header, "error": "missing_source"})
                continue
            sn_key, lines, tables = loaded
            val, used_method, snippet, extra = _extract_value_for_serial(
                term=term,
                term_label=term_label,
                header_anchor=header_anchor,
                group_after=group_after,
                table_label=table_label,
                want_type=want_type,
                lines=lines,
                tables=tables,
            )
            if val in (None, ""):
                missing_value += 1
                debug_failures.append({"row": int(row), "term": term, "serial": sn_header, "error": "missing_value"})
                continue
            cell.value = val
            updated_cells += 1

            # Fill units/min/max from table if missing (row-level)
            if col_units and not units and extra.get("units"):
                try:
                    ws.cell(row, col_units).value = extra.get("units")
                    units = str(extra.get("units") or "").strip()
                except Exception:
                    pass
            if (col_min or col_max) and (mn is None or mx is None):
                req_raw = extra.get("requirement_raw") or ""
                min_raw = extra.get("min_raw") or ""
                max_raw = extra.get("max_raw") or ""
                if col_min and min_raw:
                    try:
                        mn = _parse_float_loose(min_raw)
                        if mn is not None and ws.cell(row, col_min).value in (None, ""):
                            ws.cell(row, col_min).value = mn
                    except Exception:
                        pass
                if col_max and max_raw:
                    try:
                        mx = _parse_float_loose(max_raw)
                        if mx is not None and ws.cell(row, col_max).value in (None, ""):
                            ws.cell(row, col_max).value = mx
                    except Exception:
                        pass
                if (mn is None or mx is None) and req_raw:
                    rmin, rmax = _parse_requirement_range(req_raw)
                    if col_min and mn is None and rmin is not None:
                        mn = rmin
                        try:
                            if ws.cell(row, col_min).value in (None, ""):
                                ws.cell(row, col_min).value = mn
                        except Exception:
                            pass
                    if col_max and mx is None and rmax is not None:
                        mx = rmax
                        try:
                            if ws.cell(row, col_max).value in (None, ""):
                                ws.cell(row, col_max).value = mx
                        except Exception:
                            pass

            try:
                debug_events.append(
                    {
                        "row": int(row),
                        "serial": sn_key,
                        "term": term,
                        "term_label": term_label,
                        "header_anchor": header_anchor,
                        "group_after": group_after,
                        "value_type": want_type,
                        "value": val,
                        "method": used_method,
                        "snippet": snippet,
                        "artifacts_dir": artifacts_used.get(sn_key, ""),
                    }
                )
            except Exception:
                pass

    try:
        wb.save(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc
    finally:
        try:
            wb.close()
        except Exception:
            pass

    debug_json_path = ""
    if mirror_dir:
        try:
            if project_name:
                _mirror_project_to_local(project_dir, project_name=project_name)
            else:
                mirror_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(wb_path, mirror_dir / wb_path.name)
                except Exception:
                    pass
            debug_json_path = str((mirror_dir / PROJECT_UPDATE_DEBUG_JSON).resolve())
            payload = {
                "workbook": str(wb_path),
                "global_repo": str(repo),
                "project_name": project_name,
                "project_dir": str(project_dir),
                "overwrite": bool(overwrite),
                "window_lines": int(window_lines),
                "updated_cells": int(updated_cells),
                "skipped_existing": int(skipped_existing),
                "missing_source": int(missing_source),
                "missing_value": int(missing_value),
                "serials_in_workbook": int(len(sn_cols) if not use_row_layout else (ws.max_row - 1)),
                "serials_with_source": int(len(combined_cache)),
                "events_sample": debug_events[-500:],
                "failures": debug_failures[-2000:],
            }
            payload_json = json.dumps(payload, indent=2)
            (mirror_dir / PROJECT_UPDATE_DEBUG_JSON).write_text(payload_json, encoding="utf-8")
            if project_dir and project_dir.exists():
                try:
                    (project_dir / PROJECT_UPDATE_DEBUG_JSON).write_text(payload_json, encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            debug_json_path = ""

    return {
        "workbook": str(wb_path),
        "updated_cells": int(updated_cells),
        "skipped_existing": int(skipped_existing),
        "missing_source": int(missing_source),
        "missing_value": int(missing_value),
        "serials_in_workbook": int(len(sn_cols) if not use_row_layout else (ws.max_row - 1)),
        "serials_with_source": int(len(combined_cache)),
        "debug_json": debug_json_path,
    }


def _registry_json_path(global_repo: Path) -> Path:
    return eidat_projects_root(global_repo) / EIDAT_PROJECTS_REGISTRY_JSON


def _registry_db_path(global_repo: Path) -> Path:
    return eidat_projects_root(global_repo) / EIDAT_PROJECTS_REGISTRY_DB


def _connect_projects_registry(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA busy_timeout=2500;")
    except Exception:
        pass
    return conn


def _ensure_projects_registry_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          type TEXT NOT NULL,
          project_dir TEXT NOT NULL,
          workbook TEXT NOT NULL,
          created_by TEXT,
          created_epoch_ns INTEGER NOT NULL,
          updated_epoch_ns INTEGER NOT NULL,
          UNIQUE(name, type)
        );
        CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_epoch_ns);
        """
    )


def _ensure_projects_writable(global_repo: Path) -> None:
    repo = Path(global_repo).expanduser()
    root = eidat_projects_root(repo)
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".__eidat_write_probe_{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        try:
            probe.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if probe.exists():
                probe.unlink()
    except PermissionError as exc:
        raise RuntimeError(
            f"Projects folder is not writable; contact admin to grant Modify rights to: {root}"
        ) from exc
    except Exception:
        # Non-fatal for listing; create/delete will also validate.
        return


def _migrate_projects_json_to_sqlite(repo: Path, *, db_path: Path) -> None:
    json_path = _registry_json_path(repo)
    if not json_path.exists() or db_path.exists():
        return
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, list):
        return
    conn = _connect_projects_registry(db_path)
    try:
        _ensure_projects_registry_schema(conn)
        now_ns = __import__("time").time_ns()
        for p in raw:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            ptype = str(p.get("type") or "").strip()
            pdir = str(p.get("project_dir") or "").strip()
            wb = str(p.get("workbook") or "").strip()
            if not (name and ptype and pdir and wb):
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO projects(name, type, project_dir, workbook, created_by, created_epoch_ns, updated_epoch_ns)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, ptype, pdir, wb, None, now_ns, now_ns),
                )
            except Exception:
                pass
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_eidat_projects(global_repo: Path) -> list[dict]:
    repo = Path(global_repo).expanduser()
    _ensure_projects_writable(repo)
    db_path = _registry_db_path(repo)
    _migrate_projects_json_to_sqlite(repo, db_path=db_path)

    conn = _connect_projects_registry(db_path)
    try:
        _ensure_projects_registry_schema(conn)
        rows = conn.execute(
            """
            SELECT name, type, project_dir, workbook
            FROM projects
            ORDER BY updated_epoch_ns DESC, id DESC
            """
        ).fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    projects_root = eidat_projects_root(repo)
    out: list[dict] = []
    for r in rows:
        p = dict(r)
        raw_dir = str(p.get("project_dir") or "").strip()
        if raw_dir:
            proj_path = Path(raw_dir).expanduser()
            if not proj_path.is_absolute():
                proj_path = repo / proj_path
            try:
                proj_res = proj_path.resolve()
            except Exception:
                proj_res = proj_path.absolute()
            try:
                root_res = projects_root.resolve()
            except Exception:
                root_res = projects_root.absolute()
            try:
                proj_res.relative_to(root_res)
                if not proj_res.exists():
                    continue
            except Exception:
                continue
        out.append(p)

    return out


def delete_eidat_project(global_repo: Path, project_dir: Path) -> dict:
    """Delete a project folder and remove it from the projects registry."""
    repo = Path(global_repo).expanduser()
    _ensure_projects_writable(repo)
    proj_dir = resolve_path_within_global_repo(repo, Path(project_dir), "Project folder")
    if not proj_dir.exists():
        raise FileNotFoundError(f"Project folder not found: {proj_dir}")

    proj_name = ""
    try:
        pj = proj_dir / EIDAT_PROJECT_META
        if pj.exists():
            proj_name = str(json.loads(pj.read_text(encoding="utf-8")).get("name") or "").strip()
    except Exception:
        proj_name = ""

    try:
        proj_rel = str(proj_dir.resolve().relative_to(repo.resolve()))
    except Exception:
        proj_rel = str(proj_dir)

    # Delete folder (after registry filtering so a partially-deleted folder isn't still listed).
    shutil.rmtree(proj_dir)

    removed = 0
    db_path = _registry_db_path(repo)
    conn = _connect_projects_registry(db_path)
    try:
        _ensure_projects_registry_schema(conn)
        try:
            cur = conn.execute(
                "DELETE FROM projects WHERE project_dir = ? OR name = ?",
                (proj_rel, proj_dir.name),
            )
            removed = int(cur.rowcount or 0)
        except Exception:
            removed = 0
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Best-effort delete local mirror folder too.
    try:
        if proj_name:
            mirror = local_projects_mirror_root() / _safe_project_slug(proj_name)
            if mirror.exists():
                shutil.rmtree(mirror, ignore_errors=True)
    except Exception:
        pass

    return {"deleted_dir": str(proj_dir), "removed_registry_entries": int(removed)}


def _mirror_project_to_local(project_dir: Path, *, project_name: str) -> Path:
    """Copy project folder (workbook + project.json) into the repo-local mirror."""
    src = Path(project_dir).expanduser()
    dest_root = local_projects_mirror_root()
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / _safe_project_slug(project_name)

    # Safety: if the "local mirror" resolves into the same folder as the source
    # (e.g., when global_run_mirror is a junction into the Global Repo), do not delete/copy.
    try:
        src_res = src.resolve()
    except Exception:
        src_res = src.absolute()
    try:
        dest_res = dest.resolve()
    except Exception:
        dest_res = dest.absolute()

    try:
        if src_res == dest_res or src_res in dest_res.parents or dest_res in src_res.parents:
            return dest
    except Exception:
        # If we can't safely compare, proceed without deleting anything.
        pass

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy primary artifacts
    for name in (EIDAT_PROJECT_META,):
        p = src / name
        if p.exists():
            shutil.copy2(p, dest / p.name)
    for wb in src.glob("*.xlsx"):
        try:
            shutil.copy2(wb, dest / wb.name)
        except Exception:
            pass
    return dest


def _safe_project_slug(name: str) -> str:
    cleaned = "".join(ch for ch in (name or "").strip() if ch not in '<>:"/\\|?*').strip()
    cleaned = cleaned.replace("\t", " ").replace("\n", " ").strip()
    return cleaned or "New Project"


def _ensure_file_written(path: Path, *, create_fn) -> None:
    """
    Ensure a file exists by attempting to create it (best-effort) and verifying via stat/open.

    `create_fn` should write the file to `path`.
    """
    try:
        if path.exists():
            # Verify it's actually readable (guards against edge cases where exists() lies).
            with path.open("rb"):
                return
    except Exception:
        pass

    try:
        create_fn()
    except Exception:
        # We'll verify below and raise a clear error if still missing.
        pass

    try:
        with path.open("rb"):
            return
    except Exception as exc:
        raise RuntimeError(f"Failed to create file: {path} ({exc})") from exc


def _load_excel_trend_config(path: Path) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Excel trend config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Excel trend config must be a JSON object.")
    cols = data.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("Excel trend config must include a non-empty `columns` list.")
    stats = data.get("statistics") or ["mean", "min", "max", "std", "median", "count"]
    if not isinstance(stats, list) or not all(isinstance(s, str) and s for s in stats):
        raise ValueError("Excel trend config `statistics` must be a list of strings.")
    data["statistics"] = [str(s).strip().lower() for s in stats]
    if "header_row" not in data:
        data["header_row"] = 0
    return data


def _write_test_data_trending_workbook(
    path: Path,
    *,
    global_repo: Path | None,
    serials: list[str],
    docs: list[dict] | None,
    config: dict,
) -> None:
    try:
        from openpyxl import Workbook  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to create the Test Data Trending workbook. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    cfg_cols = config.get("columns") or []
    cfg_stats = config.get("statistics") or []
    col_names: list[str] = []
    for c in cfg_cols:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if name:
            col_names.append(name)
    stats = [str(s).strip().lower() for s in cfg_stats if str(s).strip()]
    if not stats:
        stats = ["mean", "min", "max", "std", "median", "count"]

    def _read_runs_from_sqlite(sqlite_path: Path) -> list[str]:
        p = Path(sqlite_path).expanduser()
        if not p.exists() or not p.is_file():
            return []
        try:
            with sqlite3.connect(str(p)) as conn:
                rows = conn.execute("SELECT sheet_name FROM __sheet_info ORDER BY sheet_name").fetchall()
            out = []
            for r in rows:
                try:
                    sname = str(r[0] or "").strip()
                except Exception:
                    sname = ""
                if sname:
                    out.append(sname)
            return out
        except Exception:
            return []

    runs: list[str] = []
    seen_runs: set[str] = set()
    repo = Path(global_repo).expanduser() if global_repo is not None else None
    for d in (docs or []):
        raw = str((d or {}).get("excel_sqlite_rel") or "").strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute() and repo is not None:
            p = repo / p
        for rn in _read_runs_from_sqlite(p):
            key = rn.strip().lower()
            if not key or key in seen_runs:
                continue
            seen_runs.add(key)
            runs.append(rn)
    if not runs:
        runs = ["Run1"]

    wb = Workbook()

    sn_cols = [str(s).strip() for s in serials if str(s).strip()]

    # New user-editable EIDP-style Data sheet (definitions + values).
    ws_data = wb.active
    ws_data.title = "Data"
    data_headers = ["Term", "Header", "GroupAfter", "Table Label", "ValueType", "Data Group", "Term Label", "Units", "Min", "Max"]
    data_headers.extend(sn_cols)
    ws_data.append(data_headers)
    try:
        ws_data.freeze_panes = "A2"
    except Exception:
        pass

    # Seed Data definitions from config (one row per run/column/stat).
    cfg_by_name: dict[str, dict] = {}
    for c in cfg_cols:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if name:
            cfg_by_name[name] = c

    for run_name in runs:
        for name in col_names:
            c = cfg_by_name.get(name) or {}
            units = str(c.get("units") or "").strip()
            rmin = c.get("range_min")
            rmax = c.get("range_max")
            for stat in stats:
                ws_data.append(
                    [
                        name,  # Term (display)
                        name,  # Header (TD column)
                        "",  # GroupAfter
                        stat,  # Table Label (TD stat)
                        "number",  # ValueType
                        str(run_name),  # Data Group (TD run/sheet)
                        f"{run_name}.{name}.{stat}",  # Term Label (stable key)
                        units,
                        ("" if rmin is None else rmin),
                        ("" if rmax is None else rmax),
                    ]
                    + [""] * len(sn_cols)
                )
        # visual separator between runs
        ws_data.append([""] * len(data_headers))

    # Data_calc is computed/debug: metric keys + serial columns.
    ws_data_calc = wb.create_sheet("Data_calc")
    ws_data_calc.append(["Metric"] + sn_cols)
    try:
        ws_data_calc.freeze_panes = "B2"
    except Exception:
        pass

    # Rows are grouped by run/sheet, then by configured columns/statistics.
    try:
        from openpyxl.styles import Font  # type: ignore
    except Exception:
        Font = None  # type: ignore
    bold = Font(bold=True) if Font is not None else None
    for run_name in runs:
        # run group header row
        ws_data_calc.append([str(run_name)] + [""] * len(sn_cols))
        if bold is not None:
            try:
                ws_data_calc.cell(row=ws_data_calc.max_row, column=1).font = bold
            except Exception:
                pass
        for name in col_names:
            for stat in stats:
                ws_data_calc.append([f"{run_name}.{name}.{stat}"] + [""] * len(sn_cols))
        ws_data_calc.append([""] + [""] * len(sn_cols))

    ws_cfg = wb.create_sheet("Config")
    ws_cfg.append(["Key", "Value"])
    for k in ("description", "data_group", "sheet_name", "header_row"):
        ws_cfg.append([k, json.dumps(config.get(k), ensure_ascii=False)])
    ws_cfg.append(["statistics", ", ".join(stats)])
    ws_cfg.append(["", ""])
    ws_cfg.append(["Columns", ""])
    ws_cfg.append(["name", "units", "range_min", "range_max"])
    for c in cfg_cols:
        if not isinstance(c, dict):
            continue
        ws_cfg.append(
            [
                str(c.get("name") or "").strip(),
                str(c.get("units") or "").strip(),
                json.dumps(c.get("range_min"), ensure_ascii=False),
                json.dumps(c.get("range_max"), ensure_ascii=False),
            ]
        )

    ws_src = wb.create_sheet("Sources")
    ws_src.append(["serial_number", "program_title", "document_type", "metadata_rel", "artifacts_rel", "excel_sqlite_rel"])
    docs_by_serial: dict[str, list[dict]] = {}
    for d in (docs or []):
        sn = str(d.get("serial_number") or "").strip()
        if sn:
            docs_by_serial.setdefault(sn, []).append(d)
    support_dir = eidat_support_dir(repo) if repo is not None else None
    for sn in sn_cols:
        doc = (
            _best_doc_for_serial(support_dir, docs_by_serial.get(sn, []))
            if support_dir is not None
            else (docs_by_serial.get(sn, [{}])[0] if docs_by_serial.get(sn) else {})
        )
        ws_src.append(
            [
                str(sn or "").strip(),
                str(doc.get("program_title") or "").strip(),
                str(doc.get("document_type") or "").strip(),
                str(doc.get("metadata_rel") or "").strip(),
                str(doc.get("artifacts_rel") or "").strip(),
                str(doc.get("excel_sqlite_rel") or "").strip(),
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    try:
        wb.close()
    except Exception:
        pass


def _write_eidp_trending_workbook(
    path: Path,
    *,
    serials: list[str],
    docs: list[dict] | None = None,
    support_dir: Path | None = None,
) -> None:
    """
    Create a blank EIDP trending workbook with header row and serial columns.

    The workbook has two population modes:
    - Blank: Just headers and serial columns, user adds terms manually
    - Auto-populate: Terms and values filled from extracted acceptance data

    Both modes include a 'metadata' sheet with document metadata for each serial.
    """
    try:
        from openpyxl import Workbook  # type: ignore
        from openpyxl.worksheet.datavalidation import DataValidation  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to create the EIDP Trending workbook. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    headers = ["Term", "Header", "GroupAfter", "Table Label", "ValueType", "Data Group", "Term Label", "Units", "Min", "Max"]
    headers.extend([str(s) for s in serials if str(s).strip()])
    wb = Workbook()
    ws = wb.active
    ws.title = "master"
    ws.append(headers)

    # ValueType dropdown to prevent invalid inputs (blank = auto).
    try:
        value_type_col = headers.index("ValueType") + 1  # 1-based
        max_rows = 500  # Reasonable limit for manual entry
        dv = DataValidation(type="list", formula1='"auto,string,number"', allow_blank=True, showDropDown=True)
        ws.add_data_validation(dv)
        col_letter = get_column_letter(int(value_type_col))
        dv.add(f"{col_letter}2:{col_letter}{int(max_rows)}")
    except Exception:
        pass

    # Add metadata rows to master sheet.
    # These appear as the first rows before any term data (both manual + auto-populate flows).
    if docs:
        # Build serial -> best doc mapping (use newest metadata JSON when available).
        docs_by_serial: dict[str, list[dict]] = {}
        for d in docs:
            sn = str(d.get("serial_number") or "").strip()
            if sn:
                docs_by_serial.setdefault(sn, []).append(d)

        # Metadata fields to add as rows in master sheet
        # Format: (term_name, doc_field, term_label)
        metadata_fields = [
            ("Program", "program_title", "Program"),
            ("Asset Type", "asset_type", "Asset Type"),
            ("Asset Specific Type", "asset_specific_type", "Asset Specific Type"),
            ("Vendor", "vendor", "Vendor"),
            ("Acceptance Test Plan", "acceptance_test_plan_number", "Acceptance Test Plan"),
            ("Part Number", "part_number", "Part Number"),
            ("Revision", "revision", "Revision Letter"),
            ("Test Date", "test_date", "Test Date"),
            ("Report Date", "report_date", "Report Date"),
            ("Document Type", "document_type", "Document Type"),
            ("Document Acronym", "document_type_acronym", "Document Acronym"),
            ("Similarity Group", "similarity_group", "Similarity Group"),
        ]

        for term_name, doc_field, term_label in metadata_fields:
            # Build row: Term, Header, GroupAfter, Table Label, ValueType, Data Group, Term Label, Units, Min, Max, then serial values
            row = [term_name, "", "", "", "string", "Metadata", term_label, "", "", ""]
            for sn in serials:
                sn_clean = str(sn).strip()
                doc = (
                    _best_doc_for_serial(support_dir, docs_by_serial.get(sn_clean, []))
                    if support_dir is not None
                    else (docs_by_serial.get(sn_clean, [{}])[0] if docs_by_serial.get(sn_clean) else {})
                )
                row.append(str(doc.get(doc_field) or ""))
            ws.append(row)

        # Blank separator row between metadata and user data inputs.
        ws.append(["" for _ in headers])

    ws.freeze_panes = "A2"

    # Create metadata sheet with full document info for each serial (for reference)
    ws_meta = wb.create_sheet("metadata")
    meta_headers = ["Serial Number", "Program", "Asset Type", "Part Number", "Revision", "Test Date", "Report Date", "Document Type", "Similarity Group"]
    # Keep metadata sheet ordered similarly to master-sheet metadata rows.
    meta_headers = [
        "Serial Number",
        "Program",
        "Asset Type",
        "Asset Specific Type",
        "Vendor",
        "Acceptance Test Plan",
        "Part Number",
        "Revision",
        "Test Date",
        "Report Date",
        "Document Type",
        "Document Acronym",
        "Similarity Group",
    ]
    ws_meta.append(meta_headers)

    if docs:
        # Populate metadata rows for each serial in workbook order
        for sn in serials:
            sn_clean = str(sn).strip()
            doc = (
                _best_doc_for_serial(support_dir, docs_by_serial.get(sn_clean, []))
                if support_dir is not None
                else (docs_by_serial.get(sn_clean, [{}])[0] if docs_by_serial.get(sn_clean) else {})
            )
            ws_meta.append([
                sn_clean,
                str(doc.get("program_title") or ""),
                str(doc.get("asset_type") or ""),
                str(doc.get("asset_specific_type") or ""),
                str(doc.get("vendor") or ""),
                str(doc.get("acceptance_test_plan_number") or ""),
                str(doc.get("part_number") or ""),
                str(doc.get("revision") or ""),
                str(doc.get("test_date") or ""),
                str(doc.get("report_date") or ""),
                str(doc.get("document_type") or ""),
                str(doc.get("document_type_acronym") or ""),
                str(doc.get("similarity_group") or ""),
            ])

    ws_meta.freeze_panes = "A2"

    # Restrict metadata-driven fields (doc type/acronym, etc.) to dropdown lists.
    try:
        meta_rows: dict[str, int] = {}
        col_term = headers.index("Term") + 1
        col_data_group = headers.index("Data Group") + 1
        for row in range(2, ws.max_row + 1):
            term = str(ws.cell(row, col_term).value or "").strip()
            if not term:
                continue
            dg = str(ws.cell(row, col_data_group).value or "").strip()
            if _normalize_text(dg) != _normalize_text("Metadata"):
                continue
            if term not in meta_rows:
                meta_rows[term] = row

        fixed_cols = headers.index("Max") + 1
        sn_count = len([s for s in serials if str(s).strip()])
        if sn_count > 0:
            _ensure_project_metadata_validations(
                wb,
                ws_master=ws,
                ws_metadata=ws_meta,
                docs=(docs or []),
                extra_meta=None,
                master_meta_rows=meta_rows,
                master_sn_col_range=(fixed_cols + 1, fixed_cols + sn_count),
            )
    except Exception:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    try:
        wb.close()
    except Exception:
        pass


def _write_eidp_raw_trending_workbook(
    path: Path,
    *,
    serials: list[str],
    docs: list[dict] | None = None,
) -> None:
    """
    Create a blank EIDP raw file trending workbook.

    This workbook is intentionally minimal and is filled by combined.txt extraction.
    """
    try:
        from openpyxl import Workbook  # type: ignore
        from openpyxl.worksheet.datavalidation import DataValidation  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to create the EIDP Raw File Trending workbook. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    headers = ["Term", "Header", "GroupAfter", "Table Label", "ValueType", "Data Group", "Term Label", "Units", "Min", "Max"]
    headers.extend([str(s) for s in serials if str(s).strip()])

    wb = Workbook()
    ws = wb.active
    ws.title = "master"
    ws.append(headers)

    # Value Type dropdown to prevent invalid inputs (blank = auto).
    try:
        value_type_col = headers.index("ValueType") + 1  # 1-based
        max_rows = 1000
        dv = DataValidation(type="list", formula1='"auto,string,number"', allow_blank=True, showDropDown=True)
        ws.add_data_validation(dv)
        col_letter = get_column_letter(int(value_type_col))
        dv.add(f"{col_letter}2:{col_letter}{int(max_rows)}")
    except Exception:
        pass

    # Add metadata rows at the top of the sheet (like EIDP Trending).
    if docs:
        docs_by_serial: dict[str, dict] = {}
        for d in docs:
            sn = str(d.get("serial_number") or "").strip()
            if sn and sn not in docs_by_serial:
                docs_by_serial[sn] = d

        metadata_fields = [
            ("Program", "program_title", "Program"),
            ("Asset Type", "asset_type", "Asset Type"),
            ("Asset Specific Type", "asset_specific_type", "Asset Specific Type"),
            ("Vendor", "vendor", "Vendor"),
            ("Acceptance Test Plan", "acceptance_test_plan_number", "Acceptance Test Plan"),
            ("Part Number", "part_number", "Part Number"),
            ("Revision", "revision", "Revision Letter"),
            ("Test Date", "test_date", "Test Date"),
            ("Report Date", "report_date", "Report Date"),
            ("Document Type", "document_type", "Document Type"),
            ("Document Acronym", "document_type_acronym", "Document Acronym"),
            ("Similarity Group", "similarity_group", "Similarity Group"),
        ]

        for term_name, doc_field, term_label in metadata_fields:
            row = [term_name, "", "", "", "string", "Metadata", term_label, "", "", ""]
            for sn in serials:
                sn_clean = str(sn).strip()
                doc = docs_by_serial.get(sn_clean, {})
                row.append(str(doc.get(doc_field) or ""))
            ws.append(row)

        # Blank separator row between metadata and user data inputs.
        ws.append(["" for _ in headers])

    ws.freeze_panes = "A2"

    # Restrict metadata-driven fields (doc type/acronym, etc.) to dropdown lists.
    try:
        meta_rows: dict[str, int] = {}
        col_term = headers.index("Term") + 1
        col_data_group = headers.index("Data Group") + 1
        for row in range(2, ws.max_row + 1):
            term = str(ws.cell(row, col_term).value or "").strip()
            if not term:
                continue
            dg = str(ws.cell(row, col_data_group).value or "").strip()
            if _normalize_text(dg) != _normalize_text("Metadata"):
                continue
            if term not in meta_rows:
                meta_rows[term] = row

        fixed_cols = headers.index("Max") + 1
        sn_count = len([s for s in serials if str(s).strip()])
        if sn_count > 0:
            _ensure_project_metadata_validations(
                wb,
                ws_master=ws,
                ws_metadata=None,
                docs=(docs or []),
                extra_meta=None,
                master_meta_rows=meta_rows,
                master_sn_col_range=(fixed_cols + 1, fixed_cols + sn_count),
            )
    except Exception:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    try:
        wb.close()
    except Exception:
        pass


def _extract_acceptance_data_for_serial(
    support_dir: Path,
    artifacts_rel: str,
) -> dict[str, list[dict]]:
    """
    Extract acceptance test data from extracted_terms.db (preferred) or combined.txt (fallback).

    Returns dict of term -> list[{value, units, requirement_type, min, max, result, computed_pass}]
    """
    try:
        from extraction.term_value_extractor import CombinedTxtParser
    except ImportError:
        return {}

    def _merge_term_data(base: dict[str, list[dict]], incoming: dict[str, list[dict]]) -> dict[str, list[dict]]:
        if not incoming:
            return base
        if not base:
            return dict(incoming)
        for term, data_list in incoming.items():
            if term not in base:
                base[term] = list(data_list)
                continue
            target_list = base[term]
            for idx, data in enumerate(data_list):
                if idx >= len(target_list):
                    target_list.append(dict(data))
                    continue
                target = target_list[idx]
                # Fill missing fields from incoming (by instance index).
                for key in (
                    "value",
                    "raw",
                    "units",
                    "requirement_type",
                    "requirement_min",
                    "requirement_max",
                    "requirement_raw",
                    "result",
                    "computed_pass",
                    "description",
                ):
                    if target.get(key) in (None, "", []):
                        if data.get(key) not in (None, "", []):
                            target[key] = data.get(key)
        return base

    def _parse_combined_txt(path: Path) -> dict[str, list[dict]]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}
        heuristics = None
        try:
            if DEFAULT_ACCEPTANCE_HEURISTICS.exists():
                heuristics = json.loads(DEFAULT_ACCEPTANCE_HEURISTICS.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            heuristics = None
        try:
            parser = CombinedTxtParser(content, heuristics=heuristics)
            result = parser.parse()
        except Exception:
            return {}
        term_data: dict[str, list[dict]] = {}
        for test in result.acceptance_tests:
            term_data.setdefault(test.term, []).append({
                "value": test.measured.value,
                "raw": test.measured.raw,
                "units": test.units,
                "requirement_type": test.requirement.type,
                "requirement_min": test.requirement.min_value,
                "requirement_max": test.requirement.max_value,
                "requirement_raw": test.requirement.raw,
                "result": test.result,
                "computed_pass": test.computed_pass,
                "description": test.description,
            })
        return term_data

    art_dir = _resolve_support_path(support_dir, artifacts_rel)
    if not art_dir or not art_dir.exists():
        return {}

    # Preferred: extracted_terms.db produced by term_value_extractor during processing
    db_path = art_dir / "extracted_terms.db"
    if db_path.exists():
        try:
            term_data: dict[str, list[dict]] = {}
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT term, description,
                           requirement_type, requirement_min, requirement_max, requirement_raw,
                           measured_value, measured_raw, units, result, computed_pass
                    FROM acceptance_tests
                    ORDER BY term, page, table_index, id
                    """
                ).fetchall()
            for r in rows:
                term = str(r["term"] or "").strip()
                if not term:
                    continue
                term_data.setdefault(term, []).append({
                    "value": r["measured_value"],
                    "raw": r["measured_raw"],
                    "units": r["units"],
                    "requirement_type": r["requirement_type"],
                    "requirement_min": r["requirement_min"],
                    "requirement_max": r["requirement_max"],
                    "requirement_raw": r["requirement_raw"],
                    "result": r["result"],
                    "computed_pass": (None if r["computed_pass"] is None else bool(int(r["computed_pass"]))),
                    "description": r["description"],
                })
            # Always attempt to enrich with combined.txt (for user-defined or missed terms)
            combined_path = art_dir / "combined.txt"
            if combined_path.exists():
                term_data = _merge_term_data(term_data, _parse_combined_txt(combined_path))
            return term_data
        except Exception:
            # Fall back to parsing combined.txt
            pass

    # If extracted_terms.db is missing but combined.txt exists (older artifacts),
    # try to generate the DB so Trending + certification share a single source.
    combined_path = art_dir / "combined.txt"
    if not combined_path.exists():
        return {}

    if not db_path.exists():
        try:
            from extraction.term_value_extractor import extract_from_combined_txt

            extract_from_combined_txt(combined_path, output_db=db_path, auto_project=False)
        except Exception:
            pass

    if db_path.exists():
        try:
            term_data: dict[str, list[dict]] = {}
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT term, description,
                           requirement_type, requirement_min, requirement_max, requirement_raw,
                           measured_value, measured_raw, units, result, computed_pass
                    FROM acceptance_tests
                    ORDER BY term, page, table_index, id
                    """
                ).fetchall()
            for r in rows:
                term = str(r["term"] or "").strip()
                if not term:
                    continue
                term_data.setdefault(term, []).append({
                    "value": r["measured_value"],
                    "raw": r["measured_raw"],
                    "units": r["units"],
                    "requirement_type": r["requirement_type"],
                    "requirement_min": r["requirement_min"],
                    "requirement_max": r["requirement_max"],
                    "requirement_raw": r["requirement_raw"],
                    "result": r["result"],
                    "computed_pass": (None if r["computed_pass"] is None else bool(int(r["computed_pass"]))),
                    "description": r["description"],
                })
            combined_path = art_dir / "combined.txt"
            if combined_path.exists():
                term_data = _merge_term_data(term_data, _parse_combined_txt(combined_path))
            return term_data
        except Exception:
            pass

    # Last resort: parse combined.txt directly (no DB write access / parsing errors)
    try:
        return _parse_combined_txt(combined_path)
    except Exception:
        return {}


def _populate_workbook_with_acceptance_data(
    workbook_path: Path,
    support_dir: Path,
    docs_by_serial: dict[str, list[dict]],
    discover_terms: bool = True,
) -> dict:
    """
    Populate workbook with extracted acceptance data.

    Returns summary dict with counts of terms and values populated.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl required for auto-trending") from exc

    wb = load_workbook(str(workbook_path))
    ws = wb["master"] if "master" in wb.sheetnames else wb.active

    # Read headers
    header_row = 1
    headers: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        val = ws.cell(header_row, col).value
        if val is not None:
            headers[str(val).strip()] = col

    col_term = headers.get("Term", 1)
    col_units = headers.get("Units", 7)
    col_min = headers.get("Min", 8)
    col_max = headers.get("Max", 9)

    # Find serial columns (after fixed columns)
    fixed_cols = max(col_term, col_units, col_min, col_max, 9)
    sn_cols: dict[str, int] = {}
    for col in range(fixed_cols + 1, ws.max_column + 1):
        name = str(ws.cell(header_row, col).value or "").strip()
        if name:
            sn_cols[name] = col

    # Build existing terms map (row list by term name)
    existing_terms: dict[str, list[int]] = {}
    for row in range(2, ws.max_row + 1):
        term = str(ws.cell(row, col_term).value or "").strip()
        if term:
            base_term, _ = _split_term_instance(term)
            existing_terms.setdefault(base_term, []).append(row)

    # Collect all acceptance data and discovered terms
    all_term_data: dict[str, dict[str, list[dict]]] = {}  # serial -> term -> data list
    discovered_terms: dict[str, list[dict]] = {}  # term -> data list (max length)

    known_serials = set(docs_by_serial.keys())

    for sn_header, col in sn_cols.items():
        sn = _match_serial_key(sn_header, known_serials)
        doc = _best_doc_for_serial(support_dir, docs_by_serial.get(sn, []))
        if not doc:
            continue
        artifacts_rel = str(doc.get("artifacts_rel") or "").strip()
        term_data = _extract_acceptance_data_for_serial(support_dir, artifacts_rel)
        if term_data:
            all_term_data[sn_header] = term_data
            for term, data_list in term_data.items():
                if term not in discovered_terms or len(data_list) > len(discovered_terms.get(term, [])):
                    discovered_terms[term] = list(data_list)

    # Add discovered terms if enabled
    new_terms_added = 0
    if discover_terms:
        next_row = ws.max_row + 1
        for term, data_list in discovered_terms.items():
            existing_rows = existing_terms.get(term, [])
            needed = max(0, len(data_list) - len(existing_rows))
            for idx in range(needed):
                data = data_list[len(existing_rows) + idx] if (len(existing_rows) + idx) < len(data_list) else {}
                # Add new term row
                ws.cell(next_row, col_term).value = term
                if data.get("units"):
                    ws.cell(next_row, col_units).value = data["units"]
                if data.get("requirement_min") is not None:
                    ws.cell(next_row, col_min).value = data["requirement_min"]
                if data.get("requirement_max") is not None:
                    ws.cell(next_row, col_max).value = data["requirement_max"]
                existing_terms.setdefault(term, []).append(next_row)
                next_row += 1
                new_terms_added += 1

    # Populate values
    values_populated = 0
    for sn_header, term_data in all_term_data.items():
        col = sn_cols.get(sn_header)
        if not col:
            continue
        for term, data_list in term_data.items():
            rows = existing_terms.get(term, [])
            if not rows:
                continue
            for idx, data in enumerate(data_list):
                if idx >= len(rows):
                    break
                if data.get("value") is not None:
                    ws.cell(rows[idx], col).value = data["value"]
                    values_populated += 1

    wb.save(str(workbook_path))
    try:
        wb.close()
    except Exception:
        pass

    return {
        "new_terms_added": new_terms_added,
        "values_populated": values_populated,
        "serials_processed": len(all_term_data),
    }


def create_eidat_project(
    global_repo: Path,
    *,
    project_parent_dir: Path,
    project_name: str,
    project_type: str,
    selected_metadata_rel: list[str],
    continued_population: Mapping[str, object] | None = None,
    auto_populate: bool = False,
) -> dict:
    """
    Create a project folder + workbook inside the Global Repo and register it.

    Term Population Options (two clear choices):
    - auto_populate=False: Blank workbook with header row and serial columns only.
                           User manually adds terms and values.
    - auto_populate=True:  Automatically extracts terms and values from combined.txt
                           for each EIDP. Only terms found in extractions are added.
    """
    repo = Path(global_repo).expanduser()
    _ensure_projects_writable(repo)
    safe_name = _safe_project_slug(project_name)
    parent = resolve_path_within_global_repo(repo, project_parent_dir, "Project location")
    project_dir = resolve_path_within_global_repo(repo, parent / safe_name, "Project folder")

    if project_dir.exists():
        raise FileExistsError(f"Project folder already exists: {project_dir}")

    docs = read_eidat_index_documents(repo)
    selected = {str(s).strip() for s in (selected_metadata_rel or []) if str(s).strip()}
    chosen_docs = [d for d in docs if str(d.get("metadata_rel") or "").strip() in selected]
    continued_population_clean = _sanitize_continued_population(continued_population)

    def _is_test_data_eligible(d: dict) -> bool:
        if not is_test_data_doc(d):
            return False
        try:
            sqlite_rel = str(d.get("excel_sqlite_rel") or "").strip()
        except Exception:
            sqlite_rel = ""
        if sqlite_rel:
            return True
        try:
            art = str(d.get("artifacts_rel") or "").strip().lower()
        except Exception:
            art = ""
        return "__excel" in art

    if project_type in (EIDAT_PROJECT_TYPE_TRENDING, EIDAT_PROJECT_TYPE_RAW_TRENDING):
        bad = [d for d in chosen_docs if isinstance(d, dict) and is_test_data_doc(d)]
        if bad:
            bad_sn = sorted({str(d.get("serial_number") or "").strip() for d in bad if str(d.get("serial_number") or "").strip()})
            raise RuntimeError(
                "TD reports cannot be used to create an EIDP Trending project. "
                "Create a Test Data Trending project instead. "
                + (f"(TD serials selected: {', '.join(bad_sn[:12])})" if bad_sn else "")
            )
    if project_type == EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING:
        bad = [d for d in chosen_docs if isinstance(d, dict) and not _is_test_data_eligible(d)]
        if bad:
            bad_sn = sorted({str(d.get("serial_number") or "").strip() for d in bad if str(d.get("serial_number") or "").strip()})
            raise RuntimeError(
                "Test Data Trending projects must be created from TD reports with extracted Excel data. "
                + (f"(Invalid serials selected: {', '.join(bad_sn[:12])})" if bad_sn else "")
            )

    serials = sorted({str(d.get("serial_number") or "").strip() for d in chosen_docs if str(d.get("serial_number") or "").strip()})
    if not serials:
        raise RuntimeError("Selected EIDPs have no serial numbers in the index. Re-run indexing or choose different items.")

    project_dir.mkdir(parents=True, exist_ok=False)
    workbook_path = project_dir / f"{safe_name}.xlsx"

    auto_populate_result = None
    terms_count = 0
    excel_trend_config_path = ""
    config_snapshot: dict | None = None

    if project_type == EIDAT_PROJECT_TYPE_TRENDING:
        # Create workbook with header row, serial columns, and metadata sheet.
        # Metadata sheet is always populated regardless of auto_populate setting.
        _ensure_file_written(
            workbook_path,
            create_fn=lambda: _write_eidp_trending_workbook(
                workbook_path,
                serials=serials,
                docs=chosen_docs,
                support_dir=eidat_support_dir(repo),
            ),
        )

        # Best-effort: build a project-local SQLite cache for Implementation/Trending.
        try:
            ensure_trending_project_sqlite(project_dir, workbook_path)
        except Exception:
            pass

        # Auto-populate: extract terms and values from combined.txt
        if auto_populate:
            try:
                support_dir = eidat_support_dir(repo)
                docs_by_serial: dict[str, list[dict]] = {}
                for d in docs:
                    serial = str(d.get("serial_number") or "").strip()
                    if serial:
                        docs_by_serial.setdefault(serial, []).append(d)
                auto_populate_result = _populate_workbook_with_acceptance_data(
                    workbook_path, support_dir, docs_by_serial, discover_terms=True
                )
                terms_count = auto_populate_result.get("new_terms_added", 0)
            except Exception as exc:
                auto_populate_result = {"error": str(exc)}

    elif project_type == EIDAT_PROJECT_TYPE_RAW_TRENDING:
        # Raw trending: always start blank (combined.txt-only extraction)
        auto_populate = False
        _ensure_file_written(
            workbook_path,
            create_fn=lambda: _write_eidp_raw_trending_workbook(
                workbook_path, serials=serials, docs=chosen_docs
            ),
        )

    elif project_type == EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING:
        # Test Data trending: config-driven skeleton workbook (no auto-populate yet).
        auto_populate = False
        cfg_path = CENTRAL_EXCEL_TREND_CONFIG
        cfg = _load_excel_trend_config(cfg_path)
        excel_trend_config_path = str(cfg_path)
        config_snapshot = cfg
        _ensure_file_written(
            workbook_path,
            create_fn=lambda: _write_test_data_trending_workbook(
                workbook_path,
                global_repo=repo,
                serials=serials,
                docs=chosen_docs,
                config=cfg,
            ),
        )
        # Create/populate the project-local cache DB at project creation time so
        # Trend/Analyze can plot immediately from `implementation_trending.sqlite3`.
        ensure_test_data_project_cache(project_dir, workbook_path, rebuild=True)

    else:
        raise RuntimeError(f"Unsupported project type: {project_type}")

    meta = {
        "name": safe_name,
        "type": project_type,
        "global_repo": str(repo),
        "project_dir": str(project_dir),
        "workbook": str(workbook_path),
        "selected_count": len(chosen_docs),
        "serials_count": len(serials),
        "terms_count": terms_count,
        "selected_metadata_rel": sorted(selected),
        "serials": serials,
        "continued_population": continued_population_clean,
        "auto_populate": auto_populate_result,
        "population_mode": "auto" if auto_populate else "blank",
    }
    if excel_trend_config_path:
        meta["excel_trend_config_path"] = excel_trend_config_path
    if isinstance(config_snapshot, dict):
        meta["config_snapshot"] = config_snapshot
    description = _format_continued_population_description(continued_population_clean)
    if description:
        meta["description"] = description
    meta_path = project_dir / EIDAT_PROJECT_META
    _ensure_file_written(
        meta_path,
        create_fn=lambda: meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8"),
    )
    try:
        _mirror_project_to_local(project_dir, project_name=safe_name)
    except Exception:
        pass

    # Registry entry (stored under EIDAT Support/projects as SQLite for multi-writer safety)
    _ensure_projects_writable(repo)
    try:
        rel = str(project_dir.resolve().relative_to(repo.resolve()))
    except Exception:
        rel = str(project_dir)
    wb_rel = str((Path(rel) / workbook_path.name)) if rel else str(workbook_path)
    db_path = _registry_db_path(repo)
    conn = _connect_projects_registry(db_path)
    try:
        _ensure_projects_registry_schema(conn)
        now_ns = __import__("time").time_ns()
        created_by = (os.environ.get("USERNAME") or os.environ.get("USER") or "").strip() or None
        conn.execute(
            """
            INSERT INTO projects(name, type, project_dir, workbook, created_by, created_epoch_ns, updated_epoch_ns)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, type) DO UPDATE SET
              project_dir=excluded.project_dir,
              workbook=excluded.workbook,
              updated_epoch_ns=excluded.updated_epoch_ns
            """,
            (safe_name, project_type, rel, wb_rel, created_by, now_ns, now_ns),
        )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return meta


# =============================================================================
# Certification Analysis Functions
# =============================================================================


def analyze_and_certify_document(global_repo: Path, artifacts_rel: str) -> dict:
    """
    Run certification analysis on a single document.

    Args:
        global_repo: Path to the global repository
        artifacts_rel: Relative path to artifacts folder (e.g., "debug/ocr/DOC_NAME")

    Returns:
        Dict with certification result:
            - status: CertificationStatus value string
            - total_tests: int
            - passed_tests: int
            - failed_tests: int
            - pending_tests: int
            - pass_rate: str (e.g., "5/7")
            - failed_terms: list[str]
    """
    repo = Path(global_repo).expanduser()

    # Check multiple possible support directory locations
    possible_dirs = [
        eidat_support_dir(repo) / artifacts_rel,           # EIDAT Support/
        repo / GLOBAL_RUN_MIRROR_DIRNAME / artifacts_rel,  # global_run_mirror/
        repo / artifacts_rel,                              # repo itself might be the support dir
    ]

    artifacts_dir = None
    for candidate in possible_dirs:
        if candidate.exists():
            artifacts_dir = candidate
            break

    if artifacts_dir is None:
        return {"status": "ERROR", "error": f"Artifacts folder not found"}

    # Add EIDAT_App_Files to path for extraction imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction.certification_analyzer import analyze_artifacts_folder
        result = analyze_artifacts_folder(artifacts_dir)
        if result is None:
            return {"status": "NO_DATA", "error": "No extracted_terms.db found"}
        return {
            "status": result.status.value,
            "total_tests": result.total_tests,
            "passed_tests": result.passed_tests,
            "failed_tests": result.failed_tests,
            "pending_tests": result.pending_tests,
            "pass_rate": result.pass_rate,
            "failed_terms": result.failed_terms,
            "pending_terms": result.pending_terms,
        }
    except Exception as exc:
        return {"status": "ERROR", "error": str(exc)}


def analyze_and_certify_all(global_repo: Path) -> dict:
    """
    Run certification analysis on all processed documents.

    Args:
        global_repo: Path to the global repository

    Returns:
        Dict with:
            - analyzed_count: int
            - certified_count: int
            - failed_count: int
            - pending_count: int
            - results: list[dict]  # Per-document results
    """
    repo = Path(global_repo).expanduser()

    # Add EIDAT_App_Files to path for extraction imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction.certification_analyzer import analyze_all_in_support_dir

        # Check multiple possible support directory locations
        possible_dirs = [
            eidat_support_dir(repo),           # EIDAT Support/
            repo / GLOBAL_RUN_MIRROR_DIRNAME,  # global_run_mirror/
            repo,                              # repo itself might be the support dir
        ]

        results = {}
        for support_dir in possible_dirs:
            ocr_dir = support_dir / "debug" / "ocr"
            if ocr_dir.exists():
                dir_results = analyze_all_in_support_dir(support_dir)
                results.update(dir_results)

        certified_count = sum(1 for r in results.values() if r.status.value == "CERTIFIED")
        failed_count = sum(1 for r in results.values() if r.status.value == "FAILED")
        pending_count = sum(1 for r in results.values() if r.status.value == "PENDING")

        return {
            "analyzed_count": len(results),
            "certified_count": certified_count,
            "failed_count": failed_count,
            "pending_count": pending_count,
            "results": [
                {
                    "artifacts_rel": art_rel,
                    "status": r.status.value,
                    "pass_rate": r.pass_rate,
                }
                for art_rel, r in results.items()
            ],
        }
    except Exception as exc:
        return {"analyzed_count": 0, "error": str(exc)}
