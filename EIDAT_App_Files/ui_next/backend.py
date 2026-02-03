from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import shutil
import subprocess
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT.parent  # repository root that holds user data folders
MASTER_DB_ROOT = ROOT / "Master_Database"
LEGACY_MASTER_ROOT = ROOT  # previous root-based location for master/registry
DEFAULT_TERMS_XLSX = ROOT / "user_inputs" / "terms.schema.simple.xlsx"
DEFAULT_PLOT_TERMS_XLSX = ROOT / "user_inputs" / "plot_terms.xlsx"
DEFAULT_PROPOSED_PLOTS_JSON = ROOT / "user_inputs" / "proposed_plots.json"
DEFAULT_EXCEL_TREND_CONFIG = ROOT / "user_inputs" / "excel_trend_config.json"
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
EXCEL_ARTIFACT_SUFFIX = "__excel"
# Default repository root where PDFs may live (user-organized, nested or flat)
DEFAULT_REPO_ROOT = ROOT / "Data Packages"
DEFAULT_PDF_DIR = DEFAULT_REPO_ROOT
SCANNER_ENV = ROOT / "user_inputs" / "scanner.env"
OCR_FORCE_ENV = ROOT / "user_inputs" / "ocr_force.env"
DOTENV_FILES = [
    ROOT / ".env",
    ROOT / "user_inputs" / ".env",
]
EIDAT_MANAGER_ENTRY = APP_ROOT / "Application" / "eidat_manager.py"
RUNS_DIR = ROOT / "run_data_simple"
PLOTS_DIR = ROOT / "plots"
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
REPO_NAME_FILE = ROOT / "user_inputs" / "repo_root_name.txt"


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
    expected = ensure_repo_root_name_file()
    if expected and expected.lower() != ROOT.name.lower():
        raise RuntimeError(
            f"Repo root mismatch: expected folder '{expected}' but running under '{ROOT.name}'. "
            f"Update {REPO_NAME_FILE} to match the folder containing EIDAT_App_Files."
        )


def _path_is_within_root(path: Path, root: Path = ROOT) -> bool:
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
            p = ROOT / p
        return _path_is_within_root(p, ROOT)
    except Exception:
        return False


def resolve_repo_path(path: Path, label: str = "Path") -> Path:
    _validate_repo_root_name()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not _path_is_within_root(p, ROOT):
        raise RuntimeError(f"{label} must be inside repo root: {ROOT}")
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


def save_scanner_env(env_map: Dict[str, str], path: Path = SCANNER_ENV) -> None:
    lines = [
        "# Scanner configuration (KEY=VALUE)",
        "# Edited via new GUI",
    ]
    order = [
        "QUIET",
        "REPO_ROOT",
        "OCR_MODE",
        "OCR_DPI",
        "EASYOCR_LANGS",
        "FORCE_OCR",
        "USE_EASYOCR_XY",
        "XY_LOG",
        "OCR_ROW_EPS",
        "VENV_DIR",
    ]
    written = set()
    for k in order:
        v = env_map.get(k)
        if v:
            lines.append(f"{k}={v}")
            written.add(k)
    for k in sorted(env_map.keys()):
        if k in written:
            continue
        v = env_map[k]
        if v:
            lines.append(f"{k}={v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_repo_root() -> Path:
    """Return repository root from scanner.env or DEFAULT_REPO_ROOT."""
    ensure_repo_root_name_file()
    env = parse_scanner_env(SCANNER_ENV)
    val = env.get("REPO_ROOT", "").strip()
    try:
        if val:
            p = Path(val).expanduser()
            if not p.is_absolute():
                p = ROOT / p
            return p
    except Exception:
        pass
    return DEFAULT_REPO_ROOT


def set_repo_root(p: Path) -> None:
    _validate_repo_root_name()
    target = Path(p).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    env = parse_scanner_env(SCANNER_ENV)
    try:
        rel = target.resolve().relative_to(ROOT.resolve())
        env["REPO_ROOT"] = str(rel)
    except Exception:
        env["REPO_ROOT"] = str(target)
    save_scanner_env(env)


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
    env = parse_scanner_env(SCANNER_ENV)
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
    # Merge scanner.env values for direct Python invocations
    env.update(parse_scanner_env(SCANNER_ENV))
    # Merge optional .env overrides (project-level, after scanner.env)
    for path in DOTENV_FILES:
        env.update(parse_scanner_env(path))
    # Merge OCR force overrides (if present)
    env.update(parse_scanner_env(OCR_FORCE_ENV))
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
    # Force cache to always be in project root, not executable location
    # This ensures cache is consistent whether running as script or frozen exe
    if "CACHE_ROOT" not in env and "OCR_CACHE_ROOT" not in env:
        env["CACHE_ROOT"] = str(ROOT)
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
    """Run Excel extraction on all .xlsx files under the provided directory."""
    cfg = Path(config) if config else DEFAULT_EXCEL_TREND_CONFIG
    data_root = resolve_pdf_root(data_dir)
    excels = [p for p in data_root.rglob("*.xlsx") if p.is_file()
              and not p.name.startswith("~$")]
    if not excels:
        raise RuntimeError(f"No Excel data files found under: {data_root}")
    return run_excel_extraction(excels, cfg, data_root)


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
    (ROOT / "user_inputs").mkdir(parents=True, exist_ok=True)
    DEFAULT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    # DISABLED: run_data_simple and Master_Database are no longer used
    # RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # MASTER_DB_ROOT.mkdir(parents=True, exist_ok=True)
    if not SCANNER_ENV.exists():
        save_scanner_env({"QUIET": "1"})


# --- Health checks / analysis helpers ---

def check_environment() -> subprocess.Popen:
    """Spawn a short Python check that imports key packages and reports status.

    Returns a process whose stdout can be streamed for UI feedback.
    """
    py = resolve_project_python()
    code = (
        "import sys;\n"
        "print('[INFO] Checking environment for EIDAT...');\n"
        "mods=['fitz','pandas','openpyxl','matplotlib','easyocr'];\n"
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
EIDAT_PROJECTS_REGISTRY = "projects.json"
EIDAT_PROJECT_META = "project.json"
EIDAT_PROJECT_TYPE_TRENDING = "EIDP Trending"
GLOBAL_RUN_MIRROR_DIRNAME = "global_run_mirror"
LOCAL_PROJECTS_MIRROR_DIRNAME = "projects"
PROJECT_UPDATE_DEBUG_JSON = "update_debug.json"


def eidat_support_dir(global_repo: Path) -> Path:
    return Path(global_repo).expanduser() / "EIDAT Support"


def eidat_projects_root(global_repo: Path) -> Path:
    return eidat_support_dir(global_repo) / EIDAT_PROJECTS_DIRNAME


def eidat_debug_ocr_root(global_repo: Path) -> Path:
    return eidat_support_dir(global_repo) / "debug" / "ocr"


def global_run_mirror_root() -> Path:
    """Repo-local root for mirroring Global Repo outputs."""
    return ROOT / GLOBAL_RUN_MIRROR_DIRNAME


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
        rows = conn.execute(
            """
            SELECT
              id, program_title, asset_type, serial_number, part_number,
              revision, test_date, report_date, document_type,
              metadata_rel, artifacts_rel, similarity_group
            FROM documents
            ORDER BY program_title, asset_type, serial_number, metadata_rel
            """
        ).fetchall()
    docs: list[dict] = []
    for r in rows:
        docs.append({k: r[k] for k in r.keys()})
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

    # Read all files
    files_by_path: dict[str, dict] = {}
    if support_db.exists():
        with sqlite3.connect(str(support_db)) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT * FROM files").fetchall():
                files_by_path[r["rel_path"]] = dict(r)

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
        # Try to match file to document via artifacts path
        pdf_stem = Path(rel_path).stem
        matched_doc = None
        for art_rel, doc in docs_by_artifacts.items():
            if pdf_stem in art_rel:
                matched_doc = doc
                break

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
            blocks.append({"heading": current_heading, "rows": rows, "kv": kv})
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
) -> tuple[str | float | None, str]:
    """Return (value, provenance_snippet)."""
    term_n = _normalize_text(term)
    if not term_n:
        return None, ""

    start = 0
    if header_anchor.strip():
        anchor_n = _normalize_text(header_anchor)
        for i, ln in enumerate(lines):
            if anchor_n and anchor_n in _normalize_text(ln):
                start = i
                break

    end = min(len(lines), start + max(50, int(window_lines)))

    if group_after.strip():
        ga_n = _normalize_text(group_after)
        for i in range(start, end):
            if ga_n and ga_n in _normalize_text(lines[i]):
                start = min(i + 1, end)
                break

    for i in range(start, end):
        ln = lines[i]
        ln_n = _normalize_text(ln)
        if term_n not in ln_n:
            continue
        # try to parse a number from the text after the term occurrence
        idx = ln_n.find(term_n)
        tail = ln[idx + len(term_n) :]
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
        tail_s = re.sub(r"^[\s:\-–—]+", "", ln[idx + len(term) :]).strip()
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


_META_TERM_MAP: dict[str, tuple[str, str]] = {
    # term -> (metadata_key_path, value_type)
    "program": ("program_code", "string"),
    "title": ("program_title", "string"),
    "vehicle": ("vehicle_number", "string"),
    "assettype": ("asset_type", "string"),
    "serial": ("serial_number", "serial"),
    "serialnumber": ("serial_number", "serial"),
    "part": ("part_number", "string"),
    "partnumber": ("part_number", "string"),
    "model": ("part_number", "string"),
    "modelnumber": ("part_number", "string"),
    "rev": ("revision", "string"),
    "revision": ("revision", "string"),
    "testdate": ("test_date", "string"),
    "reportdate": ("report_date", "string"),
    "operator": ("operator", "string"),
    "facility": ("facility", "string"),
}


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
) -> tuple[str | None, str]:
    term_k = _normalize_key(term)
    term_label_n = _normalize_text(term_label)
    if not term_k:
        return None, ""

    anchor = _normalize_text(header_anchor).strip()
    ga = _normalize_text(group_after).strip()

    for b in blocks:
        heading = _normalize_text(str(b.get("heading") or ""))
        if anchor and anchor not in heading:
            continue
        # group_after as secondary check: it can be a phrase that occurs in heading too
        if ga and ga not in heading and ga != anchor:
            pass
        kv = b.get("kv") or {}
        if isinstance(kv, dict):
            v = kv.get(term_k)
            if not v and len(term_k) >= 4:
                for kk, vv in kv.items():
                    try:
                        kkn = _normalize_key(kk)
                    except Exception:
                        kkn = str(kk)
                    if not kkn:
                        continue
                    if term_k == kkn or term_k in kkn or kkn in term_k:
                        v = vv
                        break
            if v:
                return str(v), f"{b.get('heading')}: {term} -> {v}"
        rows = b.get("rows") or []
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, list) or not r:
                continue
            left = _normalize_key(r[0])
            if (left == term_k or (len(term_k) >= 4 and (term_k in left or left in term_k))) and len(r) >= 2:
                val = ""
                for c in reversed(r[1:]):
                    if str(c).strip():
                        val = str(c).strip()
                        break
                if val:
                    return val, f"{b.get('heading')}: {r[0]} -> {val}"
    return None, ""


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
    col_data_group = headers["datagroup"]
    col_term_label = headers["termlabel"]
    col_units = headers["units"]
    col_min = headers["min"]
    col_max = headers["max"]

    fixed_cols = max(col_term, col_header, col_group_after, col_data_group, col_term_label, col_units, col_min, col_max)
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

    # Cache combined.txt lines per SN
    combined_cache: dict[str, list[str]] = {}
    tables_cache: dict[str, list[dict]] = {}
    meta_cache: dict[str, dict] = {}
    artifacts_used: dict[str, str] = {}
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

    updated_cells = 0
    skipped_existing = 0
    missing_source = 0
    missing_value = 0

    # Optional explicit type column
    col_value_type = headers.get("valuetype") or headers.get("value type") or headers.get("value_type")

    for row in range(2, ws.max_row + 1):
        term = str(ws.cell(row, col_term).value or "").strip()
        if not term:
            continue
        header_anchor = str(ws.cell(row, col_header).value or "").strip()
        group_after = str(ws.cell(row, col_group_after).value or "").strip()
        data_group = str(ws.cell(row, col_data_group).value or "").strip()
        term_label = str(ws.cell(row, col_term_label).value or "").strip()
        units = str(ws.cell(row, col_units).value or "").strip()
        mn = _parse_float_loose(ws.cell(row, col_min).value)
        mx = _parse_float_loose(ws.cell(row, col_max).value)
        explicit_type = str(ws.cell(row, col_value_type).value or "").strip() if col_value_type else ""

        # Use Data Group as anchor if Header is generic like "Value"
        if _normalize_text(header_anchor) in ("value", "val", "result"):
            header_anchor = data_group or header_anchor

        want_type = _infer_value_type(term, explicit=explicit_type, units=units, mn=mn, mx=mx)
        term_k = _normalize_key(term)

        for sn_header, col in sn_cols.items():
            cell = ws.cell(row, col)
            if cell.value not in (None, "") and not overwrite:
                skipped_existing += 1
                continue

            meta = meta_cache.get(sn_header) or {}
            used_method = ""
            snippet = ""
            val: object | None = None
            err: str | None = None

            # 1) Metadata-first for known document-profile fields.
            if term_k in _META_TERM_MAP and isinstance(meta, Mapping):
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

            # 2) ASCII table extraction from combined.txt (key/value and general tables)
            if val in (None, ""):
                blocks = tables_cache.get(sn_header) or []
                tval, tsnip = _extract_from_tables(
                    blocks,
                    term=term,
                    term_label=term_label,
                    header_anchor=header_anchor,
                    group_after=group_after,
                )
                if tval:
                    val = _clean_cell_text(tval)
                    used_method = "table"
                    snippet = tsnip

            # 2b) Paired acceptance/measured tables (common in synthetic debug PDFs)
            if val in (None, ""):
                blocks = tables_cache.get(sn_header) or []
                tval, tsnip = _extract_from_paired_acceptance_tables(blocks, term=term, term_label=term_label)
                if tval:
                    val = _clean_cell_text(tval)
                    used_method = "paired_tables"
                    snippet = tsnip

            # 3) Fallback: line-based scan
            if val in (None, ""):
                lines = combined_cache.get(sn_header)
                if not lines:
                    # Optional synthetic golden fallback when combined isn't available.
                    golden = _maybe_load_golden_lines(repo, meta) if isinstance(meta, Mapping) else None
                    if golden:
                        gblocks = _parse_ascii_tables(golden)
                        tval, tsnip = _extract_from_tables(
                            gblocks,
                            term=term,
                            term_label=term_label,
                            header_anchor=header_anchor,
                            group_after=group_after,
                        )
                        if tval:
                            val = _clean_cell_text(tval)
                            used_method = "golden_table"
                            snippet = tsnip
                        else:
                            tval2, tsnip2 = _extract_from_paired_acceptance_tables(gblocks, term=term, term_label=term_label)
                            if tval2:
                                val = _clean_cell_text(tval2)
                                used_method = "golden_paired_tables"
                                snippet = tsnip2
                            fallback, snip = _extract_value_from_lines(
                                golden,
                                term=term,
                                header_anchor=header_anchor,
                                group_after=group_after,
                                window_lines=window_lines,
                                want_type=want_type,
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
                    )
                    if fallback not in (None, ""):
                        val = fallback
                        used_method = "lines"
                        snippet = snip

            if val in (None, ""):
                missing_value += 1
                err = "missing_value"
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
                            "snippet": snippet,
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

    try:
        wb.save(str(wb_path))
    except PermissionError as exc:
        raise RuntimeError(f"Workbook is not writable (close it in Excel first): {wb_path}") from exc
    finally:
        try:
            wb.close()
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
                "window_lines": int(window_lines),
                "updated_cells": int(updated_cells),
                "skipped_existing": int(skipped_existing),
                "missing_source": int(missing_source),
                "missing_value": int(missing_value),
                "serials_in_workbook": int(len(sn_cols)),
                "serials_with_source": int(len(combined_cache)),
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
        "debug_json": debug_json_path,
    }


def _registry_path(global_repo: Path) -> Path:
    return eidat_projects_root(global_repo) / EIDAT_PROJECTS_REGISTRY


def list_eidat_projects(global_repo: Path) -> list[dict]:
    path = _registry_path(global_repo)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for p in data:
        if isinstance(p, dict):
            out.append(p)
    return out


def _write_projects_registry(global_repo: Path, projects: list[dict]) -> None:
    root = eidat_projects_root(global_repo)
    root.mkdir(parents=True, exist_ok=True)
    _registry_path(global_repo).write_text(json.dumps(projects, indent=2), encoding="utf-8")


def delete_eidat_project(global_repo: Path, project_dir: Path) -> dict:
    """Delete a project folder and remove it from the projects registry."""
    repo = Path(global_repo).expanduser()
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

    # Remove registry entry (match by relative project_dir when possible).
    projects = list_eidat_projects(repo)
    try:
        proj_rel = str(proj_dir.resolve().relative_to(repo.resolve()))
    except Exception:
        proj_rel = str(proj_dir)

    kept: list[dict] = []
    removed = 0
    for p in projects:
        try:
            p_dir = str(p.get("project_dir") or "").strip()
        except Exception:
            p_dir = ""
        if p_dir and (p_dir == proj_rel or Path(p_dir).name == proj_dir.name):
            removed += 1
            continue
        kept.append(p)

    # Delete folder (after registry filtering so a partially-deleted folder isn't still listed).
    shutil.rmtree(proj_dir)
    _write_projects_registry(repo, kept)

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


def _write_eidp_trending_workbook(path: Path, *, serials: list[str], docs: list[dict] | None = None) -> None:
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
    except Exception as exc:
        raise RuntimeError(
            "openpyxl is required to create the EIDP Trending workbook. "
            "Install it with `py -m pip install openpyxl` within the project environment."
        ) from exc

    headers = ["Term", "Header", "GroupAfter", "ValueType", "Data Group", "Term Label", "Units", "Min", "Max"]
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
        dv.add(f"{ws.cell(2, value_type_col).coordinate}:{ws.cell(max_rows, value_type_col).coordinate}")
    except Exception:
        pass

    # Add metadata rows to master sheet (Part Number, Revision, Test Date, Report Date)
    # These appear as the first rows before any term data
    if docs:
        # Build serial -> best doc mapping
        docs_by_serial: dict[str, dict] = {}
        for d in docs:
            sn = str(d.get("serial_number") or "").strip()
            if sn and sn not in docs_by_serial:
                docs_by_serial[sn] = d

        # Metadata fields to add as rows in master sheet
        # Format: (term_name, doc_field, term_label)
        metadata_fields = [
            ("Part Number", "part_number", "Part Number"),
            ("Revision", "revision", "Revision Letter"),
            ("Test Date", "test_date", "Test Date"),
            ("Report Date", "report_date", "Report Date"),
        ]

        for term_name, doc_field, term_label in metadata_fields:
            # Build row: Term, Header, GroupAfter, ValueType, Data Group, Term Label, Units, Min, Max, then serial values
            row = [term_name, "", "", "string", "Metadata", term_label, "", "", ""]
            for sn in serials:
                sn_clean = str(sn).strip()
                doc = docs_by_serial.get(sn_clean, {})
                row.append(str(doc.get(doc_field) or ""))
            ws.append(row)

    ws.freeze_panes = "A2"

    # Create metadata sheet with full document info for each serial (for reference)
    ws_meta = wb.create_sheet("metadata")
    meta_headers = ["Serial Number", "Program", "Asset Type", "Part Number", "Revision", "Test Date", "Report Date", "Document Type", "Similarity Group"]
    ws_meta.append(meta_headers)

    if docs:
        # Populate metadata rows for each serial in workbook order
        for sn in serials:
            sn_clean = str(sn).strip()
            doc = docs_by_serial.get(sn_clean, {})
            ws_meta.append([
                sn_clean,
                str(doc.get("program_title") or ""),
                str(doc.get("asset_type") or ""),
                str(doc.get("part_number") or ""),
                str(doc.get("revision") or ""),
                str(doc.get("test_date") or ""),
                str(doc.get("report_date") or ""),
                str(doc.get("document_type") or ""),
                str(doc.get("similarity_group") or ""),
            ])

    ws_meta.freeze_panes = "A2"

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    try:
        wb.close()
    except Exception:
        pass


def _extract_acceptance_data_for_serial(
    support_dir: Path,
    artifacts_rel: str,
) -> dict[str, dict]:
    """
    Extract acceptance test data from combined.txt using term_value_extractor.

    Returns dict of term -> {value, units, requirement_type, min, max, result, computed_pass}
    """
    try:
        from extraction.term_value_extractor import CombinedTxtParser
    except ImportError:
        return {}

    art_dir = _resolve_support_path(support_dir, artifacts_rel)
    if not art_dir or not art_dir.exists():
        return {}

    combined_path = art_dir / "combined.txt"
    if not combined_path.exists():
        return {}

    try:
        content = combined_path.read_text(encoding="utf-8", errors="ignore")
        parser = CombinedTxtParser(content)
        result = parser.parse()

        term_data = {}
        for test in result.acceptance_tests:
            term_data[test.term] = {
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
            }
        return term_data
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

    # Build existing terms map (row by term name)
    existing_terms: dict[str, int] = {}
    for row in range(2, ws.max_row + 1):
        term = str(ws.cell(row, col_term).value or "").strip()
        if term:
            existing_terms[term] = row

    # Collect all acceptance data and discovered terms
    all_term_data: dict[str, dict[str, dict]] = {}  # serial -> term -> data
    discovered_terms: dict[str, dict] = {}  # term -> first occurrence data

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
            for term, data in term_data.items():
                if term not in discovered_terms:
                    discovered_terms[term] = data

    # Add discovered terms if enabled
    new_terms_added = 0
    if discover_terms:
        next_row = ws.max_row + 1
        for term, data in discovered_terms.items():
            if term not in existing_terms:
                # Add new term row
                ws.cell(next_row, col_term).value = term
                if data.get("units"):
                    ws.cell(next_row, col_units).value = data["units"]
                if data.get("requirement_min") is not None:
                    ws.cell(next_row, col_min).value = data["requirement_min"]
                if data.get("requirement_max") is not None:
                    ws.cell(next_row, col_max).value = data["requirement_max"]
                existing_terms[term] = next_row
                next_row += 1
                new_terms_added += 1

    # Populate values
    values_populated = 0
    for sn_header, term_data in all_term_data.items():
        col = sn_cols.get(sn_header)
        if not col:
            continue
        for term, data in term_data.items():
            row = existing_terms.get(term)
            if row and data.get("value") is not None:
                ws.cell(row, col).value = data["value"]
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
    safe_name = _safe_project_slug(project_name)
    parent = resolve_path_within_global_repo(repo, project_parent_dir, "Project location")
    project_dir = resolve_path_within_global_repo(repo, parent / safe_name, "Project folder")

    if project_dir.exists():
        raise FileExistsError(f"Project folder already exists: {project_dir}")

    docs = read_eidat_index_documents(repo)
    selected = {str(s).strip() for s in (selected_metadata_rel or []) if str(s).strip()}
    chosen_docs = [d for d in docs if str(d.get("metadata_rel") or "").strip() in selected]

    serials = sorted({str(d.get("serial_number") or "").strip() for d in chosen_docs if str(d.get("serial_number") or "").strip()})
    if not serials:
        raise RuntimeError("Selected EIDPs have no serial numbers in the index. Re-run indexing or choose different items.")

    project_dir.mkdir(parents=True, exist_ok=False)
    workbook_path = project_dir / f"{safe_name}.xlsx"

    if project_type != EIDAT_PROJECT_TYPE_TRENDING:
        raise RuntimeError(f"Unsupported project type: {project_type}")

    # Create workbook with header row, serial columns, and metadata sheet
    # Metadata sheet is always populated regardless of auto_populate setting
    _write_eidp_trending_workbook(workbook_path, serials=serials, docs=chosen_docs)

    # Auto-populate: extract terms and values from combined.txt
    auto_populate_result = None
    terms_count = 0
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
        "auto_populate": auto_populate_result,
        "population_mode": "auto" if auto_populate else "blank",
    }
    (project_dir / EIDAT_PROJECT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        _mirror_project_to_local(project_dir, project_name=safe_name)
    except Exception:
        pass

    # Registry entry (stored under EIDAT Support to keep project list centralized)
    projects = list_eidat_projects(repo)
    try:
        rel = str(project_dir.resolve().relative_to(repo.resolve()))
    except Exception:
        rel = str(project_dir)
    projects.append(
        {
            "name": safe_name,
            "type": project_type,
            "project_dir": rel,
            "workbook": str((Path(rel) / workbook_path.name)) if rel else str(workbook_path),
        }
    )
    _write_projects_registry(repo, projects)
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
