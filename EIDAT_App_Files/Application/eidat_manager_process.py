from __future__ import annotations

import base64
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import hashlib
import uuid

from eidat_manager_db import SupportPaths, connect_db, ensure_schema


def _fallback_build_pointer_token(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    enc = base64.urlsafe_b64encode(raw).decode("ascii")
    return f"EIDAT_PTR:{enc}"


def _fallback_embed_pointer_token(pdf_path: Path, token: str, *, overwrite: bool = False) -> bool:
    return False


def _fallback_has_pointer_token(pdf_path: Path) -> bool:
    return False


try:
    import eidat_manager_embed as _eidat_manager_embed
except Exception:
    try:
        from . import eidat_manager_embed as _eidat_manager_embed  # type: ignore
    except Exception:
        _eidat_manager_embed = None  # type: ignore[assignment]

if _eidat_manager_embed is not None:
    build_pointer_token = getattr(_eidat_manager_embed, "build_pointer_token", _fallback_build_pointer_token)
    embed_pointer_token = getattr(_eidat_manager_embed, "embed_pointer_token", _fallback_embed_pointer_token)
    has_pointer_token = getattr(_eidat_manager_embed, "has_pointer_token", _fallback_has_pointer_token)
else:
    build_pointer_token = _fallback_build_pointer_token
    embed_pointer_token = _fallback_embed_pointer_token
    has_pointer_token = _fallback_has_pointer_token

from eidat_manager_metadata import (
    canonicalize_metadata_for_file,
    derive_minimal_metadata,
    extract_metadata_from_excel,
    extract_metadata_from_text,
    load_metadata_for_pdf,
    load_metadata_from_artifacts,
    write_metadata,
)
try:
    from eidat_manager_mat_bundle import (  # type: ignore
        MatBundleMember,
        detect_mat_bundle_member,
        list_mat_bundle_members,
        mat_bundle_artifacts_dir,
        mat_bundle_sqlite_path,
    )
except Exception:
    from .eidat_manager_mat_bundle import (  # type: ignore
        MatBundleMember,
        detect_mat_bundle_member,
        list_mat_bundle_members,
        mat_bundle_artifacts_dir,
        mat_bundle_sqlite_path,
    )


@dataclass(frozen=True)
class ProcessResult:
    rel_path: str
    abs_path: str
    ok: bool
    artifacts_dir: str | None = None
    error: str | None = None


def _load_scanner_core() -> Any:
    """OBSOLETE: Load legacy scanner - kept for metadata extraction fallback only."""
    core_path = Path(__file__).resolve().parent / "eidp_term_scanner.core.py"
    spec = importlib.util.spec_from_file_location("eidp_core", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scanner core from {core_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
MAT_EXTENSIONS = {".mat"}
DATA_MATRIX_EXTENSIONS = set(EXCEL_EXTENSIONS) | set(MAT_EXTENSIONS)
EXCEL_ARTIFACT_SUFFIX = "__excel"
TD_SERIAL_AGGREGATES_DIRNAME = "td_serial_sources"
TD_SERIAL_AGGREGATE_METADATA_SOURCE = "td_serial_aggregate"

_TD_AGG_SEQ_SHEET_RE = re.compile(
    r"^\s*seq(?:uence)?[\s_-]*(?=[A-Za-z0-9\s_-]*\d)[A-Za-z0-9]+(?:[\s_-]*[A-Za-z0-9]+)*\s*$",
    flags=re.IGNORECASE,
)
_TD_AGG_SERIAL_RE = re.compile(r"\bSN[-_ ]?[A-Z0-9]+(?:[-_ ][A-Z0-9]+)*\b", flags=re.IGNORECASE)
_TD_AGG_SERIAL_LABEL_RE = re.compile(r"\b(?:serial(?:\s*(?:number|no\.?|#))?|s\s*/?\s*n)\b", flags=re.IGNORECASE)

_IGNORED_REPO_DIRNAMES_CASEFOLD = {
    "eidat",
    "eidat support",
    "edat",
    "edat support",
}


def _ignore_rel_path(rel_path: str) -> bool:
    try:
        parts = [p.casefold() for p in Path(str(rel_path or "")).parts]
    except Exception:
        parts = []
    return any(p in _IGNORED_REPO_DIRNAMES_CASEFOLD for p in parts)


def _export_extracted_terms_db(artifacts_dir: Path, combined_txt_path: Path) -> None:
    """
    Create/refresh `extracted_terms.db` in an artifacts folder from `combined.txt`.

    This enables downstream certification analysis (which expects extracted_terms.db).
    """
    try:
        from extraction.term_value_extractor import extract_from_combined_txt
    except Exception:
        return

    try:
        if not combined_txt_path.exists():
            return
        db_path = artifacts_dir / "extracted_terms.db"
        try:
            db_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            # If we can't delete, try exporting anyway (schema is CREATE IF NOT EXISTS).
            pass

        extract_from_combined_txt(combined_txt_path, output_db=db_path, auto_project=False)
    except Exception:
        # Best-effort; extraction should still succeed without this DB.
        return


def _resolve_table_label_marker() -> str:
    """
    Best-effort resolve the combined.txt label marker from user_inputs/table_label_rules.json.

    If missing/invalid, defaults to "TABLE_LABEL".
    """
    try:
        rules_path = _resolve_user_inputs_file("table_label_rules.json")
    except Exception:
        return "TABLE_LABEL"
    try:
        if not rules_path.exists():
            return "TABLE_LABEL"
        raw = json.loads(rules_path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(raw, dict):
            return "TABLE_LABEL"
        marker = str(raw.get("marker") or "TABLE_LABEL").strip() or "TABLE_LABEL"
        return marker
    except Exception:
        return "TABLE_LABEL"


def _export_labeled_tables_db(artifacts_dir: Path, combined_txt_path: Path) -> None:
    """
    Export `[TABLE_LABEL]`-labeled ASCII tables from the final combined.txt into labeled_tables.db.

    Best-effort: failures should not fail document processing.
    """
    try:
        from extraction.labeled_tables_exporter import export_labeled_tables_db
    except Exception:
        return

    try:
        marker = _resolve_table_label_marker()
    except Exception:
        marker = "TABLE_LABEL"

    try:
        export_labeled_tables_db(
            artifacts_dir=artifacts_dir,
            combined_txt_path=combined_txt_path,
            marker=marker,
            db_name="labeled_tables.db",
        )
    except Exception:
        return


def _load_excel_extractor() -> Any:
    """Load the Excel extraction helper module from scripts/."""
    project_root = Path(__file__).resolve().parent.parent
    mod_path = project_root / "scripts" / "excel_extraction.py"
    spec = importlib.util.spec_from_file_location("excel_extraction", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Excel extractor from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _run_legacy_extraction(pdf_path: Path, dpi: int | None, output_dir: Path) -> dict[str, Any]:
    """
    Legacy extraction pipeline (ExtractionPipeline).

    Returns dict with keys matching legacy scanner interface:
        - "dir": output directory path
        - "combined": path to combined.txt
        - "manifest": path to manifest.json (optional)
    """
    # Add EIDAT_App_Files to path for extraction imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction.batch_processor import ExtractionPipeline
    except ImportError as e:
        raise RuntimeError(f"Unable to import clean extraction pipeline: {e}") from e

    # Create pipeline with specified DPI
    effective_dpi = int(dpi) if dpi else 900
    pipeline = ExtractionPipeline(dpi=effective_dpi, lang="eng", psm=3)

    # Process PDF - writes to output_dir/debug/ocr/{pdf_stem}/
    results = pipeline.process_pdf(
        pdf_path=pdf_path,
        pages=None,  # Process all pages
        output_dir=output_dir,
        verbose=False
    )

    # Build output paths matching legacy interface
    doc_name = pdf_path.stem
    artifacts_dir = output_dir / "debug" / "ocr" / doc_name
    combined_path = artifacts_dir / "combined.txt"
    summary_path = artifacts_dir / "summary.json"

    return {
        "dir": str(artifacts_dir),
        "target_dir": str(artifacts_dir),
        "combined": str(combined_path) if combined_path.exists() else None,
        "manifest": str(summary_path) if summary_path.exists() else None,
        "results": results,
        "pipeline": "extraction_v2.0",
    }


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not str(raw).strip():
        return default
    val = str(raw).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _load_debug_master_config(project_root: Path) -> Dict[str, Any]:
    cfg_path = project_root / "debug_method" / "debug_master_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Debug master config not found: {cfg_path}")
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Unable to parse debug master config: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Debug master config must be a JSON object.")
    return data


def _render_pdf_pages_to_dirs(pdf_path: Path, pages_root: Path, dpi: int) -> int:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF (fitz) is required to render PDF pages.") from exc
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow (PIL) is required to save PNGs with DPI metadata.") from exc

    doc = fitz.open(str(pdf_path))
    try:
        zoom = float(dpi) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page_index in range(doc.page_count):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
            page_num = page_index + 1
            page_dir = pages_root / f"page_{page_num}"
            page_dir.mkdir(parents=True, exist_ok=True)
            out_path = page_dir / f"page_{page_num}.png"
            img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
            img.save(str(out_path), dpi=(float(dpi), float(dpi)))
        return int(doc.page_count)
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _build_debug_method_settings(dpi: int | None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction import ocr_engine
    except ImportError as e:
        raise RuntimeError(f"Unable to import extraction modules: {e}") from e

    config = _load_debug_master_config(project_root)
    render_cfg = config.get("render", {}) if isinstance(config.get("render", {}), dict) else {}
    table_grid_cfg = config.get("table_grid", {}) if isinstance(config.get("table_grid", {}), dict) else {}
    table_variants_cfg = config.get("table_variants", {}) if isinstance(config.get("table_variants", {}), dict) else {}

    def _cfg_int(cfg: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))
        except Exception:
            return default

    def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
        try:
            return float(cfg.get(key, default))
        except Exception:
            return default

    def _cfg_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
        try:
            return bool(cfg.get(key, default))
        except Exception:
            return default

    render_dpi_cfg = _cfg_int(render_cfg, "dpi", 450)
    ocr_dpi_cfg = _cfg_int(table_variants_cfg, "ocr_dpi", 450)
    detection_dpi_cfg = _cfg_int(table_variants_cfg, "detection_dpi", 900)

    env_ocr_dpi = _parse_int_env("OCR_DPI", ocr_dpi_cfg)
    table_ocr_dpi = _parse_int_env("EIDAT_TABLE_OCR_DPI", env_ocr_dpi)
    detection_dpi = _parse_int_env("EIDAT_TABLE_DETECTION_DPI", detection_dpi_cfg)
    render_dpi = _parse_int_env("EIDAT_PAGE_RENDER_DPI", table_ocr_dpi or render_dpi_cfg)
    ocr_dpi = env_ocr_dpi or table_ocr_dpi or ocr_dpi_cfg

    if dpi is not None and int(dpi) > 0:
        ocr_dpi = int(dpi)
        table_ocr_dpi = int(dpi)
        render_dpi = int(dpi)

    if render_dpi <= 0:
        render_dpi = render_dpi_cfg
    if ocr_dpi <= 0:
        ocr_dpi = ocr_dpi_cfg
    if table_ocr_dpi <= 0:
        table_ocr_dpi = ocr_dpi
    if detection_dpi <= 0:
        detection_dpi = detection_dpi_cfg

    table_grid_enabled = _cfg_bool(table_grid_cfg, "enabled", True)
    tg_merge_kx = _cfg_int(table_grid_cfg, "merge_kx", 0)
    tg_min_gap = _cfg_float(table_grid_cfg, "min_gap", 50.0)
    tg_min_gap_ratio = _cfg_float(table_grid_cfg, "min_gap_ratio", 0.0)
    tg_gap_threshold = _cfg_float(table_grid_cfg, "gap_threshold", 0.0)
    tg_offset_px = _cfg_float(table_grid_cfg, "left_offset", 24.0)
    tg_line_thickness = _cfg_int(table_grid_cfg, "line_thickness", 3)
    tg_line_pad = _cfg_float(table_grid_cfg, "line_pad", 0.25)
    tg_min_token_h = _cfg_float(table_grid_cfg, "min_token_h", 0.0)
    tg_min_token_h_ratio = _cfg_float(table_grid_cfg, "min_token_h_ratio", 0.85)
    tg_draw_hlines = _cfg_bool(table_grid_cfg, "draw_hlines", True)
    tg_draw_seps_in_tables = _cfg_bool(table_grid_cfg, "draw_seps_in_tables", False)
    tg_draw_separators = _cfg_bool(table_grid_cfg, "draw_separators", False)
    tg_border_thickness = _cfg_int(table_grid_cfg, "border_thickness", 0)
    if tg_border_thickness <= 0:
        tg_border_thickness = max(4, tg_line_thickness + 2)

    tg_draw_overlay_lines = _parse_bool_env("EIDAT_TABLE_GRID_DRAW_LINES", False)
    tg_apply_borders_to_page = _parse_bool_env("EIDAT_TABLE_GRID_APPLY_BORDERS_TO_PAGE", False)
    tg_enable_prepass = table_grid_enabled and _parse_bool_env("EIDAT_TABLE_GRID_ENABLE_PREPASS", False)

    table_variants_lang = table_variants_cfg.get("lang") or None
    if table_variants_lang is None:
        table_variants_lang = ocr_engine.get_tesseract_lang()

    return {
        "project_root": str(project_root),
        "render_dpi": int(render_dpi),
        "ocr_dpi": int(ocr_dpi),
        "table_ocr_dpi": int(table_ocr_dpi),
        "detection_dpi": int(detection_dpi),
        "table_variants_lang": table_variants_lang,
        "table_grid_enable_prepass": bool(tg_enable_prepass),
        "tg_merge_kx": int(tg_merge_kx),
        "tg_min_gap": float(tg_min_gap),
        "tg_min_gap_ratio": float(tg_min_gap_ratio),
        "tg_gap_threshold": float(tg_gap_threshold),
        "tg_offset_px": float(tg_offset_px),
        "tg_line_thickness": int(tg_line_thickness),
        "tg_line_pad": float(tg_line_pad),
        "tg_min_token_h": float(tg_min_token_h),
        "tg_min_token_h_ratio": float(tg_min_token_h_ratio),
        "tg_draw_hlines": bool(tg_draw_hlines),
        "tg_draw_seps_in_tables": bool(tg_draw_seps_in_tables),
        "tg_draw_separators": bool(tg_draw_separators),
        "tg_draw_overlay_lines": bool(tg_draw_overlay_lines),
        "tg_apply_borders_to_page": bool(tg_apply_borders_to_page),
        "tg_border_thickness": int(tg_border_thickness),
        "graph_page_bypass": _parse_bool_env("EIDAT_GRAPH_PAGE_BYPASS", True),
        "enable_borderless": _parse_bool_env("EIDAT_TABLE_ENABLE_BORDERLESS", False)
        or _parse_bool_env("EIDAT_ENABLE_BORDERLESS_TABLES", False),
        "page_timeout_sec": max(0, _parse_int_env("EIDAT_PAGE_TIMEOUT_SEC", 0)),
    }


def _page_timeout_result(page_num: int, settings: dict[str, Any], message: str, *, timeout: bool) -> dict[str, Any]:
    return {
        "page": int(page_num),
        "error": str(message),
        "timeout": bool(timeout),
        "tokens": [],
        "tables": [],
        "charts": [],
        "img_w": 0,
        "img_h": 0,
        "dpi": int(settings.get("detection_dpi") or 0),
        "ocr_dpi": int(settings.get("ocr_dpi") or 0),
        "flow": {},
    }


def _process_debug_method_page(
    pdf_path: Path,
    page_num: int,
    page_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    project_root = Path(str(settings.get("project_root") or Path(__file__).resolve().parent.parent))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction import debug_exporter, graph_page_guard, ocr_engine, page_analyzer, token_projector
    except ImportError as e:
        raise RuntimeError(f"Unable to import extraction modules: {e}") from e

    try:
        from debug_method import run_table_variants, table_grid_debug
        from debug_method.run_debug_master import _draw_table_borders
    except ImportError as e:
        raise RuntimeError(f"Unable to import debug_method pipeline: {e}") from e

    page_path = page_dir / f"page_{page_num}.png"
    if not page_path.exists():
        raise RuntimeError(f"Missing rendered page: {page_path}")

    warnings: list[str] = []
    graph_guard: dict[str, Any] = {"regions": [], "stats": {}}
    graph_page_skip: dict[str, Any] = {"skip_page": False, "reason": "", "stats": {}}
    masked_table_page_path = page_path
    page_img_gray = None
    try:
        import cv2  # type: ignore

        page_img_gray = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
        if page_img_gray is not None:
            if bool(settings.get("graph_page_bypass", True)):
                probe_lang: str | None = None

                def _probe_page_tokens() -> list[dict[str, Any]]:
                    nonlocal probe_lang
                    if probe_lang is None:
                        probe_lang = ocr_engine.get_tesseract_lang()
                    tokens_probe, _probe_w, _probe_h, _probe_img = ocr_engine.ocr_page(
                        pdf_path,
                        page_num - 1,
                        96,
                        probe_lang,
                        3,
                        debug_dir=None,
                    )
                    return list(tokens_probe or [])

                graph_page_skip = graph_page_guard.inspect_page_for_graph_item_skip(
                    page_img_gray,
                    ocr_probe=_probe_page_tokens,
                )
                if bool(graph_page_skip.get("skip_page")):
                    warnings.append("graph_page_item_skipped")
                    page_h, page_w = page_img_gray.shape[:2]
                    page_data = {
                        "page": int(page_num),
                        "tokens": [],
                        "tables": [],
                        "charts": [],
                        "img_w": int(page_w),
                        "img_h": int(page_h),
                        "dpi": int(settings.get("detection_dpi") or 900),
                        "ocr_dpi": int(settings.get("ocr_dpi") or 450),
                        "flow": {},
                        "skip_reason": str(graph_page_skip.get("reason") or "graph_page_item_skipped"),
                        "skip_marker": "{ITEM SKIPPED}",
                        "graph_page_skip_stats": graph_page_skip.get("stats") or {},
                    }
                    if warnings:
                        page_data["warnings"] = warnings
                    debug_exporter.export_page_debug(
                        pdf_path,
                        page_num - 1,
                        [],
                        [],
                        int(page_w),
                        int(page_h),
                        int(settings.get("detection_dpi") or 900),
                        page_dir,
                        charts=[],
                        flow_data={},
                        ocr_dpi=int(settings.get("ocr_dpi") or 450),
                        ocr_img_w=int(page_w),
                        ocr_img_h=int(page_h),
                    )
                    return page_data
            graph_guard = graph_page_guard.find_chart_like_regions(page_img_gray)
            chart_regions = list(graph_guard.get("regions") or [])
            if chart_regions:
                masked_img = graph_page_guard.mask_regions(page_img_gray, chart_regions, pad_px=8)
                masked_table_page_path = page_dir / f"page_{page_num}_table_masked.png"
                cv2.imwrite(str(masked_table_page_path), masked_img)
                warnings.append("graph_like_chart_region_masked")
    except Exception:
        graph_guard = {"regions": [], "stats": {}}
        masked_table_page_path = page_path

    if bool(settings.get("table_grid_enable_prepass")):
        grid_dir = page_dir / "grid_debug"
        gap_override = None
        if float(settings.get("tg_gap_threshold") or 0.0) > 0:
            gap_override = float(settings["tg_gap_threshold"])
        elif float(settings.get("tg_min_gap") or 0.0) > 0:
            gap_override = float(settings["tg_min_gap"])
        summary = table_grid_debug.run_for_image(
            masked_table_page_path,
            out_dir=grid_dir,
            merge_kx=int(settings.get("tg_merge_kx") or 0),
            min_gap=gap_override if gap_override else None,
            min_gap_ratio=float(settings.get("tg_min_gap_ratio") or 0.0),
            offset_px=float(settings.get("tg_offset_px") or 24.0),
            line_thickness=int(settings.get("tg_line_thickness") or 3),
            line_pad_factor=float(settings.get("tg_line_pad") or 0.25),
            min_token_h_px=float(settings.get("tg_min_token_h") or 0.0),
            min_token_h_ratio=float(settings.get("tg_min_token_h_ratio") or 0.85),
            draw_tables=bool(settings.get("tg_draw_overlay_lines")),
            draw_hlines=bool(settings.get("tg_draw_overlay_lines") and settings.get("tg_draw_hlines")),
            draw_seps_in_tables=bool(
                settings.get("tg_draw_overlay_lines") and settings.get("tg_draw_seps_in_tables")
            ),
            draw_separators=bool(settings.get("tg_draw_overlay_lines") and settings.get("tg_draw_separators")),
        )
        tables_grid = summary.get("tables") or []
        if bool(settings.get("tg_apply_borders_to_page")) and tables_grid:
            _draw_table_borders(
                masked_table_page_path,
                tables_grid,
                masked_table_page_path,
                line_thickness=int(settings.get("tg_border_thickness") or 4),
            )

    fused_result = run_table_variants._run_for_input(
        masked_table_page_path,
        out_dir=page_dir,
        page=1,
        ocr_dpi_base=int(settings.get("table_ocr_dpi") or settings.get("ocr_dpi") or 450),
        detection_dpi=int(settings.get("detection_dpi") or 900),
        lang=settings.get("table_variants_lang"),
        clean=False,
        fuse=True,
        emit_variants=False,
        emit_fused=True,
        allow_no_tables=True,
        enable_borderless=bool(settings.get("enable_borderless")),
        return_fused=True,
        return_variant_rows=True,
    )

    fused_tables = list(fused_result.get("tables") or [])
    detection_dpi_used = int(fused_result.get("detection_dpi") or settings.get("detection_dpi") or 0)
    child_warnings = list(fused_result.get("warnings") or []) if isinstance(fused_result, dict) else []
    for warning in child_warnings:
        if warning not in warnings:
            warnings.append(str(warning))

    try:
        enable_candidates = _parse_bool_env("EIDAT_TABLE_VARIANT_CANDIDATES", True)
    except Exception:
        enable_candidates = True
    if enable_candidates:
        try:
            variants_meta = list(fused_result.get("variants") or [])
            variant_rows_all = list(fused_result.get("variant_rows_all") or [])

            split_mode = str(os.environ.get("EIDAT_TABLE_SPLIT_MODE", "vline") or "vline").strip().lower()
            combined_split_mode = str(
                os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE", "inherit") or "inherit"
            ).strip().lower()
            if combined_split_mode in ("inherit", "same"):
                combined_split_mode = split_mode
            combined_ascii_mode = "default"
            enable_combined_vline_split = combined_split_mode in ("vline", "default", "on", "true", "1", "yes")
            try:
                combined_vline_tol_px = float(
                    os.environ.get(
                        "EIDAT_COMBINED_TABLE_SPLIT_VLINE_MISALIGN_PX",
                        os.environ.get("EIDAT_TABLE_SPLIT_VLINE_MISALIGN_PX", "6.0"),
                    )
                )
            except Exception:
                combined_vline_tol_px = 6.0
            try:
                combined_vline_min_run = int(
                    os.environ.get(
                        "EIDAT_COMBINED_TABLE_SPLIT_VLINE_MIN_RUN",
                        os.environ.get("EIDAT_TABLE_SPLIT_VLINE_MIN_RUN", "1"),
                    )
                )
            except Exception:
                combined_vline_min_run = 1
            single_cell_sep = str(
                os.environ.get(
                    "EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR",
                    os.environ.get("EIDAT_TABLE_SPLIT_SINGLE_CELL_SEPARATOR", "1"),
                )
                or "1"
            ).strip().lower() in ("1", "true", "yes", "on")
            try:
                single_cell_sep_max_h_ratio = float(
                    os.environ.get(
                        "EIDAT_COMBINED_TABLE_SPLIT_SINGLE_CELL_SEPARATOR_MAX_H_RATIO",
                        os.environ.get("EIDAT_TABLE_SPLIT_SINGLE_CELL_SEPARATOR_MAX_H_RATIO", "0.0"),
                    )
                )
            except Exception:
                single_cell_sep_max_h_ratio = 0.0

            def _part_row_col_ids_pruned(part_table: Dict[str, Any]) -> tuple[list[int], list[int]]:
                cells = part_table.get("cells") or []
                if not isinstance(cells, list) or not cells:
                    return [], []

                row_ids_set: set[int] = set()
                col_ids_set: set[int] = set()
                text_by_rc: dict[tuple[int, int], str] = {}
                for cell in cells:
                    try:
                        r = int(cell.get("row"))
                        c = int(cell.get("col"))
                    except Exception:
                        continue
                    if r < 0 or c < 0:
                        continue
                    row_ids_set.add(int(r))
                    col_ids_set.add(int(c))

                    txt = str(cell.get("text", "") or "")
                    key = (int(r), int(c))
                    prev = text_by_rc.get(key)
                    if prev is None:
                        text_by_rc[key] = txt
                        continue
                    prev_s = str(prev).strip()
                    txt_s = str(txt).strip()
                    if not prev_s and txt_s:
                        text_by_rc[key] = txt
                        continue
                    if prev_s and txt_s and len(txt_s) > len(prev_s):
                        text_by_rc[key] = txt

                row_ids = sorted(row_ids_set)
                col_ids = sorted(col_ids_set)
                if not row_ids or not col_ids:
                    return [], []

                row_ids_keep = [
                    r_id
                    for r_id in row_ids
                    if any(str(text_by_rc.get((int(r_id), int(c_id)), "") or "").strip() for c_id in col_ids)
                ]
                col_ids_keep = [
                    c_id
                    for c_id in col_ids
                    if any(str(text_by_rc.get((int(r_id), int(c_id)), "") or "").strip() for r_id in row_ids_keep)
                ]
                return row_ids_keep, col_ids_keep

            parts: list[dict[str, Any]] = []
            order_counter = 0
            for src_idx, table in enumerate(fused_tables):
                if enable_combined_vline_split:
                    try:
                        parts_local = debug_exporter._split_table_on_vline_mismatch_for_display(
                            table,
                            tol_px=float(combined_vline_tol_px),
                            min_mismatch_run=int(combined_vline_min_run),
                            enable_single_cell_separator=bool(single_cell_sep),
                            single_cell_separator_max_h_ratio=float(single_cell_sep_max_h_ratio),
                        )
                    except Exception:
                        parts_local = [table]
                else:
                    parts_local = [table]
                for part in parts_local:
                    bbox = part.get("bbox_px") or []
                    if not bbox or len(bbox) != 4:
                        continue
                    parts.append(
                        {
                            "source_table_idx": int(src_idx),
                            "part": part,
                            "y0": float(bbox[1]),
                            "x0": float(bbox[0]),
                            "order": int(order_counter),
                        }
                    )
                    order_counter += 1

            parts_sorted = sorted(parts, key=lambda p: (p.get("y0", 0.0), p.get("x0", 0.0), p.get("order", 0)))
            out_tables: list[dict[str, Any]] = []
            for entry in parts_sorted:
                src_idx = int(entry.get("source_table_idx", 0))
                part = entry.get("part") or {}
                row_ids, col_ids = _part_row_col_ids_pruned(part)
                rows_by_variant: list[list[list[str]]] = []
                for variant_tables in variant_rows_all:
                    try:
                        tbl = variant_tables[src_idx] if src_idx < len(variant_tables) else {}
                        full_rows = tbl.get("rows") if isinstance(tbl, dict) else []
                    except Exception:
                        full_rows = []
                    v_rows: list[list[str]] = []
                    for r_id in row_ids:
                        row_out: list[str] = []
                        for c_id in col_ids:
                            try:
                                val = (
                                    str(full_rows[r_id][c_id])
                                    if (r_id < len(full_rows) and c_id < len(full_rows[r_id]))
                                    else ""
                                )
                            except Exception:
                                val = ""
                            row_out.append(val)
                        v_rows.append(row_out)
                    rows_by_variant.append(v_rows)

                out_tables.append(
                    {
                        "source_table_idx": int(src_idx),
                        "rows_by_variant": rows_by_variant,
                    }
                )

            sidecar = {
                "version": 1,
                "variant_count": int(len(variants_meta)),
                "variants": variants_meta,
                "ascii_mode": str(combined_ascii_mode),
                "tables": out_tables,
            }
            (page_dir / "table_variant_candidates.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except Exception:
            pass

    ocr_lang = ocr_engine.get_tesseract_lang()
    tokens, ocr_img_w, ocr_img_h, _img_path = ocr_engine.ocr_page(
        pdf_path, page_num - 1, int(settings.get("ocr_dpi") or 450), ocr_lang, 3, debug_dir=None
    )
    try:
        img_gray_ocr, _, _ = ocr_engine.render_pdf_page(pdf_path, page_num - 1, int(settings.get("ocr_dpi") or 450))
        if img_gray_ocr is not None and tokens:
            tokens = ocr_engine.reocr_low_confidence_tokens(
                img_gray_ocr, tokens, conf_threshold=0.6, lang=ocr_lang, verbose=False
            )
    except Exception:
        pass

    det_img, det_w, det_h = ocr_engine.render_pdf_page(pdf_path, page_num - 1, detection_dpi_used)
    if det_img is None:
        det_w = int((fused_result.get("detection_size") or {}).get("w") or 0) or int(ocr_img_w or 0)
        det_h = int((fused_result.get("detection_size") or {}).get("h") or 0) or int(ocr_img_h or 0)

    flow_tokens = (
        token_projector.scale_tokens_to_dpi(tokens, int(settings.get("ocr_dpi") or 450), detection_dpi_used) if tokens else []
    )
    flow_data = page_analyzer.extract_flow_text(flow_tokens, fused_tables, det_w, det_h)

    charts: List[Dict] = []
    enable_charts = _parse_bool_env("EIDAT_ENABLE_CHART_EXTRACTION", True)
    skip_chart_extraction = bool(graph_guard.get("regions"))
    if enable_charts and det_img is not None and not skip_chart_extraction:
        try:
            from extraction import chart_detection

            charts = chart_detection.detect_charts(det_img, flow_tokens, fused_tables, det_w, det_h, flow_data)
        except Exception:
            charts = []

    page_data = {
        "page": int(page_num),
        "tokens": tokens,
        "tables": fused_tables,
        "charts": charts,
        "img_w": det_w,
        "img_h": det_h,
        "dpi": detection_dpi_used,
        "ocr_dpi": int(settings.get("ocr_dpi") or 450),
        "flow": flow_data,
    }
    if warnings:
        page_data["warnings"] = warnings
    if graph_guard.get("regions"):
        page_data["masked_chart_regions"] = graph_guard.get("regions") or []
        page_data["masked_chart_region_stats"] = graph_guard.get("stats") or {}

    debug_exporter.export_page_debug(
        pdf_path,
        page_num - 1,
        tokens,
        fused_tables,
        det_w,
        det_h,
        detection_dpi_used,
        page_dir,
        charts=charts,
        flow_data=flow_data,
        ocr_dpi=int(settings.get("ocr_dpi") or 450),
        ocr_img_w=ocr_img_w,
        ocr_img_h=ocr_img_h,
    )

    return page_data


def _process_debug_method_page_worker(
    pdf_path_str: str,
    page_num: int,
    page_dir_str: str,
    settings: dict[str, Any],
    out_q,
) -> None:
    try:
        result = _process_debug_method_page(Path(pdf_path_str), int(page_num), Path(page_dir_str), dict(settings))
        out_q.put(("ok", result))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}"))


def _run_debug_method_page_with_timeout(
    pdf_path: Path,
    page_num: int,
    page_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    timeout_sec = max(0, int(settings.get("page_timeout_sec") or 0))
    if timeout_sec <= 0:
        return _process_debug_method_page(pdf_path, page_num, page_dir, settings)

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_process_debug_method_page_worker,
        args=(str(pdf_path), int(page_num), str(page_dir), dict(settings), q),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=float(timeout_sec))

    if proc.is_alive():
        try:
            proc.terminate()
        except Exception:
            pass
        proc.join(timeout=5)
        return _page_timeout_result(page_num, settings, f"Timeout: exceeded {int(timeout_sec)}s", timeout=True)

    try:
        status, payload = q.get_nowait()
    except Exception:
        status, payload = ("err", "No result returned from page worker")
    if status == "ok" and isinstance(payload, dict):
        return payload
    return _page_timeout_result(page_num, settings, str(payload), timeout=False)


def _run_debug_method_extraction(pdf_path: Path, dpi: int | None, output_dir: Path) -> dict[str, Any]:
    """
    New default extraction path:
    - Render pages to PNG
    - Draw table borders (debug_method/table_grid_debug)
    - Run table variants on bordered PNGs (fused only)
    - Build combined.txt with existing formatter
    """
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction import debug_exporter
    except ImportError as e:
        raise RuntimeError(f"Unable to import extraction modules: {e}") from e

    settings = _build_debug_method_settings(dpi)
    doc_name = pdf_path.stem
    artifacts_dir = output_dir / "debug" / "ocr" / doc_name
    pages_root = artifacts_dir / "pages"

    # Clean prior per-page outputs to avoid stale pages
    if pages_root.exists():
        shutil.rmtree(pages_root, ignore_errors=True)
    pages_root.mkdir(parents=True, exist_ok=True)

    # Remove old root-level page artifacts if present (migrating layout)
    for old in artifacts_dir.glob("page_*.*"):
        try:
            old.unlink()
        except Exception:
            pass

    page_count = _render_pdf_pages_to_dirs(pdf_path, pages_root, int(settings.get("render_dpi") or 450))
    if page_count <= 0:
        raise RuntimeError(f"No pages rendered for {pdf_path}")

    pages_data: List[Dict] = []

    for page_num in range(1, page_count + 1):
        page_dir = pages_root / f"page_{page_num}"
        page_path = page_dir / f"page_{page_num}.png"
        if not page_path.exists():
            raise RuntimeError(f"Missing rendered page: {page_path}")
        pages_data.append(_run_debug_method_page_with_timeout(pdf_path, page_num, page_dir, settings))

    combined_path = debug_exporter.export_combined_text(pdf_path, pages_data, artifacts_dir)
    summary_path = debug_exporter.create_summary_report(pdf_path, pages_data, artifacts_dir)

    return {
        "dir": str(artifacts_dir),
        "target_dir": str(artifacts_dir),
        "combined": str(combined_path) if combined_path.exists() else None,
        "manifest": str(summary_path) if summary_path.exists() else None,
        "results": pages_data,
        "pipeline": "debug_method_v1",
    }


def _run_clean_extraction(pdf_path: Path, dpi: int | None, output_dir: Path) -> dict[str, Any]:
    """
    Run the default extraction pipeline (debug_method + fused tables).
    """
    use_legacy = str(os.environ.get("EIDAT_USE_LEGACY_PIPELINE", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "legacy",
    )
    if use_legacy:
        return _run_legacy_extraction(pdf_path, dpi=dpi, output_dir=output_dir)
    return _run_debug_method_extraction(pdf_path, dpi=dpi, output_dir=output_dir)


def _default_excel_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "user_inputs" / "excel_trend_config.json"


def _resolve_user_inputs_file(name: str) -> Path:
    """
    Resolve a user_inputs/<name> file for both dev-repo and production-node runs.

    In production-node runs, EIDAT.bat sets EIDAT_DATA_ROOT to a node-local writable
    folder (e.g. <node_root>\\EIDAT\\UserData). Prefer that first, then fall back to
    the runtime repo root's user_inputs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []

    raw_data_root = (os.environ.get("EIDAT_DATA_ROOT") or "").strip()
    if raw_data_root:
        try:
            data_root = Path(raw_data_root).expanduser()
            if not data_root.is_absolute():
                data_root = (repo_root / data_root).resolve()
            candidates.append(data_root / "user_inputs" / str(name))
        except Exception:
            pass

    candidates.append(repo_root / "user_inputs" / str(name))

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return candidates[-1]


def _excel_artifacts_dir(paths: SupportPaths, excel_path: Path) -> Path:
    if excel_path.suffix.lower() in MAT_EXTENSIONS:
        bundle = detect_mat_bundle_member(excel_path, repo_root=paths.global_repo)
        if bundle is not None:
            return mat_bundle_artifacts_dir(paths.support_dir, bundle)
    return paths.support_dir / "debug" / "ocr" / f"{excel_path.stem}{EXCEL_ARTIFACT_SUFFIX}"


def _safe_repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        try:
            return path.relative_to(repo_root).as_posix()
        except Exception:
            return str(path)


def _bundle_identity_path(member: MatBundleMember) -> Path:
    return member.file_path.with_name(f"{member.bundle_stem}.mat")


def _bundle_folder_asset_hints(member: MatBundleMember) -> dict[str, str]:
    parent = member.file_path.parent
    asset_specific_type = str(parent.name or "").strip()
    asset_type = str(parent.parent.name or "").strip() if parent.parent != parent else ""
    return {
        "asset_type": asset_type or "Unknown",
        "asset_specific_type": asset_specific_type or "Unknown",
    }


def _merge_bundle_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if key not in merged:
                merged[key] = value
                continue
            cur = merged.get(key)
            cur_text = str(cur or "").strip()
            new_text = str(value or "").strip()
            if key == "document_type_evidence":
                if cur in (None, [], "") and value not in (None, [], ""):
                    merged[key] = value
                continue
            if cur_text in {"", "Unknown", "unknown"} and new_text not in {"", "Unknown", "unknown"}:
                merged[key] = value
    return merged


def _cleanup_stale_mat_member_artifacts(paths: SupportPaths, members: list[MatBundleMember], keep_dir: Path) -> None:
    for member in members:
        old_dir = paths.support_dir / "debug" / "ocr" / f"{member.file_path.stem}{EXCEL_ARTIFACT_SUFFIX}"
        try:
            if old_dir.resolve() == keep_dir.resolve():
                continue
        except Exception:
            if str(old_dir) == str(keep_dir):
                continue
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)


def _write_mat_bundle_manifest(
    *,
    global_repo: Path,
    artifacts_dir: Path,
    bundle: MatBundleMember,
    members: list[MatBundleMember],
    sqlite_path: Path,
    metadata_path: Path | None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    meta = metadata if isinstance(metadata, dict) else {}
    payload = {
        "bundle_key": bundle.group_key,
        "bundle_stem": bundle.bundle_stem,
        "serial_number": bundle.serial_number,
        "asset_type": str(meta.get("asset_type") or "").strip(),
        "asset_specific_type": str(meta.get("asset_specific_type") or "").strip(),
        "program_title": str(meta.get("program_title") or "").strip(),
        "source_dir_rel": _safe_repo_rel(global_repo, bundle.file_path.parent),
        "sqlite_rel": _safe_repo_rel(global_repo, sqlite_path),
        "metadata_rel": _safe_repo_rel(global_repo, metadata_path) if metadata_path is not None else "",
        "members": [
            {
                "rel_path": _safe_repo_rel(global_repo, member.file_path),
                "file_name": member.file_path.name,
                "sequence_name": member.sequence_name,
                "sequence_number": int(member.sequence_number),
            }
            for member in members
        ],
    }
    out = artifacts_dir / "mat_seq_bundle.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def _td_agg_unknownish(value: object) -> bool:
    txt = str(value or "").strip()
    return not txt or txt.casefold() in {"unknown", "none", "null", "n/a"}


def _td_agg_clean_scope_value(value: object, *, fallback: str = "Unknown") -> str:
    txt = str(value or "").strip()
    if not txt:
        return fallback
    return txt


def _td_agg_safe_path_name(value: object, *, fallback: str = "unknown") -> str:
    raw = str(value or "").strip() or str(fallback or "unknown")
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", raw).strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:80] or str(fallback or "unknown")


def _td_agg_safe_ident(value: object, *, prefix: str = "sheet") -> str:
    raw = str(value or "").strip()
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:80]


def _td_agg_table_name(run_name: object) -> str:
    return f"sheet__{_td_agg_safe_ident(run_name, prefix='sheet')}"


def _td_agg_quote_ident(value: object) -> str:
    return '"' + str(value or "").replace('"', '""') + '"'


def _td_agg_is_sequence_sheet_name(value: object) -> bool:
    return bool(_TD_AGG_SEQ_SHEET_RE.match(str(value or "").strip()))


def _td_agg_sequence_token(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    m = re.match(r"^\s*seq(?:uence)?[\s_-]*(.+?)\s*$", raw, flags=re.IGNORECASE)
    if m:
        raw = str(m.group(1) or "").strip()
    raw = re.sub(r"[\s_-]+", "", raw)
    return raw


def _td_agg_natural_sequence_key(value: object) -> list[tuple[int, object]]:
    token = _td_agg_sequence_token(value) or str(value or "").strip()
    parts = re.findall(r"\d+|[A-Za-z]+|[^A-Za-z0-9]+", token)
    key: list[tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        elif part.isalpha():
            key.append((1, part.casefold()))
        else:
            key.append((2, part.casefold()))
    if not key:
        key.append((3, str(value or "").casefold()))
    return key


def _td_agg_normalize_serial(value: object) -> str:
    txt = str(value or "").strip()
    if not txt or txt.casefold() in {"unknown", "none", "null", "n/a"}:
        return ""
    txt = txt.replace("S/N", "SN").replace("s/n", "SN")
    match = _TD_AGG_SERIAL_RE.search(txt)
    if match:
        txt = match.group(0)
    cleaned = re.sub(r"\s+", "", str(txt or "").strip().upper())
    if re.fullmatch(r"\d{1,8}", cleaned):
        if len(cleaned) <= 4:
            cleaned = cleaned.zfill(4)
        return f"SN{cleaned}"
    return cleaned


def _td_agg_serial_from_text(value: object, *, allow_raw: bool = False) -> str:
    txt = str(value or "").strip()
    if not txt:
        return ""
    match = _TD_AGG_SERIAL_RE.search(txt)
    if match:
        return _td_agg_normalize_serial(match.group(0))
    if allow_raw:
        return _td_agg_normalize_serial(txt)
    return ""


def _td_agg_extract_tab_serial(
    conn: sqlite3.Connection,
    *,
    sheet_name: str,
    fallback_serial: object,
) -> tuple[str, str]:
    try:
        rows = conn.execute(
            """
            SELECT excel_row, excel_col, value
            FROM __meta_cells
            WHERE sheet_name=?
            ORDER BY excel_row, excel_col
            """,
            (str(sheet_name),),
        ).fetchall()
    except Exception:
        rows = []
    cells = {(int(r[0]), int(r[1])): str(r[2] or "").strip() for r in rows if str(r[2] or "").strip()}
    for (row, col), text in sorted(cells.items()):
        if not _TD_AGG_SERIAL_LABEL_RE.search(text):
            continue
        same_cell = _td_agg_serial_from_text(text)
        if same_cell:
            return same_cell, "tab_label_same_cell"
        for delta_col in range(1, 5):
            cand = _td_agg_serial_from_text(cells.get((row, col + delta_col)), allow_raw=True)
            if cand:
                return cand, "tab_label_right_cell"
        for delta_row in range(1, 3):
            cand = _td_agg_serial_from_text(cells.get((row + delta_row, col)), allow_raw=True)
            if cand:
                return cand, "tab_label_below_cell"
            cand = _td_agg_serial_from_text(cells.get((row + delta_row, col + 1)), allow_raw=True)
            if cand:
                return cand, "tab_label_below_right_cell"
    for _pos, text in sorted(cells.items()):
        cand = _td_agg_serial_from_text(text)
        if cand:
            return cand, "tab_regex"
    fallback = _td_agg_normalize_serial(fallback_serial)
    if fallback:
        return fallback, "metadata_fallback"
    return "", ""


def _td_agg_resolve_support_path(paths: SupportPaths, raw_path: object) -> Path:
    raw = str(raw_path or "").strip().strip('"')
    if not raw:
        return Path()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    norm = raw.replace("/", "\\").lstrip("\\")
    parts = list(Path(norm).parts)
    support_names = {"eidat support", "edat support"}
    container_names = {"eidat", "edat"}
    if parts and str(parts[0]).strip().casefold() in support_names:
        rest = Path(*parts[1:]) if len(parts) > 1 else Path()
        return paths.support_dir / rest
    for idx, part in enumerate(parts):
        if str(part).strip().casefold() not in support_names:
            continue
        prefix = [str(v).strip().casefold() for v in parts[:idx]]
        if prefix and all(v in container_names for v in prefix):
            return paths.global_repo / Path(*parts)
        rest = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        return paths.support_dir / rest
    candidate = paths.global_repo / Path(norm)
    if candidate.exists():
        return candidate
    return paths.support_dir / Path(norm)


def _td_agg_metadata_files(support_dir: Path) -> list[Path]:
    root = Path(support_dir) / "debug" / "ocr"
    if not root.exists():
        return []
    out: list[Path] = []
    aggregate_root_name = TD_SERIAL_AGGREGATES_DIRNAME.casefold()
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
        except Exception:
            continue
        low = path.name.lower()
        if not (low.endswith("_metadata.json") or low.endswith(".metadata.json")):
            continue
        try:
            parts = [p.casefold() for p in path.relative_to(root).parts]
        except Exception:
            parts = [p.casefold() for p in path.parts]
        if aggregate_root_name in parts:
            continue
        out.append(path)
    return sorted(out, key=lambda p: str(p).casefold())


def _td_agg_ordered_sheet_info(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    try:
        conn.row_factory = sqlite3.Row
        cols = {str(row[1] or "") for row in conn.execute("PRAGMA table_info(__sheet_info)").fetchall()}
    except Exception:
        cols = set()
    order_sql = "ORDER BY COALESCE(import_order, rowid), rowid" if "import_order" in cols else "ORDER BY rowid"
    try:
        return list(conn.execute(f"SELECT * FROM __sheet_info {order_sql}").fetchall())
    except Exception:
        return []


def _td_agg_collect_members(paths: SupportPaths) -> list[dict[str, Any]]:
    support_dir = Path(paths.support_dir)
    members: list[dict[str, Any]] = []
    for meta_path in _td_agg_metadata_files(support_dir):
        try:
            meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta_raw, dict):
            continue
        if str(meta_raw.get("metadata_source") or "").strip() == TD_SERIAL_AGGREGATE_METADATA_SOURCE:
            continue
        ext = str(meta_raw.get("file_extension") or "").strip().lower()
        if ext == ".mat":
            continue
        if not _is_confirmed_test_data_meta(meta_raw):
            continue
        sqlite_rel = str(meta_raw.get("excel_sqlite_rel") or "").strip()
        sqlite_path = _td_agg_resolve_support_path(paths, sqlite_rel)
        if not sqlite_path.exists() or not sqlite_path.is_file():
            continue
        try:
            metadata_rel = str(meta_path.resolve().relative_to(support_dir.resolve())).replace("\\", "/")
        except Exception:
            metadata_rel = str(meta_path)
        try:
            artifacts_rel = str(meta_path.parent.resolve().relative_to(support_dir.resolve())).replace("\\", "/")
        except Exception:
            artifacts_rel = str(meta_path.parent)
        try:
            with sqlite3.connect(str(sqlite_path)) as src:
                src.row_factory = sqlite3.Row
                sheet_rows = _td_agg_ordered_sheet_info(src)
                for sheet_row in sheet_rows:
                    sheet_name = str(sheet_row["sheet_name"] if "sheet_name" in sheet_row.keys() else "").strip()
                    source_sheet_name = str(
                        sheet_row["source_sheet_name"] if "source_sheet_name" in sheet_row.keys() else sheet_name
                    ).strip()
                    real_tab_name = source_sheet_name or sheet_name
                    if not _td_agg_is_sequence_sheet_name(real_tab_name):
                        continue
                    tab_serial, serial_source = _td_agg_extract_tab_serial(
                        src,
                        sheet_name=sheet_name,
                        fallback_serial=meta_raw.get("serial_number"),
                    )
                    if not tab_serial:
                        continue
                    program_title = _td_agg_clean_scope_value(meta_raw.get("program_title"))
                    asset_type = _td_agg_clean_scope_value(meta_raw.get("asset_type"))
                    asset_specific_type = _td_agg_clean_scope_value(meta_raw.get("asset_specific_type"))
                    table_name = str(sheet_row["table_name"] if "table_name" in sheet_row.keys() else "").strip()
                    if not table_name:
                        table_name = _td_agg_table_name(sheet_name)
                    try:
                        import_order = int(sheet_row["import_order"] if "import_order" in sheet_row.keys() else 0)
                    except Exception:
                        import_order = 0
                    sequence_order_key = _td_agg_natural_sequence_key(real_tab_name)
                    members.append(
                        {
                            "group_key": (
                                program_title.casefold(),
                                asset_type.casefold(),
                                asset_specific_type.casefold(),
                                tab_serial.casefold(),
                            ),
                            "program_title": program_title,
                            "asset_type": asset_type,
                            "asset_specific_type": asset_specific_type,
                            "serial_number": tab_serial,
                            "serial_source": serial_source,
                            "sqlite_path": sqlite_path,
                            "sqlite_rel": sqlite_rel,
                            "metadata_path": meta_path,
                            "metadata_rel": metadata_rel,
                            "artifacts_rel": artifacts_rel,
                            "source_file": str(meta_raw.get("source_file") or ""),
                            "sheet_name": sheet_name,
                            "source_sheet_name": real_tab_name,
                            "table_name": table_name,
                            "import_order": int(import_order),
                            "sequence_token": _td_agg_sequence_token(real_tab_name),
                            "sequence_order_key": sequence_order_key,
                            "metadata": dict(meta_raw),
                            "sheet_info": {k: sheet_row[k] for k in sheet_row.keys()},
                        }
                    )
        except Exception:
            continue
    return members


def _td_agg_create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE IF NOT EXISTS __workbook (
          source_file TEXT NOT NULL,
          imported_epoch_ns INTEGER NOT NULL,
          excel_size_bytes INTEGER NOT NULL,
          excel_mtime_ns INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS __sheet_info (
          sheet_name TEXT PRIMARY KEY,
          source_sheet_name TEXT,
          table_name TEXT NOT NULL,
          header_row INTEGER NOT NULL,
          import_order INTEGER,
          excel_col_indices_json TEXT NOT NULL,
          headers_json TEXT NOT NULL,
          columns_json TEXT NOT NULL,
          mapped_headers_json TEXT,
          rows_inserted INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS __column_map (
          sheet_name TEXT NOT NULL,
          header TEXT NOT NULL,
          mapped_header TEXT NOT NULL,
          sqlite_column TEXT NOT NULL,
          PRIMARY KEY(sheet_name, header)
        );

        CREATE TABLE IF NOT EXISTS __meta_cells (
          sheet_name TEXT NOT NULL,
          excel_row INTEGER NOT NULL,
          excel_col INTEGER NOT NULL,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS __sequence_context (
          sheet_name TEXT PRIMARY KEY,
          source_sheet_name TEXT,
          data_mode_raw TEXT,
          run_type TEXT,
          on_time_value REAL,
          on_time_units TEXT,
          off_time_value REAL,
          off_time_units TEXT,
          control_period REAL,
          nominal_pf_value REAL,
          nominal_pf_units TEXT,
          nominal_tf_value REAL,
          nominal_tf_units TEXT,
          suppression_voltage_value REAL,
          suppression_voltage_units TEXT,
          valve_voltage_value REAL,
          valve_voltage_units TEXT,
          extraction_status TEXT,
          extraction_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS __td_serial_aggregate_members (
          sheet_name TEXT PRIMARY KEY,
          source_sheet_name TEXT,
          source_sqlite_rel TEXT,
          source_metadata_rel TEXT,
          source_artifacts_rel TEXT,
          source_table_name TEXT,
          tab_serial TEXT,
          serial_source TEXT,
          sequence_token TEXT,
          sequence_order_json TEXT,
          output_table_name TEXT,
          duplicate_of TEXT
        );
        """
    )


def _td_agg_copy_data_table(
    src: sqlite3.Connection,
    dest: sqlite3.Connection,
    *,
    source_table: str,
    output_table: str,
) -> int:
    info = src.execute(f"PRAGMA table_info({_td_agg_quote_ident(source_table)})").fetchall()
    if not info:
        raise RuntimeError(f"Source sequence table not found: {source_table}")
    col_defs: list[str] = []
    col_names: list[str] = []
    for row in info:
        name = str(row[1] or "").strip()
        if not name:
            continue
        typ = str(row[2] or "").strip() or "REAL"
        not_null = bool(int(row[3] or 0))
        col_defs.append(f"{_td_agg_quote_ident(name)} {typ}{' NOT NULL' if not_null else ''}")
        col_names.append(name)
    dest.execute(f"DROP TABLE IF EXISTS {_td_agg_quote_ident(output_table)}")
    dest.execute(f"CREATE TABLE {_td_agg_quote_ident(output_table)} ({', '.join(col_defs)})")
    if not col_names:
        return 0
    q_cols = ", ".join(_td_agg_quote_ident(name) for name in col_names)
    placeholders = ", ".join("?" for _ in col_names)
    rows = src.execute(f"SELECT {q_cols} FROM {_td_agg_quote_ident(source_table)} ORDER BY rowid").fetchall()
    if rows:
        dest.executemany(
            f"INSERT INTO {_td_agg_quote_ident(output_table)} ({q_cols}) VALUES ({placeholders})",
            [tuple(row) for row in rows],
        )
    return len(rows)


def _td_agg_copy_sequence_context(
    src: sqlite3.Connection,
    dest: sqlite3.Connection,
    *,
    source_sheet_name: str,
    output_sheet_name: str,
    output_source_sheet_name: str,
) -> None:
    try:
        src.row_factory = sqlite3.Row
        row = src.execute("SELECT * FROM __sequence_context WHERE sheet_name=? LIMIT 1", (source_sheet_name,)).fetchone()
    except Exception:
        row = None
    if row is None:
        dest.execute(
            """
            INSERT OR REPLACE INTO __sequence_context(
              sheet_name, source_sheet_name, extraction_status, extraction_reason
            ) VALUES (?, ?, ?, ?)
            """,
            (output_sheet_name, output_source_sheet_name, "incomplete", "sequence context unavailable in source aggregate input"),
        )
        return
    payload = {k: row[k] for k in row.keys()}
    payload["sheet_name"] = output_sheet_name
    payload["source_sheet_name"] = output_source_sheet_name
    columns = [
        "sheet_name",
        "source_sheet_name",
        "data_mode_raw",
        "run_type",
        "on_time_value",
        "on_time_units",
        "off_time_value",
        "off_time_units",
        "control_period",
        "nominal_pf_value",
        "nominal_pf_units",
        "nominal_tf_value",
        "nominal_tf_units",
        "suppression_voltage_value",
        "suppression_voltage_units",
        "valve_voltage_value",
        "valve_voltage_units",
        "extraction_status",
        "extraction_reason",
    ]
    dest.execute(
        "INSERT OR REPLACE INTO __sequence_context("
        + ", ".join(columns)
        + ") VALUES ("
        + ", ".join("?" for _ in columns)
        + ")",
        tuple(payload.get(col) for col in columns),
    )


def _td_agg_write_group(
    paths: SupportPaths,
    *,
    group_members: Sequence[Mapping[str, Any]],
    export_text_mirror: Any | None,
    export_excel_mirror: Any | None,
    mirror_xlsx: bool,
) -> dict[str, Any]:
    first = dict(group_members[0])
    serial = str(first.get("serial_number") or "").strip()
    program_title = str(first.get("program_title") or "Unknown").strip() or "Unknown"
    asset_type = str(first.get("asset_type") or "Unknown").strip() or "Unknown"
    asset_specific_type = str(first.get("asset_specific_type") or "Unknown").strip() or "Unknown"
    out_dir = (
        Path(paths.support_dir)
        / "debug"
        / "ocr"
        / TD_SERIAL_AGGREGATES_DIRNAME
        / _td_agg_safe_path_name(program_title, fallback="program")
        / _td_agg_safe_path_name(asset_type, fallback="asset")
        / _td_agg_safe_path_name(asset_specific_type, fallback="asset_specific")
        / _td_agg_safe_path_name(serial, fallback="serial")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = out_dir / f"{_td_agg_safe_path_name(serial, fallback='serial')}.sqlite3"
    if sqlite_path.exists():
        try:
            sqlite_path.unlink()
        except Exception:
            pass

    warnings: list[str] = []
    manifest_members: list[dict[str, Any]] = []
    used_run_names: dict[str, int] = {}
    now_ns = time.time_ns()
    with sqlite3.connect(str(sqlite_path)) as dest:
        _td_agg_create_schema(dest)
        for idx, raw_member in enumerate(group_members, start=1):
            member = dict(raw_member)
            source_sqlite = Path(member.get("sqlite_path") or "")
            source_sheet = str(member.get("sheet_name") or "").strip()
            source_tab = str(member.get("source_sheet_name") or source_sheet).strip()
            source_table = str(member.get("table_name") or "").strip()
            base_run = source_tab or source_sheet or f"seq_{idx}"
            run_key = base_run.casefold()
            dup_idx = used_run_names.get(run_key, 0) + 1
            used_run_names[run_key] = dup_idx
            output_run = base_run if dup_idx == 1 else f"{base_run}__{dup_idx}"
            duplicate_of = "" if dup_idx == 1 else base_run
            if duplicate_of:
                warnings.append(f"Duplicate sequence name {base_run!r} was written as {output_run!r}.")
            output_table = _td_agg_table_name(output_run)
            try:
                st = source_sqlite.stat()
                size_bytes = int(st.st_size)
                mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            except Exception:
                size_bytes = 0
                mtime_ns = 0
            with sqlite3.connect(str(source_sqlite)) as src:
                src.row_factory = sqlite3.Row
                dest.execute(
                    "INSERT INTO __workbook(source_file, imported_epoch_ns, excel_size_bytes, excel_mtime_ns) VALUES (?, ?, ?, ?)",
                    (str(source_sqlite), int(now_ns), int(size_bytes), int(mtime_ns)),
                )
                rows_inserted = _td_agg_copy_data_table(
                    src,
                    dest,
                    source_table=source_table,
                    output_table=output_table,
                )
                sheet_info = dict(member.get("sheet_info") or {})
                dest.execute(
                    """
                    INSERT INTO __sheet_info(
                      sheet_name, source_sheet_name, table_name, header_row, import_order,
                      excel_col_indices_json, headers_json, columns_json, mapped_headers_json, rows_inserted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        output_run,
                        source_tab,
                        output_table,
                        int(sheet_info.get("header_row") or 0),
                        int(idx),
                        str(sheet_info.get("excel_col_indices_json") or "[]"),
                        str(sheet_info.get("headers_json") or "[]"),
                        str(sheet_info.get("columns_json") or "{}"),
                        str(sheet_info.get("mapped_headers_json") or "[]"),
                        int(rows_inserted),
                    ),
                )
                try:
                    col_rows = src.execute(
                        "SELECT header, mapped_header, sqlite_column FROM __column_map WHERE sheet_name=? ORDER BY rowid",
                        (source_sheet,),
                    ).fetchall()
                    dest.executemany(
                        "INSERT OR REPLACE INTO __column_map(sheet_name, header, mapped_header, sqlite_column) VALUES (?, ?, ?, ?)",
                        [(output_run, row[0], row[1], row[2]) for row in col_rows],
                    )
                except Exception:
                    pass
                try:
                    meta_rows = src.execute(
                        "SELECT excel_row, excel_col, value FROM __meta_cells WHERE sheet_name=? ORDER BY excel_row, excel_col",
                        (source_sheet,),
                    ).fetchall()
                    dest.executemany(
                        "INSERT INTO __meta_cells(sheet_name, excel_row, excel_col, value) VALUES (?, ?, ?, ?)",
                        [(output_run, int(row[0]), int(row[1]), str(row[2] or "")) for row in meta_rows],
                    )
                except Exception:
                    pass
                _td_agg_copy_sequence_context(
                    src,
                    dest,
                    source_sheet_name=source_sheet,
                    output_sheet_name=output_run,
                    output_source_sheet_name=source_tab,
                )
            sequence_order_json = json.dumps(member.get("sequence_order_key") or [], ensure_ascii=True)
            dest.execute(
                """
                INSERT OR REPLACE INTO __td_serial_aggregate_members(
                  sheet_name, source_sheet_name, source_sqlite_rel, source_metadata_rel,
                  source_artifacts_rel, source_table_name, tab_serial, serial_source,
                  sequence_token, sequence_order_json, output_table_name, duplicate_of
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    output_run,
                    source_tab,
                    str(member.get("sqlite_rel") or ""),
                    str(member.get("metadata_rel") or ""),
                    str(member.get("artifacts_rel") or ""),
                    source_table,
                    serial,
                    str(member.get("serial_source") or ""),
                    str(member.get("sequence_token") or ""),
                    sequence_order_json,
                    output_table,
                    duplicate_of,
                ),
            )
            manifest_members.append(
                {
                    "sheet_name": output_run,
                    "source_sheet_name": source_tab,
                    "source_sqlite_rel": str(member.get("sqlite_rel") or ""),
                    "source_metadata_rel": str(member.get("metadata_rel") or ""),
                    "source_artifacts_rel": str(member.get("artifacts_rel") or ""),
                    "source_table_name": source_table,
                    "tab_serial": serial,
                    "serial_source": str(member.get("serial_source") or ""),
                    "sequence_token": str(member.get("sequence_token") or ""),
                    "sequence_order_key": list(member.get("sequence_order_key") or []),
                    "duplicate_of": duplicate_of,
                    "rows_inserted": int(rows_inserted),
                }
            )
        dest.commit()

    sqlite_rel = _safe_repo_rel(paths.global_repo, sqlite_path)
    metadata_path = out_dir / f"{_td_agg_safe_path_name(serial, fallback='serial')}_metadata.json"
    manifest_path = out_dir / "td_serial_aggregate.json"
    source_metadata_rels = [
        str(item.get("source_metadata_rel") or "")
        for item in manifest_members
        if str(item.get("source_metadata_rel") or "").strip()
    ]
    aggregate_meta = {
        "program_title": program_title,
        "asset_type": asset_type,
        "asset_specific_type": asset_specific_type,
        "serial_number": serial,
        "part_number": str(first.get("metadata", {}).get("part_number") or "Unknown"),
        "revision": str(first.get("metadata", {}).get("revision") or "Unknown"),
        "test_date": str(first.get("metadata", {}).get("test_date") or "Unknown"),
        "report_date": str(first.get("metadata", {}).get("report_date") or "Unknown"),
        "vendor": str(first.get("metadata", {}).get("vendor") or "Unknown"),
        "acceptance_test_plan_number": str(first.get("metadata", {}).get("acceptance_test_plan_number") or "Unknown"),
        "document_type": "TD",
        "document_type_acronym": "TD",
        "document_type_status": "confirmed",
        "document_type_source": "td_serial_aggregate",
        "document_type_reason": "td_serial_aggregate",
        "document_type_evidence": [
            {
                "kind": "td_serial_aggregate",
                "serial_number": serial,
                "sequence_count": len(manifest_members),
            }
        ],
        "document_type_review_required": False,
        "excel_sqlite_rel": sqlite_rel,
        "file_extension": ".sqlite3",
        "metadata_source": TD_SERIAL_AGGREGATE_METADATA_SOURCE,
    }
    metadata_path.write_text(json.dumps(aggregate_meta, indent=2, ensure_ascii=True), encoding="utf-8")
    manifest = {
        "kind": TD_SERIAL_AGGREGATE_METADATA_SOURCE,
        "serial_number": serial,
        "program_title": program_title,
        "asset_type": asset_type,
        "asset_specific_type": asset_specific_type,
        "sqlite_rel": sqlite_rel,
        "metadata_rel": _safe_repo_rel(paths.global_repo, metadata_path),
        "source_metadata_rels": source_metadata_rels,
        "sequence_count": len(manifest_members),
        "members": manifest_members,
        "warnings": warnings,
        "built_epoch_ns": int(now_ns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    if export_text_mirror is not None:
        try:
            export_text_mirror(sqlite_path)
        except Exception:
            pass
    if mirror_xlsx and export_excel_mirror is not None:
        try:
            export_excel_mirror(sqlite_path)
        except Exception:
            pass
    return {
        "serial_number": serial,
        "sqlite_path": str(sqlite_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "sequence_count": len(manifest_members),
        "warnings": warnings,
    }


def rebuild_td_serial_aggregates(
    paths: SupportPaths,
    *,
    export_text_mirror: Any | None = None,
    export_excel_mirror: Any | None = None,
    mirror_xlsx: bool = False,
) -> dict[str, Any]:
    members = _td_agg_collect_members(paths)
    aggregate_root = Path(paths.support_dir) / "debug" / "ocr" / TD_SERIAL_AGGREGATES_DIRNAME
    if aggregate_root.exists():
        shutil.rmtree(aggregate_root, ignore_errors=True)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for member in members:
        key = tuple(member.get("group_key") or ())
        if len(key) != 4:
            continue
        groups.setdefault(key, []).append(member)
    outputs: list[dict[str, Any]] = []
    for _key, group_members in sorted(groups.items(), key=lambda item: item[0]):
        ordered = sorted(
            group_members,
            key=lambda item: (
                item.get("sequence_order_key") or [],
                str(item.get("metadata_rel") or "").casefold(),
                int(item.get("import_order") or 0),
                str(item.get("source_sheet_name") or "").casefold(),
            ),
        )
        if not ordered:
            continue
        outputs.append(
            _td_agg_write_group(
                paths,
                group_members=ordered,
                export_text_mirror=export_text_mirror,
                export_excel_mirror=export_excel_mirror,
                mirror_xlsx=bool(mirror_xlsx),
            )
        )
    return {
        "aggregate_root": str(aggregate_root),
        "source_sequence_count": len(members),
        "aggregate_count": len(outputs),
        "aggregates": outputs,
    }


def _mark_files_processed(
    conn: Any,
    *,
    global_repo: Path,
    rel_paths: list[str],
    now_ns: int,
) -> None:
    for rel_path in rel_paths:
        abs_path = (global_repo / Path(rel_path)).expanduser()
        try:
            content_sha1 = _sha1_file(abs_path)
        except Exception:
            content_sha1 = ""
        try:
            st = abs_path.stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
        except Exception:
            mtime_ns = 0
        conn.execute(
            """
            UPDATE files
            SET last_processed_epoch_ns = ?,
                last_processed_mtime_ns = ?,
                content_sha1 = ?,
                needs_processing = 0
            WHERE rel_path = ?
            """,
            (now_ns, mtime_ns, content_sha1, rel_path),
        )


def _derive_data_file_metadata(excel_mod: Any | None, data_path: Path) -> dict:
    try:
        if excel_mod is not None:
            program, vehicle, serial = excel_mod.derive_file_identity(data_path)
        else:
            raise RuntimeError("no excel identity helper")
    except Exception:
        program, vehicle, serial = "", "", ""
    if program and vehicle:
        program_title = f"{program} {vehicle}".strip()
    else:
        program_title = (program or vehicle or "Unknown").strip()
    serial_number = (serial or "Unknown").strip() or "Unknown"
    is_mat = data_path.suffix.lower() in MAT_EXTENSIONS
    out = {
        "program_title": program_title,
        "asset_type": "Unknown",
        "serial_number": serial_number,
        "part_number": "Unknown",
        "revision": "Unknown",
        "test_date": "Unknown",
        "report_date": "Unknown",
        "vendor": "Unknown",
        "acceptance_test_plan_number": "Unknown",
    }
    if is_mat:
        out.update(
            {
                "document_type": "TD",
                "document_type_acronym": "TD",
                "document_type_status": "confirmed",
                "document_type_source": "ranker",
                "document_type_reason": "mat_extension_match",
                "document_type_evidence": [{"kind": "extension", "doc_type": "TD", "value": ".mat"}],
                "document_type_review_required": False,
            }
        )
    return out


def _is_test_data_meta(meta: dict[str, Any] | None) -> bool:
    src = meta if isinstance(meta, dict) else {}
    try:
        dt = str(src.get("document_type") or "").strip().lower()
    except Exception:
        dt = ""
    try:
        acr = str(src.get("document_type_acronym") or "").strip().lower()
    except Exception:
        acr = ""
    return dt in {"test data", "testdata", "td"} or acr in {"test data", "testdata", "td"}


def _is_confirmed_test_data_meta(meta: dict[str, Any] | None) -> bool:
    src = meta if isinstance(meta, dict) else {}
    if not _is_test_data_meta(src):
        return False
    try:
        status = str(src.get("document_type_status") or "").strip().lower()
    except Exception:
        status = ""
    try:
        review_required = bool(src.get("document_type_review_required"))
    except Exception:
        review_required = False
    return status == "confirmed" and not review_required


def _run_excel_extraction(excel_mod: Any, excel_path: Path, output_dir: Path, config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise RuntimeError(f"Excel trend config not found: {config_path}")
    config = excel_mod.load_config(config_path)
    rows = excel_mod.extract_from_excel(excel_path, config)
    excel_mod.write_outputs(rows, output_dir)
    return {
        "dir": str(output_dir),
        "rows_count": len(rows),
        "config": str(config_path),
    }


def _set_env_for_support(paths: SupportPaths) -> dict[str, str | None]:
    support_root = paths.support_dir
    merged_root = support_root / "debug" / "ocr"
    merged_root.mkdir(parents=True, exist_ok=True)

    prev: dict[str, str | None] = {}
    for k, v in {
        "MERGED_OCR_ROOT": str(merged_root),
        "OCR_CACHE_ROOT": str(support_root),
        "CACHE_ROOT": str(support_root),
    }.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = v
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        try:
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        except Exception:
            pass


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_process_file_filters(paths: SupportPaths, file_paths: list[str | Path] | None) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    repo = Path(paths.global_repo).expanduser()
    try:
        repo_res = repo.resolve()
    except Exception:
        repo_res = repo.absolute()

    for raw in file_paths or []:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        rel = ""
        if path.is_absolute():
            try:
                rel = path.resolve().relative_to(repo_res).as_posix()
            except Exception:
                rel = path.as_posix()
        else:
            rel = path.as_posix()
        rel = rel.replace("\\", "/").strip()
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        key = rel.casefold()
        if rel and key not in seen:
            seen.add(key)
            requested.append(rel)
    return requested


def process_candidates(
    paths: SupportPaths,
    *,
    limit: int | None = None,
    dpi: int | None = None,
    force: bool = False,
    only_candidates: bool = False,
    file_paths: list[str | Path] | None = None,
) -> list[ProcessResult]:
    now_ns = time.time_ns()
    core = None
    excel_mod = None
    excel_config_path = _default_excel_config_path()
    excel_sqlite_helpers = None

    def _get_core() -> Any:
        nonlocal core
        if core is None:
            core = _load_scanner_core()
        return core

    def _get_excel_mod() -> Any:
        nonlocal excel_mod
        if excel_mod is None:
            excel_mod = _load_excel_extractor()
        return excel_mod

    def _get_excel_sqlite_helpers() -> tuple[Any, Any, Any, Any, Any, Any]:
        nonlocal excel_sqlite_helpers
        if excel_sqlite_helpers is None:
            try:
                from eidat_manager_excel_to_sqlite import (  # type: ignore
                    _load_test_data_env,
                    _truthy,
                    excel_to_sqlite,
                    export_sqlite_excel_mirror,
                    export_sqlite_text_mirror,
                    write_mat_bundle_sqlite,
                )
            except Exception:
                from .eidat_manager_excel_to_sqlite import (  # type: ignore
                    _load_test_data_env,
                    _truthy,
                    excel_to_sqlite,
                    export_sqlite_excel_mirror,
                    export_sqlite_text_mirror,
                    write_mat_bundle_sqlite,
                )
            excel_sqlite_helpers = (
                excel_to_sqlite,
                export_sqlite_text_mirror,
                export_sqlite_excel_mirror,
                _load_test_data_env,
                _truthy,
                write_mat_bundle_sqlite,
            )
        return excel_sqlite_helpers

    with connect_db(paths.db_path) as conn:
        ensure_schema(conn)
        requested_rel_paths = _normalize_process_file_filters(paths, file_paths)
        missing_requested: list[str] = []
        params: list[object] = []
        where: list[str] = []
        if requested_rel_paths:
            placeholders = ", ".join("?" for _ in requested_rel_paths)
            tracked = conn.execute(
                f"SELECT rel_path FROM files WHERE rel_path IN ({placeholders})",
                list(requested_rel_paths),
            ).fetchall()
            tracked_rel_paths = {str(r["rel_path"] or "") for r in tracked}
            missing_requested = [rel for rel in requested_rel_paths if rel not in tracked_rel_paths]
            where.append(f"rel_path IN ({placeholders})")
            params.extend(requested_rel_paths)
        if bool(only_candidates) or not bool(force):
            where.append("needs_processing = 1")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(
            f"""
            SELECT rel_path, mtime_ns
            FROM files
            {where_sql}
            ORDER BY last_seen_epoch_ns DESC
            """,
            params,
        ).fetchall()

        results: list[ProcessResult] = [
            ProcessResult(
                rel_path=rel,
                abs_path=str((paths.global_repo / Path(rel)).expanduser()),
                ok=False,
                error="File is not tracked; run scan before processing this file.",
            )
            for rel in missing_requested
        ]
        bundle_outcomes: dict[str, ProcessResult] = {}
        prev_env = _set_env_for_support(paths)
        try:
            processed = 0
            attempted = 0
            total = len(rows)
            for row in rows:
                if limit is not None and processed >= int(limit):
                    break
                rel_path = str(row["rel_path"])
                if _ignore_rel_path(rel_path):
                    continue
                attempted += 1
                try:
                    print(f"[PROCESS] {attempted}/{total} start: {rel_path}", file=sys.stderr, flush=True)
                except Exception:
                    pass
                abs_path = (paths.global_repo / Path(rel_path)).expanduser()
                if rel_path in bundle_outcomes:
                    cached = bundle_outcomes[rel_path]
                    results.append(cached)
                    if cached.ok:
                        processed += 1
                    try:
                        print(f"[PROCESS] {attempted}/{total} done: {rel_path}", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    continue
                try:
                    if not abs_path.exists():
                        raise FileNotFoundError(f"Missing file: {abs_path}")
                    content_sha1 = _sha1_file(abs_path)

                    ext = abs_path.suffix.lower()
                    is_excel = ext in EXCEL_EXTENSIONS
                    is_mat = ext in MAT_EXTENSIONS
                    is_data_matrix = ext in DATA_MATRIX_EXTENSIONS

                    artifacts_dir = None
                    metadata_path = None
                    pointer_token = None
                    eidat_uuid = None

                    if is_data_matrix:
                        excel_mod = _get_excel_mod() if is_excel else None
                        bundle_seed = detect_mat_bundle_member(abs_path, repo_root=paths.global_repo) if is_mat else None
                        bundle_members = list_mat_bundle_members(abs_path, repo_root=paths.global_repo) if bundle_seed is not None else []
                        is_mat_bundle = bool(bundle_seed is not None and bundle_members)
                        artifacts_root = (
                            mat_bundle_artifacts_dir(paths.support_dir, bundle_seed)
                            if is_mat_bundle and bundle_seed is not None
                            else _excel_artifacts_dir(paths, abs_path)
                        )
                        artifacts_dir = str(artifacts_root)
                        metadata_identity_path = _bundle_identity_path(bundle_seed) if is_mat_bundle and bundle_seed is not None else abs_path

                        # Load any existing artifacts metadata first (treated as curated when present).
                        try:
                            existing_meta = load_metadata_from_artifacts(artifacts_root, metadata_identity_path)
                        except Exception:
                            existing_meta = None

                        # Extract metadata from workbook cells + filename (like PDFs use combined/title).
                        if is_mat_bundle and bundle_seed is not None:
                            per_member_meta: list[dict[str, Any]] = []
                            for member in bundle_members:
                                base_meta = _derive_data_file_metadata(None, member.file_path)
                                try:
                                    resolved_meta = canonicalize_metadata_for_file(
                                        member.file_path,
                                        existing_meta=None,
                                        extracted_meta=base_meta,
                                        default_document_type="TD",
                                    )
                                except Exception:
                                    resolved_meta = base_meta if isinstance(base_meta, dict) else {}
                                if isinstance(resolved_meta, dict):
                                    per_member_meta.append(dict(resolved_meta))
                            extracted_meta = _merge_bundle_metadata(per_member_meta)
                        elif is_excel:
                            try:
                                extracted_meta = extract_metadata_from_excel(abs_path)
                            except Exception:
                                extracted_meta = _derive_data_file_metadata(excel_mod, abs_path)
                        else:
                            extracted_meta = _derive_data_file_metadata(None, abs_path)

                        # Canonicalize once up front so TD detection is stable.
                        try:
                            raw_meta = canonicalize_metadata_for_file(
                                bundle_seed.file_path if is_mat_bundle and bundle_seed is not None else abs_path,
                                existing_meta=existing_meta,
                                extracted_meta=extracted_meta,
                                default_document_type="TD" if is_mat else "Unknown",
                            )
                        except Exception:
                            raw_meta = extracted_meta if isinstance(extracted_meta, dict) else {}
                        if is_mat_bundle and bundle_seed is not None and isinstance(raw_meta, dict):
                            folder_hints = _bundle_folder_asset_hints(bundle_seed)
                            raw_meta["serial_number"] = bundle_seed.serial_number
                            raw_meta["document_type"] = "TD"
                            raw_meta["document_type_acronym"] = "TD"
                            if str(raw_meta.get("asset_type") or "").strip() in {"", "Unknown", "unknown"}:
                                raw_meta["asset_type"] = folder_hints.get("asset_type") or "Unknown"
                            if str(raw_meta.get("asset_specific_type") or "").strip() in {"", "Unknown", "unknown"}:
                                raw_meta["asset_specific_type"] = folder_hints.get("asset_specific_type") or "Unknown"

                        is_test_data = bool(is_mat) or _is_test_data_meta(raw_meta if isinstance(raw_meta, dict) else {})
                        is_confirmed_test_data = bool(is_mat) or _is_confirmed_test_data_meta(
                            raw_meta if isinstance(raw_meta, dict) else {}
                        )

                        # Always run config-driven extraction as part of processing.
                        # If pandas is missing, allow Test Data flows to continue (SQLite is the key output).
                        if is_excel:
                            try:
                                res = _run_excel_extraction(
                                    excel_mod,
                                    abs_path,
                                    artifacts_root,
                                    excel_config_path,
                                )
                            except Exception:
                                if not is_test_data:
                                    raise
                                res = {"error": "excel_extraction_failed"}
                        else:
                            res = {"dir": str(artifacts_root), "rows_count": 0, "config": None}

                        # Test Data / MAT: create a per-source SQLite DB that downstream TD flows consume.
                        if is_test_data or is_mat:
                            try:
                                (
                                    excel_to_sqlite,
                                    export_sqlite_text_mirror,
                                    export_sqlite_excel_mirror,
                                    _load_test_data_env,
                                    _truthy,
                                    write_mat_bundle_sqlite,
                                ) = _get_excel_sqlite_helpers()
                                sqlite_rel = ""
                                sqlite_abs: Path | None = None
                                if is_mat_bundle and bundle_seed is not None:
                                    artifacts_root.mkdir(parents=True, exist_ok=True)
                                    _cleanup_stale_mat_member_artifacts(paths, bundle_members, artifacts_root)
                                    sqlite_abs = mat_bundle_sqlite_path(paths.support_dir, bundle_seed)
                                    write_mat_bundle_sqlite(
                                        mat_paths=[member.file_path for member in bundle_members],
                                        sqlite_path=sqlite_abs,
                                        overwrite=True,
                                    )
                                    sqlite_rel = _safe_repo_rel(paths.global_repo, sqlite_abs)
                                else:
                                    payload = excel_to_sqlite(
                                        global_repo=paths.global_repo,
                                        excel_files=[abs_path],
                                        data_dir=None,
                                        out_dir=Path(artifacts_root),
                                        overwrite=True,
                                        synthesize_td_seq_aliases=bool(is_confirmed_test_data),
                                    )
                                    try:
                                        results_list = list((payload or {}).get("results") or [])
                                    except Exception:
                                        results_list = []
                                    sqlite_errors: list[str] = []
                                    for r0 in results_list:
                                        try:
                                            sp = str((r0 or {}).get("sqlite_path") or "").strip()
                                        except Exception:
                                            sp = ""
                                        try:
                                            err = str((r0 or {}).get("error") or "").strip()
                                        except Exception:
                                            err = ""
                                        if err:
                                            sqlite_errors.append(err)
                                        if sp:
                                            try:
                                                sqlite_abs = Path(sp).expanduser()
                                            except Exception:
                                                sqlite_abs = None
                                            try:
                                                sqlite_rel = str(Path(sp).resolve().relative_to(paths.global_repo.resolve()))
                                            except Exception:
                                                sqlite_rel = sp
                                            break
                                if not sqlite_rel:
                                    detail = next((msg for msg in sqlite_errors if msg), "")
                                    if detail:
                                        raise RuntimeError(detail)
                                    raise RuntimeError("excel_to_sqlite did not report an output sqlite_path")
                                if isinstance(raw_meta, dict):
                                    raw_meta["excel_sqlite_rel"] = sqlite_rel

                                # Mirror the SQLite contents to a readable .txt next to the DB (artifact folder).
                                # This is intentionally best-effort so it doesn't fail the TD pipeline.
                                if sqlite_abs is not None:
                                    try:
                                        export_sqlite_text_mirror(sqlite_abs)
                                    except Exception:
                                        pass
                                    try:
                                        env = _load_test_data_env()
                                        want_xlsx = _truthy(env.get("EIDAT_TEST_DATA_SQLITE_MIRROR_XLSX", "1"))
                                    except Exception:
                                        want_xlsx = False
                                    if want_xlsx:
                                        try:
                                            export_sqlite_excel_mirror(sqlite_abs)
                                        except Exception:
                                            pass
                                if is_mat_bundle and bundle_seed is not None:
                                    member_rels = [_safe_repo_rel(paths.global_repo, member.file_path) for member in bundle_members]
                                    _mark_files_processed(
                                        conn,
                                        global_repo=paths.global_repo,
                                        rel_paths=member_rels,
                                        now_ns=now_ns,
                                    )
                                    for member_rel, member in zip(member_rels, bundle_members):
                                        if member_rel == rel_path:
                                            continue
                                        bundle_outcomes[member_rel] = ProcessResult(
                                            rel_path=member_rel,
                                            abs_path=str(member.file_path),
                                            ok=True,
                                            artifacts_dir=str(artifacts_root),
                                        )
                            except Exception as exc:
                                if is_mat_bundle and bundle_seed is not None:
                                    for member in bundle_members:
                                        member_rel = _safe_repo_rel(paths.global_repo, member.file_path)
                                        if member_rel == rel_path:
                                            continue
                                        bundle_outcomes[member_rel] = ProcessResult(
                                            rel_path=member_rel,
                                            abs_path=str(member.file_path),
                                            ok=False,
                                            error=f"MAT -> SQLite failed: {exc}",
                                        )
                                # For Test Data-like matrix sources, SQLite creation is required.
                                label = "MAT" if is_mat else "Test Data Excel"
                                raise RuntimeError(f"{label} -> SQLite failed: {exc}") from exc

                        # Final canonicalization pass to preserve any existing curated fields and fill blanks.
                        clean_meta = canonicalize_metadata_for_file(
                            bundle_seed.file_path if is_mat_bundle and bundle_seed is not None else abs_path,
                            existing_meta=existing_meta,
                            extracted_meta=raw_meta,
                            default_document_type="TD" if is_mat else "Unknown",
                        )
                        if is_mat_bundle and bundle_seed is not None:
                            folder_hints = _bundle_folder_asset_hints(bundle_seed)
                            clean_meta["serial_number"] = bundle_seed.serial_number
                            clean_meta["document_type"] = "TD"
                            clean_meta["document_type_acronym"] = "TD"
                            if str(clean_meta.get("asset_type") or "").strip() in {"", "Unknown", "unknown"}:
                                clean_meta["asset_type"] = folder_hints.get("asset_type") or "Unknown"
                            if str(clean_meta.get("asset_specific_type") or "").strip() in {"", "Unknown", "unknown"}:
                                clean_meta["asset_specific_type"] = folder_hints.get("asset_specific_type") or "Unknown"
                        metadata_path = write_metadata(Path(artifacts_dir), metadata_identity_path, clean_meta)
                        if is_mat_bundle and bundle_seed is not None:
                            _write_mat_bundle_manifest(
                                global_repo=paths.global_repo,
                                artifacts_dir=artifacts_root,
                                bundle=bundle_seed,
                                members=bundle_members,
                                sqlite_path=mat_bundle_sqlite_path(paths.support_dir, bundle_seed),
                                metadata_path=metadata_path,
                                metadata=clean_meta,
                            )
                    else:
                        core = _get_core()
                        # Use default debug-method pipeline (fused tables)
                        res = _run_clean_extraction(abs_path, dpi=dpi, output_dir=paths.support_dir)

                        combined_text = ""
                        combined_txt_path: Path | None = None
                        table_labels_done = False
                        try:
                            artifacts_dir = str(res.get("dir") or res.get("target_dir") or "")
                            artifacts_dir = artifacts_dir or None
                        except Exception:
                            artifacts_dir = None
                        try:
                            combined_path = res.get("combined")
                            if combined_path:
                                combined_txt_path = Path(combined_path)
                                combined_text = Path(combined_path).read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            combined_text = ""
                            combined_txt_path = None

                        # Post-processing (combined.txt-only): optional label-driven merge of multi-page run-on tables.
                        # This is intentionally decoupled from OCR and operates only on the final merged artifact.
                        if artifacts_dir and combined_txt_path is not None:
                            try:
                                from extraction.table_multipage_merger import (
                                    load_table_merge_heuristics,
                                    merge_multipage_tables_in_combined_lines,
                                )

                                heur_path = _resolve_user_inputs_file("table_merge_heuristics.json")
                                cfg = load_table_merge_heuristics(heur_path)
                                has_label_merge_rules = False
                                try:
                                    has_label_merge_rules = bool(cfg and isinstance(cfg.get("label_merge_rules"), list) and cfg.get("label_merge_rules"))
                                except Exception:
                                    has_label_merge_rules = False

                                if combined_txt_path.exists():
                                    raw_lines = combined_txt_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

                                    # If label-based merging is configured, ensure tables are labeled before merging.
                                    if has_label_merge_rules:
                                        try:
                                            from extraction.table_labeler import load_table_label_rules, label_combined_lines

                                            rules_path = _resolve_user_inputs_file("table_label_rules.json")
                                            rules_cfg = load_table_label_rules(rules_path)
                                            if rules_cfg:
                                                labeled0 = label_combined_lines(raw_lines, rules_cfg)
                                                if labeled0 and labeled0 != raw_lines:
                                                    combined_txt_path.write_text("".join(labeled0), encoding="utf-8")
                                                    raw_lines = labeled0
                                                table_labels_done = True
                                        except Exception:
                                            pass

                                    merged_lines = merge_multipage_tables_in_combined_lines(raw_lines, cfg=cfg)
                                    if merged_lines and merged_lines != raw_lines:
                                        combined_txt_path.write_text("".join(merged_lines), encoding="utf-8")
                                        raw_lines = merged_lines

                                    # Re-label after merge to remove orphaned label blocks and renumber consistently.
                                    if has_label_merge_rules:
                                        try:
                                            from extraction.table_labeler import load_table_label_rules, label_combined_lines

                                            rules_path = _resolve_user_inputs_file("table_label_rules.json")
                                            rules_cfg = load_table_label_rules(rules_path)
                                            if rules_cfg:
                                                labeled1 = label_combined_lines(raw_lines, rules_cfg)
                                                if labeled1 and labeled1 != raw_lines:
                                                    combined_txt_path.write_text("".join(labeled1), encoding="utf-8")
                                                    raw_lines = labeled1
                                                table_labels_done = True
                                        except Exception:
                                            pass

                                    combined_text = "".join(raw_lines) if raw_lines else combined_text
                            except Exception:
                                pass
                        extracted_meta = extract_metadata_from_text(combined_text, pdf_path=abs_path) if combined_text else None
                        embedded_meta = None
                        if extracted_meta is None:
                            embedded_meta = load_metadata_for_pdf(abs_path)
                        if extracted_meta is None and embedded_meta is None and artifacts_dir:
                            embedded_meta = load_metadata_from_artifacts(Path(artifacts_dir), abs_path)
                        if extracted_meta is None and embedded_meta is None:
                            embedded_meta = derive_minimal_metadata(core, abs_path)

                        existing_meta = None
                        if artifacts_dir:
                            try:
                                existing_meta = load_metadata_from_artifacts(Path(artifacts_dir), abs_path)
                            except Exception:
                                existing_meta = None

                        clean_meta = canonicalize_metadata_for_file(
                            abs_path,
                            existing_meta=existing_meta,
                            extracted_meta=(extracted_meta if extracted_meta is not None else embedded_meta),
                            default_document_type="Unknown",
                        )
                        if artifacts_dir:
                            metadata_path = write_metadata(Path(artifacts_dir), abs_path, clean_meta)

                        # Best-effort: apply rules-driven table labels to combined.txt (post-processing).
                        # This overwrites combined.txt in-place and is intentionally decoupled from
                        # extracted_terms.db export and metadata extraction.
                        if artifacts_dir and combined_txt_path is not None:
                            if not table_labels_done:
                                try:
                                    from extraction.table_labeler import load_table_label_rules, label_combined_lines

                                    rules_path = _resolve_user_inputs_file("table_label_rules.json")
                                    rules_cfg = load_table_label_rules(rules_path)
                                    if rules_cfg and combined_txt_path.exists():
                                        raw_lines = combined_txt_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
                                        labeled = label_combined_lines(raw_lines, rules_cfg)
                                        if labeled and labeled != raw_lines:
                                            combined_txt_path.write_text("".join(labeled), encoding="utf-8")
                                except Exception:
                                    pass

                        # Best-effort: heal table cells inside combined.txt (post-processing).
                        # Runs after table labeling + multipage merge so we can key by [TABLE_LABEL] later.
                        if artifacts_dir and combined_txt_path is not None and combined_txt_path.exists():
                            try:
                                enable_heal = str(os.environ.get("EIDAT_TABLE_CELL_HEAL", "1") or "1").strip().lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                    "on",
                                )
                            except Exception:
                                enable_heal = True
                            if enable_heal:
                                try:
                                    from extraction.table_cell_healer import (
                                        heal_combined_txt_file_inplace,
                                        JsonlDebugSink,
                                        JsonSidecarVariantCandidatesProvider,
                                        load_table_cell_heal_heuristics,
                                    )

                                    heal_path = _resolve_user_inputs_file("table_cell_heal_heuristics.json")
                                    heal_cfg = load_table_cell_heal_heuristics(heal_path)
                                    variant_provider = None
                                    try:
                                        enable_candidates = _parse_bool_env("EIDAT_TABLE_VARIANT_CANDIDATES", True)
                                    except Exception:
                                        enable_candidates = True
                                    if enable_candidates and artifacts_dir:
                                        try:
                                            numeric_cfg = heal_cfg.get("numeric") if isinstance(heal_cfg.get("numeric"), dict) else {}
                                            rescue_cfg = (
                                                numeric_cfg.get("variant_rescue")
                                                if isinstance(numeric_cfg.get("variant_rescue"), dict)
                                                else {}
                                            )
                                            if bool(rescue_cfg.get("enabled", False)):
                                                sidecar_name = str(rescue_cfg.get("sidecar_filename") or "table_variant_candidates.json")
                                                variant_provider = JsonSidecarVariantCandidatesProvider(
                                                    Path(artifacts_dir),
                                                    sidecar_filename=sidecar_name,
                                                )
                                        except Exception:
                                            variant_provider = None

                                    variant_debug = None
                                    try:
                                        enable_variant_debug = _parse_bool_env("EIDAT_TABLE_CELL_HEAL_VARIANT_DEBUG", False) or _parse_bool_env(
                                            "EIDAT_TABLE_CELL_HEAL_DEBUG", False
                                        )
                                    except Exception:
                                        enable_variant_debug = False
                                    if enable_variant_debug and artifacts_dir:
                                        try:
                                            debug_path = Path(artifacts_dir) / "table_cell_heal_variant_rescue.jsonl"
                                            variant_debug = JsonlDebugSink(debug_path, truncate=True)
                                        except Exception:
                                            variant_debug = None

                                    heal_combined_txt_file_inplace(
                                        combined_txt_path,
                                        cfg=heal_cfg,
                                        history=None,
                                        variant_provider=variant_provider,
                                        variant_debug=variant_debug,
                                    )
                                except Exception:
                                    pass

                        # Export extracted_terms.db from the final post-processed combined.txt.
                        if artifacts_dir and combined_txt_path is not None:
                            _export_extracted_terms_db(Path(artifacts_dir), combined_txt_path)
                            _export_labeled_tables_db(Path(artifacts_dir), combined_txt_path)

                        if force or not has_pointer_token(abs_path):
                            eidat_uuid = uuid.uuid4().hex
                            support_rel = None
                            artifacts_rel = None
                            metadata_rel = None
                            try:
                                support_rel = str(paths.support_dir.resolve().relative_to(paths.global_repo.resolve()))
                            except Exception:
                                support_rel = str(paths.support_dir)
                            if artifacts_dir:
                                try:
                                    artifacts_rel = str(Path(artifacts_dir).resolve().relative_to(paths.global_repo.resolve()))
                                except Exception:
                                    artifacts_rel = artifacts_dir
                            if metadata_path:
                                try:
                                    metadata_rel = str(metadata_path.resolve().relative_to(paths.global_repo.resolve()))
                                except Exception:
                                    metadata_rel = str(metadata_path)
                            payload = {
                                "eidat_uuid": eidat_uuid,
                                "support_rel": support_rel,
                                "artifacts_rel": artifacts_rel,
                                "metadata_rel": metadata_rel,
                                "processed_epoch_ns": now_ns,
                            }
                            pointer_token = build_pointer_token(payload)
                            embedded = embed_pointer_token(abs_path, pointer_token, overwrite=force)
                            if not embedded and not force:
                                pointer_token = None
                                eidat_uuid = None
                    conn.execute(
                        """
                        UPDATE files
                        SET last_processed_epoch_ns = ?,
                            last_processed_mtime_ns = ?,
                            content_sha1 = ?,
                            eidat_uuid = COALESCE(?, eidat_uuid),
                            pointer_token = COALESCE(?, pointer_token),
                            needs_processing = 0
                        WHERE rel_path = ?
                        """,
                        (now_ns, int(row["mtime_ns"] or 0), content_sha1, eidat_uuid, pointer_token, rel_path),
                    )
                    if not is_excel and artifacts_dir:
                        # Run certification analysis on extracted terms (PDF-only)
                        try:
                            from extraction.certification_analyzer import analyze_artifacts_folder
                            analyze_artifacts_folder(Path(artifacts_dir))
                        except Exception:
                            pass  # Certification failure shouldn't fail processing

                    results.append(
                        ProcessResult(
                            rel_path=rel_path,
                            abs_path=str(abs_path),
                            ok=True,
                            artifacts_dir=artifacts_dir,
                        )
                    )
                    processed += 1
                    try:
                        print(f"[PROCESS] {attempted}/{total} done: {rel_path}", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                except Exception as exc:
                    results.append(
                        ProcessResult(
                            rel_path=rel_path,
                            abs_path=str(abs_path),
                            ok=False,
                            error=str(exc),
                        )
                    )
                    try:
                        print(f"[PROCESS] {attempted}/{total} failed: {rel_path} ({exc})", file=sys.stderr, flush=True)
                    except Exception:
                        pass
            conn.commit()
        finally:
            _restore_env(prev_env)

    return results
