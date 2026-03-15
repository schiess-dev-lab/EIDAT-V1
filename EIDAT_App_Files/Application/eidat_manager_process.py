from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import uuid

from eidat_manager_db import SupportPaths, connect_db, ensure_schema
from eidat_manager_embed import build_pointer_token, embed_pointer_token, has_pointer_token
from eidat_manager_metadata import (
    canonicalize_metadata_for_file,
    derive_minimal_metadata,
    extract_metadata_from_excel,
    extract_metadata_from_text,
    load_metadata_for_pdf,
    load_metadata_from_artifacts,
    write_metadata,
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
    masked_table_page_path = page_path
    try:
        import cv2  # type: ignore

        page_img_gray = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
        if page_img_gray is not None:
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
    return paths.support_dir / "debug" / "ocr" / f"{excel_path.stem}{EXCEL_ARTIFACT_SUFFIX}"


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


def process_candidates(
    paths: SupportPaths,
    *,
    limit: int | None = None,
    dpi: int | None = None,
    force: bool = False,
    only_candidates: bool = False,
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

    def _get_excel_sqlite_helpers() -> tuple[Any, Any, Any, Any, Any]:
        nonlocal excel_sqlite_helpers
        if excel_sqlite_helpers is None:
            try:
                from eidat_manager_excel_to_sqlite import (  # type: ignore
                    _load_test_data_env,
                    _truthy,
                    excel_to_sqlite,
                    export_sqlite_excel_mirror,
                    export_sqlite_text_mirror,
                )
            except Exception:
                from .eidat_manager_excel_to_sqlite import (  # type: ignore
                    _load_test_data_env,
                    _truthy,
                    excel_to_sqlite,
                    export_sqlite_excel_mirror,
                    export_sqlite_text_mirror,
                )
            excel_sqlite_helpers = (
                excel_to_sqlite,
                export_sqlite_text_mirror,
                export_sqlite_excel_mirror,
                _load_test_data_env,
                _truthy,
            )
        return excel_sqlite_helpers

    with connect_db(paths.db_path) as conn:
        ensure_schema(conn)
        if bool(only_candidates):
            rows = conn.execute(
                """
                SELECT rel_path, mtime_ns
                FROM files
                WHERE needs_processing = 1
                ORDER BY last_seen_epoch_ns DESC
                """
            ).fetchall()
        elif force:
            rows = conn.execute(
                """
                SELECT rel_path, mtime_ns
                FROM files
                ORDER BY last_seen_epoch_ns DESC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT rel_path, mtime_ns
                FROM files
                WHERE needs_processing = 1
                ORDER BY last_seen_epoch_ns DESC
                """
            ).fetchall()

        results: list[ProcessResult] = []
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
                        artifacts_root = _excel_artifacts_dir(paths, abs_path)
                        artifacts_dir = str(artifacts_root)

                        # Load any existing artifacts metadata first (treated as curated when present).
                        try:
                            existing_meta = load_metadata_from_artifacts(artifacts_root, abs_path)
                        except Exception:
                            existing_meta = None

                        # Extract metadata from workbook cells + filename (like PDFs use combined/title).
                        if is_excel:
                            try:
                                extracted_meta = extract_metadata_from_excel(abs_path)
                            except Exception:
                                extracted_meta = _derive_data_file_metadata(excel_mod, abs_path)
                        else:
                            extracted_meta = _derive_data_file_metadata(None, abs_path)

                        # Canonicalize once up front so TD detection is stable.
                        try:
                            raw_meta = canonicalize_metadata_for_file(
                                abs_path,
                                existing_meta=existing_meta,
                                extracted_meta=extracted_meta,
                                default_document_type="TD" if is_mat else "Unknown",
                            )
                        except Exception:
                            raw_meta = extracted_meta if isinstance(extracted_meta, dict) else {}

                        is_test_data = bool(is_mat) or _is_test_data_meta(raw_meta if isinstance(raw_meta, dict) else {})

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
                                ) = _get_excel_sqlite_helpers()
                                payload = excel_to_sqlite(
                                    global_repo=paths.global_repo,
                                    excel_files=[abs_path],
                                    data_dir=None,
                                    out_dir=Path(artifacts_root),
                                    overwrite=True,
                                )
                                sqlite_rel = ""
                                sqlite_abs: Path | None = None
                                try:
                                    results_list = list((payload or {}).get("results") or [])
                                except Exception:
                                    results_list = []
                                for r0 in results_list:
                                    try:
                                        sp = str((r0 or {}).get("sqlite_path") or "").strip()
                                    except Exception:
                                        sp = ""
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
                            except Exception as exc:
                                # For Test Data-like matrix sources, SQLite creation is required.
                                label = "MAT" if is_mat else "Test Data Excel"
                                raise RuntimeError(f"{label} -> SQLite failed: {exc}") from exc

                        # Final canonicalization pass to preserve any existing curated fields and fill blanks.
                        clean_meta = canonicalize_metadata_for_file(
                            abs_path,
                            existing_meta=existing_meta,
                            extracted_meta=raw_meta,
                            default_document_type="TD" if is_mat else "Unknown",
                        )
                        metadata_path = write_metadata(Path(artifacts_dir), abs_path, clean_meta)
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
