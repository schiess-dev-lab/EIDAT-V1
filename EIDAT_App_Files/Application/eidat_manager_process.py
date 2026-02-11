from __future__ import annotations

import importlib.util
import json
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
    derive_minimal_metadata,
    extract_metadata_from_text,
    load_metadata_for_pdf,
    load_metadata_from_artifacts,
    sanitize_metadata,
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


def _run_debug_method_extraction(pdf_path: Path, dpi: int | None, output_dir: Path) -> dict[str, Any]:
    """
    New default extraction path:
    - Render pages to PNG
    - Draw table borders (debug_method/table_grid_debug)
    - Run table variants on bordered PNGs (fused only)
    - Build combined.txt with existing formatter
    """
    # Add EIDAT_App_Files to path for extraction imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from extraction import ocr_engine, token_projector, page_analyzer, debug_exporter
    except ImportError as e:
        raise RuntimeError(f"Unable to import extraction modules: {e}") from e

    try:
        from debug_method import table_grid_debug, run_table_variants
        from debug_method.run_debug_master import _draw_table_borders
    except ImportError as e:
        raise RuntimeError(f"Unable to import debug_method pipeline: {e}") from e

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

    # Table-grid overlays are debug-only and have been unstable (false positive borders / crashes).
    # Default to disabling all overlay line projection unless explicitly enabled.
    tg_draw_overlay_lines = _parse_bool_env("EIDAT_TABLE_GRID_DRAW_LINES", False)
    tg_apply_borders_to_page = _parse_bool_env("EIDAT_TABLE_GRID_APPLY_BORDERS_TO_PAGE", False)

    table_variants_lang = table_variants_cfg.get("lang") or None
    if table_variants_lang is None:
        table_variants_lang = ocr_engine.get_tesseract_lang()

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

    page_count = _render_pdf_pages_to_dirs(pdf_path, pages_root, render_dpi)
    if page_count <= 0:
        raise RuntimeError(f"No pages rendered for {pdf_path}")

    pages_data: List[Dict] = []
    ocr_lang = ocr_engine.get_tesseract_lang()

    for page_num in range(1, page_count + 1):
        page_dir = pages_root / f"page_{page_num}"
        page_path = page_dir / f"page_{page_num}.png"
        if not page_path.exists():
            raise RuntimeError(f"Missing rendered page: {page_path}")

        if table_grid_enabled:
            grid_dir = page_dir / "grid_debug"
            gap_override = None
            if tg_gap_threshold and tg_gap_threshold > 0:
                gap_override = float(tg_gap_threshold)
            elif tg_min_gap and float(tg_min_gap) > 0:
                gap_override = float(tg_min_gap)
            summary = table_grid_debug.run_for_image(
                page_path,
                out_dir=grid_dir,
                merge_kx=tg_merge_kx,
                min_gap=gap_override if gap_override else None,
                min_gap_ratio=tg_min_gap_ratio,
                offset_px=tg_offset_px,
                line_thickness=tg_line_thickness,
                line_pad_factor=tg_line_pad,
                min_token_h_px=tg_min_token_h,
                min_token_h_ratio=tg_min_token_h_ratio,
                draw_tables=bool(tg_draw_overlay_lines),
                draw_hlines=bool(tg_draw_overlay_lines and tg_draw_hlines),
                draw_seps_in_tables=bool(tg_draw_overlay_lines and tg_draw_seps_in_tables),
                draw_separators=bool(tg_draw_overlay_lines and tg_draw_separators),
            )
            tables_grid = summary.get("tables") or []
            if tg_apply_borders_to_page and tables_grid:
                _draw_table_borders(
                    page_path,
                    tables_grid,
                    page_path,
                    line_thickness=tg_border_thickness,
                )

        fused_result = run_table_variants._run_for_input(
            page_path,
            out_dir=page_dir,
            page=1,
            ocr_dpi_base=table_ocr_dpi,
            detection_dpi=detection_dpi,
            lang=table_variants_lang,
            clean=False,
            fuse=True,
            emit_variants=False,
            emit_fused=True,
            allow_no_tables=True,
            enable_borderless=False,
            return_fused=True,
        )

        fused_tables = list(fused_result.get("tables") or [])
        detection_dpi_used = int(fused_result.get("detection_dpi") or detection_dpi)

        tokens, ocr_img_w, ocr_img_h, _img_path = ocr_engine.ocr_page(
            pdf_path, page_num - 1, ocr_dpi, ocr_lang, 3, debug_dir=None
        )
        try:
            img_gray_ocr, _, _ = ocr_engine.render_pdf_page(pdf_path, page_num - 1, ocr_dpi)
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

        flow_tokens = token_projector.scale_tokens_to_dpi(tokens, ocr_dpi, detection_dpi_used) if tokens else []
        flow_data = page_analyzer.extract_flow_text(flow_tokens, fused_tables, det_w, det_h)

        charts: List[Dict] = []
        enable_charts = _parse_bool_env("EIDAT_ENABLE_CHART_EXTRACTION", True)
        if enable_charts and det_img is not None:
            try:
                from extraction import chart_detection

                charts = chart_detection.detect_charts(
                    det_img, flow_tokens, fused_tables, det_w, det_h, flow_data
                )
            except Exception:
                charts = []

        page_data = {
            "page": page_num,
            "tokens": tokens,
            "tables": fused_tables,
            "charts": charts,
            "img_w": det_w,
            "img_h": det_h,
            "dpi": detection_dpi_used,
            "ocr_dpi": ocr_dpi,
            "flow": flow_data,
        }
        pages_data.append(page_data)

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
            ocr_dpi=ocr_dpi,
            ocr_img_w=ocr_img_w,
            ocr_img_h=ocr_img_h,
        )

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


def _excel_artifacts_dir(paths: SupportPaths, excel_path: Path) -> Path:
    return paths.support_dir / "debug" / "ocr" / f"{excel_path.stem}{EXCEL_ARTIFACT_SUFFIX}"


def _derive_excel_metadata(excel_mod: Any, excel_path: Path) -> dict:
    try:
        program, vehicle, serial = excel_mod.derive_file_identity(excel_path)
    except Exception:
        program, vehicle, serial = "", "", ""
    if program and vehicle:
        program_title = f"{program} {vehicle}".strip()
    else:
        program_title = (program or vehicle or "Unknown").strip()
    serial_number = (serial or "Unknown").strip() or "Unknown"
    return {
        "program_title": program_title,
        "asset_type": "Unknown",
        "serial_number": serial_number,
        "part_number": "Unknown",
        "revision": "Unknown",
        "test_date": "Unknown",
        "report_date": "Unknown",
        "document_type": "Data file",
        "document_type_acronym": "DATA",
        "vendor": "Unknown",
        "acceptance_test_plan_number": "Unknown",
    }


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
            for row in rows:
                if limit is not None and processed >= int(limit):
                    break
                rel_path = str(row["rel_path"])
                if _ignore_rel_path(rel_path):
                    continue
                abs_path = (paths.global_repo / Path(rel_path)).expanduser()
                try:
                    if not abs_path.exists():
                        raise FileNotFoundError(f"Missing file: {abs_path}")
                    content_sha1 = _sha1_file(abs_path)

                    ext = abs_path.suffix.lower()
                    is_excel = ext in EXCEL_EXTENSIONS

                    artifacts_dir = None
                    metadata_path = None
                    pointer_token = None
                    eidat_uuid = None

                    if is_excel:
                        excel_mod = _get_excel_mod()
                        artifacts_root = _excel_artifacts_dir(paths, abs_path)
                        res = _run_excel_extraction(
                            excel_mod,
                            abs_path,
                            artifacts_root,
                            excel_config_path,
                        )
                        artifacts_dir = str(artifacts_root)
                        raw_meta = _derive_excel_metadata(excel_mod, abs_path)
                        clean_meta = sanitize_metadata(raw_meta, default_document_type="Data file")
                        metadata_path = write_metadata(Path(artifacts_dir), abs_path, clean_meta)
                    else:
                        core = _get_core()
                        # Use default debug-method pipeline (fused tables)
                        res = _run_clean_extraction(abs_path, dpi=dpi, output_dir=paths.support_dir)

                        combined_text = ""
                        combined_txt_path: Path | None = None
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

                        if artifacts_dir and combined_txt_path is not None:
                            _export_extracted_terms_db(Path(artifacts_dir), combined_txt_path)
                        raw_meta = extract_metadata_from_text(combined_text, pdf_path=abs_path) if combined_text else None
                        if raw_meta is None:
                            raw_meta = load_metadata_for_pdf(abs_path)
                        if raw_meta is None and artifacts_dir:
                            raw_meta = load_metadata_from_artifacts(Path(artifacts_dir), abs_path)
                        if raw_meta is None:
                            raw_meta = derive_minimal_metadata(core, abs_path)
                        clean_meta = sanitize_metadata(raw_meta, default_document_type="EIDP")
                        if artifacts_dir:
                            metadata_path = write_metadata(Path(artifacts_dir), abs_path, clean_meta)

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
                except Exception as exc:
                    results.append(
                        ProcessResult(
                            rel_path=rel_path,
                            abs_path=str(abs_path),
                            ok=False,
                            error=str(exc),
                        )
                    )
            conn.commit()
        finally:
            _restore_env(prev_env)

    return results
