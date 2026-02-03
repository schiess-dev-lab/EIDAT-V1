#!/usr/bin/env python3
"""
Input-agnostic table OCR variant runner (PNG or PDF).

Detect tables once at detection DPI, then run the current pipeline variants
independently and write table-only outputs per variant (JSON + TXT).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _ensure_app_path(repo_root: Path) -> None:
    app_root = repo_root / "EIDAT_App_Files"
    if not app_root.exists() and (repo_root / "extraction").exists():
        app_root = repo_root
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))


def _load_gray_image(path: Path):
    try:
        import cv2  # type: ignore
    except Exception:
        raise RuntimeError("OpenCV (cv2) is required for this script.")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _resize_image(img, scale: float):
    import cv2  # type: ignore
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _read_png_dpi(path: Path) -> Optional[float]:
    if path.suffix.lower() != ".png":
        return None
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            dpi = img.info.get("dpi")
            if not dpi:
                return None
            if isinstance(dpi, (list, tuple)) and len(dpi) >= 1:
                val = float(dpi[0])
            else:
                val = float(dpi)
            if val <= 0:
                return None
            return val
    except Exception:
        return None


def _parse_int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _parse_float_env(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    raw = str(os.environ.get(key, "")).strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _format_scale(scale: float) -> str:
    text = f"{scale:.2f}".rstrip("0").rstrip(".")
    return text or "1"

def _format_line_strip_tag(remove_lines: bool, line_strip_level: Optional[str]) -> str:
    if not remove_lines:
        return "rl0"
    level = str(line_strip_level or "default").strip().lower()
    if level in ("default", "medium", ""):
        return "rl1"
    if level in ("light", "lite", "soft", "low"):
        return "rl1light"
    safe = "".join(ch for ch in level if ch.isalnum())
    return f"rl1{safe}" if safe else "rl1"


def _gather_input_files(debug_dir: Path, input_arg: Optional[str]) -> List[Path]:
    if input_arg:
        path = Path(input_arg)
        if not path.exists():
            raise RuntimeError(f"Input not found: {path}")
        if path.is_dir():
            matches = sorted(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in (".png", ".pdf")
            )
            if not matches:
                raise RuntimeError(f"No .png or .pdf files found in {path}")
            return matches
        return [path]

    candidate_dir = debug_dir / "debug_file"
    if not candidate_dir.exists():
        raise RuntimeError(f"Input folder not found: {candidate_dir}")
    matches = sorted(
        p for p in candidate_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".pdf")
    )
    if not matches:
        raise RuntimeError(f"No .png or .pdf files found in {candidate_dir}")
    return matches


def _scale_bbox_to_ocr(
    bbox: List[float],
    detection_dpi: int,
    ocr_dpi: int,
    pad_px: int = 0,
) -> Tuple[int, int, int, int]:
    scale = float(detection_dpi) / float(ocr_dpi)
    x0, y0, x1, y1 = bbox
    return (
        int(float(x0) / scale) - pad_px,
        int(float(y0) / scale) - pad_px,
        int(float(x1) / scale) + pad_px,
        int(float(y1) / scale) + pad_px,
    )


def _clip_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _tokens_to_text(tokens_list: List[Dict]) -> str:
    try:
        from extraction import ocr_engine
        return ocr_engine._sort_tokens_into_text(tokens_list)  # type: ignore[attr-defined]
    except Exception:
        ordered = sorted(
            tokens_list,
            key=lambda t: (
                float(t.get("cy", t.get("y0", 0))),
                float(t.get("x0", 0)),
            ),
        )
        return " ".join(str(t.get("text", "")).strip() for t in ordered).strip()


def _clone_cells(base_cells: List[Dict]) -> List[Dict]:
    out = []
    for c in base_cells:
        cell = {
            "bbox_px": list(c.get("bbox_px") or []),
            "row": c.get("row"),
            "col": c.get("col"),
        }
        if "borderless" in c:
            cell["borderless"] = bool(c.get("borderless"))
        out.append(cell)
    return out


def _cells_to_rows(cells: List[Dict]) -> List[List[str]]:
    max_row = -1
    max_col = -1
    for cell in cells:
        row = cell.get("row")
        col = cell.get("col")
        if row is None or col is None:
            continue
        try:
            row_i = int(row)
            col_i = int(col)
        except (TypeError, ValueError):
            continue
        max_row = max(max_row, row_i)
        max_col = max(max_col, col_i)
    if max_row < 0 or max_col < 0:
        return []
    rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        row = cell.get("row")
        col = cell.get("col")
        if row is None or col is None:
            continue
        try:
            row_i = int(row)
            col_i = int(col)
        except (TypeError, ValueError):
            continue
        text_val = str(cell.get("text", "")).strip()
        if 0 <= row_i <= max_row and 0 <= col_i <= max_col:
            rows[row_i][col_i] = text_val
    return rows


def _build_variants(ocr_dpi: int, detection_dpi: int) -> List[Dict]:
    variants: List[Dict] = []

    table_ocr_psms = (11, 6)
    table_ocr_scales = (0.66, 1.0)
    line_strip_levels = ("default", "light")

    for scale in table_ocr_scales:
        for psm in table_ocr_psms:
            if psm != 6:
                variants.append({
                    "method": "table_region_ocr",
                    "ocr_mode": "table_region",
                    "psm": int(psm),
                    "remove_lines": False,
                    "line_strip_level": None,
                    "scale": float(scale),
                })
            for level in line_strip_levels:
                variants.append({
                    "method": "table_region_ocr",
                    "ocr_mode": "table_region",
                    "psm": int(psm),
                    "remove_lines": True,
                    "line_strip_level": str(level),
                    "scale": float(scale),
                })

    return variants


def _write_run_summary(
    summary_path: Path,
    *,
    input_paths: List[Path],
    out_dir_base: Path,
    ocr_dpi_base: int,
    detection_dpi: int,
    variants: List[Dict],
    fuse: bool,
    borderless_enabled: bool = False,
) -> None:
    lines: List[str] = []
    lines.append("Run Summary (auto-generated)")
    lines.append("")
    lines.append("Inputs:")
    for p in input_paths:
        lines.append(f"- {p}")
    lines.append("")
    lines.append(f"Output root: {out_dir_base}")
    lines.append("Per-input outputs: <output root>/<input_stem>/")
    lines.append("")
    lines.append(f"Base OCR DPI: {int(ocr_dpi_base)}")
    lines.append(f"Detection DPI (table geometry): {int(detection_dpi)}")
    lines.append(f"Fuse enabled: {bool(fuse)}")
    lines.append("")
    lines.append("Variants:")
    for v in variants:
        method = v.get("method")
        psm = v.get("psm")
        scale = v.get("scale")
        remove_lines = bool(v.get("remove_lines", False))
        line_strip_level = v.get("line_strip_level")
        scale_text = f"scale={scale}" if scale is not None else "scale=?"
        line_text = "rl0" if not remove_lines else f"rl1({line_strip_level})"
        lines.append(f"- {method} psm={psm} {scale_text} {line_text}")
    lines.append("")
    lines.append("Line-strip debug images (per input):")
    lines.append("- Only nolines images are saved.")
    lines.append("- Pattern: line_strip_debug/*_rl1*_table*_nolines.png")
    lines.append("")
    lines.append("Borderless tables:")
    if borderless_enabled:
        lines.append("Borderless tables detected using full-page OCR tokens (psm=3 @ base OCR DPI),")
        lines.append("then merged into base tables for all variants.")
    else:
        lines.append("Borderless table detection is disabled (tables are assumed to be bordered).")
    lines.append("")
    lines.append("Fuse (--fuse) logic:")
    lines.append("- Per table cell, collect the candidate text from every variant.")
    lines.append("- If any non-empty candidates exist, pick the most frequent non-empty value.")
    lines.append("- If all candidates are empty, keep empty.")
    lines.append("- No confidence weighting, normalization, or row/col context.")
    lines.append("")
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_for_input(
    input_path: Path,
    *,
    out_dir: Path,
    page: int,
    ocr_dpi_base: int,
    detection_dpi: int,
    lang: Optional[str],
    clean: bool,
    fuse: bool,
    emit_variants: bool = True,
    emit_fused: bool = True,
    allow_no_tables: bool = False,
    enable_borderless: bool = False,
    return_fused: bool = False,
) -> object:
    from extraction import table_detection, token_projector, ocr_engine, debug_exporter
    if enable_borderless:
        from extraction import borderless_table_detection

    out_dir.mkdir(parents=True, exist_ok=True)
    if clean and out_dir.exists():
        for path in sorted(out_dir.rglob("*"), reverse=True):
            try:
                if path.is_file():
                    path.unlink()
            except Exception:
                pass

    input_suffix = input_path.suffix.lower()
    is_pdf = input_suffix == ".pdf"
    is_png = input_suffix == ".png"
    if not (is_pdf or is_png):
        raise RuntimeError("Input must be a .png or .pdf")

    lang = lang or ocr_engine.get_tesseract_lang()

    base_dpi = None
    detection_img = None
    detection_scale = 1.0
    input_size = None
    detection_size = None

    pdf_image_cache: Dict[int, object] = {}
    png_image_cache: Dict[float, object] = {}
    base_img = None

    if is_pdf:
        page_index = max(0, int(page) - 1)

        def _get_pdf_image(dpi: int):
            if dpi in pdf_image_cache:
                return pdf_image_cache[dpi]
            img_gray, _, _ = ocr_engine.render_pdf_page(input_path, page_index, dpi)
            if img_gray is None:
                raise RuntimeError(f"Failed to render PDF at {dpi} DPI")
            pdf_image_cache[dpi] = img_gray
            return img_gray

        detection_img = _get_pdf_image(detection_dpi)
        detection_size = {"w": int(detection_img.shape[1]), "h": int(detection_img.shape[0])}
    else:
        base_img = _load_gray_image(input_path)
        input_size = {"w": int(base_img.shape[1]), "h": int(base_img.shape[0])}
        base_dpi = _read_png_dpi(input_path)
        if base_dpi:
            if detection_dpi > float(base_dpi):
                detection_dpi = int(round(float(base_dpi)))
                detection_scale = 1.0
            else:
                detection_scale = float(detection_dpi) / float(base_dpi)
        detection_img = _resize_image(base_img, detection_scale)
        detection_size = {"w": int(detection_img.shape[1]), "h": int(detection_img.shape[0])}

        def _get_png_image(scale: float):
            key = round(scale, 6)
            if key in png_image_cache:
                return png_image_cache[key]
            img_scaled = _resize_image(base_img, scale)
            png_image_cache[key] = img_scaled
            return img_scaled

    if detection_img is None:
        raise RuntimeError("Failed to prepare detection image")

    # Detect tables once (fixed geometry)
    table_result = table_detection.detect_tables(detection_img, verbose=False)
    tables = table_result.get("tables") or []

    # Borderless table detection is disabled by default (bordered tables expected).
    if enable_borderless:
        borderless_tables: List[Dict] = []
        try:
            ocr_img_full = None
            if is_pdf:
                ocr_img_full = _get_pdf_image(ocr_dpi_base)
            else:
                if base_dpi:
                    ocr_scale = float(ocr_dpi_base) / float(base_dpi)
                else:
                    # Assume no-DPI PNGs were rendered at base OCR DPI already.
                    ocr_scale = 1.0
                ocr_img_full = _get_png_image(ocr_scale)

            if ocr_img_full is not None:
                ocr_h_full, ocr_w_full = int(ocr_img_full.shape[0]), int(ocr_img_full.shape[1])
                tokens, _meta = ocr_engine.ocr_region_tokens(
                    ocr_img_full,
                    (0, 0, ocr_w_full, ocr_h_full),
                    lang=lang,
                    psms=(3,),
                    remove_lines=False,
                )
                if tokens:
                    if base_dpi is None:
                        scaled_tokens = tokens
                    else:
                        scaled_tokens = token_projector.scale_tokens_to_dpi(
                            tokens, ocr_dpi_base, detection_dpi
                        )
                    det_w = int(detection_size["w"]) if detection_size else int(detection_img.shape[1])
                    det_h = int(detection_size["h"]) if detection_size else int(detection_img.shape[0])
                    borderless_tables = borderless_table_detection.detect_borderless_tables(
                        scaled_tokens, det_w, det_h, tables, img_gray=detection_img
                    )
        except Exception:
            borderless_tables = []

        if borderless_tables:
            tables.extend(borderless_tables)
            if emit_variants:
                try:
                    debug_dir = out_dir / "line_strip_debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    page_num = int(page) if is_pdf else 1
                    debug_exporter.export_borderless_table_debug_images(
                        detection_img, borderless_tables, debug_dir, page_num
                    )
                except Exception:
                    pass

    base_tables: List[Dict] = []
    for t in tables:
        base_cells = [dict(c) for c in (t.get("cells") or [])]
        token_projector.assign_row_col_indices(base_cells)
        base_tables.append({
            "bbox_px": list(t.get("bbox_px") or []),
            "cells": base_cells,
            "borderless": bool(t.get("borderless", False)),
        })
    if not base_tables:
        if not allow_no_tables:
            raise RuntimeError("No tables detected in input")
        # Optionally emit an empty fused payload for consistency.
        if emit_fused and fuse:
            empty_payload = {
                "input": str(input_path),
                "input_type": "pdf" if is_pdf else "png",
                "page": int(page) if is_pdf else None,
                "input_dpi": float(base_dpi) if base_dpi else None,
                "ocr_dpi": int(ocr_dpi_base),
                "detection_dpi": int(detection_dpi),
                "input_size": input_size,
                "detection_size": detection_size,
                "variant": {
                    "method": "fused_majority",
                    "source_variants": 0,
                    "threshold": None,
                    "mode": "plurality",
                },
                "tables": [],
            }
            fused_json = out_dir / "variant_fused_majority.json"
            fused_json.write_text(json.dumps(empty_payload, indent=2), encoding="utf-8")
            fused_txt = out_dir / "variant_fused_majority.txt"
            fused_txt.write_text("", encoding="utf-8")
        if return_fused:
            return {
                "tables": [],
                "input_dpi": float(base_dpi) if base_dpi else None,
                "ocr_dpi": int(ocr_dpi_base),
                "detection_dpi": int(detection_dpi),
                "input_size": input_size,
                "detection_size": detection_size,
            }
        return 0

    variants = _build_variants(ocr_dpi_base, detection_dpi)
    use_scale_naming = bool(is_png and base_dpi is None)

    def _resolve_variant_image(variant: Dict) -> Tuple[object, int, Optional[float]]:
        dpi = variant.get("dpi")
        scale = variant.get("scale")
        ocr_dpi = None
        ocr_scale = None
        if is_pdf:
            if dpi is None and scale is not None:
                ocr_dpi = int(round(detection_dpi * float(scale)))
            else:
                ocr_dpi = int(dpi)
            return _get_pdf_image(ocr_dpi), ocr_dpi, None

        # PNG
        if base_dpi:
            if dpi is not None:
                ocr_dpi = int(dpi)
                ocr_scale = float(ocr_dpi) / float(base_dpi)
            elif scale is not None:
                ocr_dpi = int(round(detection_dpi * float(scale)))
                ocr_scale = float(ocr_dpi) / float(base_dpi)
            else:
                ocr_dpi = int(round(base_dpi))
                ocr_scale = 1.0
        else:
            if scale is not None:
                ocr_scale = float(scale)
                ocr_dpi = int(round(detection_dpi * ocr_scale))
            elif dpi is not None:
                ocr_scale = float(dpi) / float(detection_dpi)
                ocr_dpi = int(dpi)
            else:
                ocr_scale = 1.0
                ocr_dpi = int(detection_dpi)
        return _get_png_image(ocr_scale), ocr_dpi, ocr_scale

    # Settings from pipeline envs
    numeric_whitelist = os.environ.get("EIDAT_NUMERIC_RESCUE_WHITELIST", "0123456789.-")
    numeric_rescue_pad = _parse_int_env("EIDAT_NUMERIC_RESCUE_PAD", 4)
    numeric_strict_pad = _parse_int_env("EIDAT_NUMERIC_STRICT_PAD", 4)
    cell_interior_shrink = _parse_int_env("EIDAT_CELL_INTERIOR_SHRINK", 2)

    results: List[Dict] = []
    variant_rows_all: List[List[List[List[str]]]] = []

    for variant in variants:
        method = variant.get("method")
        psm = variant.get("psm")
        remove_lines = bool(variant.get("remove_lines", False))
        line_strip_level = variant.get("line_strip_level")

        ocr_img, ocr_dpi, ocr_scale = _resolve_variant_image(variant)
        ocr_h, ocr_w = int(ocr_img.shape[0]), int(ocr_img.shape[1])

        line_strip_tag = _format_line_strip_tag(remove_lines, line_strip_level)
        if use_scale_naming and ocr_scale is not None:
            scale_tag = _format_scale(ocr_scale)
            tag = f"variant_{method}_scale{scale_tag}_psm{psm}_{line_strip_tag}"
        else:
            tag = f"variant_{method}_dpi{ocr_dpi}_psm{psm}_{line_strip_tag}"

        line_strip_debug_dir = out_dir / "line_strip_debug" if (remove_lines and emit_variants) else None
        tables_out: List[Dict] = []

        if method == "table_region_ocr":
            for table_idx, table in enumerate(base_tables):
                bbox = table.get("bbox_px") or []
                if len(bbox) != 4:
                    continue
                bbox_ocr = _scale_bbox_to_ocr(bbox, detection_dpi, ocr_dpi, pad_px=8)
                bbox_ocr = _clip_bbox(bbox_ocr, ocr_w, ocr_h)
                if bbox_ocr is None:
                    continue
                tokens, _meta = ocr_engine.ocr_region_tokens(
                    ocr_img,
                    bbox_ocr,
                    lang=lang,
                    psms=(int(psm),),
                    remove_lines=remove_lines,
                    line_strip_level=str(line_strip_level) if remove_lines else None,
                    debug_dir=line_strip_debug_dir,
                    debug_tag=f"{tag}_table{table_idx + 1}",
                    debug_emit={"crop": False, "nolines": True, "linemask": False},
                )
                scaled_tokens = token_projector.scale_tokens_to_dpi(tokens, ocr_dpi, detection_dpi)
                cells = _clone_cells(table["cells"])
                token_projector.project_tokens_to_cells_force(
                    scaled_tokens,
                    cells,
                    verbose=False,
                    debug_info=None,
                    ocr_dpi=ocr_dpi,
                    detection_dpi=detection_dpi,
                    reset_cells=True,
                    center_margin_px=18.0,
                    only_if_empty=False,
                    char_gap_ratio=0.35,
                    max_token_ratio=1.6,
                    max_token_area_ratio=2.5,
                )
                tables_out.append({
                    "bbox_px": table["bbox_px"],
                    "cells": cells,
                    "borderless": bool(table.get("borderless", False)),
                })

        elif method == "page_ocr_psm3":
            bbox_full = (0, 0, ocr_w, ocr_h)
            tokens, _meta = ocr_engine.ocr_region_tokens(
                ocr_img,
                bbox_full,
                lang=lang,
                psms=(int(psm),),
                remove_lines=False,
            )
            scaled_tokens = token_projector.scale_tokens_to_dpi(tokens, ocr_dpi, detection_dpi)
            for table in base_tables:
                cells = _clone_cells(table["cells"])
                token_projector.project_tokens_to_cells_force(
                    scaled_tokens,
                    cells,
                    verbose=False,
                    debug_info=None,
                    ocr_dpi=ocr_dpi,
                    detection_dpi=detection_dpi,
                    reset_cells=True,
                    center_margin_px=18.0,
                    only_if_empty=False,
                    char_gap_ratio=0.35,
                    max_token_ratio=1.6,
                    max_token_area_ratio=2.5,
                )
                tables_out.append({
                    "bbox_px": table["bbox_px"],
                    "cells": cells,
                    "borderless": bool(table.get("borderless", False)),
                })

        elif method == "numeric_rescue":
            numeric_config = ["-c", f"tessedit_char_whitelist={numeric_whitelist}"]
            if ocr_dpi:
                numeric_config += ["-c", f"user_defined_dpi={int(ocr_dpi)}"]
            pad_px = int(round(numeric_rescue_pad * (float(ocr_dpi) / float(ocr_dpi_base))))
            for table in base_tables:
                cells = _clone_cells(table["cells"])
                for cell in cells:
                    bbox = cell.get("bbox_px") or []
                    if len(bbox) != 4:
                        continue
                    bbox_ocr = _scale_bbox_to_ocr(bbox, detection_dpi, ocr_dpi, pad_px=0)
                    bbox_ocr = _clip_bbox(bbox_ocr, ocr_w, ocr_h)
                    if bbox_ocr is None:
                        continue
                    text = ocr_engine.ocr_cell_region(
                        ocr_img,
                        bbox_ocr,
                        lang=lang,
                        psm=int(psm),
                        padding=pad_px,
                        remove_borders=False,
                        tesseract_config=numeric_config,
                    ).strip()
                    cell["text"] = text
                tables_out.append({
                    "bbox_px": table["bbox_px"],
                    "cells": cells,
                    "borderless": bool(table.get("borderless", False)),
                })

        elif method == "numeric_strict_ocr":
            numeric_config = ["-c", f"tessedit_char_whitelist={numeric_whitelist}"]
            if ocr_dpi:
                numeric_config += ["-c", f"user_defined_dpi={int(ocr_dpi)}"]
            pad_px = int(round(numeric_strict_pad * (float(ocr_dpi) / float(ocr_dpi_base))))
            for table in base_tables:
                cells = _clone_cells(table["cells"])
                for cell in cells:
                    bbox = cell.get("bbox_px") or []
                    if len(bbox) != 4:
                        continue
                    bbox_ocr = _scale_bbox_to_ocr(bbox, detection_dpi, ocr_dpi, pad_px=pad_px)
                    bbox_ocr = _clip_bbox(bbox_ocr, ocr_w, ocr_h)
                    if bbox_ocr is None:
                        continue
                    tokens, _meta = ocr_engine.ocr_region_tokens(
                        ocr_img,
                        bbox_ocr,
                        lang=lang,
                        psms=(int(psm),),
                        remove_lines=remove_lines,
                        tesseract_config=numeric_config,
                    )
                    text = _tokens_to_text(tokens).strip()
                    cell["text"] = text
                tables_out.append({
                    "bbox_px": table["bbox_px"],
                    "cells": cells,
                    "borderless": bool(table.get("borderless", False)),
                })

        elif method == "cell_interior_ocr":
            for table in base_tables:
                cells = _clone_cells(table["cells"])
                for cell in cells:
                    bbox = cell.get("bbox_px") or []
                    if len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = bbox
                    x0 += cell_interior_shrink
                    y0 += cell_interior_shrink
                    x1 -= cell_interior_shrink
                    y1 -= cell_interior_shrink
                    if x1 <= x0 or y1 <= y0:
                        continue
                    bbox_ocr = _scale_bbox_to_ocr([x0, y0, x1, y1], detection_dpi, ocr_dpi, pad_px=0)
                    bbox_ocr = _clip_bbox(bbox_ocr, ocr_w, ocr_h)
                    if bbox_ocr is None:
                        continue
                    text = ocr_engine.ocr_cell_region(
                        ocr_img,
                        bbox_ocr,
                        lang=lang,
                        psm=int(psm),
                        padding=0,
                        remove_borders=False,
                    ).strip()
                    cell["text"] = text
                tables_out.append({
                    "bbox_px": table["bbox_px"],
                    "cells": cells,
                    "borderless": bool(table.get("borderless", False)),
                })

        else:
            continue

        # Build JSON-ready tables
        json_tables = []
        for table in tables_out:
            rows = _cells_to_rows(table.get("cells", []))
            json_tables.append({
                "bbox_px": table.get("bbox_px") or [],
                "borderless": bool(table.get("borderless", False)),
                "rows": rows,
            })

        # Variant metadata for JSON + file naming
        variant_meta: Dict[str, object] = {
            "method": method,
            "psm": int(psm) if psm is not None else None,
            "remove_lines": bool(remove_lines),
            "line_strip_level": str(line_strip_level) if remove_lines else None,
        }
        if use_scale_naming and ocr_scale is not None:
            variant_meta["scale"] = float(ocr_scale)
        else:
            variant_meta["dpi"] = int(ocr_dpi)

        payload = {
            "input": str(input_path),
            "input_type": "pdf" if is_pdf else "png",
            "page": int(page) if is_pdf else None,
            "input_dpi": float(base_dpi) if base_dpi else None,
            "ocr_dpi": int(ocr_dpi),
            "detection_dpi": int(detection_dpi),
            "input_size": input_size,
            "detection_size": detection_size,
            "variant": variant_meta,
            "tables": json_tables,
        }

        if emit_variants:
            json_path = out_dir / f"{tag}.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            # Write table-only ASCII text
            lines: List[str] = []
            for idx, table in enumerate(tables_out):
                lines.append("[Table]")
                lines.append("")
                lines.append(f"[Table {idx + 1}]")
                table_ascii = debug_exporter._render_table_ascii(table)
                if table_ascii:
                    lines.append(table_ascii)
                lines.append("")
            txt_path = out_dir / f"{tag}.txt"
            txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

            results.append({
                "tag": tag,
                "json": str(json_path),
                "txt": str(txt_path),
            })

        variant_rows_all.append(json_tables)

    if fuse and variant_rows_all:
        variant_count = len(variant_rows_all)
        fused_tables: List[Dict] = []

        for table_idx, base_table in enumerate(base_tables):
            base_cells = base_table.get("cells") or []
            max_row = -1
            max_col = -1
            for cell in base_cells:
                try:
                    r = int(cell.get("row"))
                    c = int(cell.get("col"))
                except (TypeError, ValueError):
                    continue
                max_row = max(max_row, r)
                max_col = max(max_col, c)
            if max_row < 0 or max_col < 0:
                fused_rows = []
            else:
                fused_rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

            for r in range(len(fused_rows)):
                for c in range(len(fused_rows[r])):
                    counts: Dict[str, int] = {}
                    for variant_tables in variant_rows_all:
                        if table_idx >= len(variant_tables):
                            candidate = ""
                        else:
                            rows = variant_tables[table_idx].get("rows", [])
                            if r < len(rows) and c < len(rows[r]):
                                candidate = str(rows[r][c])
                            else:
                                candidate = ""
                        counts[candidate] = counts.get(candidate, 0) + 1
                    non_empty_counts = {v: c for v, c in counts.items() if str(v).strip()}
                    if non_empty_counts:
                        best_non_empty = ""
                        best_non_empty_count = -1
                        for val, count in non_empty_counts.items():
                            if count > best_non_empty_count:
                                best_non_empty = val
                                best_non_empty_count = count
                        fused_rows[r][c] = best_non_empty
                    else:
                        fused_rows[r][c] = ""

            fused_cells = []
            for cell in base_cells:
                try:
                    r = int(cell.get("row"))
                    c = int(cell.get("col"))
                except (TypeError, ValueError):
                    continue
                text_val = ""
                if 0 <= r < len(fused_rows) and 0 <= c < len(fused_rows[r]):
                    text_val = fused_rows[r][c]
                fused_cells.append({
                    "bbox_px": list(cell.get("bbox_px") or []),
                    "row": r,
                    "col": c,
                    "text": text_val,
                    "ocr_method": "fused_majority",
                })
            fused_tables.append({
                "bbox_px": base_table.get("bbox_px") or [],
                "cells": fused_cells,
                "rows": fused_rows,
                "borderless": bool(base_table.get("borderless", False)),
            })

        fused_payload = {
            "input": str(input_path),
            "input_type": "pdf" if is_pdf else "png",
            "page": int(page) if is_pdf else None,
            "input_dpi": float(base_dpi) if base_dpi else None,
            "ocr_dpi": int(ocr_dpi_base),
            "detection_dpi": int(detection_dpi),
            "input_size": input_size,
            "detection_size": detection_size,
            "variant": {
                "method": "fused_majority",
                "source_variants": variant_count,
                "threshold": None,
                "mode": "plurality",
            },
            "tables": [
                {"bbox_px": t["bbox_px"], "borderless": bool(t.get("borderless", False)), "rows": t["rows"]}
                for t in fused_tables
            ],
        }
        if emit_fused:
            fused_json = out_dir / "variant_fused_majority.json"
            fused_json.write_text(json.dumps(fused_payload, indent=2), encoding="utf-8")

            lines: List[str] = []
            for idx, table in enumerate(fused_tables):
                lines.append("[Table]")
                lines.append("")
                lines.append(f"[Table {idx + 1}]")
                table_ascii = debug_exporter._render_table_ascii(table)
                if table_ascii:
                    lines.append(table_ascii)
                lines.append("")
            fused_txt = out_dir / "variant_fused_majority.txt"
            fused_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    if return_fused:
        return {
            "tables": fused_tables if fuse else [],
            "input_dpi": float(base_dpi) if base_dpi else None,
            "ocr_dpi": int(ocr_dpi_base),
            "detection_dpi": int(detection_dpi),
            "input_size": input_size,
            "detection_size": detection_size,
        }

    if emit_variants:
        print(f"Wrote {len(results)} variants to {out_dir}")
    elif emit_fused and fuse:
        print(f"Wrote fused tables to {out_dir}")
    return len(results)


def main() -> int:
    debug_dir = Path(__file__).resolve().parent
    repo_root = debug_dir.parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to .png, .pdf, or a directory")
    parser.add_argument("--out-dir", default=str(debug_dir / "results"))
    parser.add_argument("--page", type=int, default=1, help="PDF page (1-based)")
    parser.add_argument("--ocr-dpi", type=int, default=450)
    parser.add_argument("--detection-dpi", type=int, default=900)
    parser.add_argument("--lang", default=None)
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Emit fused table output using per-cell majority voting",
    )
    parser.add_argument(
        "--fused-only",
        action="store_true",
        help="Only emit fused output (skip per-variant files)",
    )
    parser.add_argument(
        "--allow-no-tables",
        action="store_true",
        help="Do not error if no tables are detected",
    )
    parser.add_argument(
        "--enable-borderless",
        action="store_true",
        help="Enable borderless table detection (disabled by default)",
    )
    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        help="Remove all files in out-dir before run",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Keep existing outputs in out-dir",
    )
    parser.set_defaults(clean=True)
    args = parser.parse_args()

    _ensure_app_path(repo_root)

    detection_dpi = int(args.detection_dpi)
    ocr_dpi_base = int(args.ocr_dpi)
    if detection_dpi <= 0 or ocr_dpi_base <= 0:
        raise RuntimeError("detection-dpi and ocr-dpi must be positive")

    input_paths = _gather_input_files(debug_dir, args.input)
    out_dir_base = Path(args.out_dir)
    variants = _build_variants(ocr_dpi_base, detection_dpi)

    emit_variants = not bool(args.fused_only)
    fuse_enabled = bool(args.fuse or args.fused_only)

    for input_path in input_paths:
        per_input_out = out_dir_base / input_path.stem
        _run_for_input(
            input_path,
            out_dir=per_input_out,
            page=int(args.page),
            ocr_dpi_base=ocr_dpi_base,
            detection_dpi=detection_dpi,
            lang=args.lang,
            clean=bool(args.clean),
            fuse=fuse_enabled,
            emit_variants=emit_variants,
            emit_fused=bool(fuse_enabled),
            allow_no_tables=bool(args.allow_no_tables),
            enable_borderless=bool(args.enable_borderless),
        )

    summary_path = debug_dir / "run_table_variants_summary.txt"
    try:
        _write_run_summary(
            summary_path,
            input_paths=input_paths,
            out_dir_base=out_dir_base,
            ocr_dpi_base=ocr_dpi_base,
            detection_dpi=detection_dpi,
            variants=variants,
            fuse=bool(fuse_enabled),
            borderless_enabled=bool(args.enable_borderless),
        )
    except Exception:
        pass

    if len(input_paths) > 1:
        print(f"Processed {len(input_paths)} inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
