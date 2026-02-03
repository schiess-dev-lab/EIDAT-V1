#!/usr/bin/env python3
"""
Master debug runner: render PDF pages to PNGs, apply table grid overlays,
then run table variants on the bordered outputs.

Defaults to fused output (--fuse) and uses debug_method/debug_master_config.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from . import debug_page_dpi, table_grid_debug
except Exception:  # pragma: no cover - fallback for script execution
    import debug_page_dpi  # type: ignore
    import table_grid_debug  # type: ignore


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("Config must be a JSON object.")
    return data


def _resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _clean_rendered_pages(out_dir: Path) -> None:
    for path in out_dir.glob("page_*.png"):
        try:
            path.unlink()
        except Exception:
            pass


def _sorted_pages(out_dir: Path) -> List[Path]:
    pages = []
    for path in out_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        stem = path.stem
        if not stem.startswith("page_"):
            continue
        try:
            page_num = int(stem.split("_", 1)[1])
        except Exception:
            continue
        pages.append((page_num, path))
    return [p for _, p in sorted(pages, key=lambda item: (item[0], item[1].name))]


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


def _save_png_with_dpi(img_gray, path: Path, dpi: Optional[float]) -> None:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow (PIL) is required to save PNGs with DPI metadata.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img_gray)
    if dpi:
        img.save(str(path), dpi=(float(dpi), float(dpi)))
    else:
        img.save(str(path))


def _draw_table_borders(
    pre_path: Path,
    tables: List[Dict[str, Any]],
    out_path: Path,
    *,
    line_thickness: int,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) is required to draw table borders.") from exc

    img_gray = cv2.imread(str(pre_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Failed to load image: {pre_path}")
    h, w = img_gray.shape[:2]
    out = img_gray.copy()

    thickness = max(2, int(line_thickness))
    for table in tables:
        cols = [float(x) for x in (table.get("columns") or [])]
        rows = [float(y) for y in (table.get("row_lines") or [])]
        if len(cols) < 2 or len(rows) < 2:
            continue
        left = int(round(min(cols)))
        right = int(round(max(cols)))
        top = int(round(min(rows)))
        bottom = int(round(max(rows)))
        left = max(0, min(w - 1, left))
        right = max(0, min(w - 1, right))
        top = max(0, min(h - 1, top))
        bottom = max(0, min(h - 1, bottom))
        if right <= left or bottom <= top:
            continue
        for x in cols:
            xi = int(round(float(x)))
            if xi < left or xi > right:
                continue
            cv2.line(out, (xi, top), (xi, bottom), 0, thickness)
        for y in rows:
            yi = int(round(float(y)))
            if yi < top or yi > bottom:
                continue
            cv2.line(out, (left, yi), (right, yi), 0, thickness)

    dpi = _read_png_dpi(pre_path)
    _save_png_with_dpi(out, out_path, dpi)


def _strip_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def _run_table_grid_for_page(
    page_path: Path,
    *,
    out_root: Path,
    merge_kx: int,
    min_gap: Optional[float],
    min_gap_ratio: float,
    offset_px: float,
    line_thickness: int,
    line_pad_factor: float,
    min_token_h_px: float,
    min_token_h_ratio: float,
    draw_tables: bool,
    draw_hlines: bool,
    draw_seps_in_tables: bool,
    draw_separators: bool,
) -> Dict[str, Any]:
    program_dir = page_path.parent.name
    page_stem = _strip_suffix(page_path.stem, "_pre")
    out_dir = out_root / program_dir / page_stem
    return table_grid_debug.run_for_image(
        page_path,
        out_dir=out_dir,
        merge_kx=merge_kx,
        min_gap=min_gap,
        min_gap_ratio=min_gap_ratio,
        offset_px=offset_px,
        line_thickness=line_thickness,
        line_pad_factor=line_pad_factor,
        min_token_h_px=min_token_h_px,
        min_token_h_ratio=min_token_h_ratio,
        draw_tables=draw_tables,
        draw_hlines=draw_hlines,
        draw_seps_in_tables=draw_seps_in_tables,
        draw_separators=draw_separators,
    )


def _run_table_variants(
    script_path: Path,
    page_path: Path,
    *,
    out_dir: Path,
    ocr_dpi: int,
    detection_dpi: int,
    lang: Optional[str],
    clean: bool,
    fuse: bool,
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(page_path),
        "--out-dir",
        str(out_dir),
        "--ocr-dpi",
        str(ocr_dpi),
        "--detection-dpi",
        str(detection_dpi),
    ]
    if lang:
        cmd.extend(["--lang", str(lang)])
    cmd.append("--clean" if clean else "--no-clean")
    if fuse:
        cmd.append("--fuse")
    subprocess.run(cmd, check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_config = root / "debug_method" / "debug_master_config.json"

    parser = argparse.ArgumentParser(description="Run debug page render + table variants.")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--pdf", type=str, default="", help="Override PDF path.")
    parser.add_argument("--dpi", type=int, default=0, help="Override render DPI.")
    parser.add_argument("--fuse", dest="fuse", action="store_true", help="Enable fused output.")
    parser.add_argument("--no-fuse", dest="fuse", action="store_false", help="Disable fused output.")
    parser.set_defaults(fuse=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = _load_config(config_path)

    input_dir = _resolve_path(
        root, _read_config_value(config, "render", "input_dir", default="debug_method/DebugFileLocation")
    )
    if input_dir is None:
        raise RuntimeError("render.input_dir is required in config.")
    input_dir.mkdir(parents=True, exist_ok=True)

    render_root = _resolve_path(
        root, _read_config_value(config, "render", "out_root", default="debug_method/debug_file")
    )
    if render_root is None:
        raise RuntimeError("render.out_root is required in config.")
    render_root.mkdir(parents=True, exist_ok=True)

    pdf_override = _resolve_path(root, args.pdf) if args.pdf else None
    pdf_config = _resolve_path(root, _read_config_value(config, "pdf", default=""))
    dpi_override = int(args.dpi) if args.dpi and args.dpi > 0 else None
    render_dpi = dpi_override or int(_read_config_value(config, "render", "dpi", default=450))
    clean_render = bool(_read_config_value(config, "render", "clean", default=True))

    if pdf_override or pdf_config:
        pdf_path = pdf_override or pdf_config
        if pdf_path is None or not pdf_path.exists():
            raise RuntimeError(f"PDF not found: {pdf_path}")
        per_out = render_root / pdf_path.stem
        per_out.mkdir(parents=True, exist_ok=True)
        if clean_render:
            _clean_rendered_pages(per_out)
        page_count = debug_page_dpi._render_pdf_to_png(pdf_path, per_out, render_dpi)
        debug_page_dpi._verify_outputs(per_out, page_count)
        render_sets = {pdf_path.stem: (per_out, page_count)}
    else:
        pdfs = debug_page_dpi._find_pdfs(input_dir)
        render_sets = debug_page_dpi.render_pdf_set(
            pdfs,
            out_root=render_root,
            dpi=render_dpi,
            clean=clean_render,
        )

    table_grid_cfg = _read_config_value(config, "table_grid", default={}) or {}
    table_grid_enabled = bool(table_grid_cfg.get("enabled", True))
    table_grid_out_root = _resolve_path(
        root, table_grid_cfg.get("out_root", "debug_method/word_gap_debug")
    )
    if table_grid_out_root is None:
        raise RuntimeError("table_grid.out_root is required in config.")
    table_grid_out_root.mkdir(parents=True, exist_ok=True)

    tg_merge_kx = int(table_grid_cfg.get("merge_kx", 0))
    tg_min_gap = table_grid_cfg.get("min_gap", 50.0)
    tg_min_gap_ratio = float(table_grid_cfg.get("min_gap_ratio", 0.0))
    tg_gap_threshold = float(table_grid_cfg.get("gap_threshold", 0.0))
    tg_offset_px = float(table_grid_cfg.get("left_offset", 24.0))
    tg_line_thickness = int(table_grid_cfg.get("line_thickness", 3))
    tg_line_pad = float(table_grid_cfg.get("line_pad", 0.25))
    tg_min_token_h = float(table_grid_cfg.get("min_token_h", 0.0))
    tg_min_token_h_ratio = float(table_grid_cfg.get("min_token_h_ratio", 0.85))
    tg_draw_hlines = bool(table_grid_cfg.get("draw_hlines", True))
    tg_draw_seps_in_tables = bool(table_grid_cfg.get("draw_seps_in_tables", False))
    tg_draw_separators = bool(table_grid_cfg.get("draw_separators", False))
    tg_border_thickness = int(table_grid_cfg.get("border_thickness", 0))
    if tg_border_thickness <= 0:
        tg_border_thickness = max(4, tg_line_thickness + 2)

    variants_script = root / "debug_method" / "run_table_variants.py"
    if not variants_script.exists():
        raise RuntimeError(f"Missing script: {variants_script}")

    variants_out_root = _resolve_path(
        root, _read_config_value(config, "table_variants", "out_root", default="debug_method/results")
    )
    if variants_out_root is None:
        raise RuntimeError("table_variants.out_root is required in config.")
    variants_out_root.mkdir(parents=True, exist_ok=True)

    ocr_dpi = int(_read_config_value(config, "table_variants", "ocr_dpi", default=450))
    detection_dpi = int(_read_config_value(config, "table_variants", "detection_dpi", default=900))
    lang = _read_config_value(config, "table_variants", "lang", default=None)
    clean = bool(_read_config_value(config, "table_variants", "clean", default=True))

    cfg_fuse = _read_config_value(config, "table_variants", "fuse", default=True)
    fuse = bool(cfg_fuse if args.fuse is None else args.fuse)

    total_pages = 0
    for stem, (render_dir, _page_count) in render_sets.items():
        pages = _sorted_pages(render_dir)
        if not pages:
            raise RuntimeError(f"No rendered PNGs found in {render_dir}")

        if table_grid_enabled:
            for page_path in pages:
                pre_path = page_path.with_name(f"{page_path.stem}_pre.png")
                post_path = page_path.with_name(f"{page_path.stem}_post.png")
                shutil.copy2(page_path, pre_path)
                gap_override = None
                if tg_gap_threshold and tg_gap_threshold > 0:
                    gap_override = float(tg_gap_threshold)
                elif tg_min_gap and float(tg_min_gap) > 0:
                    gap_override = float(tg_min_gap)

                summary = _run_table_grid_for_page(
                    pre_path,
                    out_root=table_grid_out_root,
                    merge_kx=tg_merge_kx,
                    min_gap=gap_override if gap_override else None,
                    min_gap_ratio=tg_min_gap_ratio,
                    offset_px=tg_offset_px,
                    line_thickness=tg_line_thickness,
                    line_pad_factor=tg_line_pad,
                    min_token_h_px=tg_min_token_h,
                    min_token_h_ratio=tg_min_token_h_ratio,
                    draw_tables=True,
                    draw_hlines=tg_draw_hlines,
                    draw_seps_in_tables=tg_draw_seps_in_tables,
                    draw_separators=tg_draw_separators,
                )
                tables = summary.get("tables") or []
                _draw_table_borders(
                    pre_path,
                    tables,
                    post_path,
                    line_thickness=tg_border_thickness,
                )
                shutil.copy2(post_path, page_path)

        per_out_root = variants_out_root / stem
        per_out_root.mkdir(parents=True, exist_ok=True)
        for page_path in pages:
            _run_table_variants(
                variants_script,
                page_path,
                out_dir=per_out_root,
                ocr_dpi=ocr_dpi,
                detection_dpi=detection_dpi,
                lang=lang,
                clean=clean,
                fuse=fuse,
            )
            total_pages += 1

    print(f"Processed {total_pages} rendered pages with table variants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
