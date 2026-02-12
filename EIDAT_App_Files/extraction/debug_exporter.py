"""
Debug Exporter - Export extraction results to JSON format

Creates compatible debug output matching existing format.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from . import page_analyzer

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def export_page_debug(
    pdf_path: Path,
    page_num: int,
    tokens: List[Dict],
    tables: List[Dict],
    img_w: int,
    img_h: int,
    dpi: int,
    output_dir: Path,
    charts: Optional[List[Dict]] = None,
    flow_data: Optional[Dict] = None,
    *,
    ocr_dpi: Optional[int] = None,
    ocr_img_w: Optional[int] = None,
    ocr_img_h: Optional[int] = None,
    ocr_pass_stats: Optional[Dict] = None
) -> Path:
    """
    Export page extraction results to JSON debug file.

    Args:
        pdf_path: Source PDF path
        page_num: Page number (0-indexed)
        tokens: OCR tokens
        tables: Detected tables
        img_w, img_h: Image dimensions
        dpi: Render DPI
        output_dir: Debug output directory
        flow_data: Optional flow analysis data

    Returns:
        Path to generated JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build artifacts structure
    artifacts = {
        'tokens': tokens,
        'tables': _format_tables(tables),
        'table_count': len(tables),
        'ocr_method_summary': _build_ocr_method_summary(tables)
    }
    if ocr_pass_stats is not None:
        artifacts['ocr_pass_stats'] = ocr_pass_stats

    if charts:
        artifacts['charts'] = _format_charts(charts)
        artifacts['chart_count'] = len(charts)

    # Add flow data if available
    if flow_data:
        artifacts['flow'] = {
            'headers': flow_data.get('headers', []),
            'footers': flow_data.get('footers', []),
            'lines': len(flow_data.get('lines', [])),
            'paragraphs': len(flow_data.get('paragraphs', [])),
            'table_titles': len(flow_data.get('table_titles', []))
        }

    # Build page JSON
    page_json = {
        'page': page_num + 1,
        'pdf_file': str(pdf_path),
        'img_w': img_w,
        'img_h': img_h,
        'dpi': dpi,
        'ocr_dpi': ocr_dpi,
        'ocr_img_w': ocr_img_w,
        'ocr_img_h': ocr_img_h,
        'timestamp': datetime.now().isoformat(),
        'artifacts': artifacts,
        'meta': {
            'extractor_version': '2.1.0',
            'extraction_method': 'cell_detection + token_projection (+ table_region_ocr)'
        }
    }

    # Write to file
    output_file = output_dir / f"page_{page_num + 1}_page.json"
    with open(output_file, 'w') as f:
        json.dump(page_json, f, indent=2)

    return output_file


def _format_tables(tables: List[Dict]) -> List[Dict]:
    """Format tables for JSON export."""
    formatted = []

    for table in tables:
        bbox = table.get('bbox_px', [])
        cells = table.get('cells', [])

        # Organize cells into row bands for compatibility
        row_bands = _extract_row_bands(cells)
        col_bounds = _extract_col_bounds(cells)

        formatted_table = {
            'bbox_px': bbox,
            'num_cells': len(cells),
            'row_bands_px': row_bands,
            'col_bounds_px': col_bounds,
            'cells': _format_cells(cells)
        }

        formatted.append(formatted_table)

    return formatted


def _format_charts(charts: List[Dict]) -> List[Dict]:
    formatted = []
    for chart in charts:
        bbox = chart.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        formatted.append({
            "bbox_px": bbox,
            "method": chart.get("method", "unknown"),
            "axis_tokens": chart.get("axis_tokens"),
            "title": chart.get("title")
        })
    return formatted


def _merge_nearby_positions(positions: List[float], threshold: float = 20.0) -> List[float]:
    """
    Merge nearby positions to eliminate border artifacts.

    At 900 DPI, table borders are typically 5-15px wide. Positions within
    the threshold are merged to their average value.

    Args:
        positions: Sorted list of positions
        threshold: Max distance to merge (default 20px for 900 DPI)

    Returns:
        Merged positions list
    """
    if not positions:
        return []

    merged = []
    group = [positions[0]]

    for pos in positions[1:]:
        if pos - group[-1] <= threshold:
            # Close enough - add to current group
            group.append(pos)
        else:
            # Gap too large - finalize current group and start new one
            merged.append(sum(group) / len(group))
            group = [pos]

    # Don't forget the last group
    if group:
        merged.append(sum(group) / len(group))

    return merged


def _extract_row_bands(cells: List[Dict], merge_threshold: float = 20.0) -> List[List[float]]:
    """
    Extract unique row y-positions as bands, merging border artifacts.

    Args:
        cells: List of cells with bbox_px
        merge_threshold: Max distance to merge nearby positions (default 20px)

    Returns:
        List of [y_start, y_end] bands representing actual rows
    """
    if not cells:
        return []

    # Get unique y-positions
    y_positions = set()
    for cell in cells:
        bbox = cell.get('bbox_px', [])
        if len(bbox) >= 4:
            y_positions.add(float(bbox[1]))  # y0
            y_positions.add(float(bbox[3]))  # y1

    y_sorted = sorted(y_positions)

    # Merge nearby positions to eliminate border artifacts
    y_merged = _merge_nearby_positions(y_sorted, merge_threshold)

    # Create bands from merged positions
    bands = [[y_merged[i], y_merged[i+1]] for i in range(len(y_merged) - 1)]

    return bands


def _extract_col_bounds(cells: List[Dict], merge_threshold: float = 20.0) -> List[float]:
    """
    Extract unique column x-positions, merging border artifacts.

    Args:
        cells: List of cells with bbox_px
        merge_threshold: Max distance to merge nearby positions (default 20px)

    Returns:
        List of x-positions representing column boundaries
    """
    if not cells:
        return []

    x_positions = set()
    for cell in cells:
        bbox = cell.get('bbox_px', [])
        if len(bbox) >= 4:
            x_positions.add(float(bbox[0]))  # x0
            x_positions.add(float(bbox[2]))  # x1

    x_sorted = sorted(x_positions)

    # Merge nearby positions to eliminate border artifacts
    return _merge_nearby_positions(x_sorted, merge_threshold)


def _format_cells(cells: List[Dict]) -> List[Dict]:
    """Format cells for JSON export."""
    formatted = []

    for cell in cells:
        formatted_cell = {
            'bbox_px': cell.get('bbox_px', []),
            'text': cell.get('text', ''),
            'tokens': cell.get('tokens', []),
            'ocr_method': cell.get('ocr_method'),
            'projection_text': cell.get('projection_text'),
            'row': cell.get('row'),
            'col': cell.get('col'),
            'token_count': cell.get('token_count')
        }

        formatted.append(formatted_cell)

    return formatted


def _safe_token_count(cell: Dict) -> int:
    try:
        token_count = cell.get("token_count")
        if token_count is not None:
            return int(token_count)
    except (TypeError, ValueError):
        pass

    tokens = cell.get("tokens", [])
    if tokens:
        return int(len(tokens))

    text = str(cell.get("text", "")).strip()
    return int(len(text.split())) if text else 0


def _build_ocr_method_summary(tables: List[Dict]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_cells": 0,
        "total_tokens": 0,
        "by_method": {}
    }

    for table in tables or []:
        for cell in table.get("cells", []) or []:
            method = cell.get("ocr_method") or "unknown"
            tokens = _safe_token_count(cell)
            summary["total_cells"] = int(summary.get("total_cells", 0)) + 1
            summary["total_tokens"] = int(summary.get("total_tokens", 0)) + tokens
            bucket = summary["by_method"].setdefault(method, {"cells": 0, "tokens": 0})
            bucket["cells"] = int(bucket.get("cells", 0)) + 1
            bucket["tokens"] = int(bucket.get("tokens", 0)) + tokens

    return summary


def _merge_method_summaries(summaries: List[Dict]) -> Dict[str, object]:
    merged: Dict[str, object] = {"total_cells": 0, "total_tokens": 0, "by_method": {}}
    for summary in summaries or []:
        merged["total_cells"] = int(merged.get("total_cells", 0)) + int(summary.get("total_cells", 0) or 0)
        merged["total_tokens"] = int(merged.get("total_tokens", 0)) + int(summary.get("total_tokens", 0) or 0)
        for method, stats in (summary.get("by_method") or {}).items():
            bucket = merged["by_method"].setdefault(method, {"cells": 0, "tokens": 0})
            bucket["cells"] = int(bucket.get("cells", 0)) + int(stats.get("cells", 0) or 0)
            bucket["tokens"] = int(bucket.get("tokens", 0)) + int(stats.get("tokens", 0) or 0)
    return merged


def _merge_pass_stats(stats_list: List[Optional[Dict]]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for stats in stats_list or []:
        if not stats:
            continue
        for name, values in stats.items():
            bucket = merged.setdefault(name, {})
            for key, val in (values or {}).items():
                bucket[key] = int(bucket.get(key, 0)) + int(val or 0)
    return merged


def export_borderless_table_debug_images(
    img_gray: "np.ndarray",
    tables: List[Dict],
    output_dir: Path,
    page_num: int
) -> None:
    """
    Export debug PNGs for borderless tables.

    Creates:
      - page_{n}_borderless_table_{i}_crop.png
      - page_{n}_borderless_table_{i}_crop_nolines.png
      - page_{n}_borderless_table_{i}_crop_linemask.png
      - page_{n}_borderless_table_{i}_borders.png
    """
    if not HAVE_CV2 or img_gray is None or not tables:
        return

    h, w = img_gray.shape[:2]

    for idx, table in enumerate(tables):
        bbox = table.get("bbox_px") or []
        if len(bbox) != 4:
            continue

        x0, y0, x1, y1 = (int(round(v)) for v in bbox)
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue

        crop = img_gray[y0:y1, x0:x1]
        crop_path = output_dir / f"page_{page_num}_borderless_table_{idx + 1}_crop.png"
        cv2.imwrite(str(crop_path), crop)

        try:
            from . import ocr_engine
            cleaned, line_mask = ocr_engine._remove_table_lines(crop, return_mask=True)  # type: ignore[attr-defined]
            nolines_path = output_dir / f"page_{page_num}_borderless_table_{idx + 1}_crop_nolines.png"
            cv2.imwrite(str(nolines_path), cleaned)
            mask_path = output_dir / f"page_{page_num}_borderless_table_{idx + 1}_crop_linemask.png"
            cv2.imwrite(str(mask_path), line_mask)
        except Exception:
            pass

        overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop.copy()
        # Draw outer bbox in red for reference.
        cv2.rectangle(overlay, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), (0, 0, 255), 2)

        for cell in table.get("cells", []):
            cb = cell.get("bbox_px") or []
            if len(cb) != 4:
                continue
            cx0, cy0, cx1, cy1 = (int(round(v)) for v in cb)
            cx0 -= x0
            cx1 -= x0
            cy0 -= y0
            cy1 -= y0
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            cv2.rectangle(overlay, (cx0, cy0), (cx1 - 1, cy1 - 1), (0, 200, 0), 1)

        border_path = output_dir / f"page_{page_num}_borderless_table_{idx + 1}_borders.png"
        cv2.imwrite(str(border_path), overlay)


def export_chart_debug_images(
    img_gray: "np.ndarray",
    charts: List[Dict],
    output_dir: Path,
    page_num: int
) -> None:
    """Export debug PNGs for detected charts."""
    if not HAVE_CV2 or img_gray is None or not charts:
        return
    h, w = img_gray.shape[:2]
    for idx, chart in enumerate(charts):
        bbox = chart.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in bbox)
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img_gray[y0:y1, x0:x1]
        crop_path = output_dir / f"page_{page_num}_chart_{idx + 1}_crop.png"
        cv2.imwrite(str(crop_path), crop)


def _render_table_ascii(table: Dict, *, mode: str | None = None) -> str:
    """
    Render a table as ASCII text.

    Modes:
    - "default": render using pre-assigned row/col indices (stable, but can't represent colspans well).
    - "replica": render using snapped bbox gridlines to better approximate the scanned layout.
    """
    mode = str(mode or "default").strip().lower()
    if mode in ("replica", "snap", "grid"):
        try:
            return _render_table_ascii_replica(table)
        except Exception:
            # Fall back to the original renderer if anything goes wrong.
            mode = "default"

    cells = table.get("cells", [])
    if not cells:
        return ""

    # Organize cells into rows (requires row/col indices).
    rows_dict = {}
    for cell in cells:
        row_idx = cell.get("row", 0)
        if row_idx not in rows_dict:
            rows_dict[row_idx] = []
        rows_dict[row_idx].append(cell)

    if not rows_dict:
        return ""

    # Sort rows and cells within rows
    sorted_rows = []
    for row_idx in sorted(rows_dict.keys()):
        row_cells = sorted(rows_dict[row_idx], key=lambda c: c.get("col", 0))
        sorted_rows.append(row_cells)

    # Calculate column widths
    max_cols = max(len(row) for row in sorted_rows) if sorted_rows else 0
    col_widths = [8] * max_cols  # minimum width

    for row in sorted_rows:
        for i, cell in enumerate(row):
            if i < max_cols:
                text = str(cell.get("text", ""))
                col_widths[i] = max(col_widths[i], min(40, len(text) + 2))

    # Build ASCII table
    lines = []
    separator = "+" + "+".join("-" * w for w in col_widths) + "+"
    lines.append(separator)

    for row in sorted_rows:
        row_text = []
        for i in range(max_cols):
            if i < len(row):
                text = str(row[i].get("text", ""))[: col_widths[i] - 2]
            else:
                text = ""
            row_text.append(f" {text.ljust(col_widths[i] - 2)} ")
        lines.append("|" + "|".join(row_text) + "|")
        lines.append(separator)

    return "\n".join(lines)


def _render_table_ascii_replica(table: Dict) -> str:
    cells = table.get("cells", [])
    if not cells:
        return ""

    # Collect bbox edges.
    edges_x: list[float] = []
    edges_y: list[float] = []
    usable_cells: list[dict] = []
    for cell in cells:
        bbox = cell.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        usable_cells.append(cell)
        edges_x.extend([x0, x1])
        edges_y.extend([y0, y1])

    if not usable_cells or len(edges_x) < 2 or len(edges_y) < 2:
        return ""

    min_x, max_x = min(edges_x), max(edges_x)
    min_y, max_y = min(edges_y), max(edges_y)
    table_w = max(1.0, max_x - min_x)
    table_h = max(1.0, max_y - min_y)

    def _infer_border_thickness_px(edges: list[float]) -> Optional[float]:
        vals = sorted(set(int(round(v)) for v in edges))
        if len(vals) < 3:
            return None
        gaps = [int(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        gaps_pos = [g for g in gaps if g > 0]
        if not gaps_pos:
            return None
        gaps_pos_sorted = sorted(gaps_pos)
        median_gap = float(gaps_pos_sorted[len(gaps_pos_sorted) // 2])
        small = [g for g in gaps if 2 <= g <= 12]
        if not small:
            return None
        counts: dict[int, int] = {}
        for g in small:
            counts[g] = counts.get(g, 0) + 1

        # Use the largest repeated "small" gap as a proxy for border thickness / row-to-row
        # bbox misalignment. This helps collapse tiny sliver columns created by outer-frame
        # insets (e.g. one row starts a few pixels to the right of another).
        repeated = [g for g, ct in counts.items() if int(ct) >= 2]
        if repeated:
            return float(max(repeated))

        # Fall back: allow singletons only on very small tables when the gap is clearly much
        # smaller than typical column widths.
        best_gap, best_count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        if best_count < 2:
            if len(gaps_pos) > 5:
                return None
            if median_gap <= 0 or float(best_gap) > (median_gap * 0.5):
                return None
        return float(best_gap)

    def _env_float(key: str, default: float) -> float:
        try:
            return float(str(os.environ.get(key, str(default))))
        except Exception:
            return default

    snap_tol_px = _env_float("EIDAT_TABLE_ASCII_SNAP_TOL_PX", 0.0)
    snap_ratio = _env_float("EIDAT_TABLE_ASCII_SNAP_TOL_RATIO", 0.006)
    snap_max_px = _env_float("EIDAT_TABLE_ASCII_SNAP_MAX_PX", 25.0)
    max_width = int(_env_float("EIDAT_TABLE_ASCII_MAX_WIDTH", 160.0))
    if max_width < 60:
        max_width = 60

    if snap_tol_px and snap_tol_px > 0:
        tol_x = float(snap_tol_px)
        tol_y = float(snap_tol_px)
    else:
        # Use axis-specific auto tolerances:
        # - X snapping should remain stable even for short (few-row) table segments.
        # - Y snapping needs a higher floor to collapse double-thickness horizontal borders
        #   (common in scanned tables) even when the table segment is short.
        tol_x = max(2.0, min(float(snap_max_px), float(table_w) * float(snap_ratio)))
        tol_y = max(6.0, min(float(snap_max_px), float(table_h) * float(snap_ratio)))

        # Extra robustness: collapse double-thickness borders even for small/short tables.
        thick_x = _infer_border_thickness_px(edges_x)
        thick_y = _infer_border_thickness_px(edges_y)
        if thick_x:
            tol_x = max(tol_x, min(float(snap_max_px), float(thick_x) * 1.25))
        if thick_y:
            tol_y = max(tol_y, min(float(snap_max_px), float(thick_y) * 1.25))

    def _cluster_positions(vals: list[float], tol_px: float) -> list[float]:
        if not vals:
            return []
        vals_sorted = sorted(float(v) for v in vals)
        clusters: list[list[float]] = [[vals_sorted[0]]]
        for v in vals_sorted[1:]:
            if abs(v - clusters[-1][-1]) <= tol_px:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        centers = [sum(c) / float(len(c)) for c in clusters if c]
        # Ensure strictly increasing and unique-ish.
        out: list[float] = []
        for c in centers:
            if not out or abs(c - out[-1]) > (tol_px * 0.25 + 1e-6):
                out.append(c)
        return out

    x_lines = _cluster_positions(edges_x, tol_x)
    y_lines = _cluster_positions(edges_y, tol_y)
    if len(x_lines) < 2 or len(y_lines) < 2:
        return ""

    # Ensure bounds are included.
    if abs(x_lines[0] - min_x) > tol_x:
        x_lines = [min_x] + x_lines
    if abs(x_lines[-1] - max_x) > tol_x:
        x_lines = x_lines + [max_x]
    if abs(y_lines[0] - min_y) > tol_y:
        y_lines = [min_y] + y_lines
    if abs(y_lines[-1] - max_y) > tol_y:
        y_lines = y_lines + [max_y]
    x_lines = sorted(set(float(v) for v in x_lines))
    y_lines = sorted(set(float(v) for v in y_lines))

    ncols = len(x_lines) - 1
    nrows = len(y_lines) - 1
    if ncols <= 0 or nrows <= 0:
        return ""

    # Column widths (chars) proportional to pixel widths, bounded by max width.
    col_px = [max(1.0, float(x_lines[i + 1] - x_lines[i])) for i in range(ncols)]
    total_px = float(sum(col_px)) or 1.0
    budget = max(20, int(max_width) - (ncols + 1))  # account for border/boundary chars
    col_widths = [max(3, int(round(budget * (w / total_px)))) for w in col_px]
    # Fix rounding drift.
    drift = int(budget - sum(col_widths))
    i = 0
    while drift != 0 and ncols > 0:
        j = i % ncols
        if drift > 0:
            col_widths[j] += 1
            drift -= 1
        else:
            if col_widths[j] > 3:
                col_widths[j] -= 1
                drift += 1
        i += 1

    def _closest_line_idx(lines: list[float], v: float) -> int:
        # Linear scan is fine for typical small tables.
        best_i = 0
        best_d = abs(float(lines[0]) - float(v))
        for i, lv in enumerate(lines[1:], start=1):
            d = abs(float(lv) - float(v))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    # Build snapped grid occupancy for colspans.
    grid: list[list[int | None]] = [[None for _ in range(ncols)] for _ in range(nrows)]
    cell_spans: dict[int, dict] = {}
    cells_by_area: list[tuple[float, int, dict]] = []
    for idx, cell in enumerate(usable_cells):
        bbox = cell.get("bbox_px") or []
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        area = max(1.0, (x1 - x0) * (y1 - y0))
        cells_by_area.append((area, idx, cell))
    cells_by_area.sort(key=lambda t: t[0])  # small first

    for _area, cid, cell in cells_by_area:
        bbox = cell.get("bbox_px") or []
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        xi0 = _closest_line_idx(x_lines, x0)
        xi1 = _closest_line_idx(x_lines, x1)
        yi0 = _closest_line_idx(y_lines, y0)
        yi1 = _closest_line_idx(y_lines, y1)
        if xi1 < xi0:
            xi0, xi1 = xi1, xi0
        if yi1 < yi0:
            yi0, yi1 = yi1, yi0
        if xi0 == xi1:
            xi1 = min(len(x_lines) - 1, xi0 + 1)
        if yi0 == yi1:
            yi1 = min(len(y_lines) - 1, yi0 + 1)
        if xi0 >= xi1 or yi0 >= yi1:
            continue
        cell_spans[cid] = {
            "x0": int(xi0),
            "x1": int(xi1),
            "y0": int(yi0),
            "y1": int(yi1),
            "text": str(cell.get("text", "") or ""),
        }
        for r in range(int(yi0), int(yi1)):
            for c in range(int(xi0), int(xi1)):
                if 0 <= r < nrows and 0 <= c < ncols and grid[r][c] is None:
                    grid[r][c] = int(cid)

    if not cell_spans:
        return ""

    def _v_present_for_row(r: int) -> list[bool]:
        present = [True] * (ncols + 1)
        for b in range(1, ncols):
            left = grid[r][b - 1]
            right = grid[r][b]
            # Hide boundary only when the same non-empty cell spans across it.
            present[b] = not (left is not None and left == right)
        present[0] = True
        present[-1] = True
        return present

    v_present_rows = [_v_present_for_row(r) for r in range(nrows)]

    # Drop fully empty grid rows. These can appear when snapped horizontal lines create
    # "gap" bands between real cell rows (e.g. slight bbox misalignment / scan noise).
    # Keeping them creates spurious blank ASCII rows that downstream parsers interpret
    # as real table rows.
    kept_rows: list[int] = []
    for r in range(nrows):
        if any(grid[r][c] is not None for c in range(ncols)):
            kept_rows.append(r)
    if not kept_rows:
        return ""

    def _sep_line(v_present: list[bool]) -> str:
        parts: list[str] = ["+"]
        for c in range(ncols):
            parts.append("-" * int(col_widths[c]))
            parts.append("+" if v_present[c + 1] else "-")
        return "".join(parts)

    def _row_line(r: int) -> str:
        v_present = v_present_rows[r]
        # Build segments based on which vertical boundaries are present.
        segs: list[tuple[int, int]] = []
        start = 0
        for c in range(1, ncols):
            if v_present[c]:
                segs.append((start, c))
                start = c
        segs.append((start, ncols))

        out: list[str] = ["|"]
        for (c0, c1) in segs:
            seg_ids = [grid[r][c] for c in range(c0, c1)]
            cell_id = next((cid for cid in seg_ids if cid is not None), None)
            text = ""
            if cell_id is not None and cell_id in cell_spans:
                span = cell_spans[cell_id]
                # Only render once: at the top-left of the spanning region.
                if r == int(span["y0"]) and c0 == int(span["x0"]):
                    text = str(span.get("text", "") or "")
            seg_width = int(sum(col_widths[c] for c in range(c0, c1))) + max(0, (c1 - c0 - 1))
            if seg_width <= 0:
                seg_width = 1
            clipped = text[:seg_width]
            out.append(clipped.ljust(seg_width))
            out.append("|")
        return "".join(out)

    lines: list[str] = []
    # Top separator uses the first kept row's vertical boundaries.
    first_r = kept_rows[0]
    lines.append(_sep_line(v_present_rows[first_r]))
    for idx, r in enumerate(kept_rows):
        lines.append(_row_line(r))
        if idx == len(kept_rows) - 1:
            lines.append(_sep_line(v_present_rows[r]))
        else:
            nxt = kept_rows[idx + 1]
            # Union of boundaries above/below to make the separator consistent.
            vp = [bool(a or b) for a, b in zip(v_present_rows[r], v_present_rows[nxt])]
            vp[0] = True
            vp[-1] = True
            lines.append(_sep_line(vp))

    return "\n".join(lines)


def _merge_positions_1d(values: List[float], tol_px: float) -> List[float]:
    if not values:
        return []
    tol = max(0.0, float(tol_px))
    vals = sorted(float(v) for v in values)
    if tol <= 0:
        out: List[float] = []
        last = None
        for v in vals:
            key = int(round(v))
            if last is None or key != last:
                out.append(float(key))
                last = key
        return out

    clusters: List[List[float]] = []
    cur = [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    return [sum(c) / float(len(c)) for c in clusters if c]


def _group_cells_into_rows_bbox(cells: List[Dict]) -> List[List[Dict]]:
    usable: List[Dict] = []
    heights: List[float] = []
    for c in cells:
        bbox = c.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            y0 = float(bbox[1])
            y1 = float(bbox[3])
        except Exception:
            continue
        h = y1 - y0
        if h <= 0:
            continue
        usable.append(c)
        heights.append(h)
    if not usable:
        return []

    heights.sort()
    median_h = float(heights[len(heights) // 2]) if heights else 0.0
    row_tol = max(8.0, median_h * 0.60)

    def _cy(cell: Dict) -> float:
        x0, y0, x1, y1 = cell["bbox_px"]
        return (float(y0) + float(y1)) / 2.0

    sorted_cells = sorted(usable, key=lambda c: (_cy(c), float(c["bbox_px"][0])))
    rows: List[List[Dict]] = []
    cur: List[Dict] = []
    cur_cy: Optional[float] = None

    for cell in sorted_cells:
        cy = _cy(cell)
        if cur and cur_cy is not None and abs(cy - cur_cy) > row_tol:
            rows.append(sorted(cur, key=lambda c: float(c["bbox_px"][0])))
            cur = [cell]
            cur_cy = cy
            continue
        cur.append(cell)
        if cur_cy is None:
            cur_cy = cy
        else:
            cur_cy = (cur_cy * (len(cur) - 1) + cy) / float(len(cur))

    if cur:
        rows.append(sorted(cur, key=lambda c: float(c["bbox_px"][0])))
    return rows


def _row_vline_signature_bbox(
    row_cells: List[Dict],
    *,
    table_left: float,
    table_right: float,
    tol_px: float,
) -> Optional[List[float]]:
    if len(row_cells) < 2:
        return None
    xs: List[float] = []
    for c in row_cells:
        bbox = c.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            xs.append(float(bbox[0]))
            xs.append(float(bbox[2]))
        except Exception:
            continue
    if not xs:
        return None
    merged = _merge_positions_1d(xs, tol_px=float(tol_px))
    tol = max(0.0, float(tol_px))
    internal = [
        x
        for x in merged
        if (x - float(table_left)) > tol and (float(table_right) - x) > tol
    ]
    internal.sort()
    return internal


def _vline_signatures_match(a: Optional[List[float]], b: Optional[List[float]], tol_px: float) -> bool:
    if a is None or b is None:
        return a == b
    if len(a) != len(b):
        return False
    tol = max(0.0, float(tol_px))
    for x, y in zip(a, b):
        if abs(float(x) - float(y)) > tol:
            return False
    return True


def _split_table_on_vline_mismatch_for_display(
    table: Dict,
    tol_px: float,
    *,
    min_mismatch_run: int = 1,
    enable_single_cell_separator: bool = True,
    single_cell_separator_max_h_ratio: float = 0.0,
) -> List[Dict]:
    """
    Pure ASCII/layout split: splits a detected bordered table into multiple tables when
    internal vertical gridlines are not aligned from row to row.

    This does NOT perform any OCR; it just re-groups existing cells for rendering/output order.
    """
    try:
        tol = float(tol_px)
    except Exception:
        tol = 0.0
    tol = max(0.0, tol)
    try:
        min_run = int(min_mismatch_run)
    except Exception:
        min_run = 1
    if min_run < 1:
        min_run = 1
    try:
        sep_max_h_ratio = float(single_cell_separator_max_h_ratio)
    except Exception:
        sep_max_h_ratio = 0.0
    if sep_max_h_ratio < 0:
        sep_max_h_ratio = 0.0

    if bool(table.get("borderless", False)):
        return [table]
    cells = table.get("cells") or []
    if not isinstance(cells, list) or len(cells) < 2:
        return [table]

    rows = _group_cells_into_rows_bbox(cells)
    if len(rows) < 2:
        return [table]

    # Global bounds from cells.
    xs0: List[float] = []
    xs1: List[float] = []
    for c in cells:
        bbox = c.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            xs0.append(float(bbox[0]))
            xs1.append(float(bbox[2]))
        except Exception:
            continue
    if not xs0 or not xs1:
        return [table]
    table_left = min(xs0)
    table_right = max(xs1)
    if table_right <= table_left:
        return [table]

    def _row_h(row_cells: List[Dict]) -> float:
        ys0: List[float] = []
        ys1: List[float] = []
        for c in row_cells:
            bbox = c.get("bbox_px") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                ys0.append(float(bbox[1]))
                ys1.append(float(bbox[3]))
            except Exception:
                continue
        if not ys0 or not ys1:
            return 0.0
        return max(0.0, max(ys1) - min(ys0))

    median_multi_row_h = 0.0
    multi_heights = sorted([_row_h(r) for r in rows if len(r) >= 2 and _row_h(r) > 0])
    if multi_heights:
        median_multi_row_h = float(multi_heights[len(multi_heights) // 2])

    is_multi = [len(r) >= 2 for r in rows]
    prefix_has_multi = []
    seen = False
    for flag in is_multi:
        prefix_has_multi.append(seen)
        seen = seen or bool(flag)
    suffix_has_multi = [False] * len(rows)
    seen = False
    for i in range(len(rows) - 1, -1, -1):
        suffix_has_multi[i] = seen
        seen = seen or bool(is_multi[i])

    sigs: List[Optional[List[float]]] = []
    for row in rows:
        sig = (
            _row_vline_signature_bbox(
                row, table_left=table_left, table_right=table_right, tol_px=tol
            )
            if len(row) >= 2
            else None
        )
        sigs.append(sig)

    segments: List[List[List[Dict]]] = []
    cur: List[List[Dict]] = []
    cur_sig: Optional[List[float]] = None

    i = 0
    while i < len(rows):
        row = rows[i]
        sig = sigs[i]

        if (
            enable_single_cell_separator
            and len(row) <= 1
            and prefix_has_multi[i]
            and suffix_has_multi[i]
        ):
            is_sep = True
            if sep_max_h_ratio > 0 and median_multi_row_h > 0:
                is_sep = _row_h(row) <= (median_multi_row_h * sep_max_h_ratio)
            if is_sep:
                if cur:
                    segments.append(cur)
                    cur = []
                    cur_sig = None
                segments.append([row])
                i += 1
                continue

        if not cur:
            cur = [row]
            cur_sig = sig
            i += 1
            continue

        if sig is None:
            cur.append(row)
            i += 1
            continue

        if cur_sig is None:
            cur.append(row)
            cur_sig = sig
            i += 1
            continue

        if _vline_signatures_match(cur_sig, sig, tol_px=tol):
            cur.append(row)
            i += 1
            continue

        run = 1
        j = i + 1
        while j < len(rows) and run < min_run:
            nxt = sigs[j]
            if nxt is None:
                j += 1
                continue
            if _vline_signatures_match(sig, nxt, tol_px=tol):
                run += 1
                j += 1
                continue
            break

        if run >= min_run:
            segments.append(cur)
            cur = [row]
            cur_sig = sig
            i += 1
            continue

        cur.append(row)
        i += 1

    if cur:
        segments.append(cur)

    if len(segments) <= 1:
        return [table]

    out: List[Dict] = []
    for seg_rows in segments:
        seg_cells = [c for r in seg_rows for c in r]
        if not seg_cells:
            continue
        x0 = min(float(c["bbox_px"][0]) for c in seg_cells)
        y0 = min(float(c["bbox_px"][1]) for c in seg_cells)
        x1 = max(float(c["bbox_px"][2]) for c in seg_cells)
        y1 = max(float(c["bbox_px"][3]) for c in seg_cells)
        tb = dict(table)
        tb["cells"] = seg_cells
        tb["bbox_px"] = [x0, y0, x1, y1]
        tb["num_cells"] = len(seg_cells)
        out.append(tb)

    return out or [table]


def export_combined_text(pdf_path: Path, pages_data: List[Dict],
                          output_dir: Path) -> Path:
    """
    Export combined text from all pages including tables.

    Args:
        pdf_path: Source PDF
        pages_data: List of page results
        output_dir: Output directory

    Returns:
        Path to combined text file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _line_bounds(line: List[Dict]) -> Dict[str, float]:
        if not line:
            return {'y0': 0.0, 'y1': 0.0, 'h': 0.0}
        y0 = min(t.get('y0', 0) for t in line)
        y1 = max(t.get('y1', 0) for t in line)
        h = max(1.0, float(y1) - float(y0))
        return {'y0': float(y0), 'y1': float(y1), 'h': h}

    def _line_text(line: List[Dict]) -> str:
        """
        Render a token line as text, inserting " | " when there's a single clear horizontal gap.

        This helps represent two-column "term value" lines that aren't boxed as tables, e.g.:
          Test Plan         TPL-1000  ->  Test Plan | TPL-1000
        """
        enable = str(os.environ.get("EIDAT_KV_GAP_SPLIT", "1") or "1").strip().lower() not in {
            "0",
            "false",
            "f",
            "no",
            "n",
            "off",
        }
        toks = [t for t in (line or []) if str(t.get("text", "")).strip()]
        if not toks:
            return ""
        if not enable:
            return page_analyzer.extract_line_text(toks).strip()

        # Sort tokens left-to-right; fall back to original order if x is missing.
        def _x0(tok: Dict) -> float:
            try:
                return float(tok.get("x0", tok.get("left", 0.0)) or 0.0)
            except Exception:
                return 0.0

        def _x1(tok: Dict) -> float:
            try:
                return float(tok.get("x1", tok.get("right", 0.0)) or 0.0)
            except Exception:
                return 0.0

        try:
            toks_sorted = sorted(toks, key=_x0)
        except Exception:
            toks_sorted = toks

        b = _line_bounds(toks_sorted)
        h = float(b.get("h", 0.0) or 0.0)
        try:
            min_px = float(os.environ.get("EIDAT_KV_GAP_MIN_PX", "0") or 0.0)
        except Exception:
            min_px = 0.0
        try:
            h_mult = float(os.environ.get("EIDAT_KV_GAP_H_MULT", "2.5") or 2.5)
        except Exception:
            h_mult = 2.5
        gap_thr = max(min_px, max(40.0, h * h_mult))

        # Identify large gaps between adjacent tokens.
        breaks = []
        for i in range(len(toks_sorted) - 1):
            g = _x0(toks_sorted[i + 1]) - _x1(toks_sorted[i])
            if g >= gap_thr:
                breaks.append(i)
                if len(breaks) > 1:
                    break

        # Only apply when there's a single clear split; otherwise keep normal spacing.
        if len(breaks) != 1:
            return page_analyzer.extract_line_text(toks_sorted).strip()

        i = breaks[0]
        left = page_analyzer.extract_line_text(toks_sorted[: i + 1]).strip()
        right = page_analyzer.extract_line_text(toks_sorted[i + 1 :]).strip()
        if not left or not right:
            return page_analyzer.extract_line_text(toks_sorted).strip()
        if "|" in left or "|" in right:
            return page_analyzer.extract_line_text(toks_sorted).strip()
        if not re.search(r"[A-Za-z]", left) or not re.search(r"[A-Za-z0-9]", right):
            return page_analyzer.extract_line_text(toks_sorted).strip()
        return f"{left} | {right}"

    def _paragraph_lines(paragraph: List[List[Dict]]) -> List[str]:
        parts = [_line_text(line) for line in paragraph]
        return [p for p in parts if p]

    def _normalize_repeat_key(text: str) -> str:
        key = text.lower()
        key = re.sub(r"\d+", "#", key)
        key = re.sub(r"[^a-z0-9#]+", " ", key)
        key = re.sub(r"\s+", " ", key).strip()
        return key

    def _collect_candidate_lines(flow: Dict, kind: str) -> List[Dict]:
        lines = flow.get(f"{kind}_lines")
        if not lines:
            tokens = flow.get(f"{kind}s", [])
            lines = page_analyzer.group_tokens_into_lines(tokens)
        entries = []
        for line in lines or []:
            text = _line_text(line)
            if not text:
                continue
            entries.append({
                'line': line,
                'text': text,
                'key': _normalize_repeat_key(text)
            })
        return entries

    total_pages = len(pages_data)
    min_pages = max(2, int(math.ceil(total_pages * 0.6))) if total_pages > 0 else 2
    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}
    header_page_entries: List[List[Dict]] = []
    footer_page_entries: List[List[Dict]] = []

    for page_data in pages_data:
        flow = page_data.get('flow') or {}
        header_entries = _collect_candidate_lines(flow, "header")
        footer_entries = _collect_candidate_lines(flow, "footer")
        header_page_entries.append(header_entries)
        footer_page_entries.append(footer_entries)
        for entry in header_entries:
            header_counts[entry['key']] = header_counts.get(entry['key'], 0) + 1
        for entry in footer_entries:
            footer_counts[entry['key']] = footer_counts.get(entry['key'], 0) + 1

    header_repeat_keys = {k for k, v in header_counts.items() if v >= min_pages}
    footer_repeat_keys = {k for k, v in footer_counts.items() if v >= min_pages}

    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_chart_crops(pdf_path, pages_data, output_dir)

    combined_text = []
    split_mode = str(os.environ.get("EIDAT_TABLE_SPLIT_MODE", "vline") or "vline").strip().lower()
    ascii_mode = "replica" if split_mode in ("replica", "snap", "grid") else "default"
    combined_split_mode = str(
        os.environ.get("EIDAT_COMBINED_TABLE_SPLIT_MODE", "inherit") or "inherit"
    ).strip().lower()
    if combined_split_mode in ("inherit", "same"):
        combined_split_mode = split_mode
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
    for page_idx, page_data in enumerate(pages_data):
        page_num = page_data.get('page', 0)
        tokens = page_data.get('tokens', [])
        tables = page_data.get('tables', [])
        flow = page_data.get('flow') or {}
        charts = page_data.get('charts', []) or []
        chart_bboxes = [c.get("bbox_px") for c in charts if c.get("bbox_px")]
        confirmed_footers: List[List[Dict]] = []

        def _line_bbox(line: List[Dict]) -> Dict[str, float]:
            if not line:
                return {'x0': 0.0, 'y0': 0.0, 'x1': 0.0, 'y1': 0.0}
            x0 = min(t.get('x0', 0) for t in line)
            y0 = min(t.get('y0', 0) for t in line)
            x1 = max(t.get('x1', 0) for t in line)
            y1 = max(t.get('y1', 0) for t in line)
            return {'x0': float(x0), 'y0': float(y0), 'x1': float(x1), 'y1': float(y1)}

        def _line_in_chart(line: List[Dict]) -> bool:
            if not chart_bboxes or not line:
                return False
            bbox = _line_bbox(line)
            cx = (bbox['x0'] + bbox['x1']) / 2.0
            cy = (bbox['y0'] + bbox['y1']) / 2.0
            for cb in chart_bboxes:
                if not cb or len(cb) != 4:
                    continue
                x0, y0, x1, y1 = (float(v) for v in cb)
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    return True
            return False

        combined_text.append(f"=== Page {page_num} ===\n")

        if flow:
            header_entries = header_page_entries[page_idx] if page_idx < len(header_page_entries) else []
            footer_entries = footer_page_entries[page_idx] if page_idx < len(footer_page_entries) else []
            confirmed_headers = [e['line'] for e in header_entries if e['key'] in header_repeat_keys]
            confirmed_footers = [e['line'] for e in footer_entries if e['key'] in footer_repeat_keys]
            title_entries = flow.get('table_titles', [])

            non_table_tokens = flow.get('non_table_tokens') or flow.get('body_tokens') or []
            exclude_ids = set()
            for line in confirmed_headers + confirmed_footers:
                exclude_ids.update(id(t) for t in line)
            for entry in title_entries:
                for line in entry.get('lines', []) or []:
                    exclude_ids.update(id(t) for t in line)

            body_tokens = [t for t in non_table_tokens if id(t) not in exclude_ids]
            body_lines = page_analyzer.group_tokens_into_lines(body_tokens)
            standalone_lines, paragraphs = page_analyzer.split_lines_and_paragraphs(body_lines)

            artifacts = []
            order_idx = 0

            def _artifact_bounds(lines: List[List[Dict]]) -> Dict[str, float]:
                y0 = min(_line_bounds(line)['y0'] for line in lines)
                x0 = min(min(t.get('x0', 0) for t in line) for line in lines)
                return {'y0': y0, 'x0': float(x0)}

            def _add_artifact(category: str, lines: List[List[Dict]]) -> None:
                nonlocal order_idx
                bounds = _artifact_bounds(lines)
                artifacts.append({
                    'category': category,
                    'lines': lines,
                    'y0': bounds['y0'],
                    'x0': bounds['x0'],
                    'order': order_idx
                })
                order_idx += 1

            def _add_table_artifact(table_idx: int, table: Dict) -> None:
                nonlocal order_idx
                bbox = table.get('bbox_px', [])
                if len(bbox) != 4:
                    return
                artifacts.append({
                    'category': "Table",
                    'table_idx': table_idx,
                    'table': table,
                    'y0': float(bbox[1]),
                    'x0': float(bbox[0]),
                    'order': order_idx
                })
                order_idx += 1

            for line in confirmed_headers:
                _add_artifact("Header", [line])

            for entry in title_entries:
                title_lines = entry.get('lines', []) or []
                if title_lines:
                    filtered = [ln for ln in title_lines if not _line_in_chart(ln)]
                    if filtered:
                        _add_artifact("Table/Chart Title", filtered)

            for line in standalone_lines:
                if not _line_in_chart(line):
                    _add_artifact("Line", [line])

            for paragraph in paragraphs:
                filtered = [ln for ln in paragraph if not _line_in_chart(ln)]
                if filtered:
                    _add_artifact("Paragraph", filtered)

            for i, table in enumerate(tables):
                if enable_combined_vline_split:
                    try:
                        parts = _split_table_on_vline_mismatch_for_display(
                            table,
                            tol_px=float(combined_vline_tol_px),
                            min_mismatch_run=int(combined_vline_min_run),
                            enable_single_cell_separator=bool(single_cell_sep),
                            single_cell_separator_max_h_ratio=float(single_cell_sep_max_h_ratio),
                        )
                    except Exception:
                        parts = [table]
                else:
                    parts = [table]
                for part in parts:
                    _add_table_artifact(i, part)

            artifacts.sort(key=lambda a: (a['y0'], a['x0'], a['order']))

            table_counter = 0
            for art in artifacts:
                combined_text.append(f"[{art['category']}]\n")
                if art['category'] == "Table":
                    table_counter += 1
                    combined_text.append(f"\n[Table {table_counter}]\n")
                    table_obj = art.get("table") or {}
                    table_ascii = _render_table_ascii(table_obj, mode=ascii_mode)
                    if table_ascii:
                        combined_text.append(table_ascii)
                        combined_text.append("\n")
                elif art['category'] == "Paragraph":
                    for text in _paragraph_lines(art['lines']):
                        combined_text.append(f"{text}\n")
                else:
                    for line in art['lines']:
                        text = _line_text(line)
                        if text:
                            combined_text.append(f"{text}\n")
                combined_text.append("\n")
        else:
            # Add flow text from tokens (legacy fallback)
            if chart_bboxes and page_data.get("dpi") and page_data.get("ocr_dpi"):
                scale = float(page_data.get("ocr_dpi", 1.0)) / float(page_data.get("dpi", 1.0))
                scaled_boxes = []
                for cb in chart_bboxes:
                    if not cb or len(cb) != 4:
                        continue
                    scaled_boxes.append([
                        cb[0] * scale, cb[1] * scale,
                        cb[2] * scale, cb[3] * scale
                    ])
                def _token_in_chart(tok: Dict) -> bool:
                    if not scaled_boxes:
                        return False
                    cx = float(tok.get("cx", (tok.get("x0", 0) + tok.get("x1", 0)) / 2.0))
                    cy = float(tok.get("cy", (tok.get("y0", 0) + tok.get("y1", 0)) / 2.0))
                    for cb in scaled_boxes:
                        if cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]:
                            return True
                    return False
                flow_text = ' '.join(t.get('text', '') for t in tokens if t.get('text', '') and not _token_in_chart(t))
            else:
                flow_text = ' '.join(t.get('text', '') for t in tokens if t.get('text', ''))
            if flow_text:
                combined_text.append(flow_text)
                combined_text.append("\n")

        # Footers are written last so they appear after all content.
        for line in confirmed_footers:
            combined_text.append("[Footer]\n")
            text = _line_text(line)
            if text:
                combined_text.append(f"{text}\n")
            combined_text.append("\n")

        combined_text.append("\n")

    output_file = output_dir / "combined.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(combined_text)

    return output_file


def _ensure_chart_crops(pdf_path: Path, pages_data: List[Dict], output_dir: Path) -> None:
    if not HAVE_CV2 or not pages_data:
        return
    try:
        from . import ocr_engine
    except Exception:
        return
    for page_idx, page_data in enumerate(pages_data):
        charts = page_data.get("charts", []) or []
        if not charts:
            continue
        page_num = page_data.get("page", page_idx + 1)
        expected = [
            output_dir / f"page_{page_num}_chart_{idx + 1}_crop.png"
            for idx in range(len(charts))
        ]
        if all(p.exists() for p in expected):
            continue
        dpi = int(page_data.get("dpi") or 0) or 900
        img_gray, _, _ = ocr_engine.render_pdf_page(pdf_path, page_idx, dpi)
        if img_gray is None:
            continue
        export_chart_debug_images(img_gray, charts, output_dir, page_num)


def create_summary_report(pdf_path: Path, pages_data: List[Dict],
                           output_dir: Path) -> Path:
    """
    Create extraction summary report.

    Args:
        pdf_path: Source PDF
        pages_data: List of page results
        output_dir: Output directory

    Returns:
        Path to summary JSON file
    """
    total_tokens = sum(len(p.get('tokens', [])) for p in pages_data)
    total_tables = sum(len(p.get('tables', [])) for p in pages_data)
    total_cells = sum(
        sum(t.get('num_cells', 0) for t in p.get('tables', []))
        for p in pages_data
    )

    merged_method_summary = _merge_method_summaries(
        [_build_ocr_method_summary(p.get('tables', [])) for p in pages_data]
    )
    merged_pass_stats = _merge_pass_stats([p.get('ocr_pass_stats') for p in pages_data])

    pages = []
    for p in pages_data:
        page_tables = p.get('tables', [])
        pages.append({
            'page': p.get('page'),
            'tokens': len(p.get('tokens', [])),
            'tables': len(page_tables),
            'cells': sum(t.get('num_cells', 0) for t in page_tables),
            'ocr_method_summary': _build_ocr_method_summary(page_tables)
        })

    summary = {
        'pdf_file': str(pdf_path),
        'total_pages': len(pages_data),
        'total_tokens': total_tokens,
        'total_tables': total_tables,
        'total_cells': total_cells,
        'timestamp': datetime.now().isoformat(),
        'ocr_method_summary': merged_method_summary,
        'ocr_pass_stats': merged_pass_stats,
        'pages': pages
    }

    output_file = output_dir / "summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return output_file


def export_table_projection_debug(
    page_num: int,
    tables: List[Dict],
    projection_debug: Dict,
    all_tokens: List[Dict],
    output_dir: Path,
    ocr_dpi: int = 450,
    detection_dpi: int = 900
) -> Path:
    """
    Export detailed table projection debug info showing DPI scaling and overlap.

    Creates a human-readable debug file per page showing:
    - Table-region OCR PSM selection and re-OCR summary
    - Per-cell token matches and combined text

    Args:
        page_num: Page number (0-indexed)
        tables: List of detected tables
        projection_debug: Debug info from project_tokens_to_cells_force
        all_tokens: All OCR tokens (at OCR DPI, before scaling)
        output_dir: Debug output directory
        ocr_dpi: DPI tokens were extracted at
        detection_dpi: DPI cells were detected at

    Returns:
        Path to debug file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append(f"=" * 80)
    lines.append(f"TABLE PROJECTION DEBUG - Page {page_num + 1}")
    lines.append(f"=" * 80)
    lines.append(f"")
    lines.append("Strategy: table-region OCR with PSM selection, token projection, token re-OCR")
    lines.append("")

    # Table OCR / PSM summary
    table_ocr = projection_debug.get("table_ocr", []) if projection_debug else []
    if table_ocr:
        lines.append("Table OCR / PSM summary:")
        for entry in table_ocr:
            table_idx = entry.get("table_idx")
            psms_tried = entry.get("psms_tried", [])
            primary_psm = entry.get("primary_psm")
            extra_psms = entry.get("extra_psms", [])
            primary_reocr = entry.get("primary_reocr_count", 0)
            extra_reocr = entry.get("extra_reocr_count", 0)
            extra_psm_reocr = entry.get("extra_psm_reocr", {})

            tried_str = ", ".join(str(p) for p in psms_tried) if psms_tried else "n/a"
            extra_str = ", ".join(str(p) for p in extra_psms) if extra_psms else "none"
            lines.append(
                f"  Table {table_idx}: primary_psm={primary_psm} extras={extra_str} tried={tried_str} "
                f"reocr(primary={primary_reocr}, extras={extra_reocr})"
            )
            if extra_psm_reocr:
                parts = [f"{k}:{v}" for k, v in sorted(extra_psm_reocr.items())]
                lines.append(f"    extra_psm_reocr: {', '.join(parts)}")
        lines.append("")

    # Re-OCR summary
    reocr_tokens = [t for t in all_tokens if t.get("reocr")]
    if reocr_tokens:
        by_source: Dict[str, int] = {}
        by_psm: Dict[str, int] = {}
        for tok in reocr_tokens:
            source = tok.get("_source", "page_ocr")
            by_source[source] = by_source.get(source, 0) + 1
            psm = tok.get("_psm")
            if psm is not None:
                key = str(psm)
                by_psm[key] = by_psm.get(key, 0) + 1

        lines.append("Re-OCR summary:")
        lines.append(f"  Tokens improved: {len(reocr_tokens)}")
        if by_source:
            parts = [f"{k}={v}" for k, v in sorted(by_source.items())]
            lines.append(f"  By source: {', '.join(parts)}")
        if by_psm:
            parts = [f"psm{k}={v}" for k, v in sorted(by_psm.items())]
            lines.append(f"  By PSM: {', '.join(parts)}")
        lines.append("  Sample re-OCR tokens:")
        for tok in reocr_tokens[:15]:
            text = tok.get("text", "")
            source = tok.get("_source", "page_ocr")
            psm = tok.get("_psm")
            table_idx = tok.get("_table_idx")
            loc = f"table={table_idx}" if table_idx else "table=n/a"
            lines.append(f"    '{text}' source={source} psm={psm} {loc}")
        lines.append("")

    # Summary stats
    total_cells = sum(len(t.get('cells', [])) for t in tables)
    cells_with_text = 0
    cells_empty = 0

    for table in tables:
        for cell in table.get('cells', []):
            if cell.get('text', '').strip():
                cells_with_text += 1
            else:
                cells_empty += 1

    lines.append(f"Summary:")
    lines.append(f"  Total tables:       {len(tables)}")
    lines.append(f"  Total cells:        {total_cells}")
    lines.append(f"  Cells with text:    {cells_with_text}")
    lines.append(f"  Cells EMPTY:        {cells_empty}  {'<-- PROBLEM!' if cells_empty > cells_with_text else ''}")
    lines.append(f"  Total tokens (OCR): {len(all_tokens)}")
    lines.append(f"")

    # Per-table debug
    for table_idx, table in enumerate(tables):
        table_bbox = table.get('bbox_px', [0, 0, 0, 0])
        cells = table.get('cells', [])

        lines.append(f"-" * 80)
        lines.append(f"TABLE {table_idx + 1}")
        lines.append(f"-" * 80)
        lines.append(f"Table bbox: {table_bbox}")
        lines.append(f"Cells: {len(cells)}")
        lines.append(f"")

        # Per-cell debug
        for cell_idx, cell in enumerate(cells):
            cx0, cy0, cx1, cy1 = cell.get('bbox_px', [0, 0, 0, 0])
            cell_text = cell.get('text', '')
            cell_tokens = cell.get('tokens', [])

            lines.append(f"  Cell [{cell_idx}] row={cell.get('row')} col={cell.get('col')}")
            lines.append(f"    bbox: [{cx0:.1f}, {cy0:.1f}, {cx1:.1f}, {cy1:.1f}]")
            lines.append(f"    tokens matched: {len(cell_tokens)}")

            # Determine text source
            ocr_method = cell.get('ocr_method', 'token_projection' if cell_tokens else 'none')
            if cell_text:
                if ocr_method == 'cell_ocr_fallback':
                    lines.append(f"    text: '{cell_text[:60]}{'...' if len(cell_text) > 60 else ''}' [via CELL OCR FALLBACK]")
                elif cell_tokens:
                    lines.append(f"    text: '{cell_text[:60]}{'...' if len(cell_text) > 60 else ''}' [via TOKEN PROJECTION]")
                else:
                    lines.append(f"    text: '{cell_text[:60]}{'...' if len(cell_text) > 60 else ''}' [source unknown]")
            else:
                lines.append(f"    text: '' [TRULY EMPTY]")

            if cell_tokens:
                lines.append(f"    Token details:")
                for tok in cell_tokens[:5]:  # Limit to 5 for readability
                    tok_x0 = tok.get('x0', 0)
                    tok_y0 = tok.get('y0', 0)
                    tok_x1 = tok.get('x1', 0)
                    tok_y1 = tok.get('y1', 0)
                    overlap = tok.get('overlap_ratio', 0) * 100
                    lines.append(f"      '{tok['text']}' [{tok_x0:.1f},{tok_y0:.1f},{tok_x1:.1f},{tok_y1:.1f}] overlap:{overlap:.1f}%")
                if len(cell_tokens) > 5:
                    lines.append(f"      ... and {len(cell_tokens) - 5} more tokens")
            elif not cell_text:
                lines.append(f"    ** TRULY EMPTY - no tokens matched and fallback OCR found nothing **")

            lines.append(f"")

    lines.append(f"")
    lines.append(f"=" * 80)
    lines.append(f"END DEBUG")
    lines.append(f"=" * 80)

    output_file = output_dir / f"page_{page_num + 1}_projection_debug.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_file
