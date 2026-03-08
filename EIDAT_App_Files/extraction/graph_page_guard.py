"""
Cheap guard for dense graph/grid pages that should skip table work.

The goal is not perfect chart detection. It only needs to identify pages with
large, dense, regular orthogonal ruling that tends to explode bordered-table
work while preserving OCR-only fallback for the page.
"""

from __future__ import annotations

from typing import Dict, List

try:
    import cv2
    import numpy as np

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def _cluster_positions(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    vals = sorted(float(v) for v in values)
    tol_f = max(0.0, float(tol))
    groups: List[List[float]] = [[vals[0]]]
    for val in vals[1:]:
        if abs(val - groups[-1][-1]) <= tol_f:
            groups[-1].append(val)
        else:
            groups.append([val])
    return [sum(group) / float(len(group)) for group in groups]


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _gap_stats(coords: List[float]) -> tuple[float, float]:
    if len(coords) < 3:
        return 0.0, 0.0
    gaps = [float(coords[i] - coords[i - 1]) for i in range(1, len(coords))]
    med = _median(gaps)
    if med <= 0:
        return med, 999.0
    avg_abs_dev = sum(abs(g - med) for g in gaps) / float(len(gaps))
    return med, avg_abs_dev / med


def _line_components(mask: "np.ndarray", axis: str, img_w: int, img_h: int) -> List[Dict[str, float]]:
    comps: List[Dict[str, float]] = []
    try:
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    except Exception:
        return comps

    for i in range(1, int(num)):
        x, y, w, h, area = stats[i]
        if area <= 0:
            continue
        if axis == "h":
            if w < max(80, int(round(float(img_w) * 0.18))):
                continue
            if h > max(8, int(round(float(img_h) * 0.02))):
                continue
            comps.append(
                {
                    "pos": float(y) + float(h) / 2.0,
                    "start": float(x),
                    "end": float(x + w),
                    "len": float(w),
                }
            )
        else:
            if h < max(80, int(round(float(img_h) * 0.18))):
                continue
            if w > max(8, int(round(float(img_w) * 0.02))):
                continue
            comps.append(
                {
                    "pos": float(x) + float(w) / 2.0,
                    "start": float(y),
                    "end": float(y + h),
                    "len": float(h),
                }
            )
    return comps


def inspect_page_for_graph_grid(img_gray: object) -> Dict[str, object]:
    """
    Return a best-effort graph/grid classification for a page image.

    Output shape:
      {
        "skip_table_work": bool,
        "reason": str,
        "stats": {...}
      }
    """
    result: Dict[str, object] = {
        "skip_table_work": False,
        "reason": "",
        "stats": {},
    }
    if not HAVE_CV2:
        return result
    if not isinstance(img_gray, np.ndarray) or img_gray.ndim != 2 or img_gray.size == 0:
        return result

    img_h, img_w = img_gray.shape[:2]
    if img_h < 100 or img_w < 100:
        return result

    try:
        _, bin_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        return result

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(round(float(img_w) * 0.08))), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(round(float(img_h) * 0.08)))))
    try:
        horiz = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel)
        vert = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel)
    except Exception:
        return result

    horiz_comps = _line_components(horiz, "h", img_w, img_h)
    vert_comps = _line_components(vert, "v", img_w, img_h)
    h_positions = _cluster_positions(
        [float(comp["pos"]) for comp in horiz_comps],
        tol=max(3.0, float(img_h) * 0.004),
    )
    v_positions = _cluster_positions(
        [float(comp["pos"]) for comp in vert_comps],
        tol=max(3.0, float(img_w) * 0.004),
    )

    if horiz_comps and vert_comps:
        grid_x0 = min(float(comp["start"]) for comp in horiz_comps)
        grid_x1 = max(float(comp["end"]) for comp in horiz_comps)
        grid_y0 = min(float(comp["start"]) for comp in vert_comps)
        grid_y1 = max(float(comp["end"]) for comp in vert_comps)
    else:
        grid_x0 = grid_x1 = grid_y0 = grid_y1 = 0.0

    area_ratio = 0.0
    if grid_x1 > grid_x0 and grid_y1 > grid_y0:
        area_ratio = ((grid_x1 - grid_x0) * (grid_y1 - grid_y0)) / float(max(1, img_w * img_h))

    h_gap_med, h_gap_cv = _gap_stats(h_positions)
    v_gap_med, v_gap_cv = _gap_stats(v_positions)
    h_count = len(h_positions)
    v_count = len(v_positions)
    crossings = h_count * v_count
    axis_balance = (float(min(h_count, v_count)) / float(max(h_count, v_count))) if max(h_count, v_count) > 0 else 0.0

    result["stats"] = {
        "h_lines": int(h_count),
        "v_lines": int(v_count),
        "crossings": int(crossings),
        "axis_balance": float(axis_balance),
        "grid_area_ratio": float(area_ratio),
        "h_gap_med": float(h_gap_med),
        "v_gap_med": float(v_gap_med),
        "h_gap_cv": float(h_gap_cv),
        "v_gap_cv": float(v_gap_cv),
        "grid_bbox": [float(grid_x0), float(grid_y0), float(grid_x1), float(grid_y1)],
    }

    dense_enough = h_count >= 8 and v_count >= 8 and crossings >= 80
    balanced_axes = axis_balance >= 0.7
    wide_enough = area_ratio >= 0.08
    regular_enough = h_gap_cv <= 0.45 and v_gap_cv <= 0.45
    fine_spacing = (
        h_gap_med > 0
        and v_gap_med > 0
        and h_gap_med <= max(90.0, float(img_h) * 0.08)
        and v_gap_med <= max(90.0, float(img_w) * 0.08)
    )

    if dense_enough and balanced_axes and wide_enough and regular_enough and fine_spacing:
        result["skip_table_work"] = True
        result["reason"] = "dense_regular_orthogonal_grid"

    return result
