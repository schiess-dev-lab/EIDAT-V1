"""
Cheap guard for dense graph/grid regions that should be ignored by table work.

The goal is not perfect chart detection. It only needs to identify chart-like
subregions with dense, regular orthogonal ruling so they can be masked before
bordered-table work while preserving OCR on the original page.
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


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


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
            if w < max(60, int(round(float(img_w) * 0.10))):
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
            if h < max(60, int(round(float(img_h) * 0.10))):
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


def _build_masks_and_components(img_gray: object) -> Dict[str, object]:
    result: Dict[str, object] = {"ok": False, "stats": {}}
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

    result.update(
        {
            "ok": True,
            "img_w": int(img_w),
            "img_h": int(img_h),
            "horiz": horiz,
            "vert": vert,
            "horiz_comps": _line_components(horiz, "h", img_w, img_h),
            "vert_comps": _line_components(vert, "v", img_w, img_h),
        }
    )
    return result


def _region_stats(
    bbox: List[float],
    horiz_comps: List[Dict[str, float]],
    vert_comps: List[Dict[str, float]],
    img_w: int,
    img_h: int,
) -> Dict[str, object]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    region_w = max(1.0, x1 - x0)
    region_h = max(1.0, y1 - y0)
    region_area_ratio = (region_w * region_h) / float(max(1, img_w * img_h))

    h_local = [
        comp
        for comp in horiz_comps
        if y0 <= float(comp["pos"]) <= y1
        and _overlap_1d(float(comp["start"]), float(comp["end"]), x0, x1) >= (0.65 * region_w)
    ]
    v_local = [
        comp
        for comp in vert_comps
        if x0 <= float(comp["pos"]) <= x1
        and _overlap_1d(float(comp["start"]), float(comp["end"]), y0, y1) >= (0.65 * region_h)
    ]

    h_positions = _cluster_positions([float(comp["pos"]) for comp in h_local], tol=max(3.0, float(region_h) * 0.01))
    v_positions = _cluster_positions([float(comp["pos"]) for comp in v_local], tol=max(3.0, float(region_w) * 0.01))

    h_gap_med, h_gap_cv = _gap_stats(h_positions)
    v_gap_med, v_gap_cv = _gap_stats(v_positions)
    h_count = len(h_positions)
    v_count = len(v_positions)
    crossings = h_count * v_count
    axis_balance = (float(min(h_count, v_count)) / float(max(h_count, v_count))) if max(h_count, v_count) > 0 else 0.0
    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "h_lines": int(h_count),
        "v_lines": int(v_count),
        "crossings": int(crossings),
        "axis_balance": float(axis_balance),
        "grid_area_ratio": float(region_area_ratio),
        "h_gap_med": float(h_gap_med),
        "v_gap_med": float(v_gap_med),
        "h_gap_cv": float(h_gap_cv),
        "v_gap_cv": float(v_gap_cv),
    }


def find_chart_like_regions(img_gray: object) -> Dict[str, object]:
    result: Dict[str, object] = {"regions": [], "stats": {}}
    built = _build_masks_and_components(img_gray)
    if not bool(built.get("ok")):
        return result

    img_w = int(built["img_w"])
    img_h = int(built["img_h"])
    horiz = built["horiz"]  # type: ignore[assignment]
    vert = built["vert"]  # type: ignore[assignment]
    horiz_comps = built["horiz_comps"]  # type: ignore[assignment]
    vert_comps = built["vert_comps"]  # type: ignore[assignment]

    try:
        combined = cv2.bitwise_or(horiz, vert)
        combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
    except Exception:
        return result

    regions: List[Dict[str, object]] = []
    for i in range(1, int(num)):
        x, y, w, h, area = stats[i]
        if area <= 0:
            continue
        if w < max(90, int(round(float(img_w) * 0.10))) or h < max(90, int(round(float(img_h) * 0.10))):
            continue
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        stats_local = _region_stats(bbox, horiz_comps, vert_comps, img_w, img_h)
        h_count = int(stats_local["h_lines"])
        v_count = int(stats_local["v_lines"])
        crossings = int(stats_local["crossings"])
        axis_balance = float(stats_local["axis_balance"])
        area_ratio = float(stats_local["grid_area_ratio"])
        h_gap_med = float(stats_local["h_gap_med"])
        v_gap_med = float(stats_local["v_gap_med"])
        h_gap_cv = float(stats_local["h_gap_cv"])
        v_gap_cv = float(stats_local["v_gap_cv"])

        dense_enough = h_count >= 6 and v_count >= 6 and crossings >= 36
        balanced_axes = axis_balance >= 0.6
        wide_enough = area_ratio >= 0.02
        regular_enough = h_gap_cv <= 0.5 and v_gap_cv <= 0.5
        fine_spacing = (
            h_gap_med > 0
            and v_gap_med > 0
            and h_gap_med <= max(75.0, float(h) * 0.14)
            and v_gap_med <= max(75.0, float(w) * 0.14)
        )

        if dense_enough and balanced_axes and wide_enough and regular_enough and fine_spacing:
            regions.append(stats_local)

    result["regions"] = regions
    result["stats"] = {
        "region_count": int(len(regions)),
        "h_components": int(len(horiz_comps)),
        "v_components": int(len(vert_comps)),
    }
    return result


def inspect_page_for_graph_grid(img_gray: object) -> Dict[str, object]:
    """
    Return a best-effort whole-page graph/grid classification for a page image.
    """
    result: Dict[str, object] = {"skip_table_work": False, "reason": "", "stats": {}}
    built = _build_masks_and_components(img_gray)
    if not bool(built.get("ok")):
        return result

    img_w = int(built["img_w"])
    img_h = int(built["img_h"])
    horiz_comps = built["horiz_comps"]  # type: ignore[assignment]
    vert_comps = built["vert_comps"]  # type: ignore[assignment]

    if horiz_comps and vert_comps:
        grid_x0 = min(float(comp["start"]) for comp in horiz_comps)
        grid_x1 = max(float(comp["end"]) for comp in horiz_comps)
        grid_y0 = min(float(comp["start"]) for comp in vert_comps)
        grid_y1 = max(float(comp["end"]) for comp in vert_comps)
    else:
        grid_x0 = grid_x1 = grid_y0 = grid_y1 = 0.0

    stats_page = _region_stats([grid_x0, grid_y0, grid_x1, grid_y1], horiz_comps, vert_comps, img_w, img_h)
    area_ratio = float(stats_page["grid_area_ratio"])
    h_count = int(stats_page["h_lines"])
    v_count = int(stats_page["v_lines"])
    crossings = int(stats_page["crossings"])
    axis_balance = float(stats_page["axis_balance"])
    h_gap_med = float(stats_page["h_gap_med"])
    v_gap_med = float(stats_page["v_gap_med"])
    h_gap_cv = float(stats_page["h_gap_cv"])
    v_gap_cv = float(stats_page["v_gap_cv"])

    result["stats"] = {
        "h_lines": h_count,
        "v_lines": v_count,
        "crossings": crossings,
        "axis_balance": axis_balance,
        "grid_area_ratio": area_ratio,
        "h_gap_med": h_gap_med,
        "v_gap_med": v_gap_med,
        "h_gap_cv": h_gap_cv,
        "v_gap_cv": v_gap_cv,
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


def mask_regions(img_gray: object, regions: List[Dict[str, object]], *, pad_px: int = 8) -> object:
    if not HAVE_CV2:
        return img_gray
    if not isinstance(img_gray, np.ndarray) or img_gray.ndim != 2 or img_gray.size == 0:
        return img_gray
    if not regions:
        return img_gray
    out = img_gray.copy()
    h, w = out.shape[:2]
    pad = max(0, int(pad_px))
    for region in regions:
        bbox = region.get("bbox") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
        x0 = max(0, min(w, x0 - pad))
        y0 = max(0, min(h, y0 - pad))
        x1 = max(0, min(w, x1 + pad))
        y1 = max(0, min(h, y1 + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] = 255
    return out
