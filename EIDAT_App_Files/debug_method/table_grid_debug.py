#!/usr/bin/env python3
"""
Draw left-of-word separators, then detect table patterns to draw full cell borders.

Defaults to reading rendered pages under debug_method/debug_file and writing
overlays to debug_method/word_gap_debug.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_gray(path: Path):
    import cv2  # type: ignore

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _binarize(img_gray):
    import cv2  # type: ignore

    _, bin_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_inv = cv2.morphologyEx(
        bin_inv,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    return bin_inv


def _median(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _median_component_height(bin_inv) -> float:
    import cv2  # type: ignore

    heights: List[float] = []
    try:
        h_img, w_img = bin_inv.shape[:2]
        max_component_h = max(120.0, float(h_img) * 0.05)
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
        for i in range(1, num):
            _x, _y, _w, h, area = stats[i]
            if area < 10 or h < 3:
                continue
            if float(h) > max_component_h:
                continue
            heights.append(float(h))
    except Exception:
        return 0.0
    return _median(heights, default=0.0)


def _word_tokens(bin_inv, *, median_h: float, merge_kx: int) -> List[Dict]:
    import cv2  # type: ignore

    h, w = bin_inv.shape[:2]
    if h < 8 or w < 8:
        return []

    if merge_kx <= 0:
        merge_kx = int(round(max(3.0, min(45.0, median_h * 0.8))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(merge_kx), 1))
    merged = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    tokens: List[Dict] = []
    try:
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(merged, connectivity=8)
        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            if area < 12 or ww < 3 or hh < 3:
                continue
            if ww > w * 0.98 or hh > h * 0.9:
                continue
            x0 = float(x)
            y0 = float(y)
            x1 = float(x + ww)
            y1 = float(y + hh)
            tokens.append({
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "cx": (x0 + x1) / 2.0,
                "cy": (y0 + y1) / 2.0,
            })
    except Exception:
        return []
    return tokens


def _group_tokens_into_lines(tokens: List[Dict], *, y_tolerance: float) -> List[List[Dict]]:
    if not tokens:
        return []
    tokens_sorted = sorted(tokens, key=lambda t: (t.get("cy", t.get("y0", 0)), t.get("cx", t.get("x0", 0))))
    lines: List[List[Dict]] = []
    current = [tokens_sorted[0]]
    current_y = float(tokens_sorted[0].get("cy", tokens_sorted[0].get("y0", 0)))
    for tok in tokens_sorted[1:]:
        cy = float(tok.get("cy", tok.get("y0", 0)))
        if abs(cy - current_y) <= y_tolerance:
            current.append(tok)
        else:
            lines.append(sorted(current, key=lambda t: t.get("x0", 0)))
            current = [tok]
            current_y = cy
    if current:
        lines.append(sorted(current, key=lambda t: t.get("x0", 0)))
    return lines


def _line_bounds(line: List[Dict]) -> Dict:
    y0 = min(float(t.get("y0", 0)) for t in line)
    y1 = max(float(t.get("y1", 0)) for t in line)
    x0 = min(float(t.get("x0", 0)) for t in line)
    x1 = max(float(t.get("x1", 0)) for t in line)
    return {
        "y0": y0,
        "y1": y1,
        "x0": x0,
        "x1": x1,
        "tokens": line,
    }


def _merge_lines_into_rows(lines: List[List[Dict]], *, max_gap: float) -> List[Dict]:
    if not lines:
        return []
    line_boxes = sorted((_line_bounds(line) for line in lines), key=lambda b: b["y0"])
    rows: List[Dict] = []
    current = {
        "y0": float(line_boxes[0]["y0"]),
        "y1": float(line_boxes[0]["y1"]),
        "x0": float(line_boxes[0]["x0"]),
        "x1": float(line_boxes[0]["x1"]),
        "lines": [line_boxes[0]],
        "tokens": list(line_boxes[0]["tokens"]),
    }
    for box in line_boxes[1:]:
        gap = float(box["y0"]) - float(current["y1"])
        if gap <= max_gap:
            current["y0"] = min(float(current["y0"]), float(box["y0"]))
            current["y1"] = max(float(current["y1"]), float(box["y1"]))
            current["x0"] = min(float(current["x0"]), float(box["x0"]))
            current["x1"] = max(float(current["x1"]), float(box["x1"]))
            current["lines"].append(box)
            current["tokens"].extend(box["tokens"])
        else:
            rows.append(current)
            current = {
                "y0": float(box["y0"]),
                "y1": float(box["y1"]),
                "x0": float(box["x0"]),
                "x1": float(box["x1"]),
                "lines": [box],
                "tokens": list(box["tokens"]),
            }
    rows.append(current)
    return rows


def _cluster_positions(positions: List[float], *, tol: float) -> List[float]:
    if not positions:
        return []
    positions_sorted = sorted(float(p) for p in positions)
    clusters: List[List[float]] = [[positions_sorted[0]]]
    for pos in positions_sorted[1:]:
        if abs(pos - clusters[-1][-1]) <= tol:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    return [sum(cluster) / float(len(cluster)) for cluster in clusters]




def _detect_horizontal_rule_segments(
    bin_inv,
    *,
    min_len_ratio: float = 0.12,
    max_thickness_ratio: float = 0.03,
) -> List[Dict]:
    try:
        import cv2  # type: ignore
    except Exception:
        return []

    try:
        h, w = bin_inv.shape[:2]
    except Exception:
        return []

    if h < 10 or w < 30:
        return []

    try:
        h_kernel_w = max(int(w * 0.35), 40)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
        # Close first to bridge small breaks, then open to keep long horizontals.
        horiz = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        horiz = cv2.morphologyEx(horiz, cv2.MORPH_OPEN, h_kernel)
    except Exception:
        return []

    min_len = max(1, int(w * min_len_ratio))
    max_thickness = max(3, int(min(h, w) * max_thickness_ratio))

    segments: List[Dict] = []
    try:
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(horiz, connectivity=8)
        for i in range(1, num):
            x, y, ww, hh, _area = stats[i]
            if ww < min_len or hh > max_thickness:
                continue
            segments.append({
                "x0": float(x),
                "x1": float(x + ww),
                "y": float(y) + float(hh) / 2.0,
                "thickness": float(hh),
                "width": float(ww),
            })
    except Exception:
        return []

    return segments


def _hline_positions_for_table(
    hline_segments: List[Dict],
    *,
    left: float,
    right: float,
    row_y0: float,
    row_y1: float,
    median_h: float,
) -> List[float]:
    if not hline_segments:
        return []
    table_w = max(1.0, float(right - left))
    y_pad = max(2.0, float(median_h) * 0.6)
    y_min = float(row_y0) - y_pad
    y_max = float(row_y1) + y_pad
    min_span_ratio = 0.45

    candidates: List[float] = []
    for seg in hline_segments:
        seg_y = float(seg.get("y", 0.0))
        if seg_y < y_min or seg_y > y_max:
            continue
        x0 = float(seg.get("x0", 0.0))
        x1 = float(seg.get("x1", 0.0))
        overlap = min(x1, float(right)) - max(x0, float(left))
        if overlap <= 0:
            continue
        if (overlap / table_w) < min_span_ratio:
            continue
        candidates.append(seg_y)

    if not candidates:
        return []

    y_tol = max(2.0, float(median_h) * 0.18)
    return _cluster_positions(candidates, tol=y_tol)


def _snap_row_lines_to_hlines(
    row_lines: List[float],
    hline_positions: List[float],
    *,
    max_snap_px: float,
    min_sep: float,
) -> List[float]:
    if not row_lines or not hline_positions:
        return row_lines

    candidates = sorted(float(y) for y in hline_positions)
    snapped: List[float] = []
    for idx, pos in enumerate(row_lines):
        best = None
        best_diff = max_snap_px + 1.0
        for cand in candidates:
            diff = abs(float(cand) - float(pos))
            if diff <= max_snap_px and diff < best_diff:
                best_diff = diff
                best = float(cand)
        chosen = float(pos)
        if best is not None:
            prev = snapped[-1] if snapped else None
            next_original = row_lines[idx + 1] if idx + 1 < len(row_lines) else None
            if prev is not None and best <= float(prev) + float(min_sep):
                chosen = float(pos)
            elif next_original is not None and best >= float(next_original) - float(min_sep):
                chosen = float(pos)
            else:
                chosen = best
        snapped.append(chosen)
    return snapped


def _row_separators(rows: List[Dict], separators: List[Dict], *, x_tol: float) -> List[Dict]:
    row_data: List[Dict] = []
    for row in rows:
        y0 = float(row["y0"])
        y1 = float(row["y1"])
        seps = [
            float(sep.get("x", 0))
            for sep in separators
            if y0 <= (float(sep.get("y0", 0)) + float(sep.get("y1", 0))) * 0.5 <= y1
        ]
        sep_clusters = _cluster_positions(seps, tol=x_tol)
        row_data.append({
            "y0": y0,
            "y1": y1,
            "x0": float(row["x0"]),
            "x1": float(row["x1"]),
            "tokens": row.get("tokens", []),
            "seps": sorted(sep_clusters),
        })
    return row_data


def _build_table_info(
    table_rows_sorted: List[Dict],
    col_centers: List[float],
    separators: List[Dict],
    *,
    median_h: float,
    hline_segments: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    if not table_rows_sorted or not col_centers:
        return None
    pad_y = max(2.0, median_h * 0.15)
    pad_x = max(2.0, median_h * 0.2)

    def _merge_bands(bands: List[List[float]], *, merge_gap: float) -> List[List[float]]:
        if not bands:
            return []
        bands = sorted(bands, key=lambda b: b[0])
        merged: List[List[float]] = [bands[0]]
        for band in bands[1:]:
            if band[0] <= merged[-1][1] + merge_gap:
                merged[-1][0] = min(merged[-1][0], band[0])
                merged[-1][1] = max(merged[-1][1], band[1])
            else:
                merged.append(band)
        return merged

    def _sep_bounds_for_row(row: Dict) -> Optional[Dict]:
        y0 = float(row["y0"])
        y1 = float(row["y1"])
        sep_bounds = [
            (float(sep.get("y0", 0)), float(sep.get("y1", 0)))
            for sep in separators
            if y0 <= (float(sep.get("y0", 0)) + float(sep.get("y1", 0))) * 0.5 <= y1
        ]
        if not sep_bounds:
            return None
        return {
            "min_y0": min(s[0] for s in sep_bounds),
            "max_y1": max(s[1] for s in sep_bounds),
        }

    tokens = [tok for row in table_rows_sorted for tok in row.get("tokens", [])]
    if not tokens:
        return None
    max_x1 = max(float(tok.get("x1", 0)) for tok in tokens)
    right_margin = max(pad_x, median_h * 0.8, 24.0)
    right_boundary = max_x1 + right_margin
    min_right = col_centers[-1] + right_margin
    if right_boundary < min_right:
        right_boundary = min_right
    left_boundary = col_centers[0]

    row_lines: List[float] = []
    row_bands: List[List[float]] = []
    for row in table_rows_sorted:
        sep_bounds = _sep_bounds_for_row(row)
        if sep_bounds:
            band = [float(sep_bounds["min_y0"]), float(sep_bounds["max_y1"])]
        else:
            band = [float(row["y0"]) - pad_y, float(row["y1"]) + pad_y]
        row_bands.append(band)
    merge_gap = max(2.0, median_h * 0.4)
    merged_bands = _merge_bands(row_bands, merge_gap=merge_gap)

    # Fallback: if row bands collapse too much, rebuild from row token bounds.
    expected_min = max(2, int(round(len(table_rows_sorted) * 0.7)))
    if len(table_rows_sorted) >= 2 and len(merged_bands) < expected_min:
        alt_pad = max(1.0, median_h * 0.1)
        alt_bands = [
            [float(row["y0"]) - alt_pad, float(row["y1"]) + alt_pad]
            for row in table_rows_sorted
        ]
        alt_merged = _merge_bands(alt_bands, merge_gap=max(1.0, median_h * 0.15))
        if len(alt_merged) > len(merged_bands):
            merged_bands = alt_merged

    row_lines.append(merged_bands[0][0])
    for band in merged_bands:
        bottom_line = band[1]
        if bottom_line <= row_lines[-1]:
            bottom_line = row_lines[-1] + max(1.0, median_h * 0.1)
        row_lines.append(bottom_line)

    if hline_segments:
        hline_positions = _hline_positions_for_table(
            hline_segments,
            left=float(left_boundary),
            right=float(right_boundary),
            row_y0=float(row_lines[0]),
            row_y1=float(row_lines[-1]),
            median_h=median_h,
        )
        if hline_positions:
            row_lines = _snap_row_lines_to_hlines(
                row_lines,
                hline_positions,
                max_snap_px=200.0,
                min_sep=max(1.0, float(median_h) * 0.1),
            )

    return {
        "columns": col_centers + [right_boundary],
        "row_lines": sorted(row_lines),
        "left": left_boundary,
        "right": right_boundary,
        "top": min(row_lines),
        "bottom": max(row_lines),
        "rows": table_rows_sorted,
        "count": len(col_centers),
    }


def _detect_tables(
    rows: List[Dict],
    separators: List[Dict],
    *,
    median_h: float,
    x_tol: float,
    hline_segments: Optional[List[Dict]] = None,
) -> List[Dict]:
    row_data = _row_separators(rows, separators, x_tol=x_tol)
    candidate_rows = [r for r in row_data if len(r["seps"]) > 0]
    if len(candidate_rows) < 2:
        return []
    candidate_rows_sorted = sorted(candidate_rows, key=lambda r: r["y0"])
    row_heights = [float(r["y1"]) - float(r["y0"]) for r in candidate_rows_sorted]
    row_h_med = _median(row_heights, default=median_h)
    row_gaps = [
        max(0.0, float(cur["y0"]) - float(prev["y1"]))
        for prev, cur in zip(candidate_rows_sorted, candidate_rows_sorted[1:])
    ]
    gap_med = _median(row_gaps, default=0.0)
    max_row_gap = max(8.0, row_h_med * 1.6, median_h * 1.2)
    if gap_med > 0:
        max_row_gap = max(max_row_gap, gap_med * 1.2)
    groups: List[List[Dict]] = []
    current_group = [candidate_rows_sorted[0]]
    for row in candidate_rows_sorted[1:]:
        gap = float(row["y0"]) - float(current_group[-1]["y1"])
        if gap > max_row_gap:
            groups.append(current_group)
            current_group = [row]
        else:
            current_group.append(row)
    groups.append(current_group)

    tables: List[Dict] = []
    min_table_rows = 2
    for group in groups:
        if len(group) < min_table_rows:
            continue
        group_counts = [len(r["seps"]) for r in group]
        group_count = int(round(_median([float(c) for c in group_counts], default=0.0)))
        if group_count < 2:
            continue
        guidance_sep_count = max(len(r.get("seps", [])) for r in group)
        guidance_candidates = [r for r in group if len(r.get("seps", [])) == guidance_sep_count]
        guidance_row = min(guidance_candidates, key=lambda r: r["y0"])
        guidance_enabled = guidance_sep_count >= 2
        target_col_count = max(group_count, guidance_sep_count) if guidance_enabled else group_count
        min_row_matches = max(2, int(round(group_count * 0.7)))
        min_row_seps = max(1, group_count - 1)
        group_rows = [r for r in group if len(r["seps"]) >= min_row_seps]
        if len(group_rows) < min_table_rows:
            continue
        all_positions = [pos for r in group_rows for pos in r["seps"]]
        if guidance_enabled and guidance_row not in group_rows:
            all_positions.extend(guidance_row.get("seps", []))
        if not all_positions:
            continue
        col_clusters = _cluster_positions(all_positions, tol=x_tol)
        if not col_clusters:
            continue
        min_col_support = max(2, int(round(len(group_rows) * 0.5)))
        cluster_counts: List[Dict] = []
        for center in col_clusters:
            count = 0
            for row in group_rows:
                if any(abs(center - sep) <= x_tol for sep in row["seps"]):
                    count += 1
            if (
                guidance_enabled
                and count >= min_col_support
                and any(abs(center - sep) <= x_tol for sep in guidance_row.get("seps", []))
            ):
                count += min_col_support
            if count >= min_col_support:
                cluster_counts.append({"x": center, "count": count})
        if not cluster_counts:
            continue
        cluster_counts.sort(key=lambda c: (-c["count"], c["x"]))
        selected = cluster_counts[: max(2, min(target_col_count, len(cluster_counts)))]
        col_centers = sorted(c["x"] for c in selected)
        if len(col_centers) < 2:
            continue
        table_rows = []
        for row in group_rows:
            seps = row["seps"]
            matches = 0
            for col in col_centers:
                if any(abs(col - sep) <= x_tol for sep in seps):
                    matches += 1
            if matches >= min_row_matches:
                table_rows.append(row)
        if len(table_rows) < min_table_rows:
            continue
        table_rows_sorted = sorted(table_rows, key=lambda r: r["y0"])
        table_info = _build_table_info(
            table_rows_sorted,
            col_centers,
            separators,
            median_h=median_h,
            hline_segments=hline_segments,
        )
        if table_info:
            tables.append(table_info)

    tables_sorted = sorted(tables, key=lambda t: float(t.get("top", 0)))
    return tables_sorted


# Legacy method retained for reference (not used in current flow).
def _gap_separators(
    lines: List[List[Dict]],
    *,
    median_h: float,
    min_gap: float,
    min_overlap: float,
    pad: float,
) -> List[Dict]:
    separators: List[Dict] = []
    for line in lines:
        if len(line) < 2:
            continue
        tokens = sorted(line, key=lambda t: float(t.get("x0", 0)))
        for left, right in zip(tokens, tokens[1:]):
            gap = float(right.get("x0", 0)) - float(left.get("x1", 0))
            if gap < min_gap:
                continue
            y0_overlap = max(float(left.get("y0", 0)), float(right.get("y0", 0)))
            y1_overlap = min(float(left.get("y1", 0)), float(right.get("y1", 0)))
            if (y1_overlap - y0_overlap) < min_overlap:
                continue
            y0 = min(float(left.get("y0", 0)), float(right.get("y0", 0))) - pad
            y1 = max(float(left.get("y1", 0)), float(right.get("y1", 0))) + pad
            if y1 <= y0:
                continue
            x = (float(left.get("x1", 0)) + float(right.get("x0", 0))) / 2.0
            separators.append({
                "x": x,
                "y0": y0,
                "y1": y1,
                "gap": gap,
                "left": left,
                "right": right,
            })
    return separators


def _left_of_word_separators(
    lines: List[List[Dict]],
    *,
    min_gap: float,
    pad: float,
    offset_px: float,
    include_first: bool,
    min_token_h: float,
) -> List[Dict]:
    separators: List[Dict] = []
    relaxed_min_token_h = max(2.0, float(min_token_h) * 0.6)

    def _seps_for_line(tokens: List[Dict], *, token_h_threshold: float) -> List[Dict]:
        line_seps: List[Dict] = []
        if not tokens:
            return line_seps
        tokens_sorted = sorted(tokens, key=lambda t: float(t.get("x0", 0)))
        if include_first and tokens_sorted:
            first = tokens_sorted[0]
            h_first = float(first.get("y1", 0)) - float(first.get("y0", 0))
            if h_first >= token_h_threshold:
                gap = float(first.get("x0", 0))
                if gap >= min_gap:
                    y0 = float(first.get("y0", 0)) - pad
                    y1 = float(first.get("y1", 0)) + pad
                    if y1 > y0:
                        x = float(first.get("x0", 0)) - float(offset_px)
                        if x >= 0.0 and x < float(first.get("x0", 0)):
                            line_seps.append({
                                "x": x,
                                "y0": y0,
                                "y1": y1,
                                "y0_draw": float(first.get("y0", 0)),
                                "y1_draw": float(first.get("y1", 0)),
                                "gap": gap,
                                "left": None,
                                "right": first,
                            })
        if len(tokens_sorted) < 2:
            return line_seps
        for left, right in zip(tokens_sorted, tokens_sorted[1:]):
            h_right = float(right.get("y1", 0)) - float(right.get("y0", 0))
            if h_right < token_h_threshold:
                continue
            gap = float(right.get("x0", 0)) - float(left.get("x1", 0))
            if gap <= min_gap:
                continue
            y0 = min(float(left.get("y0", 0)), float(right.get("y0", 0))) - pad
            y1 = max(float(left.get("y1", 0)), float(right.get("y1", 0))) + pad
            if y1 <= y0:
                continue
            x = float(right.get("x0", 0)) - float(offset_px)
            if x <= float(left.get("x1", 0)) or x >= float(right.get("x0", 0)) or x < 0.0:
                continue
            line_seps.append({
                "x": x,
                "y0": y0,
                "y1": y1,
                "y0_draw": float(right.get("y0", 0)),
                "y1_draw": float(right.get("y1", 0)),
                "gap": gap,
                "left": left,
                "right": right,
            })
        return line_seps

    for line in lines:
        if not line:
            continue
        tokens = sorted(line, key=lambda t: float(t.get("x0", 0)))
        line_seps = _seps_for_line(tokens, token_h_threshold=min_token_h)
        if not line_seps:
            line_seps = _seps_for_line(tokens, token_h_threshold=relaxed_min_token_h)
        separators.extend(line_seps)
    return separators


def _draw_overlay(
    img_gray,
    *,
    tokens: List[Dict],
    separators: List[Dict],
    draw_separators: bool,
    line_thickness: int,
    bin_inv,
    tables: Optional[List[Dict]] = None,
    draw_tables: bool = True,
    hline_segments: Optional[List[Dict]] = None,
    draw_hlines: bool = True,
    sep_keep_x_tol: Optional[float] = None,
    draw_seps_in_tables: bool = True,
):
    import cv2  # type: ignore

    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w = img_gray.shape[:2]
    table_line_thickness = max(4, int(line_thickness) + 2)
    for tok in tokens:
        x0 = int(round(float(tok.get("x0", 0))))
        y0 = int(round(float(tok.get("y0", 0))))
        x1 = int(round(float(tok.get("x1", 0))))
        y1 = int(round(float(tok.get("y1", 0))))
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 220, 200), 1)
    if draw_separators:
        for sep in separators:
            x = int(round(float(sep.get("x", 0))))
            y0 = int(round(float(sep.get("y0_draw", sep.get("y0", 0)))))
            y1 = int(round(float(sep.get("y1_draw", sep.get("y1", 0)))))
            if x < 0 or x >= w:
                continue
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h - 1, y1))
            if y1 <= y0:
                continue
            if tables and not draw_seps_in_tables:
                y_mid = (y0 + y1) * 0.5
                if any(
                    float(t.get("left", 0)) <= float(x) <= float(t.get("right", 0))
                    and float(t.get("top", 0)) <= y_mid <= float(t.get("bottom", 0))
                    for t in tables
                ):
                    continue
            if tables and sep_keep_x_tol is not None and sep_keep_x_tol > 0:
                y_mid = (y0 + y1) * 0.5
                skip = False
                for table_info in tables:
                    left = float(table_info.get("left", 0))
                    right = float(table_info.get("right", 0))
                    top = float(table_info.get("top", 0))
                    bottom = float(table_info.get("bottom", 0))
                    if x < left or x > right or y_mid < top or y_mid > bottom:
                        continue
                    near_col = any(
                        abs(float(col_x) - float(x)) <= float(sep_keep_x_tol)
                        for col_x in table_info.get("columns", [])
                    )
                    if not near_col:
                        skip = True
                    break
                if skip:
                    continue
            x0 = max(0, x - int(line_thickness // 2))
            x1 = min(w, x + int((line_thickness + 1) // 2))
            if x1 > x0:
                if (bin_inv[y0:y1, x0:x1] > 0).any():
                    continue
            cv2.line(img, (x, y0), (x, y1), (40, 40, 220), int(line_thickness))
    if draw_tables and tables:
        for table_info in tables:
            left = int(round(float(table_info.get("left", 0))))
            right = int(round(float(table_info.get("right", 0))))
            top = int(round(float(table_info.get("top", 0))))
            bottom = int(round(float(table_info.get("bottom", 0))))
            left = max(0, min(w - 1, left))
            right = max(0, min(w - 1, right))
            top = max(0, min(h - 1, top))
            bottom = max(0, min(h - 1, bottom))
            if right > left and bottom > top:
                for x in table_info.get("columns", []):
                    xi = int(round(float(x)))
                    if xi < left or xi > right:
                        continue
                    cv2.line(img, (xi, top), (xi, bottom), (255, 0, 0), table_line_thickness)
                for y in table_info.get("row_lines", []):
                    yi = int(round(float(y)))
                    if yi < top or yi > bottom:
                        continue
                    cv2.line(img, (left, yi), (right, yi), (255, 0, 0), table_line_thickness)
    if draw_hlines and hline_segments:
        for seg in hline_segments:
            x0 = int(round(float(seg.get("x0", 0))))
            x1 = int(round(float(seg.get("x1", 0))))
            y = int(round(float(seg.get("y", 0))))
            if x1 <= x0:
                continue
            y = max(0, min(h - 1, y))
            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w - 1, x1))
            cv2.line(img, (x0, y), (x1, y), (0, 220, 255), max(1, line_thickness))
    return img


def _collect_images(root: Path, pattern: str) -> List[Path]:
    matches: List[Path] = []
    for path in root.rglob(pattern):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            matches.append(path)
    return sorted(matches)


def run_for_image(
    path: Path,
    *,
    out_dir: Path,
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
    draw_separators: bool = True,
) -> Dict:
    import cv2  # type: ignore

    img_gray = _load_gray(path)
    bin_inv = _binarize(img_gray)
    median_h = _median_component_height(bin_inv)
    if median_h <= 0:
        median_h = max(10.0, img_gray.shape[0] * 0.04)

    tokens = _word_tokens(bin_inv, median_h=median_h, merge_kx=merge_kx)
    hline_segments = _detect_horizontal_rule_segments(bin_inv)
    y_tol = max(6.0, median_h * 0.35)
    lines = _group_tokens_into_lines(tokens, y_tolerance=y_tol)
    row_gap = max(2.0, median_h * 0.6)
    rows = _merge_lines_into_rows(lines, max_gap=row_gap)

    if min_gap is not None and min_gap > 0:
        gap_px = float(min_gap)
    elif min_gap_ratio and min_gap_ratio > 0:
        gap_px = max(1.0, float(median_h) * float(min_gap_ratio))
    else:
        gap_px = 24.0
    min_token_h = float(min_token_h_px) if min_token_h_px > 0 else max(2.0, median_h * float(min_token_h_ratio))
    pad = max(1.0, median_h * float(line_pad_factor)) + 12.0
    separators = _left_of_word_separators(
        lines,
        min_gap=gap_px,
        pad=pad,
        offset_px=offset_px,
        include_first=True,
        min_token_h=min_token_h,
    )
    x_tol = max(6.0, median_h * 0.45)
    tables = _detect_tables(
        rows,
        separators,
        median_h=median_h,
        x_tol=x_tol,
        hline_segments=hline_segments,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = _draw_overlay(
        img_gray,
        tokens=tokens,
        separators=separators,
        draw_separators=draw_separators,
        line_thickness=line_thickness,
        bin_inv=bin_inv,
        tables=tables,
        draw_tables=draw_tables,
        hline_segments=hline_segments,
        draw_hlines=draw_hlines,
        sep_keep_x_tol=x_tol,
        draw_seps_in_tables=draw_seps_in_tables,
    )
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    first_table = tables[0] if tables else None
    payload = {
        "image": str(path),
        "method": "left_of_word",
        "median_h": float(median_h),
        "merge_kx": int(merge_kx) if merge_kx > 0 else int(round(max(3.0, min(45.0, median_h * 0.8)))),
        "min_gap": float(gap_px),
        "min_gap_ratio": float(min_gap_ratio),
        "offset_px": float(offset_px),
        "line_thickness": int(line_thickness),
        "line_pad_factor": float(line_pad_factor),
        "min_token_h": float(min_token_h),
        "draw_tables": bool(draw_tables),
        "draw_hlines": bool(draw_hlines),
        "draw_separators": bool(draw_separators),
        "lines": len(lines),
        "rows": len(rows),
        "tokens": len(tokens),
        "separators": [
            {"x": float(s["x"]), "y0": float(s["y0"]), "y1": float(s["y1"]), "gap": float(s["gap"])}
            for s in separators
        ],
        "table": {
            "detected": bool(first_table),
            "columns": [] if not first_table else [float(x) for x in first_table.get("columns", [])],
            "row_lines": [] if not first_table else [float(y) for y in first_table.get("row_lines", [])],
        },
        "tables": [
            {
                "columns": [float(x) for x in table.get("columns", [])],
                "row_lines": [float(y) for y in table.get("row_lines", [])],
            }
            for table in tables
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    debug_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Draw left-of-word separators and table grids from gap patterns.")
    parser.add_argument("--input-root", default=str(debug_dir / "debug_file"))
    parser.add_argument("--output-root", default=str(debug_dir / "word_gap_debug"))
    parser.add_argument("--pattern", default="page_*.png")
    parser.add_argument("--merge-kx", type=int, default=0, help="Horizontal merge kernel width in px (0=auto).")
    parser.add_argument("--min-gap", type=float, default=50.0, help="Minimum gap between words (px).")
    parser.add_argument("--min-gap-ratio", type=float, default=0.0, help="Minimum gap vs median text height.")
    parser.add_argument("--gap-threshold", type=float, default=0.0, help="Alias for --min-gap (px).")
    parser.add_argument("--left-offset", type=float, default=24.0, help="Line offset to the left of a word (px).")
    parser.add_argument("--line-thickness", type=int, default=3, help="Line thickness in px.")
    parser.add_argument("--line-pad", type=float, default=0.25, help="Height padding factor vs median text height.")
    parser.add_argument("--min-token-h", type=float, default=0.0, help="Minimum token height in px (0=auto).")
    parser.add_argument("--min-token-h-ratio", type=float, default=0.85, help="Auto height threshold vs median text height.")
    parser.add_argument("--no-drawn-seps", action="store_true", help="Disable drawing separator overlays.")
    parser.add_argument("--no-drawn-borders", action="store_true", help="Disable drawing table borders on overlays.")
    parser.add_argument("--no-drawn-hlines", action="store_true", help="Disable drawing horizontal rule overlays.")
    parser.add_argument(
        "--draw-seps-in-tables",
        action="store_true",
        help="Draw left-of-word separators inside detected table regions.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise RuntimeError(f"Input root not found: {input_root}")

    images = _collect_images(input_root, str(args.pattern))
    if args.limit and args.limit > 0:
        images = images[: int(args.limit)]

    if not images:
        print("No images found.")
        return 1

    summaries: List[Dict] = []
    for path in images:
        rel = path.relative_to(input_root)
        out_dir = output_root / rel.parent / rel.stem
        try:
            gap_override = float(args.gap_threshold) if args.gap_threshold else float(args.min_gap)
            summary = run_for_image(
                path,
                out_dir=out_dir,
                merge_kx=int(args.merge_kx),
                min_gap=gap_override if gap_override else None,
                min_gap_ratio=float(args.min_gap_ratio),
                offset_px=float(args.left_offset),
                line_thickness=int(args.line_thickness),
                line_pad_factor=float(args.line_pad),
                min_token_h_px=float(args.min_token_h),
                min_token_h_ratio=float(args.min_token_h_ratio),
                draw_tables=not bool(args.no_drawn_borders),
                draw_hlines=not bool(args.no_drawn_hlines),
                draw_seps_in_tables=bool(args.draw_seps_in_tables),
                draw_separators=not bool(args.no_drawn_seps),
            )
            summaries.append(summary)
            print(f"Processed {rel}")
        except Exception as exc:
            print(f"Failed {rel}: {exc}")

    index_path = output_root / "index.json"
    index_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(summaries)} summaries to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
