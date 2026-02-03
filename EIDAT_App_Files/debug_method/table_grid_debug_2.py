#!/usr/bin/env python3
"""
Draw left-of-word separators, then detect table patterns by snapping tick rows
into a simple grid. Outputs the same overlays as table_grid_debug.py.
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
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
        for i in range(1, num):
            _x, _y, _w, h, area = stats[i]
            if area < 10 or h < 3:
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


def _ticks_from_separators(separators: List[Dict]) -> List[Dict]:
    ticks: List[Dict] = []
    for sep in separators:
        x = float(sep.get("x", 0.0))
        y0 = float(sep.get("y0_draw", sep.get("y0", 0.0)))
        y1 = float(sep.get("y1_draw", sep.get("y1", 0.0)))
        if y1 <= y0:
            continue
        ticks.append({
            "x": x,
            "y0": y0,
            "y1": y1,
            "yc": (y0 + y1) * 0.5,
        })
    return ticks


def _cluster_ticks_into_rows(ticks: List[Dict], *, y_tol: float) -> List[Dict]:
    if not ticks:
        return []
    ticks_sorted = sorted(ticks, key=lambda t: t["yc"])
    rows: List[Dict] = []
    current = {
        "y0": float(ticks_sorted[0]["y0"]),
        "y1": float(ticks_sorted[0]["y1"]),
        "ticks": [ticks_sorted[0]],
        "yc": float(ticks_sorted[0]["yc"]),
    }
    for tick in ticks_sorted[1:]:
        if abs(float(tick["yc"]) - float(current["yc"])) <= y_tol:
            current["ticks"].append(tick)
            current["y0"] = min(float(current["y0"]), float(tick["y0"]))
            current["y1"] = max(float(current["y1"]), float(tick["y1"]))
            current["yc"] = (float(current["y0"]) + float(current["y1"])) * 0.5
        else:
            rows.append(current)
            current = {
                "y0": float(tick["y0"]),
                "y1": float(tick["y1"]),
                "ticks": [tick],
                "yc": float(tick["yc"]),
            }
    rows.append(current)
    return rows


def _merge_close_tick_rows(rows: List[Dict], *, merge_gap: float) -> List[Dict]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: r["yc"])
    merged: List[Dict] = []
    current = rows_sorted[0].copy()
    current["ticks"] = list(current.get("ticks", []))
    for row in rows_sorted[1:]:
        if abs(float(row["yc"]) - float(current["yc"])) <= merge_gap:
            current["ticks"].extend(row.get("ticks", []))
            current["y0"] = min(float(current["y0"]), float(row["y0"]))
            current["y1"] = max(float(current["y1"]), float(row["y1"]))
            current["yc"] = (float(current["y0"]) + float(current["y1"])) * 0.5
        else:
            merged.append(current)
            current = row.copy()
            current["ticks"] = list(current.get("ticks", []))
    merged.append(current)
    return merged


def _rows_from_ticks(separators: List[Dict], *, y_tol: float, merge_gap: float, x_tol: float) -> List[Dict]:
    ticks = _ticks_from_separators(separators)
    if not ticks:
        return []
    rows = _cluster_ticks_into_rows(ticks, y_tol=y_tol)
    rows = _merge_close_tick_rows(rows, merge_gap=merge_gap)
    for row in rows:
        xs = _cluster_positions([float(t["x"]) for t in row.get("ticks", [])], tol=x_tol)
        row["xs"] = sorted(xs)
        row["count"] = len(row["xs"])
    return sorted(rows, key=lambda r: r["y0"])


def _group_tick_rows(rows: List[Dict], *, max_gap: float) -> List[List[Dict]]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: r["y0"])
    groups: List[List[Dict]] = []
    current = [rows_sorted[0]]
    for row in rows_sorted[1:]:
        gap = float(row["y0"]) - float(current[-1]["y1"])
        if gap > max_gap:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
    groups.append(current)
    return groups


def _merge_columns_to_count(columns: List[float], target_count: int) -> List[float]:
    cols = list(columns)
    if target_count < 1:
        return cols
    while len(cols) > target_count and len(cols) >= 2:
        min_gap = None
        min_idx = 0
        for idx in range(len(cols) - 1):
            gap = float(cols[idx + 1]) - float(cols[idx])
            if min_gap is None or gap < min_gap:
                min_gap = gap
                min_idx = idx
        merged = (float(cols[min_idx]) + float(cols[min_idx + 1])) * 0.5
        cols = cols[:min_idx] + [merged] + cols[min_idx + 2 :]
    return cols




def _candidate_column_sets(rows: List[Dict], *, x_tol: float) -> List[List[float]]:
    all_xs = [x for row in rows for x in row.get("xs", [])]
    if not all_xs:
        return []
    base_cols = sorted(_cluster_positions(all_xs, tol=x_tol))
    if not base_cols:
        return []
    row_counts = [len(row.get("xs", [])) for row in rows if row.get("xs")]
    median_count = int(round(_median([float(c) for c in row_counts], default=float(len(base_cols)))))
    max_count = max(row_counts) if row_counts else len(base_cols)
    target_counts = {len(base_cols), max(1, median_count)}
    if max_count - median_count <= 2:
        target_counts.add(max_count)
        if max_count - 1 >= 1:
            target_counts.add(max_count - 1)
    if median_count - 1 >= 1:
        target_counts.add(median_count - 1)
    target_counts = {c for c in target_counts if c >= 1}

    variants: List[List[float]] = []
    for target in sorted(target_counts):
        if len(base_cols) < target:
            continue
        cols = list(base_cols)
        if len(cols) > target:
            cols = _merge_columns_to_count(cols, target)
        if cols:
            variants.append(sorted(cols))

    unique: List[List[float]] = []
    for cols in variants:
        if not any(
            len(cols) == len(existing)
            and all(abs(float(a) - float(b)) <= x_tol * 0.5 for a, b in zip(cols, existing))
            for existing in unique
        ):
            unique.append(cols)
    return unique


def _row_lines_from_tick_rows(rows: List[Dict], *, median_h: float, connect_gap: float) -> List[float]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda r: r["y0"])
    row_lines: List[float] = [float(rows_sorted[0]["y0"]), float(rows_sorted[0]["y1"])]
    prev_bottom = float(rows_sorted[0]["y1"])
    for row in rows_sorted[1:]:
        top = float(row["y0"])
        bottom = float(row["y1"])
        if top - prev_bottom > connect_gap:
            row_lines.append(top)
        row_lines.append(bottom)
        prev_bottom = bottom
    row_lines = _cluster_positions(row_lines, tol=max(1.0, median_h * 0.15))
    return sorted(row_lines)


def _alignment_score(rows: List[Dict], columns: List[float], *, x_tol: float) -> float:
    if not rows or not columns:
        return 0.0
    col_count = len(columns)
    scores: List[float] = []
    for row in rows:
        xs = row.get("xs", [])
        if not xs:
            scores.append(0.0)
            continue
        matches = 0
        for col in columns:
            if any(abs(float(col) - float(x)) <= x_tol for x in xs):
                matches += 1
        scores.append(float(matches) / float(col_count))
    return sum(scores) / float(len(scores))


def _cell_fill_ratio(tokens: List[Dict], columns: List[float], row_lines: List[float]) -> Dict:
    if len(columns) < 2 or len(row_lines) < 2:
        return {"filled": 0, "total": 0, "ratio": 0.0}
    total = 0
    filled = 0
    for r_idx in range(len(row_lines) - 1):
        y0 = float(row_lines[r_idx])
        y1 = float(row_lines[r_idx + 1])
        if y1 <= y0:
            continue
        for c_idx in range(len(columns) - 1):
            x0 = float(columns[c_idx])
            x1 = float(columns[c_idx + 1])
            if x1 <= x0:
                continue
            total += 1
            if any(
                float(tok.get("x1", 0)) > x0
                and float(tok.get("x0", 0)) < x1
                and float(tok.get("y1", 0)) > y0
                and float(tok.get("y0", 0)) < y1
                for tok in tokens
            ):
                filled += 1
    ratio = float(filled) / float(total) if total > 0 else 0.0
    return {"filled": filled, "total": total, "ratio": ratio}


def _compute_right_boundary(
    columns: List[float],
    row_lines: List[float],
    tokens: List[Dict],
    *,
    median_h: float,
) -> float:
    if not columns:
        return 0.0
    top = float(min(row_lines)) if row_lines else 0.0
    bottom = float(max(row_lines)) if row_lines else 0.0
    left = float(columns[0])
    x_pad = max(2.0, median_h * 0.25)
    y_pad = max(2.0, median_h * 0.25)
    candidates = [
        tok
        for tok in tokens
        if (top - y_pad) <= float(tok.get("cy", tok.get("y0", 0))) <= (bottom + y_pad)
        and float(tok.get("x0", 0)) >= (left - x_pad)
    ]
    max_x1 = max((float(tok.get("x1", 0)) for tok in candidates), default=columns[-1])
    col_widths = [
        float(columns[i + 1]) - float(columns[i])
        for i in range(len(columns) - 1)
        if float(columns[i + 1]) > float(columns[i])
    ]
    median_w = _median(col_widths, default=median_h * 4.0)
    right = max(max_x1 + x_pad, float(columns[-1]) + max(10.0, median_w, median_h * 2.0))
    return right


def _score_candidate(*, fill_ratio: float, alignment: float, row_count: int, col_count: int) -> float:
    base = 0.7 * float(fill_ratio) + 0.3 * float(alignment)
    size_bonus = 0.55 + 0.05 * min(6, int(row_count)) + 0.05 * min(6, int(col_count))
    return min(1.0, base * size_bonus)


def _build_table_candidate(
    rows: List[Dict],
    columns: List[float],
    tokens: List[Dict],
    *,
    median_h: float,
    x_tol: float,
    hline_segments: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    if not rows or not columns:
        return None
    row_lines = _row_lines_from_tick_rows(rows, median_h=median_h, connect_gap=max(2.0, median_h * 0.9))
    if len(row_lines) < 2:
        return None
    left = float(columns[0])
    right = _compute_right_boundary(columns, row_lines, tokens, median_h=median_h)
    if right <= left:
        return None
    if hline_segments:
        hline_positions = _hline_positions_for_table(
            hline_segments,
            left=left,
            right=right,
            row_y0=float(row_lines[0]),
            row_y1=float(row_lines[-1]),
            median_h=median_h,
        )
        if hline_positions:
            row_lines = _snap_row_lines_to_hlines(
                row_lines,
                hline_positions,
                max_snap_px=max(8.0, median_h * 1.2),
                min_sep=max(1.0, median_h * 0.15),
            )
    columns_final = list(columns) + [right]
    if len(columns_final) < 2:
        return None
    table_tokens = [
        tok
        for tok in tokens
        if float(tok.get("y1", 0)) >= min(row_lines)
        and float(tok.get("y0", 0)) <= max(row_lines)
        and float(tok.get("x1", 0)) >= left
        and float(tok.get("x0", 0)) <= right
    ]
    fill = _cell_fill_ratio(table_tokens, columns_final, row_lines)
    alignment = _alignment_score(rows, columns, x_tol=x_tol)
    row_count = max(0, len(row_lines) - 1)
    col_count = max(0, len(columns_final) - 1)
    confidence = _score_candidate(
        fill_ratio=fill["ratio"],
        alignment=alignment,
        row_count=row_count,
        col_count=col_count,
    )
    return {
        "columns": columns_final,
        "row_lines": sorted(row_lines),
        "left": left,
        "right": right,
        "top": min(row_lines),
        "bottom": max(row_lines),
        "confidence": confidence,
        "fill_ratio": fill["ratio"],
        "alignment": alignment,
        "rows": rows,
        "count": len(columns),
    }


def _detect_tables(
    rows: List[Dict],
    separators: List[Dict],
    *,
    median_h: float,
    x_tol: float,
    hline_segments: Optional[List[Dict]] = None,
) -> List[Dict]:
    page_tokens = [tok for row in rows for tok in row.get("tokens", [])]
    y_tol = max(4.0, median_h * 0.3)
    merge_gap = max(6.0, median_h * 0.55)
    tick_rows = _rows_from_ticks(separators, y_tol=y_tol, merge_gap=merge_gap, x_tol=x_tol)
    if len(tick_rows) < 2:
        return []
    row_gaps = [
        max(0.0, float(cur["y0"]) - float(prev["y1"]))
        for prev, cur in zip(tick_rows, tick_rows[1:])
    ]
    gap_med = _median(row_gaps, default=0.0)
    max_row_gap = max(8.0, median_h * 1.8, gap_med * 1.3 if gap_med > 0 else 0.0)
    groups = _group_tick_rows(tick_rows, max_gap=max_row_gap)

    candidates: List[Dict] = []
    min_rows = 2
    for group in groups:
        if len(group) < min_rows:
            continue
        for start in range(len(group)):
            for end in range(start + min_rows - 1, len(group)):
                window = group[start : end + 1]
                if len(window) < min_rows:
                    continue
                columns_variants = _candidate_column_sets(window, x_tol=x_tol)
                for cols in columns_variants:
                    if not cols:
                        continue
                    candidate = _build_table_candidate(
                        window,
                        cols,
                        page_tokens,
                        median_h=median_h,
                        x_tol=x_tol,
                        hline_segments=hline_segments,
                    )
                    if candidate:
                        candidates.append(candidate)

    if not candidates:
        return []

    candidates.sort(key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
    selected: List[Dict] = []
    for cand in candidates:
        overlap = False
        for kept in selected:
            x_overlap = min(float(cand["right"]), float(kept["right"])) - max(float(cand["left"]), float(kept["left"]))
            y_overlap = min(float(cand["bottom"]), float(kept["bottom"])) - max(float(cand["top"]), float(kept["top"]))
            if x_overlap <= 0 or y_overlap <= 0:
                continue
            x_ratio = x_overlap / max(1.0, min(float(cand["right"]) - float(cand["left"]), float(kept["right"]) - float(kept["left"])))
            y_ratio = y_overlap / max(1.0, min(float(cand["bottom"]) - float(cand["top"]), float(kept["bottom"]) - float(kept["top"])))
            if x_ratio > 0.6 and y_ratio > 0.6:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
    return sorted(selected, key=lambda t: float(t.get("top", 0)))


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
        "method": "tick_grid_simple_2",
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
                "confidence": float(table.get("confidence", 0.0)),
                "fill_ratio": float(table.get("fill_ratio", 0.0)),
                "alignment": float(table.get("alignment", 0.0)),
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
