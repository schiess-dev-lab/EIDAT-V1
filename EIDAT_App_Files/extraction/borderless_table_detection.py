"""
Borderless Table Detection - Synthesize cells from token grid alignment.

Detects table-like blocks from OCR token geometry and generates synthetic
cell bounding boxes so the existing bordered-table pipeline can be reused.
"""

from typing import Dict, List, Optional, Tuple

import os

from . import page_analyzer


def detect_borderless_tables(
    tokens: List[Dict],
    img_w: int,
    img_h: int,
    existing_tables: Optional[List[Dict]] = None,
    img_gray: Optional[object] = None,
    *,
    min_rows: int = 2,
    min_cols: int = 2,
    min_tokens: int = 6,
    min_line_coverage: float = 0.6,
    y_tolerance: float = 12.0,
    max_row_gap_ratio: float = 1.6,
    min_gap_px: float = 10.0
) -> List[Dict]:
    """
    Detect table-like token grids and synthesize cell bounding boxes.

    Args:
        tokens: OCR tokens already scaled to detection DPI.
        img_w, img_h: Page dimensions at detection DPI.
        existing_tables: Optional list of already-detected tables (bordered).
        img_gray: Optional grayscale image at detection DPI for line-based rows.
        min_rows: Minimum row count to accept a table.
        min_cols: Minimum column count to accept a table.
        min_tokens: Minimum token count to consider detection.
        min_line_coverage: Fraction of rows that must exhibit each column gap.
        y_tolerance: Line grouping tolerance (pixels at detection DPI).
        max_row_gap_ratio: Max row gap relative to median line height.
        min_gap_px: Minimum gap width (pixels) between columns.
    """
    if not tokens or len(tokens) < min_tokens:
        return []

    clean_tokens = [t for t in tokens if str(t.get("text", "")).strip()]
    if len(clean_tokens) < min_tokens:
        return []

    if existing_tables:
        clean_tokens = page_analyzer.filter_table_tokens(
            clean_tokens, existing_tables, overlap_threshold=0.4
        )
        if len(clean_tokens) < min_tokens:
            return []

    lines = page_analyzer.group_tokens_into_lines(clean_tokens, y_tolerance=y_tolerance)
    if not lines:
        return []

    line_infos = []
    for idx, line in enumerate(lines):
        bounds = _line_bounds(line)
        if not bounds:
            continue
        segments, gaps, median_h = _segment_line(line)
        if not segments:
            continue
        line_infos.append({
            "idx": idx,
            "line": line,
            "bounds": bounds,
            "segments": segments,
            "gaps": gaps,
            "median_h": median_h,
            "is_tabular": len(segments) >= min_cols,
        })

    if not line_infos:
        return []

    line_infos.sort(key=lambda info: info["bounds"]["y0"])
    blocks = _group_lines_into_blocks(line_infos, max_row_gap_ratio=max_row_gap_ratio)

    tables: List[Dict] = []
    for block in blocks:
        table = _build_table_from_block(
            block,
            img_w,
            img_h,
            img_gray=img_gray,
            min_rows=min_rows,
            min_cols=min_cols,
            min_line_coverage=min_line_coverage,
            min_gap_px=min_gap_px,
            max_row_gap_ratio=max_row_gap_ratio,
        )
        if table:
            tables.append(table)

    if existing_tables and tables:
        tables = _filter_overlapping_tables(tables, existing_tables)
    return tables


def _segment_line(line: List[Dict]) -> tuple[list[list[Dict]], list[Dict], float]:
    tokens = sorted(line, key=lambda t: float(t.get("x0", 0)))
    if not tokens:
        return [], [], 0.0

    heights = [
        max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
        for t in tokens
    ]
    median_h = _median(heights, default=12.0)

    if len(tokens) < 2:
        return [tokens], [], median_h

    gap_values = []
    for i in range(1, len(tokens)):
        gap = float(tokens[i].get("x0", 0)) - float(tokens[i - 1].get("x1", 0))
        if gap > 0:
            gap_values.append(gap)

    if not gap_values:
        return [tokens], [], median_h

    if len(gap_values) <= 1:
        threshold = max(median_h * 1.6, 24.0)
    else:
        median_gap = _median(gap_values, default=0.0)
        threshold = max(median_h * 1.2, median_gap * 2.0)

    split_indices = []
    for i in range(1, len(tokens)):
        gap = float(tokens[i].get("x0", 0)) - float(tokens[i - 1].get("x1", 0))
        if gap <= 0 or gap < threshold:
            continue
        left_h = _median(
            [
                max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
                for t in tokens[:i]
            ],
            default=median_h,
        )
        right_h = _median(
            [
                max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
                for t in tokens[i:]
            ],
            default=median_h,
        )
        if 0.75 <= (left_h / right_h) <= 1.33:
            split_indices.append(i)

    if not split_indices:
        return [tokens], [], median_h

    segments: List[List[Dict]] = []
    gaps: List[Dict] = []
    start = 0
    for idx in split_indices:
        if idx > start:
            segments.append(tokens[start:idx])
            left = tokens[idx - 1]
            right = tokens[idx]
            left_x1 = float(left.get("x1", 0))
            right_x0 = float(right.get("x0", 0))
            gaps.append({
                "mid": (left_x1 + right_x0) / 2.0,
                "left_x1": left_x1,
                "right_x0": right_x0,
            })
        start = idx
    if start < len(tokens):
        segments.append(tokens[start:])

    return segments, gaps, median_h


def _group_lines_into_blocks(
    line_infos: List[Dict],
    *,
    max_row_gap_ratio: float = 1.6
) -> List[List[Dict]]:
    blocks: List[List[Dict]] = []
    current: List[Dict] = []
    heights: List[float] = []

    for info in line_infos:
        if not current:
            current = [info]
            heights = [float(info["bounds"].get("h", 0.0))]
            continue

        prev = current[-1]
        gap = float(info["bounds"]["y0"]) - float(prev["bounds"]["y1"])
        median_h = _median(heights, default=float(info["bounds"].get("h", 0.0)))
        max_gap = max(12.0, median_h * max_row_gap_ratio)

        if gap <= max_gap:
            current.append(info)
            heights.append(float(info["bounds"].get("h", 0.0)))
        else:
            blocks.append(current)
            current = [info]
            heights = [float(info["bounds"].get("h", 0.0))]

    if current:
        blocks.append(current)

    return blocks


def _build_table_from_block(
    block: List[Dict],
    img_w: int,
    img_h: int,
    *,
    img_gray: Optional[object] = None,
    min_rows: int,
    min_cols: int,
    min_line_coverage: float,
    min_gap_px: float,
    max_row_gap_ratio: float
) -> Optional[Dict]:
    tabular_lines = [line for line in block if line.get("is_tabular")]
    candidate_lines = tabular_lines if len(tabular_lines) >= min_rows else block
    if len(candidate_lines) < min_rows:
        return None

    tabular_tokens = [t for line in candidate_lines for t in line.get("line", [])]
    if not tabular_tokens:
        return None

    heights = [
        max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
        for t in tabular_tokens
    ]
    median_h = _median(heights, default=12.0)
    min_x0 = min(float(t.get("x0", 0)) for t in tabular_tokens)
    min_y0 = min(float(t.get("y0", 0)) for t in tabular_tokens)
    max_x1 = max(float(t.get("x1", 0)) for t in tabular_tokens)
    max_y1 = max(float(t.get("y1", 0)) for t in tabular_tokens)

    gap_obs = []
    for row_idx, info in enumerate(tabular_lines):
        for gap in info.get("gaps", []):
            gap_obs.append({
                "mid": gap.get("mid", 0.0),
                "left_x1": gap.get("left_x1", 0.0),
                "right_x0": gap.get("right_x0", 0.0),
                "row": row_idx,
            })

    tolerance = max(8.0, median_h * 0.9)
    clusters = _cluster_gap_ranges(gap_obs, tolerance) if gap_obs else []
    required_lines = max(min_rows, int(round(len(tabular_lines) * min_line_coverage))) if tabular_lines else min_rows
    min_gap = max(min_gap_px, median_h * 0.4)
    pad = max(2.0, median_h * 0.1)

    boundaries: List[float] = []
    for cluster in clusters:
        rows = {obs["row"] for obs in cluster}
        if len(rows) < required_lines:
            continue
        left_max = max(float(obs["left_x1"]) for obs in cluster)
        right_min = min(float(obs["right_x0"]) for obs in cluster)
        if right_min - left_max < min_gap:
            continue
        left = left_max + pad
        right = right_min - pad
        if right <= left:
            boundary = (left_max + right_min) / 2.0
        else:
            boundary = (left + right) / 2.0
        boundaries.append(boundary)

    use_alignment = False
    alignment_row_lines: Optional[List[Dict]] = None
    if len(boundaries) + 1 < min_cols:
        boundaries, alignment_row_lines = _infer_alignment_boundaries(
            block,
            median_h=median_h,
            min_cols=min_cols,
            min_rows=min_rows,
            min_line_coverage=min_line_coverage,
        )
        if not boundaries:
            return None
        use_alignment = True

    boundaries = sorted(boundaries)
    boundaries = _merge_positions(boundaries, tolerance=tolerance)

    pad_box = max(4.0, median_h * 0.25)
    x0 = max(0.0, min_x0 - pad_box)
    x1 = min(float(img_w), max_x1 + pad_box)
    y0 = max(0.0, min_y0 - pad_box)
    y1 = min(float(img_h), max_y1 + pad_box)

    if use_alignment and alignment_row_lines:
        row_lines = sorted(alignment_row_lines, key=lambda info: info["bounds"]["y0"])
        tabular_tokens = [t for line in row_lines for t in line.get("line", [])]
        if not tabular_tokens or len(row_lines) < min_rows:
            return None
        min_x0 = min(float(t.get("x0", 0)) for t in tabular_tokens)
        min_y0 = min(float(t.get("y0", 0)) for t in tabular_tokens)
        max_x1 = max(float(t.get("x1", 0)) for t in tabular_tokens)
        max_y1 = max(float(t.get("y1", 0)) for t in tabular_tokens)
    else:
        tabular_y0 = min(float(info["bounds"]["y0"]) for info in tabular_lines)
        tabular_y1 = max(float(info["bounds"]["y1"]) for info in tabular_lines)

        table_w = max(1.0, x1 - x0)
        candidates = []
        for info in block:
            bounds = info["bounds"]
            overlap = max(0.0, min(bounds["x1"], x1) - max(bounds["x0"], x0))
            if (overlap / table_w) >= 0.45:
                candidates.append(info)

        candidates.sort(key=lambda info: info["bounds"]["y0"])
        base = [
            info for info in candidates
            if info["bounds"]["y1"] >= tabular_y0 and info["bounds"]["y0"] <= tabular_y1
        ]
        if not base:
            return None

        row_lines = list(base)
        max_gap = max(12.0, median_h * max_row_gap_ratio)
        base_indices = {id(info) for info in base}
        last_idx = max(i for i, info in enumerate(candidates) if id(info) in base_indices)
        prev = candidates[last_idx]
        for info in candidates[last_idx + 1:]:
            gap = float(info["bounds"]["y0"]) - float(prev["bounds"]["y1"])
            if gap <= max_gap:
                row_lines.append(info)
                prev = info
            else:
                break

        row_lines.sort(key=lambda info: info["bounds"]["y0"])
        if len(row_lines) < min_rows:
            return None

    row_min_y0 = min(float(info["bounds"]["y0"]) for info in row_lines)
    row_max_y1 = max(float(info["bounds"]["y1"]) for info in row_lines)
    y0 = max(0.0, row_min_y0 - pad_box)
    y1 = min(float(img_h), row_max_y1 + pad_box)

    if boundaries and row_lines:
        boundaries = _augment_boundaries_with_aligned_gaps(
            boundaries,
            row_lines,
            median_h=median_h,
            min_line_coverage=min_line_coverage,
            min_gap_px=min_gap,
        )

    if boundaries:
        valleys = _detect_vertical_whitespace_valleys(
            img_gray, (x0, y0, x1, y1), median_h=median_h
        )
        if valleys:
            boundaries = _snap_positions(boundaries, valleys, tolerance=tolerance)
            boundaries = _merge_positions(boundaries, tolerance=tolerance)
        boundaries = _refine_boundaries_from_rows(
            boundaries,
            row_lines,
            min_gap=min_gap,
            median_h=median_h,
        )
        boundaries = _filter_positions_in_bbox(boundaries, x0, x1, min_gap=min_gap)
        if (len(boundaries) + 1) < min_cols:
            return None

    row_edges = _row_edges_from_horizontal_rules(
        img_gray, (x0, y0, x1, y1), min_rows=min_rows
    )
    if not row_edges:
        row_edges = [y0]
        for i in range(len(row_lines) - 1):
            prev = row_lines[i]["bounds"]
            nxt = row_lines[i + 1]["bounds"]
            boundary = (float(prev["y1"]) + float(nxt["y0"])) / 2.0
            if boundary <= row_edges[-1]:
                boundary = row_edges[-1] + 1.0
            row_edges.append(boundary)
        row_edges.append(y1)

    col_edges = [x0] + boundaries + [x1]
    if (len(col_edges) - 1) < min_cols or (len(row_edges) - 1) < min_rows:
        return None

    cells = _cells_from_synthetic_grid(
        img_w,
        img_h,
        row_edges,
        col_edges,
        median_h=median_h,
        table_bbox=(x0, y0, x1, y1),
    )
    if not cells:
        cells = []
        try:
            tiny_min_h = int(float(os.environ.get("EIDAT_TABLE_DETECT_TINY_CELL_H_PX", "10")))
        except Exception:
            tiny_min_h = 10
        try:
            tiny_min_w = int(float(os.environ.get("EIDAT_TABLE_DETECT_TINY_CELL_W_PX", "0")))
        except Exception:
            tiny_min_w = 0
        tiny_min_h = max(0, int(tiny_min_h))
        tiny_min_w = max(0, int(tiny_min_w))
        for r in range(len(row_edges) - 1):
            for c in range(len(col_edges) - 1):
                cx0 = col_edges[c]
                cy0 = row_edges[r]
                cx1 = col_edges[c + 1]
                cy1 = row_edges[r + 1]
                if (cx1 - cx0) < 2.0 or (cy1 - cy0) < 2.0:
                    continue
                if tiny_min_h and (cy1 - cy0) < float(tiny_min_h):
                    continue
                if tiny_min_w and (cx1 - cx0) < float(tiny_min_w):
                    continue
                cells.append({
                    "bbox_px": [
                        int(round(cx0)),
                        int(round(cy0)),
                        int(round(cx1)),
                        int(round(cy1)),
                    ],
                    "borderless": True,
                })

    if len(cells) < (min_rows * min_cols):
        return None

    return {
        "bbox_px": [
            int(round(x0)),
            int(round(y0)),
            int(round(x1)),
            int(round(y1)),
        ],
        "cells": cells,
        "num_cells": len(cells),
        "borderless": True,
    }


def _cluster_gap_observations(gaps: List[Dict], tolerance: float) -> List[List[Dict]]:
    if not gaps:
        return []
    gaps_sorted = sorted(gaps, key=lambda g: float(g.get("mid", 0.0)))
    clusters: List[List[Dict]] = []
    current = [gaps_sorted[0]]
    for obs in gaps_sorted[1:]:
        if (float(obs.get("mid", 0.0)) - float(current[-1].get("mid", 0.0))) <= tolerance:
            current.append(obs)
        else:
            clusters.append(current)
            current = [obs]
    if current:
        clusters.append(current)
    return clusters


def _cluster_gap_ranges(gaps: List[Dict], tolerance: float) -> List[List[Dict]]:
    """
    Cluster gap ranges by overlap instead of midpoint distance.

    Each gap provides a [left_x1, right_x0] range that should contain
    the true column boundary. Overlapping ranges are grouped together.
    """
    if not gaps:
        return []

    def _range(g: Dict) -> tuple[float, float]:
        left = float(g.get("left_x1", g.get("mid", 0.0)))
        right = float(g.get("right_x0", g.get("mid", 0.0)))
        if right < left:
            left, right = right, left
        return left, right

    gaps_sorted = sorted(gaps, key=lambda g: _range(g)[0])
    clusters: List[List[Dict]] = []
    current: List[Dict] = [gaps_sorted[0]]
    curr_left, curr_right = _range(gaps_sorted[0])

    for obs in gaps_sorted[1:]:
        left, right = _range(obs)
        # Allow slight mismatch using tolerance.
        if left <= curr_right + tolerance and right >= curr_left - tolerance:
            current.append(obs)
            curr_left = max(curr_left, left)
            curr_right = min(curr_right, right)
        else:
            clusters.append(current)
            current = [obs]
            curr_left, curr_right = left, right

    if current:
        clusters.append(current)

    return clusters


def _infer_alignment_boundaries(
    block: List[Dict],
    *,
    median_h: float,
    min_cols: int,
    min_rows: int,
    min_line_coverage: float,
) -> tuple[List[float], Optional[List[Dict]]]:
    line_candidates = [info for info in block if info.get("line")]
    if len(line_candidates) < min_rows:
        return [], None

    tolerance = max(8.0, float(median_h) * 0.9)
    positions: List[tuple[float, int]] = []
    for idx, info in enumerate(line_candidates):
        line = info.get("line") or []
        starts = _line_column_starts(line, median_h=median_h)
        for x in starts:
            positions.append((float(x), idx))

    clusters = _cluster_positions_with_rows(positions, tolerance=tolerance)
    if not clusters:
        return [], None

    required_lines = max(min_rows, int(round(len(line_candidates) * min_line_coverage)))
    clusters = [c for c in clusters if len(c["rows"]) >= required_lines]
    centers = sorted([c["center"] for c in clusters])
    centers = _merge_positions(centers, tolerance=tolerance)
    if len(centers) < min_cols:
        return [], None

    row_lines: List[Dict] = []
    for idx, info in enumerate(line_candidates):
        line = info.get("line") or []
        starts = _line_column_starts(line, median_h=median_h)
        matched = 0
        for x in starts:
            if any(abs(float(x) - center) <= tolerance for center in centers):
                matched += 1
        if matched >= 2:
            row_lines.append(info)

    if len(row_lines) < min_rows:
        return [], None

    boundaries = _boundaries_from_alignment_centers(
        centers,
        median_h=median_h,
    )
    if len(boundaries) + 1 < min_cols:
        boundaries = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
    return boundaries, row_lines


def _line_column_starts(line: List[Dict], *, median_h: float) -> List[float]:
    tokens = sorted(line, key=lambda t: float(t.get("x0", 0)))
    if not tokens:
        return []

    starts: List[float] = [float(tokens[0].get("x0", 0.0))]

    gaps = []
    for i in range(1, len(tokens)):
        gap = float(tokens[i].get("x0", 0)) - float(tokens[i - 1].get("x1", 0))
        if gap > 0:
            gaps.append(gap)

    if gaps:
        gaps_sorted = sorted(gaps)
        median_gap = gaps_sorted[len(gaps_sorted) // 2]
    else:
        median_gap = 0.0

    gap_threshold = max(10.0, float(median_h) * 0.6, median_gap * 1.6)
    for i in range(1, len(tokens)):
        gap = float(tokens[i].get("x0", 0)) - float(tokens[i - 1].get("x1", 0))
        if gap >= gap_threshold:
            starts.append(float(tokens[i].get("x0", 0.0)))

    # Always include numeric-like tokens as potential column starts.
    for tok in tokens:
        text = str(tok.get("text", "")).strip()
        if not text:
            continue
        if _is_numeric_like(text):
            starts.append(float(tok.get("x0", 0.0)))

    return starts


def _boundaries_from_alignment_centers(
    centers: List[float],
    *,
    median_h: float,
) -> List[float]:
    if not centers or len(centers) < 2:
        return []
    pad = max(1.0, float(median_h) * 0.2)
    min_sep = max(6.0, float(median_h) * 0.6)

    edges = sorted(centers)
    boundaries: List[float] = []
    for i in range(1, len(edges)):
        left = edges[i]
        prev = edges[i - 1]
        if (left - prev) < min_sep:
            boundaries.append((left + prev) / 2.0)
        else:
            boundaries.append(left - pad)

    return boundaries


def _cluster_positions_with_rows(
    positions: List[tuple[float, int]],
    *,
    tolerance: float,
) -> List[Dict]:
    if not positions:
        return []
    positions_sorted = sorted(positions, key=lambda p: p[0])
    clusters: List[Dict] = []
    for x, row_idx in positions_sorted:
        placed = False
        for cluster in clusters:
            if abs(x - cluster["center"]) <= tolerance:
                cluster["points"].append(x)
                cluster["rows"].add(row_idx)
                cluster["center"] = sum(cluster["points"]) / float(len(cluster["points"]))
                placed = True
                break
        if not placed:
            clusters.append({"center": float(x), "points": [float(x)], "rows": {row_idx}})
    return clusters


def _is_numeric_like(text: str) -> bool:
    txt = str(text or "").strip()
    if not txt:
        return False
    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1].strip()
    if txt.startswith(("+", "-")):
        txt = txt[1:].strip()
    if txt.endswith("%"):
        txt = txt[:-1].strip()
    if not txt:
        return False
    if txt.count(".") > 1:
        return False
    return all(ch.isdigit() or ch == "." for ch in txt)


def _merge_positions(positions: List[float], *, tolerance: float) -> List[float]:
    if not positions:
        return []
    positions = sorted(positions)
    merged = [positions[0]]
    for pos in positions[1:]:
        if pos - merged[-1] <= tolerance:
            merged[-1] = (merged[-1] + pos) / 2.0
        else:
            merged.append(pos)
    return merged


def _filter_overlapping_tables(tables: List[Dict], existing: List[Dict]) -> List[Dict]:
    if not tables:
        return []

    def _overlaps(a: Dict, b: Dict) -> bool:
        ax0, ay0, ax1, ay1 = (float(v) for v in a.get("bbox_px", [0, 0, 0, 0]))
        bx0, by0, bx1, by1 = (float(v) for v in b.get("bbox_px", [0, 0, 0, 0]))
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return False
        inter = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
        iou = inter / max(1.0, (area_a + area_b - inter))
        overlap = inter / max(1.0, min(area_a, area_b))
        return iou >= 0.2 or overlap >= 0.35

    pruned: List[Dict] = []
    candidates = sorted(tables, key=lambda t: t.get("num_cells", 0), reverse=True)
    for table in candidates:
        if any(_overlaps(table, other) for other in existing):
            continue
        if any(_overlaps(table, other) for other in pruned):
            continue
        pruned.append(table)

    return sorted(pruned, key=lambda t: float(t.get("bbox_px", [0, 0, 0, 0])[1]))


def _row_edges_from_horizontal_rules(
    img_gray: Optional[object],
    bbox: Tuple[float, float, float, float],
    *,
    min_rows: int,
) -> Optional[List[float]]:
    lines = _detect_horizontal_rules(img_gray, bbox)
    if not lines:
        return None
    x0, y0, x1, y1 = bbox
    height = max(1.0, float(y1) - float(y0))
    margin = max(6.0, height * 0.01)
    internal = [y for y in lines if (y0 + margin) < y < (y1 - margin)]
    if len(internal) + 1 < min_rows:
        return None
    edges = [float(y0)] + sorted(internal) + [float(y1)]
    # Ensure strict monotonicity.
    cleaned: List[float] = []
    for y in edges:
        if not cleaned:
            cleaned.append(y)
            continue
        if y <= cleaned[-1]:
            y = cleaned[-1] + 1.0
        cleaned.append(y)
    return cleaned


def _detect_horizontal_rules(
    img_gray: Optional[object],
    bbox: Tuple[float, float, float, float],
    *,
    min_len_ratio: float = 0.45,
    max_thickness_ratio: float = 0.02,
    merge_tol: float = 3.0,
) -> List[float]:
    if img_gray is None:
        return []
    try:
        import cv2  # type: ignore
    except Exception:
        return []

    try:
        h_img, w_img = img_gray.shape[:2]
    except Exception:
        return []

    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0 = max(0, min(x0, w_img - 1))
    x1 = max(0, min(x1, w_img))
    y0 = max(0, min(y0, h_img - 1))
    y1 = max(0, min(y1, h_img))
    if x1 <= x0 or y1 <= y0:
        return []

    crop = img_gray[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return []

    h, w = crop.shape[:2]
    if h < 10 or w < 30:
        return []

    try:
        _, bin_inv = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bin_inv = cv2.morphologyEx(
            bin_inv,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        h_kernel_w = max(int(w * 0.35), 40)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
        horiz = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel)
    except Exception:
        return []

    min_len = max(1, int(w * min_len_ratio))
    max_thickness = max(3, int(min(h, w) * max_thickness_ratio))

    ys: List[float] = []
    try:
        num, labels, stats, _centroids = cv2.connectedComponentsWithStats(horiz, connectivity=8)
        for i in range(1, num):
            x, y, ww, hh, _area = stats[i]
            if ww < min_len or hh > max_thickness:
                continue
            ys.append(float(y) + float(hh) / 2.0)
    except Exception:
        return []

    if not ys:
        return []

    ys.sort()
    merged: List[float] = []
    for y in ys:
        if not merged or abs(y - merged[-1]) > merge_tol:
            merged.append(y)
        else:
            merged[-1] = (merged[-1] + y) / 2.0

    return [float(y0) + y for y in merged]


def _detect_vertical_whitespace_valleys(
    img_gray: Optional[object],
    bbox: Tuple[float, float, float, float],
    *,
    median_h: float,
) -> List[float]:
    if img_gray is None:
        return []
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return []

    try:
        h_img, w_img = img_gray.shape[:2]
    except Exception:
        return []

    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0 = max(0, min(x0, w_img - 1))
    x1 = max(0, min(x1, w_img))
    y0 = max(0, min(y0, h_img - 1))
    y1 = max(0, min(y1, h_img))
    if x1 <= x0 or y1 <= y0:
        return []

    crop = img_gray[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return []

    h, w = crop.shape[:2]
    if h < 10 or w < 30:
        return []

    try:
        _, bin_inv = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        return []

    ink = (bin_inv > 0).sum(axis=0).astype(float)
    if ink.size < 10:
        return []

    win = max(3, int(round(max(3.0, float(median_h) * 0.5))))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    smooth = np.convolve(ink, kernel, mode="same")
    smooth_list = smooth.tolist()

    median_ink = _median(smooth_list, default=0.0)
    p20 = _percentile(smooth_list, 0.2)
    threshold = max(1.0, min(median_ink * 0.35, p20 * 1.2))

    min_width = max(2, int(round(float(median_h) * 0.5)))
    valleys: List[float] = []
    start = None
    for i, val in enumerate(smooth):
        if val <= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                if (end - start + 1) >= min_width:
                    segment = smooth[start:end + 1]
                    idx = int(start + int(np.argmin(segment)))
                    valleys.append(float(x0 + idx))
                start = None
    if start is not None:
        end = len(smooth) - 1
        if (end - start + 1) >= min_width:
            segment = smooth[start:end + 1]
            idx = int(start + int(np.argmin(segment)))
            valleys.append(float(x0 + idx))

    return valleys


def _snap_positions(
    positions: List[float],
    anchors: List[float],
    *,
    tolerance: float,
) -> List[float]:
    if not positions or not anchors:
        return positions
    anchors_sorted = sorted(anchors)
    snapped: List[float] = []
    for pos in positions:
        best = None
        best_dist = None
        for a in anchors_sorted:
            dist = abs(pos - a)
            if best_dist is None or dist < best_dist:
                best = a
                best_dist = dist
        if best is not None and best_dist is not None and best_dist <= tolerance:
            snapped.append(float(best))
        else:
            snapped.append(float(pos))
    return snapped


def _refine_boundaries_from_rows(
    boundaries: List[float],
    row_lines: List[Dict],
    *,
    min_gap: float,
    median_h: float,
) -> List[float]:
    if not boundaries or not row_lines:
        return boundaries
    pad = max(1.0, float(median_h) * 0.15)
    refined: List[float] = []

    for boundary in boundaries:
        midpoints: List[float] = []
        left_edges: List[float] = []
        right_edges: List[float] = []
        for info in row_lines:
            line = info.get("line") or []
            if not line:
                continue
            left_x1 = None
            right_x0 = None
            for tok in line:
                x0 = float(tok.get("x0", 0.0))
                x1 = float(tok.get("x1", 0.0))
                cx = (x0 + x1) / 2.0
                if cx <= boundary:
                    if left_x1 is None or x1 > left_x1:
                        left_x1 = x1
                else:
                    if right_x0 is None or x0 < right_x0:
                        right_x0 = x0

            if left_x1 is None or right_x0 is None:
                continue
            left = left_x1 + pad
            right = right_x0 - pad
            if right - left < max(2.0, float(min_gap) * 0.4):
                continue
            midpoints.append((left + right) / 2.0)
            left_edges.append(left)
            right_edges.append(right)

        if midpoints:
            spread_left = _spread(left_edges)
            spread_right = _spread(right_edges)
            align_thresh = max(3.0, float(median_h) * 0.35)
            if right_edges and spread_right <= align_thresh and spread_right <= (spread_left * 0.7):
                target = _median(right_edges, default=boundary)
            elif left_edges and spread_left <= align_thresh and spread_left <= (spread_right * 0.7):
                target = _median(left_edges, default=boundary)
            else:
                target = _median(midpoints, default=boundary)

            if left_edges:
                min_left = min(left_edges)
                if target < min_left:
                    target = min_left
            if right_edges:
                max_right = max(right_edges)
                if target > max_right:
                    target = max_right

            refined.append(target)
        else:
            refined.append(boundary)

    return refined


def _augment_boundaries_with_aligned_gaps(
    boundaries: List[float],
    row_lines: List[Dict],
    *,
    median_h: float,
    min_line_coverage: float,
    min_gap_px: float,
) -> List[float]:
    if not boundaries or not row_lines:
        return boundaries

    small_gap = max(4.0, float(median_h) * 0.25)
    tolerance = max(6.0, float(median_h) * 0.6)
    candidates: List[tuple[float, int, float, float]] = []

    for row_idx, info in enumerate(row_lines):
        line = info.get("line") or []
        if len(line) < 2:
            continue
        tokens = sorted(line, key=lambda t: float(t.get("x0", 0)))
        gaps_row: List[float] = []
        for i in range(1, len(tokens)):
            gap = float(tokens[i].get("x0", 0)) - float(tokens[i - 1].get("x1", 0))
            if gap > 0:
                gaps_row.append(gap)
        median_row_gap = _median(gaps_row, default=0.0)
        row_gap_threshold = max(small_gap, float(median_row_gap) * 1.8)
        for i in range(1, len(tokens)):
            left = tokens[i - 1]
            right = tokens[i]
            gap = float(right.get("x0", 0)) - float(left.get("x1", 0))
            if gap < row_gap_threshold:
                continue
            candidates.append(
                (
                    float(right.get("x0", 0)),
                    row_idx,
                    gap,
                    float(left.get("x1", 0)),
                )
            )

    if not candidates:
        return boundaries

    candidates.sort(key=lambda item: item[0])
    clusters: List[Dict] = []
    for x, row_idx, gap, left_x1 in candidates:
        placed = False
        for cluster in clusters:
            if abs(x - cluster["center"]) <= tolerance:
                cluster["points"].append(x)
                cluster["rows"].add(row_idx)
                cluster["gaps"].append(gap)
                cluster["lefts"].append(left_x1)
                cluster["center"] = sum(cluster["points"]) / float(len(cluster["points"]))
                placed = True
                break
        if not placed:
            clusters.append({
                "center": float(x),
                "points": [float(x)],
                "rows": {int(row_idx)},
                "gaps": [float(gap)],
                "lefts": [float(left_x1)],
            })

    required_lines = max(2, int(round(len(row_lines) * min_line_coverage)))
    new_bounds: List[float] = []
    pad = max(1.0, float(median_h) * 0.12)
    gap_floor = max(small_gap, float(min_gap_px) * 0.5)
    merge_dist = max(8.0, float(median_h) * 0.8)

    for cluster in clusters:
        if len(cluster["rows"]) < required_lines:
            continue
        median_gap = _median(cluster["gaps"], default=0.0)
        if median_gap < gap_floor:
            continue

        right_start = _median(cluster["points"], default=cluster["center"])
        left_end = _median(cluster["lefts"], default=right_start - median_gap)
        boundary = right_start - pad
        if boundary <= left_end + 1.0:
            boundary = (right_start + left_end) / 2.0

        if any(abs(boundary - b) <= tolerance for b in boundaries):
            continue

        new_bounds.append(boundary)

    if not new_bounds:
        return boundaries

    combined = list(boundaries)
    for boundary in new_bounds:
        nearest_idx = None
        nearest_dist = None
        for idx, existing in enumerate(combined):
            dist = abs(boundary - existing)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        if nearest_dist is not None and nearest_dist <= merge_dist and nearest_idx is not None:
            combined[nearest_idx] = (combined[nearest_idx] + boundary) / 2.0
        else:
            combined.append(boundary)

    combined = _merge_positions(
        sorted(combined),
        tolerance=max(2.0, float(min_gap_px) * 0.45, float(median_h) * 0.6),
    )
    return combined


def _filter_positions_in_bbox(
    positions: List[float],
    x0: float,
    x1: float,
    *,
    min_gap: float,
) -> List[float]:
    if not positions:
        return []
    margin = max(2.0, float(min_gap) * 0.5)
    inner_min = float(x0) + margin
    inner_max = float(x1) - margin
    filtered = [p for p in positions if inner_min < p < inner_max]
    if not filtered:
        return []
    filtered = _merge_positions(sorted(filtered), tolerance=max(2.0, float(min_gap) * 0.35))
    cleaned: List[float] = []
    for pos in filtered:
        if not cleaned:
            cleaned.append(pos)
            continue
        if pos <= cleaned[-1]:
            pos = cleaned[-1] + 1.0
        cleaned.append(pos)
    return cleaned


def _cells_from_synthetic_grid(
    img_w: int,
    img_h: int,
    row_edges: List[float],
    col_edges: List[float],
    *,
    median_h: float,
    table_bbox: Tuple[float, float, float, float],
) -> Optional[List[Dict]]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    if img_w <= 0 or img_h <= 0:
        return None
    if len(row_edges) < 2 or len(col_edges) < 2:
        return None

    try:
        from . import table_detection
    except Exception:
        return None

    thickness = max(2, int(round(float(median_h) * 0.15)))

    x0, y0, x1, y1 = table_bbox
    crop_w = int(round(max(1.0, x1 - x0)))
    crop_h = int(round(max(1.0, y1 - y0)))
    if crop_w < 10 or crop_h < 10:
        return None

    canvas = np.full((crop_h, crop_w), 255, dtype=np.uint8)

    def _draw_h(y: float, x_start: float, x_end: float) -> None:
        yy = int(round(y - y0))
        x0r = int(round(min(x_start, x_end) - x0))
        x1r = int(round(max(x_start, x_end) - x0))
        cv2.line(canvas, (x0r, yy), (x1r, yy), 0, thickness=thickness)

    def _draw_v(x: float, y_start: float, y_end: float) -> None:
        xx = int(round(x - x0))
        y0r = int(round(min(y_start, y_end) - y0))
        y1r = int(round(max(y_start, y_end) - y0))
        cv2.line(canvas, (xx, y0r), (xx, y1r), 0, thickness=thickness)

    # Outer border (crop space)
    _draw_h(y0, x0, x1)
    _draw_h(y1, x0, x1)
    _draw_v(x0, y0, y1)
    _draw_v(x1, y0, y1)

    # Internal grid lines
    for y in row_edges[1:-1]:
        _draw_h(y, x0, x1)
    for x in col_edges[1:-1]:
        _draw_v(x, y0, y1)

    result = table_detection.detect_tables(canvas, verbose=False)
    tables = result.get("tables") or []
    if not tables:
        return None

    # Pick the largest table in the crop (should be the synthetic grid).
    best = max(tables, key=lambda t: t.get("num_cells", 0))
    cells = best.get("cells") or []
    if not cells:
        return None

    out = []
    try:
        tiny_min_h = int(float(os.environ.get("EIDAT_TABLE_DETECT_TINY_CELL_H_PX", "10")))
    except Exception:
        tiny_min_h = 10
    try:
        tiny_min_w = int(float(os.environ.get("EIDAT_TABLE_DETECT_TINY_CELL_W_PX", "0")))
    except Exception:
        tiny_min_w = 0
    tiny_min_h = max(0, int(tiny_min_h))
    tiny_min_w = max(0, int(tiny_min_w))
    for cell in cells:
        bbox = cell.get("bbox_px") or []
        if len(bbox) != 4:
            continue
        bbox = [
            int(bbox[0] + x0),
            int(bbox[1] + y0),
            int(bbox[2] + x0),
            int(bbox[3] + y0),
        ]
        if tiny_min_h and (bbox[3] - bbox[1]) < int(tiny_min_h):
            continue
        if tiny_min_w and (bbox[2] - bbox[0]) < int(tiny_min_w):
            continue
        out.append({
            "bbox_px": list(bbox),
            "borderless": True,
        })
    return out


def _bbox_overlap_score(a_bbox: List[float], b_bbox: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a_bbox)
    bx0, by0, bx1, by1 = (float(v) for v in b_bbox)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    iou = inter / max(1.0, (area_a + area_b - inter))
    overlap = inter / max(1.0, min(area_a, area_b))
    return max(iou, overlap)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if q <= 0:
        return float(values[0])
    if q >= 1:
        return float(values[-1])
    idx = (len(values) - 1) * float(q)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _median(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _spread(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) < 3:
        return float(values[-1] - values[0])
    lo = values[int(len(values) * 0.1)]
    hi = values[int(len(values) * 0.9)]
    return float(hi - lo)


def _line_bounds(line: List[Dict]) -> Optional[Dict]:
    if not line:
        return None
    x0 = min(t.get("x0", 0) for t in line)
    y0 = min(t.get("y0", 0) for t in line)
    x1 = max(t.get("x1", 0) for t in line)
    y1 = max(t.get("y1", 0) for t in line)
    h = max(1.0, float(y1) - float(y0))
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "h": h,
    }
