"""
Chart Detection - Heuristics to identify chart regions and labels.

Uses axis-token alignment + optional image ink density to find chart bounds.
"""

from typing import List, Dict, Optional, Tuple
import math
import re

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def detect_charts(
    img_gray: Optional["np.ndarray"],
    tokens: List[Dict],
    tables: List[Dict],
    img_w: int,
    img_h: int,
    flow_data: Optional[Dict] = None
) -> List[Dict]:
    """Detect chart regions using axis tokens + optional image ink."""
    if not tokens or img_w <= 0 or img_h <= 0:
        return []

    numeric_tokens = _collect_numeric_tokens(tokens)
    y_clusters = _axis_clusters(numeric_tokens, axis="y", img_w=img_w, img_h=img_h)
    x_clusters = _axis_clusters(numeric_tokens, axis="x", img_w=img_w, img_h=img_h)

    candidates: List[Dict] = []
    for y_axis in y_clusters:
        for x_axis in x_clusters:
            cand = _pair_axes_to_candidate(y_axis, x_axis, img_w, img_h)
            if cand is None:
                continue
            cand["method"] = "axis"
            candidates.append(cand)

    if not candidates and y_clusters:
        for y_axis in y_clusters:
            cand = _axis_only_candidate(y_axis, img_w, img_h, tables)
            if cand is None:
                continue
            cand["method"] = "axis_y"
            candidates.append(cand)

    title_candidates = _title_based_candidates(img_gray, tokens, tables, img_w, img_h, flow_data)
    for cand in title_candidates:
        cand["method"] = "title"
        candidates.append(cand)

    refined = []
    for cand in candidates:
        bbox = cand["bbox_px"]
        if cand.get("method") in ("axis", "axis_y"):
            bbox = _expand_bbox_with_tokens(
                bbox,
                tokens,
                img_w,
                img_h,
                pad_x=_axis_pad_x(cand, img_w),
                pad_top=_axis_pad_top(cand, img_h),
                pad_bottom=_axis_pad_bottom(cand, img_h),
            )
            bbox = _expand_bbox_with_title_line(
                bbox,
                tokens,
                img_w,
                img_h,
                axis_span_y=float(cand.get("axis_span_y") or 0.0),
            )
        else:
            bbox = _expand_bbox_with_tokens(bbox, tokens, img_w, img_h)
        bbox = _tighten_bbox_with_ink(bbox, img_gray, img_w, img_h)
        bbox = _clip_bbox_away_from_tables(bbox, tables, img_w, img_h)
        bbox = _clamp_bbox(bbox, img_w, img_h)
        if not _valid_bbox(bbox, img_w, img_h):
            continue
        if _overlaps_tables(bbox, tables, threshold=0.0):
            continue
        cand["bbox_px"] = bbox
        refined.append(cand)

    return _dedupe_candidates(refined)


def _collect_numeric_tokens(tokens: List[Dict]) -> List[Dict]:
    numeric = []
    for idx, t in enumerate(tokens):
        txt = str(t.get("text") or "").strip()
        if not txt:
            continue
        cleaned = re.sub(r"[^0-9.+-]", "", txt)
        if not cleaned or cleaned in ("-", "+", "."):
            continue
        if not re.match(r"^[-+]?\d*\.?\d+$", cleaned):
            continue
        try:
            x0 = float(t.get("x0", 0.0))
            y0 = float(t.get("y0", 0.0))
            x1 = float(t.get("x1", 0.0))
            y1 = float(t.get("y1", 0.0))
        except Exception:
            continue
        numeric.append({
            "idx": idx,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "cx": (x0 + x1) / 2.0,
            "cy": (y0 + y1) / 2.0,
            "text": txt
        })
    return numeric


def _axis_clusters(numeric_tokens: List[Dict], axis: str, img_w: int, img_h: int) -> List[Dict]:
    if not numeric_tokens:
        return []
    clusters = []
    if axis == "y":
        align_tol = max(30.0, 0.015 * float(img_w))
        by_bucket: Dict[int, List[Dict]] = {}
        for t in numeric_tokens:
            bucket = int(t["cx"] / align_tol)
            by_bucket.setdefault(bucket, []).append(t)
        for items in by_bucket.values():
            if len(items) < 3:
                continue
            items_sorted = sorted(items, key=lambda t: t["cy"])
            segments = _split_by_large_gaps(items_sorted, axis="y")
            for seg in segments:
                if len(seg) < 3:
                    continue
                y_gaps = []
                for i in range(1, len(seg)):
                    gap = seg[i]["cy"] - seg[i - 1]["cy"]
                    if gap > 20:
                        y_gaps.append(gap)
                if len(y_gaps) < 2:
                    continue
                avg_gap = sum(y_gaps) / len(y_gaps)
                if avg_gap < max(30.0, 0.02 * float(img_h)):
                    continue
                if not all(abs(g - avg_gap) < 0.35 * avg_gap for g in y_gaps):
                    continue
                bbox = _bbox_from_tokens(seg)
                if (bbox[3] - bbox[1]) < (0.08 * float(img_h)):
                    continue
                clusters.append({"axis": "y", "bbox_px": bbox, "tokens": seg})
    else:
        align_tol = max(30.0, 0.015 * float(img_h))
        by_bucket = {}
        for t in numeric_tokens:
            bucket = int(t["cy"] / align_tol)
            by_bucket.setdefault(bucket, []).append(t)
        for items in by_bucket.values():
            if len(items) < 3:
                continue
            items_sorted = sorted(items, key=lambda t: t["cx"])
            segments = _split_by_large_gaps(items_sorted, axis="x")
            for seg in segments:
                if len(seg) < 3:
                    continue
                x_gaps = []
                for i in range(1, len(seg)):
                    gap = seg[i]["cx"] - seg[i - 1]["cx"]
                    if gap > 30:
                        x_gaps.append(gap)
                if len(x_gaps) < 2:
                    continue
                avg_gap = sum(x_gaps) / len(x_gaps)
                if avg_gap < max(50.0, 0.03 * float(img_w)):
                    continue
                if not all(abs(g - avg_gap) < 0.4 * avg_gap for g in x_gaps):
                    continue
                bbox = _bbox_from_tokens(seg)
                if (bbox[2] - bbox[0]) < (0.15 * float(img_w)):
                    continue
                clusters.append({"axis": "x", "bbox_px": bbox, "tokens": seg})
    return clusters


def _split_by_large_gaps(items: List[Dict], axis: str) -> List[List[Dict]]:
    if len(items) < 3:
        return [items]
    coords = [float(it["cy"] if axis == "y" else it["cx"]) for it in items]
    gaps = [coords[i] - coords[i - 1] for i in range(1, len(coords))]
    if not gaps:
        return [items]
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 0:
        return [items]
    split_thr = median_gap * 1.8
    segments = []
    start = 0
    for i, gap in enumerate(gaps, start=1):
        if gap > split_thr:
            segments.append(items[start:i])
            start = i
    segments.append(items[start:])
    return segments


def _pair_axes_to_candidate(y_axis: Dict, x_axis: Dict, img_w: int, img_h: int) -> Optional[Dict]:
    yb = y_axis["bbox_px"]
    xb = x_axis["bbox_px"]
    y_span = max(1.0, yb[3] - yb[1])
    x_span = max(1.0, xb[2] - xb[0])
    x_cy = (xb[1] + xb[3]) / 2.0
    # Allow the X-axis to sit below the Y-axis label span for tall charts.
    min_cy = yb[1] + 0.2 * y_span
    max_cy = yb[3] + max(1.2 * y_span, 0.35 * float(img_h))
    if x_cy < min_cy or x_cy > max_cy:
        return None
    if xb[2] < yb[0] + 0.05 * float(img_w):
        return None
    x0 = min(yb[0], xb[0])
    y0 = min(yb[1], xb[1])
    x1 = max(yb[2], xb[2])
    y1 = max(yb[3], xb[3])
    pad_x = 0.1 * x_span
    pad_y = 0.1 * y_span
    bbox = [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y]
    return {
        "bbox_px": bbox,
        "axis_tokens": len(y_axis["tokens"]) + len(x_axis["tokens"]),
        "axis_span_x": x_span,
        "axis_span_y": y_span
    }


def _axis_only_candidate(
    y_axis: Dict,
    img_w: int,
    img_h: int,
    tables: List[Dict]
) -> Optional[Dict]:
    yb = y_axis.get("bbox_px") or []
    if len(yb) != 4:
        return None
    y_span = max(1.0, yb[3] - yb[1])
    min_width = 0.45 * float(img_w)
    max_width = 0.75 * float(img_w)
    target_width = max(min_width, 2.2 * y_span)
    target_width = min(target_width, max_width)
    pad_top = min(0.06 * float(img_h), 0.25 * y_span)
    pad_bottom = min(0.06 * float(img_h), 0.25 * y_span)
    x0 = max(0.0, float(yb[0]) - 0.02 * float(img_w))
    x1 = min(float(img_w), x0 + target_width)
    v0 = float(yb[1]) - 0.25 * y_span
    v1 = float(yb[3]) + 0.25 * y_span
    if tables:
        margin = max(10.0, 0.01 * float(img_w))
        for table in tables:
            tb = table.get("bbox_px") or []
            if len(tb) != 4:
                continue
            tx0, ty0, tx1, ty1 = (float(v) for v in tb)
            if tx0 <= x0:
                continue
            if ty1 < v0 or ty0 > v1:
                continue
            x1 = min(x1, tx0 - margin)
    bbox = [x0, float(yb[1]) - pad_top, x1, float(yb[3]) + pad_bottom]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return {
        "bbox_px": bbox,
        "axis_tokens": len(y_axis.get("tokens") or []),
        "axis_span_x": target_width,
        "axis_span_y": y_span
    }


def _expand_bbox_with_title_line(
    bbox: List[float],
    tokens: List[Dict],
    img_w: int,
    img_h: int,
    *,
    axis_span_y: float = 0.0
) -> List[float]:
    if not tokens or not bbox or len(bbox) != 4:
        return bbox
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if y0 <= 0:
        return bbox
    max_gap = max(120.0, 0.08 * float(img_h), 0.75 * float(axis_span_y))
    min_y = max(0.0, y0 - max_gap)
    pad_x = 0.05 * float(img_w)
    hx0 = max(0.0, x0 - pad_x)
    hx1 = min(float(img_w), x1 + pad_x)

    cand_tokens = []
    for t in tokens:
        txt = str(t.get("text") or "").strip()
        if not txt:
            continue
        try:
            tx0 = float(t.get("x0", 0.0))
            ty0 = float(t.get("y0", 0.0))
            tx1 = float(t.get("x1", 0.0))
            ty1 = float(t.get("y1", 0.0))
            cy = float(t.get("cy", (ty0 + ty1) / 2.0))
        except Exception:
            continue
        if ty0 > y0 or ty0 < min_y:
            continue
        if tx1 < hx0 or tx0 > hx1:
            continue
        cand_tokens.append({
            "text": txt,
            "x0": tx0,
            "y0": ty0,
            "x1": tx1,
            "y1": ty1,
            "cy": cy,
        })

    if not cand_tokens:
        return bbox

    heights = [max(0.0, t["y1"] - t["y0"]) for t in cand_tokens if t["y1"] > t["y0"]]
    heights.sort()
    if heights:
        med_h = heights[len(heights) // 2]
    else:
        med_h = 12.0
    y_tol = max(6.0, min(35.0, 0.6 * float(med_h)))

    cand_tokens.sort(key=lambda t: (t["cy"], t["x0"]))
    lines = []
    cur = []
    last_cy = None
    for t in cand_tokens:
        if last_cy is None or abs(float(t["cy"]) - float(last_cy)) <= y_tol:
            cur.append(t)
            last_cy = t["cy"] if last_cy is None else (0.7 * float(last_cy) + 0.3 * float(t["cy"]))
        else:
            if cur:
                lines.append(cur)
            cur = [t]
            last_cy = t["cy"]
    if cur:
        lines.append(cur)

    def _is_section_header(tokens_line: List[Dict]) -> bool:
        if not tokens_line:
            return False
        first = str(tokens_line[0].get("text") or "").strip()
        if not (bool(re.match(r"^\d+\.?$", first)) and len(tokens_line) >= 2):
            return False
        text = " ".join(str(t.get("text") or "") for t in tokens_line).strip()
        if len(text) < 10:
            return False
        alpha = sum(1 for ch in text if ch.isalpha())
        if alpha < 4:
            return False
        alpha_tokens = sum(1 for t in tokens_line[1:] if any(ch.isalpha() for ch in str(t.get("text") or "")))
        return alpha_tokens >= 1

    sorted_lines = []
    section_cutoff = None
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t["x0"])
        sorted_lines.append(line_sorted)
        if _is_section_header(line_sorted):
            ly1 = max(t["y1"] for t in line_sorted)
            if section_cutoff is None or ly1 > float(section_cutoff):
                section_cutoff = float(ly1)

    section_pad = max(6.0, 0.005 * float(img_h))
    candidates = []
    for line in sorted_lines:
        lx0 = min(t["x0"] for t in line)
        ly0 = min(t["y0"] for t in line)
        lx1 = max(t["x1"] for t in line)
        ly1 = max(t["y1"] for t in line)
        if ly1 < (0.05 * float(img_h)):
            continue
        if ly1 < min_y:
            continue
        if section_cutoff is not None and ly0 <= float(section_cutoff) + float(section_pad):
            continue
        if _is_section_header(line):
            continue
        text = " ".join(t["text"] for t in line)
        alpha = sum(1 for ch in text if ch.isalpha())
        digit = sum(1 for ch in text if ch.isdigit())
        if alpha < 4 or alpha < digit:
            continue
        if len(line) < 2 and len(text) < 8:
            continue
        candidates.append({
            "x0": lx0,
            "y0": ly0,
            "x1": lx1,
            "y1": ly1,
        })

    if not candidates:
        return bbox
    best = min(candidates, key=lambda c: c["y0"])
    pad_y = max(6.0, 0.01 * float(img_h))
    y0 = min(y0, max(0.0, float(best["y0"]) - pad_y))
    return [x0, y0, x1, y1]


def _title_based_candidates(
    img_gray: Optional["np.ndarray"],
    tokens: List[Dict],
    tables: List[Dict],
    img_w: int,
    img_h: int,
    flow_data: Optional[Dict]
) -> List[Dict]:
    if not flow_data:
        return []
    titles = flow_data.get("table_titles") or []
    if not titles:
        return []
    title_re = re.compile(r"\b(figure|fig\.?|chart|graph)\b", re.IGNORECASE)
    candidates = []
    for entry in titles:
        text = str(entry.get("text") or "")
        if not title_re.search(text):
            continue
        lines = entry.get("lines") or []
        if not lines:
            continue
        bbox = _bbox_from_lines(lines)
        if not _valid_bbox(bbox, img_w, img_h):
            continue
        below_y0 = min(img_h, bbox[3] + (0.01 * float(img_h)))
        below_y1 = min(img_h, below_y0 + (0.45 * float(img_h)))
        x0 = max(0.0, bbox[0] - 0.1 * float(img_w))
        x1 = min(float(img_w), bbox[2] + 0.4 * float(img_w))
        search_bbox = [x0, below_y0, x1, below_y1]
        search_bbox = _tighten_bbox_with_ink(search_bbox, img_gray, img_w, img_h)
        search_bbox = _expand_bbox_with_tokens(search_bbox, tokens, img_w, img_h)
        search_bbox = _clamp_bbox(search_bbox, img_w, img_h)
        if not _valid_bbox(search_bbox, img_w, img_h):
            continue
        if _overlaps_tables(search_bbox, tables):
            continue
        candidates.append({"bbox_px": search_bbox, "title": text.strip()})
    return candidates


def _bbox_from_tokens(tokens: List[Dict]) -> List[float]:
    x0 = min(t["x0"] for t in tokens)
    y0 = min(t["y0"] for t in tokens)
    x1 = max(t["x1"] for t in tokens)
    y1 = max(t["y1"] for t in tokens)
    return [float(x0), float(y0), float(x1), float(y1)]


def _bbox_from_lines(lines: List[List[Dict]]) -> List[float]:
    x0 = math.inf
    y0 = math.inf
    x1 = 0.0
    y1 = 0.0
    for line in lines:
        if not line:
            continue
        lx0 = min(float(t.get("x0", 0.0)) for t in line)
        lx1 = max(float(t.get("x1", 0.0)) for t in line)
        ly0 = min(float(t.get("y0", 0.0)) for t in line)
        ly1 = max(float(t.get("y1", 0.0)) for t in line)
        x0 = min(x0, lx0)
        y0 = min(y0, ly0)
        x1 = max(x1, lx1)
        y1 = max(y1, ly1)
    if not math.isfinite(x0):
        return [0.0, 0.0, 0.0, 0.0]
    return [x0, y0, x1, y1]


def _expand_bbox_with_tokens(
    bbox: List[float],
    tokens: List[Dict],
    img_w: int,
    img_h: int,
    *,
    pad_x: Optional[float] = None,
    pad_top: Optional[float] = None,
    pad_bottom: Optional[float] = None
) -> List[float]:
    if not tokens:
        return bbox
    x0, y0, x1, y1 = bbox
    if pad_x is None:
        pad_x = 0.08 * float(img_w)
    if pad_top is None:
        pad_top = 0.08 * float(img_h)
    if pad_bottom is None:
        pad_bottom = 0.08 * float(img_h)
    ex0 = max(0.0, x0 - pad_x)
    ex1 = min(float(img_w), x1 + pad_x)
    ey0 = max(0.0, y0 - pad_top)
    ey1 = min(float(img_h), y1 + pad_bottom)
    for t in tokens:
        txt = str(t.get("text") or "").strip()
        if not txt:
            continue
        cx = float(t.get("cx", (t.get("x0", 0.0) + t.get("x1", 0.0)) / 2.0))
        cy = float(t.get("cy", (t.get("y0", 0.0) + t.get("y1", 0.0)) / 2.0))
        if not (ex0 <= cx <= ex1 and ey0 <= cy <= ey1):
            continue
        tx0 = float(t.get("x0", 0.0))
        ty0 = float(t.get("y0", 0.0))
        tx1 = float(t.get("x1", 0.0))
        ty1 = float(t.get("y1", 0.0))
        x0 = min(x0, tx0)
        y0 = min(y0, ty0)
        x1 = max(x1, tx1)
        y1 = max(y1, ty1)
    return [x0, y0, x1, y1]


def _axis_pad_x(cand: Dict, img_w: int) -> float:
    span = float(cand.get("axis_span_x") or 0.0)
    if span <= 0:
        return 0.08 * float(img_w)
    return min(0.12 * float(img_w), 0.2 * span)


def _axis_pad_top(cand: Dict, img_h: int) -> float:
    span = float(cand.get("axis_span_y") or 0.0)
    if span <= 0:
        return 0.08 * float(img_h)
    return min(0.06 * float(img_h), 0.33 * span)


def _axis_pad_bottom(cand: Dict, img_h: int) -> float:
    span = float(cand.get("axis_span_y") or 0.0)
    if span <= 0:
        return 0.06 * float(img_h)
    return min(0.06 * float(img_h), 0.12 * span)


def _tighten_bbox_with_ink(
    bbox: List[float],
    img_gray: Optional["np.ndarray"],
    img_w: int,
    img_h: int
) -> List[float]:
    if not HAVE_CV2 or img_gray is None:
        return bbox
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0 = max(0, min(x0, img_w - 1))
    x1 = max(0, min(x1, img_w))
    y0 = max(0, min(y0, img_h - 1))
    y1 = max(0, min(y1, img_h))
    if x1 <= x0 or y1 <= y0:
        return bbox
    crop = img_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return bbox
    try:
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        return bbox
    ys, xs = (bw > 0).nonzero()
    if len(xs) < 50:
        return bbox
    ix0 = int(xs.min()) + x0
    ix1 = int(xs.max()) + x0
    iy0 = int(ys.min()) + y0
    iy1 = int(ys.max()) + y0
    pad_x = max(4, int(0.01 * float(img_w)))
    pad_y = max(4, int(0.01 * float(img_h)))
    return [ix0 - pad_x, iy0 - pad_y, ix1 + pad_x, iy1 + pad_y]


def _overlaps_tables(bbox: List[float], tables: List[Dict], threshold: float = 0.6) -> bool:
    if not tables:
        return False
    bx0, by0, bx1, by1 = bbox
    b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
    for table in tables:
        tb = table.get("bbox_px") or []
        if len(tb) != 4:
            continue
        tx0, ty0, tx1, ty1 = (float(v) for v in tb)
        ix0 = max(bx0, tx0)
        iy0 = max(by0, ty0)
        ix1 = min(bx1, tx1)
        iy1 = min(by1, ty1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter_area = (ix1 - ix0) * (iy1 - iy0)
        if inter_area / b_area > threshold:
            return True
    return False


def _clip_bbox_away_from_tables(
    bbox: List[float],
    tables: List[Dict],
    img_w: int,
    img_h: int
) -> List[float]:
    if not tables or not bbox or len(bbox) != 4:
        return bbox
    margin = max(4.0, 0.005 * float(min(img_w, img_h)))
    x0, y0, x1, y1 = bbox

    def _intersects(a: List[float], b: List[float]) -> bool:
        ix0 = max(a[0], b[0])
        iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2])
        iy1 = min(a[3], b[3])
        return ix1 > ix0 and iy1 > iy0

    for table in tables:
        tb = table.get("bbox_px") or []
        if len(tb) != 4:
            continue
        tb = [float(v) for v in tb]
        cur = [x0, y0, x1, y1]
        if not _intersects(cur, tb):
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        tcx = (tb[0] + tb[2]) / 2.0
        tcy = (tb[1] + tb[3]) / 2.0
        dx = tcx - cx
        dy = tcy - cy
        # Prefer trimming along the axis where the table is more offset.
        trim_h = abs(dx) >= abs(dy)
        trimmed = False

        def _try_vertical() -> bool:
            nonlocal y0, y1
            if dy >= 0:
                y1 = min(y1, tb[1] - margin)
            else:
                y0 = max(y0, tb[3] + margin)
            return True

        def _try_horizontal() -> bool:
            nonlocal x0, x1
            if dx >= 0:
                x1 = min(x1, tb[0] - margin)
            else:
                x0 = max(x0, tb[2] + margin)
            return True

        if trim_h:
            trimmed = _try_horizontal()
        else:
            trimmed = _try_vertical()

        cur = [x0, y0, x1, y1]
        if _intersects(cur, tb):
            # Try the other axis as a fallback if still overlapping.
            if trim_h:
                _try_vertical()
            else:
                _try_horizontal()

    return [x0, y0, x1, y1]


def _dedupe_candidates(candidates: List[Dict]) -> List[Dict]:
    if not candidates:
        return []
    kept: List[Dict] = []
    for cand in sorted(candidates, key=lambda c: _bbox_area(c["bbox_px"]), reverse=True):
        if any(_iou(cand["bbox_px"], k["bbox_px"]) > 0.55 for k in kept):
            continue
        kept.append(cand)
    return kept


def _bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _valid_bbox(bbox: List[float], img_w: int, img_h: int) -> bool:
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return False
    area = (x1 - x0) * (y1 - y0)
    if area < (0.01 * float(img_w) * float(img_h)):
        return False
    return True


def _clamp_bbox(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(x0, float(img_w)))
    x1 = max(0.0, min(x1, float(img_w)))
    y0 = max(0.0, min(y0, float(img_h)))
    y1 = max(0.0, min(y1, float(img_h)))
    return [x0, y0, x1, y1]
