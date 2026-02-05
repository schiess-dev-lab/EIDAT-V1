"""
Table Detection - Bordered cell detection and clustering

Detects table cells using pure geometric analysis of borders,
then clusters cells into logical tables.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def detect_tables(
    img_gray: np.ndarray,
    verbose: bool = False,
    *,
    cluster_gap_ratio: float = 0.02,
    cluster_gap_px: Optional[float] = None,
) -> Dict:
    """
    Main table detection function.

    Args:
        img_gray: Grayscale image (numpy array)
        verbose: Print detection stats

    Returns:
        Dict with 'tables' (list of table dicts) and 'cells' (all detected cells)
    """
    if not HAVE_CV2:
        return {'tables': [], 'cells': []}

    h, w = img_gray.shape

    # Step 1: Detect cells using contour and corner methods
    cells_contour = _find_cells_contour(img_gray)
    cells_corner = _find_cells_corner(img_gray)

    # Step 2: Merge and deduplicate
    all_cells = cells_contour + cells_corner
    all_cells = _remove_duplicate_cells(all_cells)

    # Step 2.5: Filter out thin cells (border line artifacts)
    # At 900 DPI, real cells should be at least 30px in both dimensions
    # A 2px wide "cell" is clearly a border line being detected as a cell
    min_cell_dim = 30  # ~0.85mm at 900 DPI
    all_cells = [c for c in all_cells
                 if (c['bbox_px'][2] - c['bbox_px'][0]) >= min_cell_dim and
                    (c['bbox_px'][3] - c['bbox_px'][1]) >= min_cell_dim]

    # Step 3: Filter out container cells (large frames around actual cells)
    actual_cells = _filter_containers(all_cells)

    # Step 4: Filter out oversized cells (likely chart frames)
    page_area = w * h
    max_cell_area = page_area * 0.15
    actual_cells = [c for c in actual_cells
                    if (c['bbox_px'][2] - c['bbox_px'][0]) * (c['bbox_px'][3] - c['bbox_px'][1]) <= max_cell_area]

    # Step 5: Cluster cells into tables
    tables = _cluster_into_tables(
        actual_cells,
        w,
        h,
        gap_ratio=float(cluster_gap_ratio),
        gap_px=cluster_gap_px,
    )

    # Step 5.5: Merge overlapping cells within each row band (dedupe artifacts)
    drop_ids = set()
    for table in tables:
        table_cells = table.get('cells', [])
        merged, dropped = _merge_overlapping_cells_in_rows(table_cells)
        if dropped:
            drop_ids.update(id(c) for c in dropped)
            table['cells'] = merged
            table['num_cells'] = len(merged)
            if merged:
                x0 = min(c['bbox_px'][0] for c in merged)
                y0 = min(c['bbox_px'][1] for c in merged)
                x1 = max(c['bbox_px'][2] for c in merged)
                y1 = max(c['bbox_px'][3] for c in merged)
                table['bbox_px'] = [x0, y0, x1, y1]

    if drop_ids:
        actual_cells = [c for c in actual_cells if id(c) not in drop_ids]

    # Step 6: Filter out single-cell tables (noise)
    tables = [t for t in tables if t['num_cells'] >= 2]

    if verbose:
        print(f"  Detected {len(actual_cells)} cells in {len(tables)} tables")

    return {
        'tables': tables,
        'cells': actual_cells
    }


def _find_cells_contour(img_gray: np.ndarray) -> List[Dict]:
    """Find cells by detecting closed rectangular contours."""
    h, w = img_gray.shape

    # Multi-threshold binarization
    _, bin_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_high = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    bin_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 8)
    combined = cv2.bitwise_or(cv2.bitwise_or(bin_otsu, bin_high), bin_adapt)

    # Extract lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.03), 20), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.015), 15)))

    horiz = cv2.dilate(cv2.morphologyEx(combined, cv2.MORPH_OPEN, h_kernel),
                       cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    vert = cv2.dilate(cv2.morphologyEx(combined, cv2.MORPH_OPEN, v_kernel),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))

    # Combine and close gaps
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Find contours in inverted grid
    contours, _ = cv2.findContours(cv2.bitwise_not(grid), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    min_area = (w * 0.02) * (h * 0.01)
    max_area = w * h * 0.5

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        if aspect < 0.1 or aspect > 20:
            continue

        fill_ratio = area / (cw * ch) if cw * ch > 0 else 0
        if fill_ratio < 0.7:
            continue

        cells.append({'bbox_px': [x, y, x + cw, y + ch]})

    return cells


def _find_cells_corner(img_gray: np.ndarray) -> List[Dict]:
    """Find cells by detecting line intersections (corners) and connecting lines."""
    h, w = img_gray.shape

    # Binarize
    _, bin_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_high = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    bin_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 8)
    combined = cv2.bitwise_or(cv2.bitwise_or(bin_otsu, bin_high), bin_adapt)

    # Extract lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.03), 20), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.015), 15)))

    horiz = cv2.morphologyEx(combined, cv2.MORPH_OPEN, h_kernel)
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, v_kernel)

    # Find intersections
    horiz_d = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    vert_d = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    intersections = cv2.bitwise_and(horiz_d, vert_d)

    # Get corners
    corners = _extract_corners(intersections)

    # Get line segments
    h_segs = _extract_line_segments(horiz, 'h', w, h)
    v_segs = _extract_line_segments(vert, 'v', w, h)

    # Build cells from corners and segments
    cells = _build_cells_from_corners(corners, h_segs, v_segs)

    return _remove_duplicate_cells(cells)


def _extract_corners(mask: np.ndarray, tolerance: int = 10) -> List[Tuple[int, int]]:
    """Extract corner positions from intersection mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            corners.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))

    # Cluster nearby
    merged = []
    for (x, y) in sorted(corners):
        found = False
        for i, (mx, my) in enumerate(merged):
            if abs(x - mx) < tolerance and abs(y - my) < tolerance:
                merged[i] = ((mx + x) // 2, (my + y) // 2)
                found = True
                break
        if not found:
            merged.append((x, y))

    return merged


def _extract_line_segments(mask: np.ndarray, axis: str, w: int, h: int) -> List[Dict]:
    """Extract line segments with position and extent."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if axis == 'h':
            if cw < w * 0.02:
                continue
            segments.append({'pos': y + ch // 2, 'extent': (x, x + cw)})
        else:
            if ch < h * 0.01:
                continue
            segments.append({'pos': x + cw // 2, 'extent': (y, y + ch)})

    return segments


def _build_cells_from_corners(corners: List[Tuple[int, int]],
                               h_segs: List[Dict], v_segs: List[Dict],
                               tolerance: int = 15) -> List[Dict]:
    """Build cells from corners with connecting lines."""
    cells = []
    corners_sorted = sorted(corners, key=lambda c: (c[1], c[0]))

    for i, (x1, y1) in enumerate(corners_sorted):
        for j, (x2, y2) in enumerate(corners_sorted):
            if j <= i or x2 <= x1 or y2 <= y1:
                continue

            # Check all 4 corners exist
            has_tr = any(abs(cx - x2) < tolerance and abs(cy - y1) < tolerance for (cx, cy) in corners)
            has_bl = any(abs(cx - x1) < tolerance and abs(cy - y2) < tolerance for (cx, cy) in corners)
            if not (has_tr and has_bl):
                continue

            # Check all 4 lines exist
            has_top = any(abs(s['pos'] - y1) < tolerance and
                         s['extent'][0] <= x1 + tolerance and s['extent'][1] >= x2 - tolerance
                         for s in h_segs)
            has_bot = any(abs(s['pos'] - y2) < tolerance and
                         s['extent'][0] <= x1 + tolerance and s['extent'][1] >= x2 - tolerance
                         for s in h_segs)
            has_left = any(abs(s['pos'] - x1) < tolerance and
                          s['extent'][0] <= y1 + tolerance and s['extent'][1] >= y2 - tolerance
                          for s in v_segs)
            has_right = any(abs(s['pos'] - x2) < tolerance and
                           s['extent'][0] <= y1 + tolerance and s['extent'][1] >= y2 - tolerance
                           for s in v_segs)

            if has_top and has_bot and has_left and has_right:
                cells.append({'bbox_px': [x1, y1, x2, y2]})

    return cells


def _remove_duplicate_cells(cells: List[Dict]) -> List[Dict]:
    """Remove cells with nearly identical bboxes."""
    if not cells:
        return cells

    cells_sorted = sorted(cells, key=lambda c: (c['bbox_px'][2] - c['bbox_px'][0]) *
                                                (c['bbox_px'][3] - c['bbox_px'][1]))

    keep = []
    for cell in cells_sorted:
        x0, y0, x1, y1 = cell['bbox_px']
        w, h = x1 - x0, y1 - y0

        is_dup = False
        for kept in keep:
            kx0, ky0, kx1, ky1 = kept['bbox_px']
            if (abs(x0 - kx0) < w * 0.05 and abs(y0 - ky0) < h * 0.05 and
                abs(x1 - kx1) < w * 0.05 and abs(y1 - ky1) < h * 0.05):
                is_dup = True
                break

        if not is_dup:
            keep.append(cell)

    return keep


def _filter_containers(cells: List[Dict]) -> List[Dict]:
    """Remove cells that contain other cells (table frames)."""
    if len(cells) < 2:
        return cells

    actual = []
    for i, cell in enumerate(cells):
        x0, y0, x1, y1 = cell['bbox_px']
        area = max(1, (x1 - x0) * (y1 - y0))
        contained_areas = []

        for j, other in enumerate(cells):
            if i == j:
                continue

            ox0, oy0, ox1, oy1 = other['bbox_px']
            other_area = max(1, (ox1 - ox0) * (oy1 - oy0))

            # Compute overlap area
            ix0 = max(x0, ox0)
            iy0 = max(y0, oy0)
            ix1 = min(x1, ox1)
            iy1 = min(y1, oy1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            overlap_area = (ix1 - ix0) * (iy1 - iy0)
            overlap_ratio = overlap_area / float(other_area)

            # Treat "mostly inside" as contained (tolerates minor misalignment)
            if overlap_ratio >= 0.85:
                contained_areas.append(other_area)

        # If a cell is mostly covered by 2+ other cells, it is a container/frame.
        if len(contained_areas) >= 2 and (sum(contained_areas) / float(area)) >= 0.6:
            continue

        actual.append(cell)

    return actual


def _cluster_into_tables(
    cells: List[Dict],
    img_w: int,
    img_h: int,
    gap_ratio: float = 0.02,
    gap_px: Optional[float] = None,
) -> List[Dict]:
    """Cluster cells into tables based on direct adjacency."""
    if not cells:
        return []

    gap = max(img_w, img_h) * gap_ratio
    if gap_px is not None:
        try:
            gap_px_val = float(gap_px)
        except (TypeError, ValueError):
            gap_px_val = None
        if gap_px_val is not None and gap_px_val > 0:
            gap = min(gap, gap_px_val)
    n = len(cells)
    adj = [set() for _ in range(n)]

    for i in range(n):
        x0i, y0i, x1i, y1i = cells[i]['bbox_px']
        wi, hi = x1i - x0i, y1i - y0i

        for j in range(i + 1, n):
            x0j, y0j, x1j, y1j = cells[j]['bbox_px']
            wj, hj = x1j - x0j, y1j - y0j

            y_ovr = max(0, min(y1i, y1j) - max(y0i, y0j))
            x_ovr = max(0, min(x1i, x1j) - max(x0i, x0j))

            same_row = y_ovr > min(hi, hj) * 0.5
            same_col = x_ovr > min(wi, wj) * 0.5

            x_gap = max(0, max(x0i, x0j) - min(x1i, x1j))
            y_gap = max(0, max(y0i, y0j) - min(y1i, y1j))

            if (same_row and x_gap < gap) or (same_col and y_gap < gap):
                adj[i].add(j)
                adj[j].add(i)

    # DFS for connected components
    visited = [False] * n
    tables = []

    def dfs(node, comp):
        visited[node] = True
        comp.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                dfs(nb, comp)

    for i in range(n):
        if not visited[i]:
            comp = []
            dfs(i, comp)
            if comp:
                table_cells = [cells[idx] for idx in comp]
                x0 = min(c['bbox_px'][0] for c in table_cells)
                y0 = min(c['bbox_px'][1] for c in table_cells)
                x1 = max(c['bbox_px'][2] for c in table_cells)
                y1 = max(c['bbox_px'][3] for c in table_cells)

                tables.append({
                    'bbox_px': [x0, y0, x1, y1],
                    'cells': table_cells,
                    'num_cells': len(table_cells)
                })

    return tables


def _merge_overlapping_cells_in_rows(cells: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Merge near-duplicate cells that overlap heavily within the same row band.

    This removes detection artifacts where a single logical cell is detected twice
    with slightly offset borders.
    """
    if not cells:
        return cells, []

    heights = sorted(
        [c['bbox_px'][3] - c['bbox_px'][1] for c in cells if c.get('bbox_px')]
    )
    median_h = heights[len(heights) // 2] if heights else 0
    row_tol = max(8.0, float(median_h) * 0.5)

    def _cy(cell: Dict) -> float:
        y0, y1 = cell['bbox_px'][1], cell['bbox_px'][3]
        return (y0 + y1) / 2.0

    def _area(cell: Dict) -> float:
        x0, y0, x1, y1 = cell['bbox_px']
        return max(1.0, float(x1 - x0) * float(y1 - y0))

    def _overlap_stats(a: Dict, b: Dict) -> Tuple[float, float]:
        ax0, ay0, ax1, ay1 = a['bbox_px']
        bx0, by0, bx1, by1 = b['bbox_px']
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0, 0.0
        overlap = float(ix1 - ix0) * float(iy1 - iy0)
        area_a = _area(a)
        area_b = _area(b)
        iou = overlap / max(1.0, (area_a + area_b - overlap))
        overlap_min = overlap / max(1.0, min(area_a, area_b))
        return iou, overlap_min

    # Build row bands by y-center proximity
    cells_sorted = sorted(cells, key=lambda c: (_cy(c), c['bbox_px'][0]))
    rows: List[List[Dict]] = []
    current: List[Dict] = []
    current_cy = None
    for cell in cells_sorted:
        cy = _cy(cell)
        if current and current_cy is not None and abs(cy - current_cy) > row_tol:
            rows.append(current)
            current = [cell]
            current_cy = cy
        else:
            current.append(cell)
            if current_cy is None:
                current_cy = cy
            else:
                current_cy = (current_cy * (len(current) - 1) + cy) / float(len(current))
    if current:
        rows.append(current)

    drop = set()
    for row in rows:
        for i in range(len(row)):
            a = row[i]
            if id(a) in drop:
                continue
            for j in range(i + 1, len(row)):
                b = row[j]
                if id(b) in drop:
                    continue
                iou, overlap_min = _overlap_stats(a, b)
                if iou >= 0.8 or overlap_min >= 0.9:
                    # Drop the smaller (or later) cell
                    if _area(a) >= _area(b):
                        drop.add(id(b))
                    else:
                        drop.add(id(a))
                        break

    kept = [c for c in cells if id(c) not in drop]
    dropped = [c for c in cells if id(c) in drop]
    return kept, dropped
