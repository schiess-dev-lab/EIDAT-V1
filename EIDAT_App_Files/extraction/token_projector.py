"""
Token Projector - Project OCR tokens into table cells

Assigns OCR tokens to table cells based on spatial overlap,
then organizes cells into row/column structure.
"""

import os
import re
from typing import List, Dict, Tuple, Optional

# Legacy constant - kept for backward compatibility
REOCR_CONFIDENCE_THRESHOLD = 0.5

# Minimum overlap to consider assigning a token to a cell
CELL_OVERLAP_MIN = 0.3


_TABLE_VERTICAL_ARTIFACT_CHARS = "|¦│┃丨"
_TABLE_SECONDARY_PREFIX_ARTIFACT_CHARS = "[](){}\"'`“”‘’\\/"  # common OCR border/quote variants
_TABLE_QUOTE_ARTIFACT_CHARS = "\"'`“”‘’"
_TABLE_OPENERS = {"[": "]", "(": ")", "{": "}"}


def normalize_table_cell_text(text: str) -> str:
    """
    Normalize table cell text for common border/line-detection OCR artifacts.

    We frequently see left-border remnants recognized as punctuation like:
    - |"Pressure  -> Pressure
    - ["Flow     -> Flow
    - │ 10.0     -> 10.0

    This is intentionally conservative and only strips *leading* artifacts.

    Disable with: EIDAT_TABLE_CELL_PREFIX_CLEAN=0
    """
    s = "" if text is None else str(text)
    s = s.replace("\u00a0", " ").strip()
    if not s:
        return ""

    enabled = str(os.environ.get("EIDAT_TABLE_CELL_PREFIX_CLEAN", "1")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return s

    s2 = s.lstrip()

    # 1) Strip vertical border artifacts (and any immediately-adjacent punctuation).
    if s2 and s2[0] in _TABLE_VERTICAL_ARTIFACT_CHARS:
        s2 = re.sub(rf"^[{re.escape(_TABLE_VERTICAL_ARTIFACT_CHARS)}]+", "", s2).lstrip()
        s2 = re.sub(rf"^[{re.escape(_TABLE_SECONDARY_PREFIX_ARTIFACT_CHARS)}]+", "", s2).lstrip()
        return s2.strip()

    # 2) Strip bracket-like artifacts when they don't look like a balanced short reference (e.g. [1]).
    if s2 and s2[0] in _TABLE_OPENERS:
        closer = _TABLE_OPENERS[s2[0]]
        first_token = s2.split(None, 1)[0]
        opens_with_quote = len(s2) > 1 and s2[1] in _TABLE_QUOTE_ARTIFACT_CHARS
        has_early_closer = (closer in first_token) or (closer in s2[:6])
        if opens_with_quote or not has_early_closer:
            s2 = s2[1:].lstrip()
            s2 = re.sub(rf"^[{re.escape(_TABLE_QUOTE_ARTIFACT_CHARS)}]+", "", s2).lstrip()
            return s2.strip()

    return s


def scale_tokens_to_dpi(tokens: List[Dict], from_dpi: int, to_dpi: int) -> List[Dict]:
    """
    Scale token coordinates from one DPI to another.

    Args:
        tokens: List of OCR tokens with x0, y0, x1, y1, cx, cy coordinates
        from_dpi: Source DPI the tokens were extracted at
        to_dpi: Target DPI to scale coordinates to

    Returns:
        New list of tokens with scaled coordinates (original tokens unchanged)
    """
    if from_dpi == to_dpi:
        return tokens

    scale = to_dpi / from_dpi
    scaled = []

    for t in tokens:
        scaled_token = dict(t)  # Copy all fields
        # Scale coordinate fields
        for key in ['x0', 'y0', 'x1', 'y1', 'cx', 'cy']:
            if key in scaled_token:
                scaled_token[key] = float(scaled_token[key]) * scale
        scaled.append(scaled_token)

    return scaled


def _build_cell_text(tokens: List[Dict], char_gap_ratio: Optional[float] = None,
                     line_tol_ratio: float = 0.5) -> str:
    """
    Build cell text from sorted tokens.

    If char_gap_ratio is None, tokens are joined with single spaces (legacy behavior).
    Otherwise, gaps smaller than a fraction of token height are treated as
    character gaps (no space), while larger gaps become word separators.
    """
    if not tokens:
        return ""

    if char_gap_ratio is None:
        joined = ' '.join(t.get('text', '') for t in tokens).strip()
        return normalize_table_cell_text(joined)

    parts: List[str] = []
    prev = tokens[0]
    first_text = str(prev.get('text', ''))
    if first_text:
        parts.append(first_text)

    for tok in tokens[1:]:
        text = str(tok.get('text', ''))
        if not text:
            continue

        prev_cy = float(prev.get('cy', (prev.get('y0', 0) + prev.get('y1', 0)) / 2))
        tok_cy = float(tok.get('cy', (tok.get('y0', 0) + tok.get('y1', 0)) / 2))
        prev_h = float(prev.get('y1', 0)) - float(prev.get('y0', 0))
        tok_h = float(tok.get('y1', 0)) - float(tok.get('y0', 0))

        line_tol = max(1.0, min(prev_h, tok_h) * line_tol_ratio)
        same_line = abs(tok_cy - prev_cy) <= line_tol

        gap = float(tok.get('x0', 0)) - float(prev.get('x1', 0))
        gap_thresh = max(1.0, min(prev_h, tok_h) * char_gap_ratio)

        if not same_line or gap > gap_thresh:
            parts.append(' ')

        parts.append(text)
        prev = tok

    joined = ''.join(parts).strip()
    return normalize_table_cell_text(joined)


def _sort_tokens_reading_order(
    tokens: List[Dict],
    *,
    line_tol_ratio: float = 0.5
) -> List[Dict]:
    if not tokens:
        return tokens

    heights = sorted(
        max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
        for t in tokens
    )
    median_h = heights[len(heights) // 2]
    line_tol = max(1.0, median_h * line_tol_ratio)

    def _cy(t: Dict) -> float:
        return float(t.get("cy", (t.get("y0", 0) + t.get("y1", 0)) / 2))

    sorted_tokens = sorted(
        tokens,
        key=lambda t: (_cy(t), float(t.get("x0", 0)))
    )

    lines = []
    for tok in sorted_tokens:
        cy = _cy(tok)
        assigned = False
        for line in lines:
            if abs(cy - line["cy"]) <= line_tol:
                line["tokens"].append(tok)
                line["cy"] = (line["cy"] * line["count"] + cy) / (line["count"] + 1)
                line["count"] += 1
                assigned = True
                break
        if not assigned:
            lines.append({"cy": cy, "count": 1, "tokens": [tok]})

    lines.sort(key=lambda l: l["cy"])
    ordered: List[Dict] = []
    for line in lines:
        ordered.extend(sorted(line["tokens"], key=lambda t: float(t.get("x0", 0))))

    return ordered


def _dedupe_overlapping_tokens(tokens: List[Dict],
                               overlap_min: float = 0.85) -> List[Dict]:
    """
    Deduplicate tokens with the same text that heavily overlap in bbox.

    Keeps the higher-confidence token (or smaller area on ties).
    """
    if not tokens:
        return tokens

    def _area(t: Dict) -> float:
        return max(1.0, float(t.get('x1', 0)) - float(t.get('x0', 0))) * max(
            1.0, float(t.get('y1', 0)) - float(t.get('y0', 0))
        )

    def _overlap_ratio(a: Dict, b: Dict) -> float:
        ax0, ay0, ax1, ay1 = float(a.get('x0', 0)), float(a.get('y0', 0)), float(a.get('x1', 0)), float(a.get('y1', 0))
        bx0, by0, bx1, by1 = float(b.get('x0', 0)), float(b.get('y0', 0)), float(b.get('x1', 0)), float(b.get('y1', 0))
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0
        overlap = (ix1 - ix0) * (iy1 - iy0)
        return overlap / max(1.0, min(_area(a), _area(b)))

    kept = []
    for tok in tokens:
        text = tok.get('text', '')
        if not text:
            kept.append(tok)
            continue

        replaced = False
        for i, existing in enumerate(kept):
            if existing.get('text', '') != text:
                continue
            if _overlap_ratio(existing, tok) < overlap_min:
                continue

            # Keep higher-confidence token; on tie keep tighter bbox.
            conf_existing = float(existing.get('conf', 0.0))
            conf_tok = float(tok.get('conf', 0.0))
            if conf_tok > conf_existing:
                kept[i] = tok
            elif conf_tok == conf_existing and _area(tok) < _area(existing):
                kept[i] = tok
            replaced = True
            break

        if not replaced:
            kept.append(tok)

    return kept


def _normalize_token_text(text: str) -> str:
    if not text:
        return ""
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _dedupe_overlapping_tokens_any_text(
    tokens: List[Dict],
    overlap_min: float = 0.90,
    strong_overlap_min: float = 0.98
) -> List[Dict]:
    """
    Deduplicate tokens that heavily overlap, even if text differs.

    Prefers higher-confidence tokens; uses normalized text to treat
    punctuation variants or substrings as duplicates.
    """
    if not tokens:
        return tokens

    def _area(t: Dict) -> float:
        return max(1.0, float(t.get("x1", 0)) - float(t.get("x0", 0))) * max(
            1.0, float(t.get("y1", 0)) - float(t.get("y0", 0))
        )

    def _overlap_ratio(a: Dict, b: Dict) -> float:
        ax0, ay0, ax1, ay1 = float(a.get("x0", 0)), float(a.get("y0", 0)), float(a.get("x1", 0)), float(a.get("y1", 0))
        bx0, by0, bx1, by1 = float(b.get("x0", 0)), float(b.get("y0", 0)), float(b.get("x1", 0)), float(b.get("y1", 0))
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0
        overlap = (ix1 - ix0) * (iy1 - iy0)
        return overlap / max(1.0, min(_area(a), _area(b)))

    def _quality(t: Dict) -> Tuple[float, int, int]:
        text = str(t.get("text", ""))
        norm = _normalize_token_text(text)
        return (float(t.get("conf", 0.0)), len(norm), len(text))

    sorted_tokens = sorted(tokens, key=_quality, reverse=True)
    kept: List[Dict] = []
    kept_norm: List[str] = []

    for tok in sorted_tokens:
        tok_norm = _normalize_token_text(tok.get("text", ""))
        drop = False
        for existing, existing_norm in zip(kept, kept_norm):
            overlap = _overlap_ratio(tok, existing)
            if overlap < overlap_min:
                continue
            similar = False
            if tok_norm and existing_norm:
                similar = (
                    tok_norm == existing_norm
                    or tok_norm in existing_norm
                    or existing_norm in tok_norm
                )
            if similar or overlap >= strong_overlap_min:
                drop = True
                break
        if not drop:
            kept.append(tok)
            kept_norm.append(tok_norm)

    return kept


def project_tokens_to_cells(tokens: List[Dict], cells: List[Dict],
                             overlap_threshold: float = 0.5) -> List[Dict]:
    """
    Assign OCR tokens to cells based on spatial overlap.

    Args:
        tokens: List of OCR tokens with x0, y0, x1, y1, text, conf
        cells: List of cells with bbox_px
        overlap_threshold: Minimum overlap ratio (0-1)

    Returns:
        Updated cells with 'tokens', 'text', and 'token_count' fields
    """
    for cell in cells:
        cx0, cy0, cx1, cy1 = cell['bbox_px']
        cell_tokens = []

        for token in tokens:
            tx0 = float(token.get('x0', 0))
            ty0 = float(token.get('y0', 0))
            tx1 = float(token.get('x1', 0))
            ty1 = float(token.get('y1', 0))

            if tx1 <= tx0 or ty1 <= ty0:
                continue

            # Calculate overlap
            ox0 = max(cx0, tx0)
            oy0 = max(cy0, ty0)
            ox1 = min(cx1, tx1)
            oy1 = min(cy1, ty1)

            if ox0 < ox1 and oy0 < oy1:
                overlap_area = (ox1 - ox0) * (oy1 - oy0)
                token_area = max(1, (tx1 - tx0) * (ty1 - ty0))

                if overlap_area / token_area >= overlap_threshold:
                    cell_tokens.append({
                        'text': str(token.get('text', '')),
                        'x0': tx0,
                        'y0': ty0,
                        'cx': float(token.get('cx', (tx0 + tx1) / 2)),
                        'cy': float(token.get('cy', (ty0 + ty1) / 2)),
                        'conf': float(token.get('conf', 0))
                    })

        cell_tokens = _sort_tokens_reading_order(cell_tokens)

        cell['tokens'] = cell_tokens
        cell['text'] = _build_cell_text(cell_tokens)
        cell['token_count'] = len(cell_tokens)

    return cells


def project_tokens_to_cells_with_confidence(
    tokens: List[Dict],
    cells: List[Dict],
    overlap_threshold: float = 0.5,
    reocr_threshold: float = None
) -> List[Dict]:
    """
    Project tokens to cells and calculate confidence scores.

    This is the PRIMARY method for cell text extraction. It matches
    full-page OCR tokens to detected cells based on spatial overlap,
    then calculates confidence to determine if re-OCR is needed.

    Args:
        tokens: List of OCR tokens with x0, y0, x1, y1, text, conf
        cells: List of cells with bbox_px
        overlap_threshold: Minimum overlap ratio for token assignment (0-1)
        reocr_threshold: Confidence threshold below which re-OCR is flagged
                        (defaults to REOCR_CONFIDENCE_THRESHOLD)

    Returns:
        Updated cells with:
        - tokens: list of matched tokens
        - text: combined text in reading order
        - token_count: number of tokens
        - projection_confidence: 0-1 confidence score
        - needs_reocr: bool flag for cells requiring re-OCR
        - ocr_method: 'token_projection'
    """
    if reocr_threshold is None:
        reocr_threshold = REOCR_CONFIDENCE_THRESHOLD

    for cell in cells:
        cx0, cy0, cx1, cy1 = cell['bbox_px']
        cell_area = max(1, (cx1 - cx0) * (cy1 - cy0))
        cell_tokens = []
        total_token_area_in_cell = 0
        total_overlap_ratio = 0

        for token in tokens:
            tx0 = float(token.get('x0', 0))
            ty0 = float(token.get('y0', 0))
            tx1 = float(token.get('x1', 0))
            ty1 = float(token.get('y1', 0))

            if tx1 <= tx0 or ty1 <= ty0:
                continue

            # Calculate overlap
            ox0 = max(cx0, tx0)
            oy0 = max(cy0, ty0)
            ox1 = min(cx1, tx1)
            oy1 = min(cy1, ty1)

            if ox0 < ox1 and oy0 < oy1:
                overlap_area = (ox1 - ox0) * (oy1 - oy0)
                token_area = max(1, (tx1 - tx0) * (ty1 - ty0))
                overlap_ratio = overlap_area / token_area

                if overlap_ratio >= overlap_threshold:
                    cell_tokens.append({
                        'text': str(token.get('text', '')),
                        'x0': tx0,
                        'y0': ty0,
                        'x1': tx1,
                        'y1': ty1,
                        'cx': float(token.get('cx', (tx0 + tx1) / 2)),
                        'cy': float(token.get('cy', (ty0 + ty1) / 2)),
                        'conf': float(token.get('conf', 0)),
                        'overlap_ratio': overlap_ratio
                    })
                    # Track coverage metrics
                    total_token_area_in_cell += overlap_area
                    total_overlap_ratio += overlap_ratio

        cell_tokens = _sort_tokens_reading_order(cell_tokens)

        # Calculate confidence score
        # Based on: token coverage + average fit quality + having tokens
        if cell_tokens:
            coverage_ratio = min(1.0, total_token_area_in_cell / cell_area)
            avg_overlap = total_overlap_ratio / len(cell_tokens)
            avg_token_conf = sum(t['conf'] for t in cell_tokens) / len(cell_tokens)

            # Weighted confidence: coverage (40%) + fit (30%) + OCR conf (30%)
            confidence = (
                0.4 * coverage_ratio +
                0.3 * avg_overlap +
                0.3 * avg_token_conf
            )
        else:
            # No tokens found - zero confidence
            confidence = 0.0

        cell['tokens'] = cell_tokens
        cell['text'] = _build_cell_text(cell_tokens)
        cell['token_count'] = len(cell_tokens)
        cell['projection_confidence'] = round(confidence, 3)
        cell['needs_reocr'] = confidence < reocr_threshold
        cell['ocr_method'] = 'token_projection'

    return cells


def project_tokens_to_cells_force(
    tokens: List[Dict],
    cells: List[Dict],
    verbose: bool = False,
    debug_info: Optional[Dict] = None,
    ocr_dpi: int = 450,
    detection_dpi: int = 900,
    *,
    reset_cells: bool = True,
    center_margin_px: float = 12.0,
    only_if_empty: bool = False,
    char_gap_ratio: Optional[float] = None,
    max_token_ratio: Optional[float] = None,
    max_token_area_ratio: Optional[float] = None
) -> Tuple[List[Dict], Dict]:
    """
    Force-project tokens into cells using "highest overlap wins" rule.

    Unlike project_tokens_to_cells_with_confidence, this function:
    - Assigns tokens to cells with ANY overlap (no minimum threshold)
    - Uses "highest overlap wins" when token spans multiple cells
    - Does NOT flag cells for re-OCR based on projection confidence
    - Tracks unassigned tokens for debugging

    Args:
        tokens: List of OCR tokens with x0, y0, x1, y1, text, conf
                (ALREADY SCALED to detection_dpi coordinates)
        cells: List of cells with bbox_px (at detection_dpi)
        verbose: Print debug info about unassigned tokens
        debug_info: Optional dict to populate with detailed projection debug data
        ocr_dpi: The DPI tokens were originally extracted at (for debug output)
        detection_dpi: The DPI cells were detected at (for debug output)
        char_gap_ratio: When set, remove inter-character gaps by inserting spaces
                        only for gaps larger than this fraction of token height.
        max_token_ratio: If set, skip assigning tokens that are much larger than
                        the candidate cell (width/height ratio).
        max_token_area_ratio: If set, skip tokens whose area is far larger than
                        the candidate cell area.

    Returns:
        Tuple of (updated cells, stats dict)
        - cells have: tokens, text, token_count, ocr_method
        - stats: assigned_count, unassigned_count, unassigned_tokens
    """
    # Initialize all cells (optional)
    if reset_cells:
        for cell in cells:
            cell['tokens'] = []
            cell['text'] = ''
            cell['token_count'] = 0
            cell['ocr_method'] = 'token_projection_force'
    else:
        for cell in cells:
            cell.setdefault('tokens', [])
            cell.setdefault('text', '')
            cell.setdefault('token_count', len(cell.get('tokens', [])))
            cell.setdefault('ocr_method', 'token_projection_force')

    assigned_count = 0
    unassigned_tokens = []

    # Debug tracking per cell
    cell_debug = []
    if debug_info is not None:
        debug_info['ocr_dpi'] = ocr_dpi
        debug_info['detection_dpi'] = detection_dpi
        debug_info['scale_factor'] = detection_dpi / ocr_dpi
        debug_info['cells'] = cell_debug

    # For each token, find the best cell (highest overlap)
    for token in tokens:
        tx0 = float(token.get('x0', 0))
        ty0 = float(token.get('y0', 0))
        tx1 = float(token.get('x1', 0))
        ty1 = float(token.get('y1', 0))

        if tx1 <= tx0 or ty1 <= ty0:
            continue

        token_area = (tx1 - tx0) * (ty1 - ty0)
        if token_area <= 0:
            continue

        best_cell = None
        best_cell_idx = -1
        best_overlap_area = 0
        best_overlap_ratio = 0

        # Track all overlap attempts for debugging
        token_overlaps = []

        # Prefer center-in-cell matching (with margin) to avoid brittle overlap-only assignment.
        tcx = float(token.get('cx', (tx0 + tx1) / 2))
        tcy = float(token.get('cy', (ty0 + ty1) / 2))
        candidate_indices: List[int] = []
        if center_margin_px and center_margin_px > 0:
            m = float(center_margin_px)
            for cell_idx, cell in enumerate(cells):
                if only_if_empty and cell.get("tokens"):
                    continue
                cx0, cy0, cx1, cy1 = cell['bbox_px']
                if (cx0 - m) <= tcx <= (cx1 + m) and (cy0 - m) <= tcy <= (cy1 + m):
                    candidate_indices.append(cell_idx)
        else:
            candidate_indices = [i for i, c in enumerate(cells) if not (only_if_empty and c.get("tokens"))]

        # If no center candidates, fall back to overlap scan over all cells.
        if not candidate_indices:
            candidate_indices = [i for i, c in enumerate(cells) if not (only_if_empty and c.get("tokens"))]
            if not candidate_indices:
                candidate_indices = list(range(len(cells)))

        # Find cell with highest overlap (ANY overlap counts)
        for cell_idx in candidate_indices:
            cell = cells[cell_idx]
            cx0, cy0, cx1, cy1 = cell['bbox_px']

            # Calculate overlap
            ox0 = max(cx0, tx0)
            oy0 = max(cy0, ty0)
            ox1 = min(cx1, tx1)
            oy1 = min(cy1, ty1)

            if ox0 < ox1 and oy0 < oy1:
                if max_token_ratio or max_token_area_ratio:
                    cell_w = max(1.0, float(cx1 - cx0))
                    cell_h = max(1.0, float(cy1 - cy0))
                    token_w = float(tx1 - tx0)
                    token_h = float(ty1 - ty0)
                    if max_token_ratio:
                        ratio = float(max_token_ratio)
                        if token_w > cell_w * ratio or token_h > cell_h * ratio:
                            continue
                    if max_token_area_ratio:
                        area_ratio = float(max_token_area_ratio)
                        if (token_w * token_h) > (cell_w * cell_h * area_ratio):
                            continue

                overlap_area = (ox1 - ox0) * (oy1 - oy0)
                overlap_ratio = overlap_area / token_area

                token_overlaps.append({
                    'cell_idx': cell_idx,
                    'overlap_area': overlap_area,
                    'overlap_pct': round(overlap_ratio * 100, 1)
                })

                # Check if this is the best cell so far
                if overlap_area > best_overlap_area:
                    best_cell = cell
                    best_cell_idx = cell_idx
                    best_overlap_area = overlap_area
                    best_overlap_ratio = overlap_ratio

        # Assign token to best cell if ANY overlap exists (no threshold)
        if best_cell is not None and best_overlap_area > 0:
            best_cell['tokens'].append({
                'text': str(token.get('text', '')),
                'x0': tx0,
                'y0': ty0,
                'x1': tx1,
                'y1': ty1,
                'cx': tcx,
                'cy': tcy,
                'conf': float(token.get('conf', 0)),
                'overlap_ratio': best_overlap_ratio
            })
            assigned_count += 1
        else:
            # Token has no overlap with any cell
            # Calculate original (unscaled) coordinates for debug
            scale = detection_dpi / ocr_dpi
            unassigned_tokens.append({
                'text': token.get('text', ''),
                'bbox_scaled': [tx0, ty0, tx1, ty1],
                'bbox_original': [tx0/scale, ty0/scale, tx1/scale, ty1/scale],
                'scale_factor': scale
            })

    # Sort tokens within each cell and build text
    for cell in cells:
        cell_tokens = cell['tokens']

        cell_tokens = _sort_tokens_reading_order(cell_tokens)

        # Deduplicate tokens (exact duplicates, then overlapping with same text).
        if cell_tokens:
            seen = set()
            deduped = []
            for t in cell_tokens:
                key = (
                    t.get('text', ''),
                    int(round(float(t.get('x0', 0.0)))),
                    int(round(float(t.get('y0', 0.0)))),
                    int(round(float(t.get('x1', 0.0)))),
                    int(round(float(t.get('y1', 0.0)))),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(t)
            cell_tokens = _dedupe_overlapping_tokens(deduped, overlap_min=0.85)
            cell_tokens = _dedupe_overlapping_tokens_any_text(
                cell_tokens, overlap_min=0.90, strong_overlap_min=0.98
            )
            if cell_tokens:
                heights = sorted(
                    max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
                    for t in cell_tokens
                )
                median_h = heights[len(heights) // 2]
                min_h = median_h * 0.30
                filtered = []
                for t in cell_tokens:
                    h = max(1.0, float(t.get("y1", 0)) - float(t.get("y0", 0)))
                    conf = float(t.get("conf", 0.0))
                    if h < min_h and conf < 0.65:
                        continue
                    filtered.append(t)
                if filtered:
                    cell_tokens = filtered
            cell_tokens = _sort_tokens_reading_order(cell_tokens)
            cell['tokens'] = cell_tokens

        cell['text'] = _build_cell_text(cell_tokens, char_gap_ratio=char_gap_ratio)
        cell['token_count'] = len(cell_tokens)

    # Build detailed debug info per cell
    if debug_info is not None:
        scale = detection_dpi / ocr_dpi
        for cell_idx, cell in enumerate(cells):
            cx0, cy0, cx1, cy1 = cell['bbox_px']
            cell_area = (cx1 - cx0) * (cy1 - cy0)

            # Calculate what the cell bbox would be at OCR DPI
            cell_at_ocr_dpi = [cx0/scale, cy0/scale, cx1/scale, cy1/scale]

            cell_dbg = {
                'cell_idx': cell_idx,
                'row': cell.get('row'),
                'col': cell.get('col'),
                'bbox_detection_dpi': [round(cx0, 1), round(cy0, 1), round(cx1, 1), round(cy1, 1)],
                'bbox_ocr_dpi_equivalent': [round(v, 1) for v in cell_at_ocr_dpi],
                'cell_area_px': round(cell_area, 1),
                'token_count': len(cell['tokens']),
                'text': cell['text'],
                'tokens_detail': []
            }

            # Add token details
            for tok in cell['tokens']:
                tok_original = [tok['x0']/scale, tok['y0']/scale, tok['x1']/scale, tok['y1']/scale]
                tok_dbg = {
                    'text': tok['text'],
                    'bbox_scaled_to_det_dpi': [round(tok['x0'], 1), round(tok['y0'], 1),
                                                round(tok['x1'], 1), round(tok['y1'], 1)],
                    'bbox_original_ocr_dpi': [round(v, 1) for v in tok_original],
                    'overlap_pct': round(tok['overlap_ratio'] * 100, 1),
                    'conf': round(tok['conf'], 2)
                }
                cell_dbg['tokens_detail'].append(tok_dbg)

            cell_debug.append(cell_dbg)

    stats = {
        'assigned_count': assigned_count,
        'unassigned_count': len(unassigned_tokens),
        'unassigned_tokens': unassigned_tokens
    }

    if verbose and unassigned_tokens:
        print(f"    {len(unassigned_tokens)} tokens had no cell overlap")
        # Show sample with coordinates
        for t in unassigned_tokens[:3]:
            print(f"      '{t['text']}' scaled:{t['bbox_scaled']} original:{t['bbox_original']}")

    return cells, stats


def organize_table_structure(cells: List[Dict], tolerance: float = 0.3) -> Dict:
    """
    Organize cells into row/column grid structure.

    Args:
        cells: List of cells with bbox_px
        tolerance: Position tolerance for row grouping (0-1)

    Returns:
        Dict with 'rows' (list of rows), 'num_rows', 'num_cols'
    """
    if not cells:
        return {'rows': [], 'num_rows': 0, 'num_cols': 0}

    # Sort by y position
    sorted_cells = sorted(cells, key=lambda c: c['bbox_px'][1])

    rows = []
    current_row = [sorted_cells[0]]
    row_y = sorted_cells[0]['bbox_px'][1]
    row_h = sorted_cells[0]['bbox_px'][3] - sorted_cells[0]['bbox_px'][1]

    for cell in sorted_cells[1:]:
        cell_y = cell['bbox_px'][1]
        cell_h = cell['bbox_px'][3] - cell['bbox_px'][1]

        # Check if in same row
        if abs(cell_y - row_y) < row_h * tolerance:
            current_row.append(cell)
        else:
            # Sort current row by x position
            rows.append(sorted(current_row, key=lambda c: c['bbox_px'][0]))
            current_row = [cell]
            row_y = cell_y
            row_h = cell_h

    if current_row:
        rows.append(sorted(current_row, key=lambda c: c['bbox_px'][0]))

    max_cols = max(len(row) for row in rows) if rows else 0

    return {
        'rows': rows,
        'num_rows': len(rows),
        'num_cols': max_cols
    }


def assign_row_col_indices(cells: List[Dict]) -> List[Dict]:
    """
    Assign row and col indices to cells based on their positions.

    Modifies cells in-place to add 'row' and 'col' fields.
    """
    structure = organize_table_structure(cells)

    for row_idx, row in enumerate(structure['rows']):
        for col_idx, cell in enumerate(row):
            cell['row'] = row_idx
            cell['col'] = col_idx

    return cells


def extract_table_data_matrix(table: Dict) -> List[List[str]]:
    """
    Extract table as 2D matrix of cell text values.

    Args:
        table: Table dict with 'cells'

    Returns:
        2D list of strings
    """
    cells = table.get('cells', [])
    if not cells:
        return []

    # Ensure cells have row/col indices
    if 'row' not in cells[0]:
        assign_row_col_indices(cells)

    # Find dimensions
    max_row = max(c.get('row', 0) for c in cells)
    max_col = max(c.get('col', 0) for c in cells)

    # Build matrix
    matrix = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for cell in cells:
        row = cell.get('row', 0)
        col = cell.get('col', 0)
        text = cell.get('text', '')
        matrix[row][col] = text

    return matrix


def get_cell_at_position(cells: List[Dict], row: int, col: int) -> Optional[Dict]:
    """
    Get cell at specific row/column position.

    Args:
        cells: List of cells with 'row' and 'col' fields
        row: Row index
        col: Column index

    Returns:
        Cell dict or None if not found
    """
    for cell in cells:
        if cell.get('row') == row and cell.get('col') == col:
            return cell
    return None
