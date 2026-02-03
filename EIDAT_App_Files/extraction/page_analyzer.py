"""
Page Analyzer - Detect headers, footers, and analyze text flow

Analyzes page structure to identify non-table content regions.
"""

import re
from typing import List, Dict, Tuple, Set, Optional


def detect_headers_footers(tokens: List[Dict], img_w: int, img_h: int,
                            top_threshold: float = 0.08,
                            bottom_threshold: float = 0.92) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect header and footer tokens based on page position.

    Args:
        tokens: List of OCR tokens
        img_w, img_h: Image dimensions
        top_threshold: Top N% of page for headers (default 8%)
        bottom_threshold: Bottom N% of page for footers (default 92%)

    Returns:
        Tuple of (header_tokens, footer_tokens)
    """
    headers = []
    footers = []

    top_y = img_h * top_threshold
    bottom_y = img_h * bottom_threshold

    for token in tokens:
        cy = token.get('cy', 0)

        if cy < top_y:
            headers.append(token)
        elif cy > bottom_y:
            footers.append(token)

    return headers, footers


def filter_table_tokens(tokens: List[Dict], tables: List[Dict],
                         overlap_threshold: float = 0.5) -> List[Dict]:
    """
    Remove tokens that belong to table regions.

    Args:
        tokens: All OCR tokens
        tables: List of detected tables with bbox_px
        overlap_threshold: Minimum overlap to consider token in table

    Returns:
        Tokens outside of table regions
    """
    non_table_tokens = []

    for token in tokens:
        tx0 = token.get('x0', 0)
        ty0 = token.get('y0', 0)
        tx1 = token.get('x1', 0)
        ty1 = token.get('y1', 0)
        token_area = max(1, (tx1 - tx0) * (ty1 - ty0))

        in_table = False
        for table in tables:
            table_bbox = table.get('bbox_px', [])
            if len(table_bbox) != 4:
                continue

            cx0, cy0, cx1, cy1 = table_bbox

            # Check overlap
            ox0 = max(tx0, cx0)
            oy0 = max(ty0, cy0)
            ox1 = min(tx1, cx1)
            oy1 = min(ty1, cy1)

            if ox0 < ox1 and oy0 < oy1:
                overlap_area = (ox1 - ox0) * (oy1 - oy0)
                if overlap_area / token_area >= overlap_threshold:
                    in_table = True
                    break

        if not in_table:
            non_table_tokens.append(token)

    return non_table_tokens


def group_tokens_into_lines(tokens: List[Dict], y_tolerance: float = 10) -> List[List[Dict]]:
    """
    Group tokens into lines based on y-position.

    Args:
        tokens: List of tokens
        y_tolerance: Max y-distance to be considered same line

    Returns:
        List of token groups (lines)
    """
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: (t.get('cy', 0), t.get('cx', 0)))

    lines = []
    current_line = [sorted_tokens[0]]
    line_y = sorted_tokens[0].get('cy', 0)
    line_h = max(1.0, float(sorted_tokens[0].get('y1', 0)) - float(sorted_tokens[0].get('y0', 0)))

    for token in sorted_tokens[1:]:
        token_y = token.get('cy', 0)
        token_h = max(1.0, float(token.get('y1', 0)) - float(token.get('y0', 0)))
        tol = max(float(y_tolerance), 0.35 * max(line_h, token_h))

        if abs(token_y - line_y) <= tol:
            current_line.append(token)
            # Update running line height average to keep tolerance stable.
            line_h = (line_h * (len(current_line) - 1) + token_h) / float(len(current_line))
        else:
            lines.append(sorted(current_line, key=lambda t: t.get('cx', 0)))
            current_line = [token]
            line_y = token_y
            line_h = token_h

    if current_line:
        lines.append(sorted(current_line, key=lambda t: t.get('cx', 0)))

    return lines


def _merge_overlapping_lines(
    lines: List[List[Dict]],
    overlap_ratio: float = 0.6,
    height_ratio_max: float = 1.25
) -> List[List[Dict]]:
    """
    Merge lines that overlap heavily in vertical space.

    This helps when large-font headers have slightly different baselines and
    are incorrectly split into separate lines.
    """
    if not lines:
        return []

    def _bounds(line: List[Dict]) -> Optional[Dict]:
        return _line_bounds(line)

    def _ratio(a: Dict, b: Dict) -> float:
        overlap = min(a['y1'], b['y1']) - max(a['y0'], b['y0'])
        if overlap <= 0:
            return 0.0
        min_h = max(1.0, min(a.get('h', 0.0), b.get('h', 0.0)))
        return overlap / min_h

    lines_sorted = sorted(
        lines,
        key=lambda l: (_bounds(l) or {}).get('y0', 0.0)
    )
    merged: List[List[Dict]] = []
    for line in lines_sorted:
        if not merged:
            merged.append(line)
            continue
        last = merged[-1]
        last_bounds = _bounds(last)
        curr_bounds = _bounds(line)
        if not last_bounds or not curr_bounds:
            merged.append(line)
            continue
        height_ratio = max(last_bounds.get('h', 1.0), curr_bounds.get('h', 1.0)) / max(
            1.0, min(last_bounds.get('h', 1.0), curr_bounds.get('h', 1.0))
        )
        if _ratio(last_bounds, curr_bounds) >= overlap_ratio and height_ratio <= height_ratio_max:
            merged_line = last + line
            merged_line = sorted(merged_line, key=lambda t: t.get('cx', t.get('x0', 0)))
            merged[-1] = merged_line
        else:
            merged.append(line)

    return merged


def _split_line_on_large_gap(
    line: List[Dict],
    img_w: int,
    *,
    min_gap_ratio: float = 0.2,
    min_gap_px: float = 120.0
) -> List[List[Dict]]:
    """
    Split a line into multiple segments if there is a very large horizontal gap.

    This is useful for headers with left/right columns on the same baseline.
    """
    if not line:
        return []

    tokens = sorted(line, key=lambda t: t.get('x0', 0))
    if len(tokens) < 2:
        return [tokens]

    gaps = []
    for i in range(1, len(tokens)):
        gap = float(tokens[i].get('x0', 0)) - float(tokens[i - 1].get('x1', 0))
        gaps.append((gap, i))

    max_gap, split_idx = max(gaps, key=lambda g: g[0])
    threshold = max(float(min_gap_px), float(img_w) * float(min_gap_ratio))
    if max_gap >= threshold:
        return [tokens[:split_idx], tokens[split_idx:]]

    return [tokens]


def _split_lines_on_large_gaps(
    lines: List[List[Dict]],
    img_w: int,
    *,
    min_gap_ratio: float = 0.2,
    min_gap_px: float = 120.0
) -> List[List[Dict]]:
    if not lines:
        return []
    out: List[List[Dict]] = []
    for line in lines:
        out.extend(_split_line_on_large_gap(
            line,
            img_w,
            min_gap_ratio=min_gap_ratio,
            min_gap_px=min_gap_px,
        ))
    return out


def _split_lines_on_double_space(lines: List[List[Dict]]) -> List[List[Dict]]:
    """
    Split lines when horizontal gaps exceed a "double-space" threshold.

    Uses per-line gap statistics and token height to avoid splitting
    normal word spacing.
    """
    if not lines:
        return []

    out: List[List[Dict]] = []
    for line in lines:
        tokens = sorted(line, key=lambda t: t.get('x0', 0))
        if len(tokens) < 2:
            out.append(tokens)
            continue

        heights = [
            max(1.0, float(t.get('y1', 0)) - float(t.get('y0', 0)))
            for t in tokens
        ]
        median_h = _median(heights, default=12.0)

        gaps = []
        for i in range(1, len(tokens)):
            gap = float(tokens[i].get('x0', 0)) - float(tokens[i - 1].get('x1', 0))
            if gap > 0:
                gaps.append(gap)

        if not gaps:
            out.append(tokens)
            continue

        gaps_sorted = sorted(gaps)
        median_gap = gaps_sorted[len(gaps_sorted) // 2]
        # Double-space threshold: larger than typical gap and roughly >= token height.
        threshold = max(median_h * 0.8, median_gap * 2.2)

        split_indices = []
        for i in range(1, len(tokens)):
            gap = float(tokens[i].get('x0', 0)) - float(tokens[i - 1].get('x1', 0))
            if gap < threshold:
                continue
            left_h = max(
                1.0,
                _median([
                    max(1.0, float(t.get('y1', 0)) - float(t.get('y0', 0)))
                    for t in tokens[:i]
                ], default=median_h)
            )
            right_h = max(
                1.0,
                _median([
                    max(1.0, float(t.get('y1', 0)) - float(t.get('y0', 0)))
                    for t in tokens[i:]
                ], default=median_h)
            )
            if 0.75 <= (left_h / right_h) <= 1.33:
                split_indices.append(i)

        if not split_indices:
            out.append(tokens)
            continue

        start = 0
        for idx in split_indices:
            if idx > start:
                out.append(tokens[start:idx])
            start = idx
        if start < len(tokens):
            out.append(tokens[start:])

    return out


def _split_lines_by_height_bands(
    lines: List[List[Dict]],
    *,
    min_height_ratio: float = 1.35
) -> List[List[Dict]]:
    """
    Split a line into multiple lines when tokens form distinct height bands.

    This helps separate title/subtitle pairs when OCR places them on one baseline.
    """
    if not lines:
        return []

    out: List[List[Dict]] = []

    def _upper_ratio(text: str) -> float:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0
        upp = sum(1 for ch in letters if ch.isupper())
        return float(upp) / float(len(letters))

    for line in lines:
        if len(line) < 3:
            out.append(line)
            continue

        heights = [
            max(1.0, float(t.get('y1', 0)) - float(t.get('y0', 0)))
            for t in line
        ]
        median_h = _median(heights, default=12.0)
        max_h = max(heights)
        min_h = max(1.0, min(heights))

        cy_vals = [float(t.get('cy', t.get('y0', 0))) for t in line]
        y_spread = max(cy_vals) - min(cy_vals) if cy_vals else 0.0

        if y_spread < (0.45 * median_h) or (max_h / min_h) < min_height_ratio:
            out.append(line)
            continue

        # Re-cluster with a tighter tolerance to split stacked lines.
        tol = max(3.0, 0.30 * median_h)
        tokens_sorted = sorted(line, key=lambda t: (t.get('cy', 0), t.get('x0', 0)))
        groups: List[List[Dict]] = []
        current = [tokens_sorted[0]]
        current_y = float(tokens_sorted[0].get('cy', tokens_sorted[0].get('y0', 0)))
        for tok in tokens_sorted[1:]:
            cy = float(tok.get('cy', tok.get('y0', 0)))
            if abs(cy - current_y) <= tol:
                current.append(tok)
            else:
                groups.append(sorted(current, key=lambda t: t.get('x0', 0)))
                current = [tok]
                current_y = cy
        if current:
            groups.append(sorted(current, key=lambda t: t.get('x0', 0)))

        if len(groups) < 2:
            out.append(line)
            continue

        group_heights = [
            _median([
                max(1.0, float(t.get('y1', 0)) - float(t.get('y0', 0)))
                for t in g
            ], default=median_h)
            for g in groups
        ]
        height_ratio = max(group_heights) / max(1.0, min(group_heights))
        group_texts = [" ".join(t.get('text', '') for t in g) for g in groups]
        upper_ratios = [_upper_ratio(txt) for txt in group_texts]

        # Split when heights differ, or when casing strongly differs.
        if height_ratio >= min_height_ratio or (
            len(upper_ratios) >= 2 and max(upper_ratios) >= 0.75 and min(upper_ratios) <= 0.55
        ):
            out.extend(groups)
        else:
            out.append(line)

    return out


def group_lines_into_paragraphs(lines: List[List[Dict]],
                                 gap_threshold: Optional[float] = None) -> List[List[List[Dict]]]:
    """
    Group lines into paragraphs based on vertical gaps.

    Args:
        lines: List of token lines
        gap_threshold: Min gap to split paragraphs

    Returns:
        List of paragraphs (each is a list of lines)
    """
    if not lines:
        return []

    if gap_threshold is None:
        heights = []
        for line in lines:
            bounds = _line_bounds(line)
            if bounds:
                heights.append(bounds['h'])
        median_h = _median(heights, default=12.0)
        gap_threshold = max(30.0, median_h * 1.6)

    paragraphs = []
    current_para = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = lines[i-1]
        curr_line = lines[i]

        # Get bottom of previous line and top of current line
        prev_y = max(t.get('y1', 0) for t in prev_line) if prev_line else 0
        curr_y = min(t.get('y0', 0) for t in curr_line) if curr_line else 0

        gap = curr_y - prev_y

        if gap > gap_threshold:
            # Start new paragraph
            paragraphs.append(current_para)
            current_para = [curr_line]
        else:
            current_para.append(curr_line)

    if current_para:
        paragraphs.append(current_para)

    return paragraphs


def _median(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(values[mid - 1] + values[mid]) / 2.0


def _line_bounds(line: List[Dict]) -> Optional[Dict]:
    if not line:
        return None
    x0 = min(t.get('x0', 0) for t in line)
    y0 = min(t.get('y0', 0) for t in line)
    x1 = max(t.get('x1', 0) for t in line)
    y1 = max(t.get('y1', 0) for t in line)
    h = max(1.0, float(y1) - float(y0))
    return {
        'x0': float(x0),
        'y0': float(y0),
        'x1': float(x1),
        'y1': float(y1),
        'h': h
    }


def detect_table_titles(lines: List[List[Dict]], tables: List[Dict],
                        img_h: int) -> List[Dict]:
    """
    Detect likely chart/table title lines above table bounds.

    Returns list of dicts with:
        - table_idx (1-based, or None if unbound keyword title)
        - text
        - bbox_px
        - lines (list of line token lists)
    """
    if not lines:
        return []

    line_infos = []
    heights = []
    for idx, line in enumerate(lines):
        text = extract_line_text(line).strip()
        if not text:
            continue
        bounds = _line_bounds(line)
        if not bounds:
            continue
        heights.append(bounds['h'])
        line_infos.append({
            'idx': idx,
            'line': line,
            'text': text,
            **bounds
        })

    if not line_infos:
        return []

    median_h = _median(heights, default=12.0)
    title_re = re.compile(r"^(table|figure|fig\.?|chart|graph)\b", re.IGNORECASE)
    titles = []
    used_lines: Set[int] = set()

    for table_idx, table in enumerate(tables or []):
        bbox = table.get('bbox_px', [])
        if len(bbox) != 4:
            continue
        tx0, ty0, tx1, ty1 = (float(v) for v in bbox)
        table_h = max(1.0, ty1 - ty0)
        table_w = max(1.0, tx1 - tx0)
        max_gap = max(median_h * 2.5, min(140.0, table_h * 0.15), 30.0)

        best = None
        for info in line_infos:
            if info['idx'] in used_lines:
                continue
            gap = ty0 - info['y1']
            if gap < (-0.2 * median_h) or gap > max_gap:
                continue
            overlap = max(0.0, min(info['x1'], tx1) - max(info['x0'], tx0))
            if overlap <= 0.0:
                continue
            line_w = max(1.0, info['x1'] - info['x0'])
            overlap_ratio = max(overlap / line_w, overlap / table_w)
            if overlap_ratio < 0.2:
                continue
            keyword = bool(title_re.search(info['text']))
            size_boost = 0.0
            if median_h > 0:
                size_boost = max(0.0, min(0.5, (info['h'] / median_h) - 1.0))
            score = (1.0 - (gap / max_gap)) + min(0.4, overlap_ratio) + (0.6 if keyword else 0.0) + size_boost
            if best is None or score > best['score']:
                best = dict(info)
                best['score'] = score
                best['gap'] = gap

        if not best:
            continue

        title_lines = [best]
        used_lines.add(best['idx'])

        # Include an additional line directly above if tightly stacked.
        above = [info for info in line_infos
                 if info['idx'] not in used_lines and info['y1'] <= best['y0']]
        above.sort(key=lambda i: i['y1'], reverse=True)
        max_above_gap = max(median_h * 0.8, 18.0)
        for info in above:
            if len(title_lines) >= 2:
                break
            gap = best['y0'] - info['y1']
            if gap < 0 or gap > max_above_gap:
                continue
            above_text = info['text']
            above_words = len(above_text.split())
            above_keyword = bool(title_re.search(above_text))
            if above_words > 8 and not above_keyword:
                continue
            if median_h > 0:
                if info['h'] > best['h'] * 1.2 or info['h'] < best['h'] * 0.7:
                    continue
            overlap = max(0.0, min(info['x1'], tx1) - max(info['x0'], tx0))
            if overlap <= 0.0:
                continue
            line_w = max(1.0, info['x1'] - info['x0'])
            overlap_ratio = max(overlap / line_w, overlap / table_w)
            if overlap_ratio < 0.2:
                continue
            title_lines.append(info)
            used_lines.add(info['idx'])
            break

        title_lines.sort(key=lambda i: i['y0'])
        title_text = "\n".join(i['text'] for i in title_lines)
        title_bbox = _line_bounds([t for info in title_lines for t in info['line']]) or {}

        titles.append({
            'table_idx': table_idx + 1,
            'text': title_text,
            'bbox_px': [title_bbox.get('x0', 0.0), title_bbox.get('y0', 0.0),
                        title_bbox.get('x1', 0.0), title_bbox.get('y1', 0.0)],
            'lines': [info['line'] for info in title_lines]
        })

    # Add unbound keyword titles (likely chart titles without table bounds).
    for info in line_infos:
        if info['idx'] in used_lines:
            continue
        if title_re.match(info['text'].strip().lower()):
            used_lines.add(info['idx'])
            titles.append({
                'table_idx': None,
                'text': info['text'],
                'bbox_px': [info['x0'], info['y0'], info['x1'], info['y1']],
                'lines': [info['line']]
            })

    return titles


def split_lines_and_paragraphs(
    lines: List[List[Dict]],
    *,
    min_words_for_paragraph: int = 6,
    big_line_ratio: float = 1.2,
    title_height_ratio: float = 1.35
) -> Tuple[List[List[Dict]], List[List[List[Dict]]]]:
    """
    Split lines into standalone lines and paragraph groups.

    Single-line paragraphs are treated as standalone lines if they are short
    or visually large (headline-like).
    """
    if not lines:
        return [], []

    heights = []
    for line in lines:
        bounds = _line_bounds(line)
        if bounds:
            heights.append(bounds['h'])
    median_h = _median(heights, default=12.0)

    paragraphs_raw = group_lines_into_paragraphs(lines)
    standalone_lines: List[List[Dict]] = []
    paragraphs: List[List[List[Dict]]] = []

    for paragraph in paragraphs_raw:
        if not paragraph:
            continue
        line_texts = [extract_line_text(line).strip() for line in paragraph]
        line_texts = [t for t in line_texts if t]
        if not line_texts:
            continue

        if len(paragraph) > 1:
            heights = []
            total_words = 0
            max_words = 0
            for line, text in zip(paragraph, line_texts):
                bounds = _line_bounds(line) or {}
                heights.append(float(bounds.get('h', median_h)))
                words = len(text.split())
                total_words += words
                max_words = max(max_words, words)
            avg_h = sum(heights) / max(1, len(heights))
            if max_words <= min_words_for_paragraph and total_words <= (min_words_for_paragraph * len(paragraph)):
                standalone_lines.extend(paragraph)
                continue
            if avg_h >= (median_h * big_line_ratio) and total_words <= (min_words_for_paragraph * len(paragraph)):
                standalone_lines.extend(paragraph)
                continue
            # Split title/subtitle pairs into standalone lines when height differs.
            if max(heights) / max(1.0, min(heights)) >= title_height_ratio:
                standalone_lines.extend(paragraph)
                continue

        if len(paragraph) == 1:
            line = paragraph[0]
            text = line_texts[0]
            word_count = len(text.split())
            bounds = _line_bounds(line) or {}
            line_h = float(bounds.get('h', median_h))
            big_line = line_h >= (median_h * big_line_ratio) if median_h > 0 else False

            if word_count <= min_words_for_paragraph or big_line:
                standalone_lines.append(line)
                continue

        paragraphs.append(paragraph)

    return standalone_lines, paragraphs


def extract_flow_text(tokens: List[Dict], tables: List[Dict],
                      img_w: int, img_h: int) -> Dict:
    """
    Extract structured text flow (headers, footers, paragraphs).

    Args:
        tokens: All OCR tokens
        tables: Detected tables
        img_w, img_h: Image dimensions

    Returns:
        Dict with 'headers', 'footers', 'paragraphs', 'body_tokens'
    """
    # Filter out table tokens
    non_table = filter_table_tokens(tokens, tables)

    # Detect header/footer candidates from non-table tokens
    headers, footers = detect_headers_footers(non_table, img_w, img_h)

    # Remove headers and footers from body
    header_set = set(id(t) for t in headers)
    footer_set = set(id(t) for t in footers)
    body_tokens = [t for t in non_table
                   if id(t) not in header_set and id(t) not in footer_set]

    # Group into lines and paragraphs
    lines = group_tokens_into_lines(body_tokens)
    lines = _split_lines_by_height_bands(lines)
    lines = _merge_overlapping_lines(lines)
    lines = _split_lines_on_double_space(lines)
    titles = detect_table_titles(lines, tables, img_h)
    title_line_ids = set(id(line) for t in titles for line in t.get('lines', []))
    lines = [line for line in lines if id(line) not in title_line_ids]
    standalone_lines, paragraphs = split_lines_and_paragraphs(lines)
    header_lines = group_tokens_into_lines(headers)
    header_lines = _merge_overlapping_lines(header_lines)
    header_lines = _split_lines_on_double_space(header_lines)
    header_lines = _split_lines_on_large_gaps(header_lines, img_w)
    footer_lines = group_tokens_into_lines(footers)
    footer_lines = _split_lines_on_double_space(footer_lines)

    return {
        'headers': headers,
        'footers': footers,
        'non_table_tokens': non_table,
        'body_tokens': body_tokens,
        'header_lines': header_lines,
        'footer_lines': footer_lines,
        'lines': standalone_lines,
        'paragraphs': paragraphs,
        'table_titles': titles
    }


def extract_text_from_tokens(tokens: List[Dict]) -> str:
    """Simple concatenation of token text with spaces."""
    return ' '.join(t.get('text', '') for t in tokens if t.get('text', '').strip())


def extract_line_text(line: List[Dict]) -> str:
    """Extract text from a line of tokens."""
    return ' '.join(t.get('text', '') for t in line if t.get('text', '').strip())


def extract_paragraph_text(paragraph: List[List[Dict]]) -> str:
    """Extract text from a paragraph (list of lines)."""
    return '\n'.join(extract_line_text(line) for line in paragraph)
