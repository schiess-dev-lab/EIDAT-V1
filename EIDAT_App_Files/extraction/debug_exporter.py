"""
Debug Exporter - Export extraction results to JSON format

Creates compatible debug output matching existing format.
"""

import json
import math
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


def _render_table_ascii(table: Dict) -> str:
    """Render a table as ASCII text with borders."""
    cells = table.get('cells', [])
    if not cells:
        return ""

    # Organize cells into rows
    rows_dict = {}
    for cell in cells:
        row_idx = cell.get('row', 0)
        if row_idx not in rows_dict:
            rows_dict[row_idx] = []
        rows_dict[row_idx].append(cell)

    if not rows_dict:
        return ""

    # Sort rows and cells within rows
    sorted_rows = []
    for row_idx in sorted(rows_dict.keys()):
        row_cells = sorted(rows_dict[row_idx], key=lambda c: c.get('col', 0))
        sorted_rows.append(row_cells)

    # Calculate column widths
    max_cols = max(len(row) for row in sorted_rows) if sorted_rows else 0
    col_widths = [8] * max_cols  # minimum width

    for row in sorted_rows:
        for i, cell in enumerate(row):
            if i < max_cols:
                text = str(cell.get('text', ''))
                col_widths[i] = max(col_widths[i], min(40, len(text) + 2))

    # Build ASCII table
    lines = []
    separator = '+' + '+'.join('-' * w for w in col_widths) + '+'
    lines.append(separator)

    for row in sorted_rows:
        row_text = []
        for i in range(max_cols):
            if i < len(row):
                text = str(row[i].get('text', ''))[:col_widths[i]-2]
            else:
                text = ''
            row_text.append(f" {text.ljust(col_widths[i]-2)} ")
        lines.append('|' + '|'.join(row_text) + '|')
        lines.append(separator)

    return '\n'.join(lines)


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
        return page_analyzer.extract_line_text(line).strip()

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
    for page_idx, page_data in enumerate(pages_data):
        page_num = page_data.get('page', 0)
        tokens = page_data.get('tokens', [])
        tables = page_data.get('tables', [])
        flow = page_data.get('flow') or {}
        charts = page_data.get('charts', []) or []
        chart_bboxes = [c.get("bbox_px") for c in charts if c.get("bbox_px")]

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
                _add_table_artifact(i, table)

            artifacts.sort(key=lambda a: (a['y0'], a['x0'], a['order']))

            for art in artifacts:
                combined_text.append(f"[{art['category']}]\n")
                if art['category'] == "Table":
                    table_idx = int(art.get('table_idx', 0))
                    combined_text.append(f"\n[Table {table_idx + 1}]\n")
                    table_ascii = _render_table_ascii(tables[table_idx])
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
