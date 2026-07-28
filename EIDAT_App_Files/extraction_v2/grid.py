"""Bordered-table grid detection with OpenCV morphology.

Works directly on the page bitmap, so it is identical for scanned and
digital pages. Side-by-side tables come out as separate connected
components, so they can never smear into each other (a classic failure
of row-based ASCII reconstruction).

Steps
1. binarize (ink=255)
2. long-horizontal / long-vertical morphological opens -> line masks
3. connected components over (h | v) -> candidate table regions
4. inside each region, cluster line positions -> row/col boundaries
5. boundaries -> cell rectangles

Output is purely geometric: TableGrid(bbox, row_bounds, col_bounds, cells).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Cell:
    r: int
    c: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class TableGrid:
    x0: int
    y0: int
    x1: int
    y1: int
    row_bounds: list[int]
    col_bounds: list[int]
    cells: list[Cell]

    @property
    def n_rows(self) -> int:
        return len(self.row_bounds) - 1

    @property
    def n_cols(self) -> int:
        return len(self.col_bounds) - 1


def _cluster_positions(positions: np.ndarray, tol: int) -> list[int]:
    """Cluster sorted 1-D coordinates; return one representative per cluster."""
    if positions.size == 0:
        return []
    positions = np.sort(positions)
    groups: list[list[int]] = [[int(positions[0])]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= tol:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    return [int(np.mean(g)) for g in groups]


def detect_tables(gray: np.ndarray, dpi: int = 300) -> list[TableGrid]:
    h, w = gray.shape
    # ink = white on black
    binv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    min_h_line = max(40, int(w * 0.06))
    min_v_line = max(40, int(dpi * 0.12))  # ~1/8 inch of vertical rule

    horiz = cv2.morphologyEx(
        binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_line, 1))
    )
    vert = cv2.morphologyEx(
        binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_line))
    )

    mask = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    tol = max(3, dpi // 60)  # line-position clustering tolerance (~5px @300dpi)
    tables: list[TableGrid] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < w * 0.15 or bh < dpi * 0.15:  # too small to be a table
            continue
        comp = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8) * 255
        comp_h = cv2.bitwise_and(comp, horiz[y : y + bh, x : x + bw])
        comp_v = cv2.bitwise_and(comp, vert[y : y + bh, x : x + bw])

        # row boundaries: y positions where a long horizontal run exists
        row_hits = np.where(comp_h.sum(axis=1) / 255 > bw * 0.5)[0]
        col_hits = np.where(comp_v.sum(axis=0) / 255 > bh * 0.5)[0]
        row_bounds = _cluster_positions(row_hits, tol)
        col_bounds = _cluster_positions(col_hits, tol)
        if len(row_bounds) < 3 or len(col_bounds) < 2:
            continue  # a lone rule line / underline, not a grid

        cells = [
            Cell(
                r,
                c,
                x + col_bounds[c],
                y + row_bounds[r],
                x + col_bounds[c + 1],
                y + row_bounds[r + 1],
            )
            for r in range(len(row_bounds) - 1)
            for c in range(len(col_bounds) - 1)
        ]
        tables.append(
            TableGrid(
                x0=int(x),
                y0=int(y),
                x1=int(x + bw),
                y1=int(y + bh),
                row_bounds=[int(x0 + y) for x0 in row_bounds],
                col_bounds=[int(c0 + x) for c0 in col_bounds],
                cells=cells,
            )
        )
    tables.sort(key=lambda t: (t.y0, t.x0))
    return tables
