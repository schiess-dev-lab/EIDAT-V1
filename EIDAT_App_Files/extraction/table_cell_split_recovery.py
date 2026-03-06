"""
Table Cell Split Recovery - Evidence-based splitting of merged/oversized cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from . import page_analyzer

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(key, str(default)) or str(default)).strip()))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(str(os.environ.get(key, str(default)) or str(default)).strip())
    except Exception:
        return float(default)


def _env_bool(key: str, default: bool) -> bool:
    raw = str(os.environ.get(key, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class _Cfg:
    enabled: bool

    ev_multirow: bool
    ev_whitespace: bool
    ev_token_overlap: bool
    ev_token_gap: bool

    # Grid snapping (edges inference)
    snap_tol_px: float
    snap_ratio: float
    snap_max_px: float

    # Multi-row evidence
    multirow_max_look: int
    multirow_min_run: int
    multirow_require_adjacent: bool

    # Whitespace corridor evidence
    ws_half_w: int
    ws_tb_pad: int
    ws_row_ink_max: int
    ws_min_clear_frac: float
    ws_required: bool

    # Token evidence
    boundary_margin_px: float
    min_gap_px: float
    line_support_ratio: float

    # Acceptance
    min_nonempty_subcells: int
    min_text_chars: int
    min_tokens: int
    require_all_subcells_meaningful: bool

    # Debug
    debug: bool
    debug_save_overlays: bool
    debug_max_tables: int


def _load_cfg() -> _Cfg:
    return _Cfg(
        enabled=_env_bool("EIDAT_TABLE_CELL_SPLIT_RECOVERY", True),
        ev_multirow=_env_bool("EIDAT_TABLE_CELL_SPLIT_EVIDENCE_MULTIROW", True),
        ev_whitespace=_env_bool("EIDAT_TABLE_CELL_SPLIT_EVIDENCE_WHITESPACE", True),
        ev_token_overlap=_env_bool("EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_OVERLAP", True),
        ev_token_gap=_env_bool("EIDAT_TABLE_CELL_SPLIT_EVIDENCE_TOKEN_GAP", True),
        snap_tol_px=_env_float("EIDAT_TABLE_CELL_GRID_SNAP_TOL_PX", 0.0),
        snap_ratio=_env_float("EIDAT_TABLE_CELL_GRID_SNAP_TOL_RATIO", 0.006),
        snap_max_px=_env_float("EIDAT_TABLE_CELL_GRID_SNAP_MAX_PX", 15.0),
        multirow_max_look=_env_int("EIDAT_TABLE_CELL_SPLIT_MULTIROW_MAX_LOOK", 3),
        multirow_min_run=_env_int("EIDAT_TABLE_CELL_SPLIT_MULTIROW_MIN_RUN", 2),
        multirow_require_adjacent=_env_bool("EIDAT_TABLE_CELL_SPLIT_MULTIROW_REQUIRE_ADJACENT", True),
        ws_half_w=_env_int("EIDAT_TABLE_CELL_SPLIT_WS_STRIP_HALF_WIDTH_PX", 3),
        ws_tb_pad=_env_int("EIDAT_TABLE_CELL_SPLIT_WS_TB_PAD_PX", 6),
        ws_row_ink_max=_env_int("EIDAT_TABLE_CELL_SPLIT_WS_ROW_INK_PX_MAX", 1),
        ws_min_clear_frac=_env_float("EIDAT_TABLE_CELL_SPLIT_WS_MIN_CLEAR_FRAC", 0.85),
        ws_required=_env_bool("EIDAT_TABLE_CELL_SPLIT_WS_REQUIRED", False),
        boundary_margin_px=_env_float("EIDAT_TABLE_CELL_SPLIT_BOUNDARY_MARGIN_PX", 2.0),
        min_gap_px=_env_float("EIDAT_TABLE_CELL_SPLIT_MIN_GAP_PX", 8.0),
        line_support_ratio=_env_float("EIDAT_TABLE_CELL_SPLIT_LINE_SUPPORT_RATIO", 0.7),
        min_nonempty_subcells=_env_int("EIDAT_TABLE_CELL_SPLIT_MIN_NONEMPTY_SUBCELLS", 2),
        min_text_chars=_env_int("EIDAT_TABLE_CELL_SPLIT_MIN_TEXT_CHARS", 2),
        min_tokens=_env_int("EIDAT_TABLE_CELL_SPLIT_MIN_TOKENS", 1),
        require_all_subcells_meaningful=_env_bool(
            "EIDAT_TABLE_CELL_SPLIT_REQUIRE_ALL_SUBCELLS_MEANINGFUL",
            False,
        ),
        debug=_env_bool("EIDAT_TABLE_CELL_SPLIT_DEBUG", False),
        debug_save_overlays=_env_bool("EIDAT_TABLE_CELL_SPLIT_DEBUG_SAVE_OVERLAYS", False),
        debug_max_tables=_env_int("EIDAT_TABLE_CELL_SPLIT_DEBUG_MAX_TABLES", 0),
    )


def _merge_positions_1d(values: List[float], tol_px: float) -> List[float]:
    if not values:
        return []
    tol = max(0.0, float(tol_px))
    vals = sorted(float(v) for v in values)
    if tol <= 0:
        out: List[float] = []
        last = None
        for v in vals:
            key = int(round(v))
            if last is None or key != last:
                out.append(float(key))
                last = key
        return out

    clusters: List[List[float]] = []
    cur = [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    return [sum(c) / float(len(c)) for c in clusters if c]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _closest_line_idx(lines: List[float], v: float) -> int:
    best_i = 0
    best_d = abs(float(lines[0]) - float(v))
    for i, lv in enumerate(lines[1:], start=1):
        d = abs(float(lv) - float(v))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _bbox_from_cells(cells: List[Dict]) -> Optional[List[float]]:
    xs0: List[float] = []
    ys0: List[float] = []
    xs1: List[float] = []
    ys1: List[float] = []
    for c in cells or []:
        bbox = c.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    if not xs0:
        return None
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def infer_grid_edges_from_cells(
    table_bbox: List[float],
    cells: List[Dict],
    tol_x: float,
    tol_y: float,
    *,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    x0, y0, x1, y1 = (float(v) for v in table_bbox)
    if img_w is not None and img_w > 0:
        x0 = _clamp(x0, 0.0, float(img_w))
        x1 = _clamp(x1, 0.0, float(img_w))
    if img_h is not None and img_h > 0:
        y0 = _clamp(y0, 0.0, float(img_h))
        y1 = _clamp(y1, 0.0, float(img_h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    xs: List[float] = [x0, x1]
    ys: List[float] = [y0, y1]
    for c in cells or []:
        bbox = c.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            cx0, cy0, cx1, cy1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            continue
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        xs.extend([cx0, cx1])
        ys.extend([cy0, cy1])

    x_lines = _merge_positions_1d(xs, tol_px=float(tol_x))
    y_lines = _merge_positions_1d(ys, tol_px=float(tol_y))

    x_lines = [float(_clamp(v, x0, x1)) for v in x_lines]
    y_lines = [float(_clamp(v, y0, y1)) for v in y_lines]

    x_ints = sorted({int(round(v)) for v in x_lines})
    y_ints = sorted({int(round(v)) for v in y_lines})
    if len(x_ints) < 2 or len(y_ints) < 2:
        return [], []

    # Force endpoints to match the outer bbox.
    x_ints[0] = int(round(x0))
    x_ints[-1] = int(round(x1))
    y_ints[0] = int(round(y0))
    y_ints[-1] = int(round(y1))

    out_x: List[float] = []
    for v in x_ints:
        if not out_x or float(v) > out_x[-1]:
            out_x.append(float(v))
    out_y: List[float] = []
    for v in y_ints:
        if not out_y or float(v) > out_y[-1]:
            out_y.append(float(v))

    if len(out_x) < 2 or len(out_y) < 2:
        return [], []
    return out_x, out_y


def grid_span(cell_bbox: List[float], edges_x: List[float], edges_y: List[float]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in cell_bbox)
    xi0 = _closest_line_idx(edges_x, x0)
    xi1 = _closest_line_idx(edges_x, x1)
    yi0 = _closest_line_idx(edges_y, y0)
    yi1 = _closest_line_idx(edges_y, y1)
    if xi1 < xi0:
        xi0, xi1 = xi1, xi0
    if yi1 < yi0:
        yi0, yi1 = yi1, yi0
    if xi0 == xi1:
        xi1 = min(len(edges_x) - 1, xi0 + 1)
    if yi0 == yi1:
        yi1 = min(len(edges_y) - 1, yi0 + 1)
    c0 = max(0, min(len(edges_x) - 2, int(xi0)))
    c1 = max(1, min(len(edges_x) - 1, int(xi1)))
    r0 = max(0, min(len(edges_y) - 2, int(yi0)))
    r1 = max(1, min(len(edges_y) - 1, int(yi1)))
    if c1 <= c0:
        c1 = min(len(edges_x) - 1, c0 + 1)
    if r1 <= r0:
        r1 = min(len(edges_y) - 1, r0 + 1)
    return (int(r0), int(r1), int(c0), int(c1))


def assign_row_col_indices_grid(cells: List[Dict], edges_x: List[float], edges_y: List[float]) -> None:
    if not cells or not edges_x or not edges_y:
        return
    for cell in cells:
        bbox = cell.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            r0, _r1, c0, _c1 = grid_span(list(bbox), edges_x, edges_y)
        except Exception:
            continue
        cell["row"] = int(r0)
        cell["col"] = int(c0)


def _token_center(token: Dict) -> Tuple[float, float]:
    try:
        return float(token.get("cx")), float(token.get("cy"))
    except Exception:
        x0 = float(token.get("x0", 0.0))
        y0 = float(token.get("y0", 0.0))
        x1 = float(token.get("x1", x0))
        y1 = float(token.get("y1", y0))
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _center_in_bbox(token: Dict, bbox: List[float]) -> bool:
    cx, cy = _token_center(token)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _tokens_to_text(tokens: List[Dict]) -> str:
    toks = [t for t in (tokens or []) if str(t.get("text", "")).strip()]
    if not toks:
        return ""

    def _key(t: Dict) -> Tuple[float, float]:
        cy = t.get("cy")
        if cy is None:
            y0 = float(t.get("y0", 0.0))
            y1 = float(t.get("y1", y0))
            cy = (y0 + y1) / 2.0
        return float(cy), float(t.get("x0", 0.0))

    ordered = sorted(toks, key=_key)
    return " ".join(str(t.get("text", "")).strip() for t in ordered).strip()


def _cell_meaningful(cell: Dict, *, min_chars: int, min_tokens: int) -> bool:
    txt = str(cell.get("text", "") or "").strip()
    if txt and len(txt) >= int(min_chars):
        return True
    toks = cell.get("tokens") or []
    if isinstance(toks, list):
        non_empty = [t for t in toks if str(t.get("text", "")).strip()]
        if len(non_empty) >= int(min_tokens):
            return True
    return False


def _boundary_present(grid: List[List[Optional[int]]], row: int, b: int) -> bool:
    if row < 0 or row >= len(grid):
        return False
    if not grid[row]:
        return False
    ncols = len(grid[row])
    if b <= 0 or b >= ncols:
        return False
    left = grid[row][b - 1]
    right = grid[row][b]
    return left is not None and right is not None and left != right


def _median(values: List[float], *, default: float) -> float:
    if not values:
        return float(default)
    vals = sorted(float(v) for v in values)
    return float(vals[len(vals) // 2])


def _multirow_support_ok(
    grid: List[List[Optional[int]]],
    *,
    r0: int,
    b: int,
    max_look: int,
    min_run: int,
    require_adjacent: bool,
) -> Tuple[bool, Dict[str, int]]:
    nrows = len(grid)
    max_look = max(0, int(max_look))
    min_run = max(1, int(min_run))
    rows_avail_down = max(0, min(max_look, (nrows - 1) - int(r0)))
    rows_avail_up = max(0, min(max_look, int(r0)))
    available_max = max(rows_avail_up, rows_avail_down)
    min_run_eff = min(min_run, max(1, available_max)) if available_max > 0 else 1

    run_down = 0
    for rr in range(int(r0) + 1, min(nrows, int(r0) + 1 + max_look)):
        if _boundary_present(grid, rr, int(b)):
            run_down += 1
        else:
            break

    run_up = 0
    for rr in range(int(r0) - 1, max(-1, int(r0) - 1 - max_look), -1):
        if _boundary_present(grid, rr, int(b)):
            run_up += 1
        else:
            break

    ok = False
    if require_adjacent:
        ok_down = (
            rows_avail_down > 0
            and _boundary_present(grid, int(r0) + 1, int(b))
            and run_down >= min_run_eff
        )
        ok_up = (
            rows_avail_up > 0
            and _boundary_present(grid, int(r0) - 1, int(b))
            and run_up >= min_run_eff
        )
        ok = bool(ok_down or ok_up)
    else:
        ok = max(run_up, run_down) >= min_run_eff

    return ok, {"run_up": int(run_up), "run_down": int(run_down), "min_run_eff": int(min_run_eff)}


def _token_overlap_ok(tokens: List[Dict], boundary_x: float, *, margin_px: float) -> bool:
    if not tokens:
        return True
    x_lo = float(boundary_x) - float(margin_px)
    x_hi = float(boundary_x) + float(margin_px)
    for t in tokens:
        if not str(t.get("text", "")).strip():
            continue
        try:
            x0 = float(t.get("x0", 0.0))
            x1 = float(t.get("x1", 0.0))
        except Exception:
            continue
        if x0 < x_hi and x1 > x_lo:
            return False
    return True


def _token_gap_ok(
    tokens: List[Dict],
    boundary_x: float,
    *,
    min_gap_px: float,
    line_support_ratio: float,
) -> Tuple[bool, Dict[str, Any]]:
    toks = [t for t in (tokens or []) if str(t.get("text", "")).strip()]
    if len(toks) < 2:
        return True, {"eligible_lines": 0, "support_lines": 0, "ratio": None}

    heights = [max(1.0, float(t.get("y1", 0.0)) - float(t.get("y0", 0.0))) for t in toks]
    median_h = _median(heights, default=12.0)
    gap_req = max(float(min_gap_px), float(median_h) * 0.35, 1.0)

    lines = page_analyzer.group_tokens_into_lines(toks, y_tolerance=12.0)
    eligible = 0
    supported = 0
    for line in lines:
        left = [t for t in line if float(t.get("x1", 0.0)) <= float(boundary_x)]
        right = [t for t in line if float(t.get("x0", 0.0)) >= float(boundary_x)]
        if not left or not right:
            continue
        eligible += 1
        left_max = max(float(t.get("x1", 0.0)) for t in left)
        right_min = min(float(t.get("x0", 0.0)) for t in right)
        if (right_min - left_max) >= gap_req:
            supported += 1

    if eligible <= 0:
        return True, {"eligible_lines": 0, "support_lines": 0, "ratio": None, "min_gap": float(gap_req)}

    ratio = supported / max(1, eligible)
    ok = ratio >= float(line_support_ratio)
    return ok, {
        "eligible_lines": int(eligible),
        "support_lines": int(supported),
        "ratio": float(ratio),
        "min_gap": float(gap_req),
    }


def _whitespace_corridor_ok(
    img_gray_det: object,
    cell_bbox: List[float],
    boundary_x: float,
    *,
    half_w: int,
    tb_pad_px: int,
    row_ink_max: int,
    min_clear_frac: float,
) -> Tuple[bool, Dict[str, Any]]:
    if not HAVE_CV2:
        return False, {"skipped": True, "reason": "no_cv2"}
    if img_gray_det is None:
        return False, {"skipped": True, "reason": "no_img"}
    if not isinstance(img_gray_det, np.ndarray):  # type: ignore[name-defined]
        return False, {"skipped": True, "reason": "bad_img_type"}

    h_img, w_img = img_gray_det.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in cell_bbox)
    x0 = max(0, min(x0, w_img - 1))
    x1 = max(0, min(x1, w_img))
    y0 = max(0, min(y0, h_img - 1))
    y1 = max(0, min(y1, h_img))
    if x1 <= x0 or y1 <= y0:
        return False, {"skipped": False, "reason": "empty_bbox"}

    crop = img_gray_det[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return False, {"skipped": False, "reason": "empty_crop"}

    try:
        _thr, bin_inv = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        return False, {"skipped": False, "reason": "threshold_failed"}

    bx = int(round(float(boundary_x)))
    rel_x = bx - x0
    hw = max(1, int(half_w))
    sx0 = max(0, rel_x - hw)
    sx1 = min(int(bin_inv.shape[1]), rel_x + hw)
    if sx1 <= sx0:
        return False, {"skipped": False, "reason": "strip_empty"}

    pad = max(0, int(tb_pad_px))
    y_top = max(0, pad)
    y_bot = int(bin_inv.shape[0]) - pad
    if y_bot <= y_top:
        return False, {"skipped": False, "reason": "strip_height_empty"}

    strip = bin_inv[y_top:y_bot, sx0:sx1]
    if strip is None or strip.size == 0:
        return False, {"skipped": False, "reason": "strip_empty2"}

    row_ink_max = max(0, int(row_ink_max))
    total_rows = int(strip.shape[0])
    clear_rows = 0
    for i in range(total_rows):
        ink = int(np.count_nonzero(strip[i, :]))  # type: ignore[name-defined]
        if ink <= row_ink_max:
            clear_rows += 1

    clear_frac = float(clear_rows) / float(max(1, total_rows))
    ok = clear_frac >= float(min_clear_frac)
    return ok, {
        "skipped": False,
        "clear_frac": float(clear_frac),
        "rows": int(total_rows),
        "row_ink_max": int(row_ink_max),
    }


def _grid_tolerances(cfg: _Cfg, table_bbox: List[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = (float(v) for v in table_bbox)
    w = max(1.0, float(x1) - float(x0))
    h = max(1.0, float(y1) - float(y0))
    if cfg.snap_tol_px and float(cfg.snap_tol_px) > 0:
        tol = float(cfg.snap_tol_px)
        return tol, tol
    ratio = max(0.0, float(cfg.snap_ratio))
    max_px = max(0.0, float(cfg.snap_max_px))
    tol_x = _clamp(w * ratio, 2.0, max_px if max_px > 0 else 1e9)
    tol_y = _clamp(h * ratio, 6.0, max_px if max_px > 0 else 1e9)
    return float(tol_x), float(tol_y)


def _build_occupancy_grid(
    cells: List[Dict],
    edges_x: List[float],
    edges_y: List[float],
) -> Tuple[List[List[Optional[int]]], Dict[int, Tuple[int, int, int, int]]]:
    nrows = max(0, len(edges_y) - 1)
    ncols = max(0, len(edges_x) - 1)
    grid: List[List[Optional[int]]] = [[None for _ in range(ncols)] for _ in range(nrows)]
    spans: Dict[int, Tuple[int, int, int, int]] = {}

    for cid, cell in enumerate(cells or []):
        bbox = cell.get("bbox_px") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        r0, r1, c0, c1 = grid_span(list(bbox), edges_x, edges_y)
        spans[int(cid)] = (int(r0), int(r1), int(c0), int(c1))
        for r in range(int(r0), int(r1)):
            if r < 0 or r >= nrows:
                continue
            row = grid[r]
            for c in range(int(c0), int(c1)):
                if c < 0 or c >= ncols:
                    continue
                if row[c] is None:
                    row[c] = int(cid)

    return grid, spans


def _draw_overlay(
    img_gray_det: object,
    table_bbox: List[float],
    *,
    boundaries: Dict[int, bool],
    out_path: Path,
) -> None:
    if not HAVE_CV2:
        return
    if img_gray_det is None or not isinstance(img_gray_det, np.ndarray):  # type: ignore[name-defined]
        return

    h_img, w_img = img_gray_det.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in table_bbox)
    x0 = max(0, min(x0, w_img - 1))
    x1 = max(0, min(x1, w_img))
    y0 = max(0, min(y0, h_img - 1))
    y1 = max(0, min(y1, h_img))
    if x1 <= x0 or y1 <= y0:
        return

    crop = img_gray_det[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return

    try:
        vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    except Exception:
        return

    for bx, passed in (boundaries or {}).items():
        rel_x = int(round(float(bx))) - x0
        if rel_x < 0 or rel_x >= vis.shape[1]:
            continue
        color = (0, 200, 0) if passed else (0, 0, 220)  # BGR
        try:
            cv2.line(vis, (rel_x, 0), (rel_x, vis.shape[0] - 1), color, thickness=1)
        except Exception:
            pass

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
    except Exception:
        pass


def split_merged_cells_post_projection(
    tables: List[Dict],
    *,
    img_gray_det: object | None,
    img_w: int,
    img_h: int,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Split merged/oversized spanning cells post-projection using evidence gates.

    Expects each table cell to have:
      - bbox_px: [x0,y0,x1,y1] in detection-DPI coordinates
      - tokens: optional list of token dicts with x0/y0/x1/y1 (+ optional cx/cy)
      - text: optional string

    Mutates tables in-place (table["cells"] may be replaced).
    """
    cfg = _load_cfg()
    if not cfg.enabled:
        return {
            "enabled": False,
            "tables_processed": 0,
            "cells_before": 0,
            "cells_after": 0,
            "cells_split": 0,
        }

    stats: Dict[str, Any] = {
        "enabled": True,
        "tables_seen": int(len(tables or [])),
        "tables_processed": 0,
        "cells_before": 0,
        "cells_after": 0,
        "cells_split": 0,
        "new_cells": 0,
        "boundaries_tested": 0,
        "boundaries_passed": 0,
        "tables": [],
    }

    if debug_dir is not None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            debug_dir = None

    processed = 0
    for table_idx, table in enumerate(tables or []):
        if cfg.debug_max_tables and processed >= int(cfg.debug_max_tables):
            break

        if bool(table.get("borderless", False)):
            continue

        table_cells = table.get("cells") or []
        if not isinstance(table_cells, list) or not table_cells:
            continue

        table_bbox = table.get("bbox_px") or _bbox_from_cells(table_cells)
        if not isinstance(table_bbox, (list, tuple)) or len(table_bbox) != 4:
            continue
        table_bbox_f = [float(v) for v in table_bbox]

        tol_x, tol_y = _grid_tolerances(cfg, table_bbox_f)
        edges_x, edges_y = infer_grid_edges_from_cells(
            table_bbox_f,
            table_cells,
            tol_x,
            tol_y,
            img_w=int(img_w) if img_w else None,
            img_h=int(img_h) if img_h else None,
        )
        if len(edges_x) < 2 or len(edges_y) < 2:
            continue

        grid, spans = _build_occupancy_grid(table_cells, edges_x, edges_y)

        table_debug: Dict[str, Any] = {
            "table_idx": int(table_idx + 1),
            "cells_before": int(len(table_cells)),
            "cells_after": int(len(table_cells)),
            "splits_applied": 0,
            "new_cells": 0,
            "boundaries_tested": 0,
            "boundaries_passed": 0,
            "spanning_cells_considered": 0,
            "spanning_cells_split": 0,
        }
        if cfg.debug:
            table_debug["boundaries"] = []

        out_cells: List[Dict] = []
        overlay_boundaries: Dict[int, bool] = {}

        for cid, cell in enumerate(list(table_cells)):
            span = spans.get(int(cid))
            if not span:
                out_cells.append(cell)
                continue
            r0, r1, c0, c1 = span
            row_span = int(r1) - int(r0)
            col_span = int(c1) - int(c0)
            if row_span != 1 or col_span < 2:
                out_cells.append(cell)
                continue
            if not _cell_meaningful(cell, min_chars=cfg.min_text_chars, min_tokens=cfg.min_tokens):
                out_cells.append(cell)
                continue

            tokens = cell.get("tokens") or []
            if not isinstance(tokens, list):
                tokens = []

            table_debug["spanning_cells_considered"] = int(table_debug["spanning_cells_considered"]) + 1

            candidate_bs = list(range(int(c0) + 1, int(c1)))
            passed_bs: List[int] = []

            for b in candidate_bs:
                boundary_x = float(edges_x[int(b)])
                ev: Dict[str, Any] = {
                    "cell_id": int(cid),
                    "cell_bbox": list(cell.get("bbox_px") or []),
                    "boundary_idx": int(b),
                    "boundary_x": float(boundary_x),
                    "passed": False,
                }

                ok = True

                if cfg.ev_multirow:
                    ok_mr, mr_meta = _multirow_support_ok(
                        grid,
                        r0=int(r0),
                        b=int(b),
                        max_look=int(cfg.multirow_max_look),
                        min_run=int(cfg.multirow_min_run),
                        require_adjacent=bool(cfg.multirow_require_adjacent),
                    )
                    ev["multirow"] = dict(mr_meta)
                    if not ok_mr:
                        ok = False

                if ok and cfg.ev_token_overlap:
                    ok_ov = _token_overlap_ok(tokens, boundary_x, margin_px=float(cfg.boundary_margin_px))
                    ev["token_overlap_ok"] = bool(ok_ov)
                    if not ok_ov:
                        ok = False

                if ok and cfg.ev_token_gap:
                    ok_gap, gap_meta = _token_gap_ok(
                        tokens,
                        boundary_x,
                        min_gap_px=float(cfg.min_gap_px),
                        line_support_ratio=float(cfg.line_support_ratio),
                    )
                    ev["token_gap"] = dict(gap_meta)
                    ev["token_gap_ok"] = bool(ok_gap)
                    if not ok_gap:
                        ok = False

                if ok and cfg.ev_whitespace:
                    if img_gray_det is None or not HAVE_CV2:
                        ev["whitespace"] = {"skipped": True, "required": bool(cfg.ws_required)}
                        if cfg.ws_required:
                            ok = False
                    else:
                        ok_ws, ws_meta = _whitespace_corridor_ok(
                            img_gray_det,
                            list(cell.get("bbox_px") or []),
                            boundary_x,
                            half_w=int(cfg.ws_half_w),
                            tb_pad_px=int(cfg.ws_tb_pad),
                            row_ink_max=int(cfg.ws_row_ink_max),
                            min_clear_frac=float(cfg.ws_min_clear_frac),
                        )
                        ev["whitespace"] = dict(ws_meta)
                        ev["whitespace_ok"] = bool(ok_ws)
                        if not ok_ws:
                            ok = False

                stats["boundaries_tested"] = int(stats["boundaries_tested"]) + 1
                table_debug["boundaries_tested"] = int(table_debug["boundaries_tested"]) + 1

                if ok:
                    passed_bs.append(int(b))
                    stats["boundaries_passed"] = int(stats["boundaries_passed"]) + 1
                    table_debug["boundaries_passed"] = int(table_debug["boundaries_passed"]) + 1
                    ev["passed"] = True

                if cfg.debug:
                    table_debug["boundaries"].append(ev)

                key_x = int(round(boundary_x))
                overlay_boundaries[key_x] = bool(overlay_boundaries.get(key_x) or ok)

            if not passed_bs:
                out_cells.append(cell)
                continue

            passed_bs = sorted(set(passed_bs))
            seg_cols = [int(c0)] + passed_bs + [int(c1)]
            if len(seg_cols) < 3:
                out_cells.append(cell)
                continue

            y0 = float(edges_y[int(r0)])
            y1 = float(edges_y[int(r0) + 1])

            subcells: List[Dict] = []
            for i in range(len(seg_cols) - 1):
                sc0 = int(seg_cols[i])
                sc1 = int(seg_cols[i + 1])
                bx0 = float(edges_x[sc0])
                bx1 = float(edges_x[sc1])
                bbox = [bx0, y0, bx1, y1]
                toks_sub = [t for t in tokens if _center_in_bbox(t, bbox)]
                text_sub = _tokens_to_text(toks_sub)
                subcells.append(
                    {
                        "bbox_px": bbox,
                        "tokens": toks_sub,
                        "text": text_sub,
                        "token_count": int(len([t for t in toks_sub if str(t.get("text", "")).strip()])),
                        "ocr_method": "cell_split_recovery",
                        "split_parent_bbox": list(cell.get("bbox_px") or []),
                        "split_boundaries_x": [float(edges_x[b]) for b in passed_bs],
                    }
                )

            meaningful_flags = [
                _cell_meaningful(sc, min_chars=cfg.min_text_chars, min_tokens=cfg.min_tokens)
                for sc in subcells
            ]
            nonempty = sum(1 for ok_sc in meaningful_flags if ok_sc)
            if nonempty <= 0:
                out_cells.append(cell)
                continue

            if cfg.require_all_subcells_meaningful:
                # Strict protection: never commit a split that produces any blank/non-meaningful subcell.
                if nonempty != int(len(subcells)):
                    out_cells.append(cell)
                    continue

            # Strategy: allow blank "structural" subcells ONLY on the LEFT of the first meaningful
            # subcell. Reject splits that create any blank subcell to the RIGHT of content (or an
            # internal blank), since that tends to create columns out of nothing.
            try:
                first_meaningful = next(i for i, ok_sc in enumerate(meaningful_flags) if ok_sc)
            except StopIteration:
                out_cells.append(cell)
                continue
            if any(not ok_sc for ok_sc in meaningful_flags[int(first_meaningful) :]):
                out_cells.append(cell)
                continue

            # Count leading structural blanks toward acceptance.
            effective_nonempty = int(nonempty) + int(first_meaningful)
            if effective_nonempty < int(cfg.min_nonempty_subcells):
                out_cells.append(cell)
                continue

            table_debug["splits_applied"] = int(table_debug["splits_applied"]) + 1
            table_debug["spanning_cells_split"] = int(table_debug["spanning_cells_split"]) + 1
            stats["cells_split"] = int(stats["cells_split"]) + 1
            added = int(len(subcells) - 1)
            stats["new_cells"] = int(stats["new_cells"]) + added
            table_debug["new_cells"] = int(table_debug["new_cells"]) + added
            out_cells.extend(subcells)

        table["cells"] = out_cells
        try:
            table["num_cells"] = int(len(out_cells))
        except Exception:
            pass
        assign_row_col_indices_grid(out_cells, edges_x, edges_y)

        table_debug["cells_after"] = int(len(out_cells))
        stats["tables_processed"] = int(stats["tables_processed"]) + 1
        stats["cells_before"] = int(stats["cells_before"]) + int(table_debug["cells_before"])
        stats["cells_after"] = int(stats["cells_after"]) + int(table_debug["cells_after"])
        stats["tables"].append(table_debug)

        if debug_dir is not None and cfg.debug and cfg.debug_save_overlays:
            try:
                overlay_path = debug_dir / f"table_{table_idx + 1}_split_overlay.png"
                _draw_overlay(img_gray_det, table_bbox_f, boundaries=overlay_boundaries, out_path=overlay_path)
                table_debug["overlay"] = str(overlay_path)
            except Exception:
                pass

        processed += 1

    if debug_dir is not None and cfg.debug:
        try:
            (debug_dir / "split_recovery_summary.json").write_text(
                json.dumps(stats, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    return stats
