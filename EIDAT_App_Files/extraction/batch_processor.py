"""
Batch Processor - Main extraction pipeline orchestrator

Coordinates the full extraction pipeline:
1. PDF rendering and OCR
2. Table detection
3. Token projection
4. Page analysis
5. Debug output export
"""

from pathlib import Path
from typing import List, Dict, Optional
import traceback
import os
import multiprocessing
from multiprocessing import shared_memory

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

from . import ocr_engine
from . import table_detection
from . import borderless_table_detection
from . import token_projector
from . import page_analyzer
from . import chart_detection
from . import debug_exporter


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(key, str(default)) or str(default)).strip()))
    except Exception:
        return int(default)


def _detect_tables_worker(
    shm_name: str,
    shape: tuple[int, int],
    dtype_str: str,
    verbose: bool,
    out_q,
) -> None:
    shm = None
    try:
        import numpy as np

        shm = shared_memory.SharedMemory(name=str(shm_name))
        img = np.ndarray(tuple(shape), dtype=np.dtype(dtype_str), buffer=shm.buf)
        # Defensive copy: OpenCV ops should treat input as read-only, but don't risk mutating shared memory.
        img_local = img.copy()
        out_q.put(("ok", table_detection.detect_tables(img_local, verbose=bool(verbose))))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}"))
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def _detect_tables_with_timeout(img_gray_hires, *, verbose: bool, timeout_sec: int) -> Dict:
    """
    Run bordered table detection with a hard timeout.

    If it times out or errors, return an empty detection result so the page can continue (OCR still runs).
    """
    try:
        import numpy as np

        if img_gray_hires is None:
            return {"tables": [], "cells": []}
        if not isinstance(img_gray_hires, np.ndarray) or img_gray_hires.dtype != np.uint8 or img_gray_hires.ndim != 2:
            return table_detection.detect_tables(img_gray_hires, verbose=bool(verbose))

        shm = shared_memory.SharedMemory(create=True, size=int(img_gray_hires.nbytes))
        try:
            buf = np.ndarray(img_gray_hires.shape, dtype=img_gray_hires.dtype, buffer=shm.buf)
            buf[:] = img_gray_hires

            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=_detect_tables_worker,
                args=(shm.name, tuple(int(x) for x in img_gray_hires.shape), str(img_gray_hires.dtype), bool(verbose), q),
                daemon=True,
            )
            proc.start()
            proc.join(timeout=float(max(0, int(timeout_sec))))

            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
                proc.join(timeout=5)
                if verbose:
                    print(f"  - Table detection timeout: exceeded {int(timeout_sec)}s (skipping bordered tables)")
                return {"tables": [], "cells": []}

            try:
                status, payload = q.get_nowait()
            except Exception:
                status, payload = ("err", "No result returned from detect_tables worker")
            if status == "ok" and isinstance(payload, dict):
                return payload
            if verbose:
                print(f"  - Table detection error: {payload} (skipping bordered tables)")
            return {"tables": [], "cells": []}
        finally:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
    except Exception:
        # Best-effort: never crash page processing because table detection couldn't be isolated.
        try:
            return table_detection.detect_tables(img_gray_hires, verbose=bool(verbose))
        except Exception:
            return {"tables": [], "cells": []}


def _process_page_worker(
    pipeline_kwargs: Dict,
    pdf_path_str: str,
    page_num: int,
    debug_dir_str: Optional[str],
    verbose: bool,
    out_q,
) -> None:
    try:
        pipeline = ExtractionPipeline(**pipeline_kwargs)
        debug_dir = Path(debug_dir_str) if debug_dir_str else None
        result = pipeline.process_page(Path(pdf_path_str), int(page_num), debug_dir, bool(verbose))
        out_q.put(("ok", result))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}"))


class ExtractionPipeline:
    """Main extraction pipeline."""

    def __init__(self, ocr_dpi: int = 450, detection_dpi: int = 900,
                 lang: str = "eng", psm: int = 3,
                 token_reocr_threshold: float = 0.6,
                 reocr_threshold: float = None,  # DEPRECATED
                 dpi: int = None):
        """
        Initialize extraction pipeline.

        Uses a 2-DPI strategy:
        - Lower DPI (450) for OCR: optimal for text recognition
        - Higher DPI (900) for detection: optimal for border/cell detection

        Pipeline:
        1. Detect cells at 900 DPI (cell bbox first)
        2. OCR tokens at 450 DPI
        3. Re-OCR individual tokens with low TSV confidence
        4. Force-project tokens into cells (no rejection)

        Args:
            ocr_dpi: Render DPI for full-page OCR (default 450, optimal for text)
            detection_dpi: Render DPI for table detection (default 900, high for borders)
            lang: Tesseract language (default eng)
            psm: Tesseract PSM mode for OCR (default 3, auto page segmentation)
            token_reocr_threshold: TSV confidence threshold for token re-OCR (default 0.6)
            reocr_threshold: DEPRECATED - no longer used (cell projection doesn't trigger re-OCR)
            dpi: DEPRECATED - legacy parameter, sets detection_dpi for backward compatibility
        """
        # Handle legacy 'dpi' parameter for backward compatibility
        if dpi is not None:
            detection_dpi = dpi

        self.ocr_dpi = ocr_dpi
        self.detection_dpi = detection_dpi
        self.lang = lang
        self.psm = psm
        self.token_reocr_threshold = token_reocr_threshold
        # Keep legacy attributes for compatibility
        self.dpi = detection_dpi
        self.pass2_dpi = ocr_dpi
        self.pass2_psm = psm
        self.reocr_threshold = reocr_threshold  # Deprecated, kept for compat

    def process_page(self, pdf_path: Path, page_num: int,
                      debug_dir: Optional[Path] = None,
                      verbose: bool = False) -> Dict:
        """
        Process a single page using cell-first, force-projection approach.

        Pipeline:
        1. Table detection at 900 DPI (cell bbox first)
        2. Full-page OCR at 450 DPI (optimal for text) → tokens
        3. Re-OCR individual tokens with low TSV confidence
        4. Scale tokens to detection DPI and force-project into cells

        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            debug_dir: Optional debug output directory
            verbose: Print progress

        Returns:
            Dict with extraction results
        """
        if verbose:
            print(f"Processing page {page_num + 1}...")

        # Step 1: Table detection at HIGH DPI (900) for accurate cell borders FIRST
        if verbose:
            print(f"  - Detecting cells ({self.detection_dpi} DPI)...")

        img_gray_hires, det_img_w, det_img_h = ocr_engine.render_pdf_page(
            pdf_path, page_num, self.detection_dpi
        )

        if img_gray_hires is None:
            if verbose:
                print("  - Failed to render for table detection")
            return {
                'page': page_num + 1,
                'tokens': [],
                'tables': [],
                'img_w': 0,
                'img_h': 0,
                'dpi': self.detection_dpi
            }

        guardrails = str(os.environ.get("EIDAT_TABLE_DETECT_GUARDRAILS", "1") or "1").strip().lower() in (
            "1",
            "true",
            "t",
            "yes",
            "y",
            "on",
        )
        table_timeout_sec = _env_int("EIDAT_TABLE_DETECT_TIMEOUT_SEC", _env_int("EIDAT_PAGE_TABLE_TIMEOUT_SEC", 0))
        if table_timeout_sec < 0:
            table_timeout_sec = 0
        if guardrails and table_timeout_sec and int(table_timeout_sec) > 0:
            table_result = _detect_tables_with_timeout(
                img_gray_hires, verbose=bool(verbose), timeout_sec=int(table_timeout_sec)
            )
        else:
            table_result = table_detection.detect_tables(img_gray_hires, verbose=verbose)
        tables = table_result['tables']
        cells = table_result['cells']

        if verbose:
            print(f"  - Detected {len(tables)} tables with {len(cells)} cells")

        # Step 2: Full-page OCR at TEXT-OPTIMAL DPI (450)
        if verbose:
            print(f"  - Running full-page OCR ({self.ocr_dpi} DPI, psm={self.psm})...")

        tokens, ocr_img_w, ocr_img_h, img_path = ocr_engine.ocr_page(
            pdf_path, page_num, self.ocr_dpi, self.lang, self.psm, debug_dir
        )

        if not tokens:
            if verbose:
                print(f"  - No tokens extracted")
            return {
                'page': page_num + 1,
                'tokens': [],
                'tables': tables,
                'img_w': det_img_w,
                'img_h': det_img_h,
                'dpi': self.detection_dpi
            }

        if verbose:
            print(f"  - Extracted {len(tokens)} tokens")

        # Render OCR DPI image for region OCR and token re-OCR (reuse for consistency)
        img_gray_ocr, _, _ = ocr_engine.render_pdf_page(pdf_path, page_num, self.ocr_dpi)

        # Step 3: Re-OCR individual tokens with low TSV confidence
        low_conf_count = sum(1 for t in tokens if t.get('conf', 0) < self.token_reocr_threshold)
        if low_conf_count > 0:
            if verbose:
                print(f"  - Re-OCR'ing {low_conf_count} low-confidence tokens (conf < {self.token_reocr_threshold})...")

            if img_gray_ocr is not None:
                ocr_engine.reocr_low_confidence_tokens(
                    img_gray_ocr, tokens,
                    conf_threshold=self.token_reocr_threshold,
                    lang=self.lang,
                    verbose=verbose
                )

        # Step 4: Scale tokens to detection DPI and detect borderless tables
        scaled_tokens = token_projector.scale_tokens_to_dpi(
            tokens, self.ocr_dpi, self.detection_dpi
        )

        borderless_tables = borderless_table_detection.detect_borderless_tables(
            scaled_tokens, det_img_w, det_img_h, tables, img_gray=img_gray_hires
        )
        if borderless_tables:
            if verbose:
                print(f"  - Detected {len(borderless_tables)} borderless tables")
            tables.extend(borderless_tables)
            for table in borderless_tables:
                cells.extend(table.get("cells", []))
            if debug_dir:
                debug_exporter.export_borderless_table_debug_images(
                    img_gray_hires, borderless_tables, debug_dir, page_num + 1
                )

        # Step 4b: Force-project tokens into cells
        projection_debug = {}
        table_region_tokens_all: List[Dict] = []
        pass_candidates: Dict[str, Dict[str, object]] = {}

        def _candidate_key(table_idx: int, cell: Dict) -> Optional[str]:
            row = cell.get("row")
            col = cell.get("col")
            if row is None or col is None:
                return None
            try:
                row_i = int(row)
                col_i = int(col)
            except (TypeError, ValueError):
                return None
            return f"t{table_idx + 1}_r{row_i}_c{col_i}"

        def _record_candidate(pass_name: str, table_idx: int, cell: Dict, text: Optional[str] = None) -> None:
            key = _candidate_key(table_idx, cell)
            if not key:
                return
            entry = pass_candidates.setdefault(
                key,
                {"table_idx": int(table_idx + 1), "row": int(cell.get("row")), "col": int(cell.get("col")), "candidates": {}}
            )
            if text is None:
                text = str(cell.get("text", "")).strip()
            entry["candidates"][pass_name] = str(text or "").strip()
        ocr_pass_stats = {
            "numeric_rescue": {"attempted": 0, "replaced": 0},
            "numeric_strict": {"attempted": 0, "replaced": 0},
            "cell_interior": {"attempted": 0, "filled": 0},
            "cell_cleanup": {"attempted": 0, "replaced": 0},
        }
        if cells:
            force_table_ocr = True
            table_char_gap_ratio = 0.35
            table_ocr_debug = projection_debug.setdefault("table_ocr", [])
            proj_stats = {
                'assigned_count': 0,
                'unassigned_count': 0,
                'unassigned_tokens': []
            }
            proj_total_tokens = 0

            if not force_table_ocr:
                if verbose:
                    print(f"  - Force-projecting tokens into cells...")

                proj_total_tokens = len(scaled_tokens)

                # Force-project: always assign tokens, no rejection
                # Pass debug_info dict to capture detailed projection data
                _, proj_stats = token_projector.project_tokens_to_cells_force(
                    scaled_tokens, cells, verbose=verbose,
                    debug_info=projection_debug if debug_dir else None,
                    ocr_dpi=self.ocr_dpi,
                    detection_dpi=self.detection_dpi
                )

            # Step 4a: Table-region OCR to ensure table tokens exist for projection.
            # Full-page OCR (PSM=3) often misses text inside bordered tables entirely.
            # We do a targeted table-bbox OCR pass at OCR DPI (450) with line removal,
            # then project only into still-empty cells.
            if img_gray_ocr is not None and tables:
                table_ocr_dpis = [self.ocr_dpi]
                env_dpis = os.environ.get("EIDAT_TABLE_OCR_DPIS", "").strip()
                if env_dpis:
                    for part in env_dpis.replace(";", ",").split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            table_ocr_dpis.append(int(float(part)))
                        except ValueError:
                            continue
                else:
                    extra_dpi = int(round(self.ocr_dpi * 1.33))
                    if extra_dpi != self.ocr_dpi:
                        table_ocr_dpis.append(extra_dpi)
                    if self.detection_dpi and self.detection_dpi != self.ocr_dpi:
                        table_ocr_dpis.append(int(self.detection_dpi))

                table_ocr_dpis = [
                    min(int(dpi), self.detection_dpi)
                    for dpi in table_ocr_dpis
                    if int(dpi) > 0
                ]
                table_ocr_dpis = sorted({dpi for dpi in table_ocr_dpis if dpi >= 200})
                if self.ocr_dpi not in table_ocr_dpis:
                    table_ocr_dpis.insert(0, self.ocr_dpi)

                ocr_images = {self.ocr_dpi: img_gray_ocr}
                for dpi in table_ocr_dpis:
                    if dpi in ocr_images:
                        continue
                    img_gray_alt, _, _ = ocr_engine.render_pdf_page(pdf_path, page_num, dpi)
                    if img_gray_alt is not None:
                        ocr_images[dpi] = img_gray_alt

                def _token_center_in_bbox(tok: Dict, bbox: List[float]) -> bool:
                    cx = float(tok.get("cx", (tok.get("x0", 0) + tok.get("x1", 0)) / 2))
                    cy = float(tok.get("cy", (tok.get("y0", 0) + tok.get("y1", 0)) / 2))
                    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]

                if force_table_ocr:
                    for cell in cells:
                        cell['tokens'] = []
                        cell['text'] = ''
                        cell['token_count'] = 0

                tables_needing_help = 0
                for table_idx, table in enumerate(tables):
                    table_bbox = table.get("bbox_px") or []
                    table_cells = table.get("cells") or []
                    if len(table_bbox) != 4 or not table_cells:
                        continue
                    table_debug = {
                        "table_idx": table_idx + 1,
                        "psms_tried": [],
                        "dpis_tried": [],
                        "candidates": [],
                        "primary_psm": None,
                        "primary_ocr_dpi": None,
                        "primary_avg_conf": None,
                        "extra_psms": [],
                        "extra_candidates": [],
                        "primary_reocr_count": 0,
                        "extra_reocr_count": 0,
                        "extra_psm_reocr": {},
                    }
                    # Heuristic triggers: no tokens in table bbox, or too few cells got any tokens.
                    tokens_in_bbox = sum(1 for t in scaled_tokens if _token_center_in_bbox(t, table_bbox))
                    cells_with_tokens = sum(1 for c in table_cells if c.get("tokens"))
                    empty_ratio = 1.0 - (float(cells_with_tokens) / float(max(1, len(table_cells))))
                    # Even when full-page OCR yields *some* tokens in the table region, it often misses
                    # many table cells; run region OCR when coverage is poor.
                    if not force_table_ocr:
                        if tokens_in_bbox > 0 and empty_ratio <= 0.20:
                            continue

                    tables_needing_help += 1
                    debug_tag_base = f"page_{page_num + 1}_table_{table_idx + 1}"
                    candidate_evals = []

                    bad_chars = set("|[]")
                    try:
                        table_numeric_ratio_min = float(
                            os.environ.get("EIDAT_TABLE_NUMERIC_RATIO_MIN",
                                           os.environ.get("EIDAT_NUMERIC_RESCUE_MIN_RATIO", "0.6"))
                        )
                    except ValueError:
                        table_numeric_ratio_min = 0.6
                    try:
                        table_decimal_ratio_min = float(
                            os.environ.get("EIDAT_TABLE_NUMERIC_DEC_RATIO_MIN",
                                           os.environ.get("EIDAT_NUMERIC_RESCUE_DEC_RATIO", "0.6"))
                        )
                    except ValueError:
                        table_decimal_ratio_min = 0.6
                    try:
                        cell_override_delta = float(os.environ.get("EIDAT_TABLE_CELL_OVERRIDE_DELTA", "0.2"))
                    except ValueError:
                        cell_override_delta = 0.2

                    def _normalize_numeric_local(text_val: str) -> str:
                        return str(text_val or "").strip().replace(",", "").replace(" ", "")

                    def _is_numeric_like_local(text_val: str) -> bool:
                        txt = _normalize_numeric_local(text_val)
                        if not txt:
                            return False
                        if txt.startswith("(") and txt.endswith(")"):
                            txt = txt[1:-1]
                        if txt.startswith(("+", "-")):
                            txt = txt[1:]
                        if txt.endswith("%"):
                            txt = txt[:-1]
                        if not txt:
                            return False
                        if txt.count(".") > 1:
                            return False
                        return all(ch.isdigit() or ch == "." for ch in txt)

                    def _alnum_ratio_local(text_val: str) -> float:
                        txt = str(text_val or "").strip()
                        if not txt:
                            return 0.0
                        alnum = sum(1 for ch in txt if ch.isalnum())
                        return alnum / max(1, len(txt))

                    def _compute_numeric_cols_local(table_cells_local: List[Dict]) -> Dict[int, Dict[str, float]]:
                        col_stats: Dict[int, Dict[str, object]] = {}
                        for cell in table_cells_local:
                            if int(cell.get("row", 0)) == 0:
                                continue
                            col_idx = int(cell.get("col", 0))
                            text_val = str(cell.get("text", "")).strip()
                            if not text_val:
                                continue
                            stats = col_stats.setdefault(col_idx, {"filled": 0, "numeric": 0, "decimal": 0})
                            stats["filled"] = int(stats["filled"]) + 1
                            if _is_numeric_like_local(text_val):
                                stats["numeric"] = int(stats["numeric"]) + 1
                                norm = _normalize_numeric_local(text_val)
                                if "." in norm:
                                    stats["decimal"] = int(stats["decimal"]) + 1

                        numeric_cols_local: Dict[int, Dict[str, float]] = {}
                        for col_idx, stats in col_stats.items():
                            filled = int(stats.get("filled", 0) or 0)
                            numeric = int(stats.get("numeric", 0) or 0)
                            if filled < 2:
                                continue
                            ratio = numeric / max(1, filled)
                            if ratio < table_numeric_ratio_min:
                                continue
                            dec = int(stats.get("decimal", 0) or 0)
                            dec_ratio = dec / max(1, numeric)
                            if dec_ratio < table_decimal_ratio_min:
                                continue
                            numeric_cols_local[col_idx] = {
                                "numeric_ratio": ratio,
                                "decimal_ratio": dec_ratio,
                            }
                        return numeric_cols_local

                    def _cell_quality(cell: Dict, numeric_cols_local: Dict[int, Dict[str, float]]) -> tuple:
                        text_val = str(cell.get("text", "")).strip()
                        if not text_val:
                            return ((0, 0.0, 0.0, 0.0, 0), 0.0, 0, 0.0, False, False)
                        toks = cell.get("tokens", []) or []
                        avg_conf = (
                            sum(float(t.get("conf", 0.0)) for t in toks) / max(1, len(toks))
                            if toks else 0.0
                        )
                        bad = sum(1 for ch in text_val if ch in bad_chars)
                        alnum_ratio = _alnum_ratio_local(text_val)
                        col_idx = int(cell.get("col", 0))
                        is_numeric_col = col_idx in numeric_cols_local
                        numeric_like = _is_numeric_like_local(text_val)
                        if is_numeric_col:
                            score = (1.0 if numeric_like else 0.0, -float(bad), alnum_ratio, avg_conf, len(text_val))
                            scalar = (1.0 if numeric_like else 0.0) + (alnum_ratio * 0.6) + (avg_conf * 0.4) - (bad * 0.2)
                        else:
                            score = (-float(bad), alnum_ratio, avg_conf, len(text_val), 1.0 if numeric_like else 0.0)
                            scalar = (alnum_ratio * 0.7) + (avg_conf * 0.3) - (bad * 0.2)
                        return (score, scalar, bad, alnum_ratio, numeric_like, is_numeric_col)

                    table_ocr_psms = (11, 3, 6, 7)
                    line_strip_backup = str(os.environ.get("EIDAT_TABLE_OCR_LINESTRIP_BACKUP", "1")).strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    try:
                        line_strip_min_fill = float(os.environ.get("EIDAT_TABLE_OCR_LINESTRIP_MIN_FILL", "0.7"))
                    except ValueError:
                        line_strip_min_fill = 0.7
                    try:
                        line_strip_min_conf = float(os.environ.get("EIDAT_TABLE_OCR_LINESTRIP_MIN_CONF", "0.45"))
                    except ValueError:
                        line_strip_min_conf = 0.45

                    def _pick_best_candidate(cands: List[Dict]) -> Optional[Dict]:
                        best = None
                        best_key = None
                        for cand in cands:
                            key = cand.get("table_quality")
                            if key is None:
                                key = (
                                    cand.get("cells_with_text", 0),
                                    round(float(cand.get("avg_conf", 0.0)), 3),
                                    cand.get("assigned_count", 0),
                                )
                            if best_key is None or key > best_key:
                                best = cand
                                best_key = key
                        return best

                    def _run_table_ocr_variant(*, remove_lines: bool) -> None:
                        nonlocal candidate_evals, table_region_tokens_all
                        for ocr_dpi in table_ocr_dpis:
                            img_gray_table = ocr_images.get(ocr_dpi)
                            if img_gray_table is None:
                                continue

                            scale = float(self.detection_dpi) / float(ocr_dpi)
                            x0, y0, x1, y1 = (float(v) / scale for v in table_bbox)
                            pad = 8  # px at OCR DPI
                            bbox_ocr = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))

                            for psm in table_ocr_psms:
                                debug_tag = f"{debug_tag_base}_dpi{ocr_dpi}_psm{psm}"
                                region_tokens, region_meta = ocr_engine.ocr_region_tokens(
                                    img_gray_table,
                                    bbox_ocr,
                                    lang=self.lang,
                                    psms=(psm,),
                                    remove_lines=remove_lines,
                                    debug_dir=None,
                                    debug_tag=debug_tag,
                                )
                                if not region_tokens:
                                    continue
                                # Re-OCR low-confidence tokens before projecting into cells.
                                ocr_engine.reocr_low_confidence_tokens(
                                    img_gray_table,
                                    region_tokens,
                                    conf_threshold=self.token_reocr_threshold,
                                    lang=self.lang,
                                    verbose=verbose,
                                )
                                reocr_count = sum(1 for t in region_tokens if t.get("reocr"))

                                # Annotate provenance for debug output.
                                for t in region_tokens:
                                    t["_source"] = "table_region_ocr"
                                    t["_page"] = page_num + 1
                                    t["_table_idx"] = table_idx + 1
                                    t["_psm"] = psm
                                    t["_ocr_dpi"] = ocr_dpi
                                table_region_tokens_all.extend(region_tokens)

                                scaled_region_tokens = token_projector.scale_tokens_to_dpi(
                                    region_tokens, ocr_dpi, self.detection_dpi
                                )
                                eval_cells = [{"bbox_px": c["bbox_px"]} for c in table_cells]
                                token_projector.project_tokens_to_cells_force(
                                    scaled_region_tokens,
                                    eval_cells,
                                    verbose=False,
                                    debug_info=None,
                                    ocr_dpi=ocr_dpi,
                                    detection_dpi=self.detection_dpi,
                                    reset_cells=True,
                                    center_margin_px=18.0,
                                    only_if_empty=False,
                                    char_gap_ratio=table_char_gap_ratio,
                                    max_token_ratio=1.6,
                                    max_token_area_ratio=2.5,
                                )
                                token_projector.assign_row_col_indices(eval_cells)
                                numeric_cols_local = _compute_numeric_cols_local(eval_cells)
                                candidate_cells = []
                                for eval_cell in eval_cells:
                                    score, scalar, bad_count, alnum_ratio, numeric_like, is_numeric_col = _cell_quality(
                                        eval_cell, numeric_cols_local
                                    )
                                    candidate_cells.append({
                                        "score": score,
                                        "scalar": scalar,
                                        "bad_count": bad_count,
                                        "alnum_ratio": alnum_ratio,
                                        "numeric_like": numeric_like,
                                        "is_numeric_col": is_numeric_col,
                                        "col": int(eval_cell.get("col", 0)),
                                        "tokens": eval_cell.get("tokens", []) or [],
                                        "text": eval_cell.get("text", ""),
                                        "token_count": int(eval_cell.get("token_count", 0)),
                                        "ocr_psm": psm,
                                        "ocr_dpi": ocr_dpi,
                                        "remove_lines": remove_lines,
                                    })
                                eval_cells_with_text = sum(
                                    1 for c in eval_cells if c.get("text", "").strip()
                                )
                                assigned_count = sum(len(c.get("tokens", [])) for c in eval_cells)
                                confs = [
                                    float(t.get("conf", 0.0))
                                    for t in region_tokens
                                    if t.get("text", "").strip()
                                ]
                                avg_conf = sum(confs) / max(1, len(confs))

                                non_empty_cells = [c for c in candidate_cells if str(c.get("text", "")).strip()]
                                clean_avg = (
                                    sum(float(c.get("scalar", 0.0)) for c in non_empty_cells) / max(1, len(non_empty_cells))
                                )
                                numeric_cells = [
                                    c for c in candidate_cells
                                    if c.get("is_numeric_col")
                                    and str(c.get("text", "")).strip()
                                ]
                                numeric_ok = (
                                    sum(1 for c in numeric_cells if c.get("numeric_like")) / max(1, len(numeric_cells))
                                    if numeric_cells else 0.0
                                )
                                table_quality = (
                                    eval_cells_with_text,
                                    round(float(numeric_ok), 3),
                                    round(float(clean_avg), 3),
                                    round(float(avg_conf), 3),
                                    assigned_count,
                                )

                                candidate_evals.append({
                                    "psm": psm,
                                    "ocr_dpi": ocr_dpi,
                                    "remove_lines": remove_lines,
                                    "tokens": region_tokens,
                                    "meta": region_meta,
                                    "cells": candidate_cells,
                                    "cells_with_text": eval_cells_with_text,
                                    "assigned_count": assigned_count,
                                    "avg_conf": avg_conf,
                                    "table_quality": table_quality,
                                    "clean_avg": clean_avg,
                                    "numeric_ok": numeric_ok,
                                    "reocr_count": reocr_count,
                                })

                    # Primary pass: no line stripping.
                    _run_table_ocr_variant(remove_lines=False)
                    # Backup pass: enable line stripping only if results look weak.
                    if line_strip_backup:
                        no_strip_candidates = [
                            c for c in candidate_evals if not c.get("remove_lines", False)
                        ]
                        best_no_strip = _pick_best_candidate(no_strip_candidates)
                        should_try_strip = False
                        if not best_no_strip:
                            should_try_strip = True
                        else:
                            cells_with_text = int(best_no_strip.get("cells_with_text", 0))
                            fill_ratio = cells_with_text / max(1, len(table_cells))
                            avg_conf = float(best_no_strip.get("avg_conf", 0.0) or 0.0)
                            if fill_ratio < line_strip_min_fill or avg_conf < line_strip_min_conf:
                                should_try_strip = True
                        if should_try_strip:
                            _run_table_ocr_variant(remove_lines=True)
                        table_debug["line_strip_backup"] = bool(should_try_strip)
                        table_debug["line_strip_min_fill"] = round(float(line_strip_min_fill), 3)
                        table_debug["line_strip_min_conf"] = round(float(line_strip_min_conf), 3)

                    best_candidate = _pick_best_candidate(candidate_evals)

                    region_meta = {"psm": None, "ocr_dpi": None, "avg_conf": None}
                    if best_candidate:
                        region_meta = dict(best_candidate.get("meta") or {})
                        region_meta["psm"] = best_candidate.get("psm")
                        region_meta["ocr_dpi"] = best_candidate.get("ocr_dpi")
                        region_meta["avg_conf"] = best_candidate.get("avg_conf")
                        region_meta["remove_lines"] = best_candidate.get("remove_lines")

                    if candidate_evals:
                        table_debug["psms_tried"] = sorted({c.get("psm") for c in candidate_evals if c.get("psm") is not None})
                        table_debug["dpis_tried"] = sorted({c.get("ocr_dpi") for c in candidate_evals if c.get("ocr_dpi") is not None})
                        table_debug["candidates"] = [
                            {
                                "psm": c.get("psm"),
                                "ocr_dpi": c.get("ocr_dpi"),
                                "remove_lines": c.get("remove_lines"),
                                "cells_with_text": c.get("cells_with_text"),
                                "assigned_count": c.get("assigned_count"),
                                "avg_conf": round(float(c.get("avg_conf", 0.0)), 3),
                                "clean_avg": round(float(c.get("clean_avg", 0.0)), 3),
                                "numeric_ok": round(float(c.get("numeric_ok", 0.0)), 3),
                            }
                            for c in candidate_evals
                        ]
                        table_debug["primary_psm"] = region_meta.get("psm")
                        table_debug["primary_ocr_dpi"] = region_meta.get("ocr_dpi")
                        table_debug["primary_remove_lines"] = region_meta.get("remove_lines")
                        if region_meta.get("avg_conf") is not None:
                            table_debug["primary_avg_conf"] = round(float(region_meta.get("avg_conf", 0.0)), 3)
                        if best_candidate is not None:
                            if best_candidate.get("clean_avg") is not None:
                                table_debug["primary_clean_avg"] = round(float(best_candidate.get("clean_avg", 0.0)), 3)
                            if best_candidate.get("numeric_ok") is not None:
                                table_debug["primary_numeric_ok"] = round(float(best_candidate.get("numeric_ok", 0.0)), 3)
                        if best_candidate and best_candidate.get("reocr_count") is not None:
                            table_debug["primary_reocr_count"] = int(best_candidate.get("reocr_count", 0))

                    extra_candidates = [
                        c for c in candidate_evals
                        if best_candidate is None or c is not best_candidate
                    ]
                    if extra_candidates:
                        extra_candidates.sort(
                            key=lambda c: c.get("table_quality") or (
                                c.get("cells_with_text", 0),
                                round(float(c.get("avg_conf", 0.0)), 3),
                                c.get("assigned_count", 0),
                            ),
                            reverse=True,
                        )
                        table_debug["extra_psms"] = [c.get("psm") for c in extra_candidates if c.get("psm") is not None]
                        table_debug["extra_candidates"] = [
                            {
                                "psm": c.get("psm"),
                                "ocr_dpi": c.get("ocr_dpi"),
                                "remove_lines": c.get("remove_lines"),
                                "cells_with_text": c.get("cells_with_text"),
                                "assigned_count": c.get("assigned_count"),
                                "avg_conf": round(float(c.get("avg_conf", 0.0)), 3),
                                "clean_avg": round(float(c.get("clean_avg", 0.0)), 3),
                                "numeric_ok": round(float(c.get("numeric_ok", 0.0)), 3),
                            }
                            for c in extra_candidates
                        ]
                        table_debug["extra_reocr_count"] = sum(
                            int(c.get("reocr_count", 0) or 0) for c in extra_candidates
                        )
                        for extra in extra_candidates:
                            psm_key = f"{extra.get('psm')}@{extra.get('ocr_dpi')}"
                            table_debug["extra_psm_reocr"][psm_key] = int(extra.get("reocr_count", 0) or 0)

                    if verbose:
                        total_region_tokens = sum(len(c.get("tokens", [])) for c in candidate_evals)
                        print(
                            f"    - Table OCR pass: table {table_idx + 1} "
                            f"(tokens_in_bbox={tokens_in_bbox}, cells_with_tokens={cells_with_tokens}/{len(table_cells)}) "
                            f"-> region_tokens={total_region_tokens} psm={region_meta.get('psm')} dpi={region_meta.get('ocr_dpi')}"
                        )

                    if not candidate_evals:
                        continue
                    # Apply best table pass with limited per-cell overrides.
                    if force_table_ocr:
                        assigned_tokens = 0
                        best_cells = (best_candidate or {}).get("cells") or []
                        override_count = 0
                        for idx, cell in enumerate(table_cells):
                            chosen = best_cells[idx] if idx < len(best_cells) else {}
                            is_override = False
                            for cand in extra_candidates:
                                cand_cells = cand.get("cells") or []
                                if idx >= len(cand_cells):
                                    continue
                                alt = cand_cells[idx]
                                if not str(alt.get("text", "")).strip():
                                    continue
                                if not str(chosen.get("text", "")).strip():
                                    chosen = alt
                                    is_override = True
                                    continue
                                alt_score = alt.get("score")
                                chosen_score = chosen.get("score")
                                if alt_score is None or chosen_score is None:
                                    continue
                                if alt_score <= chosen_score:
                                    if chosen.get("bad_count", 0) > 0 and alt.get("bad_count", 0) == 0:
                                        chosen = alt
                                        is_override = True
                                    continue
                                if (float(alt.get("scalar", 0.0)) - float(chosen.get("scalar", 0.0))) < cell_override_delta:
                                    continue
                                if alt.get("bad_count", 0) > chosen.get("bad_count", 0) and chosen.get("bad_count", 0) == 0:
                                    continue
                                chosen = alt
                                is_override = True

                            cell["tokens"] = chosen.get("tokens", []) or []
                            cell["text"] = chosen.get("text", "")
                            cell["token_count"] = int(chosen.get("token_count", 0))
                            cell["ocr_method"] = "table_region_ocr_override" if is_override else "table_region_ocr"
                            cell["ocr_psm"] = chosen.get("ocr_psm")
                            cell["ocr_dpi"] = chosen.get("ocr_dpi")
                            if "remove_lines" in chosen:
                                cell["ocr_remove_lines"] = bool(chosen.get("remove_lines"))
                            assigned_tokens += len(cell["tokens"])
                            if is_override:
                                override_count += 1
                        proj_total_tokens += assigned_tokens
                        proj_stats["assigned_count"] += assigned_tokens
                        if best_candidate and override_count and table_debug is not None:
                            table_debug["override_count"] = override_count

                    # Only emit debug crops for passes that actually won at least one cell.
                    if debug_dir and HAVE_CV2:
                        used_passes = {
                            (cell.get("ocr_psm"), cell.get("ocr_dpi"))
                            for cell in table_cells
                            if cell.get("ocr_psm") and cell.get("ocr_dpi")
                        }

                        def _clamp_bbox_local(bbox, w, h):
                            x0, y0, x1, y1 = bbox
                            x0 = max(0, min(int(x0), w))
                            x1 = max(0, min(int(x1), w))
                            y0 = max(0, min(int(y0), h))
                            y1 = max(0, min(int(y1), h))
                            if x1 < x0:
                                x0, x1 = x1, x0
                            if y1 < y0:
                                y0, y1 = y1, y0
                            return x0, y0, x1, y1

                        for psm, ocr_dpi in sorted(used_passes):
                            img_gray_table = ocr_images.get(int(ocr_dpi))
                            if img_gray_table is None:
                                continue
                            scale = float(self.detection_dpi) / float(ocr_dpi)
                            x0, y0, x1, y1 = (float(v) / scale for v in table_bbox)
                            pad = 8  # px at OCR DPI
                            bbox_ocr = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))
                            h, w = img_gray_table.shape[:2]
                            x0c, y0c, x1c, y1c = _clamp_bbox_local(bbox_ocr, w, h)
                            if x1c - x0c < 2 or y1c - y0c < 2:
                                continue
                            crop = img_gray_table[y0c:y1c, x0c:x1c]
                            debug_tag = f"{debug_tag_base}_dpi{int(ocr_dpi)}_psm{int(psm)}"
                            try:
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(debug_dir / f"{debug_tag}_crop.png"), crop)
                            except Exception:
                                pass

                    if table_debug["psms_tried"] or table_debug["primary_psm"] or table_debug["extra_psms"]:
                        table_debug["selection_mode"] = "table_best_with_overrides"
                        table_ocr_debug.append(table_debug)

                if verbose and tables_needing_help:
                    print(f"  - Table-region OCR attempted for {tables_needing_help} table(s)")

                if force_table_ocr:
                    for cell in cells:
                        cell['ocr_method'] = 'table_region_ocr'

                if force_table_ocr:
                    for table_idx, table in enumerate(tables):
                        table_cells = table.get("cells", []) or []
                        if not table_cells:
                            continue
                        token_projector.assign_row_col_indices(table_cells)
                        for cell in table_cells:
                            _record_candidate("table_region_ocr", table_idx, cell)

            # Helper functions for numeric inference and interior OCR.
            def _normalize_numeric(text_val: str) -> str:
                return str(text_val or "").strip().replace(",", "").replace(" ", "")

            def _is_numeric_like(text_val: str) -> bool:
                txt = _normalize_numeric(text_val)
                if not txt:
                    return False
                if txt.startswith("(") and txt.endswith(")"):
                    txt = txt[1:-1]
                if txt.startswith(("+", "-")):
                    txt = txt[1:]
                if txt.endswith("%"):
                    txt = txt[:-1]
                if not txt:
                    return False
                if txt.count(".") > 1:
                    return False
                return all(ch.isdigit() or ch == "." for ch in txt)

            def _shrink_bbox(bbox: List[float], shrink: int) -> Optional[tuple]:
                if not bbox or len(bbox) != 4:
                    return None
                x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
                x0 += shrink
                y0 += shrink
                x1 -= shrink
                y1 -= shrink
                if x1 <= x0 or y1 <= y0:
                    return None
                return (x0, y0, x1, y1)

            def _compute_numeric_cols(table_cells: List[Dict],
                                      numeric_ratio_min: float,
                                      decimal_ratio_min: float) -> Dict[int, Dict[str, float]]:
                # Build per-column stats (skip header row 0).
                col_stats: Dict[int, Dict[str, object]] = {}
                for cell in table_cells:
                    if int(cell.get("row", 0)) == 0:
                        continue
                    col_idx = int(cell.get("col", 0))
                    text_val = str(cell.get("text", "")).strip()
                    if not text_val:
                        continue
                    stats = col_stats.setdefault(col_idx, {"filled": 0, "numeric": 0, "decimal": 0})
                    stats["filled"] = int(stats["filled"]) + 1
                    if _is_numeric_like(text_val):
                        stats["numeric"] = int(stats["numeric"]) + 1
                        norm = _normalize_numeric(text_val)
                        if "." in norm:
                            stats["decimal"] = int(stats["decimal"]) + 1

                numeric_cols: Dict[int, Dict[str, float]] = {}
                for col_idx, stats in col_stats.items():
                    filled = int(stats.get("filled", 0) or 0)
                    numeric = int(stats.get("numeric", 0) or 0)
                    if filled < 2:
                        continue
                    ratio = numeric / max(1, filled)
                    if ratio < numeric_ratio_min:
                        continue
                    dec = int(stats.get("decimal", 0) or 0)
                    dec_ratio = dec / max(1, numeric)
                    numeric_cols[col_idx] = {
                        "numeric_ratio": ratio,
                        "decimal_ratio": dec_ratio,
                    }
                return numeric_cols

            def _has_bad_chars(text_val: str, bad_chars: set) -> bool:
                return any(ch in bad_chars for ch in str(text_val))

            def _alnum_ratio(text_val: str) -> float:
                txt = str(text_val or "").strip()
                if not txt:
                    return 0.0
                alnum = sum(1 for ch in txt if ch.isalnum())
                return alnum / max(1, len(txt))

            def _clean_score(text_val: str, *, is_numeric_col: bool, bad_chars: set) -> tuple:
                txt = str(text_val or "").strip()
                if not txt:
                    return (-1.0, -1.0, -1)
                bad = sum(1 for ch in txt if ch in bad_chars)
                ratio = _alnum_ratio(txt)
                numeric_like = _is_numeric_like(txt)
                if is_numeric_col:
                    return (1.0 if numeric_like else 0.0, -float(bad), ratio)
                return (-float(bad), ratio, len(txt))

            # Header-row recovery: use page OCR projection or a small PSM 3 pass.
            if scaled_tokens and img_gray_hires is not None and tables:
                header_projection_filled = 0
                header_psm3_filled = 0
                try:
                    header_shrink = int(os.environ.get("EIDAT_HEADER_ROW_SHRINK", "2"))
                except ValueError:
                    header_shrink = 2
                for table_idx, table in enumerate(tables):
                    table_cells = table.get("cells", []) or []
                    if not table_cells:
                        continue
                    token_projector.assign_row_col_indices(table_cells)
                    header_cells = [c for c in table_cells if int(c.get("row", 0)) == 0]
                    if not header_cells:
                        continue
                    empty_headers = [c for c in header_cells if not str(c.get("text", "")).strip()]
                    if empty_headers:
                        before_ids = {id(c) for c in empty_headers}
                        token_projector.project_tokens_to_cells_force(
                            scaled_tokens,
                            empty_headers,
                            verbose=False,
                            debug_info=None,
                            ocr_dpi=self.ocr_dpi,
                            detection_dpi=self.detection_dpi,
                            reset_cells=False,
                            center_margin_px=18.0,
                            only_if_empty=True,
                            char_gap_ratio=table_char_gap_ratio,
                            max_token_ratio=1.6,
                            max_token_area_ratio=2.5,
                        )
                        for cell in empty_headers:
                            if id(cell) in before_ids and str(cell.get("text", "")).strip():
                                cell["ocr_method"] = "header_row_projection"
                                header_projection_filled += 1

                    for cell in header_cells:
                        if str(cell.get("text", "")).strip():
                            continue
                        bbox = _shrink_bbox(cell.get("bbox_px"), header_shrink)
                        if not bbox:
                            continue
                        text = ocr_engine.ocr_cell_region(
                            img_gray_hires,
                            bbox,
                            lang=self.lang,
                            psm=3,
                            padding=0,
                            remove_borders=False,
                        )
                        text = token_projector.normalize_table_cell_text(text)
                        if text:
                            cell["text"] = text
                            cell["ocr_method"] = "header_row_psm3"
                            cell["ocr_psm"] = 3
                            cell["token_count"] = len(text.split())
                            header_psm3_filled += 1

                if verbose and (header_projection_filled or header_psm3_filled):
                    print(
                        f"  - Header recovery: projected={header_projection_filled}, psm3_filled={header_psm3_filled}"
                    )

            def _force_level(_env_key: str) -> int:
                # Deprecated: force flags are no longer used.
                return 0

            # Numeric rescue pass: try strict numeric OCR for suspicious numeric cells.
            enable_numeric_rescue = str(os.environ.get("EIDAT_NUMERIC_RESCUE", "1")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if force_table_ocr and enable_numeric_rescue and tables:
                force_numeric_rescue_level = _force_level("EIDAT_FORCE_NUMERIC_RESCUE")
                force_numeric_rescue = force_numeric_rescue_level >= 1
                force_numeric_rescue_override = force_numeric_rescue_level >= 2
                numeric_rescue_img = img_gray_hires if img_gray_hires is not None else img_gray_ocr
                numeric_rescue_dpi = self.detection_dpi if img_gray_hires is not None else self.ocr_dpi
                if numeric_rescue_img is None:
                    numeric_rescue_dpi = self.ocr_dpi
                numeric_rescue_scale = None
                try:
                    numeric_psm = int(os.environ.get("EIDAT_NUMERIC_RESCUE_PSM", "7"))
                except ValueError:
                    numeric_psm = 7
                try:
                    numeric_padding = int(os.environ.get("EIDAT_NUMERIC_RESCUE_PAD", "4"))
                except ValueError:
                    numeric_padding = 4
                try:
                    numeric_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_MIN_RATIO", "0.6"))
                except ValueError:
                    numeric_ratio_min = 0.6
                try:
                    numeric_ratio_strict = float(os.environ.get("EIDAT_NUMERIC_RESCUE_STRICT_RATIO", "0.8"))
                except ValueError:
                    numeric_ratio_strict = 0.8
                try:
                    decimal_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_DEC_RATIO", "0.6"))
                except ValueError:
                    decimal_ratio_min = 0.6
                try:
                    cleanup_numeric_psm = int(os.environ.get("EIDAT_NUMERIC_RESCUE_PSM", "7"))
                except ValueError:
                    cleanup_numeric_psm = 7
                numeric_whitelist = os.environ.get("EIDAT_NUMERIC_RESCUE_WHITELIST", "0123456789.-")
                numeric_config = ["-c", f"tessedit_char_whitelist={numeric_whitelist}"]
                if numeric_rescue_dpi:
                    numeric_config += ["-c", f"user_defined_dpi={int(numeric_rescue_dpi)}"]

                rescue_attempts = 0
                rescue_filled = 0
                interior_cols_cache = None
                if numeric_rescue_img is None:
                    numeric_rescue_scale = None
                else:
                    numeric_rescue_scale = float(self.detection_dpi) / float(numeric_rescue_dpi)
                if numeric_rescue_scale is None:
                    numeric_rescue_scale = float(self.detection_dpi) / float(self.ocr_dpi)
                numeric_padding_px = int(
                    round(numeric_padding * (float(numeric_rescue_dpi) / float(self.ocr_dpi)))
                ) if numeric_rescue_dpi else numeric_padding
                for table in tables:
                    table_cells = table.get("cells", []) or []
                    if not table_cells:
                        continue
                    token_projector.assign_row_col_indices(table_cells)
                    if force_numeric_rescue:
                        if interior_cols_cache is None:
                            interior_cols_cache = {}
                        numeric_cols = {
                            int(cell.get("col", 0)): {"numeric_ratio": 1.0, "decimal_ratio": 1.0}
                            for cell in table_cells
                            if int(cell.get("row", 0)) > 0
                        }
                    else:
                        numeric_cols = _compute_numeric_cols(table_cells, numeric_ratio_min, decimal_ratio_min)

                    if not numeric_cols:
                        continue

                    for cell in table_cells:
                        if int(cell.get("row", 0)) == 0:
                            continue
                        col_idx = int(cell.get("col", 0))
                        stats = numeric_cols.get(col_idx)
                        if not stats:
                            continue
                        text_val = str(cell.get("text", "")).strip()
                        norm = _normalize_numeric(text_val)
                        allow_non_empty = (
                            stats["numeric_ratio"] >= numeric_ratio_strict
                            and stats["decimal_ratio"] >= decimal_ratio_min
                        )
                        suspicious = False
                        if force_numeric_rescue:
                            suspicious = True
                        elif not text_val:
                            suspicious = True
                        elif not allow_non_empty:
                            continue
                        else:
                            if not _is_numeric_like(text_val):
                                suspicious = True
                            elif stats["decimal_ratio"] >= decimal_ratio_min and "." not in norm:
                                suspicious = True
                            elif any(ch in text_val for ch in "|_[]"):
                                suspicious = True

                        if not suspicious:
                            continue

                        bbox = cell.get("bbox_px")
                        if not bbox or len(bbox) != 4:
                            continue
                        x0, y0, x1, y1 = bbox
                        bbox_ocr = (
                            int(float(x0) / numeric_rescue_scale),
                            int(float(y0) / numeric_rescue_scale),
                            int(float(x1) / numeric_rescue_scale),
                            int(float(y1) / numeric_rescue_scale),
                        )
                        rescue_attempts += 1
                        rescued_text = ocr_engine.ocr_cell_region(
                            numeric_rescue_img,
                            bbox_ocr,
                            lang=self.lang,
                            psm=numeric_psm,
                            padding=numeric_padding_px,
                            remove_borders=False,
                            tesseract_config=numeric_config,
                        ).strip()
                        rescued_text = token_projector.normalize_table_cell_text(rescued_text)
                        _record_candidate("numeric_rescue", table_idx, cell, rescued_text)
                        if not rescued_text or not _is_numeric_like(rescued_text):
                            continue

                        if force_numeric_rescue_override:
                            cell["text"] = rescued_text
                            cell["ocr_method"] = "numeric_rescue"
                            cell["ocr_psm"] = numeric_psm
                            if numeric_rescue_dpi:
                                cell["ocr_dpi"] = int(numeric_rescue_dpi)
                            cell["ocr_numeric_whitelist"] = True
                            cell["token_count"] = len(rescued_text.split())
                            rescue_filled += 1
                            continue

                        norm_rescued = _normalize_numeric(rescued_text)
                        replace = False
                        if not text_val:
                            replace = True
                        elif not _is_numeric_like(text_val):
                            replace = True
                        elif stats["decimal_ratio"] >= decimal_ratio_min and "." in norm_rescued and "." not in norm:
                            replace = True
                        elif any(ch in text_val for ch in "|_[]"):
                            replace = True

                        if replace:
                            cell["text"] = rescued_text
                            cell["ocr_method"] = "numeric_rescue"
                            cell["ocr_psm"] = numeric_psm
                            if numeric_rescue_dpi:
                                cell["ocr_dpi"] = int(numeric_rescue_dpi)
                            cell["ocr_numeric_whitelist"] = True
                            cell["token_count"] = len(rescued_text.split())
                            rescue_filled += 1

                ocr_pass_stats["numeric_rescue"]["attempted"] += rescue_attempts
                ocr_pass_stats["numeric_rescue"]["replaced"] += rescue_filled
                if verbose and rescue_attempts:
                    print(f"  - Numeric rescue: attempted={rescue_attempts}, replaced={rescue_filled}")

            # Numeric strict OCR pass: re-OCR numeric columns and replace only on higher confidence.
            enable_numeric_strict = str(os.environ.get("EIDAT_NUMERIC_STRICT_OCR", "1")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if enable_numeric_strict and tables:
                force_numeric_strict_level = _force_level("EIDAT_FORCE_NUMERIC_STRICT")
                force_numeric_strict = force_numeric_strict_level >= 1
                force_numeric_strict_override = force_numeric_strict_level >= 2
                numeric_strict_img = img_gray_hires if img_gray_hires is not None else img_gray_ocr
                numeric_strict_dpi = self.detection_dpi if img_gray_hires is not None else self.ocr_dpi
                if numeric_strict_img is None:
                    numeric_strict_dpi = self.ocr_dpi
                try:
                    numeric_strict_psm = int(os.environ.get("EIDAT_NUMERIC_STRICT_PSM", "7"))
                except ValueError:
                    numeric_strict_psm = 7
                try:
                    numeric_strict_pad = int(os.environ.get("EIDAT_NUMERIC_STRICT_PAD", "4"))
                except ValueError:
                    numeric_strict_pad = 4
                strict_remove_lines = str(
                    os.environ.get("EIDAT_NUMERIC_STRICT_REMOVE_LINES", "1")
                ).strip().lower() in ("1", "true", "yes", "on")
                try:
                    numeric_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_MIN_RATIO", "0.6"))
                except ValueError:
                    numeric_ratio_min = 0.6
                try:
                    decimal_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_DEC_RATIO", "0.6"))
                except ValueError:
                    decimal_ratio_min = 0.6

                numeric_whitelist = os.environ.get("EIDAT_NUMERIC_RESCUE_WHITELIST", "0123456789.-")
                numeric_config = ["-c", f"tessedit_char_whitelist={numeric_whitelist}"]
                if numeric_strict_dpi:
                    numeric_config += ["-c", f"user_defined_dpi={int(numeric_strict_dpi)}"]

                def _avg_conf(tokens_list: List[Dict]) -> float:
                    confs = [
                        float(t.get("conf", 0.0))
                        for t in tokens_list
                        if str(t.get("text", "")).strip()
                    ]
                    return sum(confs) / max(1, len(confs)) if confs else 0.0

                def _tokens_to_text(tokens_list: List[Dict]) -> str:
                    try:
                        return ocr_engine._sort_tokens_into_text(tokens_list)
                    except Exception:
                        ordered = sorted(
                            tokens_list,
                            key=lambda t: (
                                float(t.get("cy", t.get("y0", 0))),
                                float(t.get("x0", 0)),
                            ),
                        )
                        return " ".join(str(t.get("text", "")).strip() for t in ordered).strip()

                strict_attempts = 0
                strict_replaced = 0
                strict_scale = float(self.detection_dpi) / float(numeric_strict_dpi)
                strict_pad_px = int(
                    round(numeric_strict_pad * (float(numeric_strict_dpi) / float(self.ocr_dpi)))
                ) if numeric_strict_dpi else numeric_strict_pad
                for table_idx, table in enumerate(tables):
                    table_cells = table.get("cells", []) or []
                    if not table_cells:
                        continue
                    token_projector.assign_row_col_indices(table_cells)
                    for cell in table_cells:
                        if int(cell.get("row", 0)) == 0:
                            continue
                        text_val = str(cell.get("text", "")).strip()
                        if not _is_numeric_like(text_val):
                            continue
                        bbox = cell.get("bbox_px")
                        if not bbox or len(bbox) != 4:
                            continue
                        x0, y0, x1, y1 = bbox
                        bbox_ocr = (
                            int(float(x0) / strict_scale) - strict_pad_px,
                            int(float(y0) / strict_scale) - strict_pad_px,
                            int(float(x1) / strict_scale) + strict_pad_px,
                            int(float(y1) / strict_scale) + strict_pad_px,
                        )
                        strict_attempts += 1
                        region_tokens, _meta = ocr_engine.ocr_region_tokens(
                            numeric_strict_img,
                            bbox_ocr,
                            lang=self.lang,
                            psms=(numeric_strict_psm,),
                            remove_lines=strict_remove_lines,
                            tesseract_config=numeric_config,
                        )
                        if not region_tokens:
                            continue
                        new_text = _tokens_to_text(region_tokens).strip()
                        _record_candidate("numeric_strict_ocr", table_idx, cell, new_text)
                        if not new_text or not _is_numeric_like(new_text):
                            continue
                        if not force_numeric_strict_override:
                            new_conf = _avg_conf(region_tokens)
                            old_conf = _avg_conf(cell.get("tokens", []) or [])
                            if new_conf <= old_conf:
                                continue
                        cell["text"] = new_text
                        scaled_region_tokens = token_projector.scale_tokens_to_dpi(
                            region_tokens, numeric_strict_dpi, self.detection_dpi
                        )
                        cell["tokens"] = scaled_region_tokens
                        cell["token_count"] = len(scaled_region_tokens)
                        cell["ocr_method"] = "numeric_strict_ocr"
                        cell["ocr_psm"] = numeric_strict_psm
                        if numeric_strict_dpi:
                            cell["ocr_dpi"] = int(numeric_strict_dpi)
                        cell["ocr_numeric_whitelist"] = True
                        strict_replaced += 1

                ocr_pass_stats["numeric_strict"]["attempted"] += strict_attempts
                ocr_pass_stats["numeric_strict"]["replaced"] += strict_replaced
                if verbose and strict_attempts:
                    print(f"  - Numeric strict OCR: attempted={strict_attempts}, replaced={strict_replaced}")

            # Empty-cell interior OCR fallback (no line stripping; numeric whitelist when applicable).
            if img_gray_hires is not None and tables:
                try:
                    interior_shrink = int(os.environ.get("EIDAT_CELL_INTERIOR_SHRINK", "2"))
                except ValueError:
                    interior_shrink = 2
                try:
                    numeric_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_MIN_RATIO", "0.6"))
                except ValueError:
                    numeric_ratio_min = 0.6
                try:
                    decimal_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_DEC_RATIO", "0.6"))
                except ValueError:
                    decimal_ratio_min = 0.6
                numeric_whitelist = os.environ.get("EIDAT_NUMERIC_RESCUE_WHITELIST", "0123456789.-")
                numeric_config = ["-c", f"tessedit_char_whitelist={numeric_whitelist}"]

                interior_filled = 0
                interior_attempts = 0
                force_cell_interior_level = _force_level("EIDAT_FORCE_CELL_INTERIOR")
                force_cell_interior = force_cell_interior_level >= 1
                force_cell_interior_override = force_cell_interior_level >= 2
                for table_idx, table in enumerate(tables):
                    table_cells = table.get("cells", []) or []
                    if not table_cells:
                        continue
                    token_projector.assign_row_col_indices(table_cells)
                    numeric_cols = _compute_numeric_cols(table_cells, numeric_ratio_min, decimal_ratio_min)
                    for cell in table_cells:
                        if not force_cell_interior and str(cell.get("text", "")).strip():
                            continue
                        bbox = _shrink_bbox(cell.get("bbox_px"), interior_shrink)
                        if not bbox:
                            continue
                        interior_attempts += 1
                        col_idx = int(cell.get("col", 0))
                        tesseract_config = numeric_config if col_idx in numeric_cols else None
                        text = ocr_engine.ocr_cell_region(
                            img_gray_hires,
                            bbox,
                            lang=self.lang,
                            psm=6,
                            padding=0,
                            remove_borders=False,
                            tesseract_config=tesseract_config,
                        )
                        text = token_projector.normalize_table_cell_text(text)
                        _record_candidate("cell_interior_ocr", table_idx, cell, text)
                        if text:
                            if force_cell_interior_override or not str(cell.get("text", "")).strip():
                                cell["text"] = text
                                cell["ocr_method"] = "cell_interior_ocr"
                                cell["ocr_psm"] = 6
                                if tesseract_config is not None:
                                    cell["ocr_numeric_whitelist"] = True
                                cell["token_count"] = len(text.split())
                                interior_filled += 1

                ocr_pass_stats["cell_interior"]["attempted"] += interior_attempts
                ocr_pass_stats["cell_interior"]["filled"] += interior_filled
                if verbose and interior_filled:
                    print(f"  - Cell interior OCR filled {interior_filled} empty cells")

            # Cleanup pass: re-OCR noisy non-empty cells to reduce artifacts.
            enable_cleanup = str(os.environ.get("EIDAT_CELL_CLEANUP", "1")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if enable_cleanup and img_gray_hires is not None and tables:
                try:
                    cleanup_shrink = int(os.environ.get("EIDAT_CELL_CLEANUP_SHRINK", "2"))
                except ValueError:
                    cleanup_shrink = 2
                try:
                    cleanup_min_alnum = float(os.environ.get("EIDAT_CELL_CLEANUP_MIN_ALNUM", "0.6"))
                except ValueError:
                    cleanup_min_alnum = 0.6
                try:
                    numeric_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_MIN_RATIO", "0.6"))
                except ValueError:
                    numeric_ratio_min = 0.6
                try:
                    decimal_ratio_min = float(os.environ.get("EIDAT_NUMERIC_RESCUE_DEC_RATIO", "0.6"))
                except ValueError:
                    decimal_ratio_min = 0.6
                numeric_whitelist = os.environ.get("EIDAT_NUMERIC_RESCUE_WHITELIST", "0123456789.-")
                numeric_config = ["-c", f"tessedit_char_whitelist={numeric_whitelist}"]

                bad_chars = set("|[]")
                cleanup_attempts = 0
                cleanup_replaced = 0
                for table_idx, table in enumerate(tables):
                    table_cells = table.get("cells", []) or []
                    if not table_cells:
                        continue
                    token_projector.assign_row_col_indices(table_cells)
                    numeric_cols = _compute_numeric_cols(table_cells, numeric_ratio_min, decimal_ratio_min)
                    for cell in table_cells:
                        if int(cell.get("row", 0)) == 0:
                            continue
                        text_val = str(cell.get("text", "")).strip()
                        if not text_val:
                            continue
                        col_idx = int(cell.get("col", 0))
                        is_numeric_col = col_idx in numeric_cols
                        noisy = False
                        if _has_bad_chars(text_val, bad_chars):
                            noisy = True
                        elif _alnum_ratio(text_val) < cleanup_min_alnum:
                            noisy = True
                        elif is_numeric_col and not _is_numeric_like(text_val):
                            noisy = True
                        if not noisy:
                            continue
                        bbox = _shrink_bbox(cell.get("bbox_px"), cleanup_shrink)
                        if not bbox:
                            continue
                        cleanup_attempts += 1
                        tesseract_config = numeric_config if is_numeric_col else None
                        psm = cleanup_numeric_psm if is_numeric_col else 6
                        new_text = ocr_engine.ocr_cell_region(
                            img_gray_hires,
                            bbox,
                            lang=self.lang,
                            psm=psm,
                            padding=0,
                            remove_borders=False,
                            tesseract_config=tesseract_config,
                        ).strip()
                        new_text = token_projector.normalize_table_cell_text(new_text)
                        _record_candidate("cell_cleanup_ocr", table_idx, cell, new_text)
                        if not new_text:
                            continue
                        if _has_bad_chars(new_text, bad_chars) and not _has_bad_chars(text_val, bad_chars):
                            continue
                        if is_numeric_col and not _is_numeric_like(new_text):
                            continue
                        if _clean_score(new_text, is_numeric_col=is_numeric_col, bad_chars=bad_chars) <= _clean_score(
                            text_val, is_numeric_col=is_numeric_col, bad_chars=bad_chars
                        ):
                            continue
                        cell["text"] = new_text
                        cell["ocr_method"] = "cell_cleanup_ocr"
                        cell["ocr_psm"] = psm
                        if tesseract_config is not None:
                            cell["ocr_numeric_whitelist"] = True
                        cell["token_count"] = len(new_text.split())
                        cleanup_replaced += 1

                ocr_pass_stats["cell_cleanup"]["attempted"] += cleanup_attempts
                ocr_pass_stats["cell_cleanup"]["replaced"] += cleanup_replaced
                if verbose and cleanup_attempts:
                    print(f"  - Cell cleanup: attempted={cleanup_attempts}, replaced={cleanup_replaced}")

            # Count results
            cells_with_text = sum(1 for c in cells if c.get('text', '').strip())
            cells_empty = len(cells) - cells_with_text
            if verbose:
                print(f"  - Token projection: {cells_with_text}/{len(cells)} cells have text, "
                      f"{proj_stats['assigned_count']}/{proj_total_tokens} tokens assigned")
                if proj_stats['unassigned_count'] > 0:
                    # Show sample of unassigned tokens
                    sample = proj_stats['unassigned_tokens'][:5]
                    for t in sample:
                        print(f"      Unassigned: '{t['text']}' scaled:{t.get('bbox_scaled', [])} original:{t.get('bbox_original', [])}")

            if not force_table_ocr:
                # Step 4b: Cell-based OCR fallback for empty cells AND low-confidence projections
                # This handles cases where:
                # 1. Tesseract PSM=3 misses text inside table cells (empty cells)
                # 2. PSM=3 produces garbage output for bordered tables (low-confidence)
                # Use 0.7 threshold - PSM=3 often produces wrong text with moderate confidence
                LOW_CONF_THRESHOLD = 0.7  # Below this, prefer cell OCR over projection
                ENABLE_CELL_OCR_OVERRIDE = str(os.environ.get("EIDAT_CELL_OCR_OVERRIDE", "")).strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )

                # Count cells that need fallback
                cells_needing_fallback = []
                for cell in cells:
                    text = cell.get('text', '').strip()
                    if not text:
                        cells_needing_fallback.append(cell)
                    elif ENABLE_CELL_OCR_OVERRIDE:
                        # Check if projection tokens had low confidence
                        cell_tokens = cell.get('tokens', [])
                        if cell_tokens:
                            avg_conf = sum(t.get('conf', 0) for t in cell_tokens) / len(cell_tokens)
                            if avg_conf < LOW_CONF_THRESHOLD:
                                cell['_low_conf_projection'] = True
                                cells_needing_fallback.append(cell)

                if cells_needing_fallback:
                    empty_count = sum(1 for c in cells_needing_fallback if not c.get('text', '').strip())
                    low_conf_count = len(cells_needing_fallback) - empty_count
                    if verbose:
                        msg = f"  - Running cell-based OCR on {len(cells_needing_fallback)} cells"
                        if empty_count > 0 and low_conf_count > 0:
                            msg += f" ({empty_count} empty, {low_conf_count} low-confidence)"
                        elif empty_count > 0:
                            msg += f" ({empty_count} empty)"
                        else:
                            msg += f" ({low_conf_count} low-confidence)"
                        print(msg + " (PSM=6)...")

                    fallback_count = 0
                    improved_count = 0
                    for cell in cells_needing_fallback:
                        bbox = cell.get('bbox_px')
                        if not bbox or len(bbox) != 4:
                            continue

                        # Store original text for comparison
                        original_text = cell.get('text', '')
                        was_low_conf = cell.get('_low_conf_projection', False)

                        # OCR the cell region directly from the hires image
                        text = ocr_engine.ocr_cell_region(
                            img_gray_hires, tuple(bbox),
                            lang=self.lang, psm=6, padding=5
                        )
                        text = token_projector.normalize_table_cell_text(text)
                        if text:
                            if was_low_conf:
                                # For low-conf projections, prefer cell OCR result
                                cell['text'] = text
                                cell['ocr_method'] = 'cell_ocr_override'
                                cell['projection_text'] = original_text  # Keep for debug
                                cell['token_count'] = len(text.split())
                                improved_count += 1
                            else:
                                # For empty cells, use cell OCR
                                cell['text'] = text
                                cell['ocr_method'] = 'cell_ocr_fallback'
                                cell['token_count'] = len(text.split())
                                fallback_count += 1

                        # Clean up temp flag
                        cell.pop('_low_conf_projection', None)

                    if verbose:
                        if fallback_count > 0:
                            print(f"  - Cell OCR fallback filled {fallback_count} previously empty cells")
                        if improved_count > 0:
                            print(f"  - Cell OCR override replaced {improved_count} low-confidence projections")

        # Assign row/col indices to cells
        for table in tables:
            table_cells = table.get('cells', [])
            token_projector.assign_row_col_indices(table_cells)

        # Step 5.5: Build per-page pass summary (attempts + matches)
        def _build_pass_summary() -> Dict[str, object]:
            summary: Dict[str, object] = {"total_cells": 0, "passes": {}}

            def _pass_bucket(name: str) -> Dict[str, int]:
                bucket = summary["passes"].setdefault(
                    name,
                    {
                        "attempted": 0,
                        "replaced": 0,
                        "candidate_cells": 0,
                        "candidate_non_empty": 0,
                        "candidate_matches_final": 0,
                        "final_wins": 0,
                        "final_wins_baseline_same": 0,
                    }
                )
                return bucket

            baseline_map = {}
            for key, entry in pass_candidates.items():
                candidates = entry.get("candidates") or {}
                baseline_map[key] = str(candidates.get("table_region_ocr", "")).strip()

            final_map = {}
            for table_idx, table in enumerate(tables):
                table_cells = table.get("cells", []) or []
                if not table_cells:
                    continue
                token_projector.assign_row_col_indices(table_cells)
                for cell in table_cells:
                    key = _candidate_key(table_idx, cell)
                    if not key:
                        continue
                    summary["total_cells"] = int(summary.get("total_cells", 0)) + 1
                    final_text = str(cell.get("text", "")).strip()
                    final_method = cell.get("ocr_method") or "unknown"
                    final_map[key] = {
                        "text": final_text,
                        "method": final_method,
                        "baseline": baseline_map.get(key, "")
                    }
                    _pass_bucket(final_method)["final_wins"] += 1

            for pass_name, stats in ocr_pass_stats.items():
                bucket = _pass_bucket(pass_name)
                bucket["attempted"] += int(stats.get("attempted", 0) or 0)
                bucket["replaced"] += int(stats.get("replaced", 0) or 0)
                if "filled" in stats:
                    bucket["replaced"] += int(stats.get("filled", 0) or 0)

            for key, entry in pass_candidates.items():
                candidates = entry.get("candidates") or {}
                final_info = final_map.get(key)
                final_text = str(final_info.get("text")) if final_info else ""
                baseline_text = str(final_info.get("baseline")) if final_info else ""
                final_method = final_info.get("method") if final_info else None
                for pass_name, candidate in candidates.items():
                    cand_text = str(candidate or "").strip()
                    bucket = _pass_bucket(pass_name)
                    bucket["candidate_cells"] += 1
                    if cand_text:
                        bucket["candidate_non_empty"] += 1
                    if cand_text and cand_text == final_text:
                        bucket["candidate_matches_final"] += 1
                        if final_method == pass_name and baseline_text and baseline_text == final_text:
                            bucket["final_wins_baseline_same"] += 1

            return summary

        pass_summary = _build_pass_summary()

        # Step 6: Analyze page flow
        if verbose:
            print("  - Analyzing page flow...")

        # Use detection DPI dimensions for flow analysis (matches cell coordinates)
        flow_tokens = token_projector.scale_tokens_to_dpi(tokens, self.ocr_dpi, self.detection_dpi)
        flow_data = page_analyzer.extract_flow_text(flow_tokens, tables, det_img_w, det_img_h)

        # Step 6b: Detect charts (axis/label regions)
        enable_charts = str(os.environ.get("EIDAT_ENABLE_CHART_EXTRACTION", "1")).strip().lower() not in (
            "0",
            "false",
            "f",
            "no",
            "n",
            "off",
        )
        charts = []
        if enable_charts:
            charts = chart_detection.detect_charts(
                img_gray_hires, flow_tokens, tables, det_img_w, det_img_h, flow_data
            )

        # Step 7: Build result
        result = {
            'page': page_num + 1,
            'tokens': tokens,
            'tables': tables,
            'charts': charts,
            'img_w': det_img_w,
            'img_h': det_img_h,
            'dpi': self.detection_dpi,
            'ocr_dpi': self.ocr_dpi,
            'flow': flow_data,
            'ocr_pass_stats': ocr_pass_stats,
            'ocr_pass_summary': pass_summary
        }

        # Step 8: Export debug if requested
        if debug_dir:
            if verbose:
                print("  - Exporting debug output...")

            debug_exporter.export_page_debug(
                pdf_path, page_num, tokens, tables,
                det_img_w, det_img_h, self.detection_dpi, debug_dir,
                charts=charts,
                flow_data=flow_data,
                ocr_dpi=self.ocr_dpi,
                ocr_img_w=ocr_img_w,
                ocr_img_h=ocr_img_h,
                ocr_pass_stats=ocr_pass_stats
            )

            try:
                import json
                pass_summary_path = debug_dir / f"page_{page_num + 1}_debug_pass_summary.json"
                with open(pass_summary_path, "w", encoding="utf-8") as f:
                    json.dump(pass_summary, f, indent=2)
            except Exception:
                pass

            if charts:
                debug_exporter.export_chart_debug_images(
                    img_gray_hires, charts, debug_dir, page_num + 1
                )

            # Export detailed table projection debug
            if tables and projection_debug:
                debug_exporter.export_table_projection_debug(
                    page_num, tables, projection_debug, (tokens + table_region_tokens_all), debug_dir,
                    ocr_dpi=self.ocr_dpi, detection_dpi=self.detection_dpi
                )

        return result

    def process_pdf(self, pdf_path: Path, pages: Optional[List[int]] = None,
                     output_dir: Optional[Path] = None,
                     verbose: bool = False) -> List[Dict]:
        """
        Process multiple pages or entire PDF.

        Args:
            pdf_path: Path to PDF
            pages: List of page numbers (0-indexed) or None for all
            output_dir: Output directory for debug files
            verbose: Print progress

        Returns:
            List of page results
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Determine pages to process
        if pages is None:
            # Process all pages
            import fitz
            doc = fitz.open(str(pdf_path))
            pages = list(range(len(doc)))
            doc.close()

        if verbose:
            print(f"Processing {len(pages)} pages from {pdf_path.name}...")

        results = []

        page_timeout_sec = _env_int("EIDAT_PAGE_TIMEOUT_SEC", _env_int("EIDAT_PAGE_TIMEOUT_SECONDS", 300))
        if page_timeout_sec < 0:
            page_timeout_sec = 0

        pipeline_kwargs = {
            "ocr_dpi": int(self.ocr_dpi),
            "detection_dpi": int(self.detection_dpi),
            "lang": str(self.lang),
            "psm": int(self.psm),
            "token_reocr_threshold": float(self.token_reocr_threshold),
            "reocr_threshold": self.reocr_threshold,
            "dpi": None,
        }

        for page_num in pages:
            try:
                # Setup debug dir for this page
                page_debug_dir = None
                if output_dir:
                    page_debug_dir = output_dir / "debug" / "ocr" / pdf_path.stem
                    page_debug_dir.mkdir(parents=True, exist_ok=True)

                if page_timeout_sec and int(page_timeout_sec) > 0:
                    ctx = multiprocessing.get_context("spawn")
                    q = ctx.Queue(maxsize=1)
                    proc = ctx.Process(
                        target=_process_page_worker,
                        args=(pipeline_kwargs, str(pdf_path), int(page_num), str(page_debug_dir) if page_debug_dir else None, bool(verbose), q),
                        daemon=True,
                    )
                    proc.start()
                    proc.join(timeout=float(page_timeout_sec))

                    if proc.is_alive():
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        proc.join(timeout=5)
                        result = {
                            'page': int(page_num) + 1,
                            'error': f"Timeout: exceeded {int(page_timeout_sec)}s",
                            'timeout': True,
                            'tokens': [],
                            'tables': [],
                            'charts': [],
                            'img_w': 0,
                            'img_h': 0,
                            'dpi': int(self.detection_dpi),
                            'ocr_dpi': int(self.ocr_dpi),
                        }
                    else:
                        try:
                            status, payload = q.get_nowait()
                        except Exception:
                            status, payload = ("err", "No result returned from worker")
                        if status == "ok" and isinstance(payload, dict):
                            result = payload
                        else:
                            result = {
                                'page': int(page_num) + 1,
                                'error': str(payload),
                                'tokens': [],
                                'tables': [],
                                'charts': [],
                                'img_w': 0,
                                'img_h': 0,
                                'dpi': int(self.detection_dpi),
                                'ocr_dpi': int(self.ocr_dpi),
                            }
                else:
                    result = self.process_page(pdf_path, page_num, page_debug_dir, verbose)
                results.append(result)

            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")
                if verbose:
                    traceback.print_exc()
                # Add error result
                results.append({
                    'page': page_num + 1,
                    'error': str(e),
                    'tokens': [],
                    'tables': [],
                    'charts': [],
                    'img_w': 0,
                    'img_h': 0,
                    'dpi': int(self.detection_dpi),
                    'ocr_dpi': int(self.ocr_dpi),
                })

        # Export combined outputs
        if output_dir and results:
            doc_output_dir = output_dir / "debug" / "ocr" / pdf_path.stem

            if verbose:
                print("Exporting combined outputs...")

            debug_exporter.export_combined_text(pdf_path, results, doc_output_dir)
            debug_exporter.create_summary_report(pdf_path, results, doc_output_dir)

        if verbose:
            total_tables = sum(len(r.get('tables', [])) for r in results)
            total_tokens = sum(len(r.get('tokens', [])) for r in results)
            print(f"\nCompleted: {len(results)} pages, {total_tokens} tokens, {total_tables} tables")

        return results


def process_pdf_batch(pdf_paths: List[Path], output_dir: Path,
                       ocr_dpi: int = 450, detection_dpi: int = 900,
                       verbose: bool = False) -> Dict[Path, List[Dict]]:
    """
    Process multiple PDFs in batch.

    Args:
        pdf_paths: List of PDF paths
        output_dir: Output directory
        ocr_dpi: DPI for OCR (default 450, optimal for text)
        detection_dpi: DPI for table detection (default 900, optimal for borders)
        verbose: Print progress

    Returns:
        Dict mapping PDF paths to their results
    """
    pipeline = ExtractionPipeline(ocr_dpi=ocr_dpi, detection_dpi=detection_dpi)
    all_results = {}

    for pdf_path in pdf_paths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {pdf_path.name}")
            print('='*60)

        try:
            results = pipeline.process_pdf(pdf_path, output_dir=output_dir, verbose=verbose)
            all_results[pdf_path] = results
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
            if verbose:
                traceback.print_exc()

    return all_results
