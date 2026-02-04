"""
OCR Engine - PDF rendering and Tesseract OCR with bounding boxes

Extracts text tokens with spatial coordinates and confidence scores.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import tempfile
import math

try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except ImportError:
    HAVE_FITZ = False

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def _resolve_tesseract_cmd() -> Tuple[str, Optional[Path]]:
    """Resolve the Tesseract CLI path and optional tessdata dir."""
    for key in ("TESSERACT_CMD", "TESSERACT_BIN", "TESSERACT_PATH"):
        try:
            val = (os.getenv(key) or "").strip()
        except Exception:
            val = ""
        if val:
            cmd_path = Path(val)
            if cmd_path.is_dir():
                exe_name = "tesseract.exe" if os.name == "nt" else "tesseract"
                cmd_path = cmd_path / exe_name
            tessdata = cmd_path.parent / "tessdata"
            return str(cmd_path), tessdata if tessdata.exists() else None

    # Repo-local fallback: tools/tesseract/tesseract(.exe)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        exe_name = "tesseract.exe" if os.name == "nt" else "tesseract"
        repo_cmd = repo_root / "tools" / "tesseract" / exe_name
        if repo_cmd.exists():
            tessdata = repo_cmd.parent / "tessdata"
            return str(repo_cmd), tessdata if tessdata.exists() else None
    except Exception:
        pass

    return "tesseract", None


def render_pdf_page(pdf_path: Path, page_num: int, dpi: int = 900) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Render a PDF page to grayscale image.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: Render DPI (default 900 for high quality)

    Returns:
        Tuple of (img_gray, width, height) or (None, 0, 0) on error
    """
    if not HAVE_FITZ or not HAVE_CV2:
        return None, 0, 0

    try:
        doc = fitz.open(str(pdf_path))
        if page_num >= len(doc):
            return None, 0, 0

        page = doc[page_num]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n >= 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        # Some PDFs render into a tiny occupied band with huge empty margins.
        # If the page looks extremely sparse, re-render with no extra margins.
        try:
            nonwhite = (img_gray < 245).mean()
        except Exception:
            nonwhite = 1.0
        if nonwhite < 0.01:
            try:
                clip = page.rect
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n >= 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img
            except Exception:
                pass

        h, w = img_gray.shape
        return img_gray, w, h
    except Exception as e:
        print(f"Error rendering page {page_num}: {e}")
        return None, 0, 0


def _clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
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


def _remove_table_lines(
    img_gray: np.ndarray,
    *,
    level: str = "default",
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Remove prominent table lines while preserving text.

    Args:
        img_gray: Grayscale image crop.
        level: "default" or "light" sensitivity.
        return_mask: When True, also return the detected line mask.

    This dramatically improves TSV token output in bordered tables without
    requiring a higher DPI render.
    """
    if not HAVE_CV2:
        if return_mask:
            if "np" in globals():
                return img_gray, np.zeros_like(img_gray)
            return img_gray, img_gray
        return img_gray

    h, w = img_gray.shape
    if h < 10 or w < 10:
        if return_mask:
            return img_gray, np.zeros_like(img_gray)
        return img_gray

    level_key = str(level or "default").strip().lower()
    if level_key in ("lite", "light", "soft", "low"):
        level_key = "light"
    else:
        level_key = "default"

    # Default behavior matches the original implementation.
    if level_key == "light":
        h_kernel_ratio = 0.35
        h_kernel_min = 120
        min_horiz_len_ratio = 0.35
        max_horiz_thickness_ratio = 0.015
        max_vert_thickness_ratio = 0.012
        v_kernel_ratio = 0.95
        v_kernel_min = 30
        min_vert_len_ratio = 0.80
        min_vert_len_min = 35
        span_ratio = 0.90
        span_tol_ratio = 0.05
        hough_enable = False
        hough_min_len_ratio = 0.55
        hough_min_len_min = 80
        hough_threshold_ratio = 0.30
        hough_threshold_min = 80
        hough_max_gap_ratio = 0.025
        hough_max_gap_min = 4
        hough_angle_tol = 10.0
        hough_thickness_ratio = 0.003
        hough_thickness_min = 2
        dilate_kernel = (2, 2)
        dilate_iters = 1
    else:
        h_kernel_ratio = 0.25
        h_kernel_min = 80
        min_horiz_len_ratio = 0.20
        max_horiz_thickness_ratio = 0.015
        max_vert_thickness_ratio = 0.012
        v_kernel_ratio = 0.80
        v_kernel_min = 25
        min_vert_len_ratio = 0.60
        min_vert_len_min = 25
        span_ratio = 0.75
        span_tol_ratio = 0.08
        hough_enable = True
        hough_min_len_ratio = 0.45
        hough_min_len_min = 60
        hough_threshold_ratio = 0.25
        hough_threshold_min = 60
        hough_max_gap_ratio = 0.03
        hough_max_gap_min = 5
        hough_angle_tol = 12.0
        hough_thickness_ratio = 0.004
        hough_thickness_min = 2
        dilate_kernel = (3, 3)
        dilate_iters = 1

    # Invert binarization so lines/text are white on black.
    _, bin_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Bridge small gaps so broken rules are detected as continuous lines.
    bin_inv = cv2.morphologyEx(
        bin_inv,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    # Detect long HORIZONTAL lines (internal grid + box borders).
    # Use a long kernel so text strokes don't get detected as "lines".
    h_kernel_w = max(int(w * h_kernel_ratio), h_kernel_min)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    horiz_all = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel)

    # Filtering thresholds: keep only "line-like" components that are long enough.
    # Use separate thickness limits: horizontal rules are often thicker than vertical dividers.
    max_horiz_thickness = max(8, int(min(h, w) * max_horiz_thickness_ratio))
    max_vert_thickness = max(6, int(min(h, w) * max_vert_thickness_ratio))
    min_horiz_len = int(w * min_horiz_len_ratio)

    def _filter_components(mask: np.ndarray, *, min_w: int = 0, min_h: int = 0,
                           max_w: Optional[int] = None, max_h: Optional[int] = None) -> np.ndarray:
        out = np.zeros_like(mask)
        try:
            num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num):
                x, y, ww, hh, _area = stats[i]
                if ww < int(min_w) or hh < int(min_h):
                    continue
                if max_w is not None and ww > int(max_w):
                    continue
                if max_h is not None and hh > int(max_h):
                    continue
                out[labels == i] = 255
        except Exception:
            return mask
        return out

    # Keep only thin, long horizontal components (grid lines).
    horiz = _filter_components(
        horiz_all,
        min_w=min_horiz_len,
        min_h=1,
        max_h=max_horiz_thickness,
    )

    # Derive approximate row separators from horizontal rules so we can remove
    # vertical dividers ONLY when they span a full cell height (between adjacent rules).
    # This avoids erasing text strokes that happen to be vertical.
    horiz_y: List[float] = []
    try:
        num, labels, stats, _centroids = cv2.connectedComponentsWithStats(horiz, connectivity=8)
        for i in range(1, num):
            _x, y, _ww, hh, _area = stats[i]
            horiz_y.append(float(y) + float(hh) / 2.0)
    except Exception:
        horiz_y = []

    horiz_y.sort()
    merged_y: List[float] = []
    for y in horiz_y:
        if not merged_y or abs(y - merged_y[-1]) > 3.0:
            merged_y.append(y)
        else:
            merged_y[-1] = (merged_y[-1] + y) / 2.0

    # Estimate typical cell height.
    gaps = [merged_y[i + 1] - merged_y[i] for i in range(len(merged_y) - 1)]
    gaps = [g for g in gaps if g >= 8.0]
    gaps.sort()
    median_gap = gaps[len(gaps) // 2] if gaps else float(h) * 0.12

    # Detect vertical lines at approximately "cell height" scale (not full-table height).
    v_kernel_h = max(int(median_gap * v_kernel_ratio), v_kernel_min)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    vert_all = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel)

    # Candidate vertical components: thin and reasonably tall.
    min_vert_len = max(int(median_gap * min_vert_len_ratio), min_vert_len_min)
    vert_candidates = _filter_components(
        vert_all,
        min_w=1,
        min_h=min_vert_len,
        max_w=max_vert_thickness,
    )

    # Keep only vertical components that span (nearly) the full height between adjacent
    # horizontal rules (i.e. "full height of a cell").
    vert_remove = np.zeros_like(vert_candidates)
    if len(merged_y) >= 2:
        y_lines = [0.0] + merged_y + [float(h - 1)]
        try:
            num, labels, stats, _centroids = cv2.connectedComponentsWithStats(vert_candidates, connectivity=8)
            for i in range(1, num):
                x, y, ww, hh, _area = stats[i]
                top = float(y)
                bot = float(y + hh)
                mid = (top + bot) / 2.0

                # Find the enclosing band [y0, y1] based on horizontal rules.
                y0 = 0.0
                y1 = float(h - 1)
                for j in range(len(y_lines) - 1):
                    if y_lines[j] <= mid <= y_lines[j + 1]:
                        y0 = y_lines[j]
                        y1 = y_lines[j + 1]
                        break
                band_h = max(1.0, y1 - y0)
                tol = max(2.0, band_h * span_tol_ratio)

                spans_band = (top <= y0 + tol) and (bot >= y1 - tol) and (hh >= span_ratio * band_h)
                if spans_band:
                    vert_remove[labels == i] = 255
        except Exception:
            vert_remove = vert_candidates
    else:
        # No reliable row separators found; fall back to removing only outer box borders.
        border_margin = max(12, int(w * 0.01))
        border_margin = min(border_margin, max(12, int(w * 0.03)))
        vert_remove[:, :border_margin] = vert_candidates[:, :border_margin]
        vert_remove[:, w - border_margin:] = vert_candidates[:, w - border_margin:]

    def _hough_line_mask(mask: np.ndarray) -> np.ndarray:
        if not hough_enable:
            return np.zeros_like(mask)
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        min_dim = min(h, w)
        if min_dim < 60:
            return np.zeros_like(mask)
        min_len = max(hough_min_len_min, int(min_dim * hough_min_len_ratio))
        max_gap = max(hough_max_gap_min, int(min_dim * hough_max_gap_ratio))
        threshold = max(hough_threshold_min, int(min_dim * hough_threshold_ratio))
        lines_p = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        out = np.zeros_like(mask)
        if lines_p is None:
            return out
        angle_tol = hough_angle_tol
        thickness = max(hough_thickness_min, int(min_dim * hough_thickness_ratio))
        for x1, y1, x2, y2 in lines_p[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle > 90.0:
                angle = 180.0 - angle
            if angle <= angle_tol or angle >= (90.0 - angle_tol):
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=thickness)
        return out

    lines = cv2.bitwise_or(horiz, vert_remove)
    hough_lines = _hough_line_mask(bin_inv)
    lines = cv2.bitwise_or(lines, hough_lines)

    # Expand slightly to cover thick borders; keep this small to avoid erasing text.
    if dilate_iters > 0:
        lines = cv2.dilate(
            lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel),
            iterations=dilate_iters,
        )

    # Paint detected lines white on the original image.
    cleaned = img_gray.copy()
    cleaned[lines > 0] = 255
    if return_mask:
        return cleaned, lines
    return cleaned


def run_tesseract_tsv(
    img_path: Path,
    lang: str = "eng",
    psm: int = 6,
    config: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run Tesseract OCR in TSV output mode.

    Args:
        img_path: Path to image file
        lang: Tesseract language (default: eng)
        psm: Page segmentation mode (default: 6 - uniform block)

    Returns:
        Tuple of (tsv_output, error_message)
    """
    try:
        tesseract_cmd, tessdata_dir = _resolve_tesseract_cmd()
        env = None
        if tessdata_dir:
            env = os.environ.copy()
            env.setdefault("TESSDATA_PREFIX", str(tessdata_dir))

        # Build tesseract command
        cmd = [
            tesseract_cmd,
            str(img_path),
            "stdout",
            "-l", lang,
            "--psm", str(psm),
            "--oem", "3",
            "tsv"
        ]
        if config:
            cmd.extend([str(c) for c in config if str(c).strip()])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        if result.returncode != 0:
            return None, result.stderr

        return result.stdout, None

    except subprocess.TimeoutExpired:
        return None, "Tesseract timeout"
    except FileNotFoundError:
        return None, "Tesseract not found in PATH"
    except Exception as e:
        return None, str(e)


def parse_tesseract_tsv(tsv_text: str) -> List[Dict]:
    """
    Parse Tesseract TSV output into token list.

    Returns list of dicts with:
        - text: The recognized text
        - conf: Confidence (0-1)
        - x0, y0, x1, y1: Bounding box in pixels
        - cx, cy: Center coordinates
        - level: Tesseract hierarchy level
        - page_num, block_num, par_num, line_num, word_num
    """
    tokens = []
    lines = tsv_text.strip().split('\n')

    if len(lines) < 2:
        return tokens

    # Skip header line
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 12:
            continue

        try:
            level = int(parts[0])
            page_num = int(parts[1])
            block_num = int(parts[2])
            par_num = int(parts[3])
            line_num = int(parts[4])
            word_num = int(parts[5])
            left = int(parts[6])
            top = int(parts[7])
            width = int(parts[8])
            height = int(parts[9])
            conf = float(parts[10]) if parts[10] != '-1' else 0
            text = parts[11] if len(parts) > 11 else ""

            # Filter out empty text or very low confidence
            if not text.strip() or conf < 30:
                continue

            # Only keep word-level tokens (level 5)
            if level != 5:
                continue

            x0, y0 = left, top
            x1, y1 = left + width, top + height
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            tokens.append({
                'text': text,
                'conf': conf / 100.0,  # Normalize to 0-1
                'x0': float(x0),
                'y0': float(y0),
                'x1': float(x1),
                'y1': float(y1),
                'cx': float(cx),
                'cy': float(cy),
                'level': level,
                'page_num': page_num,
                'block_num': block_num,
                'par_num': par_num,
                'line_num': line_num,
                'word_num': word_num
            })

        except (ValueError, IndexError):
            continue

    return tokens


def ocr_region_tokens(
    img_gray: np.ndarray,
    bbox_px: Tuple[int, int, int, int],
    *,
    lang: str,
    psms: Tuple[int, ...] = (6, 3, 11),
    remove_lines: bool = True,
    line_strip_level: Optional[str] = None,
    tesseract_config: Optional[List[str]] = None,
    debug_dir: Optional[Path] = None,
    debug_tag: str = "region",
    debug_emit: Optional[Dict[str, bool]] = None,
) -> Tuple[List[Dict], Dict[str, object]]:
    """
    OCR a region of an already-rendered page image and return TSV word tokens.

    Unlike `ocr_cell_region`, this is designed for projection: it returns
    per-token bboxes so tokens can be mapped into detected cells.

    Args:
        img_gray: Full page grayscale image at OCR DPI (e.g. 450)
        bbox_px: Region bbox in *this* image's pixel coordinates
        lang: Tesseract language
        psms: Ordered list of PSM modes to try; best result is selected
        remove_lines: Remove strong table borders before OCR
        line_strip_level: Optional sensitivity level ("default" or "light")
        debug_dir: Optional folder for saving crops
        debug_tag: Label used in debug filenames
        debug_emit: Optional flags for debug outputs (crop/nolines/linemask)

    Returns:
        (tokens, meta) where tokens are in full-page coordinates.
    """
    if not HAVE_CV2:
        return [], {"error": "OpenCV not available"}

    h, w = img_gray.shape
    x0, y0, x1, y1 = _clamp_bbox(bbox_px, w, h)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return [], {"error": "region too small"}

    crop = img_gray[y0:y1, x0:x1]
    line_mask = None
    if remove_lines:
        strip_level = line_strip_level or "default"
        if debug_dir:
            proc, line_mask = _remove_table_lines(crop, level=strip_level, return_mask=True)
        else:
            proc = _remove_table_lines(crop, level=strip_level)
    else:
        proc = crop

    if debug_dir:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            emit_crop = True
            emit_nolines = True
            emit_mask = True
            if debug_emit:
                emit_crop = bool(debug_emit.get("crop", emit_crop))
                emit_nolines = bool(debug_emit.get("nolines", emit_nolines))
                emit_mask = bool(debug_emit.get("linemask", emit_mask))
            if emit_crop:
                cv2.imwrite(str(debug_dir / f"{debug_tag}_crop.png"), crop)
            if remove_lines:
                if emit_nolines:
                    cv2.imwrite(str(debug_dir / f"{debug_tag}_nolines.png"), proc)
                if emit_mask and line_mask is not None:
                    cv2.imwrite(str(debug_dir / f"{debug_tag}_linemask.png"), line_mask)
        except Exception:
            pass

    best_tokens: List[Dict] = []
    best_meta: Dict[str, object] = {
        "psm": None,
        "score": -1.0,
        "remove_lines": remove_lines,
        "line_strip_level": line_strip_level or ("default" if remove_lines else None),
    }

    # Try multiple PSMs and keep the best token set by a simple quality score.
    for psm in psms:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), proc)

        try:
            tsv_text, error = run_tesseract_tsv(
                tmp_path, lang, int(psm), config=tesseract_config
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if error or not tsv_text:
            continue

        toks = parse_tesseract_tsv(tsv_text)
        if not toks:
            continue

        # Score: reward count of reasonably-confident tokens, lightly reward avg confidence.
        confs = [float(t.get("conf", 0.0)) for t in toks if t.get("text", "").strip()]
        good = sum(1 for c in confs if c >= 0.35)
        avg = sum(confs) / max(1, len(confs))
        score = good + avg

        if score > float(best_meta.get("score", -1.0)):
            best_tokens = toks
            best_meta = {
                "psm": int(psm),
                "score": float(score),
                "remove_lines": remove_lines,
                "line_strip_level": line_strip_level or ("default" if remove_lines else None),
            }

    # Offset region-local coords back to full-page coords.
    out: List[Dict] = []
    for t in best_tokens:
        tt = dict(t)
        for key in ("x0", "x1", "cx"):
            if key in tt:
                tt[key] = float(tt[key]) + float(x0)
        for key in ("y0", "y1", "cy"):
            if key in tt:
                tt[key] = float(tt[key]) + float(y0)
        out.append(tt)

    best_meta["token_count"] = len(out)
    return out, best_meta


def ocr_page(pdf_path: Path, page_num: int, dpi: int = 900,
             lang: str = "eng", psm: int = 6,
             debug_dir: Optional[Path] = None) -> Tuple[List[Dict], int, int, Optional[Path]]:
    """
    Full pipeline: Render PDF page and run OCR.

    Args:
        pdf_path: Path to PDF
        page_num: Page number (0-indexed)
        dpi: Render DPI
        lang: Tesseract language
        psm: Page segmentation mode
        debug_dir: Optional directory to save rendered image

    Returns:
        Tuple of (tokens, img_width, img_height, img_path)
    """
    # Render page
    img_gray, img_w, img_h = render_pdf_page(pdf_path, page_num, dpi)

    if img_gray is None:
        return [], 0, 0, None

    # Save to temp file for Tesseract
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img_path = Path(tmp.name)
        cv2.imwrite(str(img_path), img_gray)

    # Optionally save debug copy
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_img_path = debug_dir / f"page_{page_num + 1}.png"
        cv2.imwrite(str(debug_img_path), img_gray)

    # Run OCR
    tsv_text, error = run_tesseract_tsv(img_path, lang, psm)

    if error or not tsv_text:
        os.unlink(img_path)
        return [], img_w, img_h, img_path if debug_dir else None

    # Parse tokens
    tokens = parse_tesseract_tsv(tsv_text)

    # Clean up temp file
    if not debug_dir:
        os.unlink(img_path)

    return tokens, img_w, img_h, img_path if debug_dir else None


def get_tesseract_lang() -> str:
    """Get Tesseract language from environment or default to 'eng'."""
    return os.getenv("EIDAT_TESS_LANG", "eng")


def get_tesseract_psm() -> int:
    """Get Tesseract PSM from environment or default to 6."""
    try:
        return int(os.getenv("EIDAT_TESS_PSM", "6"))
    except ValueError:
        return 6


def _sort_tokens_into_text(tokens: List[Dict], y_tolerance: float = 15) -> str:
    """
    Sort tokens into reading order and combine into text.

    Groups tokens into lines based on y-position (with tolerance for slight
    variations due to character heights), then sorts each line by x-position.

    Args:
        tokens: List of token dicts with x0, y0, text
        y_tolerance: Max y-difference to be considered same line (pixels)

    Returns:
        Combined text in reading order
    """
    if not tokens:
        return ""

    # Filter tokens with text
    valid_tokens = [t for t in tokens if t.get('text', '').strip()]
    if not valid_tokens:
        return ""

    # Sort by y-center first to process top-to-bottom
    valid_tokens.sort(key=lambda t: t.get('cy', t.get('y0', 0)))

    # Group into lines using y-tolerance
    lines = []
    current_line = [valid_tokens[0]]
    current_y = valid_tokens[0].get('cy', valid_tokens[0].get('y0', 0))

    for token in valid_tokens[1:]:
        token_y = token.get('cy', token.get('y0', 0))

        if abs(token_y - current_y) <= y_tolerance:
            # Same line
            current_line.append(token)
        else:
            # New line - finalize current and start new
            lines.append(current_line)
            current_line = [token]
            current_y = token_y

    # Don't forget the last line
    if current_line:
        lines.append(current_line)

    # Sort each line by x-position and combine
    result_lines = []
    for line in lines:
        line.sort(key=lambda t: t.get('x0', 0))
        line_text = ' '.join(t.get('text', '') for t in line)
        result_lines.append(line_text)

    return ' '.join(result_lines)


def _remove_cell_borders(cell_img: np.ndarray, border_width: int = 8) -> np.ndarray:
    """
    Remove table border lines from a cell crop to improve OCR.

    Borders appear as dark lines at the edges of the cell.
    We detect and remove them by looking for dark pixels in edge regions.

    Args:
        cell_img: Grayscale cell image
        border_width: Max width of border to remove (pixels)

    Returns:
        Cell image with borders removed (cropped or whitened)
    """
    if cell_img is None or cell_img.size == 0:
        return cell_img

    h, w = cell_img.shape[:2]
    if h < border_width * 3 or w < border_width * 3:
        return cell_img  # Cell too small to safely remove borders

    # Create a copy to modify
    result = cell_img.copy()

    # Strategy: Look for dark edge regions and either crop or whiten them
    # A border is detected if the mean of edge pixels is significantly darker

    def is_border_region(region, threshold=180):
        """Check if a region is likely a border (mostly dark pixels)."""
        if region.size == 0:
            return False
        return np.mean(region) < threshold

    # Check and remove left border
    left_region = result[:, :border_width]
    if is_border_region(left_region):
        # Find where the border ends (first column that's mostly white)
        for i in range(min(border_width * 2, w // 3)):
            if np.mean(result[:, i]) > 200:
                result = result[:, i:]
                break
        else:
            result[:, :border_width] = 255  # Whiten if can't find clean edge

    # Check and remove right border
    h, w = result.shape[:2]
    if w > border_width * 2:
        right_region = result[:, -border_width:]
        if is_border_region(right_region):
            for i in range(min(border_width * 2, w // 3)):
                if np.mean(result[:, -(i+1)]) > 200:
                    result = result[:, :-(i+1)] if i > 0 else result
                    break
            else:
                result[:, -border_width:] = 255

    # Check and remove top border
    h, w = result.shape[:2]
    if h > border_width * 2:
        top_region = result[:border_width, :]
        if is_border_region(top_region):
            for i in range(min(border_width * 2, h // 3)):
                if np.mean(result[i, :]) > 200:
                    result = result[i:, :]
                    break
            else:
                result[:border_width, :] = 255

    # Check and remove bottom border
    h, w = result.shape[:2]
    if h > border_width * 2:
        bottom_region = result[-border_width:, :]
        if is_border_region(bottom_region):
            for i in range(min(border_width * 2, h // 3)):
                if np.mean(result[-(i+1), :]) > 200:
                    result = result[:-(i+1), :] if i > 0 else result
                    break
            else:
                result[-border_width:, :] = 255

    return result


def _parse_tesseract_tsv_low_conf(tsv_text: str, min_conf: int = 10) -> List[Dict]:
    """
    Parse Tesseract TSV output with lower confidence threshold.

    For cell OCR fallback, we use a lower threshold because:
    1. We know there should be text in the cell
    2. Tesseract confidence drops when borders are present
    3. Even low-confidence text is better than empty

    Args:
        tsv_text: Raw TSV output from Tesseract
        min_conf: Minimum confidence threshold (default 10, much lower than standard 30)

    Returns:
        List of token dicts
    """
    tokens = []
    lines = tsv_text.strip().split('\n')

    if len(lines) < 2:
        return tokens

    # Skip header line
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 12:
            continue

        try:
            level = int(parts[0])
            page_num = int(parts[1])
            block_num = int(parts[2])
            par_num = int(parts[3])
            line_num = int(parts[4])
            word_num = int(parts[5])
            left = int(parts[6])
            top = int(parts[7])
            width = int(parts[8])
            height = int(parts[9])
            conf = float(parts[10]) if parts[10] != '-1' else 0
            text = parts[11] if len(parts) > 11 else ""

            # Filter out empty text or very low confidence (using lower threshold)
            if not text.strip() or conf < min_conf:
                continue

            # Only keep word-level tokens (level 5)
            if level != 5:
                continue

            x0, y0 = left, top
            x1, y1 = left + width, top + height
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            tokens.append({
                'text': text,
                'conf': conf / 100.0,  # Normalize to 0-1
                'x0': float(x0),
                'y0': float(y0),
                'x1': float(x1),
                'y1': float(y1),
                'cx': float(cx),
                'cy': float(cy),
                'level': level,
                'page_num': page_num,
                'block_num': block_num,
                'par_num': par_num,
                'line_num': line_num,
                'word_num': word_num
            })

        except (ValueError, IndexError):
            continue

    return tokens


def ocr_cell_region(
    img_gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    lang: str = "eng",
    psm: int = 6,
    padding: int = 5,
    remove_borders: bool = True,
    debug_path: Optional[Path] = None,
    tesseract_config: Optional[List[str]] = None,
) -> str:
    """
    OCR a specific cell region from the image.

    This is the 2-pass approach:
    1. Cell detection finds cell boundaries
    2. This function crops and OCRs each cell individually

    Args:
        img_gray: Full page grayscale image
        bbox: Cell bounding box (x0, y0, x1, y1) in pixels
        lang: Tesseract language
        psm: Page segmentation mode (6=block, 7=single line)
        padding: Pixels to expand crop region
        remove_borders: Whether to preprocess to remove table borders
        debug_path: Optional path to save cell crop for debugging

    Returns:
        Extracted text from the cell
    """
    if not HAVE_CV2:
        return ""

    x0, y0, x1, y1 = bbox
    h, w = img_gray.shape[:2]

    # Expand with padding but stay within image bounds
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)

    if x1 <= x0 or y1 <= y0:
        return ""

    # Crop cell region
    cell_img = img_gray[y0:y1, x0:x1]

    # Remove border lines to improve OCR
    if remove_borders:
        cell_img = _remove_cell_borders(cell_img, border_width=10)

    # Save debug crop if requested
    if debug_path:
        cv2.imwrite(str(debug_path), cell_img)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), cell_img)

    try:
        # Run tesseract with specified PSM
        # PSM 6 (uniform block) is generally best for cell OCR
        tsv_text, error = run_tesseract_tsv(tmp_path, lang, psm, config=tesseract_config)

        if error or not tsv_text:
            return ""

        # Parse tokens with LOW confidence threshold for cell OCR
        # We use 10% instead of 30% because borders can reduce confidence
        tokens = _parse_tesseract_tsv_low_conf(tsv_text, min_conf=10)

        # Group tokens into lines using y-tolerance, then sort by x within each line
        # This handles slight y-variations due to character height differences
        text = _sort_tokens_into_text(tokens)
        return text.strip()

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def ocr_all_cells(img_gray: np.ndarray, cells: List[Dict],
                  lang: str = "eng", psm: int = 6) -> List[Dict]:
    """
    OCR all detected cells individually using 2-pass approach.

    Args:
        img_gray: Full page grayscale image
        cells: List of cells with 'bbox_px' field
        lang: Tesseract language
        psm: Page segmentation mode

    Returns:
        Updated cells with 'text' and 'ocr_method' fields
    """
    for cell in cells:
        bbox = cell.get('bbox_px')
        if not bbox or len(bbox) != 4:
            cell['text'] = ''
            cell['ocr_method'] = 'skipped'
            continue

        text = ocr_cell_region(img_gray, tuple(bbox), lang, psm)
        cell['text'] = text
        cell['ocr_method'] = 'cell_crop'
        cell['token_count'] = len(text.split()) if text else 0

    return cells


def ocr_cells_selective(img_gray: np.ndarray, cells: List[Dict],
                        lang: str = "eng", psm: int = 6,
                        verbose: bool = False) -> List[Dict]:
    """
    OCR only specific cells that need re-OCR (low confidence from projection).

    This is used after token projection as a fallback for cells where
    projection confidence was below threshold.

    Args:
        img_gray: Full page grayscale image
        cells: List of cells to re-OCR (should have 'bbox_px' and 'needs_reocr' fields)
        lang: Tesseract language
        psm: Page segmentation mode
        verbose: Print progress

    Returns:
        Updated cells with 'text' and 'ocr_method' fields
    """
    reocr_count = 0

    for cell in cells:
        # Only re-OCR cells flagged as needing it
        if not cell.get('needs_reocr', False):
            continue

        bbox = cell.get('bbox_px')
        if not bbox or len(bbox) != 4:
            continue

        # Store original projection text for comparison
        projection_text = cell.get('text', '')
        projection_conf = cell.get('projection_confidence', 0)

        # Run fresh OCR on cell crop
        text = ocr_cell_region(img_gray, tuple(bbox), lang, psm)

        if text.strip():
            # Re-OCR produced text - use it
            cell['text'] = text
            cell['ocr_method'] = 'reocr_fallback'
            cell['token_count'] = len(text.split())
            cell['projection_text'] = projection_text  # Keep for debugging
            reocr_count += 1
        else:
            # Re-OCR failed - keep projection result
            cell['ocr_method'] = 'token_projection_kept'

        cell['needs_reocr'] = False  # Clear flag

    if verbose and reocr_count > 0:
        print(f"    Re-OCR'd {reocr_count} low-confidence cells")

    return cells


# Default threshold for token re-OCR (60% TSV confidence)
TOKEN_REOCR_CONF_THRESHOLD = 0.6


def reocr_low_confidence_tokens(
    img_gray: np.ndarray,
    tokens: List[Dict],
    conf_threshold: float = TOKEN_REOCR_CONF_THRESHOLD,
    lang: str = "eng",
    psm: int = 7,  # PSM 7 = single text line, good for individual tokens
    padding: int = 5,
    verbose: bool = False
) -> List[Dict]:
    """
    Re-OCR individual tokens with low TSV confidence.

    For each token where conf < conf_threshold:
    1. Crop token bbox from image (with padding)
    2. Run tesseract on cropped region (psm=7 for single line)
    3. If new result has higher confidence, use it

    Args:
        img_gray: Full page grayscale image (at OCR DPI, e.g. 450)
        tokens: List of OCR tokens with x0, y0, x1, y1, text, conf
        conf_threshold: Tokens below this confidence get re-OCR'd (default 0.6)
        lang: Tesseract language
        psm: Page segmentation mode for re-OCR (default 7 = single line)
        padding: Pixels to expand crop region
        verbose: Print progress

    Returns:
        Updated tokens list with improved text/confidence where applicable
    """
    if not HAVE_CV2:
        return tokens

    reocr_count = 0
    improved_count = 0
    h, w = img_gray.shape[:2]

    for token in tokens:
        # Skip tokens already above threshold
        if token.get('conf', 0) >= conf_threshold:
            continue

        # Get token bbox
        x0 = int(token.get('x0', 0))
        y0 = int(token.get('y0', 0))
        x1 = int(token.get('x1', 0))
        y1 = int(token.get('y1', 0))

        if x1 <= x0 or y1 <= y0:
            continue

        # Expand with padding but stay within image bounds
        crop_x0 = max(0, x0 - padding)
        crop_y0 = max(0, y0 - padding)
        crop_x1 = min(w, x1 + padding)
        crop_y1 = min(h, y1 + padding)

        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            continue

        # Crop token region
        token_img = img_gray[crop_y0:crop_y1, crop_x0:crop_x1]

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), token_img)

        try:
            # Run tesseract with PSM 7 (single text line) for better token OCR
            tsv_text, error = run_tesseract_tsv(tmp_path, lang, psm)

            if error or not tsv_text:
                continue

            # Parse result
            new_tokens = parse_tesseract_tsv(tsv_text)
            reocr_count += 1

            if new_tokens:
                # Get best result (highest confidence)
                best_new = max(new_tokens, key=lambda t: t.get('conf', 0))
                new_conf = best_new.get('conf', 0)
                new_text = best_new.get('text', '')

                # Use new result if confidence improved
                if new_conf > token.get('conf', 0) and new_text.strip():
                    token['text'] = new_text
                    token['conf'] = new_conf
                    token['reocr'] = True  # Flag for debugging
                    improved_count += 1

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if verbose and reocr_count > 0:
        print(f"    Re-OCR'd {reocr_count} low-confidence tokens, {improved_count} improved")

    return tokens
