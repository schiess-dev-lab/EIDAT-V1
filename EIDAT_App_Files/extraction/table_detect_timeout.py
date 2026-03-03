"""
Hard-timeout wrapper for bordered table detection.

Purpose: prevent pathological pages (dense chart grids / "too many cells") from hanging the
extraction pipeline. On timeout, we return an empty result so the caller can continue/skip.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import multiprocessing
from multiprocessing import shared_memory


def _detect_tables_worker(
    shm_name: str,
    shape: tuple[int, int],
    dtype_str: str,
    verbose: bool,
    detect_kwargs: Dict[str, Any],
    out_q,
) -> None:
    shm: Optional[shared_memory.SharedMemory] = None
    try:
        import numpy as np
        from . import table_detection

        shm = shared_memory.SharedMemory(name=str(shm_name))
        img = np.ndarray(tuple(shape), dtype=np.dtype(dtype_str), buffer=shm.buf)
        img_local = img.copy()
        out_q.put(("ok", table_detection.detect_tables(img_local, verbose=bool(verbose), **(detect_kwargs or {}))))
    except Exception as e:
        out_q.put(("err", f"{type(e).__name__}: {e}"))
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def detect_tables_hard_timeout(
    img_gray_hires: object,
    *,
    verbose: bool,
    timeout_sec: int,
    detect_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run bordered table detection with a hard timeout.

    Returns:
      - On success: the dict returned by `table_detection.detect_tables`.
      - On timeout/error: {"tables": [], "cells": [], "timed_out": True/False, "reason": "..."}
    """
    timeout_sec_i = int(timeout_sec) if int(timeout_sec) > 0 else 0
    detect_kwargs = dict(detect_kwargs or {})
    try:
        import numpy as np
        from . import table_detection

        if img_gray_hires is None:
            return {"tables": [], "cells": [], "timed_out": False, "reason": "no_image"}

        # If the input is not a plain grayscale uint8 image, fall back to direct detection.
        if not isinstance(img_gray_hires, np.ndarray) or img_gray_hires.dtype != np.uint8 or img_gray_hires.ndim != 2:
            out = table_detection.detect_tables(img_gray_hires, verbose=bool(verbose), **detect_kwargs)
            if isinstance(out, dict):
                return out
            return {"tables": [], "cells": [], "timed_out": False, "reason": "unexpected_result"}

        if timeout_sec_i <= 0:
            out = table_detection.detect_tables(img_gray_hires, verbose=bool(verbose), **detect_kwargs)
            if isinstance(out, dict):
                return out
            return {"tables": [], "cells": [], "timed_out": False, "reason": "unexpected_result"}

        shm = shared_memory.SharedMemory(create=True, size=int(img_gray_hires.nbytes))
        try:
            buf = np.ndarray(img_gray_hires.shape, dtype=img_gray_hires.dtype, buffer=shm.buf)
            buf[:] = img_gray_hires

            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=_detect_tables_worker,
                args=(
                    shm.name,
                    tuple(int(x) for x in img_gray_hires.shape),
                    str(img_gray_hires.dtype),
                    bool(verbose),
                    detect_kwargs,
                    q,
                ),
                daemon=True,
            )
            proc.start()
            proc.join(timeout=float(timeout_sec_i))

            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
                proc.join(timeout=5)
                return {"tables": [], "cells": [], "timed_out": True, "reason": f"timeout:{timeout_sec_i}s"}

            try:
                status, payload = q.get_nowait()
            except Exception:
                status, payload = ("err", "No result returned from detect_tables worker")
            if status == "ok" and isinstance(payload, dict):
                return payload
            return {"tables": [], "cells": [], "timed_out": False, "reason": f"worker_error:{payload}"}
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
        # Best-effort: never crash a page because timeouts couldn't be enforced.
        try:
            from . import table_detection

            out = table_detection.detect_tables(img_gray_hires, verbose=bool(verbose), **detect_kwargs)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
        return {"tables": [], "cells": [], "timed_out": False, "reason": "timeout_wrapper_failed"}

