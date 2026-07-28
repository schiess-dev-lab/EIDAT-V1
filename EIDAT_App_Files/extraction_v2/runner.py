"""V2 node scan runner — completely separate from the legacy admin runner.

scan_node() walks a node root for PDFs, extracts each into
`EIDAT/EIDAT Support/extract_v2/<stem>/document.json`, auto-maps against
the central rule library, and queues unknown shapes. It NEVER blocks on a
human: unknowns accumulate for the Mapping Review screen.

Re-scan is incremental: a document whose sha256 is unchanged is only
re-mapped (cheap), not re-OCR'd.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import store
from .document import extract_document, save_json

_SKIP_DIRS = {"EIDAT", "EIDAT Support", "node_mirror", ".git", "__pycache__"}


def _iter_pdfs(node_root: Path):
    for p in sorted(node_root.rglob("*.pdf")):
        if any(part in _SKIP_DIRS for part in p.relative_to(node_root).parts[:-1]):
            continue
        yield p


def _make_crop_fn(pdf_path: Path, out_dir: Path, scan_dpi: int = 300, crop_dpi: int = 150):
    """Lazy page-crop renderer for queue previews."""

    def crop(page_number: int, bbox) -> str | None:
        try:
            import cv2
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(str(pdf_path))
            try:
                page = doc[page_number - 1]
                arr = page.render(scale=crop_dpi / 72.0, grayscale=True).to_numpy()
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                s = crop_dpi / scan_dpi
                x0, y0, x1, y1 = (int(v * s) for v in bbox)
                pad = 12
                h, w = arr.shape
                crop_img = arr[
                    max(0, y0 - pad) : min(h, y1 + pad),
                    max(0, x0 - pad) : min(w, x1 + pad),
                ]
                out = out_dir / f"crop_p{page_number}_{x0}_{y0}.png"
                cv2.imwrite(str(out), crop_img)
                return str(out)
            finally:
                doc.close()
        except Exception:
            return None

    return crop


def scan_node(
    node_root: str | Path,
    runtime_root: str | Path,
    dpi: int = 300,
    log=print,
) -> dict:
    node_root = Path(node_root)
    runtime_root = Path(runtime_root)
    out_root = store.node_store_dir(node_root)
    out_root.mkdir(parents=True, exist_ok=True)

    node_db = store.open_node(node_root)
    rules_db = store.open_rules(runtime_root)

    n_docs = n_new = n_facts = n_queued = 0
    t0 = time.time()
    for pdf in _iter_pdfs(node_root):
        n_docs += 1
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        row = node_db.execute(
            "SELECT doc_id, sha256, json_path FROM docs WHERE file=?",
            (str(pdf),),
        ).fetchone()

        doc_dir = out_root / pdf.stem
        doc_dir.mkdir(exist_ok=True)
        json_path = doc_dir / "document.json"

        if row and row[1] == sha and Path(row[2]).exists():
            doc_id = row[0]
            doc = json.loads(Path(row[2]).read_text())
            log(f"[V2] unchanged, re-map only: {pdf.name}")
        else:
            log(f"[V2] extracting: {pdf.name}")
            doc = extract_document(pdf, dpi=dpi)
            save_json(doc, json_path)
            meta = {k.upper(): v for k, v in doc.get("metadata", {}).items()}
            node_db.execute(
                "INSERT INTO docs(file, sha256, json_path, program, serial,"
                " test_date, scanned_ts) VALUES(?,?,?,?,?,?,?)"
                " ON CONFLICT(file) DO UPDATE SET sha256=excluded.sha256,"
                " json_path=excluded.json_path, program=excluded.program,"
                " serial=excluded.serial, test_date=excluded.test_date,"
                " scanned_ts=excluded.scanned_ts",
                (
                    str(pdf),
                    sha,
                    str(json_path),
                    meta.get("PROGRAM"),
                    meta.get("SERIAL"),
                    meta.get("TEST DATE"),
                    time.time(),
                ),
            )
            if row and row[1] != sha:
                # content changed: prior facts/queue entries are stale
                node_db.execute("DELETE FROM facts WHERE doc_id=?", (row[0],))
                node_db.execute("DELETE FROM queue WHERE doc_id=?", (row[0],))
            doc_id = node_db.execute(
                "SELECT doc_id FROM docs WHERE file=?", (str(pdf),)
            ).fetchone()[0]
            node_db.commit()
            n_new += 1

        f, q = store.map_document(
            node_db,
            rules_db,
            doc_id,
            doc,
            crop_fn=_make_crop_fn(pdf, doc_dir, scan_dpi=dpi),
        )
        n_facts += f
        n_queued += q

    pending = store.pending_count(node_db)
    summary = {
        "docs": n_docs,
        "extracted": n_new,
        "facts_written": n_facts,
        "newly_queued": n_queued,
        "pending_mappings": pending,
        "elapsed_s": round(time.time() - t0, 1),
    }
    log(f"[V2] done: {summary}")
    node_db.close()
    rules_db.close()
    return summary


def replay(node_root: str | Path, runtime_root: str | Path, log=print) -> dict:
    node_db = store.open_node(node_root)
    rules_db = store.open_rules(runtime_root)
    facts, queued = store.replay_node(node_db, rules_db)
    pending = store.pending_count(node_db)
    log(f"[V2] replay: +{facts} facts, {pending} still pending")
    node_db.close()
    rules_db.close()
    return {"facts_written": facts, "pending_mappings": pending}
