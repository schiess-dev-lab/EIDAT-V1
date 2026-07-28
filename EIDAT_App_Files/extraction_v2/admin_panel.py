"""Admin Dashboard integration for Extraction V2.

Provides the cell widget for the dashboard's dedicated "Extraction V2"
column. Entirely separate from the legacy runner: its buttons call only
extraction_v2.runner and the Mapping Review dialog. If this package (or a
dependency) is unavailable, admin_gui degrades gracefully — the legacy
dashboard is never affected.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets  # type: ignore

from . import store


class V2ScanWorker(QtCore.QThread):
    log = QtCore.Signal(str)
    done = QtCore.Signal(dict)

    def __init__(self, node_root: str, runtime_root: str, parent=None):
        super().__init__(parent)
        self._node_root = node_root
        self._runtime_root = runtime_root

    def run(self) -> None:  # type: ignore[override]
        from .runner import scan_node

        try:
            summary = scan_node(
                self._node_root, self._runtime_root, log=lambda s: self.log.emit(str(s))
            )
        except Exception as exc:
            self.log.emit(f"[V2][ERROR] {exc}")
            summary = {"error": str(exc)}
        self.done.emit(summary)


def _pending(node_root: str) -> int:
    try:
        db = store.open_node(node_root)
        n = store.pending_count(db)
        db.close()
        return n
    except Exception:
        return 0


def make_v2_cell(win, node_id: str, node_root: str, runtime_root: str) -> QtWidgets.QWidget:
    """Build the per-node cell for the 'Extraction V2' column.

    `win` is the AdminWindow; only its `_append_log` is used (for run logs).
    """
    cell = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(cell)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)

    btn_scan = QtWidgets.QPushButton("Scan V2")
    pending = _pending(node_root)
    btn_review = QtWidgets.QPushButton(
        f"Review ({pending})" if pending else "Review"
    )
    if pending:
        btn_review.setStyleSheet("font-weight:bold; color:#b45309;")
    for b in (btn_scan, btn_review):
        b.setMinimumHeight(26)
        b.setMaximumWidth(150)
    btn_scan.setToolTip(
        "Extraction V2: geometric table+KV extraction to document.json, "
        "auto-map via the central rule library, queue unknown shapes. "
        "Does not touch legacy artifacts."
    )
    btn_review.setToolTip("Name unidentified tables/KV groups; rules replay retroactively.")

    def _on_scan(_=False):
        worker = V2ScanWorker(node_root, runtime_root, parent=win)
        cell._v2_worker = worker  # keep a reference
        btn_scan.setEnabled(False)
        btn_scan.setText("Scanning…")
        worker.log.connect(win._append_log)

        def _finish(summary: dict):
            btn_scan.setEnabled(True)
            btn_scan.setText("Scan V2")
            n = summary.get("pending_mappings", _pending(node_root))
            btn_review.setText(f"Review ({n})" if n else "Review")
            btn_review.setStyleSheet(
                "font-weight:bold; color:#b45309;" if n else ""
            )
            win._append_log(f"[V2] {Path(node_root).name}: scan finished {summary}")

        worker.done.connect(_finish)
        worker.start()

    def _on_review(_=False):
        from .review_dialog import MappingReviewDialog

        dlg = MappingReviewDialog(node_root, runtime_root, parent=win)
        dlg.exec()
        n = _pending(node_root)
        btn_review.setText(f"Review ({n})" if n else "Review")
        btn_review.setStyleSheet("font-weight:bold; color:#b45309;" if n else "")

    btn_scan.clicked.connect(_on_scan)
    btn_review.clicked.connect(_on_review)
    lay.addWidget(btn_scan)
    lay.addWidget(btn_review)
    return cell
