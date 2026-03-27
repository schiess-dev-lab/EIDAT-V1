from __future__ import annotations

import re
import html
import json
import time
import shutil
import sys
import threading
import math
import statistics
import datetime
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

# Allow running as a module or as a script
try:
    from . import backend as be  # type: ignore
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    import ui_next.backend as be  # type: ignore


def _fit_widget_to_screen(widget: QtWidgets.QWidget, margin: int = 40) -> None:
    try:
        screen = QtGui.QGuiApplication.screenAt(widget.frameGeometry().center())
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        max_w = max(320, available.width() - margin * 2)
        max_h = max(240, available.height() - margin * 2)
        widget.setMaximumSize(max_w, max_h)
        widget.resize(min(widget.width(), max_w), min(widget.height(), max_h))
        frame = widget.frameGeometry()
        frame.moveCenter(available.center())
        widget.move(frame.topLeft())
    except Exception:
        pass


class _CollapsibleSection(QtWidgets.QFrame):
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, body: QtWidgets.QWidget, *, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._body = body
        self._expanded = False
        self.setStyleSheet("QFrame { background: transparent; border: 0; }")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_toggle = QtWidgets.QToolButton(self)
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(bool(expanded))
        self.btn_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_toggle.setStyleSheet(
            """
            QToolButton {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 800;
                color: #0f172a;
                text-align: left;
            }
            QToolButton:hover { background: #f8fafc; }
            """
        )
        self.btn_toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_toggle.clicked.connect(lambda checked: self.set_expanded(bool(checked)))
        layout.addWidget(self.btn_toggle)
        layout.addWidget(self._body)
        self.set_title(title)
        self.set_expanded(bool(expanded))

    def set_title(self, title: str) -> None:
        self.btn_toggle.setText(str(title or "").strip())

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self._expanded = expanded
        self.btn_toggle.blockSignals(True)
        self.btn_toggle.setChecked(expanded)
        self.btn_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )
        self.btn_toggle.blockSignals(False)
        self._body.setVisible(expanded)
        self.updateGeometry()
        self.toggled.emit(expanded)


def _td_serial_metadata_by_serial(rows: list[dict]) -> dict[str, dict]:
    by_sn: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        sn = str(row.get("serial") or row.get("serial_number") or "").strip()
        if sn and sn not in by_sn:
            by_sn[sn] = row
    return by_sn


TD_UNKNOWN_PROGRAM_LABEL = "Unknown Program"


def _td_display_program_title(value: object) -> str:
    return str(value or "").strip() or TD_UNKNOWN_PROGRAM_LABEL


def _td_serial_value(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("serial") or row.get("serial_number") or "").strip()


def _td_compact_filter_value(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        try:
            num = float(value)
        except Exception:
            return ""
        if not math.isfinite(num):
            return ""
        return f"{num:g}"
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        num = float(raw)
    except Exception:
        return raw
    if not math.isfinite(num):
        return raw
    return f"{num:g}"


def _td_suppression_voltage_filter_value(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ""
    return _td_compact_filter_value(row.get("suppression_voltage"))


def _td_control_period_filter_value(row: dict | None) -> str:
    if not isinstance(row, dict):
        return ""
    return _td_compact_filter_value(row.get("control_period"))


def _td_compact_filter_sort_key(value: object) -> tuple[int, float, str]:
    label = _td_compact_filter_value(value)
    if not label:
        return (2, 0.0, "")
    try:
        return (0, float(label), label.casefold())
    except Exception:
        return (1, 0.0, label.casefold())


def _td_metric_program_segments(labels: list[str], serial_rows: list[dict]) -> list[dict]:
    meta_by_sn = _td_serial_metadata_by_serial(serial_rows)
    segments: list[dict] = []
    for idx, raw_sn in enumerate(labels or []):
        sn = str(raw_sn or "").strip()
        if not sn:
            continue
        row = meta_by_sn.get(sn) or {}
        program = _td_display_program_title(row.get("program_title"))
        if segments and str(segments[-1].get("program") or "") == program:
            segments[-1]["end"] = idx
            serials = segments[-1].setdefault("serials", [])
            if isinstance(serials, list):
                serials.append(sn)
        else:
            segments.append({
                "program": program,
                "start": idx,
                "end": idx,
                "serials": [sn],
            })
    return segments


def _td_order_metric_serials(labels: list[str], serial_rows: list[dict]) -> list[str]:
    meta_by_sn = _td_serial_metadata_by_serial(serial_rows)
    serials: list[str] = []
    seen: set[str] = set()
    for raw_sn in labels or []:
        sn = str(raw_sn or "").strip()
        if not sn or sn in seen:
            continue
        seen.add(sn)
        serials.append(sn)

    def _sort_key(sn: str) -> tuple[int, str, str]:
        row = meta_by_sn.get(sn) or {}
        program = _td_display_program_title(row.get("program_title"))
        return (
            1 if program == TD_UNKNOWN_PROGRAM_LABEL else 0,
            program.casefold(),
            sn.casefold(),
        )

    return sorted(serials, key=_sort_key)


class ProcWorker(QtCore.QThread):
    line = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self, popen_factory, parent=None):
        super().__init__(parent)
        self._popen_factory = popen_factory
        self._stop = threading.Event()
        self._proc = None

    def run(self):
        rc = 0
        try:
            self._proc = self._popen_factory()
            stream = self._proc.stdout
            if stream is not None:
                for line in stream:
                    if self._stop.is_set():
                        break
                    self.line.emit(line.rstrip("\n"))
            rc = self._proc.wait()
        except Exception as e:
            self.line.emit(f"[ERROR] {e}")
            rc = 1
        finally:
            self.finished.emit(rc)

    def stop(self):
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass


class WorkspaceSyncWorker(QtCore.QThread):
    completed = QtCore.Signal(object, object, object)
    failed = QtCore.Signal(str)

    def __init__(self, repo: Path, terms: Path, auto: bool, parent=None):
        super().__init__(parent)
        self.repo = Path(repo)
        self.terms = Path(terms)
        self.auto = auto

    def run(self):
        warnings: list[str] = []
        try:
            if not self.auto:
                try:
                    be.rebuild_registry_from_run_data()
                except Exception as e:
                    warnings.append(f"Registry rebuild failed: {e}")
                try:
                    from scripts.master_cell_state import sync_cell_state_with_master
                    sync_cell_state_with_master()
                except Exception as e:
                    warnings.append(f"Cell state sync failed: {e}")
            summary, details = be.compute_workspace_sync(self.repo, self.terms)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.completed.emit(summary, details, warnings)


class ManagerTaskWorker(QtCore.QThread):
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self._task = task

    def run(self):
        try:
            payload = self._task()
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.completed.emit(payload)


class ProjectTaskWorker(QtCore.QThread):
    progress = QtCore.Signal(str)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, task_factory, *, log_path: Path | None = None, parent=None):
        super().__init__(parent)
        self._task_factory = task_factory
        self._log_path = Path(log_path).expanduser() if log_path else None

    def _write_log_line(self, handle, text: str) -> None:
        if handle is None:
            return
        try:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"[{stamp}] {text}\n")
            handle.flush()
        except Exception:
            pass

    def run(self):
        handle = None
        try:
            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = self._log_path.open("a", encoding="utf-8")

            def _report(message: str) -> None:
                txt = str(message or "").strip()
                if not txt:
                    return
                self._write_log_line(handle, txt)
                self.progress.emit(txt)

            if self._log_path is not None:
                _report(f"Log file: {self._log_path}")

            payload = self._task_factory(_report)
            if isinstance(payload, dict) and self._log_path is not None:
                payload = dict(payload)
                payload["log_path"] = str(self._log_path)
                timings = payload.get("timings")
                if isinstance(timings, dict):
                    self._write_log_line(handle, f"Timings JSON: {json.dumps(timings, sort_keys=True)}")
            self.completed.emit(payload)
        except Exception as e:
            msg = str(e or "").strip() or type(e).__name__
            if self._log_path is not None:
                for line in (msg.splitlines() or [msg]):
                    self._write_log_line(handle, f"ERROR: {line}")
                self.failed.emit(f"{msg}\n\nLog: {self._log_path}")
            else:
                self.failed.emit(msg)
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass


class RunProgressDialog(QtWidgets.QDialog):
    """Large popup that visualizes term progress with a simple spinner animation."""

    canceled = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running EIDP Scanner")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.resize(480, 260)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0b1526;
                color: #f6fbff;
            }
            QDialog QLabel {
                color: #f6fbff;
            }
            QProgressBar {
                background-color: #14253d;
                color: #0b1526;
                border: 1px solid #284364;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #3db6ff;
            }
            QPushButton[variant="ghost"] {
                background: transparent;
                color: #f6fbff;
                border: 1px solid #3db6ff;
            }
            QPushButton[variant="ghost"]:disabled {
                color: #94a6c3;
                border-color: #2a3b52;
            }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(16)

        self.lbl_heading = QtWidgets.QLabel("Executing run...")
        font = self.lbl_heading.font()
        font.setPointSize(18)
        font.setBold(True)
        self.lbl_heading.setFont(font)
        self.lbl_heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.lbl_status = QtWidgets.QLabel("Preparing scanner")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 13px;")

        self.spinner_label = QtWidgets.QLabel("•")
        spin_font = self.spinner_label.font()
        spin_font.setPointSize(24)
        spin_font.setBold(False)
        self.spinner_label.setFont(spin_font)
        self.spinner_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate until totals stream in
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Working...")

        self.detail_label = QtWidgets.QLabel("Waiting for scanner progress\u2026")
        self.detail_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setStyleSheet("font-size: 12px; color: #e0e7f0;")

        self.hint_label = QtWidgets.QLabel("This window closes automatically when the run finishes.")
        self.hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("color: #a5b8d6; font-size: 11px;")

        self.btn_cancel = QtWidgets.QPushButton("Abort Run")
        self.btn_cancel.setProperty("variant", "ghost")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        layout.addWidget(self.lbl_heading)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.spinner_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.btn_cancel)

        self._spinner_frames = ["•", "·", "•", "·"]
        self._spinner_index = 0
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(170)
        self._anim_timer.timeout.connect(self._advance_spinner)

    def _advance_spinner(self):
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self.spinner_label.setText(self._spinner_frames[self._spinner_index])

        # Base status text shown at the top of the dialog
        self._base_status_text: str = ""

    def begin(self, status_text: str):
        self._base_status_text = status_text
        self.lbl_status.setText(status_text)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Working...")
        self.detail_label.setText("Waiting for scanner progress\u2026")
        self._spinner_index = 0
        self.spinner_label.setText(self._spinner_frames[0])
        self._anim_timer.start()
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("Abort Run")
        self.adjustSize()
        _fit_widget_to_screen(self)
        self.show()
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def update_progress(
        self,
        completed: int,
        total: int,
        found: int = 0,
        *,
        current_file: str | None = None,
        file_index: int | None = None,
        file_total: int | None = None,
    ):
        if total <= 0:
            if self.progress_bar.maximum() != 0:
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setFormat("Working...")
            if completed <= 0:
                terms_text = "Waiting for term counts"
            else:
                terms_text = f"Searching: {completed} term{'s' if completed != 1 else ''}"
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            pct = max(0, min(100, int(round((completed * 100) / max(1, total)))))
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}%")
            terms_text = f"Searching: {completed} / {total} terms"
        detail_parts = []
        file_label = None
        if current_file:
            idx = (file_index or 0)
            total_files = (file_total or 0)
            if total_files > 0 and idx > 0:
                file_label = f"{current_file} ({idx}/{total_files})"
            elif idx > 0:
                file_label = f"{current_file} ({idx})"
            else:
                file_label = current_file
        elif file_total:
            idx = (file_index or 0)
            if idx > 0:
                file_label = f"EIDPs: {idx}/{file_total}"
            else:
                file_label = f"EIDPs: {file_total}"
        if file_label:
            detail_parts.append(file_label)
        if terms_text:
            detail_parts.append(f"{terms_text} \u2022 Found: {found}")
        if not detail_parts:
            detail_parts.append(f"Found: {found}")
        self.detail_label.setText(" \u2022 ".join(detail_parts))

    def finish(self, message: str, success: bool = True):
        self._anim_timer.stop()
        self.spinner_label.setText("✓" if success else "×")
        self.lbl_status.setText(message)
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else self.progress_bar.value())
        self.progress_bar.setFormat("Done")
        self.btn_cancel.setEnabled(False)
        QtCore.QTimer.singleShot(1200, self.hide)

    def abort(self):
        self._anim_timer.stop()
        self.btn_cancel.setEnabled(False)
        self.hide()

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[override]
        self._anim_timer.stop()
        super().closeEvent(event)

    def _on_cancel_clicked(self):
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Aborting...")
        self.lbl_status.setText("Stopping run...")
        self.canceled.emit()


class RepoScanDialog(QtWidgets.QDialog):
    """Popup that runs a repository sync without blocking the main UI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Repo Scan")
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.resize(460, 230)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0b1526;
                color: #f6fbff;
            }
            QDialog QLabel {
                color: #f6fbff;
            }
            QProgressBar {
                background-color: #14253d;
                color: #0b1526;
                border: 1px solid #284364;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #3db6ff;
            }
            QPushButton {
                background: transparent;
                color: #f6fbff;
                border: 1px solid #3db6ff;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:disabled {
                color: #94a6c3;
                border-color: #2a3b52;
            }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        self.lbl_heading = QtWidgets.QLabel("Global Repo Scan")
        font = self.lbl_heading.font()
        font.setPointSize(16)
        font.setBold(True)
        self.lbl_heading.setFont(font)
        self.lbl_heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.lbl_status = QtWidgets.QLabel("Preparing scan")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 12px;")

        self.lbl_detail = QtWidgets.QLabel("")
        self.lbl_detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #dbeafe; font-size: 11px;")

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Scanning...")

        self.lbl_hint = QtWidgets.QLabel("This window closes automatically when the scan finishes.")
        self.lbl_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_hint.setStyleSheet("color: #a5b8d6; font-size: 11px;")

        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.hide)

        layout.addWidget(self.lbl_heading)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_detail)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_hint)
        layout.addWidget(self.btn_close)

        self._base_status = ""
        self._dot_frames = ["", ".", "..", "..."]
        self._dot_index = 0
        self._dot_timer = QtCore.QTimer(self)
        self._dot_timer.setInterval(280)
        self._dot_timer.timeout.connect(self._advance_dots)

    def _advance_dots(self):
        self._dot_index = (self._dot_index + 1) % len(self._dot_frames)
        self.lbl_status.setText(f"{self._base_status}{self._dot_frames[self._dot_index]}")

    def begin(self, status_text: str, heading: str | None = None):
        if heading:
            self.lbl_heading.setText(heading)
        self._base_status = status_text
        self.lbl_status.setText(status_text)
        self.lbl_detail.setText("")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Scanning...")
        self.btn_close.setEnabled(False)
        self._dot_index = 0
        self._dot_timer.start()
        self.adjustSize()
        _fit_widget_to_screen(self)
        self.show()
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def set_status_text(self, status_text: str) -> None:
        self._base_status = str(status_text or "").strip()
        self.lbl_status.setText(self._base_status)

    def set_detail_text(self, detail_text: str) -> None:
        self.lbl_detail.setText(str(detail_text or "").strip())

    def finish(self, message: str, success: bool = True):
        self._dot_timer.stop()
        self.lbl_status.setText(message)
        self.lbl_detail.setText("")
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else self.progress_bar.value())
        self.progress_bar.setFormat("Done" if success else "Error")
        self.btn_close.setEnabled(True)
        QtCore.QTimer.singleShot(1200, self.hide)

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[override]
        self._dot_timer.stop()
        super().closeEvent(event)


class _DropZone(QtWidgets.QFrame):
    def __init__(self, hint: str, on_drop, parent=None):
        super().__init__(parent)
        self._on_drop = on_drop
        self._label = QtWidgets.QLabel(hint)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(1)
        lay.addWidget(self._label)
        lay.addStretch(1)
        self.setAcceptDrops(True)
        self.setFrameStyle(
            QtWidgets.QFrame.Shape.StyledPanel | QtWidgets.QFrame.Shadow.Plain
        )
        self.setStyleSheet("""
            QFrame { border: 2px dashed #c9d3e3; border-radius: 8px; min-height: 140px; background: #fafbfd; }
            QLabel { color: #5b6b7a; font-size: 14px; }
        """)

    def set_hint(self, text: str):
        self._label.setText(text)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):  # type: ignore[override]
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):  # type: ignore[override]
        e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):  # type: ignore[override]
        try:
            urls = e.mimeData().urls()
            paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
            if paths:
                self._on_drop(paths)
            e.acceptProposedAction()
        except Exception:
            e.ignore()


class ToastNotification(QtWidgets.QWidget):
    """Cookie banner / toast notification that appears at bottom of window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:1 #0f172a);
                border: 2px solid #3b82f6;
                border-radius: 12px;
                padding: 0px;
            }
            QLabel {
                color: #f8fafc;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, message: str, duration: int = 5000):
        """Show toast notification with message for duration milliseconds."""
        self.label.setText(message)
        self.adjustSize()

        # Position at bottom center of parent
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + parent_rect.height() - self.height() - 40
            self.move(x, y)

        self.show()
        self.raise_()
        self._timer.start(duration)


class TermsEditorDialog(QtWidgets.QDialog):
    """In-app editor for the simple schema terms spreadsheet."""

    VISIBLE_COLUMN_ORDER = [
        "Data Group",
        "Term Label",
        "Term",
        "Header",
        "GroupAfter",
        "GroupBefore",
        "Units",
        "Range (min)",
        "Range (max)",
        "Report Mode",
        "Fuzzy",
    ]
    COLUMN_DISPLAY_NAMES = {
        "Data Group": "Data Group",
        "Term Label": "Term Label",
        "Term": "Search Term",
        "Header": "Header",
        "GroupAfter": "Group After",
        "GroupBefore": "Group Before",
        "Units": "Units",
        "Range (min)": "Range (min)",
        "Range (max)": "Range (max)",
        "Report Mode": "Report Mode",
        "Fuzzy": "Fuzzy Threshold",
    }
    DEFAULT_HIDDEN: set[str] = set()

    def __init__(self, terms_path: Path, parent=None):
        super().__init__(parent)
        self._terms_path = Path(terms_path)
        self._all_headers: list[str] = []
        self._visible_headers: list[str] = []
        self._hidden_headers: list[str] = []
        self._hidden_rows: list[dict[str, str]] = []
        self._dirty = False
        self._loading = False
        self._pending_save_notice = False

        self.setWindowTitle("Simple Schema Terms Editor")
        self.resize(1280, 680)
        self.setObjectName("termsEditorDialog")
        self.setStyleSheet("""
            #termsEditorDialog {
                background-color: #f8fafc;
                color: #0f172a;
            }
            #termsEditorDialog QLabel {
                color: #0f172a;
            }
            #termsEditorDialog QLabel#termsPathLabel {
                color: #475569;
            }
            #termsEditorDialog QTableWidget {
                background-color: #ffffff;
                border: 1px solid #d4d4d8;
                gridline-color: #e4e4e7;
                selection-background-color: #d1d5db;
                selection-color: #0f172a;
            }
            #termsEditorDialog QTableWidget::item:selected {
                background-color: #d1d5db;
                color: #0f172a;
            }
            #termsEditorDialog QHeaderView::section {
                background-color: #eef2ff;
                color: #0f172a;
                padding: 8px;
                border: 1px solid #cbd5f5;
                font-weight: 600;
            }
            #termsEditorDialog QPushButton {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5f5;
                border-radius: 6px;
                padding: 8px 14px;
            }
            #termsEditorDialog QPushButton:hover {
                background-color: #f1f5f9;
            }
            #termsEditorDialog QPushButton[variant="primary"] {
                background-color: #2563eb;
                border-color: #2563eb;
                color: #ffffff;
            }
            #termsEditorDialog QPushButton[variant="primary"]:hover {
                background-color: #1d4ed8;
            }
        """)

        self._combo_defs = {
            "Report Mode": {
                "options": self._report_mode_options(),
                "default": "value",
            },
        }

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Edit the simple schema spreadsheet directly in the app.")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #0f172a;")
        layout.addWidget(title)
        hint = QtWidgets.QLabel(f"File: {self._terms_path}")
        hint.setObjectName("termsPathLabel")
        hint.setStyleSheet("color: #556070;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QtWidgets.QTableWidget()
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.table.verticalHeader().setDefaultSectionSize(42)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #d4d4d8;
                gridline-color: #e4e4e7;
                selection-background-color: #d1d5db;
                selection-color: #0f172a;
            }
            QTableWidget::item {
                padding: 10px;
                color: #0f172a;
            }
            QTableWidget::item:selected {
                background-color: #d1d5db;
                color: #0f172a;
            }
            QTableWidget QLineEdit {
                background-color: #ffffff;
                border: 1px solid #94a3b8;
                border-radius: 4px;
                padding: 6px 8px;
                color: #0f172a;
                selection-background-color: #bfdbfe;
                min-height: 26px;
            }
        """)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        # Clicking the row number highlights the full row
        self.table.verticalHeader().sectionClicked.connect(self._on_vertical_header_clicked)
        layout.addWidget(self.table, 1)

        row_btns = QtWidgets.QHBoxLayout()
        self.btn_add_row = QtWidgets.QPushButton("Add Row")
        self.btn_duplicate_row = QtWidgets.QPushButton("Duplicate Row")
        self.btn_move_up = QtWidgets.QPushButton("Move Up")
        self.btn_move_down = QtWidgets.QPushButton("Move Down")
        self.btn_delete_row = QtWidgets.QPushButton("Delete Selected")
        self.btn_add_row.clicked.connect(self._add_blank_row)
        self.btn_duplicate_row.clicked.connect(self._duplicate_row)
        self.btn_move_up.clicked.connect(lambda: self._move_rows(-1))
        self.btn_move_down.clicked.connect(lambda: self._move_rows(1))
        self.btn_delete_row.clicked.connect(self._delete_rows)
        row_btns.addWidget(self.btn_add_row)
        row_btns.addWidget(self.btn_duplicate_row)
        row_btns.addWidget(self.btn_move_up)
        row_btns.addWidget(self.btn_move_down)
        row_btns.addWidget(self.btn_delete_row)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        bottom = QtWidgets.QHBoxLayout()
        self._status_label = QtWidgets.QLabel("Loading...")
        self._status_label.setObjectName("termsStatusLabel")
        self._status_label.setStyleSheet("color: #0f172a; font-weight: 600;")
        bottom.addWidget(self._status_label)
        bottom.addStretch(1)
        self.btn_open_excel = QtWidgets.QPushButton("Open in Excel")
        self.btn_open_excel.clicked.connect(self._open_in_excel)
        self.btn_save_close = QtWidgets.QPushButton("Save & Close")
        self.btn_save_close.setProperty("variant", "primary")
        self.btn_save_close.clicked.connect(self._save_and_close)
        bottom.addWidget(self.btn_open_excel)
        bottom.addWidget(self.btn_save_close)
        layout.addLayout(bottom)

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self, activated=self._save_and_close)
        self.table.itemChanged.connect(self._on_table_item_changed)

        self._load_rows()

    def _report_mode_options(self) -> list[tuple[str, str]]:
        opts: list[tuple[str, str]] = []
        seen: set[str] = set()
        values = be.TERMS_REPORT_CHOICES or ["value", "cell"]
        for opt in values:
            val = (opt or "").strip().lower()
            if not val or val in seen:
                continue
            seen.add(val)
            label = "Value Only" if val == "value" else "Full Cell"
            opts.append((label, val))
        if not opts:
            opts.append(("Value Only", "value"))
        return opts

    def _cell_indices_for_widget(self, widget: QtWidgets.QWidget) -> tuple[int | None, int | None]:
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                if self.table.cellWidget(row, col) is widget:
                    return row, col
        return None, None

    def _determine_visible_headers(self, headers: list[str]) -> None:
        order: list[str] = []
        present = set(headers)
        for col in self.VISIBLE_COLUMN_ORDER:
            if col in present and col not in order:
                order.append(col)
        for h in headers:
            if h in self.DEFAULT_HIDDEN:
                continue
            if h not in order:
                order.append(h)
        self._visible_headers = [h for h in order if h in headers and h not in self.DEFAULT_HIDDEN]
        self._hidden_headers = [h for h in headers if h not in self._visible_headers]

    def _load_rows(self) -> None:
        headers, rows = be.read_terms_rows(self._terms_path)
        self._loading = True
        try:
            self._all_headers = headers
            self._determine_visible_headers(headers)
            self._hidden_rows = []
            self.table.clear()
            self.table.setColumnCount(len(self._visible_headers))
            header_labels = [self._column_display_name(h) for h in self._visible_headers]
            self.table.setHorizontalHeaderLabels(header_labels)
            self.table.setRowCount(0)
            for row in rows:
                idx = self.table.rowCount()
                self.table.insertRow(idx)
                self._hidden_rows.append({h: row.get(h, "") for h in self._hidden_headers})
                self._populate_row(idx, row)
        finally:
            self._loading = False
        self._dirty = False
        self._status_label.setText("All changes saved")

        # Set custom column widths - double width for Data Group and Term Label
        self.table.resizeColumnsToContents()
        for col_idx, header in enumerate(self._visible_headers):
            if header in ("Data Group", "Term Label"):
                current_width = self.table.columnWidth(col_idx)
                self.table.setColumnWidth(col_idx, current_width * 2)

    def _column_display_name(self, header: str) -> str:
        return self.COLUMN_DISPLAY_NAMES.get(header, header)

    def _populate_row(self, row_idx: int, row_data: dict[str, str]) -> None:
        for col_idx, header in enumerate(self._visible_headers):
            value = row_data.get(header, "")
            if header in self._combo_defs:
                combo = self._build_combo(header, value)
                self.table.setCellWidget(row_idx, col_idx, combo)
            else:
                item = QtWidgets.QTableWidgetItem(value)
                self.table.setItem(row_idx, col_idx, item)

    def _build_combo(self, header: str, value: str) -> QtWidgets.QComboBox:
        config = self._combo_defs[header]
        combo = QtWidgets.QComboBox()
        combo.setEditable(False)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #94a3b8;
                border-radius: 4px;
                padding: 4px;
                color: #0f172a;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #0f172a;
                selection-background-color: #bfdbfe;
                selection-color: #0f172a;
            }
        """)
        seen_values = set()
        for label, val in config["options"]:
            combo.addItem(label, val)
            seen_values.add(val)
        if value and value not in seen_values:
            combo.addItem(value, value)
        target = value if value else config.get("default", "")
        combo.blockSignals(True)
        idx = combo.findData(target)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        combo.setProperty("header", header)
        combo.currentIndexChanged.connect(self._on_combo_changed)
        return combo

    def _on_combo_changed(self, *args) -> None:
        if self._loading:
            return
        self._mark_dirty()

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._loading:
            return
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if self._dirty:
            return
        self._dirty = True
        self._status_label.setText("Unsaved changes")

    def _selected_rows(self) -> list[int]:
        rows = {idx.row() for idx in self.table.selectionModel().selectedRows()}
        return sorted(rows)

    def _on_vertical_header_clicked(self, logical_index: int) -> None:
        try:
            self.table.selectRow(logical_index)
        except Exception:
            pass

    def _add_blank_row(self) -> None:
        payload = {h: "" for h in self._all_headers}
        default_report = "value"
        if getattr(be, "TERMS_REPORT_CHOICES", None):
            if default_report not in be.TERMS_REPORT_CHOICES:
                default_report = be.TERMS_REPORT_CHOICES[0]
        payload.setdefault("Report Mode", default_report)
        self._insert_row(payload)
        self.table.scrollToBottom()
        self._mark_dirty()

    def _duplicate_row(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        source = rows[0]
        payload = self._combine_row_payload(source)
        self._insert_row(payload)
        self.table.scrollToBottom()
        self._mark_dirty()

    def _delete_rows(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        for row in reversed(rows):
            self.table.removeRow(row)
            if 0 <= row < len(self._hidden_rows):
                self._hidden_rows.pop(row)
        if self.table.rowCount() == 0:
            self._insert_row({h: "" for h in self._all_headers})
        self._mark_dirty()

    def _move_rows(self, direction: int) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        max_row = self.table.rowCount() - 1
        if max_row <= 0:
            return
        if direction < 0 and rows[0] == 0:
            return
        if direction > 0 and rows[-1] >= max_row:
            return
        payloads = [self._combine_row_payload(i) for i in range(self.table.rowCount())]
        if direction < 0:
            for row in rows:
                payloads[row - 1], payloads[row] = payloads[row], payloads[row - 1]
        else:
            for row in reversed(rows):
                payloads[row + 1], payloads[row] = payloads[row], payloads[row + 1]
        self._reset_table_from_payloads(payloads)
        sel = self.table.selectionModel()
        sel.clearSelection()
        new_rows = [row + direction for row in rows]
        for row in new_rows:
            index = self.table.model().index(row, 0)
            sel.select(
                index,
                QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows,
            )
        if new_rows:
            self.table.scrollTo(self.table.model().index(new_rows[0], 0))
        self._mark_dirty()

    def _insert_row(self, payload: dict[str, str]) -> None:
        self._loading = True
        try:
            idx = self.table.rowCount()
            self.table.insertRow(idx)
            hidden_payload = {h: payload.get(h, "") for h in self._hidden_headers}
            if idx <= len(self._hidden_rows):
                self._hidden_rows.insert(idx, hidden_payload)
            else:
                self._hidden_rows.append(hidden_payload)
            self._populate_row(idx, payload)
        finally:
            self._loading = False

    def _reset_table_from_payloads(self, payloads: list[dict[str, str]]) -> None:
        self._loading = True
        try:
            self.table.setRowCount(0)
            self._hidden_rows = []
            if not payloads:
                payloads = [{h: "" for h in self._all_headers}]
            for payload in payloads:
                idx = self.table.rowCount()
                self.table.insertRow(idx)
                self._hidden_rows.append({h: payload.get(h, "") for h in self._hidden_headers})
                self._populate_row(idx, payload)
        finally:
            self._loading = False

    def _row_data_from_table(self, row_idx: int) -> dict[str, str]:
        data: dict[str, str] = {}
        for col_idx, header in enumerate(self._visible_headers):
            widget = self.table.cellWidget(row_idx, col_idx)
            if isinstance(widget, QtWidgets.QComboBox):
                current = widget.currentData()
                val = current if current is not None else widget.currentText()
                data[header] = str(val)
                continue
            item = self.table.item(row_idx, col_idx)
            data[header] = item.text() if item else ""
        return data

    def _combine_row_payload(self, row_idx: int) -> dict[str, str]:
        payload: dict[str, str] = {}
        payload.update(self._hidden_rows[row_idx] if row_idx < len(self._hidden_rows) else {})
        payload.update(self._row_data_from_table(row_idx))
        for header in self._all_headers:
            payload.setdefault(header, "")
        return payload

    def _gather_rows(self) -> list[dict[str, str]]:
        rows_out: list[dict[str, str]] = []
        for row_idx in range(self.table.rowCount()):
            combined = self._combine_row_payload(row_idx)
            if not any((combined.get(h, "") or "").strip() for h in self._visible_headers if h in combined):
                continue
            rows_out.append({h: combined.get(h, "") for h in self._all_headers})
        if not rows_out:
            rows_out.append({h: "" for h in self._all_headers})
        return rows_out

    def _save_rows(self) -> bool:
        try:
            rows = self._gather_rows()
            be.write_terms_rows(rows, path=self._terms_path, headers=self._all_headers)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self._dirty = False
        self._status_label.setText("All changes saved")
        self._pending_save_notice = True
        return True

    def _save_and_close(self) -> None:
        """Save changes and close the dialog."""
        if self._save_rows():
            self.accept()  # Close with success code

    def _open_in_excel(self) -> None:
        """Open the terms file in Excel (or default spreadsheet application)."""
        try:
            import subprocess
            import platform

            # Save any pending changes first
            if self._dirty:
                resp = QtWidgets.QMessageBox.question(
                    self,
                    "Save before opening?",
                    "You have unsaved changes. Save before opening in Excel?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No | QtWidgets.QMessageBox.StandardButton.Cancel,
                )
                if resp == QtWidgets.QMessageBox.StandardButton.Cancel:
                    return
                elif resp == QtWidgets.QMessageBox.StandardButton.Yes:
                    if not self._save_rows():
                        return

            file_path = str(self._terms_path.resolve())

            # Open file with default application
            if platform.system() == 'Windows':
                subprocess.Popen(['start', 'excel', file_path], shell=True)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', file_path])
            else:  # Linux
                subprocess.Popen(['xdg-open', file_path])

            self._status_label.setText("Opened in Excel")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open failed", f"Could not open file in Excel:\n{exc}")

    def reject(self) -> None:  # type: ignore[override]
        if self._dirty:
            resp = QtWidgets.QMessageBox.question(
                self,
                "Discard unsaved edits?",
                "You have unsaved changes. Close without saving?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        super().reject()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.matches(QtGui.QKeySequence.StandardKey.Save):
            if self._save_rows():
                event.accept()
                return
        super().keyPressEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        _fit_widget_to_screen(self)


# NOTE: SeriesDropdown, PlotRowWidget, and ProposedPlotsDialog classes were removed
# as they were part of the unused Plot tab UI (see git history for original code).


class NewProjectWizardDialog(QtWidgets.QDialog):
    def __init__(self, global_repo: Path, parent=None):
        super().__init__(parent)
        self._global_repo = Path(global_repo).expanduser()
        self.project_meta: dict | None = None

        self.setWindowTitle("Create New Project")
        self.resize(900, 620)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog { background: #ffffff; color: #1f2937; }
            QLabel { color: #1f2937; }
            QLineEdit, QComboBox {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 10px;
                min-height: 28px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
                border: 1px solid #d1d5db;
            }
            QRadioButton { color: #1f2937; spacing: 8px; }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #6b7280;
                border-radius: 9px;
                background: #ffffff;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #2563eb;
                background: #2563eb;
            }
            QRadioButton::indicator:checked::after {
                background: #ffffff;
            }
            QCheckBox { color: #1f2937; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #6b7280;
                border-radius: 4px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #2563eb;
                background: #2563eb;
            }
            QPushButton {
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f9fafb; }
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f9fafb;
                color: #1f2937;
                gridline-color: #e5e7eb;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
            }
            QTableWidget::item { color: #1f2937; }
            QListWidget {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #1f2937;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #1f2937;
                padding: 6px 8px;
                border: 1px solid #e5e7eb;
                font-weight: 600;
            }
            """
        )

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        title = QtWidgets.QLabel("New Project")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #0f172a;")
        self.lbl_subtitle = QtWidgets.QLabel("Projects live inside the selected Global Repo and start as a project workbook.")
        self.lbl_subtitle.setStyleSheet("font-size: 12px; color: #475569;")
        v.addWidget(title)
        v.addWidget(self.lbl_subtitle)

        self._stack = QtWidgets.QStackedWidget()
        v.addWidget(self._stack, 1)

        self._page_details = QtWidgets.QWidget()
        self._page_select = QtWidgets.QWidget()
        self._stack.addWidget(self._page_details)
        self._stack.addWidget(self._page_select)

        self._build_page_details()
        self._build_page_select()

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_back = QtWidgets.QPushButton("Back")
        self.btn_next = QtWidgets.QPushButton("Next")
        self.btn_create = QtWidgets.QPushButton("Create Project")
        self.btn_back.clicked.connect(self._act_back)
        self.btn_next.clicked.connect(self._act_next)
        self.btn_create.clicked.connect(self._act_create)
        btns.addWidget(self.btn_back)
        btns.addWidget(self.btn_next)
        btns.addWidget(self.btn_create)
        v.addLayout(btns)

        self._refresh_filters()
        self._apply_filter_and_refresh_table(select_all=True)
        self._update_nav()

    def _build_page_details(self) -> None:
        form = QtWidgets.QFormLayout(self._page_details)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self.ed_name = QtWidgets.QLineEdit()
        self.ed_name.setPlaceholderText("Enter project name...")
        self.ed_name.setText("EIDP Trending Project")
        self.ed_name.textChanged.connect(lambda _: self._update_nav())

        self.cb_type = QtWidgets.QComboBox()
        self.cb_type.addItems([
            getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"),
            getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
        ])
        self.cb_type.currentIndexChanged.connect(self._on_project_type_changed)

        loc_row = QtWidgets.QHBoxLayout()
        self.ed_location = QtWidgets.QLineEdit(str(getattr(be, "eidat_projects_root")(self._global_repo)))
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_location)
        loc_row.addWidget(self.ed_location, 1)
        loc_row.addWidget(self.btn_browse)
        self.ed_location.textChanged.connect(lambda _: self._update_nav())

        form.addRow("Project Name", self.ed_name)
        form.addRow("Project Type", self.cb_type)
        form.addRow("Project Location (folder)", loc_row)

        hint = QtWidgets.QLabel("A new folder is created at: Location / Project Name")
        hint.setStyleSheet("color:#64748b; font-size: 11px;")
        form.addRow("", hint)

        # Term population options - clear choice between two modes
        term_label = QtWidgets.QLabel("Term Population")
        term_label.setStyleSheet("font-weight: 600; color: #374151; margin-top: 12px;")
        form.addRow("", term_label)

        self.rb_auto_populate = QtWidgets.QRadioButton("Auto-populate from extracted acceptance data")
        self.rb_auto_populate.setChecked(True)
        self.rb_auto_populate.setToolTip(
            "Automatically extracts terms and values from combined.txt for each EIDP.\n"
            "Only terms found in the extraction are added. Values are pre-filled."
        )

        self.rb_blank = QtWidgets.QRadioButton("Blank workbook (manual entry)")
        self.rb_blank.setToolTip(
            "Creates a workbook with serial columns only.\n"
            "You manually add terms and values."
        )

        form.addRow("", self.rb_auto_populate)
        form.addRow("", self.rb_blank)
        self._on_project_type_changed()

    def _on_project_type_changed(self) -> None:
        ptype = str(self.cb_type.currentText() or "").strip()
        is_raw = ptype == getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending")
        is_test_data = ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending")
        is_eidp = ptype == getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending")
        if is_raw or is_test_data:
            self.rb_auto_populate.setChecked(False)
            self.rb_blank.setChecked(True)
            self.rb_auto_populate.setEnabled(False)
            self.rb_blank.setEnabled(False)
        else:
            self.rb_auto_populate.setEnabled(True)
            self.rb_blank.setEnabled(True)

        try:
            if is_test_data and (self.ed_name.text() or "").strip() == "EIDP Trending Project":
                self.ed_name.setText("Test Data Trending Project")
            elif is_eidp and (self.ed_name.text() or "").strip() == "Test Data Trending Project":
                self.ed_name.setText("EIDP Trending Project")
        except Exception:
            pass

        cont = getattr(self, "_continued_container", None)
        if cont is not None:
            cont.setEnabled(not is_raw)
            if is_raw:
                for cb in (self.cb_cont_program, self.cb_cont_part, self.cb_cont_vendor, self.cb_cont_asset, getattr(self, "cb_cont_asset_specific", None), getattr(self, "cb_cont_test_plan", None)):
                    if cb is None:
                        continue
                    cb.setChecked(False)

        try:
            if is_test_data:
                self.lbl_subtitle.setText(
                    "Test Data Trending projects are created from TD reports with extracted Excel SQLite packages."
                )
            else:
                self.lbl_subtitle.setText(
                    "Projects live inside the selected Global Repo and start as a project workbook."
                )
        except Exception:
            pass

        # Allow the same program/asset/group pre-filtering for Test Data projects as for EIDP projects.
        try:
            if is_test_data:
                self.rb_all.setChecked(True)
                self.rb_all.setText("All indexed TD reports")
                for w in (self.rb_program, self.rb_asset, self.rb_asset_specific, self.rb_group):
                    w.setEnabled(True)
                for w in (self.cb_program, self.btn_program_multi, self.cb_asset, self.cb_asset_specific, self.cb_group):
                    w.setEnabled(True)
                if hasattr(self, "lbl_select_hint"):
                    self.lbl_select_hint.setText(
                        "Select TD reports only (must have extracted Excel data). EIDP Trending projects cannot be created from TD reports."
                    )
                if hasattr(self, "lbl_select_hint_sub"):
                    self.lbl_select_hint_sub.setText(
                        "Tip: keep this list short by selecting continued population filters below (optional)."
                    )
                if hasattr(self, "_hint_frame"):
                    self._hint_frame.setStyleSheet(
                        """
                        QFrame {
                            background: #eff6ff;
                            border: 1px solid #93c5fd;
                            border-radius: 8px;
                        }
                        """
                    )
                if hasattr(self, "_cont_label"):
                    self._cont_label.setVisible(True)
                    self._cont_label.setText("Allow Continued Test Data Population By:")
                if hasattr(self, "_cont_hint"):
                    self._cont_hint.setVisible(True)
                    self._cont_hint.setText(
                        "Any future TD Excel document that matches these metadata values is auto-added to the project on refresh/update."
                    )
                if hasattr(self, "_continued_container"):
                    self._continued_container.setVisible(True)
            else:
                self.rb_all.setText("All indexed EIDPs")
                for w in (self.rb_program, self.rb_asset, self.rb_asset_specific, self.rb_group):
                    w.setEnabled(True)
                for w in (self.cb_program, self.btn_program_multi, self.cb_asset, self.cb_asset_specific, self.cb_group):
                    w.setEnabled(True)
                if hasattr(self, "lbl_select_hint"):
                    self.lbl_select_hint.setText(
                        "Select EIDPs only (non-TD). Use Test Data Trending for TD reports."
                    )
                if hasattr(self, "lbl_select_hint_sub"):
                    self.lbl_select_hint_sub.setText("")
                if hasattr(self, "_hint_frame"):
                    self._hint_frame.setStyleSheet(
                        """
                        QFrame {
                            background: #f8fafc;
                            border: 1px solid #e5e7eb;
                            border-radius: 8px;
                        }
                        """
                    )
                if hasattr(self, "_cont_label"):
                    self._cont_label.setVisible(True)
                    self._cont_label.setText("Allow Continued EIDP Population By:")
                if hasattr(self, "_cont_hint"):
                    self._cont_hint.setVisible(True)
                    self._cont_hint.setText(
                        "Any future EIDP that matches these metadata values is auto-added to the project on refresh/update."
                    )
                if hasattr(self, "_continued_container"):
                    self._continued_container.setVisible(True)
            if is_raw:
                if hasattr(self, "_cont_label"):
                    self._cont_label.setVisible(False)
                if hasattr(self, "_cont_hint"):
                    self._cont_hint.setVisible(False)
                if hasattr(self, "_continued_container"):
                    self._continued_container.setVisible(False)
                if hasattr(self, "lbl_select_hint_sub"):
                    self.lbl_select_hint_sub.setText("")
                if hasattr(self, "_hint_frame"):
                    self._hint_frame.setStyleSheet(
                        """
                        QFrame {
                            background: #f8fafc;
                            border: 1px solid #e5e7eb;
                            border-radius: 8px;
                        }
                        """
                    )
        except Exception:
            pass

        try:
            self._refresh_filters()
        except Exception:
            pass

        try:
            self._apply_filter_and_refresh_table(select_all=True)
        except Exception:
            pass

    def _build_page_select(self) -> None:
        v = QtWidgets.QVBoxLayout(self._page_select)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        self._hint_frame = QtWidgets.QFrame()
        self._hint_frame.setStyleSheet(
            """
            QFrame {
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            """
        )
        hint_lay = QtWidgets.QVBoxLayout(self._hint_frame)
        hint_lay.setContentsMargins(10, 8, 10, 8)
        hint_lay.setSpacing(4)

        self.lbl_select_hint = QtWidgets.QLabel("Select EIDPs only (non-TD). Use Test Data Trending for TD reports.")
        self.lbl_select_hint.setStyleSheet("color: #0f172a; font-size: 12px; font-weight: 600;")
        self.lbl_select_hint.setWordWrap(True)
        hint_lay.addWidget(self.lbl_select_hint)

        self.lbl_select_hint_sub = QtWidgets.QLabel("")
        self.lbl_select_hint_sub.setStyleSheet("color: #475569; font-size: 11px;")
        self.lbl_select_hint_sub.setWordWrap(True)
        hint_lay.addWidget(self.lbl_select_hint_sub)

        v.addWidget(self._hint_frame)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(12)

        self.rb_all = QtWidgets.QRadioButton("All indexed EIDPs")
        self.rb_program = QtWidgets.QRadioButton("Program trending")
        self.rb_asset = QtWidgets.QRadioButton("Asset-type trending")
        self.rb_asset_specific = QtWidgets.QRadioButton("Asset-specific trending")
        self.rb_group = QtWidgets.QRadioButton("Similarity group")
        self.rb_all.setChecked(True)

        for rb in (self.rb_all, self.rb_program, self.rb_asset, self.rb_asset_specific, self.rb_group):
            rb.toggled.connect(lambda _=False: self._apply_filter_and_refresh_table(select_all=True))

        self.cb_program = QtWidgets.QComboBox()
        self.cb_asset = QtWidgets.QComboBox()
        self.cb_asset_specific = QtWidgets.QComboBox()
        self.cb_group = QtWidgets.QComboBox()
        self._selected_program_filters: list[str] = []
        self.cb_program.setEditable(True)
        try:
            self.cb_program.lineEdit().setReadOnly(True)
        except Exception:
            pass
        self.btn_program_multi = QtWidgets.QPushButton("Select...")
        self.btn_program_multi.clicked.connect(self._open_program_multi_select)
        self.cb_program.currentIndexChanged.connect(self._on_program_filter_changed)
        self.cb_asset.currentIndexChanged.connect(lambda *_: self._apply_filter_and_refresh_table(select_all=True))
        self.cb_asset_specific.currentIndexChanged.connect(lambda *_: self._apply_filter_and_refresh_table(select_all=True))
        self.cb_group.currentIndexChanged.connect(lambda *_: self._apply_filter_and_refresh_table(select_all=True))

        top.addWidget(self.rb_all)
        top.addWidget(self.rb_program)
        top.addWidget(self.cb_program, 1)
        top.addWidget(self.btn_program_multi)
        top.addWidget(self.rb_asset)
        top.addWidget(self.cb_asset, 1)
        top.addWidget(self.rb_asset_specific)
        top.addWidget(self.cb_asset_specific, 1)
        top.addWidget(self.rb_group)
        top.addWidget(self.cb_group, 1)
        v.addLayout(top)

        tbl_cols = ["Select", "Program", "Serial", "Doc Type", "Asset Type", "Asset Specific Type", "Metadata (rel)", "Group"]
        self.tbl = QtWidgets.QTableWidget(0, len(tbl_cols))
        self.tbl.setHorizontalHeaderLabels(tbl_cols)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.lbl_count = QtWidgets.QLabel("0 selected")
        self.lbl_count.setStyleSheet("font-size: 12px; color: #475569;")
        bottom.addWidget(self.lbl_count)
        bottom.addStretch(1)
        self.btn_sel_all = QtWidgets.QPushButton("Select All")
        self.btn_sel_none = QtWidgets.QPushButton("Select None")
        self.btn_sel_all.clicked.connect(lambda: self._set_all_checks(True))
        self.btn_sel_none.clicked.connect(lambda: self._set_all_checks(False))
        bottom.addWidget(self.btn_sel_all)
        bottom.addWidget(self.btn_sel_none)
        v.addLayout(bottom)

        self._cont_label = QtWidgets.QLabel("Allow Continued Population By:")
        self._cont_label.setStyleSheet("font-weight: 600; color: #374151; margin-top: 6px;")
        v.addWidget(self._cont_label)

        self._cont_hint = QtWidgets.QLabel(
            "Any future document that matches these metadata values is auto-added to the project on refresh/update."
        )
        self._cont_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        self._cont_hint.setWordWrap(True)
        v.addWidget(self._cont_hint)

        self.cb_cont_program = QtWidgets.QCheckBox("Program")
        self.cb_cont_part = QtWidgets.QCheckBox("Part Number")
        self.cb_cont_vendor = QtWidgets.QCheckBox("Vendor Name")
        self.cb_cont_asset = QtWidgets.QCheckBox("Asset Type")
        self.cb_cont_asset_specific = QtWidgets.QCheckBox("Asset Specific Type")
        self.cb_cont_test_plan = QtWidgets.QCheckBox("Acceptance Test Plan")

        self.list_cont_program = QtWidgets.QListWidget()
        self.list_cont_part = QtWidgets.QListWidget()
        self.list_cont_vendor = QtWidgets.QListWidget()
        self.list_cont_asset = QtWidgets.QListWidget()
        self.list_cont_asset_specific = QtWidgets.QListWidget()
        self.list_cont_test_plan = QtWidgets.QListWidget()

        for lst in (
            self.list_cont_program,
            self.list_cont_part,
            self.list_cont_vendor,
            self.list_cont_asset,
            self.list_cont_asset_specific,
            self.list_cont_test_plan,
        ):
            lst.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            lst.setMinimumHeight(70)
            lst.setMaximumHeight(120)

        def _bind_toggle(cb: QtWidgets.QCheckBox, lst: QtWidgets.QListWidget) -> None:
            def _apply(checked: bool) -> None:
                lst.setEnabled(bool(checked))
                lst.setVisible(bool(checked))
            cb.toggled.connect(_apply)
            _apply(False)

        _bind_toggle(self.cb_cont_program, self.list_cont_program)
        _bind_toggle(self.cb_cont_part, self.list_cont_part)
        _bind_toggle(self.cb_cont_vendor, self.list_cont_vendor)
        _bind_toggle(self.cb_cont_asset, self.list_cont_asset)
        _bind_toggle(self.cb_cont_asset_specific, self.list_cont_asset_specific)
        _bind_toggle(self.cb_cont_test_plan, self.list_cont_test_plan)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        def _add_cont_row(
            row: int,
            col: int,
            cb: QtWidgets.QCheckBox,
            lst: QtWidgets.QListWidget,
            *,
            col_span: int = 1,
        ) -> None:
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(6)
            box.addWidget(cb)
            box.addWidget(lst)
            grid.addLayout(box, row, col, 1, int(col_span))

        _add_cont_row(0, 0, self.cb_cont_program, self.list_cont_program)
        _add_cont_row(0, 1, self.cb_cont_vendor, self.list_cont_vendor)
        _add_cont_row(1, 0, self.cb_cont_part, self.list_cont_part)
        _add_cont_row(1, 1, self.cb_cont_asset, self.list_cont_asset)
        _add_cont_row(2, 0, self.cb_cont_asset_specific, self.list_cont_asset_specific)
        _add_cont_row(2, 1, self.cb_cont_test_plan, self.list_cont_test_plan)

        self._continued_container = QtWidgets.QWidget()
        self._continued_container.setLayout(grid)
        v.addWidget(self._continued_container)

    def _browse_location(self) -> None:
        try:
            start = str(self._global_repo)
            chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Location", start)
            if chosen:
                self.ed_location.setText(chosen)
        except Exception:
            pass

    def _read_docs(self) -> list[dict]:
        return getattr(be, "read_eidat_index_documents")(self._global_repo)

    def _refresh_filters(self) -> None:
        try:
            docs = self._read_docs()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Index not available", str(exc))
            docs = []

        ptype = str(getattr(self, "cb_type", None).currentText() if hasattr(self, "cb_type") else "").strip()
        is_td_mode = ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending")

        def _is_td(d: dict) -> bool:
            fn = getattr(be, "is_test_data_doc", None)
            if callable(fn):
                try:
                    return bool(fn(d))
                except Exception:
                    return False
            try:
                dt = str(d.get("document_type") or "").strip().lower()
            except Exception:
                dt = ""
            try:
                acr = str(d.get("document_type_acronym") or "").strip().lower()
            except Exception:
                acr = ""
            return dt in {"test data", "testdata", "td"} or acr in {"test data", "testdata", "td"}

        def _is_td_excel(d: dict) -> bool:
            if not _is_td(d):
                return False
            try:
                sqlite_rel = str(d.get("excel_sqlite_rel") or "").strip()
            except Exception:
                sqlite_rel = ""
            if sqlite_rel:
                return True
            try:
                art = str(d.get("artifacts_rel") or "").strip().lower()
            except Exception:
                art = ""
            return "__excel" in art

        if is_td_mode:
            docs = [d for d in docs if isinstance(d, dict) and _is_td_excel(d)]
        else:
            docs = [d for d in docs if isinstance(d, dict) and not _is_td(d)]

        cand: dict = {}
        try:
            load_candidates = getattr(be, "_load_metadata_candidates", None)
            if callable(load_candidates):
                raw = load_candidates()
                cand = raw if isinstance(raw, dict) else {}
        except Exception:
            cand = {}

        def _candidate_names(key: str) -> list[str]:
            try:
                canonical = getattr(be, "_canonical_names", None)
                if callable(canonical):
                    return list(canonical(cand.get(key) if isinstance(cand, dict) else []))
            except Exception:
                pass
            raw = cand.get(key) if isinstance(cand, dict) else []
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            for item in raw:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                    continue
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    if name:
                        out.append(name)
            return sorted({v for v in out if v}, key=lambda v: v.casefold())

        def _merge_values(*groups: list[str]) -> list[str]:
            merged = {
                str(value).strip()
                for group in groups
                for value in group
                if str(value).strip()
            }
            return sorted(merged, key=lambda v: v.casefold())

        programs = sorted({str(d.get("program_title") or "").strip() for d in docs if str(d.get("program_title") or "").strip()})
        assets = sorted({str(d.get("asset_type") or "").strip() for d in docs if str(d.get("asset_type") or "").strip()})
        asset_specifics = sorted({str(d.get("asset_specific_type") or "").strip() for d in docs if str(d.get("asset_specific_type") or "").strip()})
        vendors = sorted({str(d.get("vendor") or "").strip() for d in docs if str(d.get("vendor") or "").strip()})
        parts = sorted({str(d.get("part_number") or "").strip() for d in docs if str(d.get("part_number") or "").strip()})
        test_plans = sorted(
            {
                str(d.get("acceptance_test_plan_number") or "").strip()
                for d in docs
                if str(d.get("acceptance_test_plan_number") or "").strip()
            }
        )

        programs = _merge_values(programs, _candidate_names("program_titles"))
        assets = _merge_values(assets, _candidate_names("asset_types"))
        asset_specifics = _merge_values(asset_specifics, _candidate_names("asset_specific_types"))
        vendors = _merge_values(vendors, _candidate_names("vendors"))
        parts = _merge_values(parts, _candidate_names("part_numbers"))
        test_plans = _merge_values(test_plans, _candidate_names("acceptance_test_plan_numbers"))

        self.cb_program.blockSignals(True)
        self.cb_program.clear()
        self.cb_asset.clear()
        self.cb_asset_specific.clear()
        self.cb_group.clear()

        self.cb_program.addItems(programs or ["(none)"])
        self._selected_program_filters = [
            value for value in (self._selected_program_filters or []) if value in set(programs)
        ]
        if self._selected_program_filters:
            idx = self.cb_program.findText(self._selected_program_filters[0])
            if idx >= 0:
                self.cb_program.setCurrentIndex(idx)
        self._update_program_filter_summary()
        self.cb_program.blockSignals(False)
        self.cb_asset.addItems(assets or ["(none)"])
        self.cb_asset_specific.addItems(asset_specifics or ["(none)"])

        def _populate_list(widget: QtWidgets.QListWidget, items: list[str]) -> None:
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(items or ["(none)"])
            widget.blockSignals(False)

        if hasattr(self, "list_cont_program"):
            _populate_list(self.list_cont_program, programs)
        if hasattr(self, "list_cont_part"):
            _populate_list(self.list_cont_part, parts)
        if hasattr(self, "list_cont_vendor"):
            _populate_list(self.list_cont_vendor, vendors)
        if hasattr(self, "list_cont_asset"):
            _populate_list(self.list_cont_asset, assets)
        if hasattr(self, "list_cont_asset_specific"):
            _populate_list(self.list_cont_asset_specific, asset_specifics)
        if hasattr(self, "list_cont_test_plan"):
            _populate_list(self.list_cont_test_plan, test_plans)

        groups = []
        try:
            groups = getattr(be, "read_eidat_index_groups")(self._global_repo)
        except Exception:
            groups = []
        if groups:
            for g in groups:
                gid = str(g.get("group_id") or "").strip()
                if not gid:
                    continue
                title = str(g.get("title_norm") or "").strip()
                count = int(g.get("member_count") or 0)
                label = f"{gid} ({count}) {title}".strip()
                self.cb_group.addItem(label, gid)
        else:
            self.cb_group.addItem("(none)", "")

    def _program_filter_values(self) -> list[str]:
        values: list[str] = []
        for idx in range(self.cb_program.count() if hasattr(self, "cb_program") else 0):
            txt = str(self.cb_program.itemText(idx) or "").strip()
            if txt and txt != "(none)":
                values.append(txt)
        return values

    def _update_program_filter_summary(self) -> None:
        if not hasattr(self, "cb_program"):
            return
        selected = [value for value in (self._selected_program_filters or []) if str(value).strip()]
        if not selected:
            text = str(self.cb_program.currentText() or "").strip()
        elif len(selected) == 1:
            text = selected[0]
        else:
            text = f"{len(selected)} programs selected"
        try:
            self.cb_program.setEditText(text or "(none)")
            self.cb_program.setToolTip(", ".join(selected) if selected else "")
        except Exception:
            pass

    def _selected_program_filter_values(self) -> list[str]:
        selected = [value for value in (self._selected_program_filters or []) if str(value).strip()]
        if selected:
            return selected
        current = str(self.cb_program.currentText() or "").strip() if hasattr(self, "cb_program") else ""
        return [current] if current and current != "(none)" else []

    def _on_program_filter_changed(self, *_args) -> None:
        current = str(self.cb_program.currentText() or "").strip() if hasattr(self, "cb_program") else ""
        if current and current != "(none)" and current not in set(self._selected_program_filters or []):
            self._selected_program_filters = [current]
        self._update_program_filter_summary()
        self._apply_filter_and_refresh_table(select_all=True)

    def _open_program_multi_select(self) -> None:
        values = self._program_filter_values()
        if not values:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Programs")
        dlg.resize(420, 420)
        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        listw = QtWidgets.QListWidget()
        selected = set(self._selected_program_filter_values())
        for value in values:
            item = QtWidgets.QListWidgetItem(value)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if value in selected else QtCore.Qt.CheckState.Unchecked)
            listw.addItem(item)
        root.addWidget(listw, 1)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addWidget(btn_apply)
        btns.addWidget(btn_cancel)
        root.addLayout(btns)

        def _apply() -> None:
            chosen = [
                str(listw.item(i).text() or "").strip()
                for i in range(listw.count())
                if listw.item(i) and listw.item(i).checkState() == QtCore.Qt.CheckState.Checked
            ]
            self._selected_program_filters = [value for value in chosen if value]
            if self._selected_program_filters:
                blocker = QtCore.QSignalBlocker(self.cb_program)
                try:
                    idx = self.cb_program.findText(self._selected_program_filters[0])
                    if idx >= 0:
                        self.cb_program.setCurrentIndex(idx)
                finally:
                    del blocker
            self._update_program_filter_summary()
            dlg.accept()
            self._apply_filter_and_refresh_table(select_all=True)

        btn_apply.clicked.connect(_apply)
        btn_cancel.clicked.connect(dlg.reject)
        _fit_widget_to_screen(dlg)
        dlg.exec()

    def _filtered_docs(self, docs: list[dict]) -> list[dict]:
        ptype = str(self.cb_type.currentText() or "").strip()
        is_test_data = ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending")
        def _is_td(d: dict) -> bool:
            fn = getattr(be, "is_test_data_doc", None)
            if callable(fn):
                try:
                    return bool(fn(d))
                except Exception:
                    pass
            # Back-compat fallback if backend helper isn't available.
            try:
                dt = str(d.get("document_type") or "").strip().lower()
            except Exception:
                dt = ""
            try:
                acr = str(d.get("document_type_acronym") or "").strip().lower()
            except Exception:
                acr = ""
            return dt in {"test data", "testdata", "td"} or acr in {"test data", "testdata", "td"}

        if is_test_data:
            def _is_test_data_excel(d: dict) -> bool:
                ok_dt = _is_td(d)
                try:
                    art = str(d.get("artifacts_rel") or "").strip().lower()
                except Exception:
                    art = ""
                try:
                    sqlite_rel = str(d.get("excel_sqlite_rel") or "").strip()
                except Exception:
                    sqlite_rel = ""
                ok_excel = bool(sqlite_rel) or "__excel" in art
                return bool(ok_dt and ok_excel)

            docs = [d for d in docs if isinstance(d, dict) and _is_test_data_excel(d)]
        else:
            # EIDP projects must not be created from TD reports.
            docs = [d for d in docs if isinstance(d, dict) and not _is_td(d)]
        if self.rb_program.isChecked():
            wanted_values = set(self._selected_program_filter_values())
            if not wanted_values:
                return []
            return [d for d in docs if str(d.get("program_title") or "").strip() in wanted_values]
        if self.rb_asset.isChecked():
            wanted = str(self.cb_asset.currentText() or "").strip()
            return [d for d in docs if str(d.get("asset_type") or "").strip() == wanted]
        if self.rb_asset_specific.isChecked():
            wanted = str(self.cb_asset_specific.currentText() or "").strip()
            return [d for d in docs if str(d.get("asset_specific_type") or "").strip() == wanted]
        if self.rb_group.isChecked():
            gid = str(self.cb_group.currentData() or "").strip()
            if not gid:
                return []
            return [d for d in docs if str(d.get("similarity_group") or "").strip() == gid]
        return docs

    def _apply_filter_and_refresh_table(self, *, select_all: bool) -> None:
        try:
            docs = self._read_docs()
        except Exception:
            docs = []
        filtered = self._filtered_docs(docs)
        self._populate_table(filtered, select_all=select_all)
        self._update_counts()
        self._update_nav()

    def _populate_table(self, docs: list[dict], *, select_all: bool) -> None:
        self.tbl.setRowCount(0)
        for r, d in enumerate(docs):
            self.tbl.insertRow(r)
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(select_all))
            checkbox.stateChanged.connect(lambda *_: self._update_counts())
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(10, 0, 0, 0)
            lay.addWidget(checkbox)
            lay.addStretch(1)
            self.tbl.setCellWidget(r, 0, w)

            self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(str(d.get("program_title") or "")))
            self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(str(d.get("serial_number") or "")))
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(str(d.get("document_type") or d.get("document_type_acronym") or "")))
            self.tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(str(d.get("asset_type") or "")))
            self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(str(d.get("asset_specific_type") or "")))
            self.tbl.setItem(r, 6, QtWidgets.QTableWidgetItem(str(d.get("metadata_rel") or "")))
            self.tbl.setItem(r, 7, QtWidgets.QTableWidgetItem(str(d.get("similarity_group") or "")))

        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeMode.Stretch)

    def _set_all_checks(self, checked: bool) -> None:
        for r in range(self.tbl.rowCount()):
            widget = self.tbl.cellWidget(r, 0)
            if not widget:
                continue
            cb = widget.findChild(QtWidgets.QCheckBox)
            if cb:
                cb.setChecked(checked)
        self._update_counts()

    def _selected_metadata_rel(self) -> list[str]:
        selected: list[str] = []
        for r in range(self.tbl.rowCount()):
            widget = self.tbl.cellWidget(r, 0)
            cb = widget.findChild(QtWidgets.QCheckBox) if widget else None
            if cb and cb.isChecked():
                item = self.tbl.item(r, 6)
                if item and item.text().strip():
                    selected.append(item.text().strip())
        return selected

    def _selected_continued_population(self) -> tuple[dict, list[str]]:
        rules: dict[str, list[str]] = {}
        missing: list[str] = []

        def _collect(cb: QtWidgets.QCheckBox, lst: QtWidgets.QListWidget, key: str, label: str) -> None:
            if not cb.isChecked():
                return
            values = [
                i.text().strip()
                for i in lst.selectedItems()
                if i.text().strip() and i.text().strip() != "(none)"
            ]
            if values:
                rules[key] = values
            else:
                missing.append(label)

        _collect(self.cb_cont_program, self.list_cont_program, "program_title", "Program")
        _collect(self.cb_cont_part, self.list_cont_part, "part_number", "Part Number")
        _collect(self.cb_cont_vendor, self.list_cont_vendor, "vendor", "Vendor Name")
        _collect(self.cb_cont_asset, self.list_cont_asset, "asset_type", "Asset Type")
        if hasattr(self, "cb_cont_asset_specific") and hasattr(self, "list_cont_asset_specific"):
            _collect(self.cb_cont_asset_specific, self.list_cont_asset_specific, "asset_specific_type", "Asset Specific Type")
        if hasattr(self, "cb_cont_test_plan") and hasattr(self, "list_cont_test_plan"):
            _collect(self.cb_cont_test_plan, self.list_cont_test_plan, "acceptance_test_plan_number", "Acceptance Test Plan")

        return rules, missing

    def _update_counts(self) -> None:
        selected = self._selected_metadata_rel()
        self.lbl_count.setText(f"{len(selected)} selected")

    def _details_valid(self) -> bool:
        name = (self.ed_name.text() or "").strip()
        if not name:
            return False
        loc = (self.ed_location.text() or "").strip()
        if not loc:
            return False
        try:
            getattr(be, "resolve_path_within_global_repo")(self._global_repo, Path(loc), "Project location")
        except Exception:
            return False
        return True

    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setVisible(idx == 0)
        self.btn_create.setVisible(idx == 1)
        self.btn_next.setEnabled(self._details_valid())
        if idx == 1:
            self.btn_create.setEnabled(bool(self._selected_metadata_rel()))

    def _act_back(self) -> None:
        self._stack.setCurrentIndex(max(0, self._stack.currentIndex() - 1))
        self._update_nav()

    def _act_next(self) -> None:
        if not self._details_valid():
            QtWidgets.QMessageBox.information(self, "Missing info", "Enter a project name and choose a location inside the Global Repo.")
            return
        self._stack.setCurrentIndex(1)
        self._update_nav()

    def _act_create(self) -> None:
        try:
            project_name = (self.ed_name.text() or "").strip()
            project_type = str(self.cb_type.currentText() or "").strip()
            location = Path((self.ed_location.text() or "").strip())
            selected = self._selected_metadata_rel()
            if not selected:
                ptype = str(self.cb_type.currentText() or "").strip()
                if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                    raise RuntimeError("Select at least one Test Data Excel document.")
                raise RuntimeError("Select at least one EIDP.")
            continued_rules, missing = self._selected_continued_population()
            if missing:
                raise RuntimeError(
                    "Select at least one value for: " + ", ".join(missing)
                )
            # Two clear options: auto-populate or blank
            auto_populate = self.rb_auto_populate.isChecked()
            if project_type == getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"):
                auto_populate = False
            meta = getattr(be, "create_eidat_project")(
                self._global_repo,
                project_parent_dir=location,
                project_name=project_name,
                project_type=project_type,
                selected_metadata_rel=selected,
                continued_population=continued_rules,
                auto_populate=auto_populate,
            )
            self.project_meta = meta if isinstance(meta, dict) else {}
            self.accept()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Create project failed", str(exc))

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        _fit_widget_to_screen(self)


class ImplementationTrendDialog(QtWidgets.QDialog):
    def __init__(self, project_dir: Path, workbook_path: Path, parent=None):
        super().__init__(parent)
        self._project_dir = Path(project_dir).expanduser()
        self._workbook_path = Path(workbook_path).expanduser()
        self._db_path: Path | None = None
        self._serials: list[str] = []
        self._terms: list[dict] = []
        self._plot_ready = False
        self._auto_plots: list[dict] = []
        self._last_plot_payloads: list[dict] = []
        self._auto_plot_path = self._project_dir / "auto_plots.json"
        self._auto_plot_refreshing = False
        self._last_plot_source_label = ""

        self.setWindowTitle("Implementation - Trend / Analyze Data")
        self.resize(960, 640)
        self.setStyleSheet(
            """
            QDialog {
                background: #f8fafc;
                color: #0f172a;
            }
            QLabel {
                color: #0f172a;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item {
                color: #0f172a;
                padding: 4px 6px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #1e3a8a;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                color: #0f172a;
            }
            QHeaderView::section {
                background: #f1f5f9;
                color: #0f172a;
                padding: 6px 8px;
                border: 1px solid #e2e8f0;
                font-weight: 600;
            }
            """
        )

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left: term selection
        left = QtWidgets.QFrame()
        left.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }"
        )
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        lbl = QtWidgets.QLabel("Select Terms")
        lbl.setStyleSheet("font-size: 14px; font-weight: 700;")
        left_layout.addWidget(lbl)

        self.ed_filter_terms = QtWidgets.QLineEdit()
        self.ed_filter_terms.setPlaceholderText("Filter terms (e.g., OPEN_T)")
        self.ed_filter_terms.textChanged.connect(self._apply_term_filter)
        left_layout.addWidget(self.ed_filter_terms)

        self.list_terms = QtWidgets.QListWidget()
        self.list_terms.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_terms.itemSelectionChanged.connect(self._on_term_selection_changed)
        left_layout.addWidget(self.list_terms, 1)

        select_row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("Select All")
        self.btn_clear_all = QtWidgets.QPushButton("Clear")
        for b in (self.btn_select_all, self.btn_clear_all):
            b.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 10px;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #1f2937;
                    border: 1px solid #cbd5e1;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f8fafc; }
                """
            )
        self.btn_select_all.clicked.connect(self._select_all_terms)
        self.btn_clear_all.clicked.connect(self._clear_all_terms)
        select_row.addWidget(self.btn_select_all)
        select_row.addWidget(self.btn_clear_all)
        left_layout.addLayout(select_row)

        self.btn_plot = QtWidgets.QPushButton("Plot Selected Terms")
        self.btn_plot.setEnabled(False)
        self.btn_plot.setStyleSheet(
            """
            QPushButton {
                padding: 10px 14px;
                border-radius: 8px;
                background: #0f766e;
                color: #ffffff;
                border: 1px solid #0f766e;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background: #0d9488; }
            QPushButton:disabled { background: #94a3b8; border-color: #94a3b8; }
            """
        )
        self.btn_plot.clicked.connect(self._plot_selected_terms)
        left_layout.addWidget(self.btn_plot)

        auto_label = QtWidgets.QLabel("Auto-Plots")
        auto_label.setStyleSheet("font-size: 13px; font-weight: 700; margin-top: 6px;")
        left_layout.addWidget(auto_label)

        auto_hint = QtWidgets.QLabel("Select an auto-plot to open it in the plot pane.")
        auto_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        auto_hint.setWordWrap(True)
        left_layout.addWidget(auto_hint)

        self.list_auto_plots = QtWidgets.QListWidget()
        self.list_auto_plots.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_auto_plots.itemSelectionChanged.connect(self._update_auto_plot_actions)
        self.list_auto_plots.itemDoubleClicked.connect(lambda *_: self._open_selected_auto_plot())
        left_layout.addWidget(self.list_auto_plots, 1)

        auto_btn_row = QtWidgets.QHBoxLayout()
        self.btn_open_auto_panel = QtWidgets.QPushButton("Open Selected Auto-Plots")
        self.btn_open_auto_panel.setEnabled(False)
        self.btn_open_auto_panel.setStyleSheet(
            """
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f8fafc; }
            QPushButton:disabled { color: #94a3b8; border-color: #cbd5e1; }
            """
        )
        self.btn_open_auto_panel.clicked.connect(self._open_auto_plot_panel)
        self.btn_save_all_auto = QtWidgets.QPushButton("Save All Auto-Plots")
        self.btn_save_all_auto.setEnabled(False)
        self.btn_save_all_auto.setStyleSheet(
            """
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #cbd5e1;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f8fafc; }
            QPushButton:disabled { color: #94a3b8; border-color: #cbd5e1; }
            """
        )
        self.btn_save_all_auto.clicked.connect(self._save_all_auto_plots_pdf)
        auto_btn_row.addWidget(self.btn_open_auto_panel)
        auto_btn_row.addWidget(self.btn_save_all_auto)
        left_layout.addLayout(auto_btn_row)

        splitter.addWidget(left)

        # Right: plot + stats
        right = QtWidgets.QFrame()
        right.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }"
        )
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Trend Plot")
        title.setStyleSheet("font-size: 14px; font-weight: 700;")
        self.btn_add_auto_plot = QtWidgets.QPushButton("Add to Auto-Plots")
        self.btn_add_auto_plot.setEnabled(False)
        self.btn_add_auto_plot.setStyleSheet(
            """
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #0f766e;
                border: 1px solid #0f766e;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover { background: #ecfdf3; }
            QPushButton:disabled { color: #94a3b8; border-color: #cbd5e1; }
            """
        )
        self.btn_add_auto_plot.clicked.connect(self._add_current_plot_to_autoplots)

        self.btn_save_plot_pdf = QtWidgets.QPushButton("Save Plot PDF")
        self.btn_save_plot_pdf.setEnabled(False)
        self.btn_save_plot_pdf.setStyleSheet(
            """
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #1d4ed8;
                border: 1px solid #1d4ed8;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover { background: #eff6ff; }
            QPushButton:disabled { color: #94a3b8; border-color: #cbd5e1; }
            """
        )
        self.btn_save_plot_pdf.clicked.connect(self._save_current_plot_pdf)
        self.lbl_source = QtWidgets.QLabel("")
        self.lbl_source.setStyleSheet("color: #64748b; font-size: 11px;")
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(self.btn_add_auto_plot)
        header_row.addWidget(self.btn_save_plot_pdf)
        header_row.addWidget(self.lbl_source)
        right_layout.addLayout(header_row)

        self.plot_container = QtWidgets.QFrame()
        plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = None
        self._figure = None
        self._axes = None
        self._init_plot_area(plot_layout)
        right_layout.addWidget(self.plot_container, 2)

        stats_label = QtWidgets.QLabel("Analysis Summary")
        stats_label.setStyleSheet("font-size: 13px; font-weight: 700;")
        right_layout.addWidget(stats_label)

        cols = [
            "Term",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Max",
            "Min Limit",
            "Max Limit",
            "Exceed Low",
            "Exceed High",
            "Exceed SNs",
        ]
        self.tbl_stats = QtWidgets.QTableWidget(0, len(cols))
        self.tbl_stats.setHorizontalHeaderLabels(cols)
        self.tbl_stats.verticalHeader().setVisible(False)
        self.tbl_stats.setAlternatingRowColors(True)
        self.tbl_stats.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.tbl_stats.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_stats.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.tbl_stats, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._load_terms()
        self._load_auto_plots()

    def _init_plot_area(self, layout: QtWidgets.QVBoxLayout) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self._figure = Figure(figsize=(8, 4), dpi=100)
            self._axes = self._figure.add_subplot(111)
            self._axes.set_facecolor("#ffffff")
            self._figure.patch.set_facecolor("#ffffff")
            self._canvas = FigureCanvas(self._figure)
            layout.addWidget(self._canvas)
            self._plot_ready = True
        except Exception as exc:
            self._plot_ready = False
            label = QtWidgets.QLabel(
                "Plotting unavailable. Install matplotlib to enable charts.\n"
                f"Details: {exc}"
            )
            label.setWordWrap(True)
            label.setStyleSheet("color: #b91c1c; font-size: 12px;")
            layout.addWidget(label)

    def _load_terms(self) -> None:
        try:
            self._db_path = be.ensure_trending_project_sqlite(self._project_dir, self._workbook_path)
            self._serials = be.load_trending_project_serials(self._db_path)
            self._terms = be.load_trending_project_terms(self._db_path)
            self.lbl_source.setText(str(self._db_path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Implementation", str(exc))
            return

        self.list_terms.clear()
        for t in self._terms:
            term = str(t.get("term") or "").strip()
            label = str(t.get("term_label") or "").strip()
            units = str(t.get("units") or "").strip()
            display = label if label else term
            if units:
                display = f"{display} ({units})"
            if not display:
                display = term
            item = QtWidgets.QListWidgetItem(display)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, t)
            self.list_terms.addItem(item)

        if self.list_terms.count() == 0:
            self.list_terms.addItem("No terms available in this project workbook.")
            self.list_terms.setEnabled(False)
        else:
            self.list_terms.setEnabled(True)

    def _apply_term_filter(self) -> None:
        needle = (self.ed_filter_terms.text() or "").strip().lower()
        for i in range(self.list_terms.count()):
            item = self.list_terms.item(i)
            text = item.text().lower()
            item.setHidden(bool(needle) and needle not in text)

    def _on_term_selection_changed(self) -> None:
        selected = self.list_terms.selectedItems()
        self.btn_plot.setEnabled(bool(selected) and self._plot_ready)

    def _update_auto_plot_actions(self) -> None:
        if not hasattr(self, "list_auto_plots"):
            return
        if not self.list_auto_plots.isEnabled():
            self.btn_open_auto_panel.setEnabled(False)
            return
        selected = self.list_auto_plots.selectedItems()
        has_plots = bool(selected) and self._plot_ready and bool(self._db_path)
        self.btn_open_auto_panel.setEnabled(has_plots)

    def _select_all_terms(self) -> None:
        if not self.list_terms.isEnabled():
            return
        self.list_terms.selectAll()

    def _clear_all_terms(self) -> None:
        self.list_terms.clearSelection()

    def _plot_terms_payloads(self, term_payloads: list[dict], *, source_label: str = "") -> None:
        if not self._plot_ready or not self._db_path:
            return
        if not term_payloads:
            return

        self._axes.clear()
        self._axes.set_title(self._compose_title(term_payloads, source_label=source_label))
        self._axes.set_xlabel("Serial Number")
        self._axes.set_ylabel(self._units_label(term_payloads))

        x = list(range(len(self._serials)))

        stats_rows: list[dict] = []
        for payload in term_payloads:
            term_key = str(payload.get("term_key") or "").strip()
            term_name = str(payload.get("term_label") or payload.get("term") or "").strip() or "Term"
            min_limit = payload.get("min_val")
            max_limit = payload.get("max_val")
            units = str(payload.get("units") or "").strip()
            if units:
                term_name = f"{term_name} ({units})"

            series = be.load_trending_project_series(self._db_path, term_key)
            values_num: list[float | None] = []
            for row in series:
                values_num.append(row.get("value_num"))

            y = [v if v is not None else float("nan") for v in values_num]
            line = self._axes.plot(x, y, marker="o", linewidth=1.6, label=term_name)[0]
            color = line.get_color()

            if min_limit is not None:
                self._axes.axhline(float(min_limit), color=color, linestyle="--", alpha=0.4)
            if max_limit is not None:
                self._axes.axhline(float(max_limit), color=color, linestyle="--", alpha=0.4)

            numeric_values = [v for v in values_num if isinstance(v, (int, float)) and not math.isnan(float(v))]
            if numeric_values:
                mean_val = statistics.mean(numeric_values)
                std_val = statistics.pstdev(numeric_values) if len(numeric_values) > 1 else 0.0
                min_val = min(numeric_values)
                max_val = max(numeric_values)
            else:
                mean_val = None
                std_val = None
                min_val = None
                max_val = None

            exceed_low = []
            exceed_high = []
            for sn, val in zip(self._serials, values_num):
                if val is None:
                    continue
                if min_limit is not None and val < float(min_limit):
                    exceed_low.append(sn)
                if max_limit is not None and val > float(max_limit):
                    exceed_high.append(sn)

            stats_rows.append(
                {
                    "term": term_name,
                    "count": len(numeric_values),
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "min_limit": min_limit,
                    "max_limit": max_limit,
                    "exceed_low": len(exceed_low),
                    "exceed_high": len(exceed_high),
                    "exceed_sns": ", ".join(sorted(set(exceed_low + exceed_high))),
                }
            )

        self._axes.set_xticks(x)
        self._axes.set_xticklabels(self._serials, rotation=45, ha="right", fontsize=8)
        self._axes.grid(True, alpha=0.25)
        self._axes.legend(fontsize=8, loc="best")
        self._figure.tight_layout()
        self._canvas.draw()

        self._populate_stats(stats_rows)
        self._last_plot_payloads = list(term_payloads)
        self._last_plot_source_label = source_label
        if source_label:
            self.lbl_source.setText(f"{self._db_path} • {source_label}")
        else:
            self.lbl_source.setText(str(self._db_path))
        self.btn_add_auto_plot.setEnabled(bool(self._last_plot_payloads))
        self.btn_save_plot_pdf.setEnabled(bool(self._last_plot_payloads))

    def _plot_selected_terms(self) -> None:
        if not self._plot_ready or not self._db_path:
            return
        items = self.list_terms.selectedItems()
        if not items:
            return

        if not self._serials:
            QtWidgets.QMessageBox.information(self, "Implementation", "No serials found in the project workbook.")
            return

        term_payloads = []
        for item in items:
            payload = item.data(QtCore.Qt.ItemDataRole.UserRole) or {}
            term_key = str(payload.get("term_key") or "").strip()
            if term_key:
                term_payloads.append(payload)

        if not term_payloads:
            return

        self._plot_terms_payloads(term_payloads, source_label="Selected terms")

    def _populate_stats(self, rows: list[dict]) -> None:
        self.tbl_stats.setRowCount(0)
        for r, row in enumerate(rows):
            self.tbl_stats.insertRow(r)
            values = [
                row.get("term", ""),
                row.get("count", ""),
                self._fmt_num(row.get("mean")),
                self._fmt_num(row.get("std")),
                self._fmt_num(row.get("min")),
                self._fmt_num(row.get("max")),
                self._fmt_num(row.get("min_limit")),
                self._fmt_num(row.get("max_limit")),
                row.get("exceed_low", ""),
                row.get("exceed_high", ""),
                row.get("exceed_sns", ""),
            ]
            for c, v in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(v))
                self.tbl_stats.setItem(r, c, item)
        self.tbl_stats.resizeColumnsToContents()

    @staticmethod
    def _fmt_num(value: object | None) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.4g}"
        except Exception:
            return str(value)

    def _set_plot_note(self, text: str = "") -> None:
        if not hasattr(self, "lbl_plot_note"):
            return
        note = str(text or "").strip()
        self.lbl_plot_note.setText(note)
        self.lbl_plot_note.setVisible(bool(note))

    @staticmethod
    def _metric_title_suffix(stats: list[str] | tuple[str, ...] | None) -> str:
        items = [str(s).strip().lower() for s in (stats or []) if str(s).strip()]
        title_items = [s for s in items if s != "average"]
        return "/".join(title_items)

    def _default_plot_name(self, term_payloads: list[dict]) -> str:
        labels: list[str] = []
        for payload in term_payloads:
            label = str(payload.get("term_label") or payload.get("term") or "").strip()
            if label:
                labels.append(label)
        if not labels:
            return "Auto Plot"
        if len(labels) <= 3:
            return ", ".join(labels)
        return f"{', '.join(labels[:2])} (+{len(labels) - 2})"

    def _units_label(self, term_payloads: list[dict]) -> str:
        units = sorted({str(p.get("units") or "").strip() for p in term_payloads if str(p.get("units") or "").strip()})
        if not units:
            return "Value"
        if len(units) == 1:
            return f"Value ({units[0]})"
        return "Value (mixed units)"

    def _compose_title(self, term_payloads: list[dict], *, source_label: str = "") -> str:
        if source_label:
            return f"Trend Plot — {source_label}"
        labels: list[str] = []
        for payload in term_payloads:
            label = str(payload.get("term_label") or payload.get("term") or "").strip()
            if label:
                labels.append(label)
        if not labels:
            return "Trend Plot"
        if len(labels) == 1:
            return f"Trend Plot — {labels[0]}"
        return f"Trend Plot — {self._default_plot_name(term_payloads)}"

    def _load_auto_plots(self) -> None:
        self._auto_plots = []
        try:
            if self._auto_plot_path.exists():
                payload = json.loads(self._auto_plot_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    plots = payload.get("plots")
                else:
                    plots = payload
                if isinstance(plots, list):
                    self._auto_plots = [p for p in plots if isinstance(p, dict)]
        except Exception:
            self._auto_plots = []
        self._refresh_auto_plot_list()

    def _save_auto_plots(self) -> None:
        try:
            data = {"plots": self._auto_plots}
            self._auto_plot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _refresh_auto_plot_list(self) -> None:
        if not hasattr(self, "list_auto_plots"):
            return
        self._auto_plot_refreshing = True
        self.list_auto_plots.clear()
        if not self._auto_plots:
            self.list_auto_plots.addItem("No auto-plots yet.")
            self.list_auto_plots.setEnabled(False)
        else:
            self.list_auto_plots.setEnabled(True)
            for plot in self._auto_plots:
                name = str(plot.get("name") or "Auto Plot").strip()
                terms = plot.get("terms") or []
                count = len(terms) if isinstance(terms, list) else 0
                label = f"{name} ({count})"
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, plot)
                self.list_auto_plots.addItem(item)
        self._auto_plot_refreshing = False
        self.btn_save_all_auto.setEnabled(bool(self._auto_plots) and self._plot_ready and bool(self._db_path))
        self._update_auto_plot_actions()

    def _payloads_from_term_keys(self, keys: list[str]) -> list[dict]:
        by_key: dict[str, dict] = {}
        for t in self._terms:
            key = str(t.get("term_key") or "").strip()
            if key:
                by_key[key] = t
        payloads: list[dict] = []
        for key in keys:
            k = str(key or "").strip()
            if not k:
                continue
            payload = by_key.get(k)
            if payload:
                payloads.append(payload)
        return payloads

    def _open_selected_auto_plot(self) -> None:
        if self._auto_plot_refreshing:
            return
        if not self.list_auto_plots.isEnabled():
            return
        items = self.list_auto_plots.selectedItems()
        if not items:
            return
        plot = items[0].data(QtCore.Qt.ItemDataRole.UserRole) or {}
        term_keys = plot.get("terms") or []
        if not isinstance(term_keys, list) or not term_keys:
            return
        payloads = self._payloads_from_term_keys(term_keys)
        missing = len(term_keys) - len(payloads)
        if not payloads:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No matching terms found for this auto-plot.")
            return
        if missing > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Auto-Plots",
                f"{missing} term(s) no longer exist in the workbook and were skipped.",
            )
        name = str(plot.get("name") or "").strip()
        label = f"Auto-plot: {name}" if name else "Auto-plot"
        self._plot_terms_payloads(payloads, source_label=label)

    def _add_current_plot_to_autoplots(self) -> None:
        if not self._last_plot_payloads:
            return
        default_name = self._default_plot_name(self._last_plot_payloads)
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add to Auto-Plots",
            "Name this auto-plot:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not ok:
            return
        name = (name or "").strip() or default_name
        term_keys = [str(p.get("term_key") or "").strip() for p in self._last_plot_payloads if str(p.get("term_key") or "").strip()]
        if not term_keys:
            return
        plot = {
            "id": f"auto_{int(time.time()*1000)}",
            "name": name,
            "terms": term_keys,
        }
        self._auto_plots.append(plot)
        self._save_auto_plots()
        self._refresh_auto_plot_list()

    def _open_auto_plot_panel(self) -> None:
        if not self._plot_ready or not self._db_path:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "Plotting is unavailable.")
            return
        if not self.list_auto_plots.isEnabled():
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No auto-plots available.")
            return
        items = self.list_auto_plots.selectedItems()
        if not items:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "Select one or more auto-plots first.")
            return

        plots_payloads: list[tuple[str, list[dict]]] = []
        for item in items:
            plot = item.data(QtCore.Qt.ItemDataRole.UserRole) or {}
            name = str(plot.get("name") or "Auto Plot").strip()
            term_keys = plot.get("terms") or []
            if not isinstance(term_keys, list) or not term_keys:
                continue
            payloads = self._payloads_from_term_keys(term_keys)
            if not payloads:
                continue
            plots_payloads.append((name, payloads))

        if not plots_payloads:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No matching terms found for the selected auto-plots.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Auto-Plots")
        dlg.resize(980, 720)
        dlg.setStyleSheet(
            """
            QDialog { background: #ffffff; color: #1f2937; }
            QLabel { color: #1f2937; }
            QTabWidget::pane { border: 1px solid #e2e8f0; }
            QTabBar::tab {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                color: #0f172a;
                padding: 6px 10px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0f172a;
                border-bottom-color: #ffffff;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc:
            QtWidgets.QMessageBox.warning(dlg, "Auto-Plots", f"Plotting unavailable: {exc}")
            return

        for name, payloads in plots_payloads:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            tab_layout.setContentsMargins(8, 8, 8, 8)
            fig = self._render_plot_to_figure(payloads, title_text=f"Trend Plot — {name}")
            canvas = FigureCanvas(fig)
            tab_layout.addWidget(canvas, 1)
            tabs.addTab(tab, name or "Auto Plot")

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dlg.exec()

    def _render_plot_to_figure(self, term_payloads: list[dict], *, title_text: str = ""):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 4), dpi=150)
        ax = fig.add_subplot(111)
        ax.set_title(title_text or self._compose_title(term_payloads))
        ax.set_xlabel("Serial Number")
        ax.set_ylabel(self._units_label(term_payloads))

        x = list(range(len(self._serials)))
        for payload in term_payloads:
            term_key = str(payload.get("term_key") or "").strip()
            term_name = str(payload.get("term_label") or payload.get("term") or "").strip() or "Term"
            min_limit = payload.get("min_val")
            max_limit = payload.get("max_val")
            units = str(payload.get("units") or "").strip()
            if units:
                term_name = f"{term_name} ({units})"

            series = be.load_trending_project_series(self._db_path, term_key)
            values_num: list[float | None] = [row.get("value_num") for row in series]
            y = [v if v is not None else float("nan") for v in values_num]
            line = ax.plot(x, y, marker="o", linewidth=1.6, label=term_name)[0]
            color = line.get_color()
            if min_limit is not None:
                ax.axhline(float(min_limit), color=color, linestyle="--", alpha=0.4)
            if max_limit is not None:
                ax.axhline(float(max_limit), color=color, linestyle="--", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(self._serials, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        return fig

    def _save_current_plot_pdf(self) -> None:
        if not self._plot_ready or not self._last_plot_payloads:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Plot PDF",
            str(self._project_dir / "trend_plot.pdf"),
            "PDF Files (*.pdf)",
        )
        if not path:
            return
        try:
            title_text = self._compose_title(self._last_plot_payloads, source_label=self._last_plot_source_label)
            fig = self._render_plot_to_figure(self._last_plot_payloads, title_text=title_text)
            fig.savefig(path, format="pdf")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Plot PDF", str(exc))

    def _save_all_auto_plots_pdf(self) -> None:
        if not self._plot_ready or not self._auto_plots:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save All Auto-Plots",
            str(self._project_dir / "auto_plots.pdf"),
            "PDF Files (*.pdf)",
        )
        if not path:
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(path) as pdf:
                for plot in self._auto_plots:
                    keys = plot.get("terms") or []
                    if not isinstance(keys, list) or not keys:
                        continue
                    payloads = self._payloads_from_term_keys(keys)
                    if not payloads:
                        continue
                    name = str(plot.get("name") or "").strip()
                    title_text = f"Trend Plot — {name}" if name else self._compose_title(payloads)
                    fig = self._render_plot_to_figure(payloads, title_text=title_text)
                    pdf.savefig(fig)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save All Auto-Plots", str(exc))


class TestDataTrendDialog(QtWidgets.QDialog):
    def __init__(self, project_dir: Path, workbook_path: Path, parent=None):
        super().__init__(parent)
        self._project_dir = Path(project_dir).expanduser()
        self._workbook_path = Path(workbook_path).expanduser()
        self._db_path: Path | None = None
        self._plot_ready = False
        self._mode = "curves"  # curves | metrics | performance
        self._highlight_sn = ""
        self._highlight_sns: list[str] = []
        self._serial_source_rows: list[dict] = []
        self._serial_source_by_serial: dict[str, dict] = {}
        self._global_filter_rows: list[dict] = []
        self._available_program_filters: list[str] = []
        self._available_serial_filter_rows: list[dict] = []
        self._available_control_period_filters: list[str] = []
        self._available_suppression_voltage_filters: list[str] = []
        self._checked_program_filters: list[str] = []
        self._checked_serial_filters: list[str] = []
        self._checked_control_period_filters: list[str] = []
        self._checked_suppression_voltage_filters: list[str] = []
        self._auto_plots: list[dict] = []
        self._last_plot_def: dict | None = None
        self._auto_plot_path = self._project_dir / "auto_plots_test_data.json"
        self._plot_base_xlim: tuple[float, float] | None = None
        self._plot_base_ylim: tuple[float, float] | None = None
        self._plot_note_base_text = ""
        self._plot_last_cursor_xy: tuple[float, float] | None = None
        self._zone_zoom_press_xy: tuple[float, float] | None = None
        self._zone_zoom_rect = None
        self._perf_plotters: list[dict] = []
        self._cache_worker: ProjectTaskWorker | None = None
        self._export_worker: ProjectTaskWorker | None = None
        self._cache_progress_visible = False
        self._cache_progress_heading = "Test Data Cache"
        self._cache_progress_status = "Preparing cache"
        self._cache_progress_detail = ""
        self._export_progress_visible = False
        self._cache_progress_timer = QtCore.QTimer(self)
        self._cache_progress_timer.setSingleShot(True)
        self._cache_progress_timer.timeout.connect(self._show_cache_progress_dialog)
        self._plot_band_popup: QtWidgets.QDialog | None = None
        self._main_plot_legend_entries: list[dict[str, str]] = []
        self._left_panel_scroll: QtWidgets.QScrollArea | None = None
        self._left_panel_locked_width: int | None = None
        self._left_panel_width_initialized = False
        self._startup_size_locked = False
        self._perf_equations_popup: QtWidgets.QDialog | None = None

        self.setWindowTitle("Test Data - Trend / Analyze")
        self.resize(1280, 760)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(
            """
            QDialog { background: #f8fafc; color: #0f172a; }
            QLabel { color: #0f172a; }
            QListWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item { color: #0f172a; padding: 4px 6px; }
            QListWidget::item:selected { background: #dbeafe; color: #1e3a8a; }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
            }
            QHeaderView::section {
                background: #f1f5f9;
                color: #0f172a;
                padding: 6px 8px;
                border: 1px solid #e2e8f0;
                font-weight: 600;
            }
            """
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        filter_frame = QtWidgets.QFrame()
        self.filter_frame = filter_frame
        filter_frame.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }")
        filter_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        filter_layout = QtWidgets.QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(12, 10, 12, 10)
        filter_layout.setSpacing(12)

        filter_text_layout = QtWidgets.QVBoxLayout()
        filter_text_layout.setContentsMargins(0, 0, 0, 0)
        filter_text_layout.setSpacing(2)
        filter_title = QtWidgets.QLabel("Global Filters")
        filter_title.setStyleSheet("font-size: 12px; font-weight: 800; color: #0f172a;")
        self.lbl_program_filter_summary = QtWidgets.QLabel("Programs: -")
        self.lbl_program_filter_summary.setStyleSheet("color: #334155; font-size: 11px;")
        self.lbl_program_filter_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.lbl_serial_filter_summary = QtWidgets.QLabel("Serials: -")
        self.lbl_serial_filter_summary.setStyleSheet("color: #334155; font-size: 11px;")
        self.lbl_serial_filter_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.lbl_suppression_voltage_filter_summary = QtWidgets.QLabel("Suppression Voltage: -")
        self.lbl_suppression_voltage_filter_summary.setStyleSheet("color: #334155; font-size: 11px;")
        self.lbl_suppression_voltage_filter_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.lbl_control_period_filter_summary = QtWidgets.QLabel("Control Period: -")
        self.lbl_control_period_filter_summary.setStyleSheet("color: #334155; font-size: 11px;")
        self.lbl_control_period_filter_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        filter_text_layout.addWidget(filter_title)
        filter_text_layout.addWidget(self.lbl_program_filter_summary)
        filter_text_layout.addWidget(self.lbl_serial_filter_summary)
        filter_text_layout.addWidget(self.lbl_suppression_voltage_filter_summary)
        filter_text_layout.addWidget(self.lbl_control_period_filter_summary)
        filter_text_layout.addStretch(1)
        filter_layout.addLayout(filter_text_layout, 1)

        self.btn_program_filters = QtWidgets.QPushButton("Programs...")
        self.btn_program_filters.clicked.connect(self._open_program_filter_popup)
        self.btn_serial_filters = QtWidgets.QPushButton("Serials...")
        self.btn_serial_filters.clicked.connect(self._open_serial_filter_popup)
        self.btn_suppression_voltage_filters = QtWidgets.QPushButton("Suppression Voltage...")
        self.btn_suppression_voltage_filters.clicked.connect(self._open_suppression_voltage_filter_popup)
        self.btn_control_period_filters = QtWidgets.QPushButton("Control Period...")
        self.btn_control_period_filters.clicked.connect(self._open_control_period_filter_popup)
        self.btn_reset_global_filters = QtWidgets.QPushButton("Reset Filters")
        self.btn_reset_global_filters.clicked.connect(self._reset_global_filters)
        for btn in (
            self.btn_program_filters,
            self.btn_serial_filters,
            self.btn_suppression_voltage_filters,
            self.btn_control_period_filters,
            self.btn_reset_global_filters,
        ):
            btn.setMinimumHeight(32)
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 12px;
                    border-radius: 8px;
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    font-size: 12px;
                    font-weight: 700;
                    color: #0f172a;
                }
                QPushButton:hover { background: #f8fafc; }
                QPushButton:disabled { color: #94a3b8; border-color: #e2e8f0; }
                """
            )
            filter_layout.addWidget(btn)
        root.addWidget(filter_frame, 0)

        splitter = QtWidgets.QSplitter()
        self.main_splitter = splitter
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Left panel: controls
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { background: transparent; border: 0; }")
        self._left_panel_scroll = left_scroll
        left = QtWidgets.QFrame()
        left.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }")
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left)

        title = QtWidgets.QLabel("Test Data Trend / Analyze")
        title.setStyleSheet("font-size: 14px; font-weight: 700;")
        left_layout.addWidget(title)

        # Clear mode switcher (tabs hidden; these buttons are the main UI).
        switch_row = QtWidgets.QHBoxLayout()
        switch_lbl = QtWidgets.QLabel("Mode:")
        switch_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #334155;")
        self.btn_mode_curves = QtWidgets.QPushButton("Plot Curves")
        self.btn_mode_metrics = QtWidgets.QPushButton("Plot Metrics")
        self.btn_mode_perf = QtWidgets.QPushButton("Performance")
        for b in (self.btn_mode_curves, self.btn_mode_metrics, self.btn_mode_perf):
            b.setCheckable(True)
            b.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 10px;
                    border-radius: 8px;
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    font-size: 12px;
                    font-weight: 700;
                    color: #0f172a;
                }
                QPushButton:checked {
                    background: #dbeafe;
                    border-color: #60a5fa;
                    color: #1e3a8a;
                }
                """
            )
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.btn_mode_curves)
        self._mode_group.addButton(self.btn_mode_metrics)
        self._mode_group.addButton(self.btn_mode_perf)
        self.btn_mode_curves.clicked.connect(lambda: self._set_mode("curves"))
        self.btn_mode_metrics.clicked.connect(lambda: self._set_mode("metrics"))
        self.btn_mode_perf.clicked.connect(lambda: self._set_mode("performance"))

        switch_row.addWidget(switch_lbl)
        switch_row.addWidget(self.btn_mode_curves)
        switch_row.addWidget(self.btn_mode_metrics)
        switch_row.addWidget(self.btn_mode_perf)
        switch_row.addStretch(1)
        left_layout.addLayout(switch_row)

        cache_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_cache = QtWidgets.QPushButton("Build / Refresh Cache")
        self.btn_refresh_cache.setMinimumHeight(40)
        self.btn_refresh_cache.clicked.connect(lambda: self._load_cache(rebuild=True))
        self.btn_open_support = QtWidgets.QPushButton("Open Support Workbook")
        self.btn_open_support.setMinimumHeight(40)
        self.btn_open_support.clicked.connect(self._open_support_workbook)
        self.btn_export_debug_excels = QtWidgets.QPushButton("Generate Debug Excel Files")
        self.btn_export_debug_excels.setMinimumHeight(40)
        self.btn_export_debug_excels.clicked.connect(self._generate_debug_excel_files)
        self.btn_plot = QtWidgets.QPushButton("Plot Curves")
        self.btn_plot.setMinimumHeight(40)
        self.btn_plot.setStyleSheet(
            """
            QPushButton {
                padding: 10px 16px;
                border-radius: 10px;
                background: #2563eb;
                border: 1px solid #1d4ed8;
                font-size: 13px;
                font-weight: 800;
                color: #ffffff;
            }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton:pressed { background: #1e40af; }
            QPushButton:disabled { background: #94a3b8; border-color: #94a3b8; }
            """
        )
        self.btn_plot.clicked.connect(self._plot_current_mode)
        self.btn_plot_perf_cached = QtWidgets.QPushButton("Plot Performance (Run Conditions)")
        self.btn_plot_perf_cached.setMinimumHeight(40)
        self.btn_plot_perf_cached.setStyleSheet(
            """
            QPushButton {
                padding: 10px 16px;
                border-radius: 10px;
                background: #0f766e;
                border: 1px solid #115e59;
                font-size: 13px;
                font-weight: 800;
                color: #ffffff;
            }
            QPushButton:hover { background: #115e59; }
            QPushButton:pressed { background: #134e4a; }
            QPushButton:disabled { background: #94a3b8; border-color: #94a3b8; }
            """
        )
        self.btn_plot_perf_cached.clicked.connect(
            lambda: self._plot_performance_cached_condition_means(user_initiated=True)
        )
        self.btn_plot_perf_cached.setVisible(False)
        self.lbl_cache = QtWidgets.QLabel("")
        self.lbl_cache.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_cache.setWordWrap(True)
        cache_row.addWidget(self.btn_refresh_cache)
        cache_row.addWidget(self.btn_open_support)
        cache_row.addWidget(self.btn_export_debug_excels)
        cache_row.addWidget(self.btn_plot)
        cache_row.addWidget(self.btn_plot_perf_cached)
        cache_row.addStretch(1)
        left_layout.addLayout(cache_row)
        left_layout.addWidget(self.lbl_cache)

        self._runs_ex: list[dict] = []
        self._run_display_by_name: dict[str, str] = {}
        self._run_name_by_display: dict[str, str] = {}
        self._run_selection_views: dict[str, list[dict]] = {"sequence": [], "condition": []}

        tabs = QtWidgets.QStackedWidget()
        tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._tabs = tabs

        # Metrics tab
        tab_metrics = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QVBoxLayout(tab_metrics)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(8)
        metrics_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)

        self.list_y_metrics = QtWidgets.QListWidget(tab_metrics)
        self.list_y_metrics.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_y_metrics.hide()
        self.list_y_metrics.itemSelectionChanged.connect(self._on_metric_y_selection_changed)

        metrics_y_row = QtWidgets.QHBoxLayout()
        metrics_y_row.setSpacing(8)
        self.btn_metric_y_columns_popup = QtWidgets.QPushButton("Y Columns...")
        self.btn_metric_y_columns_popup.clicked.connect(self._open_metric_y_columns_popup)
        self.lbl_metric_y_columns_summary = QtWidgets.QLabel("Y Columns: -")
        self.lbl_metric_y_columns_summary.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_metric_y_columns_summary.setWordWrap(False)
        self.lbl_metric_y_columns_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        metrics_y_row.addWidget(self.btn_metric_y_columns_popup)
        metrics_y_row.addWidget(self.lbl_metric_y_columns_summary, 1)
        metrics_layout.addLayout(metrics_y_row)

        self.list_stats = QtWidgets.QListWidget(tab_metrics)
        self.list_stats.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_stats.setMaximumHeight(92)
        self.list_stats.hide()
        self.list_stats.itemSelectionChanged.connect(self._refresh_metric_stats_summary)
        for st in ["mean", "min", "max", "std"]:
            self.list_stats.addItem(QtWidgets.QListWidgetItem(st))
        # Default selection: mean (matches prior single-select default behavior).
        for i in range(self.list_stats.count()):
            it = self.list_stats.item(i)
            if it.text().strip().lower() == "mean":
                it.setSelected(True)
                break

        stats_row = QtWidgets.QHBoxLayout()
        stats_row.setSpacing(8)
        self.btn_metric_stats_popup = QtWidgets.QPushButton("Stats...")
        self.btn_metric_stats_popup.clicked.connect(self._open_metric_stats_popup)
        self.lbl_metric_stats_summary = QtWidgets.QLabel("Stats: mean")
        self.lbl_metric_stats_summary.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_metric_stats_summary.setWordWrap(False)
        self.lbl_metric_stats_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        stats_row.addWidget(self.btn_metric_stats_popup)
        stats_row.addWidget(self.lbl_metric_stats_summary, 1)
        metrics_layout.addLayout(stats_row)

        self.cb_metric_average = QtWidgets.QCheckBox("Average")
        self.cb_metric_average.setChecked(False)
        self.cb_metric_average.setToolTip(
            "Plot the aggregate average of the per-SN mean points as a separate flat series."
        )
        metrics_layout.addWidget(self.cb_metric_average)

        self.btn_metric_plot_source = QtWidgets.QPushButton("View All Sequences")
        self.btn_metric_plot_source.setCheckable(True)
        self.btn_metric_plot_source.setChecked(False)
        self.btn_metric_plot_source.setToolTip(
            "Off: current aggregate TD Metrics Calc data. On: one point per sequence from the sequence-level cache."
        )
        self.btn_metric_plot_source.clicked.connect(lambda *_: self._refresh_stats_preview())
        metrics_layout.addWidget(self.btn_metric_plot_source)

        self.cb_plot_metric_bounds = QtWidgets.QCheckBox("Plot parameter bounds from support workbook")
        self.cb_plot_metric_bounds.setChecked(False)
        self.cb_plot_metric_bounds.setToolTip("Draw min/max bounds entered in the support spreadsheet for the selected run/parameter.")
        metrics_layout.addWidget(self.cb_plot_metric_bounds)

        # Curves tab
        tab_curves = QtWidgets.QWidget()
        curves_layout = QtWidgets.QVBoxLayout(tab_curves)
        curves_layout.setContentsMargins(10, 10, 10, 10)
        curves_layout.setSpacing(8)
        curves_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)

        curves_axes_body = QtWidgets.QFrame()
        self.curves_axes_panel = curves_axes_body
        curves_axes_body.setStyleSheet("QFrame { background: transparent; border: 0; }")
        curves_axes_body.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        curves_axes_layout = QtWidgets.QVBoxLayout(curves_axes_body)
        curves_axes_layout.setContentsMargins(0, 0, 0, 0)
        curves_axes_layout.setSpacing(8)

        self.cb_y_curve = QtWidgets.QComboBox(tab_curves)
        self.cb_y_curve.hide()
        self.cb_y_curve.currentIndexChanged.connect(self._on_curve_y_column_changed)

        row_y_curve = QtWidgets.QHBoxLayout()
        row_y_curve.setSpacing(8)
        self.btn_curve_y_column_popup = QtWidgets.QPushButton("Y Column...")
        self.btn_curve_y_column_popup.clicked.connect(self._open_curve_y_column_popup)
        self.lbl_curve_y_column_summary = QtWidgets.QLabel("Y Column: -")
        self.lbl_curve_y_column_summary.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_curve_y_column_summary.setWordWrap(False)
        self.lbl_curve_y_column_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        row_y_curve.addWidget(self.btn_curve_y_column_popup)
        row_y_curve.addWidget(self.lbl_curve_y_column_summary, 1)
        curves_axes_layout.addLayout(row_y_curve)

        self.cb_x = QtWidgets.QComboBox(tab_curves)
        self.cb_x.hide()
        self.cb_x.currentIndexChanged.connect(self._on_curve_x_column_changed)

        row_curves = QtWidgets.QHBoxLayout()
        row_curves.setSpacing(8)
        self.btn_curve_x_column_popup = QtWidgets.QPushButton("X Column...")
        self.btn_curve_x_column_popup.clicked.connect(self._open_curve_x_column_popup)
        self.lbl_curve_x_column_summary = QtWidgets.QLabel("X Column: -")
        self.lbl_curve_x_column_summary.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_curve_x_column_summary.setWordWrap(False)
        self.lbl_curve_x_column_summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        row_curves.addWidget(self.btn_curve_x_column_popup)
        row_curves.addWidget(self.lbl_curve_x_column_summary, 1)
        curves_axes_layout.addLayout(row_curves)

        curves_layout.addWidget(curves_axes_body, 0)

        # Performance tab (candidate discovery + project-wide plotting)
        tab_perf = QtWidgets.QWidget()
        perf_layout = QtWidgets.QVBoxLayout(tab_perf)
        perf_layout.setContentsMargins(10, 10, 10, 10)
        perf_layout.setSpacing(8)

        lbl_perf = QtWidgets.QLabel("Performance equations and plots (all runs, all serials)")
        lbl_perf.setStyleSheet("font-size: 12px; font-weight: 800; color: #0f172a;")
        perf_layout.addWidget(lbl_perf)

        perf_panel = QtWidgets.QFrame()
        perf_panel.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }")
        perf_plot_layout = QtWidgets.QVBoxLayout(perf_panel)
        perf_plot_layout.setContentsMargins(10, 10, 10, 10)
        perf_plot_layout.setSpacing(8)
        perf_layout.addWidget(perf_panel, 1)

        row_perf_output = QtWidgets.QHBoxLayout()
        row_perf_output.addWidget(QtWidgets.QLabel("Output:"))
        self.cb_perf_y_col = QtWidgets.QComboBox()
        row_perf_output.addWidget(self.cb_perf_y_col, 1)
        perf_plot_layout.addLayout(row_perf_output)

        row_perf_x = QtWidgets.QHBoxLayout()
        row_perf_x.addWidget(QtWidgets.QLabel("Input 1:"))
        self.cb_perf_x_col = QtWidgets.QComboBox()
        row_perf_x.addWidget(self.cb_perf_x_col, 1)
        perf_plot_layout.addLayout(row_perf_x)

        row_perf_z = QtWidgets.QHBoxLayout()
        row_perf_z.addWidget(QtWidgets.QLabel("Input 2 (Optional):"))
        self.cb_perf_z_col = QtWidgets.QComboBox()
        row_perf_z.addWidget(self.cb_perf_z_col, 1)
        perf_plot_layout.addLayout(row_perf_z)

        self.lbl_perf_axes = QtWidgets.QLabel("Output: - | Input 1: - | Input 2: -")
        self.lbl_perf_axes.setStyleSheet("color: #334155; font-size: 11px;")
        self.lbl_perf_axes.setWordWrap(True)
        perf_plot_layout.addWidget(self.lbl_perf_axes)

        self.lbl_perf_common_runs = QtWidgets.QLabel("Common runs for selected variables: -")
        self.lbl_perf_common_runs.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_perf_common_runs.setWordWrap(True)
        perf_plot_layout.addWidget(self.lbl_perf_common_runs)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Condition family:"))
        self.rb_perf_steady_state = QtWidgets.QRadioButton("Steady-state")
        self.rb_perf_pulsed_mode = QtWidgets.QRadioButton("Pulsed mode")
        self.rb_perf_steady_state.setChecked(True)
        self._perf_run_type_group = QtWidgets.QButtonGroup(self)
        self._perf_run_type_group.setExclusive(True)
        self._perf_run_type_group.addButton(self.rb_perf_steady_state)
        self._perf_run_type_group.addButton(self.rb_perf_pulsed_mode)
        mode_row.addWidget(self.rb_perf_steady_state)
        mode_row.addWidget(self.rb_perf_pulsed_mode)
        mode_row.addStretch(1)
        perf_plot_layout.addLayout(mode_row)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("PM control-period filter:"))
        self.cb_perf_filter_mode = QtWidgets.QComboBox()
        self.cb_perf_filter_mode.addItem("All run conditions", "all_conditions")
        self.cb_perf_filter_mode.addItem("Match control period", "match_control_period")
        filter_row.addWidget(self.cb_perf_filter_mode, 1)
        filter_row.addWidget(QtWidgets.QLabel("Control period:"))
        self.cb_perf_control_period = QtWidgets.QComboBox()
        self.cb_perf_control_period.setEnabled(False)
        filter_row.addWidget(self.cb_perf_control_period, 1)
        perf_plot_layout.addLayout(filter_row)

        fit_row = QtWidgets.QHBoxLayout()
        self.cb_perf_fit = QtWidgets.QCheckBox("Fit equation")
        self.cb_perf_fit.setChecked(True)
        self.cb_perf_fit_model = QtWidgets.QComboBox()
        self.cb_perf_fit_model.addItem("Auto", "auto")
        self.cb_perf_fit_model.addItem("Polynomial", "polynomial")
        self.cb_perf_fit_model.addItem("Logarithmic", "logarithmic")
        self.cb_perf_fit_model.addItem("Saturating Exponential", "saturating_exponential")
        self.cb_perf_fit_model.addItem("Hybrid Saturating + Linear", "hybrid_saturating_linear")
        self.cb_perf_fit_model.addItem("Hybrid + Quadratic Residual", "hybrid_quadratic_residual")
        self.cb_perf_fit_model.addItem("Monotone PCHIP", "monotone_pchip")
        self.cb_perf_fit_model.addItem("Piecewise Auto", "piecewise_auto")
        self.cb_perf_fit_model.addItem("Piecewise 2-Segment", "piecewise_2")
        self.cb_perf_fit_model.addItem("Piecewise 3-Segment", "piecewise_3")
        self.sp_perf_degree = QtWidgets.QSpinBox()
        self.sp_perf_degree.setRange(1, 6)
        self.sp_perf_degree.setValue(2)
        self.cb_perf_norm_x = QtWidgets.QCheckBox("Normalize X")
        self.cb_perf_norm_x.setChecked(True)
        self.cb_perf_surface_model = QtWidgets.QComboBox()
        self.cb_perf_surface_model.addItem("Auto Surface", "auto_surface")
        self.cb_perf_surface_model.addItem("Quadratic Surface", "quadratic_surface")
        self.cb_perf_surface_model.addItem("Quadratic Surface + Control Period", "quadratic_surface_control_period")
        self.cb_perf_surface_model.addItem("Plane", "plane")
        fit_row.addWidget(self.cb_perf_fit)
        fit_row.addWidget(QtWidgets.QLabel("Model:"))
        fit_row.addWidget(self.cb_perf_fit_model)
        fit_row.addWidget(QtWidgets.QLabel("3D Model:"))
        fit_row.addWidget(self.cb_perf_surface_model)
        fit_row.addWidget(QtWidgets.QLabel("Degree:"))
        fit_row.addWidget(self.sp_perf_degree)
        fit_row.addWidget(self.cb_perf_norm_x)
        fit_row.addStretch(1)
        perf_plot_layout.addLayout(fit_row)

        # Runs are always all runs in the cache (no selector).

        # Serials are always all serials in the cache (no selector); highlighted serial is chosen below.

        self.lbl_stats_perf = QtWidgets.QLabel("Stats (applies to both axes)")
        self.lbl_stats_perf.setStyleSheet("font-size: 12px; font-weight: 700; color: #0f172a;")
        self.lbl_stats_perf.setVisible(False)
        perf_plot_layout.addWidget(self.lbl_stats_perf)

        self.list_perf_stats = QtWidgets.QListWidget()
        self.list_perf_stats.setMaximumHeight(110)
        self.list_perf_stats.setVisible(False)
        perf_plot_layout.addWidget(self.list_perf_stats)

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("View stat:"))
        self.cb_perf_view_stat = QtWidgets.QComboBox()
        view_row.addWidget(self.cb_perf_view_stat, 1)
        perf_plot_layout.addLayout(view_row)

        self.btn_perf_equations_popup = QtWidgets.QPushButton("Performance Equations...")
        perf_plot_layout.addWidget(self.btn_perf_equations_popup, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        perf_eq_body = QtWidgets.QWidget()
        perf_eq_layout = QtWidgets.QVBoxLayout(perf_eq_body)
        perf_eq_layout.setContentsMargins(0, 0, 0, 0)
        perf_eq_layout.setSpacing(8)

        eq_row = QtWidgets.QHBoxLayout()
        lbl_eq = QtWidgets.QLabel("Equations (per stat)")
        lbl_eq.setStyleSheet("font-size: 12px; font-weight: 700; color: #0f172a;")
        eq_row.addWidget(lbl_eq)
        eq_row.addStretch(1)
        self.btn_perf_save_equation = QtWidgets.QPushButton("Save Performance Equation")
        self.btn_perf_save_equation.setEnabled(False)
        eq_row.addWidget(self.btn_perf_save_equation)
        self.btn_perf_saved_equations = QtWidgets.QPushButton("Saved Performance Equations")
        self.btn_perf_saved_equations.setEnabled(True)
        eq_row.addWidget(self.btn_perf_saved_equations)
        self.cb_perf_include_reg_checker = QtWidgets.QCheckBox("Include Mean Regression Checker")
        self.cb_perf_include_reg_checker.setChecked(True)
        eq_row.addWidget(self.cb_perf_include_reg_checker)
        self.btn_perf_export_interactive = QtWidgets.QPushButton("Export Interactive Workbook")
        self.btn_perf_export_interactive.setEnabled(False)
        eq_row.addWidget(self.btn_perf_export_interactive)
        self.btn_perf_export_equations = QtWidgets.QPushButton("Export Equation to Excel")
        self.btn_perf_export_equations.setEnabled(False)
        eq_row.addWidget(self.btn_perf_export_equations)
        perf_eq_layout.addLayout(eq_row)

        self.tbl_perf_equations = QtWidgets.QTableWidget(0, 9)
        self.tbl_perf_equations.setHorizontalHeaderLabels(
            [
                "stat",
                "master_family",
                "master_model",
                "master_x_norm",
                "master_rmse",
                "highlighted_family",
                "highlighted_model",
                "highlighted_x_norm",
                "highlighted_rmse",
            ]
        )
        self.tbl_perf_equations.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_perf_equations.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_perf_equations.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        try:
            self.tbl_perf_equations.horizontalHeader().setStretchLastSection(True)
        except Exception:
            pass
        self.tbl_perf_equations.setMinimumHeight(0)
        perf_eq_layout.addWidget(self.tbl_perf_equations, 1)

        perf_eq_popup = QtWidgets.QDialog(self)
        perf_eq_popup.setWindowTitle("Performance Equations")
        perf_eq_popup.resize(1180, 420)
        perf_eq_popup_layout = QtWidgets.QVBoxLayout(perf_eq_popup)
        perf_eq_popup_layout.setContentsMargins(12, 12, 12, 12)
        perf_eq_popup_layout.setSpacing(8)
        perf_eq_popup_layout.addWidget(perf_eq_body, 1)
        perf_eq_close_row = QtWidgets.QHBoxLayout()
        perf_eq_close_row.addStretch(1)
        btn_close_perf_eq_popup = QtWidgets.QPushButton("Close")
        btn_close_perf_eq_popup.clicked.connect(perf_eq_popup.close)
        perf_eq_close_row.addWidget(btn_close_perf_eq_popup)
        perf_eq_popup_layout.addLayout(perf_eq_close_row)
        self._perf_equations_popup = perf_eq_popup

        tabs.addWidget(tab_metrics)
        tabs.addWidget(tab_curves)
        tabs.addWidget(tab_perf)
        left_layout.addWidget(tabs, 0)

        self.run_selector_frame = QtWidgets.QFrame()
        self.run_selector_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }"
        )
        run_selector_layout = QtWidgets.QVBoxLayout(self.run_selector_frame)
        run_selector_layout.setContentsMargins(10, 10, 10, 10)
        run_selector_layout.setSpacing(8)
        run_selector_title = QtWidgets.QLabel("Run Selection")
        run_selector_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #334155;")
        run_selector_layout.addWidget(run_selector_title)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self.cb_run_mode = QtWidgets.QComboBox()
        self.cb_run_mode.addItem("Sequence", "sequence")
        self.cb_run_mode.addItem("Run Conditions", "condition")
        self.cb_run_mode.currentIndexChanged.connect(lambda *_: self._refresh_run_dropdown())
        form.addRow("Select By:", self.cb_run_mode)

        self.lbl_run_combo = QtWidgets.QLabel("Run:")
        self.cb_run = QtWidgets.QComboBox()
        self.cb_run.currentIndexChanged.connect(self._refresh_columns_for_run)
        form.addRow(self.lbl_run_combo, self.cb_run)

        self.metrics_condition_frame = QtWidgets.QFrame()
        metrics_condition_layout = QtWidgets.QVBoxLayout(self.metrics_condition_frame)
        metrics_condition_layout.setContentsMargins(0, 0, 0, 0)
        metrics_condition_layout.setSpacing(6)
        self.lbl_metrics_condition_picker = QtWidgets.QLabel("Run Conditions:")
        self.lbl_metrics_condition_picker.setStyleSheet("font-size: 12px; font-weight: 700; color: #334155;")
        metrics_condition_layout.addWidget(self.lbl_metrics_condition_picker)
        self.list_metric_run_conditions = QtWidgets.QListWidget()
        self.list_metric_run_conditions.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.list_metric_run_conditions.itemChanged.connect(self._on_metric_condition_selection_changed)
        metrics_condition_layout.addWidget(self.list_metric_run_conditions, 1)
        metrics_condition_actions = QtWidgets.QHBoxLayout()
        self.btn_metric_run_conditions_all = QtWidgets.QPushButton("Select All")
        self.btn_metric_run_conditions_all.clicked.connect(lambda: self._set_metric_condition_selection_ids(None))
        self.btn_metric_run_conditions_clear = QtWidgets.QPushButton("Clear")
        self.btn_metric_run_conditions_clear.clicked.connect(lambda: self._set_metric_condition_selection_ids([]))
        metrics_condition_actions.addWidget(self.btn_metric_run_conditions_all)
        metrics_condition_actions.addWidget(self.btn_metric_run_conditions_clear)
        metrics_condition_actions.addStretch(1)
        metrics_condition_layout.addLayout(metrics_condition_actions)

        self.lbl_run_details = QtWidgets.QLabel("Sequence: -")
        self.lbl_run_details.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_run_details.setWordWrap(True)
        form.addRow("", self.lbl_run_details)
        run_selector_layout.addLayout(form)
        run_selector_layout.addWidget(self.metrics_condition_frame)
        left_layout.addWidget(self.run_selector_frame)

        # Serial highlight (not selection/filtering for plotting)
        lbl_serials = QtWidgets.QLabel("Highlight Serial (optional)")
        lbl_serials.setStyleSheet("font-size: 12px; font-weight: 700;")
        left_layout.addWidget(lbl_serials)

        serial_pick_row = QtWidgets.QHBoxLayout()
        self.btn_highlight_serials = QtWidgets.QPushButton("Select Highlighted Serials...")
        self.btn_highlight_serials.clicked.connect(self._open_highlight_serials_popup)
        serial_pick_row.addWidget(self.btn_highlight_serials)
        serial_pick_row.addStretch(1)
        left_layout.addLayout(serial_pick_row)

        self.lbl_highlight_serials = QtWidgets.QLabel("Highlighted serials: -")
        self.lbl_highlight_serials.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_highlight_serials.setWordWrap(True)
        left_layout.addWidget(self.lbl_highlight_serials)
        left_layout.addStretch(1)

        splitter.addWidget(left_scroll)

        # Right panel: plot + stats
        right = QtWidgets.QFrame()
        right.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }")
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        plot_header = QtWidgets.QHBoxLayout()
        plot_title = QtWidgets.QLabel("Plot")
        plot_title.setStyleSheet("font-size: 14px; font-weight: 700;")
        plot_header.addWidget(plot_title)
        plot_header.addStretch(1)

        zoom_btn_css = """
        QPushButton {
            padding: 5px 10px;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            font-size: 12px;
            font-weight: 800;
            color: #0f172a;
        }
        QPushButton:hover { background: #f8fafc; }
        QPushButton:disabled { color: #94a3b8; border-color: #e2e8f0; }
        """
        self.btn_zone_zoom = QtWidgets.QPushButton("Magnify")
        self.btn_zone_zoom.setCheckable(True)
        self.btn_zoom_out = QtWidgets.QPushButton("Zoom -")
        self.btn_zoom_in = QtWidgets.QPushButton("Zoom +")
        self.btn_zoom_reset = QtWidgets.QPushButton("Reset")
        self.btn_expand_plot = QtWidgets.QPushButton("Expand")
        for b in (
            self.btn_zone_zoom,
            self.btn_zoom_out,
            self.btn_zoom_in,
            self.btn_zoom_reset,
            self.btn_expand_plot,
        ):
            b.setEnabled(False)
            b.setStyleSheet(zoom_btn_css)
        self.btn_zone_zoom.setToolTip("Magnify zones: click-drag a rectangle on the plot to zoom to that area")
        self.btn_zoom_out.setToolTip("Zoom out (mouse wheel also works)")
        self.btn_zoom_in.setToolTip("Zoom in (mouse wheel also works)")
        self.btn_zoom_reset.setToolTip("Reset zoom to the default view")
        self.btn_expand_plot.setToolTip("Open a larger popup for zooming/panning")
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_main_plot(1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_main_plot(0.8))
        self.btn_zoom_reset.clicked.connect(self._reset_main_plot_zoom)
        self.btn_expand_plot.clicked.connect(self._open_main_plot_popup)
        self.btn_view_bands = QtWidgets.QPushButton("View Bands...")
        self.btn_view_bands.setStyleSheet(zoom_btn_css)
        self.btn_view_bands.setToolTip("Set optional view bands for the current plot axes")
        self.btn_view_bands.clicked.connect(self._open_plot_band_popup)
        self.btn_plot_legend = QtWidgets.QPushButton("Legend...")
        self.btn_plot_legend.setStyleSheet(zoom_btn_css)
        self.btn_plot_legend.setVisible(False)
        self.btn_plot_legend.setEnabled(False)
        self.btn_plot_legend.clicked.connect(self._open_main_plot_legend_popup)
        plot_header.addWidget(self.btn_zone_zoom)
        plot_header.addWidget(self.btn_zoom_out)
        plot_header.addWidget(self.btn_zoom_in)
        plot_header.addWidget(self.btn_zoom_reset)
        plot_header.addWidget(self.btn_expand_plot)
        plot_header.addWidget(self.btn_view_bands)
        plot_header.addWidget(self.btn_plot_legend)
        right_layout.addLayout(plot_header)

        self.plot_band_frame = QtWidgets.QFrame()
        self.plot_band_frame.setStyleSheet("QFrame { border: 0; background: transparent; }")
        self.plot_band_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        plot_band_layout = QtWidgets.QVBoxLayout(self.plot_band_frame)
        plot_band_layout.setContentsMargins(0, 0, 0, 0)
        plot_band_layout.setSpacing(6)

        self.plot_band_x_row = QtWidgets.QHBoxLayout()
        self.plot_band_x_row.addWidget(QtWidgets.QLabel("View X band:"))
        self.ed_plot_x_band_min = QtWidgets.QLineEdit()
        self.ed_plot_x_band_min.setPlaceholderText("min")
        self.ed_plot_x_band_min.setToolTip("Optional lower X-axis view bound. Blank leaves it unbounded.")
        self.plot_band_x_row.addWidget(self.ed_plot_x_band_min, 1)
        self.plot_band_x_row.addWidget(QtWidgets.QLabel("to"))
        self.ed_plot_x_band_max = QtWidgets.QLineEdit()
        self.ed_plot_x_band_max.setPlaceholderText("max")
        self.ed_plot_x_band_max.setToolTip("Optional upper X-axis view bound. Blank leaves it unbounded.")
        self.plot_band_x_row.addWidget(self.ed_plot_x_band_max, 1)
        plot_band_layout.addLayout(self.plot_band_x_row)

        self.plot_band_y_row = QtWidgets.QHBoxLayout()
        self.plot_band_y_row.addWidget(QtWidgets.QLabel("View Y band:"))
        self.ed_plot_y_band_min = QtWidgets.QLineEdit()
        self.ed_plot_y_band_min.setPlaceholderText("min")
        self.ed_plot_y_band_min.setToolTip("Optional lower Y-axis view bound. Blank leaves it unbounded.")
        self.plot_band_y_row.addWidget(self.ed_plot_y_band_min, 1)
        self.plot_band_y_row.addWidget(QtWidgets.QLabel("to"))
        self.ed_plot_y_band_max = QtWidgets.QLineEdit()
        self.ed_plot_y_band_max.setPlaceholderText("max")
        self.ed_plot_y_band_max.setToolTip("Optional upper Y-axis view bound. Blank leaves it unbounded.")
        self.plot_band_y_row.addWidget(self.ed_plot_y_band_max, 1)
        plot_band_layout.addLayout(self.plot_band_y_row)
        for edit in (
            self.ed_plot_x_band_min,
            self.ed_plot_x_band_max,
            self.ed_plot_y_band_min,
            self.ed_plot_y_band_max,
        ):
            edit.textChanged.connect(lambda *_: self._apply_current_plot_view_bands())
            edit.editingFinished.connect(self._finalize_plot_view_bands)

        self.plot_container = QtWidgets.QFrame()
        plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._figure = None
        self._axes = None
        self._init_plot_area(plot_layout)
        right_layout.addWidget(self.plot_container, 6)

        self.lbl_plot_note = QtWidgets.QLabel("")
        self.lbl_plot_note.setVisible(False)
        self.lbl_plot_note.setWordWrap(True)
        self.lbl_plot_note.setStyleSheet(
            "color: #0f172a; font-size: 12px; font-weight: 800; padding: 2px 2px 0 2px;"
        )
        right_layout.addWidget(self.lbl_plot_note)

        # Highlighted stats: centered dropdown button + single-row summary (keeps the plot area larger).
        self.btn_stats_toggle = QtWidgets.QPushButton("Highlighted Serial Stats ▸")
        self.btn_stats_toggle.setCheckable(True)
        self.btn_stats_toggle.setChecked(False)
        self.btn_stats_toggle.setStyleSheet(
            """
            QPushButton {
                padding: 6px 12px;
                border-radius: 10px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                font-size: 12px;
                font-weight: 800;
                color: #0f172a;
            }
            QPushButton:hover { background: #f8fafc; }
            QPushButton:checked {
                background: #f1f5f9;
                border-color: #cbd5e1;
            }
            """
        )
        stats_toggle_row = QtWidgets.QHBoxLayout()
        stats_toggle_row.addStretch(1)
        stats_toggle_row.addWidget(self.btn_stats_toggle)
        stats_toggle_row.addStretch(1)
        right_layout.addLayout(stats_toggle_row)

        self._stats_frame = QtWidgets.QFrame()
        self._stats_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }"
        )
        stats_lay = QtWidgets.QVBoxLayout(self._stats_frame)
        stats_lay.setContentsMargins(10, 8, 10, 8)
        stats_lay.setSpacing(8)

        def _stat_pair(label: str) -> tuple[QtWidgets.QLabel, QtWidgets.QLabel]:
            k = QtWidgets.QLabel(label)
            k.setStyleSheet("color:#334155; font-size: 11px; font-weight: 800;")
            v = QtWidgets.QLabel("—")
            v.setStyleSheet("color:#0f172a; font-size: 11px; font-weight: 700;")
            return k, v

        self._stats_values: dict[str, QtWidgets.QLabel] = {}
        stats_top_row = QtWidgets.QHBoxLayout()
        stats_top_row.setSpacing(16)
        for key, label in (
            ("count", "Count"),
            ("mean", "Mean"),
            ("std", "Std"),
            ("min", "Min"),
            ("max", "Max"),
        ):
            k_lbl, v_lbl = _stat_pair(label)
            box = QtWidgets.QHBoxLayout()
            box.setSpacing(6)
            box.addWidget(k_lbl)
            box.addWidget(v_lbl)
            stats_top_row.addLayout(box)
            self._stats_values[key] = v_lbl
        stats_top_row.addStretch(1)
        stats_lay.addLayout(stats_top_row)

        serial_key_lbl, serial_val_lbl = _stat_pair("Highlighted SN")
        serial_val_lbl.setWordWrap(True)
        serial_row = QtWidgets.QHBoxLayout()
        serial_row.setSpacing(6)
        serial_row.addWidget(serial_key_lbl)
        serial_row.addWidget(serial_val_lbl, 1)
        stats_lay.addLayout(serial_row)
        self._stats_values["serial"] = serial_val_lbl
        self._stats_frame.setVisible(False)
        right_layout.addWidget(self._stats_frame, 0)

        def _toggle_stats(opened: bool) -> None:
            self._stats_frame.setVisible(bool(opened))
            self.btn_stats_toggle.setText("Highlighted Serial Stats ▾" if opened else "Highlighted Serial Stats ▸")

        self.btn_stats_toggle.toggled.connect(_toggle_stats)
        _toggle_stats(False)

        # Bottom actions: saving + auto-plots
        self.btn_add_auto_plot = QtWidgets.QPushButton("Add to Auto-Plots")
        self.btn_add_auto_plot.setEnabled(False)
        self.btn_add_auto_plot.clicked.connect(self._add_current_plot_to_autoplots)
        self.btn_save_plot_pdf = QtWidgets.QPushButton("Save Plot PDF")
        self.btn_save_plot_pdf.setEnabled(False)
        self.btn_save_plot_pdf.clicked.connect(self._save_plot_pdf)
        self.btn_view_auto_plots = QtWidgets.QPushButton("View Auto-Plots...")
        self.btn_view_auto_plots.setEnabled(False)
        self.btn_view_auto_plots.clicked.connect(self._open_auto_plots_popup)
        self.lbl_source = QtWidgets.QLabel("")
        self.lbl_source.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_source.setWordWrap(True)

        footer_frame = QtWidgets.QFrame()
        footer_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }"
        )
        footer_layout = QtWidgets.QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        footer_layout.setSpacing(8)

        footer_row = QtWidgets.QHBoxLayout()
        footer_row.setSpacing(8)
        footer_row.addWidget(self.btn_add_auto_plot)
        footer_row.addWidget(self.btn_save_plot_pdf)
        footer_row.addWidget(self.btn_view_auto_plots)
        footer_row.addStretch(1)
        footer_layout.addLayout(footer_row)
        footer_layout.addWidget(self.lbl_source)
        right_layout.addWidget(footer_frame, 0)

        splitter.addWidget(right)
        splitter.setHandleWidth(0)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        try:
            splitter.handle(1).setEnabled(False)
            splitter.handle(1).hide()
        except Exception:
            pass

        self._report_progress = RunProgressDialog(self)
        try:
            self._report_progress.btn_cancel.hide()
        except Exception:
            pass

        self._refresh_global_filter_summaries()
        self._load_cache(rebuild=False)
        self._load_auto_plots()
        self._refresh_metric_selector_summaries()
        self._set_mode("curves")

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._schedule_mode_panel_height_sync()
        QtCore.QTimer.singleShot(0, self._finalize_initial_window_size)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._schedule_mode_panel_height_sync()

    def _preferred_left_panel_width(self) -> int:
        panel = self._left_panel_scroll
        if panel is None:
            return 0
        panel_widget = panel.widget()
        if panel_widget is None:
            return 0
        try:
            panel_widget.ensurePolished()
            layout = panel_widget.layout()
            if layout is not None:
                layout.activate()
        except Exception:
            pass

        width_candidates: list[int] = []
        for candidate in (
            getattr(panel_widget, "sizeHint", None),
            getattr(panel_widget, "minimumSizeHint", None),
        ):
            if callable(candidate):
                try:
                    width_candidates.append(int(candidate().width()))
                except Exception:
                    pass

        try:
            for child in panel_widget.findChildren(QtWidgets.QWidget):
                if not child.isVisibleTo(panel_widget):
                    continue
                pos = child.mapTo(panel_widget, QtCore.QPoint(0, 0))
                child_width = max(
                    int(child.sizeHint().width()),
                    int(child.minimumSizeHint().width()),
                    int(child.minimumWidth()),
                )
                width_candidates.append(int(pos.x()) + child_width)
        except Exception:
            pass

        try:
            width_candidates.append(panel.frameWidth() * 2)
        except Exception:
            pass

        preferred = max(width_candidates or [0])
        if preferred <= 0:
            return 0
        return preferred + 24

    def _initialize_left_panel_width(self) -> None:
        if self._left_panel_width_initialized:
            return
        panel = self._left_panel_scroll
        splitter = getattr(self, "main_splitter", None)
        if panel is None or splitter is None:
            return
        width = max(panel.width(), self._preferred_left_panel_width())
        if width <= 0:
            try:
                width = max(panel.sizeHint().width(), panel.minimumSizeHint().width())
            except Exception:
                width = 0
        if width <= 0:
            return
        self._left_panel_locked_width = int(width)
        self._left_panel_width_initialized = True
        panel.setMinimumWidth(self._left_panel_locked_width)
        panel.setMaximumWidth(self._left_panel_locked_width)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        total_width = max(1, splitter.width())
        right_width = max(1, total_width - self._left_panel_locked_width)
        splitter.setSizes([self._left_panel_locked_width, right_width])

    def _finalize_initial_window_size(self) -> None:
        if self._startup_size_locked:
            return
        try:
            layout = self.layout()
            if layout is not None:
                layout.activate()
        except Exception:
            pass
        self._sync_mode_panel_height()

        panel = self._left_panel_scroll
        splitter = getattr(self, "main_splitter", None)
        left_width = self._preferred_left_panel_width()

        right_width = 0
        if splitter is not None and splitter.count() >= 2:
            right_panel = splitter.widget(1)
            if right_panel is not None:
                try:
                    right_width = max(right_panel.sizeHint().width(), right_panel.minimumSizeHint().width())
                except Exception:
                    right_width = 0

        filter_width = 0
        try:
            filter_width = max(self.filter_frame.sizeHint().width(), self.filter_frame.minimumSizeHint().width())
        except Exception:
            filter_width = 0

        try:
            size_hint = self.sizeHint()
            minimum_hint = self.minimumSizeHint()
            target_width = max(
                self.width(),
                size_hint.width(),
                minimum_hint.width(),
                filter_width,
                left_width + max(right_width, 720) + max(0, splitter.handleWidth() if splitter is not None else 0) + 32,
            )
            target_height = max(self.height(), size_hint.height(), minimum_hint.height(), 760)
        except Exception:
            target_width = self.width()
            target_height = self.height()

        self.resize(target_width, target_height)
        _fit_widget_to_screen(self)
        self._initialize_left_panel_width()
        self.setMinimumSize(self.size())
        self._startup_size_locked = True

    def _schedule_mode_panel_height_sync(self) -> None:
        if not hasattr(self, "_tabs"):
            return
        QtCore.QTimer.singleShot(0, self._sync_mode_panel_height)

    def _sync_mode_panel_height(self) -> None:
        stack = getattr(self, "_tabs", None)
        if stack is None or not hasattr(stack, "currentWidget"):
            return
        current = stack.currentWidget()
        if current is None:
            return
        try:
            layout = current.layout()
            if layout is not None:
                layout.activate()
        except Exception:
            pass
        try:
            hint_height = max(current.minimumSizeHint().height(), current.sizeHint().height())
        except Exception:
            hint_height = 0
        if hint_height <= 0:
            return
        try:
            stack.setMinimumHeight(hint_height)
            stack.setMaximumHeight(hint_height)
            stack.updateGeometry()
        except Exception:
            pass

    @staticmethod
    def _metric_title_suffix(stats: list[str] | tuple[str, ...] | None) -> str:
        items = [str(s).strip().lower() for s in (stats or []) if str(s).strip()]
        title_items = [s for s in items if s != "average"]
        return "/".join(title_items)

    @staticmethod
    def _popup_selection_summary(
        selected_items: list[str] | tuple[str, ...] | None,
        *,
        total_count: int,
        empty_text: str = "-",
    ) -> str:
        selected = [str(item).strip() for item in (selected_items or []) if str(item).strip()]
        total = max(0, int(total_count or 0))
        if total <= 0:
            return empty_text
        if not selected:
            return "None selected"
        if len(selected) >= total:
            return f"All ({total})"
        if len(selected) <= 2:
            return ", ".join(selected)
        return f"{len(selected)} of {total} selected"

    @staticmethod
    def _list_widget_item_texts(list_widget: QtWidgets.QListWidget | None) -> list[str]:
        if list_widget is None:
            return []
        return [
            str(list_widget.item(i).text() or "").strip()
            for i in range(list_widget.count())
            if list_widget.item(i) is not None and str(list_widget.item(i).text() or "").strip()
        ]

    @staticmethod
    def _selected_list_widget_texts(list_widget: QtWidgets.QListWidget | None) -> list[str]:
        if list_widget is None:
            return []
        return [str(item.text() or "").strip() for item in list_widget.selectedItems() if str(item.text() or "").strip()]

    @staticmethod
    def _combo_box_current_value(combo_box: QtWidgets.QComboBox | None) -> str:
        if combo_box is None:
            return ""
        try:
            value = combo_box.currentData()
        except Exception:
            value = None
        txt = str(value if value is not None else combo_box.currentText() or "").strip()
        if txt:
            return txt
        return str(combo_box.currentText() or "").strip()

    def _set_list_widget_selection(self, list_widget: QtWidgets.QListWidget, values: list[str] | tuple[str, ...]) -> None:
        wanted = {str(value).strip() for value in (values or []) if str(value).strip()}
        list_widget.blockSignals(True)
        try:
            list_widget.clearSelection()
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item is not None and item.text() in wanted:
                    item.setSelected(True)
        finally:
            list_widget.blockSignals(False)

    def _on_metric_y_selection_changed(self) -> None:
        self._refresh_metric_y_columns_summary()
        self._refresh_stats_preview()

    def _refresh_metric_y_columns_summary(self) -> None:
        if not hasattr(self, "lbl_metric_y_columns_summary"):
            return
        selected = self._selected_list_widget_texts(getattr(self, "list_y_metrics", None))
        all_items = self._list_widget_item_texts(getattr(self, "list_y_metrics", None))
        text = "Y Columns: " + self._popup_selection_summary(selected, total_count=len(all_items))
        self.lbl_metric_y_columns_summary.setText(text)
        self.lbl_metric_y_columns_summary.setToolTip(", ".join(selected))
        self._schedule_mode_panel_height_sync()

    def _refresh_metric_stats_summary(self) -> None:
        if not hasattr(self, "lbl_metric_stats_summary"):
            return
        selected = self._selected_list_widget_texts(getattr(self, "list_stats", None))
        all_items = self._list_widget_item_texts(getattr(self, "list_stats", None))
        text = "Stats: " + self._popup_selection_summary(selected, total_count=len(all_items))
        self.lbl_metric_stats_summary.setText(text)
        self.lbl_metric_stats_summary.setToolTip(", ".join(selected))
        self._schedule_mode_panel_height_sync()

    def _refresh_metric_selector_summaries(self) -> None:
        self._refresh_metric_y_columns_summary()
        self._refresh_metric_stats_summary()

    def _current_curve_y_name(self) -> str:
        return self._combo_box_current_value(getattr(self, "cb_y_curve", None))

    def _current_curve_x_key(self) -> str:
        return self._combo_box_current_value(getattr(self, "cb_x", None))

    def _current_curve_x_label(self) -> str:
        if not hasattr(self, "cb_x"):
            return ""
        label = str(self.cb_x.currentText() or "").strip()
        return label or self._current_curve_x_key()

    def _refresh_curve_y_column_summary(self) -> None:
        if not hasattr(self, "lbl_curve_y_column_summary"):
            return
        selected = self._current_curve_y_name()
        text = f"Y Column: {selected or '-'}"
        self.lbl_curve_y_column_summary.setText(text)
        self.lbl_curve_y_column_summary.setToolTip(selected)
        self._schedule_mode_panel_height_sync()

    def _refresh_curve_x_column_summary(self) -> None:
        if not hasattr(self, "lbl_curve_x_column_summary"):
            return
        selected = self._current_curve_x_label()
        text = f"X Column: {selected or '-'}"
        self.lbl_curve_x_column_summary.setText(text)
        self.lbl_curve_x_column_summary.setToolTip(selected)
        self._schedule_mode_panel_height_sync()

    def _refresh_curve_selector_summaries(self) -> None:
        self._refresh_curve_y_column_summary()
        self._refresh_curve_x_column_summary()

    def _on_curve_y_column_changed(self) -> None:
        self._refresh_curve_y_column_summary()
        self._refresh_stats_preview()

    def _on_curve_x_column_changed(self) -> None:
        self._refresh_curve_x_column_summary()
        self._refresh_curve_y_columns()

    def _open_metric_y_columns_popup(self) -> None:
        if not hasattr(self, "list_y_metrics"):
            return
        entries = [
            {"value": text, "label": text, "search": text.lower()}
            for text in self._list_widget_item_texts(self.list_y_metrics)
        ]
        chosen = self._show_filter_checklist_popup(
            title="Metric Y Columns",
            entries=entries,
            selected_values=self._selected_list_widget_texts(self.list_y_metrics),
        )
        if chosen is None:
            return
        self._set_list_widget_selection(self.list_y_metrics, chosen)
        self._on_metric_y_selection_changed()

    def _open_metric_stats_popup(self) -> None:
        if not hasattr(self, "list_stats"):
            return
        entries = [
            {"value": text, "label": text, "search": text.lower()}
            for text in self._list_widget_item_texts(self.list_stats)
        ]
        chosen = self._show_filter_checklist_popup(
            title="Metric Stats",
            entries=entries,
            selected_values=self._selected_list_widget_texts(self.list_stats),
        )
        if chosen is None:
            return
        self._set_list_widget_selection(self.list_stats, chosen)
        self._refresh_metric_stats_summary()

    def _show_filter_single_select_popup(
        self,
        *,
        title: str,
        entries: list[dict],
        selected_value: str,
    ) -> str | None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(720, 540)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        ed_filter = QtWidgets.QLineEdit()
        ed_filter.setPlaceholderText("Filter...")
        layout.addWidget(ed_filter)

        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(listw, 1)

        want = self._perf_norm_name(selected_value)
        current_item = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("value") or "").strip()
            label = str(entry.get("label") or value).strip() or value
            if not value:
                continue
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, str(entry.get("search") or label).strip().lower())
            listw.addItem(item)
            if want and self._perf_norm_name(value) == want:
                current_item = item

        if current_item is None and listw.count() > 0:
            current_item = listw.item(0)
        if current_item is not None:
            listw.setCurrentItem(current_item)

        def _apply_filter() -> None:
            needle = str(ed_filter.text() or "").strip().lower()
            first_visible = None
            for idx in range(listw.count()):
                item = listw.item(idx)
                hay = str(item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or "").strip()
                hidden = bool(needle) and needle not in hay
                item.setHidden(hidden)
                if not hidden and first_visible is None:
                    first_visible = item
            if listw.currentItem() is None or (listw.currentItem() and listw.currentItem().isHidden()):
                if first_visible is not None:
                    listw.setCurrentItem(first_visible)

        ed_filter.textChanged.connect(_apply_filter)
        _apply_filter()

        chosen: dict[str, str | None] = {"value": None}

        def _apply() -> None:
            item = listw.currentItem()
            if item is None or item.isHidden():
                return
            value = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
            if not value:
                return
            chosen["value"] = value
            dlg.accept()

        listw.itemDoubleClicked.connect(lambda *_: _apply())

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        action_row.addWidget(btn_apply)
        action_row.addWidget(btn_cancel)
        layout.addLayout(action_row)

        btn_apply.clicked.connect(_apply)
        btn_cancel.clicked.connect(dlg.reject)
        _fit_widget_to_screen(dlg)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None
        return str(chosen.get("value") or "").strip() or None

    def _open_curve_y_column_popup(self) -> None:
        if not hasattr(self, "cb_y_curve"):
            return
        entries = [
            {"value": self._combo_box_current_value(self.cb_y_curve) if i == self.cb_y_curve.currentIndex() else str(self.cb_y_curve.itemData(i) or self.cb_y_curve.itemText(i) or "").strip(),
             "label": str(self.cb_y_curve.itemText(i) or "").strip(),
             "search": str(self.cb_y_curve.itemText(i) or "").strip().lower()}
            for i in range(self.cb_y_curve.count())
            if str(self.cb_y_curve.itemText(i) or "").strip()
        ]
        chosen = self._show_filter_single_select_popup(
            title="Curve Y Column",
            entries=entries,
            selected_value=self._current_curve_y_name(),
        )
        if chosen is None:
            return
        self._set_combo_to_value(self.cb_y_curve, chosen)
        self._on_curve_y_column_changed()

    def _open_curve_x_column_popup(self) -> None:
        if not hasattr(self, "cb_x"):
            return
        entries = [
            {
                "value": str(self.cb_x.itemData(i) if self.cb_x.itemData(i) is not None else self.cb_x.itemText(i) or "").strip(),
                "label": str(self.cb_x.itemText(i) or "").strip(),
                "search": " ".join(
                    [
                        str(self.cb_x.itemText(i) or "").strip().lower(),
                        str(self.cb_x.itemData(i) or "").strip().lower(),
                    ]
                ).strip(),
            }
            for i in range(self.cb_x.count())
            if str(self.cb_x.itemText(i) or "").strip()
        ]
        chosen = self._show_filter_single_select_popup(
            title="Curve X Column",
            entries=entries,
            selected_value=self._current_curve_x_key() or self._current_curve_x_label(),
        )
        if chosen is None:
            return
        self._set_combo_to_value(self.cb_x, chosen)
        self._on_curve_x_column_changed()

    def _plot_band_enabled_axes(self, mode: str) -> tuple[str, ...]:
        current = str(mode or "").strip().lower()
        if current == "metrics":
            return ("y",)
        if current in {"curves", "performance"}:
            return ("x", "y")
        return tuple()

    @staticmethod
    def _legend_handle_color(handle: object) -> str:
        try:
            from matplotlib import colors as mcolors
        except Exception:
            mcolors = None
        for getter_name in ("get_color", "get_facecolor", "get_edgecolor"):
            getter = getattr(handle, getter_name, None)
            if not callable(getter):
                continue
            try:
                raw_value = getter()
            except Exception:
                continue
            value = raw_value
            if hasattr(value, "tolist"):
                try:
                    value = value.tolist()
                except Exception:
                    pass
            if isinstance(value, (list, tuple)) and value:
                first = value[0]
                if isinstance(first, (list, tuple)) and first:
                    value = tuple(first)
            if mcolors is not None:
                try:
                    return str(mcolors.to_hex(value))
                except Exception:
                    pass
            text = str(value or "").strip()
            if text:
                return text
        return "#64748b"

    def _collect_legend_entries(self, ax) -> list[dict[str, str]]:
        if ax is None or not hasattr(ax, "get_legend_handles_labels"):
            return []
        try:
            handles, labels = ax.get_legend_handles_labels()
        except Exception:
            return []
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for handle, label in zip(handles or [], labels or []):
            label_text = str(label or "").strip()
            if not label_text or label_text.startswith("_"):
                continue
            if label_text in seen:
                continue
            seen.add(label_text)
            out.append({"label": label_text, "color": self._legend_handle_color(handle)})
        return out

    def _apply_interactive_legend_policy(
        self,
        ax,
        *,
        overflow_button: QtWidgets.QPushButton | None = None,
    ) -> list[dict[str, str]]:
        entries = self._collect_legend_entries(ax)
        try:
            legend = ax.get_legend()
        except Exception:
            legend = None
        if legend is not None:
            try:
                legend.remove()
            except Exception:
                pass
        overflow = len(entries) > 8
        if not overflow and entries:
            try:
                ax.legend(fontsize=8, loc="best")
            except Exception:
                pass
        if overflow_button is not None:
            overflow_button.setVisible(bool(entries) and overflow)
            overflow_button.setEnabled(bool(entries) and overflow)
        return entries

    def _open_legend_popup(
        self,
        entries: list[dict[str, str]] | tuple[dict[str, str], ...] | None,
        *,
        title: str,
    ) -> None:
        legend_entries = [dict(entry) for entry in (entries or []) if isinstance(entry, dict)]
        if not legend_entries:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(460, 520)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        ed_filter = QtWidgets.QLineEdit()
        ed_filter.setPlaceholderText("Filter legend entries...")
        layout.addWidget(ed_filter)

        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(listw, 1)

        for entry in legend_entries:
            label_text = str(entry.get("label") or "").strip()
            if not label_text:
                continue
            item = QtWidgets.QListWidgetItem(label_text)
            color_text = str(entry.get("color") or "#64748b").strip() or "#64748b"
            pixmap = QtGui.QPixmap(12, 12)
            pixmap.fill(QtGui.QColor(color_text))
            item.setIcon(QtGui.QIcon(pixmap))
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, label_text.lower())
            listw.addItem(item)

        def _apply_filter() -> None:
            needle = str(ed_filter.text() or "").strip().lower()
            for idx in range(listw.count()):
                item = listw.item(idx)
                hay = str(item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or "").strip()
                item.setHidden(bool(needle) and needle not in hay)

        ed_filter.textChanged.connect(_apply_filter)
        _apply_filter()

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        _fit_widget_to_screen(dlg)
        dlg.exec()

    def _open_main_plot_legend_popup(self) -> None:
        self._open_legend_popup(self._main_plot_legend_entries, title="Plot Legend")

    def _ensure_plot_band_popup(self) -> QtWidgets.QDialog:
        popup = getattr(self, "_plot_band_popup", None)
        if isinstance(popup, QtWidgets.QDialog):
            return popup
        popup = QtWidgets.QDialog(self, QtCore.Qt.WindowType.Popup)
        popup.setWindowTitle("View Bands")
        popup.setModal(False)
        popup.setStyleSheet("QDialog { background: #ffffff; border: 1px solid #e2e8f0; }")
        layout = QtWidgets.QVBoxLayout(popup)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.plot_band_frame)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(popup.hide)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        self._plot_band_popup = popup
        return popup

    def _open_plot_band_popup(self) -> None:
        popup = self._ensure_plot_band_popup()
        self._refresh_plot_view_band_controls()
        popup.adjustSize()
        target = getattr(self, "btn_view_bands", None)
        if target is not None:
            pos = target.mapToGlobal(QtCore.QPoint(0, target.height() + 4))
            popup.move(pos)
        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _sync_main_auto_plot_actions(self) -> None:
        enabled = bool(self._auto_plots) and self._plot_ready and bool(self._db_path)
        if hasattr(self, "btn_view_auto_plots"):
            self.btn_view_auto_plots.setEnabled(enabled)

    def _auto_plot_display_name(self, plot_def: dict) -> str:
        name = str((plot_def or {}).get("name") or "").strip()
        if name:
            return name
        mode = str((plot_def or {}).get("mode") or "").strip().lower()
        selection = self._selection_from_plot_def(plot_def or {})
        run_disp = self._selection_display_text(selection) or str((plot_def or {}).get("run") or "").strip()
        if mode == "curves":
            y = ", ".join([str(x) for x in ((plot_def or {}).get("y") or []) if str(x).strip()])
            x = str((plot_def or {}).get("x") or "").strip()
            return f"Curves: {run_disp} {y} vs {x}".strip()
        if mode == "performance":
            output = str((plot_def or {}).get("output") or "").strip()
            input1 = str((plot_def or {}).get("input1") or "").strip()
            input2 = str((plot_def or {}).get("input2") or "").strip()
            prefix = "Performance"
            if self._perf_normalize_plot_method((plot_def or {}).get("performance_plot_method")) == "cached_condition_means":
                prefix = "Performance (Run Conditions)"
            return (
                f"{prefix}: {output} vs {input1},{input2}".strip()
                if input2
                else f"{prefix}: {output} vs {input1}".strip()
            )
        y = ", ".join([str(x) for x in ((plot_def or {}).get("y") or []) if str(x).strip()])
        stats_val = (plot_def or {}).get("stats")
        stats = (
            [str(x).strip() for x in stats_val if str(x).strip()]
            if isinstance(stats_val, list)
            else []
        )
        if not stats:
            st = str((plot_def or {}).get("stat") or "").strip()
            if st:
                stats = [st]
        stats_label = self._metric_title_suffix(stats) or "metrics"
        return f"Metrics: {run_disp} {stats_label} ({y})".strip()

    def _selected_auto_plot_definitions(
        self,
        *,
        list_widget: QtWidgets.QListWidget | None = None,
    ) -> list[dict]:
        widget = list_widget if list_widget is not None else getattr(self, "list_auto_plots", None)
        if widget is None:
            return []
        out: list[dict] = []
        for item in widget.selectedItems():
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                out.append(dict(data))
        return out

    def _init_plot_area(self, layout: QtWidgets.QVBoxLayout) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            self.lbl_perf_primary_equation = QtWidgets.QLabel("")
            self.lbl_perf_primary_equation.setVisible(False)
            self.lbl_perf_primary_equation.setWordWrap(True)
            self.lbl_perf_primary_equation.setTextFormat(QtCore.Qt.TextFormat.RichText)
            self.lbl_perf_primary_equation.setStyleSheet(
                "QLabel { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; "
                "padding: 8px 10px; color: #78350f; font-size: 11px; font-weight: 700; }"
            )
            layout.addWidget(self.lbl_perf_primary_equation)

            self._figure = Figure(figsize=(8, 4), dpi=100)
            self._axes = self._figure.add_subplot(111)
            self._axes.set_facecolor("#ffffff")
            self._figure.patch.set_facecolor("#ffffff")
            self._canvas = FigureCanvas(self._figure)
            try:
                self._canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
            except Exception:
                pass
            try:
                self._canvas.mpl_connect("scroll_event", self._on_main_plot_scroll)
            except Exception:
                pass
            try:
                self._canvas.mpl_connect("motion_notify_event", self._on_main_plot_motion)
                self._canvas.mpl_connect("button_press_event", self._on_main_plot_press)
                self._canvas.mpl_connect("button_release_event", self._on_main_plot_release)
            except Exception:
                pass
            layout.addWidget(self._canvas)
            self._plot_ready = True
        except Exception as exc:
            self._plot_ready = False
            label = QtWidgets.QLabel(
                "Plotting unavailable. Install matplotlib to enable charts.\n"
                f"Details: {exc}"
            )
            label.setWordWrap(True)
            label.setStyleSheet("color: #b91c1c; font-size: 12px;")
            layout.addWidget(label)

    def _set_plot_note(self, text: str = "") -> None:
        if not hasattr(self, "lbl_plot_note"):
            return
        self._plot_note_base_text = str(text or "").strip()
        self._refresh_plot_note()

    def _displayed_plot_mode(self) -> str:
        raw = str(((self._last_plot_def or {}).get("mode") if isinstance(self._last_plot_def, dict) else "") or "").strip().lower()
        return raw if raw in {"curves", "metrics", "performance"} else str(getattr(self, "_mode", "") or "").strip().lower()

    @staticmethod
    def _view_band_active(bounds: tuple[float | None, float | None]) -> bool:
        return bool(bounds[0] is not None or bounds[1] is not None)

    @staticmethod
    def _parse_view_band_value(raw_value: object, axis_label: str, edge_label: str) -> float | None:
        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except Exception as exc:
            raise ValueError(f"{axis_label} band {edge_label} must be a number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{axis_label} band {edge_label} must be finite.")
        return float(value)

    @classmethod
    def _normalize_view_band(
        cls,
        axis_label: str,
        lower_raw: object,
        upper_raw: object,
    ) -> tuple[float | None, float | None]:
        lower = cls._parse_view_band_value(lower_raw, axis_label, "min")
        upper = cls._parse_view_band_value(upper_raw, axis_label, "max")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{axis_label} band min cannot be greater than max.")
        return lower, upper

    @staticmethod
    def _plot_view_band_axes(mode: str) -> tuple[str, ...]:
        current = str(mode or "").strip().lower()
        if current in {"curves", "metrics", "performance"}:
            return ("x", "y")
        return tuple()

    def _plot_view_band_mode_for_display(self) -> str:
        display_mode = self._displayed_plot_mode()
        active_mode = str(getattr(self, "_mode", "") or "").strip().lower()
        if display_mode != active_mode:
            return ""
        return display_mode if display_mode in {"curves", "metrics", "performance"} else ""

    def _current_plot_view_bands(self, *, mode: str | None = None) -> dict[str, tuple[float | None, float | None]]:
        use_mode = str(mode or self._plot_view_band_mode_for_display() or self._mode or "").strip().lower()
        allowed_axes = set(self._plot_view_band_axes(use_mode))
        bands: dict[str, tuple[float | None, float | None]] = {}
        if "x" in allowed_axes:
            x_min = getattr(getattr(self, "ed_plot_x_band_min", None), "text", lambda: "")()
            x_max = getattr(getattr(self, "ed_plot_x_band_max", None), "text", lambda: "")()
            bands["x"] = self._normalize_view_band("X", x_min, x_max)
        if "y" in allowed_axes:
            y_min = getattr(getattr(self, "ed_plot_y_band_min", None), "text", lambda: "")()
            y_max = getattr(getattr(self, "ed_plot_y_band_max", None), "text", lambda: "")()
            bands["y"] = self._normalize_view_band("Y", y_min, y_max)
        return bands

    def _current_enabled_plot_view_bands(self, *, mode: str | None = None) -> dict[str, tuple[float | None, float | None]]:
        use_mode = str(mode or self._plot_view_band_mode_for_display() or self._mode or "").strip().lower()
        enabled_axes = set(self._plot_band_enabled_axes(use_mode))
        bands: dict[str, tuple[float | None, float | None]] = {}
        if "x" in enabled_axes:
            x_min = getattr(getattr(self, "ed_plot_x_band_min", None), "text", lambda: "")()
            x_max = getattr(getattr(self, "ed_plot_x_band_max", None), "text", lambda: "")()
            bands["x"] = self._normalize_view_band("X", x_min, x_max)
        if "y" in enabled_axes:
            y_min = getattr(getattr(self, "ed_plot_y_band_min", None), "text", lambda: "")()
            y_max = getattr(getattr(self, "ed_plot_y_band_max", None), "text", lambda: "")()
            bands["y"] = self._normalize_view_band("Y", y_min, y_max)
        return bands

    @staticmethod
    def _plot_view_band_note(
        axis_band_filters: dict[str, tuple[float | None, float | None]] | None,
    ) -> str:
        filters = axis_band_filters or {}
        parts: list[str] = []
        for axis in ("x", "y"):
            lower, upper = filters.get(axis, (None, None))
            if lower is None and upper is None:
                continue
            lo_txt = "-inf" if lower is None else f"{float(lower):.6g}"
            hi_txt = "inf" if upper is None else f"{float(upper):.6g}"
            parts.append(f"{axis.upper()} [{lo_txt}, {hi_txt}]")
        if not parts:
            return ""
        return "View band active: " + " | ".join(parts)

    def _refresh_plot_note(self) -> None:
        if not hasattr(self, "lbl_plot_note"):
            return
        parts: list[str] = []
        base_note = str(getattr(self, "_plot_note_base_text", "") or "").strip()
        if base_note:
            parts.append(base_note)
        mode = self._plot_view_band_mode_for_display()
        if mode:
            try:
                view_note = self._plot_view_band_note(self._current_enabled_plot_view_bands(mode=mode))
            except ValueError:
                view_note = ""
            if view_note:
                parts.append(view_note)
        note = "\n".join(parts)
        self.lbl_plot_note.setText(note)
        self.lbl_plot_note.setVisible(bool(note))

    def _refresh_plot_view_band_controls(self) -> None:
        active_mode = str(getattr(self, "_mode", "") or "").strip().lower()
        axes = set(self._plot_view_band_axes(active_mode))
        enabled_axes = set(self._plot_band_enabled_axes(active_mode))
        if hasattr(self, "btn_view_bands"):
            self.btn_view_bands.setVisible(bool(axes))
            self.btn_view_bands.setEnabled(bool(getattr(self, "_plot_ready", False)) and bool(axes))
        for attr, visible in (("plot_band_x_row", "x" in axes), ("plot_band_y_row", "y" in axes)):
            row = getattr(self, attr, None)
            if row is None:
                continue
            for i in range(row.count()):
                item = row.itemAt(i)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.setVisible(visible)
                    axis = "x" if attr == "plot_band_x_row" else "y"
                    widget.setEnabled(axis in enabled_axes)
        self._refresh_plot_note()

    def _apply_plot_view_bands_to_axes(self, ax, *, mode: str | None = None) -> None:
        if ax is None:
            return
        use_mode = str(mode or self._plot_view_band_mode_for_display() or self._mode or "").strip().lower()
        axes = set(self._plot_band_enabled_axes(use_mode))
        if not axes:
            return
        bands = self._current_enabled_plot_view_bands(mode=use_mode)
        if "x" in axes:
            x_bounds = bands.get("x", (None, None))
            if self._view_band_active(x_bounds):
                current_lo, current_hi = ax.get_xlim()
                ax.set_xlim(x_bounds[0] if x_bounds[0] is not None else current_lo, x_bounds[1] if x_bounds[1] is not None else current_hi)
        if "y" in axes:
            y_bounds = bands.get("y", (None, None))
            if self._view_band_active(y_bounds):
                current_lo, current_hi = ax.get_ylim()
                ax.set_ylim(y_bounds[0] if y_bounds[0] is not None else current_lo, y_bounds[1] if y_bounds[1] is not None else current_hi)

    def _apply_current_plot_view_bands(self) -> None:
        self._refresh_plot_note()
        mode = self._plot_view_band_mode_for_display()
        if not mode or not self._axes or not self._canvas:
            return
        try:
            self._apply_plot_view_bands_to_axes(self._axes, mode=mode)
        except ValueError:
            return
        try:
            self._canvas.draw_idle()
        except Exception:
            try:
                self._canvas.draw()
            except Exception:
                pass

    def _finalize_plot_view_bands(self) -> None:
        mode = self._plot_view_band_mode_for_display() or str(getattr(self, "_mode", "") or "").strip().lower()
        if not self._plot_view_band_axes(mode):
            self._refresh_plot_note()
            return
        try:
            self._current_enabled_plot_view_bands(mode=mode)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Plot View Bands", str(exc))
            return
        self._apply_current_plot_view_bands()

    def _ensure_main_axes(self, plot_dimension: str = "2d") -> None:
        if not getattr(self, "_figure", None):
            return
        want_3d = str(plot_dimension or "").strip().lower() == "3d"
        current_name = str(getattr(getattr(self, "_axes", None), "name", "") or "").strip().lower()
        if want_3d and current_name == "3d":
            return
        if (not want_3d) and current_name and current_name != "3d":
            return
        try:
            self._figure.clear()
            if want_3d:
                self._axes = self._figure.add_subplot(111, projection="3d")
            else:
                self._axes = self._figure.add_subplot(111)
            self._axes.set_facecolor("#ffffff")
            self._figure.patch.set_facecolor("#ffffff")
        except Exception:
            return

    @staticmethod
    def _apply_axes_zoom(ax, factor: float, *, center: tuple[float, float] | None = None, axis: str = "both") -> None:
        if ax is None:
            return
        if str(getattr(ax, "name", "") or "").strip().lower() == "3d":
            return
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
        except Exception:
            return

        def _zoom_1d(lo: float, hi: float, c: float) -> tuple[float, float]:
            span = float(hi) - float(lo)
            if span == 0:
                return lo, hi
            direction = 1.0 if span >= 0 else -1.0
            span_abs = abs(span) * float(factor)
            new_lo = float(c) - (span_abs / 2.0) * direction
            new_hi = float(c) + (span_abs / 2.0) * direction
            return new_lo, new_hi

        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        if center and all(isinstance(v, (int, float)) for v in center):
            cx, cy = center
            if isinstance(cx, (int, float)):
                xc = float(cx)
            if isinstance(cy, (int, float)):
                yc = float(cy)

        want = str(axis or "both").strip().lower()
        try:
            if want in {"x", "both"}:
                ax.set_xlim(*_zoom_1d(x0, x1, xc))
            if want in {"y", "both"}:
                ax.set_ylim(*_zoom_1d(y0, y1, yc))
        except Exception:
            return

    def _capture_main_plot_base_view(self) -> None:
        if not self._axes:
            return
        if str(getattr(self._axes, "name", "") or "").strip().lower() == "3d":
            self._plot_base_xlim = None
            self._plot_base_ylim = None
            self._update_plot_zoom_actions()
            return
        try:
            self._plot_base_xlim = tuple(float(v) for v in self._axes.get_xlim())  # type: ignore[assignment]
            self._plot_base_ylim = tuple(float(v) for v in self._axes.get_ylim())  # type: ignore[assignment]
        except Exception:
            self._plot_base_xlim = None
            self._plot_base_ylim = None
        self._update_plot_zoom_actions()

    def _update_plot_zoom_actions(self) -> None:
        is_3d = str(getattr(getattr(self, "_axes", None), "name", "") or "").strip().lower() == "3d"
        enabled = bool(self._plot_ready and self._axes and self._canvas and self._last_plot_def and not is_3d)
        for b in ("btn_zone_zoom", "btn_zoom_out", "btn_zoom_in", "btn_zoom_reset", "btn_expand_plot"):
            if hasattr(self, b):
                try:
                    getattr(self, b).setEnabled(enabled)
                except Exception:
                    pass
        if hasattr(self, "btn_view_bands"):
            try:
                self.btn_view_bands.setEnabled(bool(self._plot_ready and self._plot_view_band_axes(self._mode)))
            except Exception:
                pass
        self._sync_main_auto_plot_actions()

    def _zoom_main_plot(self, factor: float) -> None:
        if not self._axes or not self._canvas:
            return
        if str(getattr(self._axes, "name", "") or "").strip().lower() == "3d":
            return
        center = self._plot_last_cursor_xy
        self._apply_axes_zoom(self._axes, float(factor), center=center, axis="both")
        try:
            self._canvas.draw_idle()
        except Exception:
            try:
                self._canvas.draw()
            except Exception:
                pass

    def _reset_main_plot_zoom(self) -> None:
        if not self._axes or not self._canvas:
            return
        if str(getattr(self._axes, "name", "") or "").strip().lower() == "3d":
            return
        if self._plot_base_xlim and self._plot_base_ylim:
            try:
                self._axes.set_xlim(*self._plot_base_xlim)
                self._axes.set_ylim(*self._plot_base_ylim)
            except Exception:
                pass
        else:
            try:
                self._axes.relim()
                self._axes.autoscale_view()
            except Exception:
                pass
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_main_plot_scroll(self, event) -> None:
        if not self._axes or not self._canvas:
            return
        if str(getattr(self._axes, "name", "") or "").strip().lower() == "3d":
            return
        if getattr(event, "inaxes", None) is not self._axes:
            return
        direction = str(getattr(event, "button", "") or "").strip().lower()
        if direction not in {"up", "down"}:
            return
        factor = 0.8 if direction == "up" else 1.25

        key = str(getattr(event, "key", "") or "").lower()
        axis = "both"
        if "shift" in key:
            axis = "x"
        elif "control" in key or "ctrl" in key:
            axis = "y"

        self._apply_axes_zoom(
            self._axes,
            factor,
            center=(getattr(event, "xdata", None), getattr(event, "ydata", None)),
            axis=axis,
        )
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_main_plot_motion(self, event) -> None:
        if not self._axes or not self._canvas:
            return
        if getattr(event, "inaxes", None) is self._axes:
            xdata = getattr(event, "xdata", None)
            ydata = getattr(event, "ydata", None)
            if isinstance(xdata, (int, float)) and isinstance(ydata, (int, float)):
                self._plot_last_cursor_xy = (float(xdata), float(ydata))

        if not (hasattr(self, "btn_zone_zoom") and self.btn_zone_zoom.isChecked()):
            return
        if not self._zone_zoom_press_xy or self._zone_zoom_rect is None:
            return

        x0, y0 = self._zone_zoom_press_xy
        x1 = getattr(event, "xdata", None)
        y1 = getattr(event, "ydata", None)
        if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
            return
        try:
            self._zone_zoom_rect.set_bounds(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        except Exception:
            return
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_main_plot_press(self, event) -> None:
        if not self._axes or not self._canvas:
            return
        if not (hasattr(self, "btn_zone_zoom") and self.btn_zone_zoom.isChecked()):
            return
        if getattr(event, "inaxes", None) is not self._axes:
            return
        if int(getattr(event, "button", 0) or 0) != 1:
            return
        x0 = getattr(event, "xdata", None)
        y0 = getattr(event, "ydata", None)
        if not isinstance(x0, (int, float)) or not isinstance(y0, (int, float)):
            return
        self._zone_zoom_press_xy = (float(x0), float(y0))
        try:
            from matplotlib.patches import Rectangle

            if self._zone_zoom_rect is not None:
                try:
                    self._zone_zoom_rect.remove()
                except Exception:
                    pass
            rect = Rectangle(
                (float(x0), float(y0)),
                0.0,
                0.0,
                fill=False,
                linewidth=1.2,
                linestyle="--",
                edgecolor="#0f766e",
                alpha=0.9,
            )
            self._axes.add_patch(rect)
            self._zone_zoom_rect = rect
        except Exception:
            self._zone_zoom_rect = None
            return
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_main_plot_release(self, event) -> None:
        if not self._axes or not self._canvas:
            return
        if not self._zone_zoom_press_xy or self._zone_zoom_rect is None:
            return

        x0, y0 = self._zone_zoom_press_xy
        x1 = getattr(event, "xdata", None)
        y1 = getattr(event, "ydata", None)
        if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
            # If released outside axes, cancel the selection.
            try:
                self._zone_zoom_rect.remove()
            except Exception:
                pass
            self._zone_zoom_press_xy = None
            self._zone_zoom_rect = None
            try:
                self._canvas.draw_idle()
            except Exception:
                pass
            return

        x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
        try:
            self._zone_zoom_rect.remove()
        except Exception:
            pass
        self._zone_zoom_press_xy = None
        self._zone_zoom_rect = None

        # Ignore tiny drags (treat as a click; keep cursor center for Zoom +/- buttons).
        if abs(x1f - x0f) < 1e-9 or abs(y1f - y0f) < 1e-9:
            self._plot_last_cursor_xy = (x1f, y1f)
            return

        lo_x, hi_x = (x0f, x1f) if x0f <= x1f else (x1f, x0f)
        lo_y, hi_y = (y0f, y1f) if y0f <= y1f else (y1f, y0f)
        try:
            self._axes.set_xlim(lo_x, hi_x)
            self._axes.set_ylim(lo_y, hi_y)
        except Exception:
            return
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _open_main_plot_popup(self) -> None:
        if not self._plot_ready or not self._db_path or not self._last_plot_def:
            return
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Plot", f"Plotting unavailable: {exc}")
            return

        fig = self._render_plot_def_to_figure(dict(self._last_plot_def))
        ax = fig.axes[0] if getattr(fig, "axes", None) else None
        if ax is None:
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Plot (Zoom)")
        dlg.resize(1180, 820)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        btn_mag = QtWidgets.QPushButton("Magnify")
        btn_mag.setCheckable(True)
        btn_out = QtWidgets.QPushButton("Zoom -")
        btn_in = QtWidgets.QPushButton("Zoom +")
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_legend = QtWidgets.QPushButton("Legend...")
        btn_legend.setVisible(False)
        btn_legend.setEnabled(False)
        for b in (btn_mag, btn_out, btn_in, btn_reset, btn_legend):
            b.setStyleSheet(
                """
                QPushButton {
                    padding: 6px 10px;
                    border-radius: 8px;
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    font-size: 12px;
                    font-weight: 800;
                    color: #0f172a;
                }
                QPushButton:hover { background: #f8fafc; }
                """
            )
        btn_mag.setToolTip("Magnify zones: click-drag a rectangle on the plot to zoom to that area")
        hint = QtWidgets.QLabel("Tip: mouse wheel zoom (Shift = X only, Ctrl = Y only)")
        hint.setStyleSheet("color: #64748b; font-size: 11px;")
        top.addWidget(btn_mag)
        top.addWidget(btn_out)
        top.addWidget(btn_in)
        top.addWidget(btn_reset)
        top.addWidget(btn_legend)
        top.addSpacing(10)
        top.addWidget(hint)
        top.addStretch(1)
        layout.addLayout(top)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas, 1)

        plot_mode = str((self._last_plot_def or {}).get("mode") or "").strip().lower()
        if plot_mode in {"curves", "metrics", "performance"}:
            try:
                self._apply_plot_view_bands_to_axes(ax, mode=plot_mode)
            except ValueError:
                pass
        popup_legend_entries = self._apply_interactive_legend_policy(ax, overflow_button=btn_legend)
        btn_legend.clicked.connect(lambda: self._open_legend_popup(popup_legend_entries, title="Plot Legend"))
        base_xlim = ax.get_xlim()
        base_ylim = ax.get_ylim()

        def _reset():
            try:
                ax.set_xlim(*base_xlim)
                ax.set_ylim(*base_ylim)
            except Exception:
                pass
            try:
                canvas.draw_idle()
            except Exception:
                pass

        def _zoom(factor: float):
            self._apply_axes_zoom(ax, factor, axis="both")
            try:
                canvas.draw_idle()
            except Exception:
                pass

        btn_out.clicked.connect(lambda: _zoom(1.25))
        btn_in.clicked.connect(lambda: _zoom(0.8))
        btn_reset.clicked.connect(_reset)

        last_xy: tuple[float, float] | None = None
        press_xy: tuple[float, float] | None = None
        rect_patch = None

        def _on_scroll(event):
            if getattr(event, "inaxes", None) is not ax:
                return
            direction = str(getattr(event, "button", "") or "").strip().lower()
            if direction not in {"up", "down"}:
                return
            factor = 0.8 if direction == "up" else 1.25
            key = str(getattr(event, "key", "") or "").lower()
            axis = "both"
            if "shift" in key:
                axis = "x"
            elif "control" in key or "ctrl" in key:
                axis = "y"
            self._apply_axes_zoom(
                ax,
                factor,
                center=(getattr(event, "xdata", None), getattr(event, "ydata", None)),
                axis=axis,
            )
            try:
                canvas.draw_idle()
            except Exception:
                pass

        def _on_motion(event):
            nonlocal last_xy, press_xy, rect_patch
            if getattr(event, "inaxes", None) is ax:
                xdata = getattr(event, "xdata", None)
                ydata = getattr(event, "ydata", None)
                if isinstance(xdata, (int, float)) and isinstance(ydata, (int, float)):
                    last_xy = (float(xdata), float(ydata))
            if not btn_mag.isChecked():
                return
            if press_xy is None or rect_patch is None:
                return
            x1 = getattr(event, "xdata", None)
            y1 = getattr(event, "ydata", None)
            if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
                return
            x0, y0 = press_xy
            try:
                rect_patch.set_bounds(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            except Exception:
                return
            try:
                canvas.draw_idle()
            except Exception:
                pass

        def _on_press(event):
            nonlocal press_xy, rect_patch
            if not btn_mag.isChecked():
                return
            if getattr(event, "inaxes", None) is not ax:
                return
            if int(getattr(event, "button", 0) or 0) != 1:
                return
            x0 = getattr(event, "xdata", None)
            y0 = getattr(event, "ydata", None)
            if not isinstance(x0, (int, float)) or not isinstance(y0, (int, float)):
                return
            press_xy = (float(x0), float(y0))
            try:
                from matplotlib.patches import Rectangle

                if rect_patch is not None:
                    try:
                        rect_patch.remove()
                    except Exception:
                        pass
                rect_patch = Rectangle(
                    (float(x0), float(y0)),
                    0.0,
                    0.0,
                    fill=False,
                    linewidth=1.2,
                    linestyle="--",
                    edgecolor="#0f766e",
                    alpha=0.9,
                )
                ax.add_patch(rect_patch)
            except Exception:
                rect_patch = None
                press_xy = None
                return
            try:
                canvas.draw_idle()
            except Exception:
                pass

        def _on_release(event):
            nonlocal press_xy, rect_patch
            if press_xy is None or rect_patch is None:
                return
            x1 = getattr(event, "xdata", None)
            y1 = getattr(event, "ydata", None)
            try:
                rect_patch.remove()
            except Exception:
                pass
            rect_patch = None
            x0, y0 = press_xy
            press_xy = None

            if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
                return
            x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
            if abs(x1f - x0f) < 1e-9 or abs(y1f - y0f) < 1e-9:
                return
            lo_x, hi_x = (x0f, x1f) if x0f <= x1f else (x1f, x0f)
            lo_y, hi_y = (y0f, y1f) if y0f <= y1f else (y1f, y0f)
            try:
                ax.set_xlim(lo_x, hi_x)
                ax.set_ylim(lo_y, hi_y)
            except Exception:
                return
            try:
                canvas.draw_idle()
            except Exception:
                pass

        try:
            canvas.mpl_connect("scroll_event", _on_scroll)
            canvas.mpl_connect("motion_notify_event", _on_motion)
            canvas.mpl_connect("button_press_event", _on_press)
            canvas.mpl_connect("button_release_event", _on_release)
        except Exception:
            pass

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        _fit_widget_to_screen(dlg)
        dlg.exec()

    def _load_cache(self, *, rebuild: bool) -> None:
        if not hasattr(self, "_report_progress"):
            try:
                if rebuild:
                    repo = be.resolve_test_data_project_global_repo(self._project_dir, self._workbook_path)
                    payload = be.update_test_data_trending_project_workbook(
                        repo,
                        self._workbook_path,
                        overwrite=False,
                    )
                    self._db_path = Path(str(payload.get("db_path") or "")).expanduser()
                else:
                    self._db_path = be.validate_test_data_project_cache_for_open(
                        self._project_dir,
                        self._workbook_path,
                    )
                self.lbl_source.setText(str(self._db_path))
                self.lbl_cache.setText(f"Cache DB: {self._db_path}")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Test Data Cache", str(exc))
                return
            self._refresh_from_cache()
            self._update_plot_zoom_actions()
            return
        if self._cache_worker is not None and self._cache_worker.isRunning():
            return
        self._set_cache_controls_enabled(False)
        self._cache_progress_visible = False
        self._cache_progress_heading = "Rebuilding Test Data Cache" if rebuild else "Loading Test Data Cache"
        self._cache_progress_status = "Rebuilding raw cache" if rebuild else "Validating existing project cache"
        self._cache_progress_detail = ""
        self.lbl_cache.setText("Cache DB: loading...")
        self._cache_progress_timer.start(150)

        def _task(report):
            report(self._cache_progress_status)
            if rebuild:
                repo = be.resolve_test_data_project_global_repo(self._project_dir, self._workbook_path)
                payload = be.update_test_data_trending_project_workbook(
                    repo,
                    self._workbook_path,
                    overwrite=False,
                    progress_cb=report,
                )
                return Path(str(payload.get("db_path") or "")).expanduser()
            return be.validate_test_data_project_cache_for_open(
                self._project_dir,
                self._workbook_path,
            )

        self._cache_worker = ProjectTaskWorker(_task, parent=self)
        self._cache_worker.progress.connect(self._on_cache_task_progress)
        self._cache_worker.completed.connect(self._on_cache_task_done)
        self._cache_worker.failed.connect(self._on_cache_task_error)
        self._cache_worker.start()

    def _set_cache_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            getattr(self, "btn_refresh_cache", None),
            getattr(self, "btn_open_support", None),
            getattr(self, "btn_export_debug_excels", None),
            getattr(self, "btn_plot", None),
        ):
            try:
                if widget is not None:
                    widget.setEnabled(bool(enabled))
            except Exception:
                pass

    def _show_cache_progress_dialog(self) -> None:
        worker = self._cache_worker
        if worker is None or not worker.isRunning() or self._cache_progress_visible:
            return
        self._cache_progress_visible = True
        try:
            self._report_progress.lbl_heading.setText(self._cache_progress_heading)
            self._report_progress.begin(self._cache_progress_status)
            if self._cache_progress_detail:
                self._report_progress.detail_label.setText(self._cache_progress_detail)
        except Exception:
            self._cache_progress_visible = False

    def _on_cache_task_progress(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        self._cache_progress_detail = msg
        if not self._cache_progress_visible:
            self._show_cache_progress_dialog()
        if self._cache_progress_visible:
            try:
                self._report_progress.detail_label.setText(msg)
            except Exception:
                pass

    def _on_cache_task_done(self, payload: object) -> None:
        self._cache_progress_timer.stop()
        self._cache_worker = None
        self._set_cache_controls_enabled(True)
        db_path = Path(str(payload)).expanduser() if payload is not None else None
        if db_path is None:
            QtWidgets.QMessageBox.warning(self, "Test Data Cache", "Cache build returned no database path.")
            return
        self._db_path = db_path
        self.lbl_source.setText(str(self._db_path))
        self.lbl_cache.setText(f"Cache DB: {self._db_path}")
        if self._cache_progress_visible:
            try:
                self._report_progress.finish("Cache ready", success=True)
            except Exception:
                pass
            self._cache_progress_visible = False
        self._refresh_from_cache()
        self._update_plot_zoom_actions()

    def _on_cache_task_error(self, message: str) -> None:
        self._cache_progress_timer.stop()
        self._cache_worker = None
        self._set_cache_controls_enabled(True)
        self.lbl_cache.setText("Cache DB: unavailable")
        if self._cache_progress_visible:
            try:
                self._report_progress.finish(f"Cache failed: {message}", success=False)
            except Exception:
                pass
            self._cache_progress_visible = False
        try:
            QtWidgets.QMessageBox.warning(self, "Test Data Cache", str(message))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test Data Cache", str(exc))

    def _open_support_workbook(self) -> None:
        try:
            path = be.td_support_workbook_path_for(self._workbook_path, project_dir=self._project_dir)
            if not path.exists():
                raise RuntimeError(f"Support workbook not found: {path}")
            be.open_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Support Workbook", str(exc))

    def _generate_debug_excel_files(self) -> None:
        try:
            generated = be.export_test_data_project_debug_excels(
                self._project_dir,
                self._workbook_path,
                force=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Debug Excel Files", str(exc))
            return

        if not generated:
            QtWidgets.QMessageBox.information(
                self,
                "Debug Excel Files",
                "No debug Excel files were generated from the current cache.",
            )
            return

        ordered_keys = (
            "implementation_excel",
            "raw_cache_excel",
            "raw_points_excel",
        )
        lines = [str(generated[key]) for key in ordered_keys if key in generated]
        QtWidgets.QMessageBox.information(
            self,
            "Debug Excel Files",
            "Generated debug Excel files:\n\n" + "\n".join(lines),
        )

    def _refresh_from_cache(self) -> None:
        if not self._db_path:
            return
        try:
            runs_ex = be.td_list_runs_ex(self._db_path)
            serials = be.td_list_serials(self._db_path)
            run_selection_views = be.td_list_run_selection_views(
                self._db_path,
                self._workbook_path,
                project_dir=self._project_dir,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test Data Cache", str(exc))
            return

        prev_ref = self.cb_run.currentData()
        prev_id = str(prev_ref.get("id") or "").strip() if isinstance(prev_ref, dict) else ""
        if not prev_id and prev_ref is None:
            prev_ref = self.cb_run.currentText()
            prev_id = str(prev_ref or "").strip()

        self._runs_ex = [d for d in (runs_ex or []) if isinstance(d, dict) and str(d.get("run_name") or "").strip()]
        self._run_selection_views = {
            "sequence": [dict(d) for d in ((run_selection_views or {}).get("sequence") or []) if isinstance(d, dict)],
            "condition": [dict(d) for d in ((run_selection_views or {}).get("condition") or []) if isinstance(d, dict)],
        }
        self._run_display_by_name = {
            str(d.get("run_name") or "").strip(): str(d.get("display_name") or "").strip()
            for d in self._runs_ex
        }
        # Only keep unique display->run mappings (for legacy auto-plots that stored display text).
        counts: dict[str, int] = {}
        for rn, dn in self._run_display_by_name.items():
            if dn:
                counts[dn] = int(counts.get(dn, 0)) + 1
        self._run_name_by_display = {
            dn: rn for rn, dn in self._run_display_by_name.items() if dn and int(counts.get(dn, 0)) == 1
        }

        try:
            source_rows = be.td_read_sources_metadata_from_cache(self._db_path)
        except Exception:
            source_rows = []
        try:
            filter_rows = be.td_read_observation_filter_rows_from_cache(self._db_path)
        except Exception:
            filter_rows = []
        by_sn = _td_serial_metadata_by_serial(source_rows)
        self._serial_source_rows = [by_sn.get(sn, {"serial": sn, "serial_number": sn}) for sn in serials if str(sn).strip()]
        self._serial_source_by_serial = _td_serial_metadata_by_serial(self._serial_source_rows)
        self._global_filter_rows = [dict(row) for row in (filter_rows or []) if isinstance(row, dict)]
        self._refresh_global_filter_options()

        self._sync_run_mode_availability()
        prev_run = self._run_name_by_display.get(prev_id, prev_id)
        prev_selection_id = prev_id or prev_run
        if prev_selection_id and ":" not in prev_selection_id:
            prev_selection_id = f"sequence:{prev_run or prev_selection_id}"
        self._refresh_run_dropdown(prev_selection_id=prev_selection_id)

        keep = [sn for sn in (self._highlight_sns or []) if sn in set(self._active_serials())]
        self._set_highlight_serials(keep)

        self._refresh_columns_for_run()
        self._refresh_stats_preview()
        self._refresh_performance_ui()
        self._update_plot_zoom_actions()

    def _refresh_global_filter_options(self) -> None:
        serial_rows = [dict(row) for row in (self._serial_source_rows or []) if isinstance(row, dict)]
        serial_rows = [row for row in serial_rows if _td_serial_value(row)]
        serial_rows.sort(key=lambda row: _td_serial_value(row).casefold())

        filter_rows = [dict(row) for row in (self._global_filter_rows or []) if isinstance(row, dict) and _td_serial_value(row)]
        program_source_rows = filter_rows or serial_rows
        program_values = sorted(
            {_td_display_program_title(row.get("program_title")) for row in program_source_rows},
            key=lambda value: (1 if value == TD_UNKNOWN_PROGRAM_LABEL else 0, value.casefold()),
        )
        prev_programs = set(self._available_program_filters or [])
        prev_checked_programs = set(self._checked_program_filters or [])
        prev_serials = {_td_serial_value(row) for row in (self._available_serial_filter_rows or []) if _td_serial_value(row)}
        prev_checked_serials = set(self._checked_serial_filters or [])
        prev_control_periods = set(self._available_control_period_filters or [])
        prev_checked_control_periods = set(self._checked_control_period_filters or [])
        prev_suppression = set(self._available_suppression_voltage_filters or [])
        prev_checked_suppression = set(self._checked_suppression_voltage_filters or [])

        if not prev_programs:
            self._checked_program_filters = list(program_values)
        else:
            self._checked_program_filters = [
                value
                for value in program_values
                if value in prev_checked_programs or value not in prev_programs
            ]

        serial_values = [_td_serial_value(row) for row in serial_rows]
        if not prev_serials:
            self._checked_serial_filters = list(serial_values)
        else:
            self._checked_serial_filters = [
                serial
                for serial in serial_values
                if serial in prev_checked_serials or serial not in prev_serials
            ]

        control_period_values = sorted(
            {_td_control_period_filter_value(row) for row in filter_rows if _td_control_period_filter_value(row)},
            key=_td_compact_filter_sort_key,
        )
        if not prev_control_periods:
            self._checked_control_period_filters = list(control_period_values)
        else:
            self._checked_control_period_filters = [
                value
                for value in control_period_values
                if value in prev_checked_control_periods or value not in prev_control_periods
            ]

        suppression_values = sorted(
            {_td_suppression_voltage_filter_value(row) for row in filter_rows if _td_suppression_voltage_filter_value(row)},
            key=_td_compact_filter_sort_key,
        )
        if not prev_suppression:
            self._checked_suppression_voltage_filters = list(suppression_values)
        else:
            self._checked_suppression_voltage_filters = [
                value
                for value in suppression_values
                if value in prev_checked_suppression or value not in prev_suppression
            ]

        self._available_program_filters = list(program_values)
        self._available_serial_filter_rows = list(serial_rows)
        self._available_control_period_filters = list(control_period_values)
        self._available_suppression_voltage_filters = list(suppression_values)
        self._refresh_global_filter_summaries()

    def _refresh_global_filter_summaries(self) -> None:
        total_programs = len(self._available_program_filters or [])
        active_programs = self._active_program_filter_values()
        total_serials = len(self._available_serial_filter_rows or [])
        active_serial_rows = self._active_serial_rows()
        active_serials = [_td_serial_value(row) for row in active_serial_rows]
        total_control_periods = len(self._available_control_period_filters or [])
        active_control_periods = self._active_control_period_filter_values()
        total_suppression = len(self._available_suppression_voltage_filters or [])
        active_suppression = self._active_suppression_voltage_filter_values()

        if hasattr(self, "lbl_program_filter_summary"):
            if total_programs <= 0:
                program_text = "Programs: -"
            elif len(active_programs) >= total_programs:
                program_text = f"Programs: All ({total_programs})"
            elif not active_programs:
                program_text = f"Programs: None active (0/{total_programs})"
            elif len(active_programs) <= 2:
                program_text = "Programs: " + ", ".join(active_programs)
            else:
                program_text = f"Programs: {len(active_programs)} of {total_programs} active"
            self.lbl_program_filter_summary.setText(program_text)
            self.lbl_program_filter_summary.setToolTip(", ".join(active_programs))

        if hasattr(self, "lbl_serial_filter_summary"):
            if total_serials <= 0:
                serial_text = "Serials: -"
            elif len(active_serials) >= total_serials and len(self._checked_serial_filters or []) >= total_serials:
                serial_text = f"Serials: All ({total_serials})"
            elif not active_serials:
                serial_text = f"Serials: None active (0/{total_serials})"
            else:
                serial_text = f"Serials: {len(active_serials)} of {total_serials} active"
            self.lbl_serial_filter_summary.setText(serial_text)
            shown = ", ".join(active_serials[:20])
            if len(active_serials) > 20:
                shown += f" (+{len(active_serials) - 20} more)"
            self.lbl_serial_filter_summary.setToolTip(shown)

        if hasattr(self, "lbl_control_period_filter_summary"):
            if total_control_periods <= 0:
                control_text = "Control Period: -"
            elif len(active_control_periods) >= total_control_periods:
                control_text = f"Control Period: All ({total_control_periods})"
            elif not active_control_periods:
                control_text = f"Control Period: None active (0/{total_control_periods})"
            elif len(active_control_periods) <= 3:
                control_text = "Control Period: " + ", ".join(active_control_periods)
            else:
                control_text = f"Control Period: {len(active_control_periods)} of {total_control_periods} active"
            self.lbl_control_period_filter_summary.setText(control_text)
            self.lbl_control_period_filter_summary.setToolTip(", ".join(active_control_periods))

        if hasattr(self, "lbl_suppression_voltage_filter_summary"):
            if total_suppression <= 0:
                suppression_text = "Suppression Voltage: -"
            elif len(active_suppression) >= total_suppression:
                suppression_text = f"Suppression Voltage: All ({total_suppression})"
            elif not active_suppression:
                suppression_text = f"Suppression Voltage: None active (0/{total_suppression})"
            elif len(active_suppression) <= 3:
                suppression_text = "Suppression Voltage: " + ", ".join(active_suppression)
            else:
                suppression_text = f"Suppression Voltage: {len(active_suppression)} of {total_suppression} active"
            self.lbl_suppression_voltage_filter_summary.setText(suppression_text)
            self.lbl_suppression_voltage_filter_summary.setToolTip(", ".join(active_suppression))

        has_programs = bool(self._available_program_filters)
        has_serials = bool(self._available_serial_filter_rows)
        has_control_periods = bool(self._available_control_period_filters)
        has_suppression = bool(self._available_suppression_voltage_filters)
        if hasattr(self, "btn_program_filters"):
            self.btn_program_filters.setEnabled(has_programs)
        if hasattr(self, "btn_serial_filters"):
            self.btn_serial_filters.setEnabled(has_serials)
        if hasattr(self, "btn_control_period_filters"):
            self.btn_control_period_filters.setEnabled(has_control_periods)
        if hasattr(self, "btn_suppression_voltage_filters"):
            self.btn_suppression_voltage_filters.setEnabled(has_suppression)
        if hasattr(self, "btn_reset_global_filters"):
            self.btn_reset_global_filters.setEnabled(has_programs or has_serials or has_control_periods or has_suppression)

    def _active_program_filter_values(self) -> list[str]:
        selected = [value for value in (self._checked_program_filters or []) if str(value).strip()]
        valid = {value for value in (self._available_program_filters or []) if str(value).strip()}
        return [value for value in selected if value in valid]

    def _active_suppression_voltage_filter_values(self) -> list[str]:
        selected = [value for value in (self._checked_suppression_voltage_filters or []) if str(value).strip()]
        valid = {value for value in (self._available_suppression_voltage_filters or []) if str(value).strip()}
        return [value for value in selected if value in valid]

    def _active_control_period_filter_values(self) -> list[str]:
        selected = [value for value in (self._checked_control_period_filters or []) if str(value).strip()]
        valid = {value for value in (self._available_control_period_filters or []) if str(value).strip()}
        return [value for value in selected if value in valid]

    def _single_active_control_period_filter_value(self) -> object | None:
        active = self._active_control_period_filter_values()
        return active[0] if len(active) == 1 else None

    def _active_serial_rows(self) -> list[dict]:
        selected_programs = set(self._active_program_filter_values())
        selected_serials = {str(serial).strip() for serial in (self._checked_serial_filters or []) if str(serial).strip()}
        selected_control_periods = set(self._active_control_period_filter_values())
        selected_suppression = set(self._active_suppression_voltage_filter_values())
        if not self._global_filter_rows:
            out: list[dict] = []
            for row in (self._available_serial_filter_rows or []):
                serial = _td_serial_value(row)
                if not serial or serial not in selected_serials:
                    continue
                if _td_display_program_title((row or {}).get("program_title")) not in selected_programs:
                    continue
                out.append(dict(row))
            return out

        matching_serials: set[str] = set()
        for row in (self._global_filter_rows or []):
            if not isinstance(row, dict):
                continue
            serial = _td_serial_value(row)
            if not serial or serial not in selected_serials:
                continue
            if self._row_program_label(row) not in selected_programs:
                continue
            control_period = _td_control_period_filter_value(row)
            if selected_control_periods and control_period not in selected_control_periods:
                continue
            suppression_voltage = _td_suppression_voltage_filter_value(row)
            if selected_suppression and suppression_voltage not in selected_suppression:
                continue
            matching_serials.add(serial)
        out = []
        for row in (self._available_serial_filter_rows or []):
            serial = _td_serial_value(row)
            if serial and serial in matching_serials:
                out.append(dict(row))
        return out

    def _active_serials(self) -> list[str]:
        return [_td_serial_value(row) for row in self._active_serial_rows() if _td_serial_value(row)]

    def _row_program_label(self, row: dict | None) -> str:
        if not isinstance(row, dict):
            return TD_UNKNOWN_PROGRAM_LABEL
        program_title = str(row.get("program_title") or "").strip()
        if program_title:
            return program_title
        serial = _td_serial_value(row)
        source_row = (self._serial_source_by_serial or {}).get(serial) or {}
        return _td_display_program_title(source_row.get("program_title"))

    def _row_matches_global_filters(self, row: dict | None) -> bool:
        if not isinstance(row, dict):
            return False
        program_label = self._row_program_label(row)
        if program_label not in set(self._active_program_filter_values()):
            return False
        selected_control_periods = set(self._active_control_period_filter_values())
        if selected_control_periods:
            control_period = _td_control_period_filter_value(row)
            if not control_period or control_period not in selected_control_periods:
                return False
        selected_suppression = set(self._active_suppression_voltage_filter_values())
        if selected_suppression:
            suppression_voltage = _td_suppression_voltage_filter_value(row)
            if not suppression_voltage or suppression_voltage not in selected_suppression:
                return False
        serial = _td_serial_value(row)
        if serial:
            active_serials = set(self._active_serials())
            if serial not in active_serials:
                return False
        return True

    def _filter_rows_for_global_selection(self, rows: list[dict]) -> list[dict]:
        return [dict(row) for row in (rows or []) if self._row_matches_global_filters(row)]

    def _visible_run_selection_items(self, mode: str) -> list[dict]:
        selected_programs = set(self._active_program_filter_values())
        selected_control_periods = set(self._active_control_period_filter_values())
        selected_suppression = set(self._active_suppression_voltage_filter_values())
        out: list[dict] = []
        for item in (self._run_selection_views.get(mode) or []):
            if not isinstance(item, dict):
                continue
            raw_programs = item.get("member_programs") or []
            if isinstance(raw_programs, list):
                member_programs = [_td_display_program_title(value) for value in raw_programs if str(value).strip()]
            else:
                member_programs = []
            if not member_programs:
                member_programs = [_td_display_program_title(item.get("program_title"))]
            if not any(program in selected_programs for program in member_programs):
                continue
            raw_control_periods = item.get("member_control_periods") or []
            if isinstance(raw_control_periods, list):
                member_control_periods = [
                    _td_compact_filter_value(value)
                    for value in raw_control_periods
                    if _td_compact_filter_value(value)
                ]
            else:
                member_control_periods = []
            if not member_control_periods:
                single_control_period = _td_compact_filter_value(item.get("control_period"))
                if single_control_period:
                    member_control_periods = [single_control_period]
            if selected_control_periods and not any(value in selected_control_periods for value in member_control_periods):
                continue
            raw_suppression = item.get("member_suppression_voltages") or []
            if isinstance(raw_suppression, list):
                member_suppression = [_td_compact_filter_value(value) for value in raw_suppression if _td_compact_filter_value(value)]
            else:
                member_suppression = []
            if not member_suppression:
                single_suppression = _td_compact_filter_value(item.get("suppression_voltage"))
                if single_suppression:
                    member_suppression = [single_suppression]
            if selected_suppression and member_suppression and not any(value in selected_suppression for value in member_suppression):
                continue
            out.append(dict(item))
        return out

    def _sync_run_mode_availability(self) -> None:
        if not hasattr(self, "cb_run_mode"):
            return
        has_conditions = bool(self._visible_run_selection_items("condition"))
        idx = self.cb_run_mode.findData("condition")
        if idx >= 0:
            try:
                self.cb_run_mode.model().item(idx).setEnabled(has_conditions)
            except Exception:
                pass
        if self._current_run_selector_mode() == "condition" and not has_conditions:
            seq_idx = self.cb_run_mode.findData("sequence")
            if seq_idx >= 0:
                self.cb_run_mode.setCurrentIndex(seq_idx)

    def _show_filter_checklist_popup(self, *, title: str, entries: list[dict], selected_values: list[str]) -> list[str] | None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(720, 540)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        ed_filter = QtWidgets.QLineEdit()
        ed_filter.setPlaceholderText("Filter...")
        layout.addWidget(ed_filter)

        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(listw, 1)

        selected = {str(value).strip() for value in (selected_values or []) if str(value).strip()}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("value") or "").strip()
            label = str(entry.get("label") or value).strip() or value
            if not value:
                continue
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if value in selected else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, str(entry.get("search") or label).strip().lower())
            listw.addItem(item)

        def _apply_filter() -> None:
            needle = str(ed_filter.text() or "").strip().lower()
            for idx in range(listw.count()):
                item = listw.item(idx)
                hay = str(item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or "").strip()
                item.setHidden(bool(needle) and needle not in hay)

        ed_filter.textChanged.connect(_apply_filter)
        _apply_filter()

        btn_row = QtWidgets.QHBoxLayout()
        btn_select_all = QtWidgets.QPushButton("Select All")
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_select_all.clicked.connect(
            lambda: [
                listw.item(i).setCheckState(QtCore.Qt.CheckState.Checked)
                for i in range(listw.count())
                if listw.item(i) is not None
            ]
        )
        btn_clear.clicked.connect(
            lambda: [
                listw.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
                for i in range(listw.count())
                if listw.item(i) is not None
            ]
        )
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        action_row.addWidget(btn_apply)
        action_row.addWidget(btn_cancel)
        layout.addLayout(action_row)

        btn_apply.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        _fit_widget_to_screen(dlg)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None

        chosen: list[str] = []
        for idx in range(listw.count()):
            item = listw.item(idx)
            if item and item.checkState() == QtCore.Qt.CheckState.Checked:
                value = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
                if value:
                    chosen.append(value)
        return chosen

    def _open_program_filter_popup(self) -> None:
        entries = [
            {"value": value, "label": value, "search": value.lower()}
            for value in (self._available_program_filters or [])
            if str(value).strip()
        ]
        chosen = self._show_filter_checklist_popup(
            title="Visible Programs",
            entries=entries,
            selected_values=self._checked_program_filters,
        )
        if chosen is None:
            return
        self._checked_program_filters = [value for value in (self._available_program_filters or []) if value in set(chosen)]
        self._on_global_filters_changed()

    def _open_serial_filter_popup(self) -> None:
        entries: list[dict] = []
        for row in (self._available_serial_filter_rows or []):
            serial = _td_serial_value(row)
            if not serial:
                continue
            program = _td_display_program_title((row or {}).get("program_title"))
            doc_type = str((row or {}).get("document_type") or "").strip()
            parts = [serial, program]
            if doc_type:
                parts.append(doc_type)
            entries.append(
                {
                    "value": serial,
                    "label": " | ".join(parts),
                    "search": self._serial_row_filter_text(row),
                }
            )
        chosen = self._show_filter_checklist_popup(
            title="Visible Serials",
            entries=entries,
            selected_values=self._checked_serial_filters,
        )
        if chosen is None:
            return
        self._checked_serial_filters = [
            _td_serial_value(row)
            for row in (self._available_serial_filter_rows or [])
            if _td_serial_value(row) in set(chosen)
        ]
        self._on_global_filters_changed()

    def _open_suppression_voltage_filter_popup(self) -> None:
        entries = [
            {"value": value, "label": value, "search": value.lower()}
            for value in (self._available_suppression_voltage_filters or [])
            if str(value).strip()
        ]
        chosen = self._show_filter_checklist_popup(
            title="Visible Suppression Voltages",
            entries=entries,
            selected_values=self._checked_suppression_voltage_filters,
        )
        if chosen is None:
            return
        self._checked_suppression_voltage_filters = [
            value for value in (self._available_suppression_voltage_filters or []) if value in set(chosen)
        ]
        self._on_global_filters_changed()

    def _open_control_period_filter_popup(self) -> None:
        entries = [
            {"value": value, "label": value, "search": value.lower()}
            for value in (self._available_control_period_filters or [])
            if str(value).strip()
        ]
        chosen = self._show_filter_checklist_popup(
            title="Visible Control Periods",
            entries=entries,
            selected_values=self._checked_control_period_filters,
        )
        if chosen is None:
            return
        self._checked_control_period_filters = [
            value for value in (self._available_control_period_filters or []) if value in set(chosen)
        ]
        self._on_global_filters_changed()

    def _reset_global_filters(self) -> None:
        self._checked_program_filters = list(self._available_program_filters or [])
        self._checked_serial_filters = [
            _td_serial_value(row)
            for row in (self._available_serial_filter_rows or [])
            if _td_serial_value(row)
        ]
        self._checked_control_period_filters = list(self._available_control_period_filters or [])
        self._checked_suppression_voltage_filters = list(self._available_suppression_voltage_filters or [])
        self._on_global_filters_changed()

    def _on_global_filters_changed(self) -> None:
        self._refresh_global_filter_summaries()
        self._sync_run_mode_availability()
        prev_selection_id = str((self._current_run_selection() or {}).get("id") or "").strip()
        self._refresh_run_dropdown(prev_selection_id=(prev_selection_id or None))
        if hasattr(self, "_refresh_perf_control_period_options"):
            self._refresh_perf_control_period_options()
        if hasattr(self, "_update_perf_control_period_state"):
            self._update_perf_control_period_state()
        if hasattr(self, "_clear_perf_results"):
            self._clear_perf_results()
        self._set_highlight_serials(list(self._highlight_sns or []))

        last_mode = str((self._last_plot_def or {}).get("mode") or "").strip().lower()
        if last_mode and last_mode == self._mode:
            if last_mode == "performance":
                self._plot_performance()
            elif last_mode == "metrics":
                self._plot_metrics()
            elif last_mode == "curves":
                self._plot_curves()
        self._update_plot_zoom_actions()

    @staticmethod
    def _set_all_perf_checks(listw: QtWidgets.QListWidget, checked: bool) -> None:
        for i in range(listw.count()):
            it = listw.item(i)
            if not it:
                continue
            it.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def _open_auto_report_options(self) -> None:
        if not self._plot_ready:
            QtWidgets.QMessageBox.information(self, "Auto Report", "Plotting is unavailable (install matplotlib).")
            return
        if not self._db_path:
            QtWidgets.QMessageBox.information(self, "Auto Report", "Build/refresh cache first.")
            return

        # Build a lightweight options dialog on-demand (keeps startup fast).
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Auto Report Options")
        dlg.resize(860, 680)
        # Force readable light theme regardless of OS/app palette (prevents white-on-white / low-contrast text).
        dlg.setStyleSheet(
            """
            QDialog { background: #ffffff; color: #000000; }
            QLabel { color: #000000; }
            QCheckBox { color: #000000; }
            QLineEdit {
                background: #ffffff;
                color: #000000;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QLineEdit::placeholder { color: #6b7280; }
            QListWidget {
                background: #ffffff;
                color: #000000;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item:selected { background: #dbeafe; color: #000000; }
            QPushButton {
                background: #ffffff;
                color: #000000;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f3f4f6; }
            """
        )
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Auto Report (Certification Style)")
        title.setStyleSheet("font-size: 14px; font-weight: 800;")
        layout.addWidget(title)

        cfg = {}
        try:
            cfg = be.load_trend_auto_report_config(self._project_dir)
        except Exception:
            cfg = {}

        # Output path
        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(QtWidgets.QLabel("Output PDF:"))
        ed_out = QtWidgets.QLineEdit()
        default_name = f"auto_report_{time.strftime('%Y-%m-%d')}.pdf"
        ed_out.setText(str(self._project_dir / default_name))
        btn_browse = QtWidgets.QPushButton("Browse…")
        def _browse():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Save Auto Report",
                ed_out.text().strip() or str(self._project_dir / default_name),
                "PDF Files (*.pdf)",
            )
            if path:
                ed_out.setText(path)
        btn_browse.clicked.connect(_browse)
        row_out.addWidget(ed_out, 1)
        row_out.addWidget(btn_browse)
        layout.addLayout(row_out)

        # Options checkboxes
        opts_row = QtWidgets.QHBoxLayout()
        cb_rebuild = QtWidgets.QCheckBox("Rebuild cache before report")
        cb_rebuild.setChecked(False)
        cb_update_cfg = QtWidgets.QCheckBox("Update excel_trend_config.json (fill missing units/ranges)")
        cb_update_cfg.setChecked(True)
        cb_add_missing = QtWidgets.QCheckBox("Add missing columns to excel_trend_config.json")
        cb_add_missing.setChecked(False)
        cb_metrics = QtWidgets.QCheckBox("Include metrics section/pages")
        include_metrics_default = True
        try:
            if isinstance(cfg, dict):
                rep = cfg.get("report") or {}
                if isinstance(rep, dict) and rep.get("include_metrics") is not None:
                    include_metrics_default = bool(rep.get("include_metrics"))
        except Exception:
            include_metrics_default = True
        cb_metrics.setChecked(include_metrics_default)
        for c in (cb_rebuild, cb_update_cfg, cb_add_missing, cb_metrics):
            opts_row.addWidget(c)
        opts_row.addStretch(1)
        layout.addLayout(opts_row)

        note = QtWidgets.QLabel("Highlight policy: watch-only (highlighted overlays are shown only when watch items trigger).")
        note.setStyleSheet("color: #64748b; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # Left: Serials under certification
        left = QtWidgets.QFrame()
        left_l = QtWidgets.QVBoxLayout(left)
        left_l.setContentsMargins(10, 10, 10, 10)
        left_l.setSpacing(8)
        left_l.addWidget(QtWidgets.QLabel("Serials Under Certification (required)"))
        ed_sn_filter = QtWidgets.QLineEdit()
        ed_sn_filter.setPlaceholderText("Filter serials…")
        left_l.addWidget(ed_sn_filter)
        list_sn = QtWidgets.QListWidget()
        list_sn.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left_l.addWidget(list_sn, 1)

        serials = []
        try:
            serials = be.td_list_serials(self._db_path) if self._db_path else []
        except Exception:
            serials = []
        for sn in serials:
            list_sn.addItem(QtWidgets.QListWidgetItem(str(sn)))

        default_hi = []
        try:
            default_hi = (cfg.get("highlight") or {}).get("default_serials") if isinstance(cfg, dict) else []
        except Exception:
            default_hi = []
        want_hi = {str(s).strip() for s in (default_hi or []) if str(s).strip()}
        if want_hi:
            for i in range(list_sn.count()):
                it = list_sn.item(i)
                if it and it.text().strip() in want_hi:
                    it.setSelected(True)

        def _apply_sn_filter():
            needle = (ed_sn_filter.text() or "").strip().lower()
            for i in range(list_sn.count()):
                it = list_sn.item(i)
                if not it:
                    continue
                it.setHidden(bool(needle) and needle not in it.text().lower())
        ed_sn_filter.textChanged.connect(_apply_sn_filter)

        splitter.addWidget(left)

        # Right: Runs + report analysis params (checkbox lists)
        right = QtWidgets.QFrame()
        right_l = QtWidgets.QVBoxLayout(right)
        right_l.setContentsMargins(10, 10, 10, 10)
        right_l.setSpacing(10)

        list_runs = QtWidgets.QListWidget()
        cb_run_scope = QtWidgets.QComboBox()
        cb_run_scope.addItem("Sequence", "sequence")
        run_selection_views = {
            "sequence": [dict(d) for d in (self._run_selection_views.get("sequence") or []) if isinstance(d, dict)],
            "condition": [dict(d) for d in (self._run_selection_views.get("condition") or []) if isinstance(d, dict)],
        }
        if run_selection_views.get("condition"):
            cb_run_scope.addItem("Run Conditions", "condition")
        cur_scope = self._current_run_selector_mode()
        idx_scope = cb_run_scope.findData(cur_scope)
        if idx_scope >= 0:
            cb_run_scope.setCurrentIndex(idx_scope)

        ed_param_filter = QtWidgets.QLineEdit()

        ed_param_filter.setPlaceholderText("Filter params…")
        list_params = QtWidgets.QListWidget()
        lbl_runs_auto = QtWidgets.QLabel("Selected runs: -")
        lbl_runs_auto.setStyleSheet("color: #64748b; font-size: 11px;")
        lbl_runs_auto.setWordWrap(True)
        row_runs = QtWidgets.QHBoxLayout()
        row_runs.addWidget(QtWidgets.QLabel("Runs included"))
        row_runs.addWidget(cb_run_scope)
        btn_runs_popup = QtWidgets.QPushButton("Select Runs...")
        row_runs.addWidget(btn_runs_popup)
        row_runs.addStretch(1)
        right_l.addLayout(row_runs)
        right_l.addWidget(lbl_runs_auto)

        lbl_params_auto = QtWidgets.QLabel("Selected params: —")
        lbl_params_auto.setStyleSheet("color: #64748b; font-size: 11px;")
        lbl_params_auto.setWordWrap(True)
        row_params = QtWidgets.QHBoxLayout()
        row_params.addWidget(QtWidgets.QLabel("Report Analysis Params (required)"))
        btn_params_popup = QtWidgets.QPushButton("Select Params...")
        row_params.addWidget(btn_params_popup)
        row_params.addStretch(1)
        right_l.addLayout(row_params)
        right_l.addWidget(lbl_params_auto)

        gb_metrics = QtWidgets.QGroupBox("Metrics Pages (optional)")
        gb_metrics.setStyleSheet("QGroupBox { font-weight: 700; }")
        metrics_pick_l = QtWidgets.QVBoxLayout(gb_metrics)
        metrics_pick_l.setContentsMargins(10, 10, 10, 10)
        metrics_pick_l.setSpacing(6)
        lbl_metrics_note = QtWidgets.QLabel("Choose which metric pages to include before WATCH plots.")
        lbl_metrics_note.setWordWrap(True)
        lbl_metrics_note.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 400;")
        metrics_pick_l.addWidget(lbl_metrics_note)
        ed_metric_filter = QtWidgets.QLineEdit()
        ed_metric_filter.setPlaceholderText("Filter metrics pagesâ€¦")
        list_metric_params = QtWidgets.QListWidget()
        lbl_metrics_auto = QtWidgets.QLabel("Selected metric params: -")
        lbl_metrics_auto.setStyleSheet("color: #64748b; font-size: 11px;")
        lbl_metrics_auto.setWordWrap(True)
        btn_metrics_popup = QtWidgets.QPushButton("Select Metrics Pages...")
        metrics_pick_l.addWidget(btn_metrics_popup)
        metrics_pick_l.addWidget(lbl_metrics_auto)
        list_metric_stats = QtWidgets.QListWidget()
        for st in ("mean", "min", "max", "median", "std"):
            it = QtWidgets.QListWidgetItem(st)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if st == "mean" else QtCore.Qt.CheckState.Unchecked)
            list_metric_stats.addItem(it)
        right_l.addWidget(gb_metrics)

        def _collect_checked(listw: QtWidgets.QListWidget) -> list[str]:
            out = []
            for i in range(listw.count()):
                it = listw.item(i)
                if it and it.checkState() == QtCore.Qt.CheckState.Checked:
                    out.append(it.text().strip())
            return [x for x in out if x]

        def _norm_name(s: str) -> str:
            return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

        time_norms = {_norm_name(x) for x in ("time", "time_s", "time(sec)", "time(s)", "time (s)", "time_sec", "times")}
        pulse_norms = {_norm_name(x) for x in ("pulse number", "pulse#", "pulse #", "pulse_number", "pulsenumber", "cycle")}
        x_exclude_norms = time_norms | pulse_norms | {_norm_name("excel_row")}

        def _set_filtered_hidden(listw: QtWidgets.QListWidget, needle: str) -> None:
            needle = (needle or "").strip().lower()
            for i in range(listw.count()):
                it = listw.item(i)
                if not it:
                    continue
                it.setHidden(bool(needle) and needle not in it.text().lower())

        def _selection_summary(items: list[str], total: int) -> str:
            if total <= 0:
                return "-"
            if not items:
                return f"0 / {total}"
            preview = ", ".join(items[:3])
            extra = "" if len(items) <= 3 else f" +{len(items) - 3} more"
            return f"{len(items)} / {total} ({preview}{extra})"

        def _open_checklist_popup(
            *,
            title_text: str,
            target_list: QtWidgets.QListWidget,
            filter_placeholder: str,
            extra_setup=None,
            extra_apply=None,
        ) -> None:
            pop = QtWidgets.QDialog(dlg)
            pop.setWindowTitle(title_text)
            pop.resize(640, 560)
            pop_l = QtWidgets.QVBoxLayout(pop)
            pop_l.setContentsMargins(12, 12, 12, 12)
            pop_l.setSpacing(8)

            ed_filter = QtWidgets.QLineEdit()
            ed_filter.setPlaceholderText(filter_placeholder)
            pop_l.addWidget(ed_filter)

            work_list = QtWidgets.QListWidget()
            work_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            pop_l.addWidget(work_list, 1)

            for i in range(target_list.count()):
                src = target_list.item(i)
                if not src:
                    continue
                it = QtWidgets.QListWidgetItem(src.text())
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(src.checkState())
                it.setData(QtCore.Qt.ItemDataRole.UserRole, src.data(QtCore.Qt.ItemDataRole.UserRole))
                work_list.addItem(it)

            ed_filter.textChanged.connect(lambda text: _set_filtered_hidden(work_list, text))

            ctx = {"popup": pop, "layout": pop_l, "list": work_list}
            if callable(extra_setup):
                extra_setup(ctx)

            btns = QtWidgets.QHBoxLayout()
            btns.addStretch(1)
            btn_ok = QtWidgets.QPushButton("Apply")
            btn_cancel2 = QtWidgets.QPushButton("Cancel")
            btns.addWidget(btn_ok)
            btns.addWidget(btn_cancel2)
            pop_l.addLayout(btns)

            def _apply() -> None:
                states: dict[str, QtCore.Qt.CheckState] = {}
                for i in range(work_list.count()):
                    it = work_list.item(i)
                    if it:
                        states[it.text()] = it.checkState()
                target_list.blockSignals(True)
                for i in range(target_list.count()):
                    it = target_list.item(i)
                    if it and it.text() in states:
                        it.setCheckState(states[it.text()])
                target_list.blockSignals(False)
                if callable(extra_apply):
                    extra_apply(ctx)
                pop.accept()

            btn_ok.clicked.connect(_apply)
            btn_cancel2.clicked.connect(pop.reject)
            _fit_widget_to_screen(pop)
            pop.exec()

        def _selection_label(selection: dict) -> str:
            return self._selection_display_text(selection) or str(selection.get("sequence_name") or selection.get("run_name") or "").strip()

        def _populate_run_selections() -> None:
            mode = str(cb_run_scope.currentData() or "sequence").strip().lower()
            items = [dict(d) for d in (run_selection_views.get(mode) or []) if isinstance(d, dict)]
            list_runs.blockSignals(True)
            list_runs.clear()
            for selection in items:
                label = _selection_label(selection)
                if not label:
                    continue
                it = QtWidgets.QListWidgetItem(label)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Checked)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, selection)
                list_runs.addItem(it)
            list_runs.blockSignals(False)

        def _collect_checked_run_selections() -> list[dict]:
            out: list[dict] = []
            for i in range(list_runs.count()):
                it = list_runs.item(i)
                if not it or it.checkState() != QtCore.Qt.CheckState.Checked:
                    continue
                data = it.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, dict):
                    out.append(dict(data))
            return out

        def _selected_member_runs() -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for selection in _collect_checked_run_selections():
                members = selection.get("member_runs") or []
                if isinstance(members, list):
                    for run in members:
                        rn = str(run or "").strip()
                        if not rn or rn in seen:
                            continue
                        seen.add(rn)
                        out.append(rn)
                rn = str(selection.get("run_name") or "").strip()
                if rn and rn not in seen:
                    seen.add(rn)
                    out.append(rn)
            return out

        def _update_runs_label() -> None:
            sel = [_selection_label(d) for d in _collect_checked_run_selections() if _selection_label(d)]
            lbl_runs_auto.setText(f"Selected runs: {_selection_summary(sel, list_runs.count())}")

        def _update_params_label():
            sel = _collect_checked(list_params)
            lbl_params_auto.setText(f"Selected params: {_selection_summary(sel, list_params.count())}")

        def _update_metric_params_label():
            params_sel = _collect_checked(list_metric_params)
            stats_sel = _collect_checked(list_metric_stats)
            stats_text = ", ".join(stats_sel) if stats_sel else "none"
            lbl_metrics_auto.setText(
                f"Selected metric params: {_selection_summary(params_sel, list_metric_params.count())} | Stats: {stats_text}"
            )

        def _refresh_params_from_runs():
            runs_sel = _selected_member_runs()
            if not runs_sel or not self._db_path:
                list_params.clear()
                list_metric_params.clear()
                _update_runs_label()
                _update_params_label()
                _update_metric_params_label()
                try:
                    _refresh_perf_eq_options()
                except Exception:
                    pass
                return

            prev_checked: dict[str, bool] = {}
            for i in range(list_params.count()):
                it = list_params.item(i)
                if not it:
                    continue
                prev_checked[_norm_name(it.text())] = it.checkState() == QtCore.Qt.CheckState.Checked

            prev_metric_checked: dict[str, bool] = {}
            for i in range(list_metric_params.count()):
                it = list_metric_params.item(i)
                if not it:
                    continue
                prev_metric_checked[_norm_name(it.text())] = it.checkState() == QtCore.Qt.CheckState.Checked

            y_norms: set[str] = set()
            y_names: list[str] = []
            y_by_run_norm: dict[str, set[str]] = {}
            try:
                for rn in runs_sel:
                    y_by_run_norm.setdefault(rn, set())
                    for c in be.td_list_y_columns(self._db_path, rn):
                        name = str((c or {}).get("name") or "").strip()
                        if not name:
                            continue
                        nk = _norm_name(name)
                        if nk in x_exclude_norms:
                            continue
                        try:
                            y_by_run_norm[rn].add(nk)
                        except Exception:
                            pass
                        if nk in y_norms:
                            continue
                        y_norms.add(nk)
                        y_names.append(name)
            except Exception:
                y_names = []
                y_by_run_norm = {}

            y_names = sorted(y_names, key=lambda s: s.lower())
            list_params.blockSignals(True)
            list_params.clear()
            for name in y_names:
                it = QtWidgets.QListWidgetItem(name)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                checked = prev_checked.get(_norm_name(name))
                if checked is None:
                    checked = True
                it.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
                list_params.addItem(it)
            list_params.blockSignals(False)

            list_metric_params.blockSignals(True)
            list_metric_params.clear()
            for name in y_names:
                it = QtWidgets.QListWidgetItem(name)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                checked = prev_metric_checked.get(_norm_name(name))
                if checked is None:
                    checked = prev_checked.get(_norm_name(name), False)
                it.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
                list_metric_params.addItem(it)
            list_metric_params.blockSignals(False)

            _update_runs_label()
            _update_params_label()
            _update_metric_params_label()

            try:
                _refresh_perf_eq_options()
            except Exception:
                pass

        def _open_runs_popup() -> None:
            _open_checklist_popup(
                title_text="Runs Included",
                target_list=list_runs,
                filter_placeholder="Filter runs...",
                extra_apply=lambda _ctx: (_update_runs_label(), _refresh_params_from_runs()),
            )

        def _open_params_popup() -> None:
            _open_checklist_popup(
                title_text="Report Analysis Params",
                target_list=list_params,
                filter_placeholder="Filter params...",
                extra_apply=lambda _ctx: (_update_params_label(), _refresh_perf_eq_options()),
            )

        def _open_metrics_popup() -> None:
            def _metrics_extra_setup(ctx: dict) -> None:
                layout2 = ctx["layout"]
                layout2.addWidget(QtWidgets.QLabel("Stats"))
                stats_list = QtWidgets.QListWidget()
                stats_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
                for i in range(list_metric_stats.count()):
                    src = list_metric_stats.item(i)
                    if not src:
                        continue
                    it = QtWidgets.QListWidgetItem(src.text())
                    it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(src.checkState())
                    stats_list.addItem(it)
                layout2.addWidget(stats_list)
                ctx["stats_list"] = stats_list

            def _metrics_extra_apply(ctx: dict) -> None:
                stats_list = ctx.get("stats_list")
                if isinstance(stats_list, QtWidgets.QListWidget):
                    list_metric_stats.blockSignals(True)
                    for i in range(list_metric_stats.count()):
                        dst = list_metric_stats.item(i)
                        src = stats_list.item(i)
                        if dst and src:
                            dst.setCheckState(src.checkState())
                    list_metric_stats.blockSignals(False)
                _update_metric_params_label()

            _open_checklist_popup(
                title_text="Metrics Pages",
                target_list=list_metric_params,
                filter_placeholder="Filter metrics pages...",
                extra_setup=_metrics_extra_setup,
                extra_apply=_metrics_extra_apply,
            )

        btn_runs_popup.clicked.connect(_open_runs_popup)
        btn_params_popup.clicked.connect(_open_params_popup)
        btn_metrics_popup.clicked.connect(_open_metrics_popup)

        cb_run_scope.currentIndexChanged.connect(lambda *_: (_populate_run_selections(), _refresh_params_from_runs()))
        list_runs.itemChanged.connect(lambda *_: _refresh_params_from_runs())
        list_params.itemChanged.connect(lambda *_: _update_params_label())
        list_metric_params.itemChanged.connect(lambda *_: _update_metric_params_label())
        list_metric_stats.itemChanged.connect(lambda *_: _update_metric_params_label())
        _populate_run_selections()
        _refresh_params_from_runs()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Performance equations (optional, user-selectable in GUI)
        gb_perf = QtWidgets.QGroupBox("Performance Equations (optional)")
        gb_perf.setStyleSheet("QGroupBox { font-weight: 700; }")
        perf_l = QtWidgets.QVBoxLayout(gb_perf)
        perf_l.setContentsMargins(10, 10, 10, 10)
        perf_l.setSpacing(8)

        lbl_perf_note = QtWidgets.QLabel(
            "Define X vs Y equations from metrics (pooled across selected runs where both columns exist). "
            "X and Y must be Y-columns/parameters on at least two common selected runs."
        )
        lbl_perf_note.setWordWrap(True)
        lbl_perf_note.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 400;")
        perf_l.addWidget(lbl_perf_note)

        tbl_perf = QtWidgets.QTableWidget(0, 4)
        tbl_perf.setHorizontalHeaderLabels(["X Param", "Y Param", "Degree", "Normalize X"])
        try:
            hdr = tbl_perf.horizontalHeader()
            hdr.setStretchLastSection(False)
            hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        except Exception:
            pass
        tbl_perf.setMaximumHeight(180)
        perf_l.addWidget(tbl_perf)

        perf_btn_row = QtWidgets.QHBoxLayout()
        btn_perf_add = QtWidgets.QPushButton("Add Equation")
        btn_perf_del = QtWidgets.QPushButton("Remove Selected")
        perf_btn_row.addWidget(btn_perf_add)
        perf_btn_row.addWidget(btn_perf_del)
        perf_btn_row.addStretch(1)
        perf_l.addLayout(perf_btn_row)

        def _combo_for_perf() -> QtWidgets.QComboBox:
            cb = QtWidgets.QComboBox()
            cb.setEditable(False)
            return cb

        def _perf_available_names() -> list[str]:
            # Pull from the current param list (y columns across checked runs).
            out: list[str] = []
            for i in range(list_params.count()):
                it = list_params.item(i)
                if it and it.text().strip():
                    out.append(it.text().strip())
            return out

        def _refresh_perf_eq_options() -> None:
            names = _perf_available_names()
            for r in range(tbl_perf.rowCount()):
                for c in (0, 1):
                    cb = tbl_perf.cellWidget(r, c)
                    if not isinstance(cb, QtWidgets.QComboBox):
                        continue
                    prev = str(cb.currentText() or "").strip()
                    cb.blockSignals(True)
                    cb.clear()
                    for n in names:
                        cb.addItem(n, n)
                    if prev:
                        cb.setCurrentText(prev)
                    cb.blockSignals(False)

        def _add_perf_row(
            x_val: str = "",
            y_val: str = "",
            degree: int = 2,
            normx: bool = True,
            *,
            stats_list: list[str] | None = None,
            require_min_points: int | None = None,
            display_name: str | None = None,
        ) -> None:
            r = tbl_perf.rowCount()
            tbl_perf.insertRow(r)

            cbx = _combo_for_perf()
            cby = _combo_for_perf()
            if stats_list is not None:
                try:
                    cbx.setProperty("_perf_stats", list(stats_list))
                except Exception:
                    pass
            if require_min_points is not None:
                try:
                    cbx.setProperty("_perf_require_min_points", int(require_min_points))
                except Exception:
                    pass
            if display_name is not None:
                try:
                    cbx.setProperty("_perf_name", str(display_name))
                except Exception:
                    pass
            sp_deg = QtWidgets.QSpinBox()
            sp_deg.setRange(0, 6)
            sp_deg.setValue(int(degree))
            cb_norm = QtWidgets.QCheckBox()
            cb_norm.setChecked(bool(normx))

            tbl_perf.setCellWidget(r, 0, cbx)
            tbl_perf.setCellWidget(r, 1, cby)
            tbl_perf.setCellWidget(r, 2, sp_deg)
            tbl_perf.setCellWidget(r, 3, cb_norm)

            _refresh_perf_eq_options()
            if x_val:
                cbx.setCurrentText(str(x_val))
            if y_val:
                cby.setCurrentText(str(y_val))
            if not x_val or not y_val:
                names = _perf_available_names()
                if names:
                    if not x_val:
                        cbx.setCurrentText(names[0])
                    if not y_val:
                        cby.setCurrentText(names[1] if len(names) > 1 else names[0])

        def _remove_perf_selected() -> None:
            r = tbl_perf.currentRow()
            if r >= 0:
                tbl_perf.removeRow(r)

        btn_perf_add.clicked.connect(lambda *_: _add_perf_row())
        btn_perf_del.clicked.connect(lambda *_: _remove_perf_selected())

        # Pre-populate from excel_trend_config.json if available (user can edit in-place)
        try:
            xl_cfg = be.load_excel_trend_config(Path(be.DEFAULT_EXCEL_TREND_CONFIG).expanduser())
            plotters = xl_cfg.get("performance_plotters") if isinstance(xl_cfg, dict) else []
            if isinstance(plotters, list):
                for pd in plotters:
                    if not isinstance(pd, dict):
                        continue
                    nm = str(pd.get("name") or "").strip() or None
                    x_spec = pd.get("x") or {}
                    y_spec = pd.get("y") or {}
                    x_col = str((x_spec.get("column") if isinstance(x_spec, dict) else "") or "").strip()
                    y_col = str((y_spec.get("column") if isinstance(y_spec, dict) else "") or "").strip()
                    raw_stats = pd.get("stats")
                    if isinstance(raw_stats, list) and all(isinstance(s, str) for s in raw_stats):
                        stats_list = [str(s).strip().lower() for s in raw_stats if str(s).strip()]
                    else:
                        legacy = str((x_spec.get("stat") if isinstance(x_spec, dict) else "mean") or "mean").strip().lower()
                        stats_list = [legacy] if legacy else ["mean"]
                    req_pts = int(pd.get("require_min_points") or 2)
                    fit_cfg = pd.get("fit") or {}
                    deg = int((fit_cfg.get("degree") if isinstance(fit_cfg, dict) else 0) or 0)
                    normx = bool((fit_cfg.get("normalize_x") if isinstance(fit_cfg, dict) else True))
                    if x_col and y_col:
                        _add_perf_row(x_col, y_col, deg, normx, stats_list=stats_list, require_min_points=req_pts, display_name=nm)
        except Exception:
            pass

        layout.addWidget(gb_perf)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_gen = QtWidgets.QPushButton("Generate Report")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(btn_gen)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        def _do_generate():
            out_path = (ed_out.text() or "").strip()
            if not out_path:
                QtWidgets.QMessageBox.information(dlg, "Auto Report", "Select an output PDF path.")
                return
            run_selections_sel = _collect_checked_run_selections()
            runs_sel = _selected_member_runs()
            hi_sel = [it.text().strip() for it in list_sn.selectedItems() if it and it.text().strip()]
            if not runs_sel:
                QtWidgets.QMessageBox.information(dlg, "Auto Report", "Select at least one run.")
                return
            if not hi_sel:
                QtWidgets.QMessageBox.information(dlg, "Auto Report", "Select at least one serial under certification.")
                return
            params_sel = _collect_checked(list_params)
            if not params_sel:
                QtWidgets.QMessageBox.information(dlg, "Auto Report", "Select at least one report analysis parameter.")
                return
            metric_params_sel = _collect_checked(list_metric_params)
            metric_stats_sel = _collect_checked(list_metric_stats)
            if cb_metrics.isChecked():
                if not metric_params_sel:
                    QtWidgets.QMessageBox.information(dlg, "Auto Report", "Select at least one metric parameter for metrics pages.")
                    return
                if not metric_stats_sel:
                    QtWidgets.QMessageBox.information(dlg, "Auto Report", "Select at least one metric statistic for metrics pages.")
                    return

            # Performance plotters (optional override; if empty, backend falls back to excel_trend_config presets)
            perf_plotters = []
            try:
                perf_stat = "mean"
                try:
                    xl_cfg2 = be.load_excel_trend_config(Path(be.DEFAULT_EXCEL_TREND_CONFIG).expanduser())
                    st_list = xl_cfg2.get("statistics") if isinstance(xl_cfg2, dict) else None
                    if isinstance(st_list, list) and st_list:
                        perf_stat = str(st_list[0] or "").strip().lower() or "mean"
                except Exception:
                    perf_stat = "mean"
                if perf_stat not in {"mean", "min", "max", "std"}:
                    perf_stat = "mean"

                # Build a quick per-run availability map (normed names) to validate "same run" constraint.
                by_run_norm: dict[str, set[str]] = {}
                if self._db_path:
                    for rn in runs_sel:
                        by_run_norm[rn] = {_norm_name(str((c or {}).get("name") or "")) for c in be.td_list_y_columns(self._db_path, rn)}

                for r in range(tbl_perf.rowCount()):
                    cbx = tbl_perf.cellWidget(r, 0)
                    cby = tbl_perf.cellWidget(r, 1)
                    sp = tbl_perf.cellWidget(r, 2)
                    cn = tbl_perf.cellWidget(r, 3)
                    x_col = str(cbx.currentText() if isinstance(cbx, QtWidgets.QComboBox) else "").strip()
                    y_col = str(cby.currentText() if isinstance(cby, QtWidgets.QComboBox) else "").strip()
                    deg = int(sp.value() if isinstance(sp, QtWidgets.QSpinBox) else 2)
                    normx = bool(cn.isChecked() if isinstance(cn, QtWidgets.QCheckBox) else True)
                    nm = str(cbx.property("_perf_name") or "").strip() if isinstance(cbx, QtWidgets.QComboBox) else ""
                    stats_list = cbx.property("_perf_stats") if isinstance(cbx, QtWidgets.QComboBox) else None
                    req_pts = cbx.property("_perf_require_min_points") if isinstance(cbx, QtWidgets.QComboBox) else None
                    if not isinstance(stats_list, list) or not all(isinstance(s, str) for s in stats_list):
                        stats_list = [perf_stat]
                    req_pts_i = 2
                    try:
                        req_pts_i = max(2, int(req_pts or 2))
                    except Exception:
                        req_pts_i = 2

                    if not x_col and not y_col:
                        continue
                    if not x_col or not y_col:
                        QtWidgets.QMessageBox.information(dlg, "Auto Report", "Performance equation rows must include both X and Y.")
                        return
                    if _norm_name(x_col) == _norm_name(y_col):
                        QtWidgets.QMessageBox.information(dlg, "Auto Report", "Performance equation X and Y must be different parameters.")
                        return

                    common_runs = 0
                    nx = _norm_name(x_col)
                    ny = _norm_name(y_col)
                    for rn in runs_sel:
                        avail = by_run_norm.get(rn) or set()
                        if nx in avail and ny in avail:
                            common_runs += 1
                    if common_runs < req_pts_i:
                        QtWidgets.QMessageBox.information(
                            dlg,
                            "Auto Report",
                            f"Performance equation '{y_col} vs {x_col}' requires at least {req_pts_i} selected runs where both columns exist.",
                        )
                        return

                    perf_plotters.append(
                        {
                            "name": nm or f"{y_col} vs {x_col}",
                            "x": {"column": x_col},
                            "y": {"column": y_col},
                            "stats": stats_list,
                            "require_min_points": req_pts_i,
                            "fit": {"degree": max(0, min(int(deg), 6)), "normalize_x": bool(normx)},
                        }
                    )
            except Exception:
                perf_plotters = []

            payload = {
                "output_pdf": out_path,
                "runs": runs_sel,
                "run_selections": run_selections_sel,
                "run_selection_labels": [_selection_label(d) for d in run_selections_sel if _selection_label(d)],
                "highlighted_serials": hi_sel,
                "params": params_sel,
                "metric_params": metric_params_sel,
                "metric_stats": metric_stats_sel,
                "rebuild_cache": bool(cb_rebuild.isChecked()),
                "update_excel_trend_config": bool(cb_update_cfg.isChecked()),
                "add_missing_columns": bool(cb_add_missing.isChecked()),
                "include_metrics": bool(cb_metrics.isChecked()),
            }
            if perf_plotters:
                payload["performance_plotters"] = perf_plotters
            dlg.setProperty("_auto_report_payload", payload)
            dlg.accept()

        btn_gen.clicked.connect(_do_generate)
        btn_cancel.clicked.connect(dlg.reject)

        _fit_widget_to_screen(dlg)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        payload = dlg.property("_auto_report_payload") or {}
        if not isinstance(payload, dict):
            return
        self._run_auto_report(payload)

    def _run_auto_report(self, payload: dict) -> None:
        try:
            out_pdf = Path(str(payload.get("output_pdf") or "")).expanduser()
            if not out_pdf:
                raise RuntimeError("Missing output path.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Auto Report", str(exc))
            return

        runs = payload.get("runs") or []
        hi = payload.get("highlighted_serials") or []

        options = {
            "runs": runs,
            "run_selections": payload.get("run_selections") or [],
            "run_selection_labels": payload.get("run_selection_labels") or [],
            "params": payload.get("params") or [],
            "metric_params": payload.get("metric_params") or [],
            "metric_stats": payload.get("metric_stats") or [],
            "rebuild_cache": bool(payload.get("rebuild_cache")),
            "update_excel_trend_config": bool(payload.get("update_excel_trend_config", True)),
            "add_missing_columns": bool(payload.get("add_missing_columns")),
            "include_metrics": bool(payload.get("include_metrics", True)),
        }
        if isinstance(payload.get("performance_plotters"), list):
            options["performance_plotters"] = payload.get("performance_plotters")

        self._report_progress.lbl_heading.setText("Generating Auto Report")
        self._report_progress.begin("Building report…")
        self._report_progress.show()

        def _task():
            return be.generate_test_data_auto_report(
                self._project_dir,
                self._workbook_path,
                out_pdf,
                highlighted_serials=list(hi) if isinstance(hi, list) else [],
                options=options,
            )

        worker = ManagerTaskWorker(_task, self)

        def _done(result):
            try:
                self._report_progress.finish("Report complete", success=True)
            except Exception:
                pass
            try:
                pdf_path = str(result.get("output_pdf") or out_pdf)
                summary_path = str(result.get("summary_json") or "")
                msg = f"Auto report generated:\n{pdf_path}"
                if summary_path:
                    msg += f"\n\nSummary JSON:\n{summary_path}"
                mbox = QtWidgets.QMessageBox(self)
                mbox.setWindowTitle("Auto Report")
                try:
                    mbox.setStyleSheet("QMessageBox { background: #ffffff; color: #000000; } QLabel { color: #000000; }")
                except Exception:
                    pass
                mbox.setText(msg)
                btn_open = mbox.addButton("Open PDF", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                btn_close = mbox.addButton("Close", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                mbox.exec()
                if mbox.clickedButton() == btn_open:
                    try:
                        be.open_path(Path(pdf_path))
                    except Exception:
                        pass
                _ = btn_close
            except Exception:
                pass

        def _fail(err: str):
            try:
                self._report_progress.finish(f"Report failed: {err}", success=False)
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Auto Report", str(err))

        worker.completed.connect(_done)
        worker.failed.connect(_fail)
        worker.start()

    def _current_run_selector_mode(self) -> str:
        mode = str(self.cb_run_mode.currentData() if hasattr(self, "cb_run_mode") else "sequence" or "sequence").strip().lower()
        return mode if mode in {"sequence", "condition"} else "sequence"

    @staticmethod
    def _selection_summary_text(items: list[str] | tuple[str, ...] | None) -> str:
        return ", ".join([str(item).strip() for item in (items or []) if str(item).strip()])

    def _metrics_condition_multiselect_active(self) -> bool:
        return (
            str(getattr(self, "_mode", "") or "").strip().lower() == "metrics"
            and self._current_run_selector_mode() == "condition"
            and hasattr(self, "list_metric_run_conditions")
        )

    def _checked_metric_condition_selections(self) -> list[dict]:
        if not hasattr(self, "list_metric_run_conditions"):
            return []
        out: list[dict] = []
        for i in range(self.list_metric_run_conditions.count()):
            it = self.list_metric_run_conditions.item(i)
            if not it or it.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            data = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                out.append(dict(data))
        return out

    def _set_metric_condition_selection_ids(self, selection_ids: list[str] | tuple[str, ...] | set[str] | None) -> None:
        if not hasattr(self, "list_metric_run_conditions"):
            return
        ids = (
            {str(v).strip() for v in selection_ids if str(v).strip()}
            if selection_ids is not None
            else None
        )
        self.list_metric_run_conditions.blockSignals(True)
        try:
            for i in range(self.list_metric_run_conditions.count()):
                it = self.list_metric_run_conditions.item(i)
                if not it:
                    continue
                data = it.data(QtCore.Qt.ItemDataRole.UserRole)
                sel_id = str(data.get("id") or "").strip() if isinstance(data, dict) else ""
                checked = True if ids is None else sel_id in ids
                it.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        finally:
            self.list_metric_run_conditions.blockSignals(False)
        self._on_metric_condition_selection_changed()

    def _combine_run_selections(self, selections: list[dict] | tuple[dict, ...] | None) -> dict:
        items = [dict(item) for item in (selections or []) if isinstance(item, dict)]
        if not items:
            return {}
        if len(items) == 1:
            return dict(items[0])

        first = dict(items[0])
        mode = str(first.get("mode") or "condition").strip().lower() or "condition"
        selection_ids: list[str] = []
        selection_labels: list[str] = []
        run_conditions: list[str] = []
        member_runs: list[str] = []
        member_sequences: list[str] = []
        detail_rows: list[str] = []
        seen_runs: set[str] = set()
        seen_sequences: set[str] = set()
        seen_labels: set[str] = set()
        seen_conditions: set[str] = set()

        for item in items:
            selection_id = str(item.get("id") or "").strip()
            if selection_id:
                selection_ids.append(selection_id)
            label = self._selection_display_text(item)
            if label and label.lower() not in seen_labels:
                seen_labels.add(label.lower())
                selection_labels.append(label)
            condition_label = self._selection_condition_label(item)
            if condition_label and condition_label.lower() not in seen_conditions:
                seen_conditions.add(condition_label.lower())
                run_conditions.append(condition_label)
            for run_name in (item.get("member_runs") or []):
                rn = str(run_name or "").strip()
                if rn and rn.lower() not in seen_runs:
                    seen_runs.add(rn.lower())
                    member_runs.append(rn)
            for sequence_name in (item.get("member_sequences") or []):
                seq = str(sequence_name or "").strip()
                if seq and seq.lower() not in seen_sequences:
                    seen_sequences.add(seq.lower())
                    member_sequences.append(seq)
            details = str(item.get("details_text") or "").strip()
            if details:
                detail_rows.append(details)

        label_text = self._selection_summary_text(selection_labels)
        detail_text = self._selection_summary_text(detail_rows)
        combined: dict = {
            "mode": mode,
            "id": f"{mode}:multi:{'|'.join(selection_ids)}" if selection_ids else f"{mode}:multi",
            "run_name": str(first.get("run_name") or "").strip(),
            "display_text": label_text,
            "run_condition": label_text,
            "run_conditions": list(run_conditions or selection_labels),
            "selection_ids": list(selection_ids),
            "selection_labels": list(selection_labels),
            "member_runs": list(member_runs),
            "member_sequences": list(member_sequences),
            "details_text": detail_text,
        }
        return combined

    def _current_run_selections(self) -> list[dict]:
        if self._metrics_condition_multiselect_active():
            return self._checked_metric_condition_selections()
        ref = self.cb_run.currentData() if hasattr(self, "cb_run") else None
        if isinstance(ref, dict):
            return [dict(ref)]
        run = str(ref or (self.cb_run.currentText() if hasattr(self, "cb_run") else "") or "").strip()
        run = str(self._run_name_by_display.get(run, run) or "").strip()
        if not run:
            return []
        return [
            {
                "mode": "sequence",
                "id": f"sequence:{run}",
                "run_name": run,
                "sequence_name": run,
                "display_text": run,
                "member_runs": [run],
                "member_sequences": [run],
                "run_condition": "",
                "details_text": f"Sequence: {run}",
            }
        ]

    def _current_run_selection(self) -> dict:
        selections = self._current_run_selections()
        if not selections:
            return {}
        if len(selections) == 1:
            return dict(selections[0])
        return self._combine_run_selections(selections)

    def _current_member_runs(self) -> list[str]:
        runs: list[str] = []
        seen: set[str] = set()
        for selection in self._current_run_selections():
            members = selection.get("member_runs") or []
            if isinstance(members, list):
                for run_name in members:
                    rn = str(run_name or "").strip()
                    if rn and rn.lower() not in seen:
                        seen.add(rn.lower())
                        runs.append(rn)
            run_name = str(selection.get("run_name") or "").strip()
            if run_name and run_name.lower() not in seen:
                seen.add(run_name.lower())
                runs.append(run_name)
        return runs

    def _current_run_name(self) -> str:
        runs = self._current_member_runs()
        if runs:
            return runs[0]
        selection = self._current_run_selection()
        return str(selection.get("run_name") or "").strip()

    def _run_display_text(self, run_name: str) -> str:
        rn = str(run_name or "").strip()
        if not rn:
            return ""
        dn = str((self._run_display_by_name or {}).get(rn) or "").strip()
        return dn or rn

    @staticmethod
    def _is_internal_run_label(text: object) -> bool:
        return str(text or "").strip().lower() in {"sequence", "condition"}

    def _selection_condition_label(self, selection: dict | None) -> str:
        if not isinstance(selection, dict):
            return ""
        labels = [str(v).strip() for v in (selection.get("selection_labels") or selection.get("run_conditions") or []) if str(v).strip()]
        if len(labels) > 1:
            return self._selection_summary_text(labels)
        fallback = ""
        for candidate in (selection.get("run_condition"), selection.get("display_text")):
            text = str(candidate or "").strip()
            if not text:
                continue
            if not self._is_internal_run_label(text):
                return text
            if not fallback:
                fallback = text
        run_text = self._run_display_text(str(selection.get("run_name") or "").strip())
        if run_text and not self._is_internal_run_label(run_text):
            return run_text
        return fallback or run_text

    def _selection_display_text(self, selection: dict | None) -> str:
        if not isinstance(selection, dict):
            return ""
        mode = str(selection.get("mode") or "sequence").strip().lower()
        if mode == "condition":
            return self._selection_condition_label(selection)
        display_text = str(selection.get("display_text") or "").strip()
        if display_text:
            return display_text
        run = str(selection.get("run_name") or "").strip()
        return self._run_display_text(run) or str(selection.get("sequence_name") or run).strip()

    def _selection_title_parts(self, selection: dict | None) -> tuple[str, str]:
        if not isinstance(selection, dict):
            return "", ""
        seqs = [str(v).strip() for v in (selection.get("member_sequences") or []) if str(v).strip()]
        run_condition = self._selection_condition_label(selection)
        mode = str(selection.get("mode") or "sequence").strip().lower()
        if mode == "condition":
            seq_text = ", ".join(seqs)
        else:
            seq_text = str(selection.get("sequence_name") or (seqs[0] if seqs else selection.get("run_name")) or "").strip()
        return seq_text, run_condition

    def _compose_run_title(self, selection: dict | None, suffix: str = "") -> str:
        seq_text, run_condition = self._selection_title_parts(selection)
        mode = str((selection or {}).get("mode") or "").strip().lower()
        multi_condition = len([str(v).strip() for v in ((selection or {}).get("selection_labels") or (selection or {}).get("run_conditions") or []) if str(v).strip()]) > 1
        parts: list[str] = []
        if mode == "condition":
            if run_condition:
                parts.append(f"{'Run Conditions' if multi_condition else 'Run Condition'}: {run_condition}")
            if seq_text:
                parts.append(f"Sequences: {seq_text}")
        else:
            if seq_text:
                parts.append(f"Sequence: {seq_text}")
            if run_condition:
                parts.append(f"Run Condition: {run_condition}")
        if suffix:
            parts.append(str(suffix).strip())
        return " | ".join([p for p in parts if str(p).strip()])

    @staticmethod
    def _selection_observation_filters(selection: dict | None) -> tuple[str, str]:
        if not isinstance(selection, dict):
            return "", ""
        if str(selection.get("mode") or "sequence").strip().lower() != "sequence":
            return "", ""
        program_title = str(selection.get("program_title") or "").strip()
        source_run_name = str(selection.get("source_run_name") or selection.get("sequence_name") or "").strip()
        return program_title, source_run_name

    def _selected_metric_plot_source(self) -> str:
        checked = bool(
            getattr(self, "btn_metric_plot_source", None)
            and self.btn_metric_plot_source.isChecked()
        )
        raw = (
            getattr(be, "TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES", "all_sequences")
            if checked
            else getattr(be, "TD_METRIC_PLOT_SOURCE_AGGREGATE", "aggregate")
        )
        normalizer = getattr(be, "td_metric_normalize_plot_source", None)
        if callable(normalizer):
            try:
                return str(normalizer(raw)).strip().lower()
            except Exception:
                pass
        return str(raw).strip().lower() or "aggregate"

    def _set_metric_plot_source(self, value: object) -> None:
        normalizer = getattr(be, "td_metric_normalize_plot_source", None)
        if callable(normalizer):
            try:
                source = str(normalizer(value)).strip().lower()
            except Exception:
                source = str(value or "").strip().lower()
        else:
            source = str(value or "").strip().lower()
        want_all_sequences = source == getattr(be, "TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES", "all_sequences")
        if hasattr(self, "btn_metric_plot_source"):
            self.btn_metric_plot_source.setChecked(bool(want_all_sequences))

    def _load_metric_series_for_selection(
        self,
        run_name: str,
        column_name: str,
        stat: str,
        *,
        selection: dict | None = None,
        control_period_filter: object = None,
        run_type_filter: object = None,
        metric_source: object = None,
    ) -> list[dict]:
        if not getattr(self, "_db_path", None):
            return []
        metric_source_value = (
            metric_source
            if metric_source is not None
            else getattr(be, "TD_METRIC_PLOT_SOURCE_AGGREGATE", "aggregate")
        )
        normalizer = getattr(be, "td_metric_normalize_plot_source", None)
        if callable(normalizer):
            try:
                metric_source_norm = str(normalizer(metric_source_value)).strip().lower()
            except Exception:
                metric_source_norm = str(metric_source_value or "").strip().lower()
        else:
            metric_source_norm = str(metric_source_value or "").strip().lower()
        program_title, source_run_name = self._selection_observation_filters(selection)
        if metric_source_norm != getattr(be, "TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES", "all_sequences"):
            program_title, source_run_name = "", ""
        loader_control_period_filter = (
            control_period_filter
            if control_period_filter not in (None, "")
            else self._single_active_control_period_filter_value()
        )
        try:
            rows = be.td_load_metric_series(
                self._db_path,
                run_name,
                column_name,
                stat,
                program_title=(program_title or None),
                source_run_name=(source_run_name or None),
                control_period_filter=loader_control_period_filter,
                run_type_filter=run_type_filter,
                metric_source=metric_source_norm,
            )
            return self._filter_rows_for_global_selection(rows)
        except Exception:
            return []

    def _load_curves_for_selection(
        self,
        run_name: str,
        y_name: str,
        x_name: str,
        *,
        selection: dict | None = None,
        serials: list[str] | None = None,
    ) -> list[dict]:
        if not getattr(self, "_db_path", None):
            return []
        program_title, source_run_name = self._selection_observation_filters(selection)
        active_serials = [str(value).strip() for value in (serials or self._active_serials()) if str(value).strip()]
        try:
            rows = be.td_load_curves(
                self._db_path,
                run_name,
                y_name,
                x_name,
                serials=active_serials,
                program_title=(program_title or None),
                source_run_name=(source_run_name or None),
                control_period_filter=self._single_active_control_period_filter_value(),
            )
            return self._filter_rows_for_global_selection(rows)
        except Exception:
            return []

    def _select_run_by_id(self, selection_id: str) -> None:
        key = str(selection_id or "").strip()
        if not key or not hasattr(self, "cb_run"):
            return
        for idx in range(self.cb_run.count()):
            data = self.cb_run.itemData(idx)
            if isinstance(data, dict) and str(data.get("id") or "").strip() == key:
                self.cb_run.setCurrentIndex(idx)
                return

    def _selection_from_plot_def(self, d: dict | None) -> dict:
        if not isinstance(d, dict):
            return {}
        want_mode = str(d.get("selector_mode") or "sequence").strip().lower()
        want_mode = want_mode if want_mode in {"sequence", "condition"} else "sequence"
        want_ids = (
            [str(v).strip() for v in (d.get("selection_ids") or []) if str(v).strip()]
            if isinstance(d.get("selection_ids"), list)
            else []
        )
        want_id = str(d.get("selection_id") or "").strip()
        views = self._run_selection_views.get(want_mode) or []
        if want_mode == "condition" and len(want_ids) > 1:
            items: list[dict] = []
            by_id = {
                str(item.get("id") or "").strip(): dict(item)
                for item in views
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            }
            for selection_id in want_ids:
                item = by_id.get(selection_id)
                if item:
                    items.append(dict(item))
            if items:
                combined = self._combine_run_selections(items)
                if combined:
                    return combined
            labels = (
                [str(v).strip() for v in (d.get("selection_labels") or d.get("run_conditions") or []) if str(v).strip()]
                if isinstance(d.get("selection_labels") or d.get("run_conditions"), list)
                else []
            )
            runs = [str(v).strip() for v in (d.get("member_runs") or []) if str(v).strip()] if isinstance(d.get("member_runs"), list) else []
            seqs = [str(v).strip() for v in (d.get("member_sequences") or []) if str(v).strip()] if isinstance(d.get("member_sequences"), list) else []
            return {
                "mode": "condition",
                "id": want_id or f"condition:multi:{'|'.join(want_ids)}",
                "display_text": self._selection_summary_text(labels),
                "run_condition": self._selection_summary_text(labels),
                "run_conditions": list(labels),
                "selection_ids": list(want_ids),
                "selection_labels": list(labels),
                "member_runs": runs,
                "member_sequences": seqs,
                "details_text": f"Source Sequences: {', '.join(seqs)}" if seqs else "",
            }
        for item in views:
            if isinstance(item, dict) and str(item.get("id") or "").strip() == want_id:
                return dict(item)
        runs = [str(v).strip() for v in (d.get("member_runs") or []) if str(v).strip()] if isinstance(d.get("member_runs"), list) else []
        run = str(d.get("run") or "").strip()
        if not runs and run:
            runs = [run]
        if not runs:
            return {}
        if want_mode == "condition":
            seqs_raw = d.get("member_sequences")
            seqs = (
                [str(v).strip() for v in seqs_raw if str(v).strip()]
                if isinstance(seqs_raw, list)
                else [str(v).strip() for v in runs if str(v).strip()]
            )
            label = str(d.get("run_condition") or d.get("display_text") or "").strip()
            return {
                "mode": "condition",
                "id": want_id or f"condition:{'|'.join(runs)}",
                "display_text": label,
                "run_condition": label,
                "member_runs": runs,
                "member_sequences": seqs,
                "details_text": f"Sequences: {', '.join(seqs)}",
            }
        run0 = runs[0]
        return {
            "mode": "sequence",
            "id": want_id or f"sequence:{run0}",
            "run_name": run0,
            "sequence_name": run0,
            "display_text": run0,
            "member_runs": [run0],
            "member_sequences": [run0],
            "run_condition": "",
            "details_text": f"Sequence: {run0}",
        }

    def _populate_metric_condition_list(self) -> None:
        if not hasattr(self, "list_metric_run_conditions"):
            return
        prev_checked_ids = {
            str(data.get("id") or "").strip()
            for data in self._checked_metric_condition_selections()
            if isinstance(data, dict) and str(data.get("id") or "").strip()
        }
        had_items = self.list_metric_run_conditions.count() > 0
        items = self._visible_run_selection_items("condition")
        self.list_metric_run_conditions.blockSignals(True)
        self.list_metric_run_conditions.clear()
        try:
            for item in items:
                label = self._selection_display_text(item)
                if not label:
                    continue
                it = QtWidgets.QListWidgetItem(label)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                sel_id = str(item.get("id") or "").strip()
                checked = (sel_id in prev_checked_ids) if had_items else True
                it.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, item)
                self.list_metric_run_conditions.addItem(it)
        finally:
            self.list_metric_run_conditions.blockSignals(False)

    def _refresh_run_selection_visibility(self) -> None:
        multi_active = self._metrics_condition_multiselect_active()
        if hasattr(self, "lbl_run_combo"):
            self.lbl_run_combo.setVisible(not multi_active)
        if hasattr(self, "cb_run"):
            self.cb_run.setVisible(not multi_active)
        if hasattr(self, "metrics_condition_frame"):
            self.metrics_condition_frame.setVisible(multi_active)

    def _on_metric_condition_selection_changed(self) -> None:
        selection = self._current_run_selection()
        if hasattr(self, "lbl_run_details"):
            details = str(selection.get("details_text") or "").strip()
            if not details and self._metrics_condition_multiselect_active():
                details = "Run Conditions: -"
            self.lbl_run_details.setText(details or "Sequence: -")
        try:
            self._refresh_columns_for_run()
            self._refresh_stats_preview()
        except Exception:
            pass

    def _metric_bounds_for_run(self, run_name: str) -> dict[str, dict]:
        run = str(run_name or "").strip()
        if not run or not getattr(self, "_workbook_path", None) or not getattr(self, "_project_dir", None):
            return {}
        reader = getattr(be, "_read_td_support_workbook", None)
        if not callable(reader):
            return {}
        try:
            support_cfg = reader(self._workbook_path, project_dir=self._project_dir)
        except Exception:
            return {}
        condition_bounds = {
            str(k).strip(): dict(v)
            for k, v in (support_cfg.get("condition_bounds") or {}).items()
            if str(k).strip() and isinstance(v, dict)
        }
        if run in condition_bounds:
            return dict(condition_bounds.get(run) or {})
        sequences = [
            dict(s)
            for s in (support_cfg.get("sequences") or [])
            if isinstance(s, dict) and bool(s.get("enabled", True))
        ]
        bounds_by_sequence = {
            str(k).strip(): dict(v)
            for k, v in (support_cfg.get("bounds_by_sequence") or {}).items()
            if str(k).strip() and isinstance(v, dict)
        }

        def _norm_name(value: object) -> str:
            return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())

        seq_name = ""
        for seq in sequences:
            source_match = str(seq.get("source_run_name") or "").strip()
            if source_match and _norm_name(source_match) == _norm_name(run):
                seq_name = str(seq.get("sequence_name") or source_match).strip() or source_match
                break
        if not seq_name:
            for seq in sequences:
                candidate = str(seq.get("sequence_name") or "").strip()
                if candidate and _norm_name(candidate) == _norm_name(run):
                    seq_name = candidate
                    break
        if not seq_name:
            seq_name = run
        return dict(bounds_by_sequence.get(seq_name) or {})

    def _plot_metric_bound_lines(self, axes, bound: dict | None) -> None:
        specs_fn = getattr(be, "td_metric_bound_line_specs", None)
        specs = specs_fn(bound) if callable(specs_fn) else []
        for spec in specs:
            try:
                axes.axhline(
                    float(spec.get("value")),
                    color=str(spec.get("color") or "red"),
                    linestyle=str(spec.get("linestyle") or "--"),
                    alpha=float(spec.get("alpha") if spec.get("alpha") is not None else 0.8),
                    linewidth=float(spec.get("linewidth") if spec.get("linewidth") is not None else 1.2),
                )
            except Exception:
                continue

    def _apply_metric_program_segments(self, axes, labels: list[str]) -> None:
        if axes is None or not labels:
            return
        try:
            from matplotlib.patches import Rectangle
            from matplotlib.transforms import blended_transform_factory
        except Exception:
            return
        segments = _td_metric_program_segments(labels, self._active_serial_rows())
        if not segments:
            return
        palette = ["#1d4ed8", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#334155"]
        transform = blended_transform_factory(axes.transData, axes.transAxes)
        for idx, segment in enumerate(segments):
            try:
                start = int(segment.get("start"))
                end = int(segment.get("end"))
            except Exception:
                continue
            color = palette[idx % len(palette)]
            x0 = float(start) - 0.5
            width = float(end - start + 1)
            try:
                axes.add_patch(
                    Rectangle(
                        (x0, 0.0),
                        width,
                        1.0,
                        transform=transform,
                        fill=False,
                        linewidth=1.4,
                        linestyle="-",
                        edgecolor=color,
                        alpha=0.95,
                        zorder=0.2,
                    )
                )
            except Exception:
                continue
            label = str(segment.get("program") or "Unknown Program")
            span = end - start + 1
            try:
                axes.text(
                    (float(start) + float(end)) / 2.0,
                    0.985,
                    label,
                    transform=transform,
                    ha="center",
                    va="top",
                    fontsize=(8 if span >= 2 else 7),
                    fontweight="bold",
                    rotation=(90 if span == 1 and len(label) > 14 else 0),
                    color=color,
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "#ffffff",
                        "edgecolor": color,
                        "linewidth": 1.0,
                        "alpha": 0.92,
                    },
                    clip_on=True,
                    zorder=4.0,
                )
            except Exception:
                continue

    @staticmethod
    def _metric_points_for_serial_labels(series_rows: list[dict], labels: list[str]) -> list[dict]:
        index_by_serial = {
            str(sn).strip(): idx
            for idx, sn in enumerate(labels or [])
            if str(sn).strip()
        }
        rows_by_serial: dict[str, list[dict]] = {}
        for row in series_rows or []:
            if not isinstance(row, dict):
                continue
            sn = str(row.get("serial") or "").strip()
            val = row.get("value_num")
            if sn not in index_by_serial or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                continue
            rows_by_serial.setdefault(sn, []).append(dict(row))
        out: list[dict] = []
        for sn, rows in rows_by_serial.items():
            rows_sorted = sorted(
                rows,
                key=lambda row: (
                    str(row.get("program_title") or "").lower(),
                    str(row.get("source_run_name") or "").lower(),
                    str(row.get("observation_id") or "").lower(),
                ),
            )
            count = len(rows_sorted)
            if count <= 1:
                offsets = [0.0]
            else:
                step = min(0.16, 0.38 / float(max(1, count - 1)))
                offsets = [((idx - ((count - 1) / 2.0)) * step) for idx in range(count)]
            base_x = float(index_by_serial.get(sn, 0))
            for offset, row in zip(offsets, rows_sorted):
                out.append(
                    {
                        "x": base_x + float(offset),
                        "y": float(row.get("value_num") or 0.0),
                        "serial": sn,
                        "observation_id": str(row.get("observation_id") or "").strip(),
                        "program_title": str(row.get("program_title") or "").strip(),
                        "source_run_name": str(row.get("source_run_name") or "").strip(),
                    }
                )
        out.sort(key=lambda row: (float(row.get("x") or 0.0), str(row.get("observation_id") or "").lower()))
        return out

    def _curve_trace_label(self, run_name: str, curve_row: dict, *, multi_run: bool) -> str:
        serial = str(curve_row.get("serial") or "").strip() or "SN"
        program_title = str(curve_row.get("program_title") or "").strip()
        source_run_name = str(curve_row.get("source_run_name") or "").strip()
        parts: list[str] = []
        if multi_run:
            parts.append(self._run_display_text(run_name) or run_name)
        parts.append(serial)
        if program_title:
            parts.append(program_title)
        if source_run_name:
            parts.append(source_run_name)
        return " | ".join([part for part in parts if str(part).strip()])

    def _select_run_by_name(self, run_name: str) -> None:
        rn = str(run_name or "").strip()
        if not rn:
            return
        self._select_run_by_id(f"sequence:{rn}")

    def _refresh_run_dropdown(self, prev_selection_id: str | None = None) -> None:
        if not hasattr(self, "cb_run"):
            return
        if prev_selection_id is None:
            current = self._current_run_selection()
            prev_selection_id = str(current.get("id") or "").strip()
        mode = self._current_run_selector_mode()
        items = self._visible_run_selection_items(mode)
        self._populate_metric_condition_list()
        self.cb_run.blockSignals(True)
        self.cb_run.clear()
        for item in items:
            text = self._selection_display_text(item) or str(item.get("sequence_name") or item.get("run_name") or "").strip()
            if not text:
                continue
            self.cb_run.addItem(text, item)
        if prev_selection_id:
            self._select_run_by_id(prev_selection_id)
        if self.cb_run.currentIndex() < 0 and self.cb_run.count() > 0:
            self.cb_run.setCurrentIndex(0)
        self.cb_run.blockSignals(False)
        self._refresh_run_selection_visibility()
        selection = self._current_run_selection()
        if hasattr(self, "lbl_run_details"):
            details = str(selection.get("details_text") or "").strip()
            if not details and self._metrics_condition_multiselect_active():
                details = "Run Conditions: -"
            self.lbl_run_details.setText(details or "Sequence: -")
        try:
            self._refresh_columns_for_run()
            self._refresh_stats_preview()
        except Exception:
            pass

    def _refresh_columns_for_run(self) -> None:
        if not self._db_path:
            return
        selection = self._current_run_selection()
        if hasattr(self, "lbl_run_details"):
            details = str(selection.get("details_text") or "").strip()
            self.lbl_run_details.setText(details or "Sequence: -")
        runs = self._current_member_runs()
        if not runs:
            self.cb_y_curve.clear()
            self.cb_x.clear()
            self.list_y_metrics.clear()
            self._refresh_metric_selector_summaries()
            self._refresh_curve_selector_summaries()
            self._schedule_mode_panel_height_sync()
            return

        y_cols: list[dict] = []
        metric_y_cols: list[dict] = []
        x_cols: list[str] = []
        seen_y: set[str] = set()
        seen_metric_y: set[str] = set()
        seen_x: set[str] = set()
        for run in runs:
            try:
                raw_cols = be.td_list_raw_y_columns(self._db_path, run)
            except Exception:
                raw_cols = []
            try:
                metric_cols = be.td_list_metric_y_columns(self._db_path, run)
            except Exception:
                metric_cols = []
            try:
                x_vals = be.td_list_x_columns(self._db_path, run)
            except Exception:
                x_vals = []
            for col in raw_cols:
                name = str((col or {}).get("name") or "").strip()
                if name and name not in seen_y:
                    seen_y.add(name)
                    y_cols.append(dict(col))
            for col in metric_cols:
                name = str((col or {}).get("name") or "").strip()
                if name and name not in seen_metric_y:
                    seen_metric_y.add(name)
                    metric_y_cols.append(dict(col))
            for x_val in x_vals:
                x_name = str(x_val or "").strip()
                if x_name and x_name not in seen_x:
                    seen_x.add(x_name)
                    x_cols.append(x_name)

        def _norm_name(s: str) -> str:
            return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

        # Keep curve/metric Y selectors free of curve X-axis columns.
        time_norms = {_norm_name(x) for x in ("time", "time_s", "time(sec)", "time(s)", "time (s)", "time_sec", "times")}
        pulse_norms = {_norm_name(x) for x in ("pulse number", "pulse#", "pulse #", "pulse_number", "pulsenumber", "cycle")}
        x_exclude_norms = time_norms | pulse_norms | {_norm_name("excel_row")}

        y_names = [
            str(c.get("name") or "")
            for c in y_cols
            if str(c.get("name") or "").strip() and _norm_name(str(c.get("name") or "")) not in x_exclude_norms
        ]

        metric_y_names = [str(c.get("name") or "").strip() for c in metric_y_cols if str(c.get("name") or "").strip()]
        if not metric_y_names:
            metric_y_names = list(y_names)

        # Multi-select Y list for metrics mode
        prev_selected = {it.text() for it in self.list_y_metrics.selectedItems()} if hasattr(self, "list_y_metrics") else set()
        self.list_y_metrics.blockSignals(True)
        self.list_y_metrics.clear()
        for name in metric_y_names:
            self.list_y_metrics.addItem(QtWidgets.QListWidgetItem(name))
        # Restore selection if possible; otherwise select all by default.
        restored = False
        if prev_selected:
            for i in range(self.list_y_metrics.count()):
                it = self.list_y_metrics.item(i)
                if it.text() in prev_selected:
                    it.setSelected(True)
                    restored = True
        if not restored and self.list_y_metrics.count() > 0:
            self.list_y_metrics.selectAll()
        self.list_y_metrics.blockSignals(False)
        self._refresh_metric_y_columns_summary()

        prev_x = self._current_curve_x_key() or self._current_curve_x_label()
        self.cb_x.blockSignals(True)
        self.cb_x.clear()
        xs_raw = [str(x or "").strip() for x in (x_cols or []) if str(x or "").strip()]
        xs_by_norm: dict[str, str] = {}
        for x in xs_raw:
            n = _norm_name(x)
            if n and n not in xs_by_norm:
                xs_by_norm[n] = x

        def _pick_best(preferred: list[str]) -> str:
            have = set(xs_raw)
            for p in preferred:
                if p in have:
                    return p
            for p in preferred:
                v = xs_by_norm.get(_norm_name(p))
                if v:
                    return v
            return ""

        # Only show Time + Pulse Number as curve X-axis options (never excel_row).
        time_key = _pick_best(["Time", "Time (s)", "Time(s)", "time_s", "time", "time_sec", "time (s)", "time(s)"])
        pulse_key = _pick_best(["Pulse Number", "Pulse #", "cycle", "Cycle", "pulse_number", "pulsenumber", "Pulse", "pulse"])
        if time_key:
            self.cb_x.addItem("Time", time_key)
        if pulse_key and pulse_key != time_key:
            self.cb_x.addItem("Pulse Number", pulse_key)

        want = (prev_x or "").strip()
        if _norm_name(want) in time_norms:
            want = "Time"
        elif _norm_name(want) in pulse_norms:
            want = "Pulse Number"
        if want:
            self._set_combo_to_value(self.cb_x, want)
        self.cb_x.blockSignals(False)
        self._refresh_curve_x_column_summary()
        self._refresh_curve_y_columns()
        self._schedule_mode_panel_height_sync()

    def _refresh_curve_y_columns(self) -> None:
        if not self._db_path or not hasattr(self, "cb_y_curve") or not hasattr(self, "cb_x"):
            return
        runs = self._current_member_runs()
        prev_y = self._current_curve_y_name()
        self.cb_y_curve.blockSignals(True)
        self.cb_y_curve.clear()
        if not runs:
            self.cb_y_curve.blockSignals(False)
            self._refresh_curve_y_column_summary()
            return
        x_key = self._current_curve_x_key()
        x_label = self._current_curve_x_label()
        seen_y: set[str] = set()
        y_names: list[str] = []
        for run in runs:
            x_col = self._resolve_curve_x_key(run, x_key or x_label)
            try:
                y_cols = be.td_list_curve_y_columns(self._db_path, run, x_col)
            except Exception:
                y_cols = []
            for col in y_cols:
                name = str((col or {}).get("name") or "").strip()
                if name and name not in seen_y:
                    seen_y.add(name)
                    y_names.append(name)
        for name in y_names:
            self.cb_y_curve.addItem(name, name)
        if prev_y:
            self._set_combo_to_value(self.cb_y_curve, prev_y)
        elif y_names:
            self.cb_y_curve.setCurrentIndex(0)
        self.cb_y_curve.blockSignals(False)
        self._refresh_curve_y_column_summary()
        if self._mode == "curves":
            self._refresh_stats_preview()
        self._schedule_mode_panel_height_sync()

    def _resolve_curve_x_key(self, run: str, x_label: str) -> str:
        """
        Resolve a user-facing X label ("Time" / "Pulse Number") to the underlying td_curves.x_name key.

        Supports older caches where X keys were stored as variants like "time_s" / "cycle".
        """
        if not self._db_path:
            return str(x_label or "").strip()
        run_name = str(run or "").strip()
        label = str(x_label or "").strip()
        if not run_name or not label:
            return label

        def _norm_name(s: str) -> str:
            return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

        time_norms = {_norm_name(x) for x in ("time", "time_s", "time(sec)", "time(s)", "time (s)", "time_sec", "times")}
        pulse_norms = {_norm_name(x) for x in ("pulse number", "pulse#", "pulse #", "pulse_number", "pulsenumber", "cycle")}

        try:
            xs = be.td_list_x_columns(self._db_path, run_name)  # type: ignore[arg-type]
        except Exception:
            xs = []
        xs = [str(x or "").strip() for x in (xs or []) if str(x or "").strip()]
        if label in xs:
            return label

        by_norm: dict[str, str] = {}
        for x in xs:
            n = _norm_name(x)
            if n and n not in by_norm:
                by_norm[n] = x

        n = _norm_name(label)
        if n == _norm_name("excel_row"):
            for pref in ("Time", "Time (s)", "Time(s)", "time_s", "time"):
                v = by_norm.get(_norm_name(pref))
                if v:
                    return v
            for pref in ("Pulse Number", "Pulse #", "cycle", "Cycle", "pulse_number", "pulsenumber"):
                v = by_norm.get(_norm_name(pref))
                if v:
                    return v
        if n in time_norms:
            for pref in ("Time", "Time (s)", "Time(s)", "time_s", "time"):
                v = by_norm.get(_norm_name(pref))
                if v:
                    return v
        if n in pulse_norms:
            for pref in ("Pulse Number", "Pulse #", "cycle", "Cycle", "pulse_number", "pulsenumber"):
                v = by_norm.get(_norm_name(pref))
                if v:
                    return v
        return label

    def _serial_row_filter_text(self, row: dict) -> str:
        if not isinstance(row, dict):
            return ""
        parts = [str(v).strip() for v in row.values() if str(v).strip()]
        return " | ".join(parts).lower()

    def _highlight_summary_text(self) -> str:
        sels = [str(sn).strip() for sn in (self._highlight_sns or []) if str(sn).strip()]
        if not sels:
            return "Highlighted serials: -"
        shown = ", ".join(sels[:3])
        extra = "" if len(sels) <= 3 else f" +{len(sels) - 3} more"
        return f"Highlighted serials: {len(sels)} ({shown}{extra})"

    def _set_highlight_serials(self, serials: list[str]) -> None:
        allowed_serials = {serial for serial in self._active_serials() if str(serial).strip()}
        cleaned: list[str] = []
        seen: set[str] = set()
        for sn in (serials or []):
            val = str(sn or "").strip()
            if not val or val in seen or val not in allowed_serials:
                continue
            seen.add(val)
            cleaned.append(val)
        self._highlight_sns = cleaned
        self._highlight_sn = cleaned[0] if cleaned else ""
        if hasattr(self, "lbl_highlight_serials"):
            self.lbl_highlight_serials.setText(self._highlight_summary_text())
        self._refresh_stats_preview()
        if self._mode == "performance" and getattr(self, "_perf_results_by_stat", None):
            self._update_perf_highlight_models()
            self._fill_perf_equations_table()
            self._redraw_performance_view()

    def _open_highlight_serials_popup(self) -> None:
        rows = [r for r in self._active_serial_rows() if isinstance(r, dict)]
        if not rows:
            QtWidgets.QMessageBox.information(self, "Highlighted Serials", "No serial metadata available.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Highlighted Serials")
        dlg.resize(760, 560)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        ed_filter = QtWidgets.QLineEdit()
        ed_filter.setPlaceholderText("Filter by serial or metadata...")
        layout.addWidget(ed_filter)

        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(listw, 1)

        selected = {str(sn).strip() for sn in (self._highlight_sns or []) if str(sn).strip()}
        for row in rows:
            sn = _td_serial_value(row)
            if not sn:
                continue
            program = _td_display_program_title((row or {}).get("program_title"))
            doc_type = str(row.get("document_type") or "").strip()
            label_parts = [sn]
            if program:
                label_parts.append(program)
            if doc_type:
                label_parts.append(doc_type)
            item = QtWidgets.QListWidgetItem(" | ".join(label_parts))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if sn in selected else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, sn)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, self._serial_row_filter_text(row))
            listw.addItem(item)

        def _apply_filter() -> None:
            needle = (ed_filter.text() or "").strip().lower()
            for i in range(listw.count()):
                item = listw.item(i)
                blob = str(item.data(QtCore.Qt.ItemDataRole.UserRole + 1) or "").strip()
                item.setHidden(bool(needle) and needle not in blob)

        ed_filter.textChanged.connect(_apply_filter)
        _apply_filter()

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btn_ok = QtWidgets.QPushButton("Apply")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        def _apply_selection() -> None:
            picked: list[str] = []
            for i in range(listw.count()):
                item = listw.item(i)
                if item and item.checkState() == QtCore.Qt.CheckState.Checked:
                    sn = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
                    if sn:
                        picked.append(sn)
            self._set_highlight_serials(picked)
            dlg.accept()

        btn_ok.clicked.connect(_apply_selection)
        btn_cancel.clicked.connect(dlg.reject)
        _fit_widget_to_screen(dlg)
        dlg.exec()

    def _populate_stats_table(self, run: str, y_col: str, highlight_sn: str) -> None:
        def _set_val(key: str, v: object) -> None:
            lbl = (self._stats_values or {}).get(key)
            if lbl is None:
                return
            if v is None or str(v).strip() == "":
                lbl.setText("—")
            else:
                lbl.setText(str(v))

        if not getattr(self, "_db_path", None) or not str(run or "").strip() or not str(y_col or "").strip():
            for k in ("serial", "count", "mean", "std", "min", "max"):
                _set_val(k, None)
            return

        sn = str(highlight_sn or "").strip()
        if not sn:
            for k in ("serial", "count", "mean", "std", "min", "max"):
                _set_val(k, None)
            return

        _set_val("serial", sn)
        selection = self._current_run_selection()
        metric_source = self._selected_metric_plot_source()
        for st in ("count", "mean", "std", "min", "max"):
            val = None
            series = self._load_metric_series_for_selection(
                run,
                y_col,
                st,
                selection=selection,
                metric_source=metric_source,
            )
            matches = [r for r in series if str(r.get("serial") or "").strip() == sn]
            if (
                metric_source == getattr(be, "TD_METRIC_PLOT_SOURCE_ALL_SEQUENCES", "all_sequences")
                and len(matches) > 1
            ):
                if st == "count":
                    _set_val(st, f"{len(matches)} seqs")
                else:
                    _set_val(st, f"Multiple ({len(matches)})")
                continue
            for r in matches:
                val = r.get("value_num")
                break
            if isinstance(val, (int, float)) and st != "count":
                _set_val(st, f"{float(val):.6g}")
            else:
                _set_val(st, val)

    def _refresh_stats_preview(self) -> None:
        run = self._current_run_name()
        y_col = ""
        if self._mode == "curves":
            y_col = self._current_curve_y_name()
        else:
            # Preview first selected Y column (keeps table compact).
            selected = self._selected_list_widget_texts(getattr(self, "list_y_metrics", None))
            y_col = (selected[0] if selected else "").strip()
        self._populate_stats_table(run, y_col, self._highlight_sn)

    def _set_mode(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        if m not in {"curves", "metrics", "performance"}:
            return
        self._mode = m
        if hasattr(self, "_tabs"):
            if m == "metrics":
                self._tabs.setCurrentIndex(0)
            elif m == "curves":
                self._tabs.setCurrentIndex(1)
            else:
                self._tabs.setCurrentIndex(2)
        self.btn_mode_curves.setChecked(m == "curves")
        self.btn_mode_metrics.setChecked(m == "metrics")
        if hasattr(self, "btn_mode_perf"):
            self.btn_mode_perf.setChecked(m == "performance")
        if hasattr(self, "run_selector_frame"):
            self.run_selector_frame.setVisible(m != "performance")
        self._refresh_run_selection_visibility()
        self._refresh_plot_view_band_controls()
        self.btn_plot.setText(
            "Plot Curves" if m == "curves" else ("Plot Metrics" if m == "metrics" else "Plot Performance (Legacy)")
        )
        if hasattr(self, "btn_plot_perf_cached"):
            self.btn_plot_perf_cached.setVisible(m == "performance")
            self.btn_plot_perf_cached.setEnabled(m == "performance")
        if m != "performance":
            self._refresh_stats_preview()
        self._update_perf_primary_equation_banner()
        self._schedule_mode_panel_height_sync()

    def _plot_current_mode(self) -> None:
        self._set_plot_note("")
        if self._mode == "performance":
            self._plot_performance(user_initiated=True)
        elif self._mode == "metrics":
            self._plot_metrics()
        else:
            self._plot_curves()

    def _plot_metrics(self) -> None:
        if not self._plot_ready or not self._db_path:
            return
        self._set_plot_note("")
        self._ensure_main_axes("2d")
        selection = self._current_run_selection()
        selections = self._current_run_selections()
        runs = self._current_member_runs()
        stats = [text.lower() for text in self._selected_list_widget_texts(getattr(self, "list_stats", None))]
        stats = [s for s in stats if s]
        if hasattr(self, "cb_metric_average") and self.cb_metric_average.isChecked() and "average" not in stats:
            stats.append("average")
        if not runs or not stats:
            return
        y_cols = self._selected_list_widget_texts(getattr(self, "list_y_metrics", None))
        y_cols = [c for c in y_cols if c]
        if not y_cols:
            QtWidgets.QMessageBox.information(self, "Plot Metrics", "Select at least one Y column.")
            return

        try:
            serial_rows = self._active_serial_rows()
            labels = _td_order_metric_serials(
                [_td_serial_value(row) for row in serial_rows if _td_serial_value(row)],
                serial_rows,
            )
        except Exception:
            labels = []
        if not labels:
            QtWidgets.QMessageBox.information(self, "Plot Metrics", "No serial numbers available.")
            return

        metric_source = self._selected_metric_plot_source()
        stats_label = self._metric_title_suffix(stats)
        y_label = stats[0] if len(stats) == 1 else "Metric value"
        plot_bounds = bool(getattr(self, "cb_plot_metric_bounds", None) and self.cb_plot_metric_bounds.isChecked())
        self._axes.clear()
        self._axes.set_title(self._compose_run_title(selection, stats_label))
        self._axes.set_xlabel("Serial Number")
        self._axes.set_ylabel(y_label)
        x = list(range(len(labels)))
        any_plotted = False
        average_summaries: list[str] = []
        multi_run = len(runs) > 1
        run_selection_pairs: list[tuple[str, dict | None]] = []
        if selections:
            seen_pairs: set[tuple[str, str]] = set()
            for selected in selections:
                selected_runs = [str(v).strip() for v in (selected.get("member_runs") or []) if str(v).strip()]
                if not selected_runs:
                    selected_run = str(selected.get("run_name") or "").strip()
                    if selected_run:
                        selected_runs = [selected_run]
                for run_name in selected_runs:
                    pair_key = (run_name.lower(), str(selected.get("id") or "").strip().lower())
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    run_selection_pairs.append((run_name, dict(selected)))
        else:
            run_selection_pairs = [(run_name, selection) for run_name in runs]
        for run, run_selection in run_selection_pairs:
            metric_bounds = self._metric_bounds_for_run(run) if plot_bounds else {}
            for y_col in y_cols:
                for stat in stats:
                    source_stat = "mean" if str(stat).strip().lower() == "average" else stat
                    series = self._load_metric_series_for_selection(
                        run,
                        y_col,
                        source_stat,
                        selection=run_selection,
                        metric_source=metric_source,
                    )
                    try:
                        if str(stat).strip().lower() == "average":
                            y = be.td_metric_average_plot_values(series, labels)
                        else:
                            y = be.td_metric_plot_values(series, labels, stat)
                    except Exception:
                        vmap = {
                            str(r.get("serial") or "").strip(): r.get("value_num")
                            for r in series
                            if str(r.get("serial") or "").strip()
                        }
                        y = [
                            (float(vmap.get(sn)) if isinstance(vmap.get(sn), (int, float)) else float("nan"))
                            for sn in labels
                        ]
                    stat_label = str(stat)
                    label = f"{run}.{y_col}.{stat_label}" if multi_run else f"{y_col}.{stat_label}"
                    try:
                        is_average = str(stat).strip().lower() == "average"
                        if is_average:
                            line = self._axes.plot(
                                x,
                                y,
                                marker=None,
                                linewidth=1.4,
                                label=label,
                            )[0]
                        else:
                            points = self._metric_points_for_serial_labels(series, labels)
                            if not points:
                                continue
                            xs = [float(p.get("x") or 0.0) for p in points]
                            ys = [float(p.get("y") or 0.0) for p in points]
                            line = self._axes.plot(
                                xs,
                                ys,
                                linestyle="",
                                marker="o",
                                markersize=5.2,
                                alpha=0.88,
                                label=label,
                            )[0]
                        bound = dict(metric_bounds.get(str(y_col)) or {})
                        self._plot_metric_bound_lines(self._axes, bound)
                        if is_average:
                            avg_val = next((float(v) for v in y if isinstance(v, (int, float)) and math.isfinite(float(v))), None)
                            if avg_val is not None:
                                average_summaries.append(f"{label} = {self._fmt_num(avg_val)}")
                        else:
                            hi_set = {str(sn).strip() for sn in (self._highlight_sns or []) if str(sn).strip()}
                            for point in points:
                                if str(point.get("serial") or "").strip() in hi_set:
                                    self._axes.plot(
                                        [float(point.get("x") or 0.0)],
                                        [float(point.get("y") or 0.0)],
                                        linestyle="",
                                        marker="o",
                                        markersize=9,
                                        color=line.get_color(),
                                    )
                        any_plotted = True
                    except Exception:
                        continue
        if not any_plotted:
            QtWidgets.QMessageBox.information(self, "Plot Metrics", "No metric values found for this selection.")
            return
        self._apply_metric_program_segments(self._axes, labels)
        self._axes.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
        self._axes.set_xticks(x)
        self._axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        self._axes.grid(True, alpha=0.25)
        self._main_plot_legend_entries = self._apply_interactive_legend_policy(
            self._axes,
            overflow_button=getattr(self, "btn_plot_legend", None),
        )
        if average_summaries:
            prefix = "Average value: " if len(average_summaries) == 1 else "Average values: "
            self._set_plot_note(prefix + " | ".join(average_summaries))
        self._apply_plot_view_bands_to_axes(self._axes, mode="metrics")
        self._refresh_plot_note()
        try:
            self._figure.tight_layout()
        except Exception:
            pass
        try:
            self._canvas.draw()
        except Exception:
            pass
        self._capture_main_plot_base_view()
        self._update_perf_primary_equation_banner()
        self.btn_save_plot_pdf.setEnabled(True)
        selection_ids = [str(item.get("id") or "").strip() for item in selections if isinstance(item, dict) and str(item.get("id") or "").strip()]
        selection_labels = [self._selection_display_text(item) for item in selections if isinstance(item, dict) and self._selection_display_text(item)]
        run_conditions = [self._selection_condition_label(item) for item in selections if isinstance(item, dict) and self._selection_condition_label(item)]
        self._last_plot_def = {
            "mode": "metrics",
            "run": runs[0],
            "selector_mode": str(selection.get("mode") or "sequence"),
            "selection_id": str(selection.get("id") or ""),
            "selection_ids": list(selection_ids),
            "selection_labels": list(selection_labels),
            "display_text": self._selection_display_text(selection),
            "run_condition": self._selection_condition_label(selection),
            "run_conditions": list(run_conditions),
            "member_sequences": list(selection.get("member_sequences") or []),
            "member_runs": list(runs),
            "stats": list(stats),
            "y": list(y_cols),
            "plot_bounds": bool(plot_bounds),
            "metric_plot_source": metric_source,
        }
        self.btn_add_auto_plot.setEnabled(True)
        self._refresh_stats_preview()

    def _plot_curves(self) -> None:
        if not self._plot_ready or not self._db_path:
            return
        self._set_plot_note("")
        self._ensure_main_axes("2d")
        selection = self._current_run_selection()
        runs = self._current_member_runs()
        y_col = self._current_curve_y_name()
        x_key = self._current_curve_x_key()
        x_label = self._current_curve_x_label()
        if not runs or not y_col:
            return
        if not x_label:
            QtWidgets.QMessageBox.information(
                self,
                "Plot Curves",
                "No valid X column found.\n\nOnly Time / Pulse Number are supported for the X-axis (never excel_row).",
            )
            return

        self._axes.clear()
        self._axes.set_title(self._compose_run_title(selection, f"{y_col} vs {x_label}"))
        self._axes.set_xlabel(x_label)
        self._axes.set_ylabel(y_col)
        any_plotted = False
        multi_run = len(runs) > 1
        x_col_title = ""
        for run in runs:
            x_col = self._resolve_curve_x_key(run, x_key or x_label)
            if not x_col:
                continue
            if not x_col_title:
                x_col_title = x_col
            try:
                curves = self._load_curves_for_selection(run, y_col, x_col, selection=selection, serials=self._active_serials())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Plot Curves", str(exc))
                return
            for s in curves:
                sn = str(s.get("serial") or "").strip()
                xs = s.get("x") or []
                ys = s.get("y") or []
                if not isinstance(xs, list) or not isinstance(ys, list) or not xs or not ys:
                    continue
                try:
                    hi_set = {str(v).strip() for v in (self._highlight_sns or []) if str(v).strip()}
                    is_hi = bool(hi_set) and sn in hi_set
                    label = self._curve_trace_label(run, s, multi_run=multi_run)
                    self._axes.plot(
                        xs,
                        ys,
                        linewidth=(2.6 if is_hi else 1.1),
                        alpha=(1.0 if is_hi else 0.75),
                        label=label,
                    )
                    any_plotted = True
                except Exception:
                    continue
        if not any_plotted:
            QtWidgets.QMessageBox.information(self, "Plot Curves", "No curve data found for this selection.")
            return
        self._axes.grid(True, alpha=0.25)
        self._main_plot_legend_entries = self._apply_interactive_legend_policy(
            self._axes,
            overflow_button=getattr(self, "btn_plot_legend", None),
        )
        self._apply_plot_view_bands_to_axes(self._axes, mode="curves")
        try:
            self._figure.tight_layout()
        except Exception:
            pass
        try:
            self._canvas.draw()
        except Exception:
            pass
        self._capture_main_plot_base_view()
        self.btn_save_plot_pdf.setEnabled(True)
        self._last_plot_def = {
            "mode": "curves",
            "run": runs[0],
            "selector_mode": str(selection.get("mode") or "sequence"),
            "selection_id": str(selection.get("id") or ""),
            "display_text": self._selection_display_text(selection),
            "run_condition": self._selection_condition_label(selection),
            "member_sequences": list(selection.get("member_sequences") or []),
            "member_runs": list(runs),
            "x": (x_label or x_col_title),
            "y": [y_col],
        }
        self.btn_add_auto_plot.setEnabled(True)
        self._populate_stats_table(runs[0], y_col, self._highlight_sn)

    def _selected_perf_runs(self) -> list[str]:
        if not getattr(self, "_db_path", None):
            return []
        try:
            return be.td_list_runs(self._db_path)
        except Exception:
            return []

    def _selected_perf_serials(self) -> list[str]:
        if not getattr(self, "_db_path", None):
            return []
        return self._active_serials()

    def _selected_perf_filter_mode(self) -> str:
        if not hasattr(self, "cb_perf_filter_mode"):
            return "all_conditions"
        try:
            mode = str(self.cb_perf_filter_mode.currentData() or "").strip().lower()
        except Exception:
            mode = ""
        return mode if mode in {"all_conditions", "match_control_period"} else "all_conditions"

    def _selected_perf_run_type_mode(self) -> str:
        if hasattr(self, "rb_perf_pulsed_mode") and self.rb_perf_pulsed_mode.isChecked():
            return "pulsed_mode"
        return "steady_state"

    def _set_perf_run_type_mode(self, mode: object) -> None:
        normalized = be.td_perf_normalize_run_type_mode(mode)
        target = self.rb_perf_pulsed_mode if normalized == "pulsed_mode" else self.rb_perf_steady_state
        try:
            target.setChecked(True)
        except Exception:
            pass

    def _selected_perf_control_period(self) -> object | None:
        if (
            self._selected_perf_run_type_mode() != "pulsed_mode"
            or not hasattr(self, "cb_perf_control_period")
        ):
            return None
        try:
            value = self.cb_perf_control_period.currentData()
        except Exception:
            value = None
        if value in (None, ""):
            txt = str(self.cb_perf_control_period.currentText() or "").strip()
            return txt or None
        return value

    def _perf_normalize_plot_method(self, value: object) -> str:
        normalizer = getattr(be, "td_perf_normalize_plot_method", None)
        if callable(normalizer):
            try:
                return str(normalizer(value)).strip().lower()
            except Exception:
                pass
        raw = str(value or "").strip().lower()
        if raw == "cached_condition_means":
            return "cached_condition_means"
        return "legacy_serial_curves"

    def _perf_plot_method_label(self, value: object) -> str:
        labeler = getattr(be, "td_perf_plot_method_label", None)
        if callable(labeler):
            try:
                return str(labeler(value)).strip()
            except Exception:
                pass
        return "Run Conditions" if self._perf_normalize_plot_method(value) == "cached_condition_means" else "Legacy Serial Curves"

    def _perf_plot_method_file_slug(self, value: object) -> str:
        return "run_conditions" if self._perf_normalize_plot_method(value) == "cached_condition_means" else "legacy"

    def _perf_available_control_periods(self) -> list[object]:
        rows = [
            dict(row)
            for row in (getattr(self, "_global_filter_rows", None) or [])
            if isinstance(row, dict)
        ]
        if rows and callable(getattr(self, "_row_matches_global_filters", None)):
            values_by_label: dict[str, object] = {}
            for row in rows:
                if not self._row_matches_global_filters(row):
                    continue
                raw_value = row.get("control_period")
                label = _td_compact_filter_value(raw_value)
                if not label or label in values_by_label:
                    continue
                values_by_label[label] = raw_value
            if values_by_label:
                return [
                    values_by_label[label]
                    for label in sorted(values_by_label.keys(), key=_td_compact_filter_sort_key)
                ]
            return []
        if not getattr(self, "_db_path", None):
            return []
        try:
            return list(be.td_list_control_periods(self._db_path) or [])
        except Exception:
            return []

    def _refresh_perf_control_period_options(self) -> None:
        if not hasattr(self, "cb_perf_control_period"):
            return
        prev_control_period = self._selected_perf_control_period()
        control_periods = self._perf_available_control_periods()
        self.cb_perf_control_period.blockSignals(True)
        self.cb_perf_control_period.clear()
        for cp in control_periods:
            self.cb_perf_control_period.addItem(str(cp), cp)
        if prev_control_period not in (None, ""):
            match_idx = self.cb_perf_control_period.findData(prev_control_period)
            if match_idx >= 0:
                self.cb_perf_control_period.setCurrentIndex(match_idx)
        elif self.cb_perf_control_period.count() > 0:
            self.cb_perf_control_period.setCurrentIndex(0)
        self.cb_perf_control_period.blockSignals(False)

    def _update_perf_control_period_state(self) -> None:
        if not hasattr(self, "cb_perf_control_period"):
            return
        _output_name, _input1_name, input2_name = self._perf_var_names()
        is_surface = bool(str(input2_name or "").strip())
        cp_surface_family = str(getattr(be, "TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD", "quadratic_surface_control_period"))
        use_as_slice = (
            self._selected_perf_run_type_mode() == "pulsed_mode"
            and is_surface
            and self._perf_requested_surface_family() == cp_surface_family
        )
        enabled = (
            self._selected_perf_run_type_mode() == "pulsed_mode"
            and (self._selected_perf_filter_mode() == "match_control_period" or use_as_slice)
            and self.cb_perf_control_period.count() > 0
        )
        self.cb_perf_control_period.setEnabled(enabled)
        if use_as_slice and self._selected_perf_filter_mode() != "match_control_period":
            self.cb_perf_control_period.setToolTip("Select the displayed control-period slice for the control-period-aware surface.")
        elif enabled:
            self.cb_perf_control_period.setToolTip("Select the control period used to filter pulsed-mode data.")
        else:
            self.cb_perf_control_period.setToolTip("")

    @staticmethod
    def _perf_norm_name(value: object) -> str:
        return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())

    def _perf_plot_stat_candidates(self) -> list[str]:
        allowed = {"mean", "min", "max", "std"}
        stats = [
            str(s).strip().lower()
            for s in (getattr(self, "_perf_available_stats", []) or [])
            if str(s).strip().lower() in allowed
        ]
        out: list[str] = []
        for st in stats:
            if st not in out:
                out.append(st)
        if "mean" not in out:
            out.insert(0, "mean")
        return out or ["mean", "min", "max", "std"]

    def _perf_var_names(self) -> tuple[str, str, str]:
        output_name = self._perf_current_col_name(self.cb_perf_y_col) if hasattr(self, "cb_perf_y_col") else ""
        input1_name = self._perf_current_col_name(self.cb_perf_x_col) if hasattr(self, "cb_perf_x_col") else ""
        input2_name = self._perf_current_col_name(self.cb_perf_z_col) if hasattr(self, "cb_perf_z_col") else ""
        return output_name, input1_name, input2_name

    def _perf_axis_names(self) -> tuple[str, str]:
        x_name = self._perf_current_col_name(self.cb_perf_x_col) if hasattr(self, "cb_perf_x_col") else ""
        y_name = self._perf_current_col_name(self.cb_perf_y_col) if hasattr(self, "cb_perf_y_col") else ""
        return x_name, y_name

    def _perf_requested_fit_mode(self) -> str:
        if not hasattr(self, "cb_perf_fit_model"):
            return "auto"
        try:
            data = self.cb_perf_fit_model.currentData()
        except Exception:
            data = None
        normalize = getattr(be, "td_perf_normalize_fit_mode", None)
        if callable(normalize):
            try:
                return str(normalize(data if data is not None else self.cb_perf_fit_model.currentText())).strip().lower()
            except Exception:
                pass
        raw = str(data if data is not None else self.cb_perf_fit_model.currentText()).strip().lower()
        return raw or "auto"

    def _perf_requested_surface_family(self) -> str:
        if not hasattr(self, "cb_perf_surface_model"):
            return "auto_surface"
        try:
            data = self.cb_perf_surface_model.currentData()
        except Exception:
            data = None
        normalize = getattr(be, "td_perf_normalize_surface_family", None)
        if callable(normalize):
            try:
                return str(normalize(data if data is not None else self.cb_perf_surface_model.currentText())).strip().lower()
            except Exception:
                pass
        raw = str(data if data is not None else self.cb_perf_surface_model.currentText()).strip().lower()
        return raw or "auto_surface"

    def _perf_fit_family_label(self, family: object) -> str:
        labeler = getattr(be, "td_perf_fit_family_label", None)
        if callable(labeler):
            try:
                return str(labeler(family)).strip()
            except Exception:
                pass
        return str(family or "").strip()

    def _update_perf_fit_controls(self) -> None:
        fit_enabled = bool(getattr(self, "cb_perf_fit", None) and self.cb_perf_fit.isChecked())
        fit_mode = self._perf_requested_fit_mode()
        _output_name, _input1_name, input2_name = self._perf_var_names()
        is_surface = bool(str(input2_name or "").strip())
        if hasattr(self, "cb_perf_fit_model"):
            self.cb_perf_fit_model.setEnabled(fit_enabled and not is_surface)
            if is_surface:
                self.cb_perf_fit_model.setToolTip("2-variable fit family selector. 3-variable mode uses surface fitting.")
            else:
                self.cb_perf_fit_model.setToolTip("")
        if hasattr(self, "cb_perf_surface_model"):
            self.cb_perf_surface_model.setEnabled(fit_enabled and is_surface)
            if is_surface:
                self.cb_perf_surface_model.setToolTip("Surface family override for 3D performance plots, including the control-period-aware surface slice model.")
            else:
                self.cb_perf_surface_model.setToolTip("Enable Input 2 to use 3-variable surface fitting.")
        allow_poly_controls = fit_enabled and (not is_surface) and fit_mode == "polynomial"
        if hasattr(self, "sp_perf_degree"):
            self.sp_perf_degree.setEnabled(allow_poly_controls)
        if hasattr(self, "cb_perf_norm_x"):
            self.cb_perf_norm_x.setEnabled(allow_poly_controls)

    def _perf_build_master_aggregate_curve(self, curves: dict[str, list[tuple[float, float, str]]]) -> tuple[list[float], list[float], list[float]]:
        aggregator = getattr(be, "td_perf_build_aggregate_curve", None)
        if callable(aggregator):
            try:
                aggregate = aggregator(curves, max_bins=24, min_serials_per_bin=1, return_meta=True)
            except Exception:
                aggregate = {}
            if isinstance(aggregate, dict):
                xs = [float(v) for v in (aggregate.get("x") or [])]
                ys = [float(v) for v in (aggregate.get("y") or [])]
                support = [float(v) for v in (aggregate.get("serial_support") or [])]
                edge_weight = [float(v) for v in (aggregate.get("edge_weight") or [])]
                weights = [
                    float(max(1.0, math.sqrt(max(0.0, support[idx])))) * float(edge_weight[idx])
                    for idx in range(min(len(xs), len(support), len(edge_weight)))
                ]
                if len(xs) == len(ys) and len(xs) == len(weights) and xs:
                    return xs, ys, weights
        pooled_x: list[float] = []
        pooled_y: list[float] = []
        for pts in (curves or {}).values():
            for point in pts or []:
                if len(point) < 2:
                    continue
                pooled_x.append(float(point[0]))
                pooled_y.append(float(point[1]))
        return pooled_x, pooled_y, []

    def _perf_fit_model_for_points(self, xs: list[float], ys: list[float], sample_weights: list[float] | None = None) -> dict | None:
        fitter = getattr(be, "td_perf_fit_model", None)
        curve_builder = getattr(be, "td_perf_build_fit_curve", None)
        if not callable(fitter):
            return None
        fit_mode = self._perf_requested_fit_mode()
        degree = int(self.sp_perf_degree.value()) if hasattr(self, "sp_perf_degree") else 2
        normx = bool(getattr(self, "cb_perf_norm_x", None) and self.cb_perf_norm_x.isChecked())
        model = fitter(xs, ys, fit_mode=fit_mode, polynomial_degree=degree, normalize_x=normx, sample_weights=sample_weights)
        if not isinstance(model, dict):
            return None
        if callable(curve_builder) and xs:
            try:
                curve = curve_builder(model, min(xs), max(xs), points=220)
            except Exception:
                curve = {}
            if isinstance(curve, dict):
                model.update(curve)
        return model

    def _perf_fit_surface_for_points(
        self,
        x1s: list[float],
        x2s: list[float],
        ys: list[float],
        *,
        control_periods: list[float] | None = None,
        display_control_period: object = None,
    ) -> dict | None:
        fitter = getattr(be, "td_perf_fit_surface_model", None)
        grid_builder = getattr(be, "td_perf_build_surface_grid", None)
        if not callable(fitter):
            return None
        surface_family = self._perf_requested_surface_family()
        model = fitter(
            x1s,
            x2s,
            ys,
            auto_surface_families=(surface_family == "auto_surface"),
            surface_family=surface_family,
            control_periods=control_periods,
        )
        if not isinstance(model, dict):
            return None
        if callable(grid_builder) and x1s and x2s:
            try:
                grid = grid_builder(
                    model,
                    min(x1s),
                    max(x1s),
                    min(x2s),
                    max(x2s),
                    points=22,
                    control_period=display_control_period,
                )
            except Exception:
                grid = {}
            if isinstance(grid, dict):
                model.update(grid)
        cp_surface_family = str(getattr(be, "TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD", "quadratic_surface_control_period"))
        if str(model.get("fit_family") or "").strip().lower() == cp_surface_family:
            domain_warning_fn = getattr(be, "_td_perf_surface_control_period_out_of_domain_warning", None)
            append_warning_fn = getattr(be, "_td_perf_append_fit_warning", None)
            if callable(domain_warning_fn):
                domain_warning = str(domain_warning_fn(model, display_control_period) or "").strip()
                if domain_warning:
                    if callable(append_warning_fn):
                        append_warning_fn(model, domain_warning)
                    else:
                        existing = str(model.get("fit_warning_text") or "").strip()
                        model["fit_warning_text"] = "\n".join([part for part in [existing, domain_warning] if part])
        return model

    def _perf_current_col_name(self, cb: QtWidgets.QComboBox) -> str:
        try:
            d = cb.currentData()
        except Exception:
            d = None
        if isinstance(d, str):
            if not d.strip():
                return ""
            return d.strip()
        txt = str(cb.currentText() or "").strip()
        if not txt or txt.strip().lower() in {"none", "(none)"}:
            return ""
        if txt.endswith(")") and " (" in txt:
            txt = txt.split(" (", 1)[0].strip()
        return txt

    def _perf_checked_stats(self) -> list[str]:
        if not hasattr(self, "list_perf_stats"):
            return []
        out: list[str] = []
        for i in range(self.list_perf_stats.count()):
            it = self.list_perf_stats.item(i)
            if it and it.checkState() == QtCore.Qt.CheckState.Checked:
                st = it.text().strip().lower()
                if st:
                    out.append(st)
        return out

    def _sync_perf_view_stats(self) -> None:
        if not hasattr(self, "cb_perf_view_stat") or not hasattr(self, "list_perf_stats"):
            return
        checked = self._perf_checked_stats()
        prev = str(self.cb_perf_view_stat.currentText() or "").strip().lower()
        self.cb_perf_view_stat.blockSignals(True)
        self.cb_perf_view_stat.clear()
        for st in checked:
            self.cb_perf_view_stat.addItem(st, st)
        if prev and prev in checked:
            self.cb_perf_view_stat.setCurrentText(prev)
        elif checked:
            self.cb_perf_view_stat.setCurrentIndex(0)
        self.cb_perf_view_stat.blockSignals(False)

    def _fill_perf_axis_combo(
        self,
        cb: QtWidgets.QComboBox,
        *,
        allowed_norms: set[str] | None = None,
        allow_blank: bool = False,
    ) -> None:
        prev = self._perf_current_col_name(cb)
        prev_norm = self._perf_norm_name(prev)
        blocker = QtCore.QSignalBlocker(cb)
        cb.clear()
        if allow_blank:
            cb.addItem("None", "")
        for col in (getattr(self, "_perf_available_columns", []) or []):
            nm = str((col or {}).get("name") or "").strip()
            if not nm:
                continue
            nk = self._perf_norm_name(nm)
            if allowed_norms is not None and nk not in allowed_norms:
                continue
            units = str((col or {}).get("units") or "").strip()
            label = f"{nm} ({units})" if units else nm
            cb.addItem(label, nm)
        if prev_norm:
            for i in range(cb.count()):
                if self._perf_norm_name(cb.itemData(i)) == prev_norm:
                    cb.setCurrentIndex(i)
                    break
        del blocker

    def _set_perf_axis_combo_by_norm(
        self,
        cb: QtWidgets.QComboBox,
        want_norm: str,
        *,
        allow_blank: bool = False,
        disallow_norms: set[str] | None = None,
    ) -> bool:
        blocked = {norm for norm in (disallow_norms or set()) if norm}
        blocker = QtCore.QSignalBlocker(cb)
        try:
            if want_norm:
                for i in range(cb.count()):
                    item_norm = self._perf_norm_name(cb.itemData(i))
                    if item_norm and item_norm == want_norm and item_norm not in blocked:
                        cb.setCurrentIndex(i)
                        return True
            if allow_blank and cb.count() > 0:
                cb.setCurrentIndex(0)
                return True
            for i in range(cb.count()):
                item_norm = self._perf_norm_name(cb.itemData(i))
                if not item_norm or item_norm in blocked:
                    continue
                cb.setCurrentIndex(i)
                return True
            if cb.count() > 0:
                cb.setCurrentIndex(0)
                return True
            return False
        finally:
            del blocker

    def _common_runs_for_perf_pair(self, x_name: str, y_name: str) -> list[str]:
        x_norm = self._perf_norm_name(x_name)
        y_norm = self._perf_norm_name(y_name)
        if not x_norm or not y_norm:
            return []
        x_runs = set((getattr(self, "_perf_col_runs", {}) or {}).get(x_norm) or set())
        y_runs = set((getattr(self, "_perf_col_runs", {}) or {}).get(y_norm) or set())
        common = x_runs & y_runs
        order = {r: i for i, r in enumerate(getattr(self, "_perf_all_runs", []) or [])}
        return sorted(common, key=lambda r: order.get(r, 10**9))

    def _update_perf_axes_label(self) -> None:
        if not hasattr(self, "lbl_perf_axes"):
            return
        output_name, input1_name, input2_name = self._perf_var_names()
        self.lbl_perf_axes.setText(
            f"Output: {output_name or '-'} | Input 1: {input1_name or '-'} | Input 2: {input2_name or '-'}"
        )

    def _update_perf_pair_summary(
        self,
        *,
        stats: list[str] | None = None,
        qualifying_serial_count: int | None = None,
        require_min_points: int | None = None,
        plot_dimension: str | None = None,
    ) -> None:
        if not hasattr(self, "lbl_perf_common_runs"):
            return
        output_name, input1_name, input2_name = self._perf_var_names()
        self._update_perf_axes_label()
        if not output_name or not input1_name:
            self.lbl_perf_common_runs.setText("Common runs for selected variables: -")
            return
        chosen = [nm for nm in (output_name, input1_name, input2_name) if str(nm).strip()]
        if len({self._perf_norm_name(v) for v in chosen}) != len(chosen):
            self.lbl_perf_common_runs.setText("Common runs for selected variables: - (all selected variables must be different)")
            return
        common = self._common_runs_for_perf_vars(output_name, input1_name, input2_name)
        if not common:
            self.lbl_perf_common_runs.setText("Common runs for selected variables: 0 (no run contains all selected variables)")
            return
        shown = ", ".join(common[:4])
        extra = f" (+{len(common) - 4})" if len(common) > 4 else ""
        mode_label = "surface" if str(plot_dimension or "").strip().lower() == "3d" or str(input2_name or "").strip() else "curve"
        text = f"Common runs: {len(common)} - {shown}{extra} | Mode: {mode_label}"
        if stats:
            text += f" | Plot stats: {', '.join(stats)}"
        if qualifying_serial_count is not None:
            text += f" | Qualifying serials: {qualifying_serial_count}"
        if require_min_points is not None:
            text += f" | Min points per serial: {int(require_min_points)}"
        self.lbl_perf_common_runs.setText(text)

    def _common_runs_for_perf_vars(self, output_name: str, input1_name: str, input2_name: str = "") -> list[str]:
        names = [str(v).strip() for v in (output_name, input1_name, input2_name) if str(v).strip()]
        if len(names) < 2:
            return []
        sets: list[set[str]] = []
        for name in names:
            key = self._perf_norm_name(name)
            sets.append(set((getattr(self, "_perf_col_runs", {}) or {}).get(key) or set()))
        common = sets[0]
        for s in sets[1:]:
            common = common & s
        order = {r: i for i, r in enumerate(getattr(self, "_perf_all_runs", []) or [])}
        return sorted(common, key=lambda r: order.get(r, 10**9))

    def _filter_perf_axis_options(self, *, changed: str | None = None) -> None:
        if not hasattr(self, "cb_perf_x_col") or not hasattr(self, "cb_perf_y_col"):
            return
        combos = {
            "x": self.cb_perf_x_col,
            "y": self.cb_perf_y_col,
        }
        if hasattr(self, "cb_perf_z_col"):
            combos["z"] = self.cb_perf_z_col
        if changed not in combos:
            return
        if getattr(self, "_perf_axis_update_in_progress", False):
            return
        self._perf_axis_update_in_progress = True
        try:
            pivot_cb = combos[changed]
            pivot = self._perf_current_col_name(pivot_cb)
            pivot_norm = self._perf_norm_name(pivot)
            pivot_runs = set((getattr(self, "_perf_col_runs", {}) or {}).get(pivot_norm) or set())
            selected_norms = {
                key: self._perf_norm_name(self._perf_current_col_name(cb))
                for key, cb in combos.items()
            }
            for key, cb in combos.items():
                if key == changed:
                    continue
                desired_norm = selected_norms.get(key, "")
                disallow = {v for k2, v in selected_norms.items() if k2 != key and v}
                allowed_norms = {
                    norm_name
                    for norm_name, runs in (getattr(self, "_perf_col_runs", {}) or {}).items()
                    if norm_name not in disallow and (not pivot_runs or (set(runs or set()) & pivot_runs))
                }
                if not allowed_norms and getattr(self, "_perf_available_columns", None):
                    allowed_norms = {
                        self._perf_norm_name(str((col or {}).get("name") or "").strip())
                        for col in (self._perf_available_columns or [])
                        if self._perf_norm_name(str((col or {}).get("name") or "").strip()) not in disallow
                    }
                self._fill_perf_axis_combo(cb, allowed_norms=allowed_norms, allow_blank=(key == "z"))
                self._set_perf_axis_combo_by_norm(
                    cb,
                    desired_norm,
                    allow_blank=(key == "z"),
                    disallow_norms=disallow,
                )
                selected_norms[key] = self._perf_norm_name(self._perf_current_col_name(cb))
        finally:
            self._perf_axis_update_in_progress = False

    def _on_perf_axis_changed(self, changed: str | None = None) -> None:
        if changed in {"x", "y", "z"}:
            if getattr(self, "_perf_axis_update_in_progress", False):
                return
            self._filter_perf_axis_options(changed=changed)
        self._update_perf_fit_controls()
        self._update_perf_control_period_state()
        self._update_perf_pair_summary()
        self._clear_perf_results()

    def _populate_perf_stats(self, stats: list[str]) -> None:
        if not hasattr(self, "list_perf_stats"):
            return
        self.list_perf_stats.blockSignals(True)
        self.list_perf_stats.clear()
        for st in stats:
            item = QtWidgets.QListWidgetItem(st)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list_perf_stats.addItem(item)
        self.list_perf_stats.blockSignals(False)
        self.list_perf_stats.setEnabled(False)

    def _clear_perf_results(self) -> None:
        self._perf_results_by_stat = {}
        self._perf_plot_view_stats = []
        self._main_plot_legend_entries = []
        if hasattr(self, "btn_plot_legend"):
            self.btn_plot_legend.setVisible(False)
            self.btn_plot_legend.setEnabled(False)
        if hasattr(self, "tbl_perf_equations"):
            try:
                self.tbl_perf_equations.setRowCount(0)
            except Exception:
                pass
        if hasattr(self, "cb_perf_view_stat"):
            try:
                self.cb_perf_view_stat.blockSignals(True)
                self.cb_perf_view_stat.clear()
                self.cb_perf_view_stat.blockSignals(False)
            except Exception:
                pass
        self._update_perf_export_button_state()
        self._update_perf_primary_equation_banner()

    def _set_combo_to_value(self, cb: QtWidgets.QComboBox, value: str) -> bool:
        want = str(value or "").strip()
        blocker = QtCore.QSignalBlocker(cb)
        if not want:
            if cb.count() > 0:
                cb.setCurrentIndex(0)
            return True
        want_norm = self._perf_norm_name(want)
        for i in range(cb.count()):
            item_data = cb.itemData(i)
            item_text = cb.itemText(i)
            if self._perf_norm_name(item_data if item_data is not None else item_text) == want_norm:
                cb.setCurrentIndex(i)
                return True
        return False

    def _perf_build_regression_checker_row(
        self,
        *,
        observation_id: str,
        run_name_value: str,
        run_display: str,
        row_input1: dict,
        row_output: dict,
        row_input2: dict | None = None,
    ) -> dict[str, object]:
        row_input1 = dict(row_input1 or {})
        row_output = dict(row_output or {})
        row_input2 = dict(row_input2 or {})
        base_row = row_output or row_input1 or row_input2
        labeler = getattr(be, "_td_perf_export_condition_label", None)
        if callable(labeler):
            try:
                condition_label = str(labeler(base_row, display_name=run_display) or "").strip()
            except Exception:
                condition_label = str(run_display or run_name_value).strip()
        else:
            condition_label = str(run_display or run_name_value).strip()
        return {
            "observation_id": str(observation_id or "").strip(),
            "run_name": str(run_name_value or "").strip(),
            "display_name": str(run_display or run_name_value).strip(),
            "program_title": str(
                base_row.get("program_title")
                or row_input1.get("program_title")
                or row_input2.get("program_title")
                or ""
            ).strip(),
            "source_run_name": str(
                base_row.get("source_run_name")
                or row_input1.get("source_run_name")
                or row_input2.get("source_run_name")
                or ""
            ).strip(),
            "control_period": base_row.get(
                "control_period",
                row_input1.get("control_period", row_input2.get("control_period")),
            ),
            "suppression_voltage": base_row.get(
                "suppression_voltage",
                row_input1.get("suppression_voltage", row_input2.get("suppression_voltage")),
            ),
            "condition_label": condition_label,
            "serial": str(
                base_row.get("serial") or row_input1.get("serial") or row_input2.get("serial") or ""
            ).strip(),
            "input_1": float(row_input1.get("value_num") or 0.0),
            "input_2": (
                float(row_input2.get("value_num") or 0.0)
                if row_input2 and isinstance(row_input2.get("value_num"), (int, float))
                else None
            ),
            "actual_mean": float(row_output.get("value_num") or 0.0),
            "sample_count": 1,
        }

    def _perf_collect_results(
        self,
        output_name: str,
        input1_name: str,
        input2_name: str,
        plot_stats: list[str],
        runs: list[str],
        serials: list[str],
        *,
        fit_enabled: bool,
        require_min_points: int,
        control_period_filter: object = None,
        display_control_period: object = None,
        run_type_filter: object = None,
    ) -> tuple[dict[str, dict], list[str], str]:
        is_surface = bool(str(input2_name or "").strip())
        surface_family = self._perf_requested_surface_family() if is_surface else ""
        cp_surface_family = str(getattr(be, "TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD", "quadratic_surface_control_period"))
        use_cp_surface = bool(is_surface and surface_family == cp_surface_family)
        equation_stats = list(plot_stats)
        for st in ("min_3sigma", "max_3sigma"):
            if st not in equation_stats:
                equation_stats.append(st)

        results: dict[str, dict] = {}
        plot_view_stats: list[str] = []
        fit_error_text = ""
        pair_label = (
            f"{output_name} vs {input1_name},{input2_name}"
            if is_surface
            else f"{output_name} vs {input1_name}"
        )
        serial_set = {str(sn).strip() for sn in serials if str(sn).strip()}

        def _series_by_observation(rows: list[dict]) -> dict[str, dict]:
            out: dict[str, dict] = {}
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                obs_id = str(row.get("observation_id") or "").strip()
                sn = str(row.get("serial") or "").strip()
                val = row.get("value_num")
                if (
                    not obs_id
                    or not sn
                    or sn not in serial_set
                    or not isinstance(val, (int, float))
                    or not math.isfinite(float(val))
                ):
                    continue
                out[obs_id] = dict(row)
            return out

        def _obs_label(run_display: str, run_name_value: str, row: dict) -> str:
            parts: list[str] = [str(run_display or run_name_value).strip() or str(run_name_value or "").strip()]
            for key in ("program_title", "source_run_name"):
                value = str((row or {}).get(key) or "").strip()
                if value and value not in parts:
                    parts.append(value)
            return " | ".join([part for part in parts if str(part).strip()])

        def _cp_matches(value: object, target: object) -> bool:
            if target in (None, ""):
                return True
            try:
                return abs(float(value) - float(target)) <= 1e-9
            except Exception:
                return str(value or "").strip().lower() == str(target or "").strip().lower()

        for st in equation_stats:
            if is_surface:
                per_run: list[tuple[str, str, dict[str, dict], dict[str, dict], dict[str, dict], str, str, str]] = []
                for rn in runs:
                    input1_col, input1_units = self._resolve_td_y_col_units(rn, input1_name)
                    input2_col, input2_units = self._resolve_td_y_col_units(rn, input2_name)
                    output_col, output_units = self._resolve_td_y_col_units(rn, output_name)
                    if not input1_col or not input2_col or not output_col:
                        continue
                    input1_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            input1_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    input2_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            input2_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    output_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            output_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    if not input1_map or not input2_map or not output_map:
                        continue
                    per_run.append(
                        (
                            rn,
                            self._run_display_text(rn),
                            input1_map,
                            input2_map,
                            output_map,
                            input1_units,
                            input2_units,
                            output_units,
                        )
                    )

                points_3d: dict[str, list[tuple[float, float, float, str]]] = {}
                checker_rows_by_serial: dict[str, list[dict]] = {}
                pooled_x1: list[float] = []
                pooled_x2: list[float] = []
                pooled_y: list[float] = []
                pooled_cp: list[float] = []
                invalid_control_period_seen = False
                min_surface_points = max(3, int(require_min_points))
                for sn in serials:
                    pts_all: list[tuple[float, float, float, str, object]] = []
                    visible_rows: list[dict] = []
                    for _rn, rdisp, input1_map, input2_map, output_map, _u1, _u2, _uy in per_run:
                        obs_ids = sorted(set(input1_map.keys()) & set(input2_map.keys()) & set(output_map.keys()))
                        for obs_id in obs_ids:
                            row_input1 = input1_map.get(obs_id) or {}
                            row_input2 = input2_map.get(obs_id) or {}
                            row_output = output_map.get(obs_id) or {}
                            if str(row_output.get("serial") or row_input1.get("serial") or "").strip() != sn:
                                continue
                            cp_value = row_output.get("control_period", row_input1.get("control_period", row_input2.get("control_period")))
                            cp_numeric = None
                            if use_cp_surface:
                                try:
                                    cp_numeric = float(cp_value)
                                    if not math.isfinite(cp_numeric):
                                        raise ValueError
                                except Exception:
                                    invalid_control_period_seen = True
                            point = (
                                float(row_input1.get("value_num") or 0.0),
                                float(row_input2.get("value_num") or 0.0),
                                float(row_output.get("value_num") or 0.0),
                                _obs_label(rdisp, _rn, row_output or row_input1 or row_input2),
                                cp_numeric if cp_numeric is not None else cp_value,
                            )
                            pts_all.append(point)
                            if not use_cp_surface or _cp_matches(point[4], display_control_period):
                                visible_rows.append(
                                    self._perf_build_regression_checker_row(
                                        observation_id=obs_id,
                                        run_name_value=_rn,
                                        run_display=rdisp,
                                        row_input1=row_input1,
                                        row_output=row_output,
                                        row_input2=row_input2,
                                    )
                                )
                    pts_slice = [
                        (point[0], point[1], point[2], point[3])
                        for point in pts_all
                        if not use_cp_surface or _cp_matches(point[4], display_control_period)
                    ]
                    distinct_x1 = {round(p[0], 12) for p in pts_all}
                    distinct_x2 = {round(p[1], 12) for p in pts_all}
                    if len(pts_all) >= min_surface_points and len(distinct_x1) >= 2 and len(distinct_x2) >= 2:
                        pts_slice.sort(key=lambda t: (t[0], t[1], t[3]))
                        if pts_slice:
                            points_3d[sn] = pts_slice
                            checker_rows_by_serial[sn] = sorted(
                                visible_rows,
                                key=lambda row: (
                                    float(row.get("input_1") or 0.0),
                                    float(row.get("input_2") or 0.0)
                                    if row.get("input_2") not in (None, "")
                                    else float("-inf"),
                                    str(row.get("condition_label") or "").lower(),
                                    str(row.get("observation_id") or "").lower(),
                                ),
                            )
                        pooled_x1.extend([p[0] for p in pts_all])
                        pooled_x2.extend([p[1] for p in pts_all])
                        pooled_y.extend([p[2] for p in pts_all])
                        if use_cp_surface:
                            for point in pts_all:
                                if isinstance(point[4], (int, float)):
                                    pooled_cp.append(float(point[4]))

                if not points_3d:
                    continue

                input1_units = next((u for *_rest, u, _, _ in per_run if str(u).strip()), "")
                input2_units = next((u for *_rest, _, u, _ in per_run if str(u).strip()), "")
                output_units = next((u for *_rest, u in per_run if str(u).strip()), "")

                master_model: dict = {}
                if fit_enabled and len(pooled_y) >= 3:
                    try:
                        if use_cp_surface and run_type_filter != "pulsed_mode":
                            raise RuntimeError("Quadratic Surface + Control Period requires pulsed-mode data.")
                        if use_cp_surface and invalid_control_period_seen:
                            raise RuntimeError("Quadratic Surface + Control Period requires usable control-period values for all fitted pulsed-mode points.")
                        cp_support = {}
                        if use_cp_surface:
                            support_fn = getattr(be, "_td_perf_analyze_quadratic_surface_control_period_fit_support", None)
                            if callable(support_fn):
                                cp_support = support_fn(
                                    pooled_x1,
                                    pooled_x2,
                                    pooled_y,
                                    pooled_cp,
                                    fit_mode=(
                                        str(getattr(be, "TD_PERF_FIT_MODE_AUTO_SURFACE", "auto_surface"))
                                        if surface_family == str(getattr(be, "TD_PERF_FIT_MODE_AUTO_SURFACE", "auto_surface"))
                                        else str(getattr(be, "TD_PERF_FIT_MODE_POLYNOMIAL_SURFACE", "polynomial_surface"))
                                    ),
                                )
                        model = self._perf_fit_surface_for_points(
                            pooled_x1,
                            pooled_x2,
                            pooled_y,
                            control_periods=(pooled_cp if use_cp_surface else None),
                            display_control_period=display_control_period,
                        )
                        if use_cp_surface and not isinstance(model, dict):
                            failure_fn = getattr(be, "_td_perf_format_surface_control_period_failure", None)
                            if callable(failure_fn):
                                raise RuntimeError(
                                    failure_fn(
                                        cp_support.get("ignored_control_periods") or [],
                                        eligible_count=len(cp_support.get("eligible_control_period_values") or []),
                                    )
                                )
                            raise RuntimeError("Quadratic Surface + Control Period requires at least two distinct control periods with valid surface coverage.")
                    except Exception as exc:
                        if not fit_error_text:
                            fit_error_text = str(exc)
                        model = None
                    if isinstance(model, dict):
                        master_model = model

                results[st] = {
                    "pair_label": pair_label,
                    "output_target": output_name,
                    "input1_target": input1_name,
                    "input2_target": input2_name,
                    "x_target": input1_name,
                    "y_target": output_name,
                    "input1_units": input1_units,
                    "input2_units": input2_units,
                    "output_units": output_units,
                    "stat_label": st,
                    "fit_mode": self._perf_requested_fit_mode(),
                    "plot_dimension": "3d",
                    "surface_fit_family": surface_family,
                    "selected_control_period": display_control_period,
                    "performance_plot_method": "legacy_serial_curves",
                    "points_3d": points_3d,
                    "regression_checker_rows": [
                        dict(row)
                        for serial in serials
                        for row in (checker_rows_by_serial.get(serial) or [])
                    ],
                    "master_model": master_model,
                    "fit_warning_text": str(master_model.get("fit_warning_text") or "").strip(),
                    "highlight_serial": "",
                    "highlight_model": {},
                }
            else:
                per_run_2d: list[tuple[str, str, dict[str, dict], dict[str, dict], str, str]] = []
                for rn in runs:
                    input1_col, input1_units = self._resolve_td_y_col_units(rn, input1_name)
                    output_col, output_units = self._resolve_td_y_col_units(rn, output_name)
                    if not input1_col or not output_col:
                        continue
                    input1_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            input1_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    output_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            output_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    if not input1_map or not output_map:
                        continue
                    per_run_2d.append((rn, self._run_display_text(rn), input1_map, output_map, input1_units, output_units))

                curves: dict[str, list[tuple[float, float, str]]] = {}
                checker_rows_by_serial: dict[str, list[dict]] = {}
                for sn in serials:
                    pts_2d: list[tuple[float, float, str]] = []
                    visible_rows: list[dict] = []
                    for _rn, rdisp, input1_map, output_map, _xu, _yu in per_run_2d:
                        obs_ids = sorted(set(input1_map.keys()) & set(output_map.keys()))
                        for obs_id in obs_ids:
                            row_input1 = input1_map.get(obs_id) or {}
                            row_output = output_map.get(obs_id) or {}
                            if str(row_output.get("serial") or row_input1.get("serial") or "").strip() != sn:
                                continue
                            pts_2d.append(
                                (
                                    float(row_input1.get("value_num") or 0.0),
                                    float(row_output.get("value_num") or 0.0),
                                    _obs_label(rdisp, _rn, row_output or row_input1),
                                )
                            )
                            visible_rows.append(
                                self._perf_build_regression_checker_row(
                                    observation_id=obs_id,
                                    run_name_value=_rn,
                                    run_display=rdisp,
                                    row_input1=row_input1,
                                    row_output=row_output,
                                )
                            )
                    if len(pts_2d) >= require_min_points:
                        pts_2d.sort(key=lambda t: t[0])
                        curves[sn] = pts_2d
                        checker_rows_by_serial[sn] = sorted(
                            visible_rows,
                            key=lambda row: (
                                float(row.get("input_1") or 0.0),
                                str(row.get("condition_label") or "").lower(),
                                str(row.get("observation_id") or "").lower(),
                            ),
                        )

                if not curves:
                    continue

                input1_units = next((u for *_rest, u, _ in per_run_2d if str(u).strip()), "")
                output_units = next((u for *_rest, _, u in per_run_2d if str(u).strip()), "")

                master_model = {}
                aggregate_x, aggregate_y, aggregate_weights = self._perf_build_master_aggregate_curve(curves)
                if fit_enabled and aggregate_x:
                    try:
                        model = self._perf_fit_model_for_points(aggregate_x, aggregate_y, sample_weights=aggregate_weights)
                    except Exception as exc:
                        if not fit_error_text:
                            fit_error_text = str(exc)
                        model = None
                    if isinstance(model, dict):
                        master_model = model

                results[st] = {
                    "pair_label": pair_label,
                    "output_target": output_name,
                    "input1_target": input1_name,
                    "input2_target": "",
                    "x_target": input1_name,
                    "y_target": output_name,
                    "x_units": input1_units,
                    "y_units": output_units,
                    "stat_label": st,
                    "fit_mode": self._perf_requested_fit_mode(),
                    "plot_dimension": "2d",
                    "performance_plot_method": "legacy_serial_curves",
                    "curves": curves,
                    "regression_checker_rows": [
                        dict(row)
                        for serial in serials
                        for row in (checker_rows_by_serial.get(serial) or [])
                    ],
                    "master_model": master_model,
                    "highlight_serial": "",
                    "highlight_model": {},
                }

            if st in plot_stats:
                plot_view_stats.append(st)

        return results, plot_view_stats, fit_error_text

    def _perf_collect_cached_condition_mean_results(
        self,
        output_name: str,
        input1_name: str,
        input2_name: str,
        plot_stats: list[str],
        runs: list[str],
        serials: list[str],
        *,
        fit_enabled: bool,
        require_min_points: int,
        control_period_filter: object = None,
        display_control_period: object = None,
        run_type_filter: object = None,
    ) -> tuple[dict[str, dict], list[str], str]:
        is_surface = bool(str(input2_name or "").strip())
        surface_family = self._perf_requested_surface_family() if is_surface else ""
        cp_surface_family = str(
            getattr(be, "TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD", "quadratic_surface_control_period")
        )
        use_cp_surface = bool(is_surface and surface_family == cp_surface_family)
        equation_stats = list(plot_stats)
        for st in ("min_3sigma", "max_3sigma"):
            if st not in equation_stats:
                equation_stats.append(st)

        results: dict[str, dict] = {}
        plot_view_stats: list[str] = []
        fit_error_text = ""
        pair_label = (
            f"{output_name} vs {input1_name},{input2_name}"
            if is_surface
            else f"{output_name} vs {input1_name}"
        )
        serial_set = {str(sn).strip() for sn in serials if str(sn).strip()}

        def _series_by_observation(rows: list[dict]) -> dict[str, dict]:
            out: dict[str, dict] = {}
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                obs_id = str(row.get("observation_id") or "").strip()
                sn = str(row.get("serial") or "").strip()
                val = row.get("value_num")
                if (
                    not obs_id
                    or not sn
                    or sn not in serial_set
                    or not isinstance(val, (int, float))
                    or not math.isfinite(float(val))
                ):
                    continue
                out[obs_id] = dict(row)
            return out

        def _obs_label(run_display: str, run_name_value: str, row: dict) -> str:
            parts: list[str] = [str(run_display or run_name_value).strip() or str(run_name_value or "").strip()]
            for key in ("program_title", "source_run_name"):
                value = str((row or {}).get(key) or "").strip()
                if value and value not in parts:
                    parts.append(value)
            return " | ".join([part for part in parts if str(part).strip()])

        def _cp_matches(value: object, target: object) -> bool:
            if target in (None, ""):
                return True
            try:
                return abs(float(value) - float(target)) <= 1e-9
            except Exception:
                return str(value or "").strip().lower() == str(target or "").strip().lower()

        for st in equation_stats:
            if is_surface:
                per_run: list[tuple[str, str, dict[str, dict], dict[str, dict], dict[str, dict], str, str, str]] = []
                for rn in runs:
                    input1_col, input1_units = self._resolve_td_y_col_units(rn, input1_name)
                    input2_col, input2_units = self._resolve_td_y_col_units(rn, input2_name)
                    output_col, output_units = self._resolve_td_y_col_units(rn, output_name)
                    if not input1_col or not input2_col or not output_col:
                        continue
                    input1_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            input1_col,
                            "mean",
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    input2_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            input2_col,
                            "mean",
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    output_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            output_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    if not input1_map or not input2_map or not output_map:
                        continue
                    per_run.append(
                        (
                            rn,
                            self._run_display_text(rn),
                            input1_map,
                            input2_map,
                            output_map,
                            input1_units,
                            input2_units,
                            output_units,
                        )
                    )

                points_3d: dict[str, list[tuple[float, float, float, str]]] = {}
                checker_rows_by_serial: dict[str, list[dict]] = {}
                pooled_x1: list[float] = []
                pooled_x2: list[float] = []
                pooled_y: list[float] = []
                pooled_cp: list[float] = []
                invalid_control_period_seen = False
                min_surface_points = max(3, int(require_min_points))
                for sn in serials:
                    pts_all: list[tuple[float, float, float, str, object]] = []
                    pts_slice: list[tuple[float, float, float, str]] = []
                    visible_rows: list[dict] = []
                    for _rn, rdisp, input1_map, input2_map, output_map, _u1, _u2, _uy in per_run:
                        obs_ids = sorted(set(input1_map.keys()) & set(input2_map.keys()) & set(output_map.keys()))
                        for obs_id in obs_ids:
                            row_input1 = input1_map.get(obs_id) or {}
                            row_input2 = input2_map.get(obs_id) or {}
                            row_output = output_map.get(obs_id) or {}
                            if str(row_output.get("serial") or row_input1.get("serial") or "").strip() != sn:
                                continue
                            cp_value = row_output.get(
                                "control_period",
                                row_input1.get("control_period", row_input2.get("control_period")),
                            )
                            cp_numeric = None
                            if use_cp_surface:
                                try:
                                    cp_numeric = float(cp_value)
                                    if not math.isfinite(cp_numeric):
                                        raise ValueError
                                except Exception:
                                    invalid_control_period_seen = True
                            point = (
                                float(row_input1.get("value_num") or 0.0),
                                float(row_input2.get("value_num") or 0.0),
                                float(row_output.get("value_num") or 0.0),
                                _obs_label(rdisp, _rn, row_output or row_input1 or row_input2),
                                cp_numeric if cp_numeric is not None else cp_value,
                            )
                            pts_all.append(point)
                            if not use_cp_surface or _cp_matches(point[4], display_control_period):
                                pts_slice.append((point[0], point[1], point[2], point[3]))
                                visible_rows.append(
                                    self._perf_build_regression_checker_row(
                                        observation_id=obs_id,
                                        run_name_value=_rn,
                                        run_display=rdisp,
                                        row_input1=row_input1,
                                        row_output=row_output,
                                        row_input2=row_input2,
                                    )
                                )
                    distinct_x1 = {round(p[0], 12) for p in pts_all}
                    distinct_x2 = {round(p[1], 12) for p in pts_all}
                    if len(pts_all) >= min_surface_points and len(distinct_x1) >= 2 and len(distinct_x2) >= 2:
                        pts_slice.sort(key=lambda t: (t[0], t[1], t[3]))
                        if pts_slice:
                            points_3d[sn] = pts_slice
                            checker_rows_by_serial[sn] = sorted(
                                visible_rows,
                                key=lambda row: (
                                    float(row.get("input_1") or 0.0),
                                    float(row.get("input_2") or 0.0)
                                    if row.get("input_2") not in (None, "")
                                    else float("-inf"),
                                    str(row.get("condition_label") or "").lower(),
                                    str(row.get("observation_id") or "").lower(),
                                ),
                            )
                        pooled_x1.extend([p[0] for p in pts_all])
                        pooled_x2.extend([p[1] for p in pts_all])
                        pooled_y.extend([p[2] for p in pts_all])
                        if use_cp_surface:
                            for point in pts_all:
                                if isinstance(point[4], (int, float)):
                                    pooled_cp.append(float(point[4]))

                if not points_3d:
                    continue

                input1_units = next((u for *_rest, u, _, _ in per_run if str(u).strip()), "")
                input2_units = next((u for *_rest, _, u, _ in per_run if str(u).strip()), "")
                output_units = next((u for *_rest, u in per_run if str(u).strip()), "")

                master_model: dict = {}
                if fit_enabled and len(pooled_y) >= 3:
                    try:
                        if use_cp_surface and run_type_filter != "pulsed_mode":
                            raise RuntimeError("Quadratic Surface + Control Period requires pulsed-mode data.")
                        if use_cp_surface and invalid_control_period_seen:
                            raise RuntimeError(
                                "Quadratic Surface + Control Period requires usable control-period values for all fitted pulsed-mode points."
                            )
                        cp_support = {}
                        if use_cp_surface:
                            support_fn = getattr(be, "_td_perf_analyze_quadratic_surface_control_period_fit_support", None)
                            if callable(support_fn):
                                cp_support = support_fn(
                                    pooled_x1,
                                    pooled_x2,
                                    pooled_y,
                                    pooled_cp,
                                    fit_mode=(
                                        str(getattr(be, "TD_PERF_FIT_MODE_AUTO_SURFACE", "auto_surface"))
                                        if surface_family == str(getattr(be, "TD_PERF_FIT_MODE_AUTO_SURFACE", "auto_surface"))
                                        else str(getattr(be, "TD_PERF_FIT_MODE_POLYNOMIAL_SURFACE", "polynomial_surface"))
                                    ),
                                )
                        model = self._perf_fit_surface_for_points(
                            pooled_x1,
                            pooled_x2,
                            pooled_y,
                            control_periods=(pooled_cp if use_cp_surface else None),
                            display_control_period=display_control_period,
                        )
                        if use_cp_surface and not isinstance(model, dict):
                            failure_fn = getattr(be, "_td_perf_format_surface_control_period_failure", None)
                            if callable(failure_fn):
                                raise RuntimeError(
                                    failure_fn(
                                        cp_support.get("ignored_control_periods") or [],
                                        eligible_count=len(cp_support.get("eligible_control_period_values") or []),
                                    )
                                )
                            raise RuntimeError(
                                "Quadratic Surface + Control Period requires at least two distinct control periods with valid surface coverage."
                            )
                    except Exception as exc:
                        if not fit_error_text:
                            fit_error_text = str(exc)
                        model = None
                    if isinstance(model, dict):
                        master_model = model

                results[st] = {
                    "pair_label": pair_label,
                    "output_target": output_name,
                    "input1_target": input1_name,
                    "input2_target": input2_name,
                    "x_target": input1_name,
                    "y_target": output_name,
                    "input1_units": input1_units,
                    "input2_units": input2_units,
                    "output_units": output_units,
                    "stat_label": st,
                    "fit_mode": self._perf_requested_fit_mode(),
                    "plot_dimension": "3d",
                    "surface_fit_family": surface_family,
                    "selected_control_period": display_control_period,
                    "performance_plot_method": "cached_condition_means",
                    "points_3d": points_3d,
                    "regression_checker_rows": [
                        dict(row)
                        for serial in serials
                        for row in (checker_rows_by_serial.get(serial) or [])
                    ],
                    "master_model": master_model,
                    "fit_warning_text": str(master_model.get("fit_warning_text") or "").strip(),
                    "highlight_serial": "",
                    "highlight_model": {},
                }
            else:
                per_run_2d: list[tuple[str, str, dict[str, dict], dict[str, dict], str, str]] = []
                for rn in runs:
                    input1_col, input1_units = self._resolve_td_y_col_units(rn, input1_name)
                    output_col, output_units = self._resolve_td_y_col_units(rn, output_name)
                    if not input1_col or not output_col:
                        continue
                    input1_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            input1_col,
                            "mean",
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    output_map = _series_by_observation(
                        self._load_perf_equation_metric_series(
                            rn,
                            output_col,
                            st,
                            control_period_filter=control_period_filter,
                            run_type_filter=run_type_filter,
                        )
                    )
                    if not input1_map or not output_map:
                        continue
                    per_run_2d.append((rn, self._run_display_text(rn), input1_map, output_map, input1_units, output_units))

                curves: dict[str, list[tuple[float, float, str]]] = {}
                checker_rows_by_serial: dict[str, list[dict]] = {}
                pooled_x: list[float] = []
                pooled_y: list[float] = []
                for sn in serials:
                    pts_2d: list[tuple[float, float, str]] = []
                    visible_rows: list[dict] = []
                    for _rn, rdisp, input1_map, output_map, _xu, _yu in per_run_2d:
                        obs_ids = sorted(set(input1_map.keys()) & set(output_map.keys()))
                        for obs_id in obs_ids:
                            row_input1 = input1_map.get(obs_id) or {}
                            row_output = output_map.get(obs_id) or {}
                            if str(row_output.get("serial") or row_input1.get("serial") or "").strip() != sn:
                                continue
                            pts_2d.append(
                                (
                                    float(row_input1.get("value_num") or 0.0),
                                    float(row_output.get("value_num") or 0.0),
                                    _obs_label(rdisp, _rn, row_output or row_input1),
                                )
                            )
                            visible_rows.append(
                                self._perf_build_regression_checker_row(
                                    observation_id=obs_id,
                                    run_name_value=_rn,
                                    run_display=rdisp,
                                    row_input1=row_input1,
                                    row_output=row_output,
                                )
                            )
                    if len(pts_2d) >= require_min_points:
                        pts_2d.sort(key=lambda t: (t[0], t[2]))
                        curves[sn] = pts_2d
                        checker_rows_by_serial[sn] = sorted(
                            visible_rows,
                            key=lambda row: (
                                float(row.get("input_1") or 0.0),
                                str(row.get("condition_label") or "").lower(),
                                str(row.get("observation_id") or "").lower(),
                            ),
                        )
                        pooled_x.extend([p[0] for p in pts_2d])
                        pooled_y.extend([p[1] for p in pts_2d])

                if not curves:
                    continue

                input1_units = next((u for *_rest, u, _ in per_run_2d if str(u).strip()), "")
                output_units = next((u for *_rest, _, u in per_run_2d if str(u).strip()), "")

                master_model = {}
                if fit_enabled and pooled_x:
                    try:
                        model = self._perf_fit_model_for_points(pooled_x, pooled_y, sample_weights=None)
                    except Exception as exc:
                        if not fit_error_text:
                            fit_error_text = str(exc)
                        model = None
                    if isinstance(model, dict):
                        master_model = model

                results[st] = {
                    "pair_label": pair_label,
                    "output_target": output_name,
                    "input1_target": input1_name,
                    "input2_target": "",
                    "x_target": input1_name,
                    "y_target": output_name,
                    "x_units": input1_units,
                    "y_units": output_units,
                    "stat_label": st,
                    "fit_mode": self._perf_requested_fit_mode(),
                    "plot_dimension": "2d",
                    "performance_plot_method": "cached_condition_means",
                    "curves": curves,
                    "regression_checker_rows": [
                        dict(row)
                        for serial in serials
                        for row in (checker_rows_by_serial.get(serial) or [])
                    ],
                    "master_model": master_model,
                    "highlight_serial": "",
                    "highlight_model": {},
                }

            if st in plot_stats:
                plot_view_stats.append(st)

        return results, plot_view_stats, fit_error_text

    def _render_performance_result(
        self,
        ax,
        result: dict,
        *,
        highlight_serial: str = "",
        title_override: str = "",
        select_equation_row: bool = False,
    ) -> None:
        if not isinstance(result, dict):
            return
        plot_dimension = str(result.get("plot_dimension") or "2d").strip().lower()
        plot_method = self._perf_normalize_plot_method(result.get("performance_plot_method"))
        use_cached_condition_means = plot_method == "cached_condition_means"
        stat_label = str(result.get("stat_label") or "").strip().lower()
        master_model = result.get("master_model") or {}
        master_family = self._perf_fit_family_label((master_model or {}).get("fit_family") or "")
        if plot_dimension == "3d":
            points_3d = result.get("points_3d") or {}
            if not isinstance(points_3d, dict) or not points_3d:
                return
            output_target = str(result.get("output_target") or "").strip()
            input1_target = str(result.get("input1_target") or "").strip()
            input2_target = str(result.get("input2_target") or "").strip()
            input1_units = str(result.get("input1_units") or "").strip()
            input2_units = str(result.get("input2_units") or "").strip()
            output_units = str(result.get("output_units") or "").strip()
            pair_label = str(result.get("pair_label") or f"{output_target} vs {input1_target},{input2_target}").strip() or "Performance"
            title = str(title_override or "").strip() or (
                f"Performance - {pair_label} - {stat_label}" + (f" ({master_family})" if master_family else "")
            )
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel(f"{input1_target}.{stat_label}" + (f" ({input1_units})" if input1_units else ""))
            ax.set_ylabel(f"{input2_target}.{stat_label}" + (f" ({input2_units})" if input2_units else ""))
            try:
                ax.set_zlabel(f"{output_target}.{stat_label}" + (f" ({output_units})" if output_units else ""))
            except Exception:
                pass

            for sn, pts in points_3d.items():
                if not isinstance(pts, list) or not pts:
                    continue
                if highlight_serial and sn == highlight_serial:
                    continue
                x1s = [p[0] for p in pts]
                x2s = [p[1] for p in pts]
                ys = [p[2] for p in pts]
                try:
                    ax.scatter(x1s, x2s, ys, s=10, alpha=0.18, color="#64748b")
                    if not use_cached_condition_means and len(pts) >= 2:
                        ax.plot(x1s, x2s, ys, linewidth=0.8, alpha=0.14, color="#64748b")
                except Exception:
                    continue

            hi_color = "#ef4444"
            if highlight_serial and highlight_serial in points_3d:
                pts = points_3d.get(highlight_serial) or []
                if isinstance(pts, list) and pts:
                    x1s = [p[0] for p in pts]
                    x2s = [p[1] for p in pts]
                    ys = [p[2] for p in pts]
                    ax.scatter(x1s, x2s, ys, s=28, alpha=0.95, color=hi_color, label=highlight_serial)
                    if not use_cached_condition_means and len(pts) >= 2:
                        ax.plot(x1s, x2s, ys, linewidth=2.0, alpha=0.9, color=hi_color)
                    for x1, x2, yv, lbl in pts:
                        try:
                            ax.text(x1, x2, yv, str(lbl), fontsize=7, alpha=0.8, color=hi_color)
                        except Exception:
                            pass

            if isinstance(master_model, dict) and master_model.get("x1_grid") and master_model.get("x2_grid") and master_model.get("z_grid"):
                try:
                    fam = self._perf_fit_family_label(master_model.get("fit_family") or "")
                    label = f"master fit ({fam})" if fam else "master fit"
                    ax.plot_wireframe(
                        master_model["x1_grid"],
                        master_model["x2_grid"],
                        master_model["z_grid"],
                        color="#0f172a",
                        linewidth=0.5,
                        alpha=0.35,
                        label=label,
                    )
                except Exception:
                    pass
            hi_model = result.get("highlight_model") or {}
            if (
                highlight_serial
                and isinstance(hi_model, dict)
                and hi_model.get("x1_grid")
                and hi_model.get("x2_grid")
                and hi_model.get("z_grid")
            ):
                try:
                    fam = self._perf_fit_family_label(hi_model.get("fit_family") or "")
                    label = f"{highlight_serial} fit ({fam})" if fam else f"{highlight_serial} fit"
                    ax.plot_wireframe(
                        hi_model["x1_grid"],
                        hi_model["x2_grid"],
                        hi_model["z_grid"],
                        color=hi_color,
                        linewidth=0.6,
                        alpha=0.45,
                        label=label,
                    )
                except Exception:
                    pass

            try:
                if highlight_serial:
                    ax.legend(fontsize=8, loc="best")
            except Exception:
                pass
        else:
            curves = result.get("curves") or {}
            if not isinstance(curves, dict) or not curves:
                return
            x_target = str(result.get("x_target") or "").strip()
            y_target = str(result.get("y_target") or "").strip()
            x_units = str(result.get("x_units") or "").strip()
            y_units = str(result.get("y_units") or "").strip()
            pair_label = str(result.get("pair_label") or f"{y_target} vs {x_target}").strip() or "Performance"
            title = str(title_override or "").strip() or (
                f"Performance - {pair_label} - {stat_label}" + (f" ({master_family})" if master_family else "")
            )

            ax.clear()
            ax.set_title(title)
            ax.set_xlabel(f"{x_target}.{stat_label}" + (f" ({x_units})" if x_units else ""))
            ax.set_ylabel(f"{y_target}.{stat_label}" + (f" ({y_units})" if y_units else ""))

            for sn, pts in curves.items():
                if not isinstance(pts, list) or not pts:
                    continue
                if highlight_serial and sn == highlight_serial:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                try:
                    if use_cached_condition_means:
                        ax.scatter(xs, ys, s=12, alpha=0.22, color="#64748b")
                    else:
                        if len(pts) < 2:
                            continue
                        ax.plot(xs, ys, linewidth=0.9, alpha=0.14, color="#64748b")
                except Exception:
                    continue

            hi_color = "#ef4444"
            if highlight_serial and highlight_serial in curves:
                pts = curves.get(highlight_serial) or []
                if isinstance(pts, list) and pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if use_cached_condition_means:
                        ax.scatter(xs, ys, s=32, alpha=0.95, color=hi_color, label=highlight_serial)
                    else:
                        if len(pts) >= 2:
                            ax.plot(xs, ys, marker="o", linewidth=2.4, alpha=0.98, color=hi_color, label=highlight_serial)
                    for x, y, lbl in pts:
                        try:
                            ax.annotate(str(lbl), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8, color=hi_color)
                        except Exception:
                            pass

            if isinstance(master_model, dict) and master_model.get("xfit") and master_model.get("yfit"):
                try:
                    fam = self._perf_fit_family_label(master_model.get("fit_family") or "")
                    label = f"master fit ({fam})" if fam else "master fit"
                    ax.plot(master_model["xfit"], master_model["yfit"], linestyle="--", linewidth=1.5, alpha=0.75, color="#0f172a", label=label)
                except Exception:
                    pass
            hi_model = result.get("highlight_model") or {}
            if isinstance(hi_model, dict) and hi_model.get("xfit") and hi_model.get("yfit"):
                try:
                    fam = self._perf_fit_family_label(hi_model.get("fit_family") or "")
                    label = f"{highlight_serial} fit ({fam})" if fam else f"{highlight_serial} fit"
                    ax.plot(hi_model["xfit"], hi_model["yfit"], linestyle="--", linewidth=1.5, alpha=0.8, color=hi_color, label=label)
                except Exception:
                    pass

            ax.grid(True, alpha=0.25)
            try:
                if highlight_serial:
                    ax.legend(fontsize=8, loc="best")
            except Exception:
                pass

        try:
            ax.figure.tight_layout()
        except Exception:
            pass
        if select_equation_row:
            self._select_perf_equation_row(stat_label)

    def _on_perf_stats_changed(self) -> None:
        self._sync_perf_view_stats()
        self._clear_perf_results()

    def _on_perf_view_stat_changed(self) -> None:
        if self._mode != "performance":
            return
        self._redraw_performance_view()

    def _selected_perf_plotter(self) -> dict | None:
        if not hasattr(self, "cb_perf_plotter"):
            return None
        try:
            d = self.cb_perf_plotter.currentData()
        except Exception:
            d = None
        return d if isinstance(d, dict) else None

    @staticmethod
    def _perf_candidate_key(candidate: dict | None) -> tuple[str, str]:
        if not isinstance(candidate, dict):
            return "", ""
        return (
            str(candidate.get("x_norm") or "").strip().lower(),
            str(candidate.get("y_norm") or "").strip().lower(),
        )

    def _set_selected_perf_candidate(self, candidate: dict | None, *, switch_to_plot: bool = False) -> None:
        if not isinstance(candidate, dict) or not hasattr(self, "cb_perf_plotter"):
            return
        want_key = self._perf_candidate_key(candidate)
        if not all(want_key):
            return
        for i in range(self.cb_perf_plotter.count()):
            data = self.cb_perf_plotter.itemData(i)
            if self._perf_candidate_key(data if isinstance(data, dict) else None) == want_key:
                if self.cb_perf_plotter.currentIndex() != i:
                    self.cb_perf_plotter.setCurrentIndex(i)
                return

    def _sync_perf_candidate_table_selection(self) -> None:
        if not hasattr(self, "tbl_perf_candidates"):
            return
        current_key = self._perf_candidate_key(self._selected_perf_plotter())
        self.tbl_perf_candidates.blockSignals(True)
        try:
            self.tbl_perf_candidates.clearSelection()
            if not all(current_key):
                return
            for row in range(self.tbl_perf_candidates.rowCount()):
                item = self.tbl_perf_candidates.item(row, 0)
                candidate = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
                if self._perf_candidate_key(candidate if isinstance(candidate, dict) else None) == current_key:
                    self.tbl_perf_candidates.selectRow(row)
                    break
        finally:
            self.tbl_perf_candidates.blockSignals(False)

    def _refresh_perf_candidate_table(self) -> None:
        if not hasattr(self, "tbl_perf_candidates"):
            return
        raw_filter = str(getattr(self, "ed_perf_candidate_filter", None).text() if hasattr(self, "ed_perf_candidate_filter") else "" or "")
        tokens = [tok for tok in raw_filter.strip().lower().split() if tok]
        all_candidates = list(getattr(self, "_perf_plotters", []) or [])
        filtered: list[dict] = []
        for candidate in all_candidates:
            if not isinstance(candidate, dict):
                continue
            hay = " ".join(
                [
                    str(candidate.get("display_name") or candidate.get("name") or ""),
                    str((candidate.get("x") or {}).get("column") if isinstance(candidate.get("x"), dict) else ""),
                    str((candidate.get("y") or {}).get("column") if isinstance(candidate.get("y"), dict) else ""),
                    str(candidate.get("x_units") or ""),
                    str(candidate.get("y_units") or ""),
                    " ".join([str(sn) for sn in (candidate.get("qualifying_serials") or []) if str(sn).strip()]),
                ]
            ).lower()
            if tokens and not all(tok in hay for tok in tokens):
                continue
            filtered.append(candidate)

        self.tbl_perf_candidates.setRowCount(len(filtered))
        for row, candidate in enumerate(filtered):
            x_spec = candidate.get("x") or {}
            y_spec = candidate.get("y") or {}
            values = [
                str(candidate.get("display_name") or candidate.get("name") or "").strip(),
                str((x_spec.get("column") if isinstance(x_spec, dict) else "") or "").strip(),
                str((y_spec.get("column") if isinstance(y_spec, dict) else "") or "").strip(),
                str(int(candidate.get("qualifying_serial_count") or 0)),
                str(int(candidate.get("distinct_x_point_count") or 0)),
                ", ".join([str(v).strip() for v in (candidate.get("available_equation_views") or []) if str(v).strip()]),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if col == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, candidate)
                self.tbl_perf_candidates.setItem(row, col, item)
        try:
            self.tbl_perf_candidates.resizeColumnsToContents()
        except Exception:
            pass

        if hasattr(self, "lbl_perf_candidate_summary"):
            total = len(all_candidates)
            shown = len(filtered)
            self.lbl_perf_candidate_summary.setText(
                f"Viable candidates: {total} | Showing: {shown} | Rule: clustered X operating points with min span"
            )
        self._sync_perf_candidate_table_selection()

    def _activate_selected_perf_candidate(self, *, switch_to_plot: bool = False) -> None:
        if not hasattr(self, "tbl_perf_candidates"):
            return
        rows = sorted({idx.row() for idx in self.tbl_perf_candidates.selectionModel().selectedRows()}) if self.tbl_perf_candidates.selectionModel() else []
        if not rows:
            return
        item = self.tbl_perf_candidates.item(rows[0], 0)
        candidate = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(candidate, dict):
            self._set_selected_perf_candidate(candidate, switch_to_plot=switch_to_plot)

    def _on_perf_candidate_selection_changed(self) -> None:
        self._activate_selected_perf_candidate(switch_to_plot=False)

    def _resolve_td_y_name(self, run_name: str, target: str) -> str:
        tgt = str(target or "").strip()
        if not tgt or not self._db_path:
            return ""

        def _norm(s: str) -> str:
            return "".join(ch.lower() for ch in str(s or "").strip() if ch.isalnum())

        want = _norm(tgt)
        try:
            cols = be.td_list_y_columns(self._db_path, run_name)
        except Exception:
            cols = []
        by_norm = {
            _norm(str(c.get("name") or "")): str(c.get("name") or "").strip()
            for c in cols
            if str(c.get("name") or "").strip()
        }
        return by_norm.get(want, "")

    def _load_metric_map(self, run_name: str, col_name: str, stat: str) -> dict[str, float]:
        if not self._db_path:
            return {}
        series = self._load_metric_series_for_selection(run_name, col_name, stat)
        out: dict[str, float] = {}
        for r in series:
            sn = str(r.get("serial") or "").strip()
            v = r.get("value_num")
            if not sn or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                continue
            out[sn] = float(v)
        return out

    def _perf_bounds_mode(self, plotter: dict | None) -> str:
        normalize = getattr(be, "td_perf_normalize_bounds_mode", None)
        if callable(normalize):
            try:
                return str(normalize((plotter or {}).get("bounds_mode"))).strip().lower()
            except Exception:
                pass
        raw = str((plotter or {}).get("bounds_mode") or "").strip().lower()
        return raw if raw in {"actual", "median_3sigma"} else "median_3sigma"

    def _perf_bounds_mode_label(self, bounds_mode: str) -> str:
        return "median +/- 3sigma" if str(bounds_mode or "").strip().lower() == "median_3sigma" else "actual min/max"

    def _load_perf_display_metric_map(self, run_name: str, col_name: str, display_stat: str, *, bounds_mode: str) -> dict[str, float]:
        st = str(display_stat or "").strip().lower()
        if not st:
            return {}
        source_stats = {st}
        if st in {"min", "max"}:
            source_stats.update({"median", "std"})
        source_maps = {src: self._load_metric_map(run_name, col_name, src) for src in source_stats}
        resolver = getattr(be, "td_perf_display_value", None)
        serials = sorted({sn for mapping in source_maps.values() for sn in mapping.keys()})
        out: dict[str, float] = {}
        for sn in serials:
            values = {src: mapping.get(sn) for src, mapping in source_maps.items()}
            try:
                val = resolver(values, st, bounds_mode=bounds_mode) if callable(resolver) else None
            except Exception:
                val = None
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                out[sn] = float(val)
        return out

    def _load_perf_equation_metric_map(self, run_name: str, col_name: str, stat: str) -> dict[str, float]:
        series = self._load_perf_equation_metric_series(run_name, col_name, stat)
        out: dict[str, float] = {}
        for row in series:
            sn = str(row.get("serial") or "").strip()
            val = row.get("value_num")
            if not sn or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                continue
            out[sn] = float(val)
        return out

    def _load_perf_equation_metric_series(
        self,
        run_name: str,
        col_name: str,
        stat: str,
        *,
        control_period_filter: object = None,
        run_type_filter: object = None,
    ) -> list[dict]:
        st = str(stat or "").strip().lower()
        if not st:
            return []
        if st in {"min_3sigma", "max_3sigma"}:
            resolver = getattr(be, "td_perf_mean_3sigma_value", None)
            if not callable(resolver):
                return []
            mean_rows = self._load_metric_series_for_selection(
                run_name,
                col_name,
                "mean",
                control_period_filter=control_period_filter,
                run_type_filter=run_type_filter,
            )
            std_rows = self._load_metric_series_for_selection(
                run_name,
                col_name,
                "std",
                control_period_filter=control_period_filter,
                run_type_filter=run_type_filter,
            )
            mean_by_obs = {
                str(row.get("observation_id") or "").strip(): dict(row)
                for row in mean_rows
                if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
            }
            std_by_obs = {
                str(row.get("observation_id") or "").strip(): dict(row)
                for row in std_rows
                if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
            }
            out: list[dict] = []
            for obs_id in sorted(set(mean_by_obs.keys()) | set(std_by_obs.keys())):
                mean_row = mean_by_obs.get(obs_id) or {}
                std_row = std_by_obs.get(obs_id) or {}
                try:
                    val = resolver(
                        {
                            "mean": (mean_row or {}).get("value_num"),
                            "std": (std_row or {}).get("value_num"),
                        },
                        st,
                    )
                except Exception:
                    val = None
                if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                    continue
                base_row = dict(mean_row or std_row)
                base_row["value_num"] = float(val)
                out.append(base_row)
            return out
        return self._load_metric_series_for_selection(
            run_name,
            col_name,
            st,
            control_period_filter=control_period_filter,
            run_type_filter=run_type_filter,
        )

    def _resolve_td_y_col_units(self, run_name: str, target: str) -> tuple[str, str]:
        tgt = str(target or "").strip()
        if not tgt or not self._db_path:
            return "", ""

        def _norm(s: str) -> str:
            return "".join(ch.lower() for ch in str(s or "").strip() if ch.isalnum())

        want = _norm(tgt)
        try:
            cols = be.td_list_y_columns(self._db_path, run_name)
        except Exception:
            cols = []
        for c in cols:
            nm = str((c or {}).get("name") or "").strip()
            if not nm:
                continue
            if _norm(nm) == want:
                return nm, str((c or {}).get("units") or "").strip()
        return tgt, ""

    @staticmethod
    def _perf_fmt_equation(coeffs: list[float], degree: int, *, x0: float | None, sx: float | None) -> tuple[str, str]:
        if not coeffs:
            return "", ""
        deg = int(degree)
        parts: list[str] = []
        for i, c in enumerate(coeffs):
            power = deg - i
            try:
                cf = float(c)
            except Exception:
                continue
            if power == 0:
                parts.append(f"{cf:+.4g}")
            elif power == 1:
                parts.append(f"{cf:+.4g}*x'")
            else:
                parts.append(f"{cf:+.4g}*x'^{power}")
        expr = " ".join(parts).lstrip("+").strip()
        if x0 is not None and sx is not None:
            return f"y = {expr}", f"x' = (x - {float(x0):.4g}) / {float(sx):.4g}"
        return f"y = {expr}", ""

    def _perf_poly_fit(
        self, xs: list[float], ys: list[float], degree: int, *, normalize_x: bool
    ) -> dict | None:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("numpy is required for polynomial fitting.") from exc
        deg = int(degree)
        if deg <= 0:
            return None
        if len(xs) < max(2, deg + 1):
            return None
        x_arr = np.array([float(v) for v in xs], dtype=float)
        y_arr = np.array([float(v) for v in ys], dtype=float)
        if normalize_x:
            x0 = float(np.mean(x_arr))
            sx = float(np.std(x_arr)) or 1.0
            xn = (x_arr - x0) / sx
        else:
            x0 = 0.0
            sx = 1.0
            xn = x_arr
        coeffs = np.polyfit(xn, y_arr, deg).tolist()
        p = np.poly1d(coeffs)
        yhat = p(xn)
        rmse = float(np.sqrt(np.mean((y_arr - yhat) ** 2)))
        poly_eqn, x_norm_eqn = self._perf_fmt_equation(
            [float(c) for c in coeffs],
            deg,
            x0=(x0 if normalize_x else None),
            sx=(sx if normalize_x else None),
        )
        return {
            "degree": deg,
            "coeffs": [float(c) for c in coeffs],
            "rmse": rmse,
            "x0": x0,
            "sx": sx,
            "normalize_x": bool(normalize_x),
            "equation": poly_eqn,
            "x_norm_equation": x_norm_eqn,
        }

    def _fill_perf_equations_table(self) -> None:
        if not hasattr(self, "tbl_perf_equations"):
            return
        rows_by_stat: dict[str, tuple[str, str, str, str, str, str, str, str, str]] = {}
        order = {
            "mean": 0,
            "min": 1,
            "max": 2,
            "std": 3,
            "min_3sigma": 4,
            "max_3sigma": 5,
        }
        for st, r in (getattr(self, "_perf_results_by_stat", {}) or {}).items():
            master = (r or {}).get("master_model") or {}
            hi = (r or {}).get("highlight_model") or {}
            rows_by_stat[str(st).strip().lower()] = (
                str(st),
                self._perf_fit_family_label(master.get("fit_family") or ""),
                str(master.get("equation") or ""),
                str(master.get("x_norm_equation") or ""),
                (f"{float(master.get('rmse')):.4g}" if isinstance(master.get("rmse"), (int, float)) else ""),
                self._perf_fit_family_label(hi.get("fit_family") or ""),
                str(hi.get("equation") or ""),
                str(hi.get("x_norm_equation") or ""),
                (f"{float(hi.get('rmse')):.4g}" if isinstance(hi.get("rmse"), (int, float)) else ""),
            )
        mean_row = rows_by_stat.get("mean")
        std_row = rows_by_stat.get("std")
        if mean_row:
            mean_master_eqn = str(mean_row[2] or "").strip()
            mean_hi_eqn = str(mean_row[6] or "").strip()
            std_master_eqn = str((std_row[2] if std_row else "") or "").strip()
            std_hi_eqn = str((std_row[6] if std_row else "") or "").strip()
            derived_master_min = f"({mean_master_eqn}) - 3*({std_master_eqn})" if mean_master_eqn and std_master_eqn else (f"{mean_master_eqn} - 3sigma" if mean_master_eqn else "")
            derived_master_max = f"({mean_master_eqn}) + 3*({std_master_eqn})" if mean_master_eqn and std_master_eqn else (f"{mean_master_eqn} + 3sigma" if mean_master_eqn else "")
            derived_hi_min = f"({mean_hi_eqn}) - 3*({std_hi_eqn})" if mean_hi_eqn and std_hi_eqn else (f"{mean_hi_eqn} - 3sigma" if mean_hi_eqn else "")
            derived_hi_max = f"({mean_hi_eqn}) + 3*({std_hi_eqn})" if mean_hi_eqn and std_hi_eqn else (f"{mean_hi_eqn} + 3sigma" if mean_hi_eqn else "")
            rows_by_stat["min_3sigma"] = (
                "min_3sigma",
                ("Derived from Mean and Std" if std_master_eqn else str(mean_row[1] or "")),
                derived_master_min,
                str(mean_row[3] or (std_row[3] if std_row else "") or ""),
                "",
                ("Derived from Mean and Std" if std_hi_eqn else str(mean_row[5] or "")),
                derived_hi_min,
                str(mean_row[7] or (std_row[7] if std_row else "") or ""),
                "",
            )
            rows_by_stat["max_3sigma"] = (
                "max_3sigma",
                ("Derived from Mean and Std" if std_master_eqn else str(mean_row[1] or "")),
                derived_master_max,
                str(mean_row[3] or (std_row[3] if std_row else "") or ""),
                "",
                ("Derived from Mean and Std" if std_hi_eqn else str(mean_row[5] or "")),
                derived_hi_max,
                str(mean_row[7] or (std_row[7] if std_row else "") or ""),
                "",
            )
        rows = list(rows_by_stat.values())
        rows.sort(key=lambda t: (order.get(str(t[0]).strip().lower(), 100), str(t[0]).lower()))
        self.tbl_perf_equations.setRowCount(len(rows))
        for r_i, row in enumerate(rows):
            for c_i, val in enumerate(row):
                it = QtWidgets.QTableWidgetItem(str(val))
                it.setFlags(it.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.tbl_perf_equations.setItem(r_i, c_i, it)
        try:
            self.tbl_perf_equations.resizeColumnsToContents()
        except Exception:
            pass
        self._update_perf_export_button_state()

    def _perf_has_exportable_models(self) -> bool:
        for result in (getattr(self, "_perf_results_by_stat", {}) or {}).values():
            master = (result or {}).get("master_model") or {}
            if isinstance(master, dict) and str(master.get("fit_family") or "").strip():
                return True
        return False

    def _update_perf_export_button_state(self) -> None:
        export_busy = bool(getattr(self, "_export_worker", None) and self._export_worker.isRunning())
        enabled = self._perf_has_exportable_models() and not export_busy
        for widget_name in ("btn_perf_export_equations", "btn_perf_export_interactive", "btn_perf_save_equation"):
            if hasattr(self, widget_name):
                try:
                    getattr(self, widget_name).setEnabled(enabled)
                except Exception:
                    pass
        if hasattr(self, "btn_perf_saved_equations"):
            try:
                self.btn_perf_saved_equations.setEnabled(bool(getattr(self, "_project_dir", None)) and not export_busy)
            except Exception:
                pass

    def _perf_exact_equation_text(self, stat: str) -> str:
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        raw = str(stat or "").strip().lower()
        if raw in {"min_3sigma", "max_3sigma"}:
            mean_model = ((results.get("mean") or {}).get("master_model") or {}) if isinstance(results, dict) else {}
            std_model = ((results.get("std") or {}).get("master_model") or {}) if isinstance(results, dict) else {}
            mean_eq = str((mean_model or {}).get("equation") or "").strip()
            std_eq = str((std_model or {}).get("equation") or "").strip()
            if mean_eq and std_eq:
                sign = "-" if raw == "min_3sigma" else "+"
                return f"({mean_eq}) {sign} 3*({std_eq})"
        model = ((results.get(raw) or {}).get("master_model") or {}) if isinstance(results, dict) else {}
        return str((model or {}).get("equation") or "").strip()

    def _update_perf_primary_equation_banner(self) -> None:
        if not hasattr(self, "lbl_perf_primary_equation"):
            return
        if str(getattr(self, "_mode", "") or "").strip().lower() != "performance":
            self.lbl_perf_primary_equation.clear()
            self.lbl_perf_primary_equation.setVisible(False)
            return
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        first_result = next((r for r in results.values() if isinstance(r, dict)), {}) or {}
        if str(first_result.get("plot_dimension") or "2d").strip().lower() == "3d":
            self.lbl_perf_primary_equation.clear()
            self.lbl_perf_primary_equation.setVisible(False)
            return
        lines: list[str] = []
        for stat in ("mean", "min", "max", "min_3sigma", "max_3sigma"):
            eq = self._perf_exact_equation_text(stat)
            if not eq:
                continue
            lines.append(f"<b>{html.escape(stat)}</b>: <span style='font-family: Consolas, monospace;'>{html.escape(eq)}</span>")
        if not lines:
            self.lbl_perf_primary_equation.clear()
            self.lbl_perf_primary_equation.setVisible(False)
            return
        output_target = html.escape(str(first_result.get("output_target") or first_result.get("y_target") or "").strip())
        input1_target = html.escape(str(first_result.get("input1_target") or first_result.get("x_target") or "").strip())
        subtitle = f"<div style='margin-top:2px; margin-bottom:6px;'>{output_target} vs {input1_target}</div>" if output_target and input1_target else ""
        body = "<br>".join(lines)
        self.lbl_perf_primary_equation.setText(f"<div><b>Performance Equation</b>{subtitle}{body}</div>")
        self.lbl_perf_primary_equation.setVisible(True)

    @staticmethod
    def _perf_export_slug(value: object) -> str:
        text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
        return text or "value"

    def _perf_export_run_specs(self, runs: list[str], output_target: str, input1_target: str, input2_target: str) -> list[dict]:
        specs: list[dict] = []
        for run_name in runs or []:
            output_col, output_units = self._resolve_td_y_col_units(run_name, output_target)
            input1_col, input1_units = self._resolve_td_y_col_units(run_name, input1_target)
            input2_col = ""
            input2_units = ""
            if str(input2_target or "").strip():
                input2_col, input2_units = self._resolve_td_y_col_units(run_name, input2_target)
            if not output_col or not input1_col:
                continue
            if str(input2_target or "").strip() and not input2_col:
                continue
            specs.append(
                {
                    "run_name": str(run_name or "").strip(),
                    "display_name": self._run_display_text(run_name),
                    "output_column": output_col,
                    "output_units": output_units,
                    "input1_column": input1_col,
                    "input1_units": input1_units,
                    "input2_column": input2_col,
                    "input2_units": input2_units,
                }
            )
        return specs

    def _perf_current_regression_checker_rows(
        self,
        run_specs: list[dict[str, object]],
        *,
        control_period_filter: object = None,
        run_type_filter: object = None,
    ) -> list[dict[str, object]]:
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        mean_result = (results.get("mean") or next((r for r in results.values() if isinstance(r, dict)), {})) or {}
        live_rows = [dict(row) for row in (mean_result.get("regression_checker_rows") or []) if isinstance(row, dict)]
        if live_rows:
            live_rows.sort(
                key=lambda row: (
                    str(row.get("run_name") or "").lower(),
                    str(row.get("program_title") or "").lower(),
                    str(row.get("serial") or "").lower(),
                    float(row.get("input_1") or 0.0),
                    float(row.get("input_2") or 0.0) if row.get("input_2") not in (None, "") else float("-inf"),
                    str(row.get("observation_id") or "").lower(),
                )
            )
            return live_rows
        plot_dimension = str(mean_result.get("plot_dimension") or "2d").strip().lower()
        qualifying_serials: set[str] = set()
        if plot_dimension == "3d":
            point_map = mean_result.get("points_3d") or {}
            if isinstance(point_map, dict):
                qualifying_serials = {
                    str(serial).strip()
                    for serial, points in point_map.items()
                    if str(serial).strip() and isinstance(points, list) and points
                }
        else:
            curve_map = mean_result.get("curves") or {}
            if isinstance(curve_map, dict):
                qualifying_serials = {
                    str(serial).strip()
                    for serial, points in curve_map.items()
                    if str(serial).strip() and isinstance(points, list) and points
                }
        if not qualifying_serials:
            qualifying_serials = {str(serial).strip() for serial in self._selected_perf_serials() if str(serial).strip()}

        out: list[dict[str, object]] = []
        for spec in run_specs or []:
            if not isinstance(spec, dict):
                continue
            run_name = str(spec.get("run_name") or "").strip()
            display_name = str(spec.get("display_name") or run_name).strip() or run_name
            input1_column = str(spec.get("input1_column") or "").strip()
            input2_column = str(spec.get("input2_column") or "").strip()
            output_column = str(spec.get("output_column") or "").strip()
            if not run_name or not input1_column or not output_column:
                continue

            x1_rows = self._load_perf_equation_metric_series(
                run_name,
                input1_column,
                "mean",
                control_period_filter=control_period_filter,
                run_type_filter=run_type_filter,
            )
            y_rows = self._load_perf_equation_metric_series(
                run_name,
                output_column,
                "mean",
                control_period_filter=control_period_filter,
                run_type_filter=run_type_filter,
            )
            if not x1_rows or not y_rows:
                continue
            x1_map = {
                str(row.get("observation_id") or "").strip(): dict(row)
                for row in x1_rows
                if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
            }
            y_map = {
                str(row.get("observation_id") or "").strip(): dict(row)
                for row in y_rows
                if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
            }
            x2_map: dict[str, dict] = {}
            obs_ids = set(x1_map.keys()) & set(y_map.keys())
            if input2_column:
                x2_rows = self._load_perf_equation_metric_series(
                    run_name,
                    input2_column,
                    "mean",
                    control_period_filter=control_period_filter,
                    run_type_filter=run_type_filter,
                )
                x2_map = {
                    str(row.get("observation_id") or "").strip(): dict(row)
                    for row in x2_rows
                    if isinstance(row, dict) and str(row.get("observation_id") or "").strip()
                }
                obs_ids &= set(x2_map.keys())

            for observation_id in sorted(obs_ids):
                row_x1 = x1_map.get(observation_id) or {}
                row_y = y_map.get(observation_id) or {}
                row_x2 = x2_map.get(observation_id) or {}
                serial = str(row_y.get("serial") or row_x1.get("serial") or row_x2.get("serial") or "").strip()
                if qualifying_serials and serial not in qualifying_serials:
                    continue
                try:
                    input_1 = float(row_x1.get("value_num"))
                    actual_mean = float(row_y.get("value_num"))
                except Exception:
                    continue
                if not (math.isfinite(input_1) and math.isfinite(actual_mean)):
                    continue
                input_2 = None
                if input2_column:
                    try:
                        input_2 = float(row_x2.get("value_num"))
                    except Exception:
                        continue
                    if not math.isfinite(float(input_2)):
                        continue
                out.append(
                    self._perf_build_regression_checker_row(
                        observation_id=observation_id,
                        run_name_value=run_name,
                        run_display=display_name,
                        row_input1=row_x1,
                        row_output=row_y,
                        row_input2=row_x2,
                    )
                )
        out.sort(
            key=lambda row: (
                str(row.get("run_name") or "").lower(),
                str(row.get("program_title") or "").lower(),
                str(row.get("serial") or "").lower(),
                float(row.get("input_1") or 0.0),
                float(row.get("input_2") or 0.0) if row.get("input_2") not in (None, "") else float("-inf"),
                str(row.get("observation_id") or "").lower(),
            )
        )
        return out

    def _open_spreadsheet_path(self, file_path: Path) -> None:
        import platform
        import subprocess

        resolved = str(Path(file_path).expanduser().resolve())
        if platform.system() == "Windows":
            subprocess.Popen(["start", "excel", resolved], shell=True)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", resolved])
        else:
            subprocess.Popen(["xdg-open", resolved])

    def _handle_perf_excel_export_success(self, payload: object, *, heading: str) -> str:
        exported_path = Path(payload).expanduser() if isinstance(payload, (str, Path)) else None
        if exported_path is None:
            raise RuntimeError(f"{heading} returned an invalid output path.")
        self._open_spreadsheet_path(exported_path)
        return f"{heading} complete: {exported_path.name}"

    def _start_perf_export_task(
        self,
        *,
        heading: str,
        status_text: str,
        task_factory,
        on_success,
    ) -> None:
        if self._cache_worker is not None and self._cache_worker.isRunning():
            QtWidgets.QMessageBox.information(self, heading, "Wait for the cache task to finish first.")
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            QtWidgets.QMessageBox.information(self, heading, "An Excel export is already running.")
            return

        self._update_perf_export_button_state()
        self._export_progress_visible = False
        try:
            self._report_progress.lbl_heading.setText(heading)
            self._report_progress.begin(status_text)
            self._report_progress.detail_label.setText("")
            self._export_progress_visible = True
        except Exception:
            self._export_progress_visible = False

        self._export_worker = ProjectTaskWorker(task_factory, parent=self)
        self._update_perf_export_button_state()
        self._export_worker.progress.connect(self._on_perf_export_task_progress)
        self._export_worker.completed.connect(lambda payload: self._on_perf_export_task_done(payload, on_success, heading))
        self._export_worker.failed.connect(lambda message: self._on_perf_export_task_error(message, heading))
        self._export_worker.start()

    def _on_perf_export_task_progress(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg or not self._export_progress_visible:
            return
        try:
            self._report_progress.detail_label.setText(msg)
        except Exception:
            pass

    def _on_perf_export_task_done(self, payload: object, on_success, heading: str) -> None:
        self._export_worker = None
        self._update_perf_export_button_state()
        try:
            msg = on_success(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, heading, str(exc))
            if self._export_progress_visible:
                try:
                    self._report_progress.finish(f"{heading} failed: {exc}", success=False)
                except Exception:
                    pass
                self._export_progress_visible = False
            return
        if self._export_progress_visible:
            try:
                self._report_progress.finish(msg or f"{heading} complete", success=True)
            except Exception:
                pass
            self._export_progress_visible = False

    def _on_perf_export_task_error(self, message: str, heading: str) -> None:
        self._export_worker = None
        self._update_perf_export_button_state()
        msg = str(message or "").strip() or f"{heading} failed."
        QtWidgets.QMessageBox.warning(self, heading, msg)
        if self._export_progress_visible:
            try:
                self._report_progress.finish(f"{heading} failed: {msg}", success=False)
            except Exception:
                pass
            self._export_progress_visible = False

    def _start_perf_equation_excel_export(
        self,
        output_path: Path,
        *,
        plot_metadata: dict[str, object],
        results_by_stat: dict[str, dict[str, object]],
        run_specs: list[dict[str, object]],
        control_period_filter: object = None,
        run_type_filter: object = None,
    ) -> None:
        output = Path(output_path).expanduser()
        project_dir = Path(getattr(self, "_project_dir", output.parent)).expanduser()

        def _task(report):
            return be.td_perf_export_equation_workbook(
                self._db_path,
                output,
                plot_metadata=plot_metadata,
                results_by_stat=results_by_stat,
                run_specs=run_specs,
                control_period_filter=control_period_filter,
                run_type_filter=run_type_filter,
                progress_cb=report,
            )

        self._start_perf_export_task(
            heading="Export Equation to Excel",
            status_text=f"Exporting equation workbook to {output.name}",
            task_factory=_task,
            on_success=lambda payload: self._handle_perf_excel_export_success(payload, heading="Export Equation to Excel"),
        )

    def _start_perf_interactive_equation_export(
        self,
        output_path: Path,
        *,
        plot_metadata: dict[str, object],
        results_by_stat: dict[str, dict[str, object]],
        run_specs: list[dict[str, object]],
        regression_checker_rows: list[dict[str, object]] | None = None,
        control_period_filter: object = None,
        run_type_filter: object = None,
        include_regression_checker: bool = True,
    ) -> None:
        output = Path(output_path).expanduser()
        checker_rows = [dict(row) for row in (regression_checker_rows or []) if isinstance(row, dict)]

        def _task(report):
            return be.td_perf_export_interactive_equation_workbook(
                self._db_path,
                output,
                plot_metadata=plot_metadata,
                results_by_stat=results_by_stat,
                run_specs=run_specs,
                regression_checker_rows=checker_rows,
                control_period_filter=control_period_filter,
                run_type_filter=run_type_filter,
                include_regression_checker=include_regression_checker,
                progress_cb=report,
            )

        self._start_perf_export_task(
            heading="Export Interactive Workbook",
            status_text=f"Exporting interactive workbook to {output.name}",
            task_factory=_task,
            on_success=lambda payload: self._handle_perf_excel_export_success(payload, heading="Export Interactive Workbook"),
        )

    def _start_saved_perf_equations_excel_export(
        self,
        output_path: Path,
        *,
        entries: list[dict],
    ) -> None:
        output = Path(output_path).expanduser()
        project_dir = Path(getattr(self, "_project_dir", output.parent)).expanduser()
        entry_count = len(entries)

        def _task(report):
            return be.td_perf_export_saved_equations_workbook(
                self._db_path,
                output,
                entries=entries,
                progress_cb=report,
            )

        self._start_perf_export_task(
            heading="Export Saved Performance Equations to Excel",
            status_text=f"Exporting {entry_count} saved equation(s) to {output.name}",
            task_factory=_task,
            on_success=lambda payload: self._handle_perf_excel_export_success(
                payload,
                heading="Export Saved Performance Equations to Excel",
            ),
        )

    def _export_perf_interactive_equation_workbook(self) -> None:
        if not self._perf_has_exportable_models():
            QtWidgets.QMessageBox.information(self, "Performance", "No exportable master equations are available.")
            return
        if not getattr(self, "_db_path", None):
            QtWidgets.QMessageBox.information(self, "Performance", "Build/refresh cache first.")
            return
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        first_result = next((r for r in results.values() if isinstance(r, dict)), {}) or {}
        output_target = str(first_result.get("output_target") or "").strip()
        input1_target = str(first_result.get("input1_target") or "").strip()
        input2_target = str(first_result.get("input2_target") or "").strip()
        if not output_target or not input1_target:
            QtWidgets.QMessageBox.information(self, "Performance", "Plot a performance fit before exporting equations.")
            return

        plot_def = dict(getattr(self, "_last_plot_def", {}) or {})
        performance_plot_method = self._perf_normalize_plot_method(plot_def.get("performance_plot_method"))
        current_selection = self._current_run_selection()
        runs = [str(v).strip() for v in (plot_def.get("member_runs") or current_selection.get("member_runs") or []) if str(v).strip()]
        run_specs = self._perf_export_run_specs(runs, output_target, input1_target, input2_target)
        run_type_mode = str(
            plot_def.get("performance_run_type_mode") or self._selected_perf_run_type_mode()
        ).strip().lower()
        perf_filter_mode = str(plot_def.get("performance_filter_mode") or self._selected_perf_filter_mode()).strip().lower()
        control_period_filter = plot_def.get("selected_control_period")
        if run_type_mode != "pulsed_mode" or perf_filter_mode != "match_control_period":
            control_period_filter = None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = (
            f"interactive_performance_equation_{self._perf_plot_method_file_slug(performance_plot_method)}_{self._perf_export_slug(output_target)}_vs_"
            f"{self._perf_export_slug(input1_target)}"
        )
        if str(input2_target or "").strip():
            default_name += f"_{self._perf_export_slug(input2_target)}"
        default_name += f"_{timestamp}.xlsx"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Interactive Workbook",
            str(self._project_dir / default_name),
            "Excel Files (*.xlsx)",
        )
        if not out_path:
            return

        plot_metadata = {
            "plot_dimension": str(first_result.get("plot_dimension") or plot_def.get("plot_dimension") or "2d"),
            "output_target": output_target,
            "output_units": str(first_result.get("output_units") or first_result.get("y_units") or "").strip(),
            "input1_target": input1_target,
            "input1_units": str(first_result.get("input1_units") or first_result.get("x_units") or "").strip(),
            "input2_target": input2_target,
            "input2_units": str(first_result.get("input2_units") or "").strip(),
            "run_selection_label": str(plot_def.get("run_condition") or plot_def.get("display_text") or self._selection_condition_label(current_selection)).strip(),
            "member_runs": list(runs),
            "performance_run_type_mode": run_type_mode,
            "performance_filter_mode": perf_filter_mode or "all_conditions",
            "selected_control_period": control_period_filter,
            "performance_plot_method": performance_plot_method,
        }
        try:
            asset_metadata = be.td_perf_collect_asset_metadata(self._db_path, be._td_perf_entry_serials(results))
        except Exception:
            asset_metadata = {}
        if isinstance(asset_metadata, dict):
            plot_metadata["asset_type"] = str(asset_metadata.get("primary_asset_type") or "").strip()
            plot_metadata["asset_specific_type"] = str(asset_metadata.get("primary_asset_specific_type") or "").strip()

        include_regression_checker = bool(
            getattr(self, "cb_perf_include_reg_checker", None) and self.cb_perf_include_reg_checker.isChecked()
        )
        try:
            regression_checker_rows = self._perf_current_regression_checker_rows(
                run_specs,
                control_period_filter=control_period_filter,
                run_type_filter=run_type_mode,
            )
            self._start_perf_interactive_equation_export(
                Path(out_path),
                plot_metadata=plot_metadata,
                results_by_stat=results,
                run_specs=run_specs,
                regression_checker_rows=regression_checker_rows,
                control_period_filter=control_period_filter,
                run_type_filter=run_type_mode,
                include_regression_checker=include_regression_checker,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Interactive Workbook", str(exc))

    def _export_perf_equations_to_excel(self) -> None:
        if not self._perf_has_exportable_models():
            QtWidgets.QMessageBox.information(self, "Performance", "No exportable master equations are available.")
            return
        if not getattr(self, "_db_path", None):
            QtWidgets.QMessageBox.information(self, "Performance", "Build/refresh cache first.")
            return
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        first_result = next((r for r in results.values() if isinstance(r, dict)), {}) or {}
        output_target = str(first_result.get("output_target") or "").strip()
        input1_target = str(first_result.get("input1_target") or "").strip()
        input2_target = str(first_result.get("input2_target") or "").strip()
        if not output_target or not input1_target:
            QtWidgets.QMessageBox.information(self, "Performance", "Plot a performance fit before exporting equations.")
            return

        plot_def = dict(getattr(self, "_last_plot_def", {}) or {})
        performance_plot_method = self._perf_normalize_plot_method(plot_def.get("performance_plot_method"))
        current_selection = self._current_run_selection()
        runs = [str(v).strip() for v in (plot_def.get("member_runs") or current_selection.get("member_runs") or []) if str(v).strip()]
        run_specs = self._perf_export_run_specs(runs, output_target, input1_target, input2_target)
        run_type_mode = str(
            plot_def.get("performance_run_type_mode") or self._selected_perf_run_type_mode()
        ).strip().lower()
        perf_filter_mode = str(plot_def.get("performance_filter_mode") or self._selected_perf_filter_mode()).strip().lower()
        control_period_filter = plot_def.get("selected_control_period")
        if run_type_mode != "pulsed_mode" or perf_filter_mode != "match_control_period":
            control_period_filter = None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = (
            f"performance_equation_export_{self._perf_plot_method_file_slug(performance_plot_method)}_{self._perf_export_slug(output_target)}_vs_"
            f"{self._perf_export_slug(input1_target)}"
        )
        if str(input2_target or "").strip():
            default_name += f"_{self._perf_export_slug(input2_target)}"
        default_name += f"_{timestamp}.xlsx"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Equation to Excel",
            str(self._project_dir / default_name),
            "Excel Files (*.xlsx)",
        )
        if not out_path:
            return

        plot_metadata = {
            "plot_dimension": str(first_result.get("plot_dimension") or plot_def.get("plot_dimension") or "2d"),
            "output_target": output_target,
            "output_units": str(first_result.get("output_units") or first_result.get("y_units") or "").strip(),
            "input1_target": input1_target,
            "input1_units": str(first_result.get("input1_units") or first_result.get("x_units") or "").strip(),
            "input2_target": input2_target,
            "input2_units": str(first_result.get("input2_units") or "").strip(),
            "run_selection_label": str(plot_def.get("run_condition") or plot_def.get("display_text") or self._selection_condition_label(current_selection)).strip(),
            "member_runs": list(runs),
            "performance_run_type_mode": run_type_mode,
            "performance_filter_mode": perf_filter_mode or "all_conditions",
            "performance_plot_method": performance_plot_method,
        }
        try:
            asset_metadata = be.td_perf_collect_asset_metadata(self._db_path, be._td_perf_entry_serials(results))
        except Exception:
            asset_metadata = {}
        if isinstance(asset_metadata, dict):
            plot_metadata["asset_type"] = str(asset_metadata.get("primary_asset_type") or "").strip()
            plot_metadata["asset_specific_type"] = str(asset_metadata.get("primary_asset_specific_type") or "").strip()
        try:
            self._start_perf_equation_excel_export(
                Path(out_path),
                plot_metadata=plot_metadata,
                results_by_stat=results,
                run_specs=run_specs,
                control_period_filter=control_period_filter,
                run_type_filter=run_type_mode,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Equation to Excel", str(exc))

    def _open_performance_equations_popup(self) -> None:
        dlg = getattr(self, "_perf_equations_popup", None)
        if dlg is None:
            return
        if dlg.isMinimized():
            dlg.showNormal()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _perf_default_saved_name(self) -> str:
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        first_result = next((r for r in results.values() if isinstance(r, dict)), {}) or {}
        output_target = str(first_result.get("output_target") or "").strip()
        input1_target = str(first_result.get("input1_target") or "").strip()
        input2_target = str(first_result.get("input2_target") or "").strip()
        method = self._perf_normalize_plot_method(
            (getattr(self, "_last_plot_def", {}) or {}).get("performance_plot_method")
            or first_result.get("performance_plot_method")
        )
        prefix = "Performance (Run Conditions)" if method == "cached_condition_means" else "Performance"
        if input2_target:
            return f"{prefix}: {output_target} vs {input1_target},{input2_target}".strip()
        return f"{prefix}: {output_target} vs {input1_target}".strip()

    def _build_current_saved_performance_entry(self, name: str, existing_entry: dict | None = None) -> dict:
        if not getattr(self, "_db_path", None):
            raise RuntimeError("Build/refresh cache first.")
        results = getattr(self, "_perf_results_by_stat", {}) or {}
        first_result = next((r for r in results.values() if isinstance(r, dict)), {}) or {}
        output_target = str(first_result.get("output_target") or "").strip()
        input1_target = str(first_result.get("input1_target") or "").strip()
        input2_target = str(first_result.get("input2_target") or "").strip()
        if not output_target or not input1_target:
            raise RuntimeError("Plot a performance fit before saving equations.")

        plot_def = dict(getattr(self, "_last_plot_def", {}) or {})
        performance_plot_method = self._perf_normalize_plot_method(plot_def.get("performance_plot_method"))
        plot_def["performance_plot_method"] = performance_plot_method
        current_selection = self._current_run_selection()
        runs = [
            str(value).strip()
            for value in (plot_def.get("member_runs") or current_selection.get("member_runs") or [])
            if str(value).strip()
        ]
        plot_def["member_runs"] = list(runs)
        plot_def["polynomial_degree"] = int(self.sp_perf_degree.value()) if hasattr(self, "sp_perf_degree") else 2
        plot_def["normalize_x"] = bool(getattr(self, "cb_perf_norm_x", None) and self.cb_perf_norm_x.isChecked())
        plot_def["require_min_points"] = int(getattr(self, "_perf_require_min_points", 2) or 2)
        plot_def["fit_mode"] = str(plot_def.get("fit_mode") or self._perf_requested_fit_mode()).strip().lower()
        plot_def["surface_fit_family"] = str(self._perf_requested_surface_family()).strip().lower()
        plot_def["output"] = output_target
        plot_def["input1"] = input1_target
        plot_def["input2"] = input2_target
        run_specs = self._perf_export_run_specs(runs, output_target, input1_target, input2_target)
        plot_metadata = {
            "plot_dimension": str(first_result.get("plot_dimension") or plot_def.get("plot_dimension") or "2d"),
            "output_target": output_target,
            "output_units": str(first_result.get("output_units") or first_result.get("y_units") or "").strip(),
            "input1_target": input1_target,
            "input1_units": str(first_result.get("input1_units") or first_result.get("x_units") or "").strip(),
            "input2_target": input2_target,
            "input2_units": str(first_result.get("input2_units") or "").strip(),
            "run_selection_label": str(plot_def.get("run_condition") or plot_def.get("display_text") or self._selection_condition_label(current_selection)).strip(),
            "display_text": str(plot_def.get("display_text") or self._selection_display_text(current_selection)).strip(),
            "run_condition": str(plot_def.get("run_condition") or self._selection_condition_label(current_selection)).strip(),
            "member_runs": list(runs),
            "performance_run_type_mode": str(plot_def.get("performance_run_type_mode") or self._selected_perf_run_type_mode()).strip().lower(),
            "performance_filter_mode": str(plot_def.get("performance_filter_mode") or self._selected_perf_filter_mode()).strip().lower() or "all_conditions",
            "selected_control_period": plot_def.get("selected_control_period"),
            "performance_plot_method": performance_plot_method,
        }
        return be.td_perf_build_saved_equation_entry(
            self._db_path,
            name=name,
            plot_definition=plot_def,
            plot_metadata=plot_metadata,
            results_by_stat=results,
            run_specs=run_specs,
            existing_id=((existing_entry or {}).get("id") if isinstance(existing_entry, dict) else None),
            existing_saved_at=((existing_entry or {}).get("saved_at") if isinstance(existing_entry, dict) else None),
        )

    def _format_saved_performance_entry_detail(self, entry: dict) -> str:
        lines: list[str] = []
        lines.append(f"Name: {str(entry.get('name') or '').strip()}")
        asset_metadata = dict(entry.get("asset_metadata") or {}) if isinstance(entry.get("asset_metadata"), dict) else {}
        primary_asset = str(asset_metadata.get("primary_asset_type") or "").strip() or "(Unspecified)"
        primary_specific = str(asset_metadata.get("primary_asset_specific_type") or "").strip() or "(Unspecified)"
        lines.append(f"Asset Type: {primary_asset}")
        lines.append(f"Asset Specific Type: {primary_specific}")
        plot_metadata = dict(entry.get("plot_metadata") or {}) if isinstance(entry.get("plot_metadata"), dict) else {}
        lines.append(f"Dimension: {str(plot_metadata.get('plot_dimension') or '').strip().upper()}")
        lines.append(f"Method: {self._perf_plot_method_label(plot_metadata.get('performance_plot_method'))}")
        lines.append(f"Output: {str(plot_metadata.get('output_target') or '').strip()}")
        lines.append(f"Input 1: {str(plot_metadata.get('input1_target') or '').strip()}")
        input2_target = str(plot_metadata.get("input2_target") or "").strip()
        if input2_target:
            lines.append(f"Input 2: {input2_target}")
        lines.append(f"Updated: {str(entry.get('updated_at') or '').strip()}")
        refresh_error = str(entry.get("refresh_error") or "").strip()
        if refresh_error:
            lines.append(f"Refresh Error: {refresh_error}")
        group_map = asset_metadata.get("serials_by_asset_group") or {}
        if isinstance(group_map, dict) and group_map:
            lines.append("")
            lines.append("Asset Groups:")
            for key, serials in sorted(group_map.items(), key=lambda item: str(item[0]).lower()):
                serial_list = ", ".join(str(v).strip() for v in (serials or []) if str(v).strip())
                lines.append(f"  {key}: {serial_list}")
        rows = entry.get("equation_rows") or []
        if isinstance(rows, list) and rows:
            lines.append("")
            lines.append("Equations:")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                lines.append(f"  [{str(row.get('stat') or '').strip()}] {str(row.get('fit_family') or '').strip()}")
                lines.append(f"    {str(row.get('equation') or '').strip()}")
                norm_text = str(row.get("x_norm_equation") or "").strip()
                if norm_text:
                    lines.append(f"    norm: {norm_text}")
        return "\n".join(lines)

    def _save_current_performance_equation(self) -> None:
        if not self._perf_has_exportable_models():
            QtWidgets.QMessageBox.information(self, "Performance", "No exportable master equations are available.")
            return
        store = be.load_td_saved_performance_equations(self._project_dir)
        entries = [dict(item) for item in (store.get("entries") or []) if isinstance(item, dict)]
        default_name = self._perf_default_saved_name()
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Performance Equation", "Saved equation name:", text=default_name)
        if not ok:
            return
        clean_name = str(name or "").strip()
        if not clean_name:
            QtWidgets.QMessageBox.information(self, "Save Performance Equation", "Enter a saved equation name.")
            return
        existing = next((item for item in entries if str(item.get("name") or "").strip().casefold() == clean_name.casefold()), None)
        if existing:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Overwrite Saved Equation",
                f"A saved performance equation named '{clean_name}' already exists.\n\nOverwrite it?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            entry = self._build_current_saved_performance_entry(clean_name, existing_entry=existing if isinstance(existing, dict) else None)
            be.td_perf_upsert_saved_equation(self._project_dir, entry)
            self._show_toast("Saved performance equation")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Performance Equation", str(exc))

    def _open_saved_performance_equations_popup(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Saved Performance Equations")
        dlg.resize(1080, 640)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        tbl = QtWidgets.QTableWidget(0, 11)
        tbl.setHorizontalHeaderLabels(
            [
                "name",
                "method",
                "asset_type",
                "asset_specific_type",
                "dimension",
                "output",
                "input_1",
                "input_2",
                "stats",
                "fit_family",
                "updated_at",
            ]
        )
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(tbl, 2)

        detail = QtWidgets.QPlainTextEdit()
        detail.setReadOnly(True)
        detail.setMinimumHeight(180)
        layout.addWidget(detail, 1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        btn_export_excel = QtWidgets.QPushButton("Export All to Excel")
        btn_export_matlab = QtWidgets.QPushButton("Export All to MATLAB")
        btn_delete = QtWidgets.QPushButton("Delete Selected")
        btn_close = QtWidgets.QPushButton("Close")
        for button in (btn_export_excel, btn_export_matlab, btn_delete, btn_close):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        def _load_entries() -> list[dict]:
            store = be.load_td_saved_performance_equations(self._project_dir)
            return [dict(item) for item in (store.get("entries") or []) if isinstance(item, dict)]

        def _selected_entry() -> dict | None:
            row = tbl.currentRow()
            if row < 0:
                return None
            item = tbl.item(row, 0)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
            return dict(data) if isinstance(data, dict) else None

        def _refresh_table() -> None:
            entries = _load_entries()
            tbl.setRowCount(len(entries))
            for row_idx, entry in enumerate(entries):
                plot_metadata = dict(entry.get("plot_metadata") or {}) if isinstance(entry.get("plot_metadata"), dict) else {}
                asset_metadata = dict(entry.get("asset_metadata") or {}) if isinstance(entry.get("asset_metadata"), dict) else {}
                equation_rows = entry.get("equation_rows") or []
                stats_text = ", ".join(
                    str(row.get("stat") or "").strip()
                    for row in equation_rows
                    if isinstance(row, dict) and str(row.get("stat") or "").strip()
                )
                family_text = next(
                    (
                        str(row.get("fit_family") or "").strip()
                        for row in equation_rows
                        if isinstance(row, dict) and str(row.get("fit_family") or "").strip()
                    ),
                    "",
                )
                values = [
                    str(entry.get("name") or "").strip(),
                    self._perf_plot_method_label(plot_metadata.get("performance_plot_method")),
                    str(asset_metadata.get("primary_asset_type") or "").strip(),
                    str(asset_metadata.get("primary_asset_specific_type") or "").strip(),
                    str(plot_metadata.get("plot_dimension") or "").strip().upper(),
                    str(plot_metadata.get("output_target") or "").strip(),
                    str(plot_metadata.get("input1_target") or "").strip(),
                    str(plot_metadata.get("input2_target") or "").strip(),
                    stats_text,
                    family_text,
                    str(entry.get("updated_at") or "").strip(),
                ]
                for col_idx, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    if col_idx == 0:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
                    tbl.setItem(row_idx, col_idx, item)
            try:
                tbl.resizeColumnsToContents()
            except Exception:
                pass
            if entries:
                tbl.selectRow(0)
            else:
                detail.clear()
            enabled = bool(entries)
            btn_export_excel.setEnabled(enabled)
            btn_export_matlab.setEnabled(enabled)
            btn_delete.setEnabled(enabled and tbl.currentRow() >= 0)

        def _sync_detail() -> None:
            entry = _selected_entry()
            if entry is None:
                detail.clear()
                btn_delete.setEnabled(False)
                return
            detail.setPlainText(self._format_saved_performance_entry_detail(entry))
            btn_delete.setEnabled(True)

        def _export_all_to_excel() -> None:
            if not getattr(self, "_db_path", None):
                QtWidgets.QMessageBox.information(dlg, "Saved Performance Equations", "Build/refresh cache first.")
                return
            entries = _load_entries()
            if not entries:
                QtWidgets.QMessageBox.information(dlg, "Saved Performance Equations", "No saved equations are available.")
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Export All Saved Equations to Excel",
                str(self._project_dir / f"saved_performance_equations_{timestamp}.xlsx"),
                "Excel Files (*.xlsx)",
            )
            if not out_path:
                return
            try:
                self._start_saved_perf_equations_excel_export(Path(out_path), entries=entries)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, "Saved Performance Equations", str(exc))

        def _export_all_to_matlab() -> None:
            entries = _load_entries()
            if not entries:
                QtWidgets.QMessageBox.information(dlg, "Saved Performance Equations", "No saved equations are available.")
                return
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Export All Saved Equations to MATLAB",
                str(self._project_dir / f"saved_performance_equations_{timestamp}.m"),
                "MATLAB Files (*.m)",
            )
            if not out_path:
                return
            try:
                exported = be.td_perf_export_saved_equations_matlab(Path(out_path), entries=entries)
                try:
                    be.open_path(Path(exported))
                except Exception:
                    pass
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, "Saved Performance Equations", str(exc))

        def _delete_selected() -> None:
            entry = _selected_entry()
            if entry is None:
                return
            answer = QtWidgets.QMessageBox.question(
                dlg,
                "Delete Saved Equation",
                f"Delete saved performance equation '{str(entry.get('name') or '').strip()}'?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            try:
                be.td_perf_delete_saved_equation(self._project_dir, entry.get("id"))
                _refresh_table()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, "Delete Saved Equation", str(exc))

        tbl.itemSelectionChanged.connect(_sync_detail)
        btn_export_excel.clicked.connect(_export_all_to_excel)
        btn_export_matlab.clicked.connect(_export_all_to_matlab)
        btn_delete.clicked.connect(_delete_selected)
        btn_close.clicked.connect(dlg.accept)
        _refresh_table()
        dlg.exec()

    def _select_perf_equation_row(self, stat: str) -> None:
        st = str(stat or "").strip().lower()
        if not st or not hasattr(self, "tbl_perf_equations"):
            return
        for i in range(self.tbl_perf_equations.rowCount()):
            it = self.tbl_perf_equations.item(i, 0)
            if it and it.text().strip().lower() == st:
                try:
                    self.tbl_perf_equations.selectRow(i)
                except Exception:
                    pass
                break

    def _update_perf_highlight_models(self) -> None:
        if not getattr(self, "_perf_results_by_stat", None):
            return
        hi_sn = str(getattr(self, "_highlight_sn", "") or "").strip()
        fit_enabled = bool(getattr(self, "cb_perf_fit", None) and self.cb_perf_fit.isChecked())
        if not fit_enabled:
            for st, r in self._perf_results_by_stat.items():
                if isinstance(r, dict):
                    r["highlight_serial"] = hi_sn
                    r["highlight_model"] = {}
            return
        for st, r in self._perf_results_by_stat.items():
            if not isinstance(r, dict):
                continue
            if str(r.get("highlight_serial") or "") == hi_sn:
                continue
            plot_dimension = str(r.get("plot_dimension") or "2d").strip().lower()
            if plot_dimension == "3d":
                point_map = r.get("points_3d") or {}
                pts = point_map.get(hi_sn) if isinstance(point_map, dict) and hi_sn else None
            else:
                curves = r.get("curves") or {}
                pts = curves.get(hi_sn) if isinstance(curves, dict) and hi_sn else None
            if not isinstance(pts, list) or not pts:
                r["highlight_serial"] = hi_sn
                r["highlight_model"] = {}
                continue
            try:
                if plot_dimension == "3d":
                    if str(((r.get("master_model") or {}).get("fit_family") or "")).strip().lower() == str(
                        getattr(be, "TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD", "quadratic_surface_control_period")
                    ):
                        model = None
                    else:
                        x1s = [float(p[0]) for p in pts]
                        x2s = [float(p[1]) for p in pts]
                        ys = [float(p[2]) for p in pts]
                        model = self._perf_fit_surface_for_points(x1s, x2s, ys) if fit_enabled else None
                else:
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    model = self._perf_fit_model_for_points(xs, ys) if fit_enabled else None
            except Exception:
                model = None
            r["highlight_serial"] = hi_sn
            r["highlight_model"] = model or {}

    def _redraw_performance_view(self) -> None:
        if not getattr(self, "_perf_results_by_stat", None):
            return
        st = ""
        if hasattr(self, "cb_perf_view_stat"):
            st = str(self.cb_perf_view_stat.currentText() or "").strip().lower()
        if not st:
            return
        r = (self._perf_results_by_stat or {}).get(st) or {}
        if not isinstance(r, dict):
            return
        plot_dimension = str(r.get("plot_dimension") or "2d").strip().lower()
        self._ensure_main_axes(plot_dimension)
        hi_sn = str(getattr(self, "_highlight_sn", "") or "").strip()
        self._render_performance_result(self._axes, r, highlight_serial=hi_sn, select_equation_row=True)
        self._main_plot_legend_entries = self._apply_interactive_legend_policy(
            self._axes,
            overflow_button=getattr(self, "btn_plot_legend", None),
        )
        self._apply_plot_view_bands_to_axes(self._axes, mode="performance")
        self._refresh_plot_note()
        try:
            self._canvas.draw()
        except Exception:
            pass
        self._capture_main_plot_base_view()

    def _open_performance_tabs_dialog(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Performance",
            "The per-serial Performance tabs view was removed in Performance Plotter v2.\n\n"
            "Use the main Performance plot (all serials) and the Highlight Serial selector to inspect a single unit.",
        )
        return
        if not self._plot_ready or not self._db_path:
            return
        serials = self._selected_perf_serials()
        if not serials:
            QtWidgets.QMessageBox.information(self, "Performance", "Select at least one serial.")
            return
        if len(serials) > 40:
            QtWidgets.QMessageBox.information(
                self, "Performance", f"Too many serials selected ({len(serials)}). Select 40 or fewer for tabs."
            )
            return
        plotter = self._selected_perf_plotter()
        if not plotter:
            return
        x_spec = plotter.get("x") or {}
        y_spec = plotter.get("y") or {}
        x_target = str((x_spec.get("column") if isinstance(x_spec, dict) else "") or "").strip()
        y_target = str((y_spec.get("column") if isinstance(y_spec, dict) else "") or "").strip()
        x_stat = str((x_spec.get("stat") if isinstance(x_spec, dict) else "mean") or "mean").strip().lower()
        y_stat = str((y_spec.get("stat") if isinstance(y_spec, dict) else "mean") or "mean").strip().lower()
        runs = self._selected_perf_runs()
        if not runs:
            QtWidgets.QMessageBox.information(self, "Performance", "Select at least one run/condition.")
            return

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Performance", f"Plotting unavailable: {exc}")
            return

        run_maps: list[tuple[str, str, dict[str, float], dict[str, float]]] = []
        for rn in runs:
            x_col = self._resolve_td_y_name(rn, x_target) or x_target
            y_col = self._resolve_td_y_name(rn, y_target) or y_target
            run_maps.append(
                (
                    rn,
                    self._run_display_text(rn),
                    self._load_metric_map(rn, x_col, x_stat),
                    self._load_metric_map(rn, y_col, y_stat),
                )
            )

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Performance Curves")
        dlg.resize(1040, 780)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)

        for sn in serials:
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title(f"{sn} — {str(plotter.get('name') or '').strip() or 'Performance'}")
            ax.set_xlabel(f"{x_target}.{x_stat}")
            ax.set_ylabel(f"{y_target}.{y_stat}")
            pts: list[tuple[float, float, str]] = []
            for _rn, dn, xmap, ymap in run_maps:
                if sn not in xmap or sn not in ymap:
                    continue
                pts.append((float(xmap[sn]), float(ymap[sn]), dn or _rn))
            pts.sort(key=lambda t: t[0])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", linewidth=1.4)
                for x, y, lbl in pts:
                    ax.annotate(str(lbl), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.75)
            ax.grid(True, alpha=0.25)
            try:
                fig.tight_layout()
            except Exception:
                pass

            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            tab_layout.setContentsMargins(8, 8, 8, 8)
            canvas = FigureCanvas(fig)
            tab_layout.addWidget(canvas, 1)
            tabs.addTab(tab, sn)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        _fit_widget_to_screen(dlg)
        dlg.exec()

    def _refresh_performance_ui(self) -> None:
        if not getattr(self, "_db_path", None):
            return
        if (
            not hasattr(self, "cb_perf_x_col")
            or not hasattr(self, "cb_perf_y_col")
            or not hasattr(self, "cb_perf_z_col")
            or not hasattr(self, "list_perf_stats")
            or not hasattr(self, "cb_perf_view_stat")
        ):
            return

        try:
            cfg = be.load_excel_trend_config(be.DEFAULT_EXCEL_TREND_CONFIG)
        except Exception:
            cfg = {}
        stats = cfg.get("statistics") if isinstance(cfg, dict) else []
        self._perf_available_stats = (
            [str(s).strip().lower() for s in stats if isinstance(s, str) and str(s).strip()]
            if isinstance(stats, list)
            else ["mean", "min", "max", "std"]
        )
        if not self._perf_available_stats:
            self._perf_available_stats = ["mean", "min", "max", "std"]
        elif "mean" not in self._perf_available_stats:
            self._perf_available_stats.insert(0, "mean")

        if hasattr(self, "list_stats"):
            try:
                prev = {it.text().strip().lower() for it in self.list_stats.selectedItems() if it and it.text().strip()}
            except Exception:
                prev = set()
            prev_average = bool(
                "average" in prev
                or (hasattr(self, "cb_metric_average") and self.cb_metric_average.isChecked())
            )
            self.list_stats.blockSignals(True)
            try:
                self.list_stats.clear()
                for st in self._perf_available_stats:
                    if str(st).strip().lower() == "average":
                        continue
                    self.list_stats.addItem(QtWidgets.QListWidgetItem(st))
                restored = False
                if prev:
                    for i in range(self.list_stats.count()):
                        it = self.list_stats.item(i)
                        if it and it.text().strip().lower() in prev:
                            it.setSelected(True)
                            restored = True
                if not restored and self.list_stats.count() > 0:
                    self.list_stats.item(0).setSelected(True)
            finally:
                self.list_stats.blockSignals(False)
            if hasattr(self, "cb_metric_average"):
                self.cb_metric_average.blockSignals(True)
                self.cb_metric_average.setChecked(prev_average)
                self.cb_metric_average.blockSignals(False)
            self._refresh_metric_stats_summary()

        prev_output, prev_input1, prev_input2 = self._perf_var_names()
        prev_run_type_mode = self._selected_perf_run_type_mode()
        prev_filter_mode = self._selected_perf_filter_mode()
        prev_control_period = self._selected_perf_control_period()

        try:
            runs = be.td_list_runs(self._db_path)
        except Exception:
            runs = []
        try:
            available_run_type_modes = be.td_list_performance_run_type_modes(self._db_path)
        except Exception:
            available_run_type_modes = ["steady_state", "pulsed_mode"]
        union: dict[str, dict] = {}
        col_runs: dict[str, set[str]] = {}
        for rn in runs:
            try:
                cols = be.td_list_y_columns(self._db_path, rn)
            except Exception:
                cols = []
            for col in cols:
                name = str((col or {}).get("name") or "").strip()
                if not name:
                    continue
                units = str((col or {}).get("units") or "").strip()
                key = self._perf_norm_name(name)
                col_runs.setdefault(key, set()).add(str(rn))
                if key not in union:
                    union[key] = {"name": name, "units": units}
                elif not str(union[key].get("units") or "").strip() and units:
                    union[key]["units"] = units
        self._perf_available_columns = sorted(union.values(), key=lambda d: str(d.get("name") or "").lower())
        self._perf_col_runs = col_runs
        self._perf_all_runs = [str(r).strip() for r in runs if str(r).strip()]

        self._fill_perf_axis_combo(self.cb_perf_x_col)
        self._fill_perf_axis_combo(self.cb_perf_y_col)
        self._fill_perf_axis_combo(self.cb_perf_z_col, allow_blank=True)
        self._set_perf_axis_combo_by_norm(self.cb_perf_x_col, self._perf_norm_name(prev_input1))
        self._set_perf_axis_combo_by_norm(self.cb_perf_y_col, self._perf_norm_name(prev_output))
        self._set_perf_axis_combo_by_norm(self.cb_perf_z_col, self._perf_norm_name(prev_input2), allow_blank=True)
        available_run_type_set = {
            be.td_perf_normalize_run_type_mode(mode)
            for mode in (available_run_type_modes or [])
            if str(mode).strip()
        }
        if not available_run_type_set:
            available_run_type_set = {"steady_state", "pulsed_mode"}
        if hasattr(self, "rb_perf_steady_state") and hasattr(self, "rb_perf_pulsed_mode"):
            self.rb_perf_steady_state.blockSignals(True)
            self.rb_perf_pulsed_mode.blockSignals(True)
            self.rb_perf_steady_state.setEnabled("steady_state" in available_run_type_set)
            self.rb_perf_pulsed_mode.setEnabled("pulsed_mode" in available_run_type_set)
            desired_run_type_mode = (
                prev_run_type_mode
                if prev_run_type_mode in available_run_type_set
                else ("steady_state" if "steady_state" in available_run_type_set else "pulsed_mode")
            )
            self._set_perf_run_type_mode(desired_run_type_mode)
            self.rb_perf_steady_state.blockSignals(False)
            self.rb_perf_pulsed_mode.blockSignals(False)
        if hasattr(self, "cb_perf_filter_mode"):
            mode_idx = self.cb_perf_filter_mode.findData(prev_filter_mode)
            if mode_idx < 0:
                mode_idx = self.cb_perf_filter_mode.findData("all_conditions")
            if mode_idx >= 0:
                self.cb_perf_filter_mode.setCurrentIndex(mode_idx)
        self._refresh_perf_control_period_options()
        self._filter_perf_axis_options(changed="x")
        self._filter_perf_axis_options(changed="z")
        self._update_perf_control_period_state()

        if not getattr(self, "_perf_signals_connected", False):
            self.cb_perf_x_col.currentIndexChanged.connect(lambda *_: self._on_perf_axis_changed("x"))
            self.cb_perf_y_col.currentIndexChanged.connect(lambda *_: self._on_perf_axis_changed("y"))
            self.cb_perf_z_col.currentIndexChanged.connect(lambda *_: self._on_perf_axis_changed("z"))
            self.cb_perf_view_stat.currentIndexChanged.connect(self._on_perf_view_stat_changed)
            self.rb_perf_steady_state.toggled.connect(lambda *_: self._update_perf_control_period_state())
            self.rb_perf_steady_state.toggled.connect(lambda *_: self._clear_perf_results())
            self.rb_perf_pulsed_mode.toggled.connect(lambda *_: self._update_perf_control_period_state())
            self.rb_perf_pulsed_mode.toggled.connect(lambda *_: self._clear_perf_results())
            self.cb_perf_filter_mode.currentIndexChanged.connect(lambda *_: self._update_perf_control_period_state())
            self.cb_perf_filter_mode.currentIndexChanged.connect(lambda *_: self._clear_perf_results())
            self.cb_perf_control_period.currentIndexChanged.connect(lambda *_: self._clear_perf_results())
            self.cb_perf_fit.toggled.connect(lambda *_: self._update_perf_fit_controls())
            self.cb_perf_fit.toggled.connect(lambda *_: self._clear_perf_results())
            self.cb_perf_fit_model.currentIndexChanged.connect(lambda *_: self._update_perf_fit_controls())
            self.cb_perf_fit_model.currentIndexChanged.connect(lambda *_: self._clear_perf_results())
            self.cb_perf_surface_model.currentIndexChanged.connect(lambda *_: self._update_perf_control_period_state())
            self.cb_perf_surface_model.currentIndexChanged.connect(lambda *_: self._clear_perf_results())
            self.sp_perf_degree.valueChanged.connect(lambda *_: self._clear_perf_results())
            self.cb_perf_norm_x.toggled.connect(lambda *_: self._clear_perf_results())
            self.btn_perf_equations_popup.clicked.connect(self._open_performance_equations_popup)
            self.btn_perf_save_equation.clicked.connect(self._save_current_performance_equation)
            self.btn_perf_saved_equations.clicked.connect(self._open_saved_performance_equations_popup)
            self.btn_perf_export_interactive.clicked.connect(self._export_perf_interactive_equation_workbook)
            self.btn_perf_export_equations.clicked.connect(self._export_perf_equations_to_excel)
            self._perf_signals_connected = True

        self.cb_perf_fit.setEnabled(True)
        self.cb_perf_fit_model.setEnabled(True)
        self._perf_require_min_points = max(2, int(getattr(self, "_perf_require_min_points", 2) or 2))
        self._populate_perf_stats(self._perf_plot_stat_candidates())
        self._update_perf_fit_controls()

        if not self._perf_available_columns:
            if hasattr(self, "lbl_perf_axes"):
                self.lbl_perf_axes.setText("Output: - | Input 1: - | Input 2: -")
            if hasattr(self, "lbl_perf_common_runs"):
                self.lbl_perf_common_runs.setText("Common runs for selected variables: 0 (no test data metric columns are available)")
            self._clear_perf_results()
            self._schedule_mode_panel_height_sync()
            return

        self._on_perf_axis_changed()
        self._schedule_mode_panel_height_sync()

    def _on_perf_preset_changed(self) -> None:
        self._on_perf_axis_changed()

    def _perf_partial_cp_fit_messages(self, stats: list[str]) -> list[str]:
        if not isinstance(stats, list) or not stats:
            return []
        cp_surface_family = str(
            getattr(be, "TD_PERF_FIT_FAMILY_QUADRATIC_SURFACE_CONTROL_PERIOD", "quadratic_surface_control_period")
        ).strip().lower()
        format_warning = getattr(be, "_td_perf_format_surface_control_period_warning", None)
        messages: list[str] = []
        for stat in stats:
            result = (self._perf_results_by_stat or {}).get(stat) or {}
            if not isinstance(result, dict):
                continue
            master_model = result.get("master_model") or {}
            if not isinstance(master_model, dict) or not master_model:
                continue
            fit_family = str(
                master_model.get("fit_family") or result.get("surface_fit_family") or ""
            ).strip().lower()
            if fit_family != cp_surface_family:
                continue
            ignored = [
                dict(entry)
                for entry in (master_model.get("ignored_control_periods") or [])
                if isinstance(entry, dict)
            ]
            if not ignored:
                continue
            if callable(format_warning):
                text = str(format_warning(ignored) or "").strip()
            else:
                text = str(
                    master_model.get("fit_warning_text")
                    or result.get("fit_warning_text")
                    or ""
                ).strip()
            if text and text not in messages:
                messages.append(text)
        return messages

    def _plot_performance(self, *, user_initiated: bool = False) -> None:
        if not self._plot_ready or not self._db_path:
            return
        self._set_plot_note("")
        output_target, input1_target, input2_target = self._perf_var_names()
        if not output_target or not input1_target:
            QtWidgets.QMessageBox.information(self, "Performance", "Select Output and Input 1 columns.")
            return
        chosen = [nm for nm in (output_target, input1_target, input2_target) if str(nm).strip()]
        if len({self._perf_norm_name(v) for v in chosen}) != len(chosen):
            QtWidgets.QMessageBox.information(self, "Performance", "Output and selected inputs must be different columns.")
            return
        plot_stats = self._perf_plot_stat_candidates()

        runs = self._selected_perf_runs()
        if not runs:
            QtWidgets.QMessageBox.information(self, "Performance", "No runs found in cache.")
            return
        serials = self._selected_perf_serials()
        if not serials:
            QtWidgets.QMessageBox.information(self, "Performance", "No serial numbers found in cache.")
            return

        common_runs = self._common_runs_for_perf_vars(output_target, input1_target, input2_target)
        if not common_runs:
            QtWidgets.QMessageBox.information(
                self, "Performance", "All selected variables must exist on at least one common run/condition."
            )
            return
        run_order = {r: i for i, r in enumerate(getattr(self, "_perf_all_runs", runs) or [])}
        runs = sorted([r for r in runs if r in set(common_runs)], key=lambda r: run_order.get(r, 10**9))

        fit_enabled = bool(getattr(self, "cb_perf_fit", None) and self.cb_perf_fit.isChecked())
        require_min_points = max(2, int(getattr(self, "_perf_require_min_points", 2) or 2))
        run_type_mode = self._selected_perf_run_type_mode()
        perf_filter_mode = self._selected_perf_filter_mode()
        control_period_filter = self._selected_perf_control_period()
        self._perf_results_by_stat, plot_view_stats, fit_error_text = self._perf_collect_results(
            output_target,
            input1_target,
            input2_target,
            plot_stats,
            runs,
            serials,
            fit_enabled=fit_enabled,
            require_min_points=require_min_points,
            run_type_filter=run_type_mode,
            control_period_filter=(control_period_filter if perf_filter_mode == "match_control_period" else None),
            display_control_period=control_period_filter,
        )

        if not plot_view_stats:
            message = "No qualifying performance data found for the selected Output/Input pairing."
            QtWidgets.QMessageBox.information(self, "Performance", message)
            return

        self._perf_plot_view_stats = list(plot_view_stats)
        self._populate_perf_stats(plot_view_stats)
        prev_view = str(self.cb_perf_view_stat.currentText() or "").strip().lower() if hasattr(self, "cb_perf_view_stat") else ""
        if hasattr(self, "cb_perf_view_stat"):
            self.cb_perf_view_stat.blockSignals(True)
            self.cb_perf_view_stat.clear()
            for st in plot_view_stats:
                self.cb_perf_view_stat.addItem(st, st)
            if prev_view and prev_view in plot_view_stats:
                self.cb_perf_view_stat.setCurrentText(prev_view)
            else:
                self.cb_perf_view_stat.setCurrentIndex(0)
            self.cb_perf_view_stat.blockSignals(False)

        qualifying_serials = {
            sn
            for st in plot_view_stats
            for sn in (
                ((self._perf_results_by_stat.get(st) or {}).get("points_3d") or {}).keys()
                if str(((self._perf_results_by_stat.get(st) or {}).get("plot_dimension") or "2d")).strip().lower() == "3d"
                else ((self._perf_results_by_stat.get(st) or {}).get("curves") or {}).keys()
            )
        }
        self._update_perf_pair_summary(
            stats=plot_view_stats,
            qualifying_serial_count=len(qualifying_serials),
            require_min_points=require_min_points,
            plot_dimension=("3d" if str(input2_target or "").strip() else "2d"),
        )
        fit_warning_notes: list[str] = []
        for stat in plot_view_stats:
            result = (self._perf_results_by_stat or {}).get(stat) or {}
            warning_text = str(result.get("fit_warning_text") or ((result.get("master_model") or {}).get("fit_warning_text") if isinstance(result.get("master_model"), dict) else "") or "").strip()
            if warning_text and warning_text not in fit_warning_notes:
                fit_warning_notes.append(warning_text)
        has_master_model = any(
            isinstance(((self._perf_results_by_stat or {}).get(stat) or {}).get("master_model"), dict)
            and bool((((self._perf_results_by_stat or {}).get(stat) or {}).get("master_model") or {}))
            for stat in plot_view_stats
        )
        if fit_error_text and not has_master_model:
            QtWidgets.QMessageBox.warning(self, "Performance", fit_error_text)
        partial_cp_messages = self._perf_partial_cp_fit_messages(plot_view_stats)
        if user_initiated and partial_cp_messages:
            QtWidgets.QMessageBox.information(self, "Performance", "\n".join(partial_cp_messages))
        self._set_plot_note("\n".join(fit_warning_notes))
        self._update_perf_highlight_models()
        self._fill_perf_equations_table()
        self._redraw_performance_view()
        self.btn_save_plot_pdf.setEnabled(True)
        selection = self._current_run_selection()
        current_stat = str(self.cb_perf_view_stat.currentText() or "").strip().lower() if hasattr(self, "cb_perf_view_stat") else ""
        master_result = (self._perf_results_by_stat or {}).get(current_stat) or {}
        master_model = (master_result or {}).get("master_model") or {}
        self._last_plot_def = {
            "mode": "performance",
            "performance_plot_method": "legacy_serial_curves",
            "selector_mode": str(selection.get("mode") or "sequence"),
            "selection_id": str(selection.get("id") or ""),
            "display_text": self._selection_display_text(selection),
            "run_condition": self._selection_condition_label(selection),
            "member_sequences": list(selection.get("member_sequences") or []),
            "member_runs": list(runs),
            "output": output_target,
            "input1": input1_target,
            "input2": input2_target,
            "plot_dimension": "3d" if str(input2_target or "").strip() else "2d",
            "stats": list(plot_view_stats),
            "view_stat": current_stat or (plot_view_stats[0] if plot_view_stats else "mean"),
            "fit_enabled": bool(fit_enabled),
            "fit_mode": str((master_model or {}).get("fit_mode") or self._perf_requested_fit_mode()),
            "fit_family": str((master_model or {}).get("fit_family") or ""),
            "surface_fit_family": self._perf_requested_surface_family(),
            "auto_surface_families": bool(self._perf_requested_surface_family() == "auto_surface"),
            "performance_run_type_mode": run_type_mode,
            "performance_filter_mode": perf_filter_mode,
            "selected_control_period": control_period_filter,
            "highlight_serial": str(getattr(self, "_highlight_sn", "") or "").strip(),
        }
        self.btn_add_auto_plot.setEnabled(True)

    def _plot_performance_cached_condition_means(self, *, user_initiated: bool = False) -> None:
        if not self._plot_ready or not self._db_path:
            return
        self._set_plot_note("")
        output_target, input1_target, input2_target = self._perf_var_names()
        if not output_target or not input1_target:
            QtWidgets.QMessageBox.information(self, "Performance", "Select Output and Input 1 columns.")
            return
        chosen = [nm for nm in (output_target, input1_target, input2_target) if str(nm).strip()]
        if len({self._perf_norm_name(v) for v in chosen}) != len(chosen):
            QtWidgets.QMessageBox.information(
                self,
                "Performance",
                "Output and selected inputs must be different columns.",
            )
            return
        plot_stats = self._perf_plot_stat_candidates()

        runs = self._selected_perf_runs()
        if not runs:
            QtWidgets.QMessageBox.information(self, "Performance", "No runs found in cache.")
            return
        serials = self._selected_perf_serials()
        if not serials:
            QtWidgets.QMessageBox.information(self, "Performance", "No serial numbers found in cache.")
            return

        common_runs = self._common_runs_for_perf_vars(output_target, input1_target, input2_target)
        if not common_runs:
            QtWidgets.QMessageBox.information(
                self,
                "Performance",
                "All selected variables must exist on at least one common run/condition.",
            )
            return
        run_order = {r: i for i, r in enumerate(getattr(self, "_perf_all_runs", runs) or [])}
        runs = sorted([r for r in runs if r in set(common_runs)], key=lambda r: run_order.get(r, 10**9))

        fit_enabled = bool(getattr(self, "cb_perf_fit", None) and self.cb_perf_fit.isChecked())
        require_min_points = max(2, int(getattr(self, "_perf_require_min_points", 2) or 2))
        run_type_mode = self._selected_perf_run_type_mode()
        perf_filter_mode = self._selected_perf_filter_mode()
        control_period_filter = self._selected_perf_control_period()
        self._perf_results_by_stat, plot_view_stats, fit_error_text = self._perf_collect_cached_condition_mean_results(
            output_target,
            input1_target,
            input2_target,
            plot_stats,
            runs,
            serials,
            fit_enabled=fit_enabled,
            require_min_points=require_min_points,
            run_type_filter=run_type_mode,
            control_period_filter=(control_period_filter if perf_filter_mode == "match_control_period" else None),
            display_control_period=control_period_filter,
        )

        if not plot_view_stats:
            message = "No qualifying performance data found for the selected Output/Input pairing."
            QtWidgets.QMessageBox.information(self, "Performance", message)
            return

        self._perf_plot_view_stats = list(plot_view_stats)
        self._populate_perf_stats(plot_view_stats)
        prev_view = str(self.cb_perf_view_stat.currentText() or "").strip().lower() if hasattr(self, "cb_perf_view_stat") else ""
        if hasattr(self, "cb_perf_view_stat"):
            self.cb_perf_view_stat.blockSignals(True)
            self.cb_perf_view_stat.clear()
            for st in plot_view_stats:
                self.cb_perf_view_stat.addItem(st, st)
            if prev_view and prev_view in plot_view_stats:
                self.cb_perf_view_stat.setCurrentText(prev_view)
            else:
                self.cb_perf_view_stat.setCurrentIndex(0)
            self.cb_perf_view_stat.blockSignals(False)

        qualifying_serials = {
            sn
            for st in plot_view_stats
            for sn in (
                ((self._perf_results_by_stat.get(st) or {}).get("points_3d") or {}).keys()
                if str(((self._perf_results_by_stat.get(st) or {}).get("plot_dimension") or "2d")).strip().lower() == "3d"
                else ((self._perf_results_by_stat.get(st) or {}).get("curves") or {}).keys()
            )
        }
        self._update_perf_pair_summary(
            stats=plot_view_stats,
            qualifying_serial_count=len(qualifying_serials),
            require_min_points=require_min_points,
            plot_dimension=("3d" if str(input2_target or "").strip() else "2d"),
        )
        fit_warning_notes: list[str] = []
        for stat in plot_view_stats:
            result = (self._perf_results_by_stat or {}).get(stat) or {}
            warning_text = str(
                result.get("fit_warning_text")
                or ((result.get("master_model") or {}).get("fit_warning_text") if isinstance(result.get("master_model"), dict) else "")
                or ""
            ).strip()
            if warning_text and warning_text not in fit_warning_notes:
                fit_warning_notes.append(warning_text)
        has_master_model = any(
            isinstance(((self._perf_results_by_stat or {}).get(stat) or {}).get("master_model"), dict)
            and bool((((self._perf_results_by_stat or {}).get(stat) or {}).get("master_model") or {}))
            for stat in plot_view_stats
        )
        if fit_error_text and not has_master_model:
            QtWidgets.QMessageBox.warning(self, "Performance", fit_error_text)
        partial_cp_messages = self._perf_partial_cp_fit_messages(plot_view_stats)
        if user_initiated and partial_cp_messages:
            QtWidgets.QMessageBox.information(self, "Performance", "\n".join(partial_cp_messages))
        self._set_plot_note("\n".join(fit_warning_notes))
        self._update_perf_highlight_models()
        self._fill_perf_equations_table()
        self._redraw_performance_view()
        self.btn_save_plot_pdf.setEnabled(True)
        selection = self._current_run_selection()
        current_stat = str(self.cb_perf_view_stat.currentText() or "").strip().lower() if hasattr(self, "cb_perf_view_stat") else ""
        master_result = (self._perf_results_by_stat or {}).get(current_stat) or {}
        master_model = (master_result or {}).get("master_model") or {}
        self._last_plot_def = {
            "mode": "performance",
            "performance_plot_method": "cached_condition_means",
            "selector_mode": str(selection.get("mode") or "sequence"),
            "selection_id": str(selection.get("id") or ""),
            "display_text": self._selection_display_text(selection),
            "run_condition": self._selection_condition_label(selection),
            "member_sequences": list(selection.get("member_sequences") or []),
            "member_runs": list(runs),
            "output": output_target,
            "input1": input1_target,
            "input2": input2_target,
            "plot_dimension": "3d" if str(input2_target or "").strip() else "2d",
            "stats": list(plot_view_stats),
            "view_stat": current_stat or (plot_view_stats[0] if plot_view_stats else "mean"),
            "fit_enabled": bool(fit_enabled),
            "fit_mode": str((master_model or {}).get("fit_mode") or self._perf_requested_fit_mode()),
            "fit_family": str((master_model or {}).get("fit_family") or ""),
            "surface_fit_family": self._perf_requested_surface_family(),
            "auto_surface_families": bool(self._perf_requested_surface_family() == "auto_surface"),
            "performance_run_type_mode": run_type_mode,
            "performance_filter_mode": perf_filter_mode,
            "selected_control_period": control_period_filter,
            "highlight_serial": str(getattr(self, "_highlight_sn", "") or "").strip(),
        }
        self.btn_add_auto_plot.setEnabled(True)

    def _save_plot_pdf(self) -> None:
        if not self._plot_ready or not self._figure:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Plot PDF",
            str(self._project_dir / "test_data_plot.pdf"),
            "PDF Files (*.pdf)",
        )
        if not path:
            return
        try:
            self._figure.savefig(path, format="pdf")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Plot PDF", str(exc))

    def _load_auto_plots(self) -> None:
        self._auto_plots = []
        try:
            if self._auto_plot_path.exists():
                data = json.loads(self._auto_plot_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._auto_plots = [d for d in data if isinstance(d, dict)]
        except Exception:
            self._auto_plots = []
        self._sync_main_auto_plot_actions()
        self._refresh_auto_plots_list()

    def _refresh_auto_plots_list(self, list_widget: QtWidgets.QListWidget | None = None) -> None:
        widget = list_widget if list_widget is not None else getattr(self, "list_auto_plots", None)
        if widget is None:
            return
        widget.clear()
        for d in self._auto_plots:
            item = QtWidgets.QListWidgetItem(self._auto_plot_display_name(d) or "Auto-Plot")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, d)
            widget.addItem(item)
        self._update_auto_actions(list_widget=widget)

    def _update_auto_actions(
        self,
        *,
        list_widget: QtWidgets.QListWidget | None = None,
        btn_open: QtWidgets.QPushButton | None = None,
        btn_open_all: QtWidgets.QPushButton | None = None,
        btn_delete: QtWidgets.QPushButton | None = None,
        btn_save_all: QtWidgets.QPushButton | None = None,
        btn_auto_report: QtWidgets.QPushButton | None = None,
    ) -> None:
        widget = list_widget if list_widget is not None else getattr(self, "list_auto_plots", None)
        selected = widget.selectedItems() if widget is not None else []
        has = bool(selected)
        open_btn = btn_open if btn_open is not None else getattr(self, "btn_open_auto", None)
        open_all_btn = btn_open_all if btn_open_all is not None else getattr(self, "btn_open_all_auto", None)
        delete_btn = btn_delete if btn_delete is not None else getattr(self, "btn_delete_auto", None)
        save_all_btn = btn_save_all if btn_save_all is not None else getattr(self, "btn_save_all_auto", None)
        auto_report_btn = btn_auto_report if btn_auto_report is not None else getattr(self, "btn_auto_report", None)
        if open_btn is not None:
            open_btn.setEnabled(has and self._plot_ready and bool(self._db_path))
        if open_all_btn is not None:
            open_all_btn.setEnabled(bool(self._auto_plots) and self._plot_ready and bool(self._db_path))
        if delete_btn is not None:
            delete_btn.setEnabled(has)
        if save_all_btn is not None:
            save_all_btn.setEnabled(bool(self._auto_plots) and self._plot_ready and bool(self._db_path))
        if auto_report_btn is not None:
            auto_report_btn.setEnabled(bool(self._plot_ready and self._db_path))
        self._sync_main_auto_plot_actions()

    def _add_current_plot_to_autoplots(self) -> None:
        if not self._last_plot_def:
            return
        d = dict(self._last_plot_def)
        if "name" not in d:
            selection = self._selection_from_plot_def(d) or self._current_run_selection()
            mode = str(d.get("mode") or "").strip().lower()
            if mode == "curves":
                run_disp = self._selection_display_text(selection)
                y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                x = str(d.get("x") or "").strip()
                d["name"] = f"Curves: {run_disp} {y} vs {x}".strip()
            elif mode == "performance":
                output = str(d.get("output") or "").strip()
                input1 = str(d.get("input1") or "").strip()
                input2 = str(d.get("input2") or "").strip()
                d["name"] = (
                    f"Performance: {output} vs {input1},{input2}".strip()
                    if input2
                    else f"Performance: {output} vs {input1}".strip()
                )
            else:
                run_disp = self._selection_display_text(selection)
                y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                stats_val = d.get("stats")
                stats = (
                    [str(x).strip() for x in stats_val if str(x).strip()]
                    if isinstance(stats_val, list)
                    else []
                )
                if not stats:
                    st = str(d.get("stat") or "").strip()
                    if st:
                        stats = [st]
                stats_label = self._metric_title_suffix(stats) or "metrics"
                d["name"] = f"Metrics: {run_disp} {stats_label} ({y})".strip()
        self._auto_plots.append(d)
        try:
            self._auto_plot_path.write_text(json.dumps(self._auto_plots, indent=2), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Auto-Plots", str(exc))
            return
        self._sync_main_auto_plot_actions()
        self._refresh_auto_plots_list()

    def _open_selected_auto_plot(
        self,
        plot_def: dict | None = None,
        *,
        list_widget: QtWidgets.QListWidget | None = None,
    ) -> None:
        if not self._db_path or not self._plot_ready:
            return
        d = dict(plot_def) if isinstance(plot_def, dict) else None
        if d is None:
            selected = self._selected_auto_plot_definitions(list_widget=list_widget)
            if not selected:
                return
            d = selected[0]
        if not isinstance(d, dict):
            return
        mode = str(d.get("mode") or "").strip().lower()
        if mode not in {"curves", "metrics", "performance"}:
            return
        want_mode = str(d.get("selector_mode") or "sequence").strip().lower()
        if hasattr(self, "cb_run_mode"):
            idx = self.cb_run_mode.findData(want_mode)
            if idx >= 0:
                self.cb_run_mode.setCurrentIndex(idx)
        selection = self._selection_from_plot_def(d)
        selection_ids = [str(v).strip() for v in (d.get("selection_ids") or []) if str(v).strip()] if isinstance(d.get("selection_ids"), list) else []
        sel_id = str(selection.get("id") or d.get("selection_id") or "").strip()
        if mode == "metrics" and want_mode == "condition":
            self._set_mode("metrics")
            restore_ids = list(selection_ids)
            if not restore_ids and sel_id:
                restore_ids = [sel_id]
            self._set_metric_condition_selection_ids(restore_ids if restore_ids else None)
        if sel_id and not (mode == "metrics" and want_mode == "condition"):
            self._select_run_by_id(sel_id)
        self._set_mode(mode)
        if mode == "curves":
            ys = d.get("y") or []
            if isinstance(ys, list) and ys:
                self._set_combo_to_value(self.cb_y_curve, str(ys[0]))
            x = str(d.get("x") or "").strip()
            if x:
                def _norm_name(s: str) -> str:
                    return "".join(ch.lower() for ch in str(s or "") if ch.isalnum())

                time_norms = {_norm_name(v) for v in ("time", "time_s", "time(sec)", "time(s)", "time (s)", "time_sec", "times")}
                pulse_norms = {_norm_name(v) for v in ("pulse number", "pulse#", "pulse #", "pulse_number", "pulsenumber", "cycle")}
                want = x
                if _norm_name(want) in time_norms:
                    want = "Time"
                elif _norm_name(want) in pulse_norms:
                    want = "Pulse Number"
                self._set_combo_to_value(self.cb_x, want)
            self._plot_curves()
        elif mode == "metrics":
            stats_val = d.get("stats")
            stats = (
                [str(x).strip().lower() for x in stats_val if str(x).strip()]
                if isinstance(stats_val, list)
                else []
            )
            if not stats:
                st = str(d.get("stat") or "").strip().lower()
                if st:
                    stats = [st]
            if not stats:
                stats = ["mean"]
            want_stats = {s for s in stats if s}
            want_average = "average" in want_stats
            want_stats.discard("average")
            if hasattr(self, "cb_metric_average"):
                self.cb_metric_average.setChecked(want_average)
            self._set_metric_plot_source(d.get("metric_plot_source"))
            if want_stats and hasattr(self, "list_stats"):
                self.list_stats.clearSelection()
                for i in range(self.list_stats.count()):
                    it = self.list_stats.item(i)
                    if it.text().strip().lower() in want_stats:
                        it.setSelected(True)
            elif hasattr(self, "list_stats"):
                self.list_stats.clearSelection()
            ys = d.get("y") or []
            want = {str(x).strip() for x in ys if str(x).strip()} if isinstance(ys, list) else set()
            if want and hasattr(self, "list_y_metrics"):
                self.list_y_metrics.clearSelection()
                for i in range(self.list_y_metrics.count()):
                    it = self.list_y_metrics.item(i)
                    if it.text() in want:
                        it.setSelected(True)
            if hasattr(self, "cb_plot_metric_bounds"):
                self.cb_plot_metric_bounds.setChecked(bool(d.get("plot_bounds")))
            self._plot_metrics()
        else:
            self._set_mode("performance")
            self._refresh_performance_ui()
            performance_plot_method = self._perf_normalize_plot_method(d.get("performance_plot_method"))
            run_type_mode = str(d.get("performance_run_type_mode") or "").strip().lower()
            if run_type_mode:
                self._set_perf_run_type_mode(run_type_mode)
            filter_mode = str(d.get("performance_filter_mode") or "all_conditions").strip().lower()
            if hasattr(self, "cb_perf_filter_mode"):
                idx = self.cb_perf_filter_mode.findData(filter_mode)
                if idx >= 0:
                    self.cb_perf_filter_mode.setCurrentIndex(idx)
            selected_cp = d.get("selected_control_period")
            if selected_cp not in (None, "") and hasattr(self, "cb_perf_control_period"):
                idx = self.cb_perf_control_period.findData(selected_cp)
                if idx >= 0:
                    self.cb_perf_control_period.setCurrentIndex(idx)
            self._update_perf_control_period_state()
            self._set_combo_to_value(self.cb_perf_y_col, str(d.get("output") or ""))
            self._set_combo_to_value(self.cb_perf_x_col, str(d.get("input1") or ""))
            self._set_combo_to_value(self.cb_perf_z_col, str(d.get("input2") or ""))
            self._on_perf_axis_changed("z" if str(d.get("input2") or "").strip() else "x")
            if hasattr(self, "cb_perf_fit"):
                self.cb_perf_fit.setChecked(bool(d.get("fit_enabled", True)))
            fit_mode = str(d.get("fit_mode") or "").strip().lower()
            if fit_mode and hasattr(self, "cb_perf_fit_model"):
                idx = self.cb_perf_fit_model.findData(fit_mode)
                if idx >= 0:
                    self.cb_perf_fit_model.setCurrentIndex(idx)
            surface_fit_family = str(
                d.get("surface_fit_family")
                or ("auto_surface" if bool(d.get("auto_surface_families")) else "quadratic_surface")
            ).strip().lower()
            if hasattr(self, "cb_perf_surface_model"):
                idx = self.cb_perf_surface_model.findData(surface_fit_family)
                if idx >= 0:
                    self.cb_perf_surface_model.setCurrentIndex(idx)
            if performance_plot_method == "cached_condition_means":
                self._plot_performance_cached_condition_means()
            else:
                self._plot_performance()
            view_stat = str(d.get("view_stat") or "").strip().lower()
            if view_stat and hasattr(self, "cb_perf_view_stat"):
                idx = self.cb_perf_view_stat.findData(view_stat)
                if idx >= 0:
                    self.cb_perf_view_stat.setCurrentIndex(idx)
                elif self.cb_perf_view_stat.count() > 0:
                    for i in range(self.cb_perf_view_stat.count()):
                        if str(self.cb_perf_view_stat.itemText(i) or "").strip().lower() == view_stat:
                            self.cb_perf_view_stat.setCurrentIndex(i)
                            break
            self._redraw_performance_view()

    def _open_auto_plots_popup(self) -> None:
        if not self._plot_ready or not self._db_path:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "Plotting is unavailable.")
            return
        if not self._auto_plots:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No auto-plots available.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Auto-Plots")
        dlg.resize(760, 560)
        dlg.setStyleSheet(
            """
            QDialog { background: #ffffff; color: #0f172a; }
            QLabel { color: #0f172a; }
            QListWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item { padding: 4px 6px; }
            QListWidget::item:selected { background: #dbeafe; color: #1e3a8a; }
            """
        )
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Saved Auto-Plots")
        title.setStyleSheet("font-size: 13px; font-weight: 800;")
        layout.addWidget(title)

        hint = QtWidgets.QLabel("Open a saved plot in the main viewer, open all in tabs, or manage saved entries.")
        hint.setStyleSheet("color: #64748b; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(list_widget, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_open = QtWidgets.QPushButton("Open")
        btn_open_all = QtWidgets.QPushButton("Open All")
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_save_all = QtWidgets.QPushButton("Save All PDF")
        btn_auto_report = QtWidgets.QPushButton("Auto Report...")
        btn_close = QtWidgets.QPushButton("Close")
        for btn in (btn_open, btn_open_all, btn_delete, btn_save_all, btn_auto_report):
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        def _sync_buttons() -> None:
            self._update_auto_actions(
                list_widget=list_widget,
                btn_open=btn_open,
                btn_open_all=btn_open_all,
                btn_delete=btn_delete,
                btn_save_all=btn_save_all,
                btn_auto_report=btn_auto_report,
            )

        def _delete_selected() -> None:
            self._delete_selected_auto_plots(list_widget=list_widget)
            _sync_buttons()

        self._refresh_auto_plots_list(list_widget)
        _sync_buttons()
        list_widget.itemDoubleClicked.connect(lambda *_: self._open_selected_auto_plot(list_widget=list_widget))
        list_widget.itemSelectionChanged.connect(_sync_buttons)
        btn_open.clicked.connect(lambda: self._open_selected_auto_plot(list_widget=list_widget))
        btn_open_all.clicked.connect(self._open_all_auto_plots_panel)
        btn_delete.clicked.connect(_delete_selected)
        btn_save_all.clicked.connect(self._save_all_auto_plots_pdf)
        btn_auto_report.clicked.connect(self._open_auto_report_options)
        btn_close.clicked.connect(dlg.accept)

        _fit_widget_to_screen(dlg)
        dlg.exec()

    def _open_all_auto_plots_panel(self) -> None:
        if not self._plot_ready or not self._db_path:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "Plotting is unavailable.")
            return
        if not self._auto_plots:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No auto-plots available.")
            return

        plots: list[tuple[str, dict]] = []
        for d in self._auto_plots:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "").strip()
            if not name:
                mode = str(d.get("mode") or "").strip()
                run_ref = str(d.get("run") or "").strip()
                run = self._run_name_by_display.get(run_ref, run_ref)
                run_disp = self._run_display_text(run)
                if mode == "curves":
                    y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                    x = str(d.get("x") or "").strip()
                    name = f"Curves: {run_disp} {y} vs {x}".strip()
                elif mode == "performance":
                    output = str(d.get("output") or "").strip()
                    input1 = str(d.get("input1") or "").strip()
                    input2 = str(d.get("input2") or "").strip()
                    name = (
                        f"Performance: {output} vs {input1},{input2}".strip()
                        if input2
                        else f"Performance: {output} vs {input1}".strip()
                    )
                else:
                    y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                    stats_val = d.get("stats")
                    stats = (
                        [str(x).strip() for x in stats_val if str(x).strip()]
                        if isinstance(stats_val, list)
                        else []
                    )
                    if not stats:
                        st = str(d.get("stat") or "").strip()
                        if st:
                            stats = [st]
                    stats_label = self._metric_title_suffix(stats) or "metrics"
                    name = f"Metrics: {run_disp} {stats_label} ({y})".strip()
            plots.append((name or "Auto Plot", d))

        if not plots:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No auto-plots available.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Auto-Plots")
        dlg.resize(980, 720)
        dlg.setStyleSheet(
            """
            QDialog { background: #ffffff; color: #1f2937; }
            QLabel { color: #1f2937; }
            QTabWidget::pane { border: 1px solid #e2e8f0; }
            QTabBar::tab {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                color: #0f172a;
                padding: 6px 10px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0f172a;
                border-bottom-color: #ffffff;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc:
            QtWidgets.QMessageBox.warning(dlg, "Auto-Plots", f"Plotting unavailable: {exc}")
            return

        for name, d in plots:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            tab_layout.setContentsMargins(8, 8, 8, 8)
            fig = self._render_plot_def_to_figure(d)
            ax = fig.axes[0] if getattr(fig, "axes", None) else None
            canvas = FigureCanvas(fig)

            # Zoom controls (also supports mouse wheel on the canvas).
            if ax is not None:
                base_xlim = ax.get_xlim()
                base_ylim = ax.get_ylim()

                ctrl = QtWidgets.QHBoxLayout()
                btn_mag = QtWidgets.QPushButton("Magnify")
                btn_mag.setCheckable(True)
                btn_out = QtWidgets.QPushButton("Zoom -")
                btn_in = QtWidgets.QPushButton("Zoom +")
                btn_reset = QtWidgets.QPushButton("Reset")
                for b in (btn_mag, btn_out, btn_in, btn_reset):
                    b.setStyleSheet(
                        """
                        QPushButton {
                            padding: 5px 10px;
                            border-radius: 8px;
                            background: #ffffff;
                            border: 1px solid #e2e8f0;
                            font-size: 12px;
                            font-weight: 800;
                            color: #0f172a;
                        }
                        QPushButton:hover { background: #f8fafc; }
                        """
                    )
                btn_mag.setToolTip("Magnify zones: click-drag a rectangle on the plot to zoom to that area")
                hint = QtWidgets.QLabel("Mouse wheel zoom (Shift = X only, Ctrl = Y only)")
                hint.setStyleSheet("color: #64748b; font-size: 11px;")
                ctrl.addWidget(btn_mag)
                ctrl.addWidget(btn_out)
                ctrl.addWidget(btn_in)
                ctrl.addWidget(btn_reset)
                ctrl.addSpacing(10)
                ctrl.addWidget(hint)
                ctrl.addStretch(1)
                tab_layout.addLayout(ctrl)

                def _reset():
                    try:
                        ax.set_xlim(*base_xlim)
                        ax.set_ylim(*base_ylim)
                    except Exception:
                        pass
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

                def _zoom(factor: float):
                    self._apply_axes_zoom(ax, factor, axis="both")
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

                btn_out.clicked.connect(lambda *_: _zoom(1.25))
                btn_in.clicked.connect(lambda *_: _zoom(0.8))
                btn_reset.clicked.connect(lambda *_: _reset())

                press_xy: tuple[float, float] | None = None
                rect_patch = None

                def _on_scroll(event):
                    if getattr(event, "inaxes", None) is not ax:
                        return
                    direction = str(getattr(event, "button", "") or "").strip().lower()
                    if direction not in {"up", "down"}:
                        return
                    factor = 0.8 if direction == "up" else 1.25
                    key = str(getattr(event, "key", "") or "").lower()
                    axis = "both"
                    if "shift" in key:
                        axis = "x"
                    elif "control" in key or "ctrl" in key:
                        axis = "y"
                    self._apply_axes_zoom(
                        ax,
                        factor,
                        center=(getattr(event, "xdata", None), getattr(event, "ydata", None)),
                        axis=axis,
                    )
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

                def _on_motion(event):
                    nonlocal press_xy, rect_patch
                    if not btn_mag.isChecked():
                        return
                    if press_xy is None or rect_patch is None:
                        return
                    x1 = getattr(event, "xdata", None)
                    y1 = getattr(event, "ydata", None)
                    if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
                        return
                    x0, y0 = press_xy
                    try:
                        rect_patch.set_bounds(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                    except Exception:
                        return
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

                def _on_press(event):
                    nonlocal press_xy, rect_patch
                    if not btn_mag.isChecked():
                        return
                    if getattr(event, "inaxes", None) is not ax:
                        return
                    if int(getattr(event, "button", 0) or 0) != 1:
                        return
                    x0 = getattr(event, "xdata", None)
                    y0 = getattr(event, "ydata", None)
                    if not isinstance(x0, (int, float)) or not isinstance(y0, (int, float)):
                        return
                    press_xy = (float(x0), float(y0))
                    try:
                        from matplotlib.patches import Rectangle

                        if rect_patch is not None:
                            try:
                                rect_patch.remove()
                            except Exception:
                                pass
                        rect_patch = Rectangle(
                            (float(x0), float(y0)),
                            0.0,
                            0.0,
                            fill=False,
                            linewidth=1.2,
                            linestyle="--",
                            edgecolor="#0f766e",
                            alpha=0.9,
                        )
                        ax.add_patch(rect_patch)
                    except Exception:
                        rect_patch = None
                        press_xy = None
                        return
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

                def _on_release(event):
                    nonlocal press_xy, rect_patch
                    if press_xy is None or rect_patch is None:
                        return
                    x1 = getattr(event, "xdata", None)
                    y1 = getattr(event, "ydata", None)
                    try:
                        rect_patch.remove()
                    except Exception:
                        pass
                    rect_patch = None
                    x0, y0 = press_xy
                    press_xy = None

                    if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)):
                        try:
                            canvas.draw_idle()
                        except Exception:
                            pass
                        return
                    x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
                    if abs(x1f - x0f) < 1e-9 or abs(y1f - y0f) < 1e-9:
                        return
                    lo_x, hi_x = (x0f, x1f) if x0f <= x1f else (x1f, x0f)
                    lo_y, hi_y = (y0f, y1f) if y0f <= y1f else (y1f, y0f)
                    try:
                        ax.set_xlim(lo_x, hi_x)
                        ax.set_ylim(lo_y, hi_y)
                    except Exception:
                        return
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass

                try:
                    canvas.mpl_connect("scroll_event", _on_scroll)
                    canvas.mpl_connect("motion_notify_event", _on_motion)
                    canvas.mpl_connect("button_press_event", _on_press)
                    canvas.mpl_connect("button_release_event", _on_release)
                except Exception:
                    pass

            tab_layout.addWidget(canvas, 1)
            tabs.addTab(tab, name or "Auto Plot")

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dlg.exec()

    def _delete_selected_auto_plots(
        self,
        *,
        list_widget: QtWidgets.QListWidget | None = None,
    ) -> None:
        to_del = self._selected_auto_plot_definitions(list_widget=list_widget)
        if not to_del:
            return
        self._auto_plots = [d for d in self._auto_plots if d not in to_del]
        try:
            self._auto_plot_path.write_text(json.dumps(self._auto_plots, indent=2), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Auto-Plots", str(exc))
            return
        self._sync_main_auto_plot_actions()
        self._refresh_auto_plots_list(list_widget=list_widget)

    def _render_plot_def_to_figure(self, d: dict):
        from matplotlib.figure import Figure

        mode = str(d.get("mode") or "").strip().lower()
        selection = self._selection_from_plot_def(d)
        runs = [str(v).strip() for v in (selection.get("member_runs") or d.get("member_runs") or []) if str(v).strip()]
        if not runs:
            run_ref = str(d.get("run") or "").strip()
            if run_ref:
                runs = [self._run_name_by_display.get(run_ref, run_ref)]
        plot_dimension = "3d" if mode == "performance" and str(d.get("input2") or "").strip() else "2d"
        fig = Figure(figsize=(8, 4), dpi=100)
        if plot_dimension == "3d":
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
        if mode == "curves":
            ys = d.get("y") or []
            y = str(ys[0] if isinstance(ys, list) and ys else "").strip()
            x_label = str(d.get("x") or "").strip()
            ax.set_title(str(d.get("name") or "") or self._compose_run_title(selection, f"{y} vs {x_label}"))
            ax.set_xlabel(x_label)
            ax.set_ylabel(y)
            multi_run = len(runs) > 1
            for run in runs:
                x = self._resolve_curve_x_key(run, x_label)
                curves = self._load_curves_for_selection(run, y, x, selection=selection, serials=self._active_serials())  # type: ignore[arg-type]
                for s in curves:
                    xs = s.get("x") or []
                    ys2 = s.get("y") or []
                    if not isinstance(xs, list) or not isinstance(ys2, list) or not xs or not ys2:
                        continue
                    ax.plot(
                        xs,
                        ys2,
                        linewidth=1.1,
                        alpha=0.85,
                        label=self._curve_trace_label(run, s, multi_run=multi_run),
                    )
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        elif mode == "metrics":
            stats_val = d.get("stats")
            stats = (
                [str(x).strip().lower() for x in stats_val if str(x).strip()]
                if isinstance(stats_val, list)
                else []
            )
            if not stats:
                st = str(d.get("stat") or "").strip().lower()
                if st:
                    stats = [st]
            if not stats:
                stats = ["mean"]
            metric_source = getattr(be, "TD_METRIC_PLOT_SOURCE_AGGREGATE", "aggregate")
            normalizer = getattr(be, "td_metric_normalize_plot_source", None)
            if callable(normalizer):
                try:
                    metric_source = str(normalizer(d.get("metric_plot_source"))).strip().lower()
                except Exception:
                    metric_source = str(d.get("metric_plot_source") or metric_source).strip().lower() or str(metric_source)
            else:
                metric_source = str(d.get("metric_plot_source") or metric_source).strip().lower() or str(metric_source)
            stats_label = self._metric_title_suffix(stats)
            ys = d.get("y") or []
            y_cols = [str(x).strip() for x in ys] if isinstance(ys, list) else []
            plot_bounds = bool(d.get("plot_bounds"))
            ax.set_title(str(d.get("name") or "") or self._compose_run_title(selection, stats_label))
            ax.set_xlabel("Serial Number")
            ax.set_ylabel(stats[0] if len(stats) == 1 else "Metric value")
            serial_rows = self._active_serial_rows()
            labels = _td_order_metric_serials(
                [_td_serial_value(row) for row in serial_rows if _td_serial_value(row)],
                serial_rows,
            )
            x_idx = list(range(len(labels)))
            multi_run = len(runs) > 1
            for run in runs:
                metric_bounds = self._metric_bounds_for_run(run) if plot_bounds else {}
                for y_col in y_cols:
                    for stat in stats:
                        source_stat = "mean" if str(stat).strip().lower() == "average" else stat
                        series = self._load_metric_series_for_selection(
                            run,
                            y_col,
                            source_stat,
                            selection=selection,
                            metric_source=metric_source,
                        )  # type: ignore[arg-type]
                        is_average = str(stat).strip().lower() == "average"
                        if str(stat).strip().lower() == "average":
                            yv = be.td_metric_average_plot_values(series, labels)  # type: ignore[arg-type]
                            ax.plot(
                                x_idx,
                                yv,
                                marker=None,
                                linewidth=1.2,
                                label=(f"{run}.{y_col}.{stat}" if multi_run else f"{y_col}.{stat}"),
                            )
                        else:
                            points = self._metric_points_for_serial_labels(series, labels)
                            if not points:
                                continue
                            ax.plot(
                                [float(p.get("x") or 0.0) for p in points],
                                [float(p.get("y") or 0.0) for p in points],
                                linestyle="",
                                marker="o",
                                markersize=5.0,
                                alpha=0.88,
                                label=(f"{run}.{y_col}.{stat}" if multi_run else f"{y_col}.{stat}"),
                            )
                        bound = dict(metric_bounds.get(str(y_col)) or {})
                        self._plot_metric_bound_lines(ax, bound)
            self._apply_metric_program_segments(ax, labels)
            ax.set_xlim(-0.5, max(len(labels) - 0.5, 0.5))
            ax.set_xticks(x_idx)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        else:
            output = str(d.get("output") or "").strip()
            input1 = str(d.get("input1") or "").strip()
            input2 = str(d.get("input2") or "").strip()
            performance_plot_method = self._perf_normalize_plot_method(d.get("performance_plot_method"))
            stats_val = d.get("stats")
            plot_stats = (
                [str(x).strip().lower() for x in stats_val if str(x).strip()]
                if isinstance(stats_val, list)
                else []
            )
            if not plot_stats:
                view_stat = str(d.get("view_stat") or "").strip().lower()
                plot_stats = [view_stat] if view_stat else ["mean"]
            if not runs:
                runs = self._selected_perf_runs()
            serials = self._selected_perf_serials()
            fit_enabled = bool(d.get("fit_enabled", True))
            require_min_points = max(2, int(getattr(self, "_perf_require_min_points", 2) or 2))
            run_type_mode = str(d.get("performance_run_type_mode") or self._selected_perf_run_type_mode()).strip().lower()
            perf_filter_mode = str(d.get("performance_filter_mode") or "all_conditions").strip().lower()
            control_period_filter = d.get("selected_control_period")
            common_runs = self._common_runs_for_perf_vars(output, input1, input2)
            if common_runs:
                common_set = set(common_runs)
                run_order = {r: i for i, r in enumerate(getattr(self, "_perf_all_runs", runs) or [])}
                runs = sorted([r for r in runs if r in common_set], key=lambda r: run_order.get(r, 10**9))
            collect_fn = (
                self._perf_collect_cached_condition_mean_results
                if performance_plot_method == "cached_condition_means"
                else self._perf_collect_results
            )
            results, plot_view_stats, _fit_error = collect_fn(
                output,
                input1,
                input2,
                plot_stats,
                runs,
                serials,
                fit_enabled=fit_enabled,
                require_min_points=require_min_points,
                run_type_filter=run_type_mode,
                control_period_filter=(
                    control_period_filter
                    if run_type_mode == "pulsed_mode" and perf_filter_mode == "match_control_period"
                    else None
                ),
                display_control_period=control_period_filter,
            )
            view_stat = str(d.get("view_stat") or "").strip().lower()
            if not view_stat or view_stat not in results:
                view_stat = plot_view_stats[0] if plot_view_stats else ""
            result = (results or {}).get(view_stat) or {}
            if isinstance(result, dict) and result:
                self._render_performance_result(
                    ax,
                    result,
                    highlight_serial=str(d.get("highlight_serial") or "").strip(),
                    title_override=str(d.get("name") or "").strip(),
                )

        try:
            fig.tight_layout()
        except Exception:
            pass
        return fig

    def _save_all_auto_plots_pdf(self) -> None:
        if not self._plot_ready or not self._auto_plots or not self._db_path:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save All Auto-Plots",
            str(self._project_dir / "auto_plots_test_data.pdf"),
            "PDF Files (*.pdf)",
        )
        if not path:
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(path) as pdf:
                for d in self._auto_plots:
                    if not isinstance(d, dict):
                        continue
                    fig = self._render_plot_def_to_figure(d)
                    pdf.savefig(fig)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save All Auto-Plots", str(exc))


class ProjectEnvDialog(QtWidgets.QDialog):
    def __init__(self, project_dir: Path, parent=None):
        super().__init__(parent)
        self._project_dir = Path(project_dir).expanduser()
        self._env_path = be.project_scanner_env_path(self._project_dir)

        self.setWindowTitle("Project Env Overrides")
        self.resize(680, 340)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Project-level scanner overrides")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        root.addWidget(title)

        self.lbl_path = QtWidgets.QLabel("")
        self.lbl_path.setStyleSheet("font-size: 11px; color: #475569;")
        self.lbl_path.setWordWrap(True)
        root.addWidget(self.lbl_path)

        note = QtWidgets.QLabel(
            "These values override Central Runtime `user_inputs/scanner.env` and `user_inputs/scanner.local.env` for this project only."
        )
        note.setStyleSheet("font-size: 11px; color: #475569;")
        note.setWordWrap(True)
        root.addWidget(note)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.cb_combined_only = QtWidgets.QCheckBox("Use combined.txt only (skip acceptance data)")
        form.addRow("Trending:", self.cb_combined_only)

        self.cb_fuzzy_header = QtWidgets.QCheckBox("Enable fuzzy header stick")
        form.addRow("KV Header:", self.cb_fuzzy_header)

        self.sp_fuzzy_header_ratio = QtWidgets.QDoubleSpinBox()
        self.sp_fuzzy_header_ratio.setRange(0.0, 1.0)
        self.sp_fuzzy_header_ratio.setSingleStep(0.01)
        self.sp_fuzzy_header_ratio.setDecimals(3)
        form.addRow("Header min ratio:", self.sp_fuzzy_header_ratio)

        self.cb_fuzzy_term = QtWidgets.QCheckBox("Enable fuzzy term stick")
        form.addRow("Trending Terms:", self.cb_fuzzy_term)

        self.sp_fuzzy_term_ratio = QtWidgets.QDoubleSpinBox()
        self.sp_fuzzy_term_ratio.setRange(0.0, 1.0)
        self.sp_fuzzy_term_ratio.setSingleStep(0.01)
        self.sp_fuzzy_term_ratio.setDecimals(3)
        form.addRow("Term min ratio:", self.sp_fuzzy_term_ratio)

        root.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_open = QtWidgets.QPushButton("Open File")
        self.btn_delete = QtWidgets.QPushButton("Delete Overrides")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        for b in (self.btn_open, self.btn_delete, self.btn_save, self.btn_cancel):
            b.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 12px;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
                """
            )
        self.btn_save.setStyleSheet(
            """
            QPushButton {
                padding: 8px 12px;
                border-radius: 8px;
                background: #0f766e;
                color: #ffffff;
                border: 1px solid #0f766e;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover { background: #0d9488; }
            """
        )
        self.btn_open.clicked.connect(self._act_open)
        self.btn_delete.clicked.connect(self._act_delete)
        self.btn_save.clicked.connect(self._act_save)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_open)
        btns.addWidget(self.btn_delete)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_cancel)
        root.addLayout(btns)

        self._load()

    def _load(self) -> None:
        try:
            env = be.load_scanner_env(project_dir=self._project_dir)
        except Exception:
            env = {}

        exists = bool(self._env_path.exists())
        self.lbl_path.setText(f"File: {self._env_path}  ({'exists' if exists else 'not created'})")
        self.btn_delete.setEnabled(exists)

        def _truthy(key: str, default: str = "0") -> bool:
            v = str(env.get(key, default) or "").strip().lower()
            return v in {"1", "true", "yes", "on", "enable", "enabled"}

        self.cb_combined_only.setChecked(
            _truthy("EIDAT_TRENDING_COMBINED_ONLY") or _truthy("EIDAT_COMBINED_ONLY")
        )
        self.cb_fuzzy_header.setChecked(_truthy("EIDAT_FUZZY_HEADER_STICK"))
        self.cb_fuzzy_term.setChecked(_truthy("EIDAT_FUZZY_TERM_STICK", "1"))

        def _float(key: str, fallback: float) -> float:
            try:
                return float(env.get(key, str(fallback)) or fallback)
            except Exception:
                return fallback

        self.sp_fuzzy_header_ratio.setValue(_float("EIDAT_FUZZY_HEADER_MIN_RATIO", 0.72))
        self.sp_fuzzy_term_ratio.setValue(_float("EIDAT_FUZZY_TERM_MIN_RATIO", 0.78))

    def _current_overrides(self) -> dict[str, str]:
        def _ratio_str(v: float) -> str:
            s = f"{float(v):.3f}"
            s = s.rstrip("0").rstrip(".")
            return s or "0"

        return {
            "EIDAT_TRENDING_COMBINED_ONLY": "1" if self.cb_combined_only.isChecked() else "0",
            "EIDAT_FUZZY_HEADER_STICK": "1" if self.cb_fuzzy_header.isChecked() else "0",
            "EIDAT_FUZZY_HEADER_MIN_RATIO": _ratio_str(self.sp_fuzzy_header_ratio.value()),
            "EIDAT_FUZZY_TERM_STICK": "1" if self.cb_fuzzy_term.isChecked() else "0",
            "EIDAT_FUZZY_TERM_MIN_RATIO": _ratio_str(self.sp_fuzzy_term_ratio.value()),
        }

    def _act_save(self) -> None:
        try:
            be.save_project_scanner_env(self._current_overrides(), self._project_dir)
            self._load()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Project Env", str(exc))

    def _act_open(self) -> None:
        try:
            if not self._env_path.exists():
                be.save_project_scanner_env(self._current_overrides(), self._project_dir)
            be.open_path(self._env_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Project Env", str(exc))

    def _act_delete(self) -> None:
        try:
            if not self._env_path.exists():
                return
            resp = QtWidgets.QMessageBox.question(
                self,
                "Delete Overrides",
                "Delete this project's overrides file and revert to inherited scanner.env values?",
            )
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            be.delete_project_scanner_env(self._project_dir)
            self._load()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete Overrides", str(exc))


class MetadataBatchEditorDialog(QtWidgets.QDialog):
    FIELD_DEFS = [
        ("Program", "program_title"),
        ("Asset Type", "asset_type"),
        ("Asset Specific Type", "asset_specific_type"),
        ("Vendor", "vendor"),
        ("Part Number", "part_number"),
        ("Revision", "revision"),
        ("Test Date", "test_date"),
        ("Report Date", "report_date"),
        ("Acceptance Test Plan", "acceptance_test_plan_number"),
        ("Document Type", "document_type"),
        ("Document Acronym", "document_type_acronym"),
    ]

    def __init__(self, rows: list[dict], *, choices: dict[str, list[str]] | None = None, parent=None):
        super().__init__(parent)
        self._rows = [dict(row) for row in (rows or []) if isinstance(row, dict)]
        self._choices = dict(choices or {})
        self._field_labels = {key: label for label, key in self.FIELD_DEFS}

        self.setWindowTitle("Edit Metadata")
        self.resize(680, 320)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Bulk-edit metadata for the selected files.")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #0f172a;")
        root.addWidget(title)

        serials = [str(row.get("serial_number") or "").strip() for row in self._rows if str(row.get("serial_number") or "").strip()]
        serial_preview = ", ".join(serials[:8])
        if len(serials) > 8:
            serial_preview += f" ... (+{len(serials) - 8})"
        summary = QtWidgets.QLabel(
            f"Selected files: {len(self._rows)}"
            + (f"\nSerials: {serial_preview}" if serial_preview else "")
            + "\nChoose one metadata field, enter the new value, and apply it to every selected file."
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("font-size: 11px; color: #475569;")
        root.addWidget(summary)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setSpacing(10)

        combo_style = (
            """
            QComboBox {
                padding: 6px 8px;
                border-radius: 6px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
                border: 1px solid #d1d5db;
            }
            QLineEdit {
                color: #1f2937;
            }
            """
        )

        self.cmb_field = QtWidgets.QComboBox()
        self.cmb_field.setMinimumWidth(280)
        self.cmb_field.setStyleSheet(combo_style)
        for label, key in self.FIELD_DEFS:
            self.cmb_field.addItem(label, key)
        form.addRow("Field:", self.cmb_field)

        self.cmb_value = QtWidgets.QComboBox()
        self.cmb_value.setEditable(True)
        self.cmb_value.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.cmb_value.setMinimumWidth(320)
        self.cmb_value.setStyleSheet(combo_style)
        try:
            le = self.cmb_value.lineEdit()
            if le is not None:
                le.setPlaceholderText("Enter new value")
        except Exception:
            pass
        form.addRow("New Value:", self.cmb_value)

        self.lbl_current = QtWidgets.QLabel("")
        self.lbl_current.setWordWrap(True)
        self.lbl_current.setStyleSheet("font-size: 11px; color: #475569;")
        form.addRow("Current:", self.lbl_current)

        root.addLayout(form)

        self.cmb_field.currentIndexChanged.connect(self._on_field_changed)
        self._on_field_changed(self.cmb_field.currentIndex())

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        for btn in (self.btn_apply, self.btn_cancel):
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 12px;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
                """
            )
        self.btn_apply.setStyleSheet(
            """
            QPushButton {
                padding: 8px 12px;
                border-radius: 8px;
                background: #0f766e;
                color: #ffffff;
                border: 1px solid #0f766e;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover { background: #0d9488; }
            """
        )
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_cancel)
        root.addLayout(buttons)

        _fit_widget_to_screen(self, margin=24)

    def _shared_value(self, key: str) -> str:
        seen: set[str] = set()
        value = ""
        for row in self._rows:
            current = str(row.get(key) or "").strip()
            if not current:
                current = ""
            seen.add(current)
            value = current
            if len(seen) > 1:
                return ""
        return value

    def _field_key(self) -> str:
        return str(self.cmb_field.currentData() or "").strip()

    def _on_field_changed(self, _index: int) -> None:
        key = self._field_key()
        self.cmb_value.blockSignals(True)
        try:
            self.cmb_value.clear()
            values = [str(v).strip() for v in (self._choices.get(key) or []) if str(v).strip()]
            if values:
                self.cmb_value.addItems(values)
            self.cmb_value.setCurrentIndex(-1)
            self.cmb_value.setEditText("")
        finally:
            self.cmb_value.blockSignals(False)
        shared = self._shared_value(key)
        label = self._field_labels.get(key, key)
        if shared:
            self.lbl_current.setText(f"{label} is the same across the selection: {shared}")
        else:
            self.lbl_current.setText(f"{label} has mixed values across the selection.")

    def field_updates(self) -> dict[str, str]:
        key = self._field_key()
        value = str(self.cmb_value.currentText() or "").strip()
        if not key or not value:
            return {}
        if value == self._shared_value(key):
            return {}
        return {key: value}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIDAT Prototype - Demonstration Only")
        self.resize(1280, 860)

        be.ensure_scaffold()
        # Initialize core OCR settings only if scanner.local.env doesn't already exist
        try:
            if not be.SCANNER_ENV_LOCAL.exists():
                env = be.parse_scanner_env(be.SCANNER_ENV_LOCAL)
                if not (env.get("OCR_ROW_EPS") or "").strip():
                    env["OCR_ROW_EPS"] = "15"
                if not (env.get("OCR_DPI") or "").strip():
                    env["OCR_DPI"] = "500"
                if env:
                    be.save_scanner_env(env)
        except Exception:
            pass
        # Global styling - polished modern design with rounded corners
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f0f4f8;
            }
            QWidget {
                background: transparent;
            }
            QLabel {
                color: #1f2937;
            }
            QLabel.subtle {
                color: #6b7280;
                font-size: 12px;
            }

            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 10px 14px;
                color: #374151;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #2563eb;
                border-width: 2px;
            }

            /* Improve QMessageBox legibility */
            QMessageBox {
                background-color: #ffffff;
                border-radius: 12px;
            }
            QMessageBox QLabel {
                color: #1f2937;
            }
            QMessageBox QPushButton {
                padding: 8px 20px;
                border-radius: 8px;
                background: #2563eb;
                color: #ffffff;
                border: none;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                background: #1d4ed8;
            }

            /* Global Menu styling */
            QMenu {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                padding: 4px;
            }
            QMenu::item {
                background: transparent;
                color: #1f2937;
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background: #dbeafe;
                color: #1f2937;
            }
            QMenu::separator {
                height: 1px;
                background: #e5e7eb;
                margin: 4px 8px;
            }

            /* Global ComboBox dropdown styling */
            QComboBox {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
                border: 1px solid #d1d5db;
            }

            /* Global Dialog styling */
            QDialog {
                background: #ffffff;
            }
            QDialog QLabel {
                color: #1f2937;
            }
            QDialog QLineEdit {
                color: #1f2937;
                background: #ffffff;
            }
            QDialog QPushButton {
                color: #374151;
                background: #ffffff;
            }

            /* Plain text edit for debug console with rounded corners */
            QPlainTextEdit {
                background: #1f2937;
                color: #e5e7eb;
                border: 1px solid #374151;
                border-radius: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 12px;
            }

            /* Scrollbars */
            QScrollBar:vertical {
                border: none;
                background: #f3f4f6;
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #d1d5db;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9ca3af;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QScrollBar:horizontal {
                border: none;
                background: #f3f4f6;
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #d1d5db;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #9ca3af;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            """
        )

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Header with gradient background - logo and tabs inline with modern design
        header = QtWidgets.QFrame()
        header.setObjectName("heroHeader")
        header.setStyleSheet("""
            #heroHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ffffff, stop:0.5 #f8fafc, stop:1 #ffffff);
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        hbox = QtWidgets.QHBoxLayout(header)
        hbox.setContentsMargins(24, 20, 24, 16)
        hbox.setSpacing(20)

        # Logo - simple and compact
        logo_container = QtWidgets.QFrame()
        logo_container.setObjectName("logoBadge")
        logo_container.setStyleSheet("""
            #logoBadge {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1e40af);
                border-radius: 12px;
                border: none;
            }
        """)
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(QtGui.QColor(15, 23, 42, 90))
        logo_container.setGraphicsEffect(shadow)
        logo_container.setFixedSize(48, 48)
        logo_layout = QtWidgets.QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        logo = QtWidgets.QLabel()
        logo_pix = self._build_logo_pixmap(size=44)
        logo.setPixmap(logo_pix)
        logo.setFixedSize(44, 44)
        logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(logo)

        hbox.addWidget(logo_container)

        # Title section with improved typography
        title = QtWidgets.QLabel("EIDAT")
        font = title.font(); font.setPointSize(22); font.setBold(True); font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 0.5); title.setFont(font)
        title.setStyleSheet("color: #0f172a; padding: 0px;")
        subtitle = QtWidgets.QLabel("End Item Data Analysis Tool (Prototype)")
        subtitle.setStyleSheet("color:#64748b; font-size: 12px; font-weight: 500; letter-spacing: 0.3px; border:none; background:transparent;")
        proto_badge = QtWidgets.QLabel("Prototype build - demonstration only")
        proto_badge.setStyleSheet("color:#991b1b; background:#fee2e2; border:1px solid #fecaca; border-radius:6px; font-size:11px; font-weight:600; padding:2px 8px; letter-spacing:0.5px;")
        proto_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        tbox = QtWidgets.QVBoxLayout();
        tbox.setSpacing(4)
        tbox.addWidget(title); tbox.addWidget(subtitle); tbox.addWidget(proto_badge)
        hbox.addLayout(tbox)

        hbox.addStretch(1)

        # Create clean, minimal tab buttons - larger and right-aligned
        self.tab_buttons = QtWidgets.QWidget()
        tab_btn_layout = QtWidgets.QHBoxLayout(self.tab_buttons)
        tab_btn_layout.setContentsMargins(0, 0, 0, 0)
        tab_btn_layout.setSpacing(0)

        self.btn_tab_setup = QtWidgets.QPushButton("⚙  Setup")
        self.btn_tab_files = QtWidgets.QPushButton("Files")
        self.btn_tab_product_center = QtWidgets.QPushButton("Product Center")
        self.btn_tab_projects = QtWidgets.QPushButton("Projects")

        for btn in [self.btn_tab_setup, self.btn_tab_files, self.btn_tab_product_center, self.btn_tab_projects]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 14px 48px;
                    margin: 0;
                    font-weight: 500;
                    font-size: 15px;
                    color: #6b7280;
                    background: transparent;
                    border: none;
                    border-bottom: 3px solid transparent;
                }
                QPushButton:checked {
                    color: #2563eb;
                    font-weight: 600;
                    background: transparent;
                    border-bottom: 3px solid #2563eb;
                }
                QPushButton:hover:!checked {
                    color: #374151;
                    background: rgba(59, 130, 246, 0.05);
                    border-bottom: 3px solid #cbd5e1;
                }
            """)
            tab_btn_layout.addWidget(btn)

        self.btn_tab_setup.setChecked(True)
        hbox.addWidget(self.tab_buttons)

        # Store status label but don't add it to layout (hidden)
        self.lbl_ready = QtWidgets.QLabel("● System Ready");
        self.lbl_ready.setObjectName("statusBadge");
        self.lbl_ready.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #d1fae5, stop:1 #a7f3d0);
            color: #065f46;
            border-radius: 12px;
            padding: 6px 16px;
            font-size: 12px;
            font-weight: 700;
            border: 2px solid #10b981;
        """)
        self.lbl_ready.setVisible(False)

        # Create tabs widget (hidden, only used for content management)
        self.tabs = QtWidgets.QTabWidget()
        self.tab_setup = QtWidgets.QWidget()
        self.tab_process = QtWidgets.QWidget()
        # NOTE: tab_plot and tab_outputs removed - they were unused
        self.tab_files = QtWidgets.QWidget()
        self.tab_product_center = QtWidgets.QWidget()
        self.tab_projects = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_setup, "Setup")
        self.tabs.addTab(self.tab_files, "Files")
        self.tabs.addTab(self.tab_product_center, "Product Center")
        self.tabs.addTab(self.tab_projects, "Projects")

        # Connect tab buttons to switch content
        self.btn_tab_setup.clicked.connect(lambda: self._switch_tab(0))
        self.btn_tab_files.clicked.connect(lambda: self._switch_tab(1))
        self.btn_tab_product_center.clicked.connect(lambda: self._switch_tab(2))
        self.btn_tab_projects.clicked.connect(lambda: self._switch_tab(3))

        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Tab pane styling - clean look with no border-radius to match flat tabs
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #e5e7eb;
                border-radius: 0;
                border-top: none;
                margin: 0px;
                background: #ffffff;
                padding: 16px;
            }
            QTabBar::tab {
                width: 0px;
                height: 0px;
                margin: 0px;
                padding: 0px;
                border: none;
            }
            """
        )

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        # Toggle to show/hide the debug log panel on demand - styled with rounded corners
        self.btn_toggle_log = QtWidgets.QPushButton("\u25B6  Debug Console")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setChecked(False)
        self.btn_toggle_log.clicked.connect(self._toggle_log_panel)
        self.btn_toggle_log.setStyleSheet("""
            QPushButton {
                background: #374151;
                color: #e5e7eb;
                border: 1px solid #4b5563;
                border-radius: 8px;
                padding: 10px 16px;
                text-align: left;
                font-size: 12px;
                font-weight: 500;
                margin: 8px;
            }
            QPushButton:hover {
                background: #4b5563;
            }
            QPushButton:checked {
                background: #1f2937;
                border-color: #374151;
            }
        """)
        pol_log = self.btn_toggle_log.sizePolicy(); pol_log.setHorizontalStretch(1); pol_log.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding); self.btn_toggle_log.setSizePolicy(pol_log)
        self.status_bar = self.statusBar()
        self._progress_dialog = RunProgressDialog(self)
        self._progress_dialog.canceled.connect(self._on_progress_canceled)
        self._repo_scan_dialog = RepoScanDialog(self)
        # Pattern supports both old format (without Found) and new format (with Found)
        self._progress_pattern = re.compile(r"\[PROGRESS\]\s*Terms:\s*(\d+)%\s*\((\d+)/(\d+)\)(?:\s*\|\s*Found:\s*(\d+))?")
        self._progress_total = 0
        self._progress_completed = 0
        self._progress_found = 0
        # File-level progress tracking
        self._progress_file_total = 0
        self._progress_file_index = 0
        self._progress_file_name: str | None = None
        # Track whether current worker is a missing-terms batch (driven by helper script)
        self._is_missing_terms_batch = False
        self._progress_popup_active = False
        self._progress_was_canceled = False
        self._last_run_dir: Path | None = None

        # Create toast notification widget
        self._toast = ToastNotification(self)
        self._toast.hide()
        self._files_external_rel_paths: set[str] | None = None
        self._files_external_filter_label = ""
        self._projects_all: list[dict] = []
        self._projects_external_dirs: set[str] | None = None
        self._projects_external_filter_label = ""
        self._product_center_products: list[dict] = []
        self._product_center_filtered: list[dict] = []
        self._product_center_current: dict | None = None

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(header)
        self.content_container = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.tabs)
        content_layout.addWidget(self.btn_toggle_log)

        self.log_container = QtWidgets.QFrame()
        self.log_container.setObjectName("debugPanel")
        log_layout = QtWidgets.QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(16, 0, 16, 12)
        log_layout.setSpacing(0)
        log_layout.addWidget(self.log)
        self.log_container.setMinimumHeight(160)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.content_container)
        self.main_splitter.addWidget(self.log_container)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.log_container.hide()

        layout.addWidget(self.main_splitter, 1)
        # Build tabs
        self._setup_tab_setup()
        self._setup_tab_process()
        self._setup_tab_files()
        self._setup_tab_product_center()
        self._setup_tab_projects()
        # Data Outputs tab replaced by top-level master button

        # Runtime
        self._worker: ProcWorker | None = None
        self._sync_worker: WorkspaceSyncWorker | None = None
        self._manager_worker: ManagerTaskWorker | None = None
        self._project_worker: ProjectTaskWorker | None = None
        self._sync_popup_active = False
        self._manager_popup_active = False
        self._project_popup_active = False
        self._enrich_after_run: bool = False
        self._registry_cache: tuple[list[str], list[list[str]]] | None = None
        self._scan_refresh()
        # Initial workspace sync on startup (quiet, no master-compile/notifications)
        try:
            self._start_workspace_sync(auto=True, show_popup=False, heading="Auto Workspace Sync")
        except Exception:
            pass
        # Periodic auto-sync every few minutes (no popup, no compile)
        try:
            self._sync_timer = QtCore.QTimer(self)
            self._sync_timer.setInterval(5 * 60 * 1000)  # 5 minutes
            self._sync_timer.timeout.connect(lambda: self._start_workspace_sync(auto=True, show_popup=False, heading="Auto Workspace Sync"))
            self._sync_timer.start()
        except Exception:
            pass
        # Periodic EIDAT Manager scan/process (quiet; uses scanner.env flag)
        try:
            self._manager_scan_timer = QtCore.QTimer(self)
            self._manager_scan_timer.setInterval(10 * 60 * 1000)  # 10 minutes
            self._manager_scan_timer.timeout.connect(self._auto_manager_tick)
            self._manager_scan_timer.start()
            QtCore.QTimer.singleShot(800, self._auto_manager_tick)
        except Exception:
            pass

    # Tabs
    def _setup_tab_setup(self):
        grid = QtWidgets.QGridLayout(self.tab_setup)
        grid.setContentsMargins(24, 24, 24, 24)
        grid.setSpacing(16)

        # Program Health
        grp_env = QtWidgets.QGroupBox("Program Health")
        grp_env.setStyleSheet("""
            QGroupBox {
                font-weight: 900;
                font-size: 24px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 16px;
                background: #ffffff;
                padding: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: #111827;
            }
        """)
        l_env = QtWidgets.QVBoxLayout(grp_env)
        l_env.setSpacing(12)

        # Header row with description and badge
        header_row = QtWidgets.QHBoxLayout()
        desc_label = QtWidgets.QLabel("Ensure all dependencies are installed and up to date")
        desc_label.setStyleSheet("color: #6b7280; font-size: 13px; font-weight: 400;")
        header_row.addWidget(desc_label)
        header_row.addStretch()

        self.lbl_env_health = QtWidgets.QLabel("Healthy")
        self.lbl_env_health.setObjectName("healthBadge")
        self.lbl_env_health.setStyleSheet("background: #d1fae5; color: #065f46; border-radius: 12px; padding: 4px 12px; font-size: 12px; font-weight: 600;")
        header_row.addWidget(self.lbl_env_health)
        l_env.addLayout(header_row)

        # Button row
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(12)
        self.btn_check = QtWidgets.QPushButton("Check Environment")
        self.btn_check.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        self.btn_install = QtWidgets.QPushButton("Update Packages")
        self.btn_install.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                background: #93c5fd;
                border-color: #93c5fd;
            }
        """)
        self.btn_check.clicked.connect(self._act_check_env)
        self.btn_install.clicked.connect(self._act_install)
        button_row.addWidget(self.btn_check)
        button_row.addWidget(self.btn_install)
        button_row.addStretch()
        l_env.addLayout(button_row)

        # Environment path display
        self.lbl_env = QtWidgets.QLabel("Env: Unknown")
        self.lbl_env.setStyleSheet("color: #6b7280; font-size: 12px; font-weight: 400; padding: 8px; background: #f9fafb; border-radius: 4px;")
        self.lbl_env.setWordWrap(True)
        l_env.addWidget(self.lbl_env)

        # Global Repository
        grp_repo = QtWidgets.QGroupBox("Global Repository")
        grp_repo.setStyleSheet(
            """
            QGroupBox {
                font-weight: 900;
                font-size: 24px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 16px;
                background: #ffffff;
                padding: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: #111827;
            }
            """
        )
        repo_layout = QtWidgets.QVBoxLayout(grp_repo)
        repo_layout.setSpacing(10)

        repo_desc = QtWidgets.QLabel(
            "Select the top-level folder where EIDPs naturally live. EIDAT will create and manage an 'EIDAT Support' folder inside it."
        )
        repo_desc.setWordWrap(True)
        repo_desc.setStyleSheet("color: #6b7280; font-size: 13px; font-weight: 400;")
        repo_layout.addWidget(repo_desc)

        repo_label = QtWidgets.QLabel("Global Repo Root")
        repo_label.setStyleSheet("color: #374151; font-size: 13px; font-weight: 600; margin-top: 2px;")
        repo_layout.addWidget(repo_label)

        button_min_h = 38
        repo_row = QtWidgets.QHBoxLayout()
        repo_row.setSpacing(6)
        self.ed_global_repo = QtWidgets.QLineEdit(str(getattr(be, "get_repo_root", lambda: be.DEFAULT_REPO_ROOT)()))
        self.ed_global_repo.setMinimumHeight(32)
        self.ed_global_repo.setStyleSheet(
            """
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 10px;
                color: #374151;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #2563eb;
            }
            """
        )
        self.ed_global_repo.editingFinished.connect(self._act_global_repo_changed)
        btn_repo = QtWidgets.QPushButton("Browse...")
        btn_repo.setMinimumHeight(button_min_h)
        btn_repo.setStyleSheet(
            """
            QPushButton {
                padding: 6px 12px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
            """
        )
        btn_repo.clicked.connect(lambda: self._browse_folder(self.ed_global_repo, be.DEFAULT_PDF_DIR))
        repo_row.addWidget(self.ed_global_repo, 1)
        repo_row.addWidget(btn_repo)
        repo_layout.addLayout(repo_row)

        self.lbl_support_status = QtWidgets.QLabel("Support: not initialized")
        self.lbl_support_status.setWordWrap(True)
        self.lbl_support_status.setStyleSheet("color: #374151; font-size: 12px;")
        repo_layout.addWidget(self.lbl_support_status)

        self.lbl_index_status = QtWidgets.QLabel("Index: not built")
        self.lbl_index_status.setWordWrap(True)
        self.lbl_index_status.setStyleSheet("color: #374151; font-size: 12px;")
        repo_layout.addWidget(self.lbl_index_status)

        scan_row = QtWidgets.QHBoxLayout()
        scan_row.setSpacing(8)
        self.btn_manager_scan_all = QtWidgets.QPushButton("Scan All (EIDAT Manager)")
        self.btn_manager_scan_all.setMinimumHeight(button_min_h)
        self.btn_manager_scan_all.setStyleSheet(
            """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                background: #93c5fd;
                border-color: #93c5fd;
            }
            """
        )
        self.btn_manager_scan_all.clicked.connect(self._act_manager_scan_all)
        scan_row.addWidget(self.btn_manager_scan_all)

        self.btn_manager_process_new = QtWidgets.QPushButton("Process New Files")
        self.btn_manager_process_new.setMinimumHeight(button_min_h)
        self.btn_manager_process_new.setStyleSheet(
            """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
            """
        )
        self.btn_manager_process_new.clicked.connect(self._act_manager_process_new)
        scan_row.addWidget(self.btn_manager_process_new)

        self.btn_manager_force_all = QtWidgets.QPushButton("Force Process All (Debug)")
        self.btn_manager_force_all.setMinimumHeight(button_min_h)
        self.btn_manager_force_all.setStyleSheet(
            """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #ffffff;
                color: #991b1b;
                border: 1px solid #fca5a5;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #fef2f2;
                border-color: #f87171;
            }
            """
        )
        self.btn_manager_force_all.clicked.connect(self._act_manager_force_all)
        scan_row.addWidget(self.btn_manager_force_all)
        scan_row.addStretch(1)
        repo_layout.addLayout(scan_row)

        index_row = QtWidgets.QHBoxLayout()
        index_row.setSpacing(8)
        self.btn_manager_index = QtWidgets.QPushButton("Show Index Summary")
        self.btn_manager_index.setMinimumHeight(button_min_h)
        self.btn_manager_index.setStyleSheet(
            """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
            """
        )
        self.btn_manager_index.clicked.connect(self._act_manager_index)
        index_row.addWidget(self.btn_manager_index)
        index_row.addStretch(1)
        repo_layout.addLayout(index_row)

        # Extraction Settings
        grp_set = QtWidgets.QGroupBox("Extraction Settings")
        grp_set.setStyleSheet("""
            QGroupBox {
                font-weight: 900;
                font-size: 24px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 16px;
                background: #ffffff;
                padding: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: #111827;
            }
        """)
        ls = QtWidgets.QVBoxLayout(grp_set)
        ls.setSpacing(16)

        # Description
        hint = QtWidgets.QLabel("Configure how EIDAT processes and extracts data from documents")
        hint.setStyleSheet("color: #6b7280; font-size: 13px; font-weight: 400;")
        ls.addWidget(hint)

        # OCR Mode
        ocr_container = QtWidgets.QWidget()
        ocr_layout = QtWidgets.QVBoxLayout(ocr_container)
        ocr_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.setSpacing(6)
        ocr_label = QtWidgets.QLabel("OCR Mode")
        ocr_label.setStyleSheet("color: #374151; font-size: 13px; font-weight: 500;")
        ocr_layout.addWidget(ocr_label)
        # Friendly OCR mode labels mapped to env values
        self._ocr_value_to_display = {
            "fallback": "Read PDF and fallback to OCR if needed",
            "ocr_only": "Only OCR read the selected documents",
            "no_ocr": "Read PDF, no OCR (may fail)",
        }
        self._ocr_display_to_value = {v: k for k, v in self._ocr_value_to_display.items()}
        self.cmb_ocr_mode = QtWidgets.QComboBox()
        self.cmb_ocr_mode.setStyleSheet("""
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px 12px;
                color: #374151;
                font-size: 13px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #9ca3af;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                color: #374151;
                background: #ffffff;
                selection-background-color: #dbeafe;
                border: 1px solid #d1d5db;
                padding: 4px;
            }
        """)
        self.cmb_ocr_mode.addItems(list(self._ocr_value_to_display.values()))
        self.cmb_ocr_mode.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        # Enable scrollbar for dropdown if needed
        self.cmb_ocr_mode.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ocr_layout.addWidget(self.cmb_ocr_mode)
        ls.addWidget(ocr_container)

        # OCR line Y tolerance (row grouping)
        ytol_container = QtWidgets.QWidget()
        ytol_layout = QtWidgets.QVBoxLayout(ytol_container)
        ytol_layout.setContentsMargins(0, 0, 0, 0)
        ytol_layout.setSpacing(6)
        ytol_header = QtWidgets.QHBoxLayout()
        ytol_label = QtWidgets.QLabel("OCR Line Y Tolerance")
        ytol_label.setStyleSheet("color: #374151; font-size: 13px; font-weight: 500;")
        ytol_header.addWidget(ytol_label)
        ytol_info = QtWidgets.QLabel("\u24D8")
        ytol_info.setStyleSheet("color: #9ca3af; font-size: 14px;")
        ytol_info.setToolTip("Vertical tolerance (pixels) for grouping OCR boxes into a single text line (higher = more lenient)")
        ytol_header.addWidget(ytol_info)
        ytol_header.addStretch()
        self.lbl_ocr_row_tol = QtWidgets.QLabel("15")
        self.lbl_ocr_row_tol.setStyleSheet("color: #111827; font-size: 14px; font-weight: 600;")
        ytol_header.addWidget(self.lbl_ocr_row_tol)
        ytol_layout.addLayout(ytol_header)
        self.sld_ocr_row_tol = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_ocr_row_tol.setRange(2, 40)
        self.sld_ocr_row_tol.setValue(15)
        self.sld_ocr_row_tol.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #2563eb;
            }
            QSlider::handle:horizontal:hover {
                background: #1d4ed8;
            }
        """)
        ytol_layout.addWidget(self.sld_ocr_row_tol)
        ls.addWidget(ytol_container)

        # OCR DPI
        dpi_container = QtWidgets.QWidget()
        dpi_layout = QtWidgets.QVBoxLayout(dpi_container)
        dpi_layout.setContentsMargins(0, 0, 0, 0)
        dpi_layout.setSpacing(6)
        dpi_header = QtWidgets.QHBoxLayout()
        dpi_label = QtWidgets.QLabel("OCR DPI")
        dpi_label.setStyleSheet("color: #374151; font-size: 13px; font-weight: 500;")
        dpi_header.addWidget(dpi_label)
        dpi_info = QtWidgets.QLabel("\u24D8")
        dpi_info.setStyleSheet("color: #9ca3af; font-size: 14px;")
        dpi_info.setToolTip("Higher DPI may improve OCR accuracy at the cost of speed")
        dpi_header.addWidget(dpi_info)
        dpi_header.addStretch()
        self.lbl_dpi_val = QtWidgets.QLabel("500")
        self.lbl_dpi_val.setStyleSheet("color: #111827; font-size: 14px; font-weight: 600;")
        dpi_header.addWidget(self.lbl_dpi_val)
        dpi_layout.addLayout(dpi_header)
        self.sld_ocr_dpi = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_ocr_dpi.setRange(100, 1000)
        self.sld_ocr_dpi.setSingleStep(50)
        self.sld_ocr_dpi.setValue(500)
        self.sld_ocr_dpi.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #2563eb;
            }
            QSlider::handle:horizontal:hover {
                background: #1d4ed8;
            }
        """)
        dpi_layout.addWidget(self.sld_ocr_dpi)
        ls.addWidget(dpi_container)

        # Fuzzy Matching Preset (for multi-word search terms with OCR errors)
        fuzzy_container = QtWidgets.QWidget()
        fuzzy_layout = QtWidgets.QVBoxLayout(fuzzy_container)
        fuzzy_layout.setContentsMargins(0, 0, 0, 0)
        fuzzy_layout.setSpacing(6)
        fuzzy_header = QtWidgets.QHBoxLayout()
        fuzzy_label = QtWidgets.QLabel("Fuzzy Matching Strictness")
        fuzzy_label.setStyleSheet("color: #374151; font-size: 13px; font-weight: 500;")
        fuzzy_header.addWidget(fuzzy_label)
        fuzzy_info = QtWidgets.QLabel("\u24D8")
        fuzzy_info.setStyleSheet("color: #9ca3af; font-size: 14px;")
        fuzzy_info.setToolTip("Controls how strictly multi-word search terms (e.g., 'Seats Closed') must match OCR text\n\nLenient: For poor OCR quality, tolerates more errors\nMedium: Balanced (recommended for typical scans)\nStrict: For high-quality OCR, requires closer matches")
        fuzzy_header.addWidget(fuzzy_info)
        fuzzy_header.addStretch()
        fuzzy_layout.addLayout(fuzzy_header)

        # Preset mapping for fuzzy matching
        self._fuzzy_value_to_display = {
            "lenient": "Lenient (Poor OCR)",
            "medium": "Medium (Balanced)",
            "strict": "Strict (High Quality OCR)"
        }
        self._fuzzy_display_to_value = {v: k for k, v in self._fuzzy_value_to_display.items()}

        self.cmb_fuzzy_preset = QtWidgets.QComboBox()
        self.cmb_fuzzy_preset.setStyleSheet("""
            QComboBox {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px 12px;
                color: #374151;
                font-size: 13px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #9ca3af;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                color: #374151;
                background: #ffffff;
                selection-background-color: #dbeafe;
                border: 1px solid #d1d5db;
                padding: 4px;
            }
        """)
        self.cmb_fuzzy_preset.addItems(list(self._fuzzy_value_to_display.values()))
        self.cmb_fuzzy_preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cmb_fuzzy_preset.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Set default to medium
        self.cmb_fuzzy_preset.setCurrentText(self._fuzzy_value_to_display["medium"])
        fuzzy_layout.addWidget(self.cmb_fuzzy_preset)
        ls.addWidget(fuzzy_container)

        # Show detailed debug logs toggle
        debug_container = QtWidgets.QWidget()
        debug_layout = QtWidgets.QHBoxLayout(debug_container)
        debug_layout.setContentsMargins(0, 0, 0, 0)
        debug_layout.setSpacing(8)
        debug_left = QtWidgets.QVBoxLayout()
        debug_left.setSpacing(2)
        debug_title = QtWidgets.QLabel("Show detailed debug logs")
        debug_title.setStyleSheet("color: #374151; font-size: 13px; font-weight: 500;")
        debug_desc = QtWidgets.QLabel("(slightly slower)")
        debug_desc.setStyleSheet("color: #9ca3af; font-size: 12px;")
        debug_left.addWidget(debug_title)
        debug_left.addWidget(debug_desc)
        debug_layout.addLayout(debug_left)
        debug_layout.addStretch()
        self.chk_logging = QtWidgets.QCheckBox()
        self.chk_logging.setStyleSheet("""
            QCheckBox::indicator {
                width: 40px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 10px;
                background: #d1d5db;
            }
            QCheckBox::indicator:checked {
                border-radius: 10px;
                background: #2563eb;
            }
        """)
        debug_layout.addWidget(self.chk_logging)
        ls.addWidget(debug_container)

        # Persist on change
        self.cmb_ocr_mode.currentTextChanged.connect(self._persist_settings_from_panel)
        self.sld_ocr_row_tol.valueChanged.connect(self._on_ocr_row_tol_slider)
        self.sld_ocr_dpi.valueChanged.connect(self._on_dpi_slider)
        self.cmb_fuzzy_preset.currentTextChanged.connect(self._persist_settings_from_panel)
        self.chk_logging.stateChanged.connect(self._persist_settings_from_panel)

        grid.addWidget(grp_env, 0, 0, 1, 2)
        grid.addWidget(grp_repo, 1, 0, 1, 2)
        grp_set.setVisible(False)
        grid.addWidget(grp_set, 2, 0, 1, 2)
        grid.setRowStretch(3, 1)

    def _setup_tab_process(self):
        main_layout = QtWidgets.QVBoxLayout(self.tab_process)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)
        main_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        intro = QtWidgets.QLabel(
            "Work through the guided steps: confirm your workspace, prepare inputs, run extraction, and update the database."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #374151; font-size: 12px;")
        main_layout.addWidget(intro)

        card_style = """
            QGroupBox {
                font-weight: 900;
                font-size: 18px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 6px;
                background: #ffffff;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: #111827;
            }
        """
        button_min_h = 38

        # Master Database reference (moved to top)
        grp_master = QtWidgets.QGroupBox("Master Database")
        grp_master.setStyleSheet(card_style)
        master_layout = QtWidgets.QVBoxLayout(grp_master)
        master_layout.setSpacing(10)

        master_desc = QtWidgets.QLabel("Open and manage the central EIDAT master database workbook.")
        master_desc.setWordWrap(True)
        master_desc.setStyleSheet("color: #6b7280; font-size: 12px; font-weight: 400;")
        master_layout.addWidget(master_desc)

        self.btn_open_master_tab = QtWidgets.QPushButton("🗄  Open Master Database")
        self.btn_open_master_tab.setMinimumHeight(button_min_h)
        self.btn_open_master_tab.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
                text-align: center;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        self.btn_open_master_tab.clicked.connect(lambda: self._safe_open(be.open_master_workbook))
        master_layout.addWidget(self.btn_open_master_tab)

        main_layout.addWidget(grp_master)

        # Two-column container for steps
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(12)

        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(12)
        right_column = QtWidgets.QVBoxLayout()
        right_column.setSpacing(12)

        # Step 1: Workspace & Sync
        grp_workspace = QtWidgets.QGroupBox("Step 1: Workspace & Sync")
        grp_workspace.setStyleSheet(card_style)
        workspace_layout = QtWidgets.QVBoxLayout(grp_workspace)
        workspace_layout.setSpacing(10)

        workspace_desc = QtWidgets.QLabel("Point to your EIDP repository and sync the workspace before processing.")
        workspace_desc.setWordWrap(True)
        workspace_desc.setStyleSheet("color: #6b7280; font-size: 12px; font-weight: 400;")
        workspace_layout.addWidget(workspace_desc)

        repo_label = QtWidgets.QLabel("Repository Root")
        repo_label.setStyleSheet("color: #374151; font-size: 12px; font-weight: 600; margin-top: 2px;")
        workspace_layout.addWidget(repo_label)

        repo_row = QtWidgets.QHBoxLayout()
        repo_row.setSpacing(6)
        self.ed_repo = QtWidgets.QLineEdit(str(getattr(be, "get_repo_root", lambda: be.DEFAULT_REPO_ROOT)()))
        self.ed_repo.setMinimumHeight(32)
        self.ed_repo.setStyleSheet("""
            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 10px;
                color: #374151;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #2563eb;
            }
        """)
        btn_repo = QtWidgets.QPushButton("Browse...")
        btn_repo.setMinimumHeight(button_min_h)
        btn_repo.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        btn_repo.clicked.connect(lambda: self._browse_folder(self.ed_repo, be.DEFAULT_PDF_DIR))
        repo_row.addWidget(self.ed_repo, 1)
        repo_row.addWidget(btn_repo)
        workspace_layout.addLayout(repo_row)

        self.btn_sync_workspace = QtWidgets.QPushButton("⭳  Sync Workspace Now")
        self.btn_sync_workspace.setMinimumHeight(button_min_h)
        self.btn_sync_workspace.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                background: #93c5fd;
                border-color: #93c5fd;
            }
        """)
        self.btn_sync_workspace.clicked.connect(self._act_sync_workspace)
        workspace_layout.addWidget(self.btn_sync_workspace)

        self.lbl_sync_banner = QtWidgets.QLabel("No sync run yet.")
        self.lbl_sync_banner.setObjectName("syncBanner")
        self.lbl_sync_banner.setWordWrap(True)
        self.lbl_sync_banner.setStyleSheet(
            "background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; border-radius: 6px; padding: 8px 10px; font-size: 11px;"
        )
        workspace_layout.addWidget(self.lbl_sync_banner)

        left_column.addWidget(grp_workspace)

        # Step 2: Prepare Inputs
        grp_inputs = QtWidgets.QGroupBox("Step 2: Prepare Inputs")
        grp_inputs.setStyleSheet(card_style)
        inputs_layout = QtWidgets.QVBoxLayout(grp_inputs)
        inputs_layout.setSpacing(10)

        inputs_desc = QtWidgets.QLabel("Define simple schema terms and refresh the spreadsheet that drives extraction.")
        inputs_desc.setWordWrap(True)
        inputs_desc.setStyleSheet("color: #6b7280; font-size: 12px; font-weight: 400;")
        inputs_layout.addWidget(inputs_desc)

        self.btn_terms_edit = QtWidgets.QPushButton("✎  Edit Simple Schema Terms")
        self.btn_terms_edit.setMinimumHeight(button_min_h)
        self.btn_terms_edit.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                background: #93c5fd;
                border-color: #93c5fd;
            }
        """)
        self.btn_terms_edit.clicked.connect(self._open_terms_editor)
        inputs_layout.addWidget(self.btn_terms_edit)

        self.btn_terms_refresh = QtWidgets.QPushButton("📄  Create/Refresh Input Spreadsheet")
        self.btn_terms_refresh.setMinimumHeight(button_min_h)
        self.btn_terms_refresh.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        self.btn_terms_refresh.clicked.connect(self._act_generate_terms)
        inputs_layout.addWidget(self.btn_terms_refresh)

        left_column.addWidget(grp_inputs)
        left_column.addStretch(1)

        # Step 3: Run & Update (combined)
        grp_run = QtWidgets.QGroupBox("Step 3: Run & Update")
        grp_run.setStyleSheet(card_style)
        run_layout = QtWidgets.QVBoxLayout(grp_run)
        run_layout.setSpacing(10)

        run_desc = QtWidgets.QLabel("Process PDFs with current terms, then push updates and clean up.")
        run_desc.setWordWrap(True)
        run_desc.setStyleSheet("color: #6b7280; font-size: 12px; font-weight: 400;")
        run_layout.addWidget(run_desc)

        self.btn_start = QtWidgets.QPushButton("▶  Extract and Update All")
        self.btn_start.setMinimumHeight(button_min_h)
        self.btn_start.setStyleSheet("""
            QPushButton {
                padding: 10px 18px;
                border-radius: 6px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                background: #93c5fd;
                border-color: #93c5fd;
            }
        """)
        self.btn_start.clicked.connect(self._show_extraction_options)
        run_layout.addWidget(self.btn_start)

        # Excel data extraction button
        self.btn_start_excel = QtWidgets.QPushButton("📊  Extract Excel Data")
        self.btn_start_excel.setMinimumHeight(button_min_h)
        self.btn_start_excel.setStyleSheet("""
            QPushButton {
                padding: 10px 18px;
                border-radius: 6px;
                background: #059669;
                color: #ffffff;
                border: 1px solid #059669;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #047857;
            }
            QPushButton:disabled {
                background: #6ee7b7;
                border-color: #6ee7b7;
            }
        """)
        self.btn_start_excel.clicked.connect(self._act_start_excel_scan)
        run_layout.addWidget(self.btn_start_excel)

        self.btn_open_run_data = QtWidgets.QPushButton("📂  Open Run Data Folder")
        self.btn_open_run_data.setMinimumHeight(button_min_h)
        self.btn_open_run_data.setStyleSheet("""
            QPushButton {
                padding: 8px 14px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        self.btn_open_run_data.clicked.connect(lambda: self._safe_open(be.open_run_data_root))
        run_layout.addWidget(self.btn_open_run_data)

        # Update & review section inside the same card
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #e5e7eb; margin: 6px 0;")
        run_layout.addWidget(divider)

        self.btn_view_outdated = QtWidgets.QPushButton("📋  MANUAL UPDATE / INDIVIDUAL SELECTION")
        self.btn_view_outdated.setMinimumHeight(button_min_h)
        self.btn_view_outdated.setStyleSheet("""
            QPushButton {
                padding: 10px 18px;
                border-radius: 6px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
        """)
        self.btn_view_outdated.clicked.connect(self._show_outdated_popup)
        run_layout.addWidget(self.btn_view_outdated)

        secondary_row = QtWidgets.QHBoxLayout()
        secondary_row.setSpacing(8)

        self.btn_view_registry2 = QtWidgets.QPushButton("📖  View Registry")
        self.btn_view_registry2.setMinimumHeight(button_min_h)
        self.btn_view_registry2.setStyleSheet("""
            QPushButton {
                padding: 8px 14px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        self.btn_view_registry2.clicked.connect(self._act_view_registry)

        self.btn_clear_old_runs = QtWidgets.QPushButton("🗑  Clear Old Run Cache")
        self.btn_clear_old_runs.setMinimumHeight(button_min_h)
        self.btn_clear_old_runs.setStyleSheet("""
            QPushButton {
                padding: 8px 14px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        self.btn_clear_old_runs.clicked.connect(self._act_clear_old_runs)

        secondary_row.addWidget(self.btn_view_registry2)
        secondary_row.addWidget(self.btn_clear_old_runs)
        run_layout.addLayout(secondary_row)

        advanced_label = QtWidgets.QLabel("Advanced: rebuild the master workbook from state (overwrites manual edits).")
        advanced_label.setWordWrap(True)
        advanced_label.setStyleSheet("color: #b91c1c; font-size: 11px; font-weight: 600;")
        run_layout.addWidget(advanced_label)

        self.btn_compile_master = QtWidgets.QPushButton("📊  Compile New Master Workbook")
        self.btn_compile_master.setMinimumHeight(button_min_h)
        self.btn_compile_master.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                background: #dc2626;
                color: #ffffff;
                border: 1px solid #dc2626;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #b91c1c;
            }
            QPushButton:disabled {
                background: #fca5a5;
                border-color: #fca5a5;
            }
        """)
        self.btn_compile_master.setToolTip(
            "Rebuild master.xlsx from simple extraction outputs ONLY.\n"
            "WARNING: This will overwrite master.xlsx and discard all manual edits!"
        )
        self.btn_compile_master.clicked.connect(self._act_compile_master_from_state)
        run_layout.addWidget(self.btn_compile_master)

        right_column.addWidget(grp_run)
        right_column.addStretch(1)

        columns.addLayout(left_column, 1)
        columns.addLayout(right_column, 1)
        main_layout.addLayout(columns)
        main_layout.addStretch(1)

        # Keep internal fields for logic
        self.ed_terms = QtWidgets.QLineEdit(str(be.DEFAULT_TERMS_XLSX))
        self.ed_terms.setVisible(False)
        self.ed_pdfs = QtWidgets.QLineEdit(str(be.DEFAULT_PDF_DIR))
        self.ed_pdfs.setVisible(False)

    # NOTE: _setup_tab_plot was removed - it was never called and the Plot tab UI was unused


    # NOTE: _setup_tab_outputs was removed - it was never called and the Outputs tab UI was unused

    def _setup_tab_files(self):
        """Setup the Files tab with tree+table split view."""
        root = QtWidgets.QHBoxLayout(self.tab_files)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(14)

        # ============ LEFT PANE: Tree Navigation ============
        left = QtWidgets.QFrame()
        left.setFixedWidth(280)
        left.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
        """)
        l = QtWidgets.QVBoxLayout(left)
        l.setContentsMargins(14, 14, 14, 14)
        l.setSpacing(10)

        # Header
        lbl = QtWidgets.QLabel("File Browser")
        lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        l.addWidget(lbl)

        # Grouping mode selectors - Primary and Secondary
        combo_style = """
            QComboBox {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
                border: 1px solid #d1d5db;
            }
        """
        grouping_options = ["Program", "Asset Type", "Serial Number", "Doc Type"]

        # Primary grouping
        primary_row = QtWidgets.QHBoxLayout()
        primary_lbl = QtWidgets.QLabel("Primary:")
        primary_lbl.setStyleSheet("font-size: 12px; color: #6b7280;")
        self.cmb_files_primary = QtWidgets.QComboBox()
        self.cmb_files_primary.addItems(grouping_options)
        self.cmb_files_primary.setStyleSheet(combo_style)
        self.cmb_files_primary.currentIndexChanged.connect(self._refresh_files_tree)
        primary_row.addWidget(primary_lbl)
        primary_row.addWidget(self.cmb_files_primary, 1)
        l.addLayout(primary_row)

        # Secondary grouping (optional)
        secondary_row = QtWidgets.QHBoxLayout()
        secondary_lbl = QtWidgets.QLabel("Secondary:")
        secondary_lbl.setStyleSheet("font-size: 12px; color: #6b7280;")
        self.cmb_files_secondary = QtWidgets.QComboBox()
        self.cmb_files_secondary.addItems(["None"] + grouping_options)
        self.cmb_files_secondary.setStyleSheet(combo_style)
        self.cmb_files_secondary.currentIndexChanged.connect(self._refresh_files_tree)
        secondary_row.addWidget(secondary_lbl)
        secondary_row.addWidget(self.cmb_files_secondary, 1)
        l.addLayout(secondary_row)

        # Tree widget
        self.tree_files = QtWidgets.QTreeWidget()
        self.tree_files.setHeaderHidden(True)
        self.tree_files.setStyleSheet("""
            QTreeWidget {
                background: #ffffff;
                color: #1f2937;
                border: none;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 4px 8px;
                color: #1f2937;
            }
            QTreeWidget::item:selected {
                background: #dbeafe;
                color: #1e40af;
            }
            QTreeWidget::item:hover:!selected {
                background: #f1f5f9;
                color: #1f2937;
            }
        """)
        self.tree_files.itemSelectionChanged.connect(self._on_files_tree_selection)
        l.addWidget(self.tree_files, 1)

        # Refresh button
        self.btn_files_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_files_refresh.setStyleSheet("""
            QPushButton {
                padding: 8px 14px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f9fafb; }
        """)
        self.btn_files_refresh.clicked.connect(self._act_files_refresh)
        l.addWidget(self.btn_files_refresh)

        # Status label
        self.lbl_files_status = QtWidgets.QLabel("Select Global Repo in Setup")
        self.lbl_files_status.setStyleSheet("font-size: 11px; color: #6b7280;")
        self.lbl_files_status.setWordWrap(True)
        l.addWidget(self.lbl_files_status)

        # ============ RIGHT PANE: Table View ============
        right = QtWidgets.QFrame()
        right.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
        """)
        r = QtWidgets.QVBoxLayout(right)
        r.setContentsMargins(14, 14, 14, 14)
        r.setSpacing(10)

        # Header with filter
        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Files")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        self.lbl_files_count = QtWidgets.QLabel("")
        self.lbl_files_count.setStyleSheet("font-size: 11px; color: #6b7280;")
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(self.lbl_files_count)
        r.addLayout(header_row)

        # Search/filter box
        filter_row = QtWidgets.QHBoxLayout()
        self.ed_files_filter = QtWidgets.QLineEdit()
        self.ed_files_filter.setPlaceholderText("Filter across all metadata columns (filename, serial, vendor, hashes, ...)")
        self.ed_files_filter.setStyleSheet("""
            QLineEdit {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
        """)
        self.ed_files_filter.textChanged.connect(self._apply_files_filter)
        filter_row.addWidget(self.ed_files_filter, 1)

        # Column visibility menu
        self.btn_files_columns = QtWidgets.QPushButton("Columns")
        self.btn_files_columns.setStyleSheet("""
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f9fafb; }
        """)
        self.btn_files_columns.clicked.connect(self._files_open_columns_menu)
        filter_row.addWidget(self.btn_files_columns)
        r.addLayout(filter_row)

        # Deep search (combined.txt content)
        deep_row = QtWidgets.QHBoxLayout()
        self.cmb_files_deep_search = QtWidgets.QComboBox()
        self.cmb_files_deep_search.setEditable(True)
        try:
            le = self.cmb_files_deep_search.lineEdit()
            le.setPlaceholderText("Deep search in combined.txt (OCR content) — click Search to find matching documents")
            le.setClearButtonEnabled(True)
            le.returnPressed.connect(self._act_files_deep_search)
        except Exception:
            pass
        self.cmb_files_deep_search.setStyleSheet("""
            QComboBox {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2937;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
                border: 1px solid #d1d5db;
            }
        """)
        try:
            self.cmb_files_deep_search.textActivated.connect(self._act_files_deep_pick_result)
        except Exception:
            # Fallback for older Qt bindings
            try:
                self.cmb_files_deep_search.activated.connect(lambda _idx: self._act_files_deep_pick_result(self.cmb_files_deep_search.currentText()))
            except Exception:
                pass
        deep_row.addWidget(self.cmb_files_deep_search, 1)

        self.btn_files_deep_search = QtWidgets.QPushButton("Search")
        self.btn_files_deep_clear = QtWidgets.QPushButton("Clear")
        for b in (self.btn_files_deep_search, self.btn_files_deep_clear):
            b.setStyleSheet("""
                QPushButton {
                    padding: 6px 10px;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
            """)
        self.btn_files_deep_search.clicked.connect(self._act_files_deep_search)
        self.btn_files_deep_clear.clicked.connect(self._act_files_deep_clear)
        deep_row.addWidget(self.btn_files_deep_search)
        deep_row.addWidget(self.btn_files_deep_clear)
        r.addLayout(deep_row)

        self.lbl_files_deep_status = QtWidgets.QLabel("")
        self.lbl_files_deep_status.setStyleSheet("font-size: 11px; color: #6b7280;")
        self.lbl_files_deep_status.setWordWrap(True)
        r.addWidget(self.lbl_files_deep_status)

        files_subset_row = QtWidgets.QHBoxLayout()
        self.lbl_files_subset = QtWidgets.QLabel("")
        self.lbl_files_subset.setWordWrap(True)
        self.lbl_files_subset.setStyleSheet(
            "font-size: 11px; color: #1e3a8a; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 6px 8px;"
        )
        self.btn_files_clear_subset = QtWidgets.QPushButton("Clear")
        self.btn_files_clear_subset.setStyleSheet(
            """
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f9fafb; }
            """
        )
        self.btn_files_clear_subset.clicked.connect(self._clear_files_external_subset)
        files_subset_row.addWidget(self.lbl_files_subset, 1)
        files_subset_row.addWidget(self.btn_files_clear_subset)
        r.addLayout(files_subset_row)

        # Table
        # Include all available metadata as columns so filtering can search across everything.
        # (Support DB: files.*) + (Index DB: documents.*)
        self._files_table_cols: list[tuple[str, str]] = [
            ("File Name", "_filename"),
            ("Program", "program_title"),
            ("Asset", "asset_type"),
            ("Asset Specific", "asset_specific_type"),
            ("Metadata Source", "metadata_source"),
            ("Manual Override Fields", "manual_override_fields"),
            ("Serial", "serial_number"),
            ("Part #", "part_number"),
            ("Revision", "revision"),
            ("Doc Type", "document_type"),
            ("Doc Acronym", "document_type_acronym"),
            ("Vendor", "vendor"),
            ("ATP #", "acceptance_test_plan_number"),
            ("Report Date", "report_date"),
            ("Test Date", "test_date"),
            ("Manual Override Updated", "manual_override_updated_at"),
            ("Applied Rule", "applied_asset_specific_type_rule"),
            ("Processed", "_processed"),
            ("Needs Processing", "needs_processing"),
            ("Certification", "_certification"),
            ("Cert Status", "certification_status"),
            ("Cert Pass Rate", "certification_pass_rate"),
            ("Similarity Group", "similarity_group"),
            ("Title Norm", "title_norm"),
            ("File Ext", "file_extension"),
            ("Indexed", "indexed_epoch_ns"),
            ("Rel Path", "rel_path"),
            ("Metadata Rel", "metadata_rel"),
            ("Artifacts Rel", "artifacts_rel"),
            ("Excel SQLite Rel", "excel_sqlite_rel"),
            ("Size (bytes)", "size_bytes"),
            ("Modified", "mtime_ns"),
            ("First Seen", "first_seen_epoch_ns"),
            ("Last Seen", "last_seen_epoch_ns"),
            ("Last Processed", "last_processed_epoch_ns"),
            ("Last Proc mtime", "last_processed_mtime_ns"),
            ("Fingerprint", "file_fingerprint"),
            ("Content SHA1", "content_sha1"),
            ("EIDAT UUID", "eidat_uuid"),
            ("Pointer Token", "pointer_token"),
            ("File ID", "id"),
            ("Document ID", "document_id"),
        ]
        cols = [c[0] for c in self._files_table_cols]
        self.tbl_files = QtWidgets.QTableWidget(0, len(cols))
        self.tbl_files.setHorizontalHeaderLabels(cols)
        self.tbl_files.verticalHeader().setVisible(False)
        self.tbl_files.setAlternatingRowColors(True)
        self.tbl_files.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tbl_files.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_files.horizontalHeader().setStretchLastSection(True)
        self.tbl_files.setSortingEnabled(True)
        self.tbl_files.setStyleSheet("""
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f9fafb;
                color: #1f2937;
                gridline-color: #e5e7eb;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
            }
            QTableWidget::item {
                color: #1f2937;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #1f2937;
                padding: 6px 8px;
                border: 1px solid #e5e7eb;
                font-weight: 600;
            }
        """)
        self.tbl_files.doubleClicked.connect(self._act_files_open_pdf)
        self.tbl_files.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tbl_files.customContextMenuRequested.connect(self._files_context_menu)
        r.addWidget(self.tbl_files, 1)

        # Header right-click to show/hide columns
        try:
            hdr = self.tbl_files.horizontalHeader()
            hdr.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            hdr.customContextMenuRequested.connect(self._files_header_context_menu)
        except Exception:
            pass

        # Default column visibility (users can override via Columns menu)
        self._files_load_column_visibility()

        # Action buttons
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.btn_files_open_pdf = QtWidgets.QPushButton("Open File")
        self.btn_files_open_metadata = QtWidgets.QPushButton("Open Metadata")
        self.btn_files_open_artifacts = QtWidgets.QPushButton("Open Artifacts")
        self.btn_files_open_combined = QtWidgets.QPushButton("Open combined.txt")
        self.btn_files_show_explorer = QtWidgets.QPushButton("Show in Explorer")
        self.btn_files_edit_metadata = QtWidgets.QPushButton("Edit Metadata")
        self.btn_files_update_metadata = QtWidgets.QPushButton("Update Metadata")
        self.btn_files_certify_all = QtWidgets.QPushButton("Certify All")

        self.btn_files_open_combined.setToolTip("Open the merged OCR artifact combined.txt for this document (if present).")

        for b in (self.btn_files_open_pdf, self.btn_files_open_metadata,
                  self.btn_files_open_artifacts, self.btn_files_open_combined, self.btn_files_show_explorer,
                  self.btn_files_edit_metadata,
                  self.btn_files_update_metadata, self.btn_files_certify_all):
            b.setStyleSheet("""
                QPushButton {
                    padding: 8px 14px;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
            """)

        self.btn_files_open_pdf.clicked.connect(self._act_files_open_pdf)
        self.btn_files_open_metadata.clicked.connect(self._act_files_open_metadata)
        self.btn_files_open_artifacts.clicked.connect(self._act_files_open_artifacts)
        self.btn_files_open_combined.clicked.connect(self._act_files_open_combined)
        self.btn_files_show_explorer.clicked.connect(self._act_files_show_explorer)
        self.btn_files_edit_metadata.clicked.connect(self._act_files_edit_metadata)
        self.btn_files_update_metadata.clicked.connect(self._act_files_update_metadata)
        self.btn_files_certify_all.clicked.connect(self._act_files_certify_all)

        actions.addWidget(self.btn_files_open_pdf)
        actions.addWidget(self.btn_files_open_metadata)
        actions.addWidget(self.btn_files_open_artifacts)
        actions.addWidget(self.btn_files_open_combined)
        actions.addWidget(self.btn_files_show_explorer)
        actions.addWidget(self.btn_files_edit_metadata)
        actions.addWidget(self.btn_files_update_metadata)
        actions.addWidget(self.btn_files_certify_all)
        r.addLayout(actions)

        root.addWidget(left)
        root.addWidget(right, 1)

        # Store file data for filtering
        self._files_data: list[dict] = []
        self._files_filtered: list[dict] = []
        self._update_files_external_filter_banner()

    def _setup_tab_product_center(self):
        root = QtWidgets.QHBoxLayout(self.tab_product_center)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(14)

        left = QtWidgets.QFrame()
        left.setFixedWidth(360)
        left.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            """
        )
        l = QtWidgets.QVBoxLayout(left)
        l.setContentsMargins(14, 14, 14, 14)
        l.setSpacing(10)

        title = QtWidgets.QLabel("Product Center")
        title.setStyleSheet("font-size: 15px; font-weight: 800; color: #0f172a;")
        desc = QtWidgets.QLabel(
            "Browse products built from Asset Type, Asset Specific Type, and Vendor. "
            "Use this as the visual launch point into documents and linked projects."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #475569;")
        l.addWidget(title)
        l.addWidget(desc)

        self.ed_product_center_search = QtWidgets.QLineEdit()
        self.ed_product_center_search.setPlaceholderText("Search products, vendors, part numbers, serials, document types ...")
        self.ed_product_center_search.setStyleSheet(
            """
            QLineEdit {
                padding: 8px 10px;
                border-radius: 8px;
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            """
        )
        self.ed_product_center_search.textChanged.connect(self._apply_product_center_filter)
        l.addWidget(self.ed_product_center_search)

        self.lbl_product_center_status = QtWidgets.QLabel("")
        self.lbl_product_center_status.setWordWrap(True)
        self.lbl_product_center_status.setStyleSheet("font-size: 11px; color: #64748b;")
        l.addWidget(self.lbl_product_center_status)

        self.list_product_center = QtWidgets.QListWidget()
        self.list_product_center.setIconSize(QtCore.QSize(72, 72))
        self.list_product_center.setSpacing(6)
        self.list_product_center.setStyleSheet(
            """
            QListWidget {
                background: #ffffff;
                color: #1f2937;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 10px;
                margin: 0px;
            }
            QListWidget::item:selected {
                background: #eff6ff;
                color: #1e3a8a;
                border: 1px solid #93c5fd;
            }
            QListWidget::item:hover:!selected {
                background: #f8fafc;
            }
            """
        )
        self.list_product_center.currentRowChanged.connect(self._on_product_center_selection_changed)
        l.addWidget(self.list_product_center, 1)

        actions = QtWidgets.QHBoxLayout()
        self.btn_product_center_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_product_center_open_image_folder = QtWidgets.QPushButton("Open Image Folder")
        for btn in (self.btn_product_center_refresh, self.btn_product_center_open_image_folder):
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 12px;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
                """
            )
        self.btn_product_center_refresh.clicked.connect(self._refresh_product_center)
        self.btn_product_center_open_image_folder.clicked.connect(self._act_product_center_open_image_folder)
        actions.addWidget(self.btn_product_center_refresh)
        actions.addWidget(self.btn_product_center_open_image_folder)
        l.addLayout(actions)

        right = QtWidgets.QFrame()
        right.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            """
        )
        r = QtWidgets.QVBoxLayout(right)
        r.setContentsMargins(0, 0, 0, 0)
        r.setSpacing(0)

        self.product_center_scroll = QtWidgets.QScrollArea()
        self.product_center_scroll.setWidgetResizable(True)
        self.product_center_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.product_center_scroll.setStyleSheet("QScrollArea { border: none; background: #ffffff; }")
        self.product_center_body = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self.product_center_body)
        body.setContentsMargins(18, 18, 18, 18)
        body.setSpacing(16)

        hero_row = QtWidgets.QHBoxLayout()
        hero_row.setSpacing(16)

        self.lbl_product_center_hero = QtWidgets.QLabel()
        self.lbl_product_center_hero.setMinimumSize(320, 220)
        self.lbl_product_center_hero.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_product_center_hero.setStyleSheet(
            """
            QLabel {
                background: #f8fafc;
                color: #0f172a;
                border: 1px solid #dbe2ea;
                border-radius: 14px;
                font-size: 16px;
                font-weight: 700;
                padding: 12px;
            }
            """
        )
        hero_row.addWidget(self.lbl_product_center_hero, 0)

        summary_col = QtWidgets.QVBoxLayout()
        summary_col.setSpacing(8)
        self.lbl_product_center_title = QtWidgets.QLabel("Select a product")
        self.lbl_product_center_title.setStyleSheet("font-size: 22px; font-weight: 800; color: #0f172a;")
        self.lbl_product_center_subtitle = QtWidgets.QLabel("")
        self.lbl_product_center_subtitle.setWordWrap(True)
        self.lbl_product_center_subtitle.setStyleSheet("font-size: 12px; color: #475569;")
        self.lbl_product_center_stats = QtWidgets.QLabel("")
        self.lbl_product_center_stats.setWordWrap(True)
        self.lbl_product_center_stats.setStyleSheet("font-size: 12px; color: #0f172a;")
        self.lbl_product_center_parts = QtWidgets.QLabel("")
        self.lbl_product_center_parts.setWordWrap(True)
        self.lbl_product_center_parts.setStyleSheet("font-size: 12px; color: #334155;")
        self.lbl_product_center_atps = QtWidgets.QLabel("")
        self.lbl_product_center_atps.setWordWrap(True)
        self.lbl_product_center_atps.setStyleSheet("font-size: 12px; color: #334155;")
        self.lbl_product_center_serials = QtWidgets.QLabel("")
        self.lbl_product_center_serials.setWordWrap(True)
        self.lbl_product_center_serials.setStyleSheet("font-size: 12px; color: #334155;")
        self.lbl_product_center_image_debug = QtWidgets.QLabel("")
        self.lbl_product_center_image_debug.setWordWrap(True)
        self.lbl_product_center_image_debug.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_product_center_image_debug.setStyleSheet(
            "font-size: 11px; color: #475569; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px;"
        )
        summary_col.addWidget(self.lbl_product_center_title)
        summary_col.addWidget(self.lbl_product_center_subtitle)
        summary_col.addWidget(self.lbl_product_center_stats)
        summary_col.addWidget(self.lbl_product_center_parts)
        summary_col.addWidget(self.lbl_product_center_atps)
        summary_col.addWidget(self.lbl_product_center_serials)
        summary_col.addWidget(self.lbl_product_center_image_debug)
        summary_col.addStretch(1)
        hero_row.addLayout(summary_col, 1)
        body.addLayout(hero_row)

        docs_hdr = QtWidgets.QHBoxLayout()
        docs_title = QtWidgets.QLabel("Documents by Type")
        docs_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #0f172a;")
        self.lbl_product_center_doc_hint = QtWidgets.QLabel("")
        self.lbl_product_center_doc_hint.setStyleSheet("font-size: 11px; color: #64748b;")
        docs_hdr.addWidget(docs_title)
        docs_hdr.addStretch(1)
        docs_hdr.addWidget(self.lbl_product_center_doc_hint)
        body.addLayout(docs_hdr)

        self.tree_product_center_docs = QtWidgets.QTreeWidget()
        self.tree_product_center_docs.setColumnCount(5)
        self.tree_product_center_docs.setHeaderLabels(["Document", "Serial", "Part #", "ATP #", "Program"])
        self.tree_product_center_docs.setRootIsDecorated(True)
        self.tree_product_center_docs.setAlternatingRowColors(True)
        self.tree_product_center_docs.setStyleSheet(
            """
            QTreeWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                color: #1f2937;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 4px;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #1f2937;
                padding: 6px 8px;
                border: 1px solid #e5e7eb;
                font-weight: 600;
            }
            """
        )
        self.tree_product_center_docs.itemSelectionChanged.connect(self._update_product_center_doc_actions)
        body.addWidget(self.tree_product_center_docs)

        doc_actions = QtWidgets.QHBoxLayout()
        doc_actions.addStretch(1)
        self.btn_product_doc_open_file = QtWidgets.QPushButton("Open File")
        self.btn_product_doc_open_metadata = QtWidgets.QPushButton("Open Metadata")
        self.btn_product_doc_open_combined = QtWidgets.QPushButton("Open combined.txt")
        self.btn_product_doc_show_files = QtWidgets.QPushButton("Show in Files")
        for btn in (
            self.btn_product_doc_open_file,
            self.btn_product_doc_open_metadata,
            self.btn_product_doc_open_combined,
            self.btn_product_doc_show_files,
        ):
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 12px;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
                """
            )
        self.btn_product_doc_open_file.clicked.connect(self._act_product_center_open_doc_file)
        self.btn_product_doc_open_metadata.clicked.connect(self._act_product_center_open_doc_metadata)
        self.btn_product_doc_open_combined.clicked.connect(self._act_product_center_open_doc_combined)
        self.btn_product_doc_show_files.clicked.connect(self._act_product_center_show_files)
        doc_actions.addWidget(self.btn_product_doc_open_file)
        doc_actions.addWidget(self.btn_product_doc_open_metadata)
        doc_actions.addWidget(self.btn_product_doc_open_combined)
        doc_actions.addWidget(self.btn_product_doc_show_files)
        body.addLayout(doc_actions)

        proj_hdr = QtWidgets.QHBoxLayout()
        proj_title = QtWidgets.QLabel("Projects")
        proj_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #0f172a;")
        self.lbl_product_center_project_hint = QtWidgets.QLabel("")
        self.lbl_product_center_project_hint.setStyleSheet("font-size: 11px; color: #64748b;")
        proj_hdr.addWidget(proj_title)
        proj_hdr.addStretch(1)
        proj_hdr.addWidget(self.lbl_product_center_project_hint)
        body.addLayout(proj_hdr)

        self.tbl_product_center_projects = QtWidgets.QTableWidget(0, 4)
        self.tbl_product_center_projects.setHorizontalHeaderLabels(["Name", "Type", "Folder", "Workbook"])
        self.tbl_product_center_projects.verticalHeader().setVisible(False)
        self.tbl_product_center_projects.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_product_center_projects.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_product_center_projects.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_product_center_projects.horizontalHeader().setStretchLastSection(True)
        self.tbl_product_center_projects.itemSelectionChanged.connect(self._update_product_center_doc_actions)
        self.tbl_product_center_projects.setStyleSheet(
            """
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                color: #1f2937;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #1f2937;
                padding: 6px 8px;
                border: 1px solid #e5e7eb;
                font-weight: 600;
            }
            """
        )
        body.addWidget(self.tbl_product_center_projects)

        project_actions = QtWidgets.QHBoxLayout()
        project_actions.addStretch(1)
        self.btn_product_project_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_product_project_open_workbook = QtWidgets.QPushButton("Open Workbook")
        self.btn_product_project_show_projects = QtWidgets.QPushButton("Show in Projects")
        for btn in (
            self.btn_product_project_open_folder,
            self.btn_product_project_open_workbook,
            self.btn_product_project_show_projects,
        ):
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 12px;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 12px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
                """
            )
        self.btn_product_project_open_folder.clicked.connect(self._act_product_center_open_project_folder)
        self.btn_product_project_open_workbook.clicked.connect(self._act_product_center_open_project_workbook)
        self.btn_product_project_show_projects.clicked.connect(self._act_product_center_show_projects)
        project_actions.addWidget(self.btn_product_project_open_folder)
        project_actions.addWidget(self.btn_product_project_open_workbook)
        project_actions.addWidget(self.btn_product_project_show_projects)
        body.addLayout(project_actions)

        eq_hdr = QtWidgets.QHBoxLayout()
        eq_title = QtWidgets.QLabel("Saved Performance Equations")
        eq_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #0f172a;")
        self.lbl_product_center_eq_hint = QtWidgets.QLabel("")
        self.lbl_product_center_eq_hint.setStyleSheet("font-size: 11px; color: #64748b;")
        eq_hdr.addWidget(eq_title)
        eq_hdr.addStretch(1)
        eq_hdr.addWidget(self.lbl_product_center_eq_hint)
        body.addLayout(eq_hdr)

        self.tbl_product_center_equations = QtWidgets.QTableWidget(0, 5)
        self.tbl_product_center_equations.setHorizontalHeaderLabels(["Project", "Equation", "Summary", "Saved", "Updated"])
        self.tbl_product_center_equations.verticalHeader().setVisible(False)
        self.tbl_product_center_equations.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_product_center_equations.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_product_center_equations.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_product_center_equations.horizontalHeader().setStretchLastSection(True)
        self.tbl_product_center_equations.setStyleSheet(
            """
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                color: #1f2937;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #1f2937;
                padding: 6px 8px;
                border: 1px solid #e5e7eb;
                font-weight: 600;
            }
            """
        )
        body.addWidget(self.tbl_product_center_equations)

        self.product_center_scroll.setWidget(self.product_center_body)
        r.addWidget(self.product_center_scroll, 1)

        root.addWidget(left)
        root.addWidget(right, 1)

        self._set_product_center_empty_state("Select a Global Repo in Setup to load Product Center.")
        self._update_product_center_doc_actions()

    def _setup_tab_projects(self):
        root = QtWidgets.QHBoxLayout(self.tab_projects)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(14)

        # Left pane: primary actions
        left = QtWidgets.QFrame()
        left.setFixedWidth(260)
        left.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            """
        )
        l = QtWidgets.QVBoxLayout(left)
        l.setContentsMargins(14, 14, 14, 14)
        l.setSpacing(10)

        lbl = QtWidgets.QLabel("Project Builder")
        lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        desc = QtWidgets.QLabel("Create projects from indexed EIDPs or TD reports.\n(EIDP Trending / Test Data Trending)")
        desc.setStyleSheet("font-size: 11px; color: #0f172a;")
        desc.setWordWrap(True)
        l.addWidget(lbl)
        l.addWidget(desc)

        self.btn_project_new = QtWidgets.QPushButton("Create New Project")
        self.btn_project_new.setStyleSheet(
            """
            QPushButton {
                padding: 10px 14px;
                border-radius: 8px;
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton:disabled { background: #93c5fd; border-color: #93c5fd; }
            """
        )
        self.btn_project_new.clicked.connect(self._act_new_project)

        self.btn_project_refresh = QtWidgets.QPushButton("Refresh List")
        self.btn_project_refresh.setStyleSheet(
            """
            QPushButton {
                padding: 10px 14px;
                border-radius: 8px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f9fafb; }
            """
        )
        self.btn_project_refresh.clicked.connect(self._refresh_projects)

        l.addWidget(self.btn_project_new)
        l.addWidget(self.btn_project_refresh)

        impl_divider = QtWidgets.QFrame()
        impl_divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        impl_divider.setStyleSheet("background-color: #e5e7eb; margin: 8px 0;")
        l.addWidget(impl_divider)

        impl_lbl = QtWidgets.QLabel("Implementation")
        impl_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #0f172a;")
        impl_desc = QtWidgets.QLabel("Trend / analyze data for a selected EIDP Trending project.")
        impl_desc.setStyleSheet("font-size: 11px; color: #475569;")
        impl_desc.setWordWrap(True)
        l.addWidget(impl_lbl)
        l.addWidget(impl_desc)

        self.btn_project_implementation = QtWidgets.QPushButton("Trend / Analyze Data")
        self.btn_project_implementation.setEnabled(False)
        self.btn_project_implementation.setStyleSheet(
            """
            QPushButton {
                padding: 10px 14px;
                border-radius: 8px;
                background: #0f766e;
                color: #ffffff;
                border: 1px solid #0f766e;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background: #0d9488; }
            QPushButton:disabled { background: #94a3b8; border-color: #94a3b8; }
            """
        )
        self.btn_project_implementation.clicked.connect(self._act_open_project_implementation)
        l.addWidget(self.btn_project_implementation)
        l.addStretch(1)

        self.lbl_projects_hint = QtWidgets.QLabel("Tip: run 'Index' first to populate metadata.")
        self.lbl_projects_hint.setStyleSheet("font-size: 11px; color: #0f172a;")
        self.lbl_projects_hint.setWordWrap(True)
        l.addWidget(self.lbl_projects_hint)

        # Right pane: project list
        right = QtWidgets.QFrame()
        right.setStyleSheet(
            """
            QFrame {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            """
        )
        r = QtWidgets.QVBoxLayout(right)
        r.setContentsMargins(14, 14, 14, 14)
        r.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Projects")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        self.lbl_projects_status = QtWidgets.QLabel("")
        self.lbl_projects_status.setStyleSheet("font-size: 11px; color: #0f172a;")
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(self.lbl_projects_status)
        r.addLayout(header_row)

        projects_filter_row = QtWidgets.QHBoxLayout()
        self.ed_projects_filter = QtWidgets.QLineEdit()
        self.ed_projects_filter.setPlaceholderText("Filter projects by name, type, folder, workbook ...")
        self.ed_projects_filter.setStyleSheet(
            """
            QLineEdit {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
            }
            """
        )
        self.ed_projects_filter.textChanged.connect(self._apply_projects_filter)
        projects_filter_row.addWidget(self.ed_projects_filter, 1)
        r.addLayout(projects_filter_row)

        projects_subset_row = QtWidgets.QHBoxLayout()
        self.lbl_projects_subset = QtWidgets.QLabel("")
        self.lbl_projects_subset.setWordWrap(True)
        self.lbl_projects_subset.setStyleSheet(
            "font-size: 11px; color: #1e3a8a; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 6px 8px;"
        )
        self.btn_projects_clear_subset = QtWidgets.QPushButton("Clear")
        self.btn_projects_clear_subset.setStyleSheet(
            """
            QPushButton {
                padding: 6px 10px;
                border-radius: 6px;
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #f9fafb; }
            """
        )
        self.btn_projects_clear_subset.clicked.connect(self._clear_projects_external_subset)
        projects_subset_row.addWidget(self.lbl_projects_subset, 1)
        projects_subset_row.addWidget(self.btn_projects_clear_subset)
        r.addLayout(projects_subset_row)

        cols = ["Name", "Type", "Folder", "Workbook"]
        self.tbl_projects = QtWidgets.QTableWidget(0, len(cols))
        self.tbl_projects.setHorizontalHeaderLabels(cols)
        self.tbl_projects.verticalHeader().setVisible(False)
        self.tbl_projects.setAlternatingRowColors(True)
        self.tbl_projects.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_projects.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_projects.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_projects.horizontalHeader().setStretchLastSection(True)
        self.tbl_projects.setStyleSheet(
            """
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f9fafb;
                color: #1f2937;
                gridline-color: #e5e7eb;
                selection-background-color: #dbeafe;
                selection-color: #1f2937;
            }
            QTableWidget::item {
                color: #1f2937;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #1f2937;
                padding: 6px 8px;
                border: 1px solid #e5e7eb;
                font-weight: 600;
            }
            """
        )
        self.tbl_projects.itemSelectionChanged.connect(self._update_project_actions)
        self.tbl_projects.doubleClicked.connect(self._act_open_project_workbook)
        r.addWidget(self.tbl_projects, 1)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.cb_project_overwrite = QtWidgets.QCheckBox("Overwrite existing cells")
        self.cb_project_overwrite.setChecked(False)
        self.cb_project_overwrite.setStyleSheet("color:#0f172a; font-size: 12px;")
        self.btn_project_update = QtWidgets.QPushButton("Update Project")
        self.btn_project_perf_sheets = QtWidgets.QPushButton("Generate Performance Sheets")
        self.btn_project_debug_excels = QtWidgets.QPushButton("Generate Debug Excel Files")
        self.btn_project_env = QtWidgets.QPushButton("Project Env")
        self.btn_project_delete = QtWidgets.QPushButton("Delete Project")
        self.btn_project_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_project_open_workbook = QtWidgets.QPushButton("Open Workbook")
        self.btn_project_open_support = QtWidgets.QPushButton("Open Support Workbook")
        for b in (
            self.btn_project_update,
            self.btn_project_perf_sheets,
            self.btn_project_debug_excels,
            self.btn_project_env,
            self.btn_project_delete,
            self.btn_project_open_folder,
            self.btn_project_open_workbook,
            self.btn_project_open_support,
        ):
            b.setStyleSheet(
                """
                QPushButton {
                    padding: 10px 14px;
                    border-radius: 8px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover { background: #f9fafb; }
                """
        )
        self.btn_project_update.clicked.connect(self._act_update_project)
        self.btn_project_perf_sheets.clicked.connect(self._act_generate_project_performance_sheets)
        self.btn_project_debug_excels.clicked.connect(self._act_generate_project_debug_excels)
        self.btn_project_env.clicked.connect(self._act_project_env)
        self.btn_project_delete.clicked.connect(self._act_delete_project)
        self.btn_project_open_folder.clicked.connect(self._act_open_project_folder)
        self.btn_project_open_workbook.clicked.connect(self._act_open_project_workbook)
        self.btn_project_open_support.clicked.connect(self._act_open_project_support_workbook)
        actions.addWidget(self.cb_project_overwrite)
        actions.addWidget(self.btn_project_update)
        actions.addWidget(self.btn_project_perf_sheets)
        actions.addWidget(self.btn_project_debug_excels)
        actions.addWidget(self.btn_project_env)
        actions.addWidget(self.btn_project_delete)
        actions.addWidget(self.btn_project_open_folder)
        actions.addWidget(self.btn_project_open_workbook)
        actions.addWidget(self.btn_project_open_support)
        r.addLayout(actions)

        root.addWidget(left)
        root.addWidget(right, 1)

        self._refresh_projects()
        self._update_projects_external_filter_banner()

    def _product_center_placeholder_pixmap(self, text: str, *, width: int, height: int) -> QtGui.QPixmap:
        pix = QtGui.QPixmap(width, height)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(1, 1, width - 2, height - 2)
        gradient = QtGui.QLinearGradient(0, 0, width, height)
        gradient.setColorAt(0, QtGui.QColor("#eff6ff"))
        gradient.setColorAt(1, QtGui.QColor("#dbeafe"))
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtGui.QPen(QtGui.QColor("#93c5fd"), 1.5))
        painter.drawRoundedRect(rect, 16, 16)

        font = painter.font()
        font.setBold(True)
        font.setPointSize(max(10, min(20, width // 14)))
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#1e3a8a"))
        painter.drawText(
            QtCore.QRectF(18, 18, width - 36, height - 36),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            str(text or "").strip() or "No Image",
        )
        painter.end()
        return pix

    def _product_center_image_status_text(self, product: dict, *, image_path: str = "", decode_ok: bool = False) -> str:
        asset_specific = str(product.get("asset_specific_type") or product.get("display_name") or "").strip()
        try:
            details = be.product_center_image_lookup_details(asset_specific)
        except Exception:
            details = {
                "asset_specific_type": asset_specific,
                "search_dir": "",
                "expected_files": [],
                "resolved_path": str(image_path or "").strip(),
                "resolved_exists": bool(image_path),
            }
        expected = ", ".join(str(x or "").strip() for x in (details.get("expected_files") or []) if str(x or "").strip())
        resolved_path = str(image_path or details.get("resolved_path") or "").strip()
        search_dir = str(details.get("search_dir") or "").strip()
        if resolved_path and decode_ok:
            status = f"Image loaded from: {resolved_path}"
        elif resolved_path:
            status = f"Image file found but Qt could not decode it: {resolved_path}"
        else:
            status = "No matching image file found."
        lines = [
            f"Asset Specific Type: {asset_specific or '(blank)'}",
            status,
        ]
        if expected:
            lines.append(f"Expected names: {expected}")
        if search_dir:
            lines.append(f"Search folder: {search_dir}")
        return "\n".join(lines)

    def _product_center_pixmap(self, product: dict, *, width: int, height: int) -> tuple[QtGui.QPixmap, str]:
        image_path = str(product.get("image_path") or "").strip()
        if image_path:
            try:
                pix = QtGui.QPixmap(image_path)
                if not pix.isNull():
                    return (
                        pix.scaled(
                            width,
                            height,
                            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            QtCore.Qt.TransformationMode.SmoothTransformation,
                        ),
                        self._product_center_image_status_text(product, image_path=image_path, decode_ok=True),
                    )
            except Exception:
                pass
        return (
            self._product_center_placeholder_pixmap(
                str(product.get("asset_specific_type") or product.get("display_name") or "No Image"),
                width=width,
                height=height,
            ),
            self._product_center_image_status_text(product, image_path=image_path, decode_ok=False),
        )

    def _product_center_search_blob(self, product: dict) -> str:
        parts: list[str] = [
            str(product.get("asset_type") or ""),
            str(product.get("asset_specific_type") or ""),
            str(product.get("vendor") or ""),
        ]
        for key in ("part_numbers", "acceptance_test_plan_numbers", "serial_numbers", "document_types"):
            for value in (product.get(key) or []):
                parts.append(str(value or ""))
        for doc in (product.get("documents") or []):
            if not isinstance(doc, dict):
                continue
            for key in (
                "document_type",
                "document_type_acronym",
                "serial_number",
                "part_number",
                "program_title",
                "metadata_rel",
                "metadata_source",
                "applied_asset_specific_type_rule",
            ):
                parts.append(str(doc.get(key) or ""))
            for value in (doc.get("manual_override_fields") or []):
                parts.append(str(value or ""))
        for project in (product.get("projects") or []):
            if not isinstance(project, dict):
                continue
            parts.append(str(project.get("name") or ""))
            parts.append(str(project.get("type") or ""))
        return "\n".join(part for part in parts if part).lower()

    def _set_product_center_empty_state(self, message: str) -> None:
        self._product_center_current = None
        self.lbl_product_center_status.setText(str(message or ""))
        self.lbl_product_center_title.setText("Product Center")
        self.lbl_product_center_subtitle.setText(str(message or ""))
        self.lbl_product_center_stats.setText("")
        self.lbl_product_center_parts.setText("")
        self.lbl_product_center_atps.setText("")
        self.lbl_product_center_serials.setText("")
        self.lbl_product_center_image_debug.setText("")
        self.lbl_product_center_doc_hint.setText("")
        self.lbl_product_center_project_hint.setText("")
        self.lbl_product_center_eq_hint.setText("")
        pix, debug_text = self._product_center_pixmap({}, width=320, height=220)
        self.lbl_product_center_hero.setPixmap(pix)
        self.lbl_product_center_image_debug.setText(debug_text)
        self.tree_product_center_docs.clear()
        self.tbl_product_center_projects.setRowCount(0)
        self.tbl_product_center_equations.setRowCount(0)
        self._update_product_center_doc_actions()

    def _refresh_product_center(self) -> None:
        repo_raw = (self.ed_global_repo.text() or "").strip() if hasattr(self, "ed_global_repo") else ""
        self.list_product_center.clear()
        self._product_center_products = []
        self._product_center_filtered = []
        if not repo_raw:
            self._set_product_center_empty_state("Select a Global Repo in Setup to load Product Center.")
            return
        try:
            products = be.list_product_center_products(Path(repo_raw))
        except FileNotFoundError:
            self._set_product_center_empty_state("No indexed products found. Run scan/index first so Product Center has source data.")
            return
        except Exception as exc:
            self._set_product_center_empty_state(f"Unable to load Product Center: {exc}")
            return
        self._product_center_products = [dict(item) for item in products if isinstance(item, dict)]
        if not self._product_center_products:
            self._set_product_center_empty_state("No indexed products found. Run scan/index first so Product Center has source data.")
            return
        self._apply_product_center_filter()

    def _apply_product_center_filter(self) -> None:
        products = list(getattr(self, "_product_center_products", []) or [])
        text = (self.ed_product_center_search.text() or "").strip().lower() if hasattr(self, "ed_product_center_search") else ""
        if text:
            products = [item for item in products if text in self._product_center_search_blob(item)]
        self._product_center_filtered = products
        self.list_product_center.clear()
        for product in products:
            counts = product.get("counts") if isinstance(product.get("counts"), dict) else {}
            subtitle = f"{product.get('asset_type') or ''} | {product.get('vendor') or ''}"
            summary = (
                f"{int(counts.get('documents') or 0)} docs | "
                f"{int(counts.get('projects') or 0)} projects | "
                f"{int(counts.get('saved_performance_equations') or 0)} saved equations"
            )
            item = QtWidgets.QListWidgetItem(
                QtGui.QIcon(self._product_center_pixmap(product, width=72, height=72)[0]),
                f"{str(product.get('display_name') or product.get('asset_specific_type') or '')}\n{subtitle}\n{summary}",
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(product))
            self.list_product_center.addItem(item)
        total = len(getattr(self, "_product_center_products", []) or [])
        shown = len(products)
        self.lbl_product_center_status.setText(f"{shown} of {total} product(s)" if shown != total else f"{shown} product(s)")
        if products:
            self.list_product_center.setCurrentRow(0)
        else:
            self._set_product_center_empty_state("No products match the current Product Center filter.")

    def _on_product_center_selection_changed(self, _row: int) -> None:
        item = self.list_product_center.currentItem()
        product = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(product, dict):
            self._set_product_center_empty_state("Select a product to see documents, projects, and saved equations.")
            return
        self._product_center_current = dict(product)
        counts = product.get("counts") if isinstance(product.get("counts"), dict) else {}
        self.lbl_product_center_title.setText(str(product.get("asset_specific_type") or ""))
        self.lbl_product_center_subtitle.setText(
            f"Asset Type: {str(product.get('asset_type') or '')}\nVendor: {str(product.get('vendor') or '')}"
        )
        self.lbl_product_center_stats.setText(
            "Documents: {documents} | EIDP: {eidp} | TD: {td} | Projects: {projects} | Saved Equations: {equations}".format(
                documents=int(counts.get("documents") or 0),
                eidp=int(counts.get("eidp_documents") or 0),
                td=int(counts.get("td_documents") or 0),
                projects=int(counts.get("projects") or 0),
                equations=int(counts.get("saved_performance_equations") or 0),
            )
        )
        self.lbl_product_center_parts.setText(
            "Part Numbers: " + (", ".join(str(value) for value in (product.get("part_numbers") or [])) or "None")
        )
        self.lbl_product_center_atps.setText(
            "Acceptance Test Plans: "
            + (", ".join(str(value) for value in (product.get("acceptance_test_plan_numbers") or [])) or "None")
        )
        self.lbl_product_center_serials.setText(
            "Serial Numbers: " + (", ".join(str(value) for value in (product.get("serial_numbers") or [])) or "None")
        )
        hero_pix, image_debug = self._product_center_pixmap(product, width=320, height=220)
        self.lbl_product_center_hero.setPixmap(hero_pix)
        self.lbl_product_center_image_debug.setText(image_debug)
        self._populate_product_center_documents(product)
        self._populate_product_center_projects(product)
        self._populate_product_center_equations(product)
        self._update_product_center_doc_actions()

    def _populate_product_center_documents(self, product: dict) -> None:
        self.tree_product_center_docs.clear()
        groups: dict[str, list[dict]] = {}
        for doc in (product.get("documents") or []):
            if not isinstance(doc, dict):
                continue
            groups.setdefault(str(doc.get("display_document_type") or "(No Doc Type)"), []).append(doc)
        for doc_type in sorted(groups.keys(), key=str.casefold):
            docs = groups[doc_type]
            parent = QtWidgets.QTreeWidgetItem([f"{doc_type} ({len(docs)})"])
            parent.setFirstColumnSpanned(True)
            parent.setFlags(parent.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.tree_product_center_docs.addTopLevelItem(parent)
            for doc in docs:
                child = QtWidgets.QTreeWidgetItem(
                    [
                        str(doc.get("metadata_rel") or doc.get("rel_path") or "(Document)"),
                        str(doc.get("serial_number") or ""),
                        str(doc.get("part_number") or ""),
                        str(doc.get("acceptance_test_plan_number") or ""),
                        str(doc.get("program_title") or ""),
                    ]
                )
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, dict(doc))
                parent.addChild(child)
        self.tree_product_center_docs.expandAll()
        self.tree_product_center_docs.resizeColumnToContents(0)
        self.lbl_product_center_doc_hint.setText(
            f"{len(product.get('documents') or [])} document(s) linked to this product"
        )

    def _populate_product_center_projects(self, product: dict) -> None:
        projects = [dict(item) for item in (product.get("projects") or []) if isinstance(item, dict)]
        self.tbl_product_center_projects.setRowCount(0)
        for row, project in enumerate(projects):
            self.tbl_product_center_projects.insertRow(row)
            item_name = QtWidgets.QTableWidgetItem(str(project.get("name") or ""))
            item_type = QtWidgets.QTableWidgetItem(str(project.get("type") or ""))
            item_folder = QtWidgets.QTableWidgetItem(str(project.get("project_dir") or ""))
            item_workbook = QtWidgets.QTableWidgetItem(str(project.get("workbook") or ""))
            item_name.setData(QtCore.Qt.ItemDataRole.UserRole, dict(project))
            self.tbl_product_center_projects.setItem(row, 0, item_name)
            self.tbl_product_center_projects.setItem(row, 1, item_type)
            self.tbl_product_center_projects.setItem(row, 2, item_folder)
            self.tbl_product_center_projects.setItem(row, 3, item_workbook)
        self.tbl_product_center_projects.resizeColumnsToContents()
        self.tbl_product_center_projects.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tbl_product_center_projects.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.lbl_product_center_project_hint.setText(f"{len(projects)} linked project(s)")

    def _populate_product_center_equations(self, product: dict) -> None:
        equations = [dict(item) for item in (product.get("saved_performance_equations") or []) if isinstance(item, dict)]
        self.tbl_product_center_equations.setRowCount(0)
        for row, entry in enumerate(equations):
            self.tbl_product_center_equations.insertRow(row)
            values = [
                str(entry.get("project_name") or ""),
                str(entry.get("name") or ""),
                str(entry.get("summary") or ""),
                str(entry.get("saved_at") or ""),
                str(entry.get("updated_at") or ""),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.tbl_product_center_equations.setItem(row, col, item)
        self.tbl_product_center_equations.resizeColumnsToContents()
        self.tbl_product_center_equations.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.lbl_product_center_eq_hint.setText(f"{len(equations)} saved equation(s)")

    def _selected_product_center_document(self) -> dict | None:
        item = self.tree_product_center_docs.currentItem()
        if item is None:
            return None
        doc = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return dict(doc) if isinstance(doc, dict) else None

    def _selected_product_center_project(self) -> dict | None:
        row = self.tbl_product_center_projects.currentRow()
        if row < 0:
            return None
        item = self.tbl_product_center_projects.item(row, 0)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return dict(payload) if isinstance(payload, dict) else None

    def _update_product_center_doc_actions(self) -> None:
        doc = self._selected_product_center_document()
        current = self._product_center_current if isinstance(self._product_center_current, dict) else {}
        has_files_target = bool(
            [
                str(item.get("rel_path") or "").strip()
                for item in (current.get("documents") or [])
                if isinstance(item, dict) and str(item.get("rel_path") or "").strip()
            ]
        )
        for btn in (
            self.btn_product_doc_open_file,
            self.btn_product_doc_open_metadata,
            self.btn_product_doc_open_combined,
        ):
            btn.setEnabled(bool(doc))
        self.btn_product_doc_show_files.setEnabled(has_files_target)
        project = self._selected_product_center_project()
        self.btn_product_project_open_folder.setEnabled(bool(project))
        self.btn_product_project_open_workbook.setEnabled(bool(project))
        self.btn_product_project_show_projects.setEnabled(bool(current.get("projects")))

    def _act_product_center_open_image_folder(self) -> None:
        try:
            be.open_path(be.product_center_images_dir())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Product Center", str(exc))

    def _act_product_center_open_doc_file(self) -> None:
        doc = self._selected_product_center_document()
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not doc or not repo_raw:
            return
        rel_path = str(doc.get("rel_path") or "").strip()
        if not rel_path:
            QtWidgets.QMessageBox.warning(self, "Product Center", "No tracked source file is linked to this document.")
            return
        try:
            be.open_path(Path(repo_raw).expanduser() / rel_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open File", str(exc))

    def _act_product_center_open_doc_metadata(self) -> None:
        doc = self._selected_product_center_document()
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not doc or not repo_raw:
            return
        metadata_rel = str(doc.get("metadata_rel") or "").strip()
        if not metadata_rel:
            QtWidgets.QMessageBox.warning(self, "Product Center", "No metadata JSON is linked to this document.")
            return
        try:
            be.open_path(be.eidat_support_dir(Path(repo_raw).expanduser()) / metadata_rel)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Metadata", str(exc))

    def _act_product_center_open_doc_combined(self) -> None:
        doc = self._selected_product_center_document()
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not doc or not repo_raw:
            return
        support_dir = be.eidat_support_dir(Path(repo_raw).expanduser())
        combined = None
        artifacts_rel = str(doc.get("artifacts_rel") or "").strip()
        if artifacts_rel:
            combined = support_dir / artifacts_rel / "combined.txt"
        if combined is None or not combined.exists():
            rel_path = str(doc.get("rel_path") or "").strip()
            if rel_path:
                artifacts = be.get_file_artifacts_path(Path(repo_raw).expanduser(), rel_path)
                if artifacts is not None:
                    combined = artifacts / "combined.txt"
        if combined is None or not combined.exists():
            QtWidgets.QMessageBox.warning(self, "Product Center", "combined.txt was not found for this document.")
            return
        try:
            be.open_path(combined)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open combined.txt", str(exc))

    def _act_product_center_show_files(self) -> None:
        current = self._product_center_current if isinstance(self._product_center_current, dict) else {}
        rel_paths = [
            str(item.get("rel_path") or "").strip()
            for item in (current.get("documents") or [])
            if isinstance(item, dict) and str(item.get("rel_path") or "").strip()
        ]
        if not rel_paths:
            QtWidgets.QMessageBox.warning(self, "Product Center", "No tracked files are linked to this product.")
            return
        label = f"Filtered from Product Center: {str(current.get('asset_specific_type') or '').strip()}"
        self._set_files_external_subset(rel_paths, label=label)
        self._switch_tab(1)

    def _act_product_center_open_project_folder(self) -> None:
        project = self._selected_product_center_project()
        if not project:
            return
        try:
            be.open_path(Path(str(project.get("project_dir") or "")).expanduser())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Folder", str(exc))

    def _act_product_center_open_project_workbook(self) -> None:
        project = self._selected_product_center_project()
        if not project:
            return
        try:
            be.open_path(Path(str(project.get("workbook") or "")).expanduser())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Workbook", str(exc))

    def _act_product_center_show_projects(self) -> None:
        current = self._product_center_current if isinstance(self._product_center_current, dict) else {}
        project_dirs = [
            str(item.get("project_dir") or "").strip()
            for item in (current.get("projects") or [])
            if isinstance(item, dict) and str(item.get("project_dir") or "").strip()
        ]
        if not project_dirs:
            QtWidgets.QMessageBox.warning(self, "Product Center", "No linked projects are available for this product.")
            return
        label = f"Filtered from Product Center: {str(current.get('asset_specific_type') or '').strip()}"
        self._set_projects_external_subset(project_dirs, label=label)
        self._switch_tab(3)

    def _refresh_projects(self) -> None:
        tbl = getattr(self, "tbl_projects", None)
        if not tbl:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip() if hasattr(self, "ed_global_repo") else ""
        self.btn_project_new.setEnabled(bool(repo_raw))
        self._projects_all = []
        tbl.setRowCount(0)
        if not repo_raw:
            self.lbl_projects_status.setText("Select a Global Repo in Setup")
            self._update_projects_external_filter_banner()
            self._update_project_actions()
            return

        repo = Path(repo_raw).expanduser()
        try:
            projects = be.list_eidat_projects(repo)
        except Exception as exc:
            self.lbl_projects_status.setText("Unable to read projects")
            self._append_log(f"[PROJECTS] Failed to load registry: {exc}")
            return

        records: list[dict] = []
        for p in projects:
            name = str(p.get("name") or "").strip()
            ptype = str(p.get("type") or "").strip()
            rel_folder = str(p.get("project_dir") or "").strip()
            rel_workbook = str(p.get("workbook") or "").strip()
            folder_abs = (repo / rel_folder).expanduser() if rel_folder and not Path(rel_folder).is_absolute() else Path(rel_folder).expanduser()
            workbook_abs = (repo / rel_workbook).expanduser() if rel_workbook and not Path(rel_workbook).is_absolute() else Path(rel_workbook).expanduser()
            records.append(
                {
                    "name": name,
                    "type": ptype,
                    "project_dir": str(folder_abs),
                    "project_dir_display": rel_folder,
                    "workbook": str(workbook_abs),
                    "workbook_display": rel_workbook,
                }
            )

        self._projects_all = records
        self._apply_projects_filter()

    def _normalize_project_dir_key(self, value: object) -> str:
        try:
            return str(Path(str(value or "")).expanduser().resolve()).casefold()
        except Exception:
            return str(Path(str(value or "")).expanduser().absolute()).casefold()

    def _update_projects_external_filter_banner(self) -> None:
        active = bool(getattr(self, "_projects_external_dirs", None))
        label = getattr(self, "_projects_external_filter_label", "").strip()
        if hasattr(self, "lbl_projects_subset"):
            self.lbl_projects_subset.setVisible(active)
            self.lbl_projects_subset.setText(label if active else "")
        if hasattr(self, "btn_projects_clear_subset"):
            self.btn_projects_clear_subset.setVisible(active)

    def _set_projects_external_subset(self, project_dirs: list[str], *, label: str) -> None:
        cleaned = {
            self._normalize_project_dir_key(value)
            for value in (project_dirs or [])
            if str(value).strip()
        }
        self._projects_external_dirs = cleaned or None
        self._projects_external_filter_label = str(label or "").strip()
        self._update_projects_external_filter_banner()
        self._apply_projects_filter()

    def _clear_projects_external_subset(self) -> None:
        self._projects_external_dirs = None
        self._projects_external_filter_label = ""
        self._update_projects_external_filter_banner()
        self._apply_projects_filter()

    def _apply_projects_filter(self) -> None:
        tbl = getattr(self, "tbl_projects", None)
        if not tbl:
            return
        records = list(getattr(self, "_projects_all", []) or [])
        external_dirs = getattr(self, "_projects_external_dirs", None)
        if external_dirs is not None:
            records = [
                item for item in records
                if self._normalize_project_dir_key(item.get("project_dir")) in external_dirs
            ]
        search = (self.ed_projects_filter.text() or "").strip().lower() if hasattr(self, "ed_projects_filter") else ""
        if search:
            records = [
                item for item in records
                if search in "\n".join(
                    [
                        str(item.get("name") or ""),
                        str(item.get("type") or ""),
                        str(item.get("project_dir_display") or ""),
                        str(item.get("workbook_display") or ""),
                    ]
                ).lower()
            ]
        self._populate_projects_table(records)

    def _populate_projects_table(self, projects: list[dict]) -> None:
        tbl = getattr(self, "tbl_projects", None)
        if not tbl:
            return
        tbl.setRowCount(0)
        for r, p in enumerate(projects):
            tbl.insertRow(r)
            item_name = QtWidgets.QTableWidgetItem(str(p.get("name") or ""))
            item_type = QtWidgets.QTableWidgetItem(str(p.get("type") or ""))
            item_folder = QtWidgets.QTableWidgetItem(str(p.get("project_dir_display") or ""))
            item_workbook = QtWidgets.QTableWidgetItem(str(p.get("workbook_display") or ""))
            item_name.setData(QtCore.Qt.ItemDataRole.UserRole, dict(p))
            item_folder.setData(QtCore.Qt.ItemDataRole.UserRole, str(p.get("project_dir") or ""))
            item_workbook.setData(QtCore.Qt.ItemDataRole.UserRole, str(p.get("workbook") or ""))
            tbl.setItem(r, 0, item_name)
            tbl.setItem(r, 1, item_type)
            tbl.setItem(r, 2, item_folder)
            tbl.setItem(r, 3, item_workbook)
        tbl.resizeColumnsToContents()
        tbl.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        total = len(getattr(self, "_projects_all", []) or [])
        shown = len(projects)
        self.lbl_projects_status.setText(f"{shown} of {total} project(s)" if shown != total else f"{shown} project(s)")
        self._update_project_actions()

    def _act_new_project(self) -> None:
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first (Setup tab).")
            repo = Path(repo_raw).expanduser()
            dlg = NewProjectWizardDialog(repo, self)
            self._prepare_dialog(dlg)
            if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self._refresh_projects()
                try:
                    meta = dlg.project_meta if isinstance(dlg.project_meta, dict) else {}
                    support_val = str(meta.get("support_workbook") or "").strip()
                    project_dir = Path(str(meta.get("project_dir") or "")).expanduser()
                    if support_val and project_dir.exists():
                        support_path = Path(support_val).expanduser()
                        if not support_path.is_absolute():
                            support_path = project_dir / support_path
                        if support_path.exists():
                            be.open_path(support_path)
                except Exception:
                    pass
                self._show_toast("Project created")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "New Project", str(exc))

    def _selected_project_item(self, col: int) -> QtWidgets.QTableWidgetItem | None:
        tbl = getattr(self, "tbl_projects", None)
        if not tbl:
            return None
        row = tbl.currentRow()
        if row < 0:
            return None
        return tbl.item(row, col)

    def _selected_project_record(self) -> dict | None:
        item_name = self._selected_project_item(0)
        item_type = self._selected_project_item(1)
        item_folder = self._selected_project_item(2)
        item_workbook = self._selected_project_item(3)
        if not item_name or not item_type or not item_folder or not item_workbook:
            return None
        return {
            "name": (item_name.text() or "").strip(),
            "type": (item_type.text() or "").strip(),
            "folder": str(item_folder.data(QtCore.Qt.ItemDataRole.UserRole) or item_folder.text() or "").strip(),
            "workbook": str(item_workbook.data(QtCore.Qt.ItemDataRole.UserRole) or item_workbook.text() or "").strip(),
        }

    def _update_project_actions(self) -> None:
        record = self._selected_project_record()
        project_busy = bool(getattr(self, "_project_worker", None) and self._project_worker.isRunning())
        is_impl_supported = False
        is_update_supported = False
        is_td = False
        if record:
            ptype = str(record.get("type") or "").strip()
            is_impl_supported = ptype in (
                getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"),
                getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"),
                getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
            )
            is_update_supported = is_impl_supported
            is_td = ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending")
        if hasattr(self, "btn_project_implementation"):
            self.btn_project_implementation.setEnabled(bool(record) and is_impl_supported)
        if hasattr(self, "btn_project_update"):
            self.btn_project_update.setEnabled(bool(record) and is_update_supported and not project_busy)
        if hasattr(self, "btn_project_perf_sheets"):
            self.btn_project_perf_sheets.setEnabled(bool(record) and is_td and not project_busy)
        if hasattr(self, "btn_project_debug_excels"):
            self.btn_project_debug_excels.setEnabled(bool(record) and is_td and not project_busy)
        if hasattr(self, "btn_project_env"):
            self.btn_project_env.setEnabled(bool(record) and not project_busy)
        if hasattr(self, "btn_project_open_support"):
            self.btn_project_open_support.setEnabled(bool(is_td))

    def _project_task_log_path(self, project_dir: Path, prefix: str) -> Path:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", str(prefix or "project_task")).strip("_") or "project_task"
        return Path(project_dir).expanduser() / "logs" / f"{safe_prefix}_{stamp}.log"

    def _start_project_task(
        self,
        *,
        heading: str,
        status_text: str,
        project_dir: Path,
        log_prefix: str,
        task_factory,
        on_success,
    ) -> None:
        if self._project_worker is not None and self._project_worker.isRunning():
            self._show_toast("Project task already running")
            return
        if self._manager_popup_active or self._sync_popup_active:
            self._show_toast("Another background task is already using the progress popup")
            return

        log_path = self._project_task_log_path(project_dir, log_prefix)
        self._project_popup_active = True
        self._repo_scan_dialog.begin(status_text, heading=heading)
        self._repo_scan_dialog.set_detail_text(f"Logging to {log_path.name}")
        self._append_log(f"[GUI] {heading} started")
        self._append_log(f"[PROJECT TASK] Log: {log_path}")

        self._project_worker = ProjectTaskWorker(task_factory, log_path=log_path, parent=self)
        self._update_project_actions()
        self._project_worker.progress.connect(self._on_project_task_progress)
        self._project_worker.completed.connect(lambda payload: self._on_project_task_done(payload, on_success, heading))
        self._project_worker.failed.connect(lambda message: self._on_project_task_error(message, heading))
        self._project_worker.start()

    def _on_project_task_progress(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        self._append_log(f"[PROJECT TASK] {msg}")
        if self._project_popup_active:
            self._repo_scan_dialog.set_detail_text(msg)

    def _on_project_task_done(self, payload: object, on_success, heading: str) -> None:
        self._project_worker = None
        self._update_project_actions()
        msg = None
        try:
            msg = on_success(payload)
        except Exception as exc:
            try:
                self._append_log(f"[PROJECT TASK ERROR] {heading}: {exc}")
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, heading, str(exc))
            if self._project_popup_active:
                self._repo_scan_dialog.finish(f"{heading} failed: {exc}", success=False)
                self._project_popup_active = False
            return
        if self._project_popup_active:
            self._repo_scan_dialog.finish(msg or f"{heading} complete", success=True)
            self._project_popup_active = False

    def _on_project_task_error(self, message: str, heading: str) -> None:
        self._project_worker = None
        self._update_project_actions()
        msg = str(message or "").strip() or f"{heading} failed."
        try:
            self._append_log(f"[PROJECT TASK ERROR] {heading}: {msg}")
        except Exception:
            pass
        QtWidgets.QMessageBox.warning(self, heading, msg)
        if self._project_popup_active:
            self._repo_scan_dialog.finish(f"{heading} failed: {msg}", success=False)
            self._project_popup_active = False

    def _handle_project_update_success(self, payload: object, *, wb_path: Path, ptype: str, started: float) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Project update returned an invalid payload.")

        def _td_excluded_source_line(item: object) -> str:
            row = item if isinstance(item, dict) else {}
            serial = str(row.get("serial") or "").strip() or "unknown serial"
            reference = (
                str(row.get("metadata_rel") or "").strip()
                or str(row.get("excel_sqlite_rel") or "").strip()
                or str(row.get("artifacts_rel") or "").strip()
            )
            reason = str(row.get("reason") or "").strip() or "Excluded from compilation."
            if reference:
                return f"{serial} ({reference}): {reason}"
            return f"{serial}: {reason}"

        updated = int(payload.get("updated_cells") or 0)
        missing_src = int(payload.get("missing_source") or 0)
        missing_val = int(payload.get("missing_value") or 0)
        serials = int(payload.get("serials_in_workbook") or 0)
        have_src = int(payload.get("serials_with_source") or 0)
        serials_added = int(payload.get("serials_added") or 0)
        added_serials = payload.get("added_serials") or []
        compiled_serials = [str(value).strip() for value in (payload.get("compiled_serials") or []) if str(value).strip()]
        compiled_serials_count = int(payload.get("compiled_serials_count") or len(compiled_serials))
        excluded_sources = [dict(item) for item in (payload.get("excluded_sources") or []) if isinstance(item, dict)]
        excluded_sources_count = int(payload.get("excluded_sources_count") or len(excluded_sources))
        warning_summary = str(payload.get("warning_summary") or "").strip()
        cache_sync_mode = str(payload.get("cache_sync_mode") or "").strip()
        cache_sync_reason = str(payload.get("cache_sync_reason") or "").strip()
        cache_sync_counts = payload.get("cache_sync_counts") if isinstance(payload.get("cache_sync_counts"), dict) else {}
        cache_state = payload.get("cache_state") if isinstance(payload.get("cache_state"), dict) else {}
        cache_validation_ok = bool(payload.get("cache_validation_ok"))
        cache_validation_error = str(payload.get("cache_validation_error") or "").strip()
        cache_validation_summary = str(payload.get("cache_validation_summary") or "").strip()
        cache_debug_path = str(payload.get("cache_debug_path") or "").strip()
        backend_module_path = str(payload.get("backend_module_path") or "").strip()
        dbg = str(payload.get("debug_json") or "").strip()
        log_path = str(payload.get("log_path") or "").strip()
        elapsed_s = round(time.perf_counter() - started, 3)
        self._append_log(
            f"[PROJECT UPDATE] Finished in {elapsed_s:.3f}s: updated={updated}, serials={serials}, added={serials_added}, sources={have_src}, missing_source={missing_src}, missing_value={missing_val}"
        )
        if cache_sync_mode:
            self._append_log(
                "[PROJECT UPDATE] Cache sync mode="
                f"{cache_sync_mode}, reason={cache_sync_reason or 'n/a'}, "
                f"added={int(cache_sync_counts.get('added') or 0)}, "
                f"changed={int(cache_sync_counts.get('changed') or 0)}, "
                f"removed={int(cache_sync_counts.get('removed') or 0)}, "
                f"unchanged={int(cache_sync_counts.get('unchanged') or 0)}, "
                f"missing={int(cache_sync_counts.get('missing') or 0)}, "
                f"invalid={int(cache_sync_counts.get('invalid') or 0)}, "
                f"reingested={int(cache_sync_counts.get('reingested') or 0)}"
            )
        if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending") and not cache_validation_ok:
            failure_lines = ["Update Project did not finish with a Trend/Analyze-ready Test Data build."]
            if cache_validation_error:
                failure_lines.append(cache_validation_error)
            if cache_validation_summary:
                failure_lines.append(f"TD cache summary: {cache_validation_summary}")
            if cache_debug_path:
                failure_lines.append(f"TD cache debug: {cache_debug_path}")
            if log_path:
                failure_lines.append(f"Log: {log_path}")
            raise RuntimeError("\n".join(failure_lines))
        if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
            self._append_log(
                "[PROJECT UPDATE] TD cache validation="
                f"{'ok' if cache_validation_ok else 'failed'}"
                + (f": {cache_validation_error}" if cache_validation_error else "")
            )
            if cache_validation_summary:
                self._append_log(f"[PROJECT UPDATE] TD cache summary: {cache_validation_summary}")
            if cache_debug_path:
                self._append_log(f"[PROJECT UPDATE] TD cache debug: {cache_debug_path}")
            if backend_module_path:
                self._append_log(f"[PROJECT UPDATE] Backend module: {backend_module_path}")
            timings = payload.get("timings")
            if not isinstance(timings, dict) and dbg:
                try:
                    dbg_payload = json.loads(dbg)
                    if isinstance(dbg_payload, dict):
                        timings = dbg_payload.get("timings_s")
                except Exception:
                    timings = None
            if isinstance(timings, dict):
                ordered_keys = [
                    "support_refresh_s",
                    "pre_cache_workbook_save_s",
                    "cache_ensure_s",
                    "cache_read_s",
                    "data_calc_build_s",
                    "metrics_long_sheet_s",
                    "raw_cache_long_sheet_s",
                    "perf_candidates_main_s",
                    "perf_candidates_cp_total_s",
                    "perf_candidates_cp_count",
                    "metadata_sync_s",
                    "post_cache_workbook_build_s",
                    "final_workbook_save_s",
                    "total_s",
                ]
                shown = set()
                for key in ordered_keys:
                    if key not in timings:
                        continue
                    shown.add(key)
                    self._append_log(f"[TD UPDATE TIMING] {key}={timings.get(key)}")
                for key in sorted(str(k) for k in timings.keys() if str(k) not in shown):
                    self._append_log(f"[TD UPDATE TIMING] {key}={timings.get(key)}")
            saved_refresh = payload.get("saved_equation_refresh")
            if isinstance(saved_refresh, dict):
                self._append_log(
                    "[PROJECT UPDATE] Saved equations refreshed="
                    f"{int(saved_refresh.get('refreshed_count') or 0)}, failed={int(saved_refresh.get('failed_count') or 0)}"
                )
                for err in (saved_refresh.get("errors") or []):
                    if str(err).strip():
                        self._append_log(f"[PROJECT UPDATE] Saved equation refresh error: {err}")
            if excluded_sources_count:
                self._append_log(
                    f"[PROJECT UPDATE WARNING] Excluded TD sources: {excluded_sources_count}"
                    + (f" | {warning_summary}" if warning_summary else "")
                )
                for item in excluded_sources[:20]:
                    self._append_log(f"[PROJECT UPDATE WARNING] {_td_excluded_source_line(item)}")
                if len(excluded_sources) > 20:
                    self._append_log(
                        f"[PROJECT UPDATE WARNING] ... and {len(excluded_sources) - 20} more excluded TD source(s)"
                    )

        lines = [
            f"Updated cells: {updated}",
            f"Serials in workbook: {serials}",
        ]
        if serials_added:
            lines.append(f"Serials auto-added: {serials_added}")
        if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
            lines.extend(
                [
                    f"Compiled serials: {compiled_serials_count}",
                    f"Excluded TD sources: {excluded_sources_count}",
                    f"No value found: {missing_val}",
                    "",
                    f"Workbook: {payload.get('workbook') or wb_path}",
                ]
            )
        else:
            lines.extend(
                [
                    f"Serials with debug source: {have_src}",
                    f"Missing debug source: {missing_src}",
                    f"No value found: {missing_val}",
                    "",
                    f"Workbook: {payload.get('workbook') or wb_path}",
                ]
            )
        if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
            if cache_sync_mode:
                lines.append(f"Cache sync mode: {cache_sync_mode}")
            lines.append(f"TD cache validation: {'ready' if cache_validation_ok else 'failed'}")
            impl_counts = cache_state.get("impl_counts") if isinstance(cache_state.get("impl_counts"), dict) else {}
            raw_counts = cache_state.get("raw_counts") if isinstance(cache_state.get("raw_counts"), dict) else {}
            source_status_counts = cache_state.get("source_status_counts") if isinstance(cache_state.get("source_status_counts"), dict) else {}
            if impl_counts or raw_counts:
                lines.append(
                    "TD cache counts: "
                    f"td_runs={int(impl_counts.get('td_runs') or 0)}, "
                    f"td_columns_calc_y={int(impl_counts.get('td_columns_calc_y') or 0)}, "
                    f"td_metrics_calc={int(impl_counts.get('td_metrics_calc') or 0)}, "
                    f"td_condition_observations_sequences={int(impl_counts.get('td_condition_observations_sequences') or 0)}, "
                    f"td_metrics_calc_sequences={int(impl_counts.get('td_metrics_calc_sequences') or 0)}, "
                    f"td_raw_sequences={int(raw_counts.get('td_raw_sequences') or 0)}, "
                    f"td_columns_raw_y={int(raw_counts.get('td_columns_raw_y') or 0)}, "
                    f"td_curves_raw={int(raw_counts.get('td_curves_raw') or 0)}"
                )
            if source_status_counts:
                lines.append(
                    "TD excluded source counts: "
                    f"missing={int(source_status_counts.get('missing') or 0)}, "
                    f"invalid={int(source_status_counts.get('invalid') or 0)}, "
                    f"non_ok={int(source_status_counts.get('non_ok') or 0)}"
                )
            if warning_summary:
                lines.append(warning_summary)
            if excluded_sources:
                lines.append("Excluded from compilation until fixed:")
                for item in excluded_sources[:10]:
                    lines.append(_td_excluded_source_line(item))
                if len(excluded_sources) > 10:
                    lines.append(f"... and {len(excluded_sources) - 10} more excluded TD source(s)")
            if cache_validation_summary:
                lines.append(f"TD cache summary: {cache_validation_summary}")
            if cache_debug_path:
                lines.append(f"TD cache debug: {cache_debug_path}")
            lines.append("Trend/Analyze is ready. No additional cache build is required.")
            lines.append("Performance candidate sheets remain on demand.")
            saved_refresh = payload.get("saved_equation_refresh")
            if isinstance(saved_refresh, dict):
                lines.append(
                    "Saved equations refreshed: "
                    f"{int(saved_refresh.get('refreshed_count') or 0)}"
                    f" (failed: {int(saved_refresh.get('failed_count') or 0)})"
                )
        if log_path:
            lines.append(f"Log: {log_path}")
        msg = "\n".join(lines)
        if serials_added and isinstance(added_serials, list) and added_serials:
            shown = ", ".join(str(s).strip() for s in added_serials[:20] if str(s).strip())
            if shown:
                msg += f"\nAdded serials: {shown}" + (" ..." if len(added_serials) > 20 else "")
        if dbg:
            self._append_log(f"[PROJECT UPDATE] Debug JSON: {dbg}")
        QtWidgets.QMessageBox.information(self, "Project Updated", msg)
        if excluded_sources_count:
            self._show_toast(f"Project updated with warnings: {updated} cell(s)")
            return f"Project updated with warnings in {elapsed_s:.1f}s"
        self._show_toast(f"Project updated and ready: {updated} cell(s)")
        return f"Project updated and ready in {elapsed_s:.1f}s"

    def _handle_project_performance_sheet_success(self, payload: object, *, wb_path: Path) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Performance-sheet generation returned an invalid payload.")
        timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
        log_path = str(payload.get("log_path") or "").strip()
        perf_count = int(payload.get("performance_candidate_count") or 0)
        cp_count = int(payload.get("performance_cp_sheet_count") or 0)
        if isinstance(timings, dict):
            for key in ("perf_candidates_main_s", "perf_candidates_cp_total_s", "perf_candidates_cp_count", "final_workbook_save_s", "total_s"):
                if key in timings:
                    self._append_log(f"[TD PERFORMANCE TIMING] {key}={timings.get(key)}")
        lines = [
            f"Workbook: {payload.get('workbook') or wb_path}",
            f"Performance candidate rows: {perf_count}",
            f"Control-period sheets: {cp_count}",
        ]
        if log_path:
            lines.append(f"Log: {log_path}")
        QtWidgets.QMessageBox.information(self, "Performance Sheets Generated", "\n".join(lines))
        self._show_toast("Performance sheets generated")
        return "Performance sheets generated"

    def _handle_project_debug_excel_success(self, payload: object, *, wb_path: Path) -> str:
        if not isinstance(payload, dict):
            raise RuntimeError("Debug Excel export returned an invalid payload.")
        ordered_keys = (
            "implementation_excel",
            "raw_cache_excel",
            "raw_points_excel",
        )
        lines = [str(payload.get(key) or "").strip() for key in ordered_keys if str(payload.get(key) or "").strip()]
        if not lines:
            raise RuntimeError(f"No debug Excel files were generated for {wb_path}.")
        QtWidgets.QMessageBox.information(
            self,
            "Debug Excel Files",
            "Generated debug Excel files:\n\n" + "\n".join(lines),
        )
        self._show_toast("Debug Excel files generated")
        return "Debug Excel files generated"

    def _act_open_project_folder(self) -> None:
        item = self._selected_project_item(2)
        if not item:
            return
        path = Path(str(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text() or "")).expanduser()
        try:
            be.open_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Folder", str(exc))

    def _act_open_project_workbook(self) -> None:
        item = self._selected_project_item(3)
        if not item:
            return
        path = Path(str(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text() or "")).expanduser()
        try:
            be.open_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Workbook", str(exc))

    def _act_open_project_support_workbook(self) -> None:
        try:
            record = self._selected_project_record()
            if not record:
                raise RuntimeError("Select a project in the list first.")
            ptype = str(record.get("type") or "").strip()
            if ptype != getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                raise RuntimeError("Support workbooks are available only for Test Data Trending projects.")
            project_dir = Path(str(record.get("folder") or "")).expanduser()
            workbook = Path(str(record.get("workbook") or "")).expanduser()
            path = be.td_support_workbook_path_for(workbook, project_dir=project_dir)
            if not path.exists():
                raise RuntimeError(f"Support workbook not found: {path}")
            be.open_path(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Support Workbook", str(exc))

    def _act_project_env(self) -> None:
        try:
            record = self._selected_project_record()
            if not record:
                raise RuntimeError("Select a project in the list first.")
            project_dir = Path(str(record.get("folder") or "")).expanduser()
            if not project_dir.exists():
                raise RuntimeError(f"Project folder not found: {project_dir}")
            dlg = ProjectEnvDialog(project_dir, self)
            self._prepare_dialog(dlg)
            dlg.exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Project Env", str(exc))

    def _act_open_project_implementation(self) -> None:
        try:
            record = self._selected_project_record()
            if not record:
                raise RuntimeError("Select a project in the list first.")
            ptype = str(record.get("type") or "").strip()
            if ptype not in (
                getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"),
                getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"),
                getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
            ):
                raise RuntimeError(
                    "Implementation is available only for EIDP Trending, Raw File Trending, or Test Data Trending projects."
                )

            project_dir = Path(str(record.get("folder") or "")).expanduser()
            workbook = Path(str(record.get("workbook") or "")).expanduser()
            if not project_dir.exists():
                raise RuntimeError(f"Project folder not found: {project_dir}")
            if not workbook.exists():
                raise RuntimeError(f"Project workbook not found: {workbook}")

            if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                dlg = TestDataTrendDialog(project_dir, workbook, self)
            else:
                dlg = ImplementationTrendDialog(project_dir, workbook, self)
            self._prepare_dialog(dlg)
            dlg.exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Implementation", str(exc))

    def _act_update_project(self) -> None:
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first (Setup tab).")
            repo = Path(repo_raw).expanduser()
            record = self._selected_project_record()
            if not record:
                raise RuntimeError("Select a project in the list first.")
            wb_path = Path(str(record.get("workbook") or "")).expanduser()
            ptype = str(record.get("type") or "").strip()
            overwrite = bool(getattr(self, "cb_project_overwrite", None) and self.cb_project_overwrite.isChecked())
            project_name = str(record.get("name") or wb_path.stem or "").strip() or wb_path.stem
            started = time.perf_counter()
            self._append_log(f"[PROJECT UPDATE] Starting: {project_name}")
            self._append_log(f"[PROJECT UPDATE] Workbook: {wb_path}")

            def _task(report):
                if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"):
                    report("Updating EIDP trending workbook")
                    return be.update_eidp_trending_project_workbook(repo, wb_path, overwrite=overwrite)
                if ptype == getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"):
                    report("Updating raw-file trending workbook")
                    return be.update_eidp_raw_trending_project_workbook(repo, wb_path, overwrite=overwrite)
                if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                    return be.update_test_data_trending_project_workbook(
                        repo,
                        wb_path,
                        overwrite=overwrite,
                        include_performance_sheets=True,
                        progress_cb=report,
                    )
                raise RuntimeError(f"Unsupported project type: {ptype}")

            self._start_project_task(
                heading="Update Project",
                status_text=f"Updating {project_name}",
                project_dir=wb_path.parent,
                log_prefix="project_update",
                task_factory=_task,
                on_success=lambda payload: self._handle_project_update_success(payload, wb_path=wb_path, ptype=ptype, started=started),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Update Project", str(exc))

    def _act_generate_project_performance_sheets(self) -> None:
        try:
            record = self._selected_project_record()
            if not record:
                raise RuntimeError("Select a project in the list first.")
            ptype = str(record.get("type") or "").strip()
            if ptype != getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                raise RuntimeError("Performance sheets are available only for Test Data Trending projects.")
            project_dir = Path(str(record.get("folder") or "")).expanduser()
            wb_path = Path(str(record.get("workbook") or "")).expanduser()
            project_name = str(record.get("name") or wb_path.stem or "").strip() or wb_path.stem

            def _task(report):
                return be.generate_test_data_project_performance_sheets(
                    project_dir,
                    wb_path,
                    progress_cb=report,
                )

            self._append_log(f"[PROJECT PERFORMANCE] Starting: {project_name}")
            self._append_log(f"[PROJECT PERFORMANCE] Workbook: {wb_path}")
            self._start_project_task(
                heading="Generate Performance Sheets",
                status_text=f"Generating performance sheets for {project_name}",
                project_dir=project_dir,
                log_prefix="project_performance_sheets",
                task_factory=_task,
                on_success=lambda payload: self._handle_project_performance_sheet_success(payload, wb_path=wb_path),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Generate Performance Sheets", str(exc))

    def _act_generate_project_debug_excels(self) -> None:
        try:
            record = self._selected_project_record()
            if not record:
                raise RuntimeError("Select a project in the list first.")
            ptype = str(record.get("type") or "").strip()
            if ptype != getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                raise RuntimeError("Debug Excel export is available only for Test Data Trending projects.")
            project_dir = Path(str(record.get("folder") or "")).expanduser()
            wb_path = Path(str(record.get("workbook") or "")).expanduser()
            project_name = str(record.get("name") or wb_path.stem or "").strip() or wb_path.stem

            def _task(report):
                return be.export_test_data_project_debug_excels(
                    project_dir,
                    wb_path,
                    force=True,
                    progress_cb=report,
                )

            self._append_log(f"[PROJECT DEBUG EXCEL] Starting: {project_name}")
            self._append_log(f"[PROJECT DEBUG EXCEL] Workbook: {wb_path}")
            self._start_project_task(
                heading="Generate Debug Excel Files",
                status_text=f"Generating debug Excel files for {project_name}",
                project_dir=project_dir,
                log_prefix="project_debug_excel_files",
                task_factory=_task,
                on_success=lambda payload: self._handle_project_debug_excel_success(payload, wb_path=wb_path),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Generate Debug Excel Files", str(exc))

    def _act_delete_project(self) -> None:
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first (Setup tab).")
            repo = Path(repo_raw).expanduser()
            folder_item = self._selected_project_item(2)
            name_item = self._selected_project_item(0)
            if not folder_item:
                raise RuntimeError("Select a project in the list first.")
            project_name = (name_item.text() if name_item else "this project").strip()
            project_dir = Path(str(folder_item.data(QtCore.Qt.ItemDataRole.UserRole) or folder_item.text() or "")).expanduser()

            resp = QtWidgets.QMessageBox.question(
                self,
                "Delete Project",
                f"Delete project '{project_name}'?\n\nThis will permanently delete:\n{project_dir}",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            payload = be.delete_eidat_project(repo, project_dir)
            self._append_log(f"[PROJECT DELETE] {payload}")
            self._refresh_projects()
            self._show_toast("Project deleted")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete Project", str(exc))

    # Actions & helpers
    def _switch_tab(self, idx: int):
        """Switch to a tab by index and update button states."""
        # Update button checked states
        self.btn_tab_setup.setChecked(idx == 0)
        self.btn_tab_files.setChecked(idx == 1)
        self.btn_tab_product_center.setChecked(idx == 2)
        self.btn_tab_projects.setChecked(idx == 3)
        # Switch the actual tab
        self.tabs.setCurrentIndex(idx)

    def _on_tab_changed(self, idx: int):
        try:
            if self.tabs.widget(idx) is self.tab_files:
                self._refresh_files_tab()
            elif self.tabs.widget(idx) is self.tab_product_center:
                self._refresh_product_center()
            elif self.tabs.widget(idx) is self.tab_projects:
                self._refresh_projects()
        except Exception:
            pass

    # ============ Files Tab Methods ============
    def _act_files_refresh(self) -> None:
        """Refresh Files tab and rescan repo so new files appear immediately."""
        try:
            # Fast path: refresh from DB first.
            self._refresh_files_tab()
        except Exception:
            pass
        # Then scan the repo to pick up new/changed files (PDF/Excel) (no processing).
        try:
            self._act_manager_scan_all(auto=True, auto_process=False)
        except Exception:
            pass

    def _refresh_files_tab(self) -> None:
        """Refresh the entire Files tab from database."""
        repo_raw = (self.ed_global_repo.text() or "").strip() if hasattr(self, "ed_global_repo") else ""
        if not repo_raw:
            self.lbl_files_status.setText("Select a Global Repo in Setup tab")
            self._files_data = []
            self._refresh_files_tree()
            return

        try:
            self._files_data = be.read_files_with_index_metadata(Path(repo_raw))
            self.lbl_files_status.setText(f"{len(self._files_data)} file(s) tracked")
        except Exception as exc:
            self._files_data = []
            self.lbl_files_status.setText(f"Error: {exc}")

        self._refresh_files_tree()

    def _refresh_files_tree(self) -> None:
        """Rebuild the tree based on current grouping mode."""
        self.tree_files.clear()
        if not self._files_data:
            self._files_filtered = []
            self._populate_files_table([])
            return

        primary = self.cmb_files_primary.currentText()
        secondary = self.cmb_files_secondary.currentText()

        # Helper to get grouping value from file
        def get_group_value(f: dict, group_type: str) -> str:
            if group_type == "Program":
                return f.get("program_title") or "(No Program)"
            elif group_type == "Asset Type":
                return f.get("asset_type") or "(No Asset)"
            elif group_type == "Serial Number":
                return f.get("serial_number") or "(No Serial)"
            elif group_type == "Doc Type":
                return f.get("document_type") or "(No Doc Type)"
            return "(Unknown)"

        # Build hierarchical tree
        tree_data: dict = {}
        for f in self._files_data:
            # Build path based on primary and optional secondary
            path_list = [get_group_value(f, primary)]
            if secondary != "None":
                path_list.append(get_group_value(f, secondary))
            path = tuple(path_list)

            # Build nested dict - only add file to the leaf (deepest) node
            current = tree_data
            for i, p in enumerate(path):
                if p not in current:
                    current[p] = {"_files": [], "_children": {}}
                # Only add file at the leaf level (last item in path)
                if i == len(path) - 1:
                    current[p]["_files"].append(f)
                current = current[p]["_children"]

        # Helper to count all files in a node (including children)
        def count_all_files(node: dict) -> int:
            total = len(node.get("_files", []))
            for child in node.get("_children", {}).values():
                total += count_all_files(child)
            return total

        # Helper to collect all files from a node (including children)
        def collect_all_files(node: dict) -> list[dict]:
            all_files = list(node.get("_files", []))
            for child in node.get("_children", {}).values():
                all_files.extend(collect_all_files(child))
            return all_files

        # Convert to tree items
        def build_tree_items(parent, data_dict):
            for key, val in sorted(data_dict.items()):
                count = count_all_files(val)
                item = QtWidgets.QTreeWidgetItem(parent, [f"{key} ({count})"])
                # Store all files from this node and children for filtering
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, collect_all_files(val))
                build_tree_items(item, val.get("_children", {}))

        build_tree_items(self.tree_files, tree_data)
        self.tree_files.expandToDepth(0)

        # Show all files initially
        self._files_filtered = list(self._files_data)
        self._files_refresh_table_view()

    def _on_files_tree_selection(self) -> None:
        """When tree selection changes, update table."""
        items = self.tree_files.selectedItems()
        if not items:
            self._files_filtered = self._files_data
        else:
            # Each node already contains all files (including children)
            # Use a set to avoid duplicates if parent and child are both selected
            seen_paths: set[str] = set()
            files: list[dict] = []

            for item in items:
                data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if data:
                    for f in data:
                        rel_path = f.get("rel_path", "")
                        if rel_path not in seen_paths:
                            seen_paths.add(rel_path)
                            files.append(f)

            self._files_filtered = files

        self._files_refresh_table_view()

    def _files_refresh_table_view(self) -> None:
        """Refresh the Files table based on tree selection + deep-search filter."""
        base_files = getattr(self, "_files_filtered", None)
        if base_files is None:
            base_files = getattr(self, "_files_data", []) or []

        external_rel_paths = getattr(self, "_files_external_rel_paths", None)
        if external_rel_paths is not None:
            files = [f for f in (base_files or []) if str(f.get("rel_path") or "") in external_rel_paths]
        else:
            files = list(base_files or [])

        deep_matches: set[str] | None = getattr(self, "_files_deep_matches", None)
        if deep_matches is not None:
            files = [f for f in files if str(f.get("rel_path") or "") in deep_matches]

        self._populate_files_table(files)

    def _update_files_external_filter_banner(self) -> None:
        label = getattr(self, "_files_external_filter_label", "").strip()
        rel_paths = getattr(self, "_files_external_rel_paths", None)
        active = bool(rel_paths)
        if hasattr(self, "lbl_files_subset"):
            self.lbl_files_subset.setVisible(active)
            self.lbl_files_subset.setText(label if active else "")
        if hasattr(self, "btn_files_clear_subset"):
            self.btn_files_clear_subset.setVisible(active)

    def _set_files_external_subset(self, rel_paths: list[str], *, label: str) -> None:
        cleaned = {str(value).strip() for value in (rel_paths or []) if str(value).strip()}
        self._files_external_rel_paths = cleaned or None
        self._files_external_filter_label = str(label or "").strip()
        self._update_files_external_filter_banner()
        self._files_refresh_table_view()

    def _clear_files_external_subset(self) -> None:
        self._files_external_rel_paths = None
        self._files_external_filter_label = ""
        self._update_files_external_filter_banner()
        self._files_refresh_table_view()

    def _populate_files_table(self, files: list[dict]) -> None:
        """Populate the table with file data."""
        self.tbl_files.setSortingEnabled(False)
        self.tbl_files.setRowCount(0)

        # New implementation: populate all available metadata columns (support DB + index DB).
        self._populate_files_table_all_metadata(files)
        try:
            # Keep metadata filter applied when the table repopulates (tree selection, refresh, deep search).
            self._apply_files_filter(self.ed_files_filter.text())
        except Exception:
            pass
        return

        for row, f in enumerate(files):
            self.tbl_files.insertRow(row)

            rel_path = f.get("rel_path", "")
            filename = Path(rel_path).name if rel_path else ""

            # Status - green checkmark if processed, red X if not
            processed = f.get("last_processed_epoch_ns")
            is_processed = bool(processed)

            items_data = [
                filename,
                f.get("program_title") or "",
                f.get("asset_type") or "",
                f.get("serial_number") or "",
                f.get("document_type") or "",
                f.get("report_date") or "",
                f.get("test_date") or "",
            ]

            for c, val in enumerate(items_data):
                item = QtWidgets.QTableWidgetItem(val)
                # Store full file info in first column
                if c == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, f)
                self.tbl_files.setItem(row, c, item)

            # Status column with colored icon
            status_item = QtWidgets.QTableWidgetItem("✓" if is_processed else "✗")
            status_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            status_item.setForeground(QtGui.QColor("#16a34a") if is_processed else QtGui.QColor("#dc2626"))
            font = status_item.font()
            font.setPointSize(14)
            font.setBold(True)
            status_item.setFont(font)
            self.tbl_files.setItem(row, 7, status_item)

            # Certification column with status and pass rate
            cert_status = f.get("certification_status") or ""
            cert_pass_rate = f.get("certification_pass_rate") or ""
            if cert_status and cert_pass_rate:
                cert_text = f"{cert_status} ({cert_pass_rate})"
            elif cert_status:
                cert_text = cert_status
            else:
                cert_text = "-"

            cert_item = QtWidgets.QTableWidgetItem(cert_text)
            cert_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            # Color coding based on status
            cert_colors = {
                "CERTIFIED": "#16a34a",  # Green
                "FAILED": "#dc2626",     # Red
                "PENDING": "#f59e0b",    # Amber
                "NO_DATA": "#6b7280",    # Gray
            }
            cert_color = cert_colors.get(cert_status, "#6b7280")
            cert_item.setForeground(QtGui.QColor(cert_color))

            cert_font = cert_item.font()
            cert_font.setBold(True)
            cert_item.setFont(cert_font)
            self.tbl_files.setItem(row, 8, cert_item)

        self.tbl_files.resizeColumnsToContents()
        self.tbl_files.setSortingEnabled(True)
        self.lbl_files_count.setText(f"{len(files)} file(s)")

    def _populate_files_table_all_metadata(self, files: list[dict]) -> None:
        cols = getattr(self, "_files_table_cols", [])
        if cols:
            try:
                if self.tbl_files.columnCount() != len(cols):
                    self.tbl_files.setColumnCount(len(cols))
                    self.tbl_files.setHorizontalHeaderLabels([c[0] for c in cols])
                    # If the column count ever changes (e.g., after an update),
                    # re-apply visibility preferences.
                    self._files_load_column_visibility()
            except Exception:
                pass

        # Cache role for fast filtering (avoid joining across many columns on every keystroke).
        search_role = int(QtCore.Qt.ItemDataRole.UserRole) + 1

        def fmt_epoch_ns(ns_val) -> str:
            if ns_val is None or ns_val == "":
                return ""
            try:
                ns_int = int(ns_val)
            except Exception:
                return str(ns_val)
            if ns_int <= 0:
                return ""
            try:
                dt = datetime.datetime.fromtimestamp(ns_int / 1_000_000_000)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(ns_int)

        epoch_ns_keys = {
            "mtime_ns",
            "first_seen_epoch_ns",
            "last_seen_epoch_ns",
            "last_processed_epoch_ns",
            "last_processed_mtime_ns",
            "indexed_epoch_ns",
        }

        for row, f in enumerate(files):
            self.tbl_files.insertRow(row)

            rel_path = f.get("rel_path", "")
            filename = Path(rel_path).name if rel_path else ""
            ext_fallback = ""
            try:
                ext_fallback = str(Path(rel_path).suffix or "").strip()
            except Exception:
                ext_fallback = ""

            processed = f.get("last_processed_epoch_ns")
            is_processed = bool(processed)

            cert_status = str(f.get("certification_status") or "").strip()
            cert_pass_rate = str(f.get("certification_pass_rate") or "").strip()
            if cert_status and cert_pass_rate:
                cert_text = f"{cert_status} ({cert_pass_rate})"
            elif cert_status:
                cert_text = cert_status
            else:
                cert_text = "-"

            search_parts: list[str] = []
            file_name_item: QtWidgets.QTableWidgetItem | None = None

            for col_idx, (_label, key) in enumerate(cols):
                item = QtWidgets.QTableWidgetItem("")

                if key == "_filename":
                    item.setText(filename)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, f)
                    file_name_item = item
                    search_parts.append(filename)
                elif key == "_processed":
                    status_text = ("\u2713 Processed") if is_processed else ("\u2717 Unprocessed")
                    item.setText(status_text)
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QtGui.QColor("#16a34a") if is_processed else QtGui.QColor("#dc2626"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    search_parts.append(status_text)
                elif key == "_certification":
                    item.setText(cert_text)
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    cert_colors = {
                        "CERTIFIED": "#16a34a",  # Green
                        "FAILED": "#dc2626",     # Red
                        "PENDING": "#f59e0b",    # Amber
                        "NO_DATA": "#6b7280",    # Gray
                    }
                    cert_color = cert_colors.get(cert_status, "#6b7280")
                    item.setForeground(QtGui.QColor(cert_color))
                    cert_font = item.font()
                    cert_font.setBold(True)
                    item.setFont(cert_font)
                    search_parts.append(cert_text)
                elif key in epoch_ns_keys:
                    val = fmt_epoch_ns(f.get(key))
                    item.setText(val)
                    search_parts.append(val)
                elif key == "needs_processing":
                    try:
                        needs = int(f.get("needs_processing") or 0)
                    except Exception:
                        needs = 0
                    val = "YES" if needs else "NO"
                    item.setText(val)
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QtGui.QColor("#dc2626") if needs else QtGui.QColor("#16a34a"))
                    search_parts.append(val)
                elif key == "file_extension":
                    raw = f.get("file_extension")
                    s = "" if (raw is None) else str(raw).strip()
                    if not s and ext_fallback:
                        s = ext_fallback
                    item.setText(s)
                    search_parts.append(s)
                else:
                    raw = f.get(key)
                    s = "" if (raw is None) else str(raw)
                    item.setText(s)
                    search_parts.append(s)

                self.tbl_files.setItem(row, col_idx, item)

            if file_name_item is not None:
                try:
                    blob = " ".join(p for p in search_parts if p).lower()
                except Exception:
                    blob = ""
                file_name_item.setData(search_role, blob)

        self.tbl_files.resizeColumnsToContents()
        self.tbl_files.setSortingEnabled(True)
        self.lbl_files_count.setText(f"{len(files)} file(s)")

    def _apply_files_filter(self, text: str) -> None:
        """Filter table rows by search text."""
        text = text.strip().lower()
        search_role = int(QtCore.Qt.ItemDataRole.UserRole) + 1
        for row in range(self.tbl_files.rowCount()):
            combined = ""
            item0 = self.tbl_files.item(row, 0)
            if item0 is not None:
                try:
                    combined = str(item0.data(search_role) or "")
                except Exception:
                    combined = ""
            if not combined:
                combined = " ".join(
                    (self.tbl_files.item(row, c).text() if self.tbl_files.item(row, c) else "")
                    for c in range(self.tbl_files.columnCount())
                ).lower()
            match = (text in combined) if text else True
            self.tbl_files.setRowHidden(row, not match)

    def _act_files_deep_clear(self) -> None:
        """Clear deep-search results and show the normal Files view again."""
        self._files_deep_matches = None
        self._files_deep_last_query = ""
        try:
            self.lbl_files_deep_status.setText("")
        except Exception:
            pass
        try:
            self.cmb_files_deep_search.blockSignals(True)
            self.cmb_files_deep_search.clear()
            self.cmb_files_deep_search.setCurrentIndex(-1)
            self.cmb_files_deep_search.setEditText("")
        except Exception:
            pass
        finally:
            try:
                self.cmb_files_deep_search.blockSignals(False)
            except Exception:
                pass
        self._files_refresh_table_view()

    def _act_files_deep_search(self) -> None:
        """Search inside combined.txt and filter the Files table to matching documents."""
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
        except Exception:
            repo_raw = ""
        if not repo_raw:
            QtWidgets.QMessageBox.warning(self, "Deep Search", "Select a Global Repo in Setup tab first.")
            return

        query = ""
        try:
            query = (self.cmb_files_deep_search.lineEdit().text() or "").strip()
        except Exception:
            try:
                query = (self.cmb_files_deep_search.currentText() or "").strip()
            except Exception:
                query = ""

        if not query:
            self._act_files_deep_clear()
            return

        base_files = getattr(self, "_files_filtered", None)
        if base_files is None:
            base_files = getattr(self, "_files_data", []) or []

        rel_paths: list[str] = []
        seen: set[str] = set()
        for f in base_files:
            rp = str(f.get("rel_path") or "").strip()
            if rp and rp not in seen:
                seen.add(rp)
                rel_paths.append(rp)

        if not rel_paths:
            QtWidgets.QMessageBox.information(self, "Deep Search", "No files available to search.")
            return

        try:
            self.btn_files_deep_search.setEnabled(False)
            self.lbl_files_deep_status.setText(f"Searching combined.txt for: {query}")
        except Exception:
            pass

        def task():
            return be.deep_search_combined_txt(Path(repo_raw), rel_paths, query)

        worker = ManagerTaskWorker(task, parent=self)
        self._files_deep_worker = worker

        def _done(payload: object) -> None:
            try:
                if not isinstance(payload, dict):
                    raise RuntimeError("Deep search returned invalid payload.")
                matched = payload.get("matched_rel_paths") or []
                if not isinstance(matched, list):
                    matched = []
                matched_rel_paths = [str(x) for x in matched if str(x).strip()]
                self._files_deep_matches = set(matched_rel_paths)
                self._files_deep_last_query = query

                scanned = int(payload.get("scanned") or 0)
                missing = int(payload.get("missing") or 0)
                self.lbl_files_deep_status.setText(
                    f"Deep search: {len(matched_rel_paths)} match(es) (scanned {scanned}, missing combined.txt {missing})"
                )

                # Show matching documents in the search bar dropdown.
                try:
                    self.cmb_files_deep_search.blockSignals(True)
                    self.cmb_files_deep_search.clear()
                    self.cmb_files_deep_search.addItems(matched_rel_paths)
                    self.cmb_files_deep_search.setCurrentIndex(-1)
                    self.cmb_files_deep_search.setEditText(query)
                finally:
                    try:
                        self.cmb_files_deep_search.blockSignals(False)
                    except Exception:
                        pass
                self._files_refresh_table_view()

                try:
                    if matched_rel_paths:
                        self.cmb_files_deep_search.showPopup()
                except Exception:
                    pass
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Deep Search", str(exc))
            finally:
                try:
                    self.btn_files_deep_search.setEnabled(True)
                except Exception:
                    pass

        def _fail(err: str) -> None:
            try:
                self.lbl_files_deep_status.setText(f"Deep search failed: {err}")
            except Exception:
                pass
            try:
                self.btn_files_deep_search.setEnabled(True)
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Deep Search", str(err))

        worker.completed.connect(_done)
        worker.failed.connect(_fail)
        worker.start()

    def _act_files_deep_pick_result(self, rel_path: str) -> None:
        """Select a matched document from the deep-search dropdown without losing the query."""
        rp = str(rel_path or "").strip()
        if not rp:
            return

        # Select the row in the table
        try:
            for row in range(self.tbl_files.rowCount()):
                item0 = self.tbl_files.item(row, 0)
                if item0 is None:
                    continue
                info = item0.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(info, dict) and str(info.get("rel_path") or "") == rp:
                    self.tbl_files.selectRow(row)
                    try:
                        self.tbl_files.scrollToItem(item0, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
                    except Exception:
                        pass
                    break
        except Exception:
            pass

        # Restore query text
        q = str(getattr(self, "_files_deep_last_query", "") or "")
        try:
            self.cmb_files_deep_search.blockSignals(True)
            self.cmb_files_deep_search.setCurrentIndex(-1)
            self.cmb_files_deep_search.setEditText(q)
        except Exception:
            pass
        finally:
            try:
                self.cmb_files_deep_search.blockSignals(False)
            except Exception:
                pass

    def _files_default_visible_column_keys(self) -> list[str]:
        # Keep the table compact by default, but still show the key “at-a-glance” metadata.
        # Always keep filename visible so rows are identifiable and selection logic remains clear.
        return [
            "_filename",
            "program_title",
            "asset_type",
            "metadata_source",
            "file_extension",
            "report_date",
            "test_date",
        ]

    def _files_column_key_to_index(self) -> dict[str, int]:
        cols = getattr(self, "_files_table_cols", []) or []
        return {key: idx for idx, (_label, key) in enumerate(cols)}

    def _files_visible_column_keys(self) -> list[str]:
        key_to_idx = self._files_column_key_to_index()
        visible: list[str] = []
        for key, idx in key_to_idx.items():
            try:
                if not self.tbl_files.isColumnHidden(idx):
                    visible.append(key)
            except Exception:
                continue
        # Always include filename for safety
        if "_filename" not in visible:
            visible.insert(0, "_filename")
        return visible

    def _files_apply_visible_column_keys(self, keys: list[str] | None) -> None:
        key_to_idx = self._files_column_key_to_index()
        visible = set(keys or [])
        visible.add("_filename")
        for key, idx in key_to_idx.items():
            try:
                self.tbl_files.setColumnHidden(idx, key not in visible)
            except Exception:
                pass

    def _files_load_column_visibility(self) -> None:
        """Apply persisted visibility; if none, apply defaults."""
        try:
            settings = QtCore.QSettings("EIDAT", "EIDAT-V1")
            raw = settings.value("files_table/visible_column_keys", "")
        except Exception:
            raw = ""

        keys: list[str] | None = None
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    keys = [str(x) for x in parsed if str(x).strip()]
            except Exception:
                keys = None
        elif isinstance(raw, (list, tuple)):
            try:
                keys = [str(x) for x in raw if str(x).strip()]
            except Exception:
                keys = None

        if not keys:
            keys = self._files_default_visible_column_keys()

        self._files_apply_visible_column_keys(keys)

    def _files_save_column_visibility(self) -> None:
        try:
            keys = self._files_visible_column_keys()
            settings = QtCore.QSettings("EIDAT", "EIDAT-V1")
            settings.setValue("files_table/visible_column_keys", json.dumps(keys))
        except Exception:
            pass

    def _files_columns_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                padding: 4px;
            }
            QMenu::item {
                background: transparent;
                color: #1f2937;
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background: #dbeafe;
                color: #1f2937;
            }
            QMenu::separator {
                height: 1px;
                background: #e5e7eb;
                margin: 4px 8px;
            }
        """)

        act_reset = menu.addAction("Reset to Default Columns")
        act_show_all = menu.addAction("Show All Columns")
        menu.addSeparator()

        def _do_reset():
            self._files_apply_visible_column_keys(self._files_default_visible_column_keys())
            self._files_save_column_visibility()

        def _do_show_all():
            key_to_idx = self._files_column_key_to_index()
            for idx in key_to_idx.values():
                try:
                    self.tbl_files.setColumnHidden(idx, False)
                except Exception:
                    pass
            self._files_save_column_visibility()

        act_reset.triggered.connect(_do_reset)
        act_show_all.triggered.connect(_do_show_all)

        cols = getattr(self, "_files_table_cols", []) or []
        key_to_idx = self._files_column_key_to_index()

        for label, key in cols:
            idx = key_to_idx.get(key)
            if idx is None:
                continue
            act = menu.addAction(label)
            act.setCheckable(True)
            try:
                act.setChecked(not self.tbl_files.isColumnHidden(idx))
            except Exception:
                act.setChecked(True)

            def _make_toggle(col_key: str, col_idx: int):
                def _toggle(checked: bool):
                    # Always keep filename visible
                    if col_key == "_filename" and not checked:
                        try:
                            self.tbl_files.setColumnHidden(col_idx, False)
                        except Exception:
                            pass
                        return
                    try:
                        self.tbl_files.setColumnHidden(col_idx, not checked)
                    except Exception:
                        pass
                    self._files_save_column_visibility()

                return _toggle

            act.toggled.connect(_make_toggle(key, idx))

        return menu

    def _files_open_columns_menu(self) -> None:
        try:
            menu = self._files_columns_menu()
            pos = self.btn_files_columns.mapToGlobal(QtCore.QPoint(0, self.btn_files_columns.height()))
            menu.exec(pos)
        except Exception:
            pass

    def _files_header_context_menu(self, pos) -> None:
        try:
            menu = self._files_columns_menu()
            global_pos = self.tbl_files.horizontalHeader().mapToGlobal(pos)
            menu.exec(global_pos)
        except Exception:
            pass

    def _selected_file_info(self) -> dict | None:
        """Return file info dict for selected row."""
        row = self.tbl_files.currentRow()
        if row < 0:
            return None
        item = self.tbl_files.item(row, 0)
        if not item:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _selected_files_info(self) -> list[dict]:
        """Return file info dicts for selected rows (fallback to current row)."""
        model = self.tbl_files.selectionModel()
        rows = {idx.row() for idx in model.selectedRows()} if model else set()
        if not rows and self.tbl_files.currentRow() >= 0:
            rows = {self.tbl_files.currentRow()}
        infos: list[dict] = []
        for row in sorted(rows):
            item = self.tbl_files.item(row, 0)
            if not item:
                continue
            info = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(info, dict):
                infos.append(info)
        return infos

    def _metadata_editor_choices(self) -> dict[str, list[str]]:
        try:
            return dict(be._build_validation_lists(self._files_data or []))
        except Exception:
            return {}

    def _selected_files_have_manual_override(self, infos: list[dict]) -> bool:
        for info in infos or []:
            if bool(info.get("has_manual_override")):
                return True
            source = str(info.get("metadata_source") or "").strip().lower()
            if source in {"manual_override", "mixed"}:
                return True
            manual_fields = str(info.get("manual_override_fields") or "").strip()
            if manual_fields:
                return True
        return False

    def _act_files_edit_metadata(self) -> None:
        infos = self._selected_files_info()
        if not infos:
            QtWidgets.QMessageBox.information(self, "Edit Metadata", "Select one or more files first.")
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            QtWidgets.QMessageBox.warning(self, "Edit Metadata", "No global repository selected.")
            return
        rel_paths = [str(i.get("rel_path") or "").strip() for i in infos if str(i.get("rel_path") or "").strip()]
        if not rel_paths:
            QtWidgets.QMessageBox.warning(self, "Edit Metadata", "Selected rows are missing file paths.")
            return

        dlg = MetadataBatchEditorDialog(infos, choices=self._metadata_editor_choices(), parent=self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        updates = dlg.field_updates()
        if not updates:
            QtWidgets.QMessageBox.information(self, "Edit Metadata", "No metadata fields were changed.")
            return
        repo = Path(repo_raw).expanduser()

        def _on_success(payload: dict):
            ok = int(payload.get("updated") or 0)
            failed = int(payload.get("failed") or 0)
            results = payload.get("results") or []
            index_error = payload.get("index_error")
            for r in results:
                if not r.get("ok") and r.get("error"):
                    self._append_log(f"[METADATA EDIT] FAILED: {r.get('rel_path')}: {r.get('error')}")
            if index_error:
                self._append_log(f"[METADATA EDIT] Index rebuild failed: {index_error}")
            if failed > 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Edit Metadata",
                    f"Updated {ok} file(s), but {failed} failed.\n\nSee Debug Console for details.",
                )
            else:
                self._show_toast(f"Updated metadata for {ok} file(s).")
            self._refresh_files_tab()
            return f"Metadata edit complete - ok={ok}, failed={failed}"

        self._start_manager_action(
            heading="Edit Metadata",
            status_text="Applying metadata edits",
            task=lambda: be.edit_metadata_for_files(repo, rel_paths, updates),
            on_success=_on_success,
        )

    def _act_files_open_pdf(self) -> None:
        """Open the source file (PDF or Excel)."""
        info = self._selected_file_info()
        if not info:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            return
        rel_path = info.get("rel_path", "")
        if not rel_path:
            return
        full_path = Path(repo_raw) / rel_path
        try:
            be.open_path(full_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open File", str(exc))

    def _act_files_open_metadata(self) -> None:
        """Open the metadata JSON file."""
        info = self._selected_file_info()
        if not info:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            return
        metadata_rel = info.get("metadata_rel", "")
        if not metadata_rel:
            QtWidgets.QMessageBox.warning(self, "Metadata", "No metadata file for this document")
            return
        full_path = be.eidat_support_dir(Path(repo_raw)) / metadata_rel
        try:
            be.open_path(full_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Metadata", str(exc))

    def _act_files_open_artifacts(self) -> None:
        """Open the artifacts folder."""
        info = self._selected_file_info()
        if not info:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            return
        rel_path = info.get("rel_path", "")
        if not rel_path:
            return
        artifacts = be.get_file_artifacts_path(Path(repo_raw), rel_path)
        if not artifacts or not artifacts.exists():
            QtWidgets.QMessageBox.warning(self, "Artifacts", "No artifacts folder found")
            return
        try:
            be.open_path(artifacts)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open Artifacts", str(exc))

    def _act_files_open_combined(self) -> None:
        """Open the combined.txt artifact for the selected file (if available)."""
        info = self._selected_file_info()
        if not info:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            return
        rel_path = info.get("rel_path", "")
        if not rel_path:
            return
        artifacts = be.get_file_artifacts_path(Path(repo_raw), rel_path)
        if not artifacts or not artifacts.exists():
            QtWidgets.QMessageBox.warning(self, "combined.txt", "No artifacts folder found")
            return
        combined = artifacts / "combined.txt"
        if not combined.exists():
            QtWidgets.QMessageBox.warning(self, "combined.txt", "combined.txt not found in artifacts folder")
            return
        try:
            be.open_path(combined)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open combined.txt", str(exc))

    def _act_files_show_explorer(self) -> None:
        """Show file in OS file explorer."""
        info = self._selected_file_info()
        if not info:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            return
        rel_path = info.get("rel_path", "")
        full_path = Path(repo_raw) / rel_path
        try:
            be.open_path(full_path.parent)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Show in Explorer", str(exc))

    def _act_files_update_metadata(self) -> None:
        """Refresh metadata JSON for selected files without re-OCR."""
        infos = self._selected_files_info()
        if not infos:
            QtWidgets.QMessageBox.information(self, "Update Metadata", "Select one or more files first.")
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            QtWidgets.QMessageBox.warning(self, "Update Metadata", "No global repository selected.")
            return
        rel_paths = [str(i.get("rel_path") or "").strip() for i in infos if str(i.get("rel_path") or "").strip()]
        if not rel_paths:
            QtWidgets.QMessageBox.warning(self, "Update Metadata", "Selected rows are missing file paths.")
            return
        repo = Path(repo_raw).expanduser()
        overwrite_manual_fields = False
        if self._selected_files_have_manual_override(infos):
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Update Metadata")
            msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg.setText("Some selected files contain manual metadata overrides.")
            msg.setInformativeText(
                "Choose whether metadata refresh is allowed to overwrite those manual fields."
            )
            overwrite_btn = msg.addButton("Overwrite", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            preserve_btn = msg.addButton("Do Not Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = msg.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(preserve_btn)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked is cancel_btn:
                return
            overwrite_manual_fields = clicked is overwrite_btn

        def _on_success(payload: dict):
            ok = int(payload.get("updated") or 0)
            failed = int(payload.get("failed") or 0)
            results = payload.get("results") or []
            index_error = payload.get("index_error")
            for r in results:
                if not r.get("ok") and r.get("error"):
                    self._append_log(f"[METADATA] FAILED: {r.get('rel_path')}: {r.get('error')}")
            if index_error:
                self._append_log(f"[METADATA] Index rebuild failed: {index_error}")
            if failed > 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Update Metadata",
                    f"Updated {ok} file(s), but {failed} failed.\n\nSee Debug Console for details.",
                )
            else:
                self._show_toast(f"Updated metadata for {ok} file(s).")
            self._refresh_files_tab()
            return f"Metadata update complete - ok={ok}, failed={failed}"

        self._start_manager_action(
            heading="Update Metadata",
            status_text="Updating metadata",
            task=lambda: be.refresh_metadata_only(repo, rel_paths, overwrite_manual_fields=overwrite_manual_fields),
            on_success=_on_success,
        )

    def _act_files_recertify(self) -> None:
        """Re-run certification analysis on selected file."""
        info = self._selected_file_info()
        if not info:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            return
        artifacts_rel = info.get("artifacts_rel", "")
        if not artifacts_rel:
            QtWidgets.QMessageBox.warning(
                self, "Certification",
                "No artifacts folder found for this document.\n"
                "The document may not have been processed yet."
            )
            return
        try:
            result = be.analyze_and_certify_document(Path(repo_raw), artifacts_rel)
            status = result.get("status", "ERROR")
            if status == "ERROR":
                QtWidgets.QMessageBox.warning(
                    self, "Certification Error",
                    result.get("error", "Unknown error during certification")
                )
                return

            msg = (
                f"Certification Status: {status}\n\n"
                f"Passed: {result.get('passed_tests', 0)}\n"
                f"Failed: {result.get('failed_tests', 0)}\n"
                f"Pending: {result.get('pending_tests', 0)}"
            )
            failed_terms = result.get("failed_terms", [])
            if failed_terms:
                msg += f"\n\nFailed Tests:\n  - " + "\n  - ".join(failed_terms[:10])
                if len(failed_terms) > 10:
                    msg += f"\n  ... and {len(failed_terms) - 10} more"

            QtWidgets.QMessageBox.information(self, "Certification Result", msg)
            # Refresh the files tab to show updated certification status
            self._refresh_files_tab()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Certification Error", str(exc))

    def _act_files_certify_all(self) -> None:
        """Run certification analysis on all processed documents."""
        repo_raw = (self.ed_global_repo.text() or "").strip()
        if not repo_raw:
            QtWidgets.QMessageBox.warning(self, "Certify All", "No global repository selected.")
            return

        try:
            result = be.analyze_and_certify_all(Path(repo_raw))
            if "error" in result:
                QtWidgets.QMessageBox.warning(
                    self, "Certification Error",
                    result.get("error", "Unknown error")
                )
                return

            msg = (
                f"Certification Complete\n\n"
                f"Analyzed: {result.get('analyzed_count', 0)} documents\n"
                f"Certified: {result.get('certified_count', 0)}\n"
                f"Failed: {result.get('failed_count', 0)}\n"
                f"Pending: {result.get('pending_count', 0)}"
            )
            QtWidgets.QMessageBox.information(self, "Certify All", msg)

            # Rebuild index to sync certification status
            self._append_log("Rebuilding index to sync certification status...")
            be.eidat_manager_index(Path(repo_raw))

            # Refresh the files tab
            self._refresh_files_tab()
            self._append_log("Certification complete and index rebuilt.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Certification Error", str(exc))

    def _files_context_menu(self, pos) -> None:
        """Show context menu for files table."""
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                padding: 4px;
            }
            QMenu::item {
                background: transparent;
                color: #1f2937;
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background: #dbeafe;
                color: #1f2937;
            }
            QMenu::separator {
                height: 1px;
                background: #e5e7eb;
                margin: 4px 8px;
            }
        """)

        act_open_pdf = menu.addAction("Open File")
        act_open_metadata = menu.addAction("Open Metadata JSON")
        act_open_artifacts = menu.addAction("Open Artifacts Folder")
        act_open_combined = menu.addAction("Open combined.txt")
        menu.addSeparator()
        act_show_explorer = menu.addAction("Show in Explorer")
        act_edit_metadata = menu.addAction("Edit Metadata")
        act_update_metadata = menu.addAction("Update Metadata Only")
        menu.addSeparator()
        act_recertify = menu.addAction("Re-analyze Certification")

        act_open_pdf.triggered.connect(self._act_files_open_pdf)
        act_open_metadata.triggered.connect(self._act_files_open_metadata)
        act_open_artifacts.triggered.connect(self._act_files_open_artifacts)
        act_open_combined.triggered.connect(self._act_files_open_combined)
        act_show_explorer.triggered.connect(self._act_files_show_explorer)
        act_edit_metadata.triggered.connect(self._act_files_edit_metadata)
        act_update_metadata.triggered.connect(self._act_files_update_metadata)
        act_recertify.triggered.connect(self._act_files_recertify)

        menu.exec(self.tbl_files.mapToGlobal(pos))

    def _append_log(self, text: str):
        self.log.appendPlainText(text)
        if not self.log.isVisible():
            try:
                label = self.btn_toggle_log.text()
                if "\u2022" not in label and "•" not in label:
                    self.btn_toggle_log.setText("Show Debug Panel •")
            except Exception:
                pass
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_worker_line(self, text: str):
        self._append_log(text)
        self._maybe_update_run_progress(text)
        self._maybe_update_run_file(text)
        self._maybe_track_run_dir(text)

    def _maybe_update_run_progress(self, text: str):
        if "[PROGRESS] Terms" not in text:
            return
        match = self._progress_pattern.search(text)
        if not match:
            return
        try:
            completed = int(match.group(2))
            total = int(match.group(3))
            # Group 4 is optional (for backward compatibility with old format)
            found_str = match.group(4)
            found = int(found_str) if found_str is not None else 0
        except Exception:
            return
        self._progress_total = max(total, 0)
        self._progress_completed = max(0, min(completed, self._progress_total or completed))
        self._progress_found = max(0, found)
        self._update_progress_widgets()

    def _maybe_update_run_file(self, text: str):
        """Track which EIDP/PDF is currently being processed based on stdout."""
        if not self._progress_popup_active:
            return
        # Excel conversion emits one line per workbook.
        if text.strip().startswith("[EXCEL]"):
            try:
                current_file = text.split("[EXCEL]", 1)[1].strip()
            except Exception:
                current_file = ""
            if current_file:
                self._progress_file_index = max(1, self._progress_file_index + 1)
                self._progress_file_name = current_file
                self._update_progress_widgets()
            return
        # Missing-terms batch helper script emits a higher-level INFO line per EIDP.
        if "[INFO] Starting missing-terms extraction for serial" in text:
            self._is_missing_terms_batch = True
            try:
                # Example: "[INFO] Starting missing-terms extraction for serial ABC123 (file.pdf)"
                if "(" in text and ")" in text:
                    file_part = text.split("(", 1)[1].rsplit(")", 1)[0]
                    current_file = file_part.strip()
                else:
                    current_file = text.rsplit(" ", 1)[-1].strip()
            except Exception:
                current_file = ""
            if current_file:
                self._progress_file_index = max(1, self._progress_file_index + 1)
                self._progress_file_name = current_file
                self._update_progress_widgets()
            return
        # Standard scanner runs (full folder / selected PDFs) emit:
        # "[INFO] Scanning: <file>  [Data: <label>]"
        if "[INFO] Scanning:" in text and not self._is_missing_terms_batch:
            try:
                _, tail = text.split("Scanning:", 1)
                # Strip optional "[Data: ...]" suffix.
                before_data = tail.split("[Data:", 1)[0]
                current_file = before_data.strip()
            except Exception:
                current_file = ""
            if current_file:
                self._progress_file_index = max(1, self._progress_file_index + 1)
                self._progress_file_name = current_file
                self._update_progress_widgets()

    def _maybe_track_run_dir(self, text: str):
        if "Outputs will be saved under:" not in text:
            return
        try:
            _, tail = text.split("Outputs will be saved under:", 1)
        except ValueError:
            return
        candidate = tail.strip().strip('"')
        if not candidate:
            return
        path = Path(candidate)
        if not path.is_absolute():
            path = Path(be.ROOT) / path
        self._last_run_dir = path

    def _on_progress_canceled(self):
        if not self._progress_popup_active or self._progress_was_canceled:
            return
        self._progress_was_canceled = True
        self._enrich_after_run = False
        try:
            self.lbl_ready.setText("Stopping run...")
        except Exception:
            pass
        self._append_log("[GUI] User requested run abort.")
        self._progress_dialog.lbl_status.setText("Stopping run...")
        self._act_stop_scan()

    def _update_progress_widgets(self):
        if not self._progress_popup_active:
            return
        total = self._progress_total
        completed = min(self._progress_completed, total if total else self._progress_completed)
        found = self._progress_found
        self._progress_dialog.update_progress(
            completed,
            total,
            found,
            current_file=self._progress_file_name,
            file_index=self._progress_file_index,
            file_total=self._progress_file_total,
        )

    def _finalize_run_progress(self, success: bool):
        if not self._progress_popup_active:
            return
        if self._progress_total and self._progress_completed < self._progress_total:
            self._progress_completed = self._progress_total
        self._update_progress_widgets()
        final_success = success and not self._progress_was_canceled
        if self._progress_was_canceled:
            message = "Run aborted"
        elif final_success:
            message = "Run complete"
        else:
            message = "Run finished with errors"
        self._progress_dialog.finish(message, success=final_success)
        self._progress_popup_active = False
        self._progress_was_canceled = False
        if final_success:
            self._last_run_dir = None

    def _cleanup_last_run_dir(self):
        path = self._last_run_dir
        if not path:
            return
        try:
            runs_root = Path(be.RUNS_DIR).resolve()
        except Exception:
            runs_root = Path(be.RUNS_DIR)
        try:
            target = path.resolve()
        except Exception:
            target = path
        if runs_root not in target.parents and target != runs_root:
            self._last_run_dir = None
            return
        try:
            shutil.rmtree(target, ignore_errors=True)
        finally:
            self._last_run_dir = None

    def _start_worker(
        self,
        popen_factory,
        *,
        status_msg: str,
        show_run_progress: bool = False,
        total_files: int | None = None,
    ):
        if self._worker is not None and self._worker.isRunning():
            return
        self._append_log(f"[GUI] {status_msg}")
        self.status_bar.showMessage(status_msg)
        self._progress_total = 0
        self._progress_completed = 0
        self._progress_found = 0
        self._progress_file_total = max(int(total_files or 0), 0)
        self._progress_file_index = 0
        self._progress_file_name = None
        self._is_missing_terms_batch = False
        self._progress_popup_active = show_run_progress
        self._progress_was_canceled = False
        self._last_run_dir = None
        # Track if this is an extraction run (show_run_progress indicates EIDP extraction)
        self._is_extraction_run = show_run_progress
        if show_run_progress:
            self._progress_dialog.begin(status_msg)
        else:
            self._progress_dialog.abort()
        self._worker = ProcWorker(popen_factory)
        self._worker.line.connect(self._on_worker_line)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _toggle_log_panel(self):
        try:
            visible = bool(self.btn_toggle_log.isChecked())
        except Exception:
            visible = True
        self.log_container.setVisible(visible)
        if visible:
            try:
                total = max(1, self.main_splitter.height())
                lower = min(320, max(160, int(total * 0.3)))
                upper = max(1, total - lower)
                self.main_splitter.setSizes([upper, lower])
                self.log_container.raise_()
            except Exception:
                pass
        else:
            try:
                self.main_splitter.setSizes([1, 0])
            except Exception:
                pass
        try:
            # Update arrow direction based on visibility
            self.btn_toggle_log.setText("\u25BC  Debug Console" if visible else "\u25B6  Debug Console")
        except Exception:
            pass

    def _prepare_dialog(self, dlg: QtWidgets.QDialog) -> None:
        try:
            explicit_size = dlg.size()
            dlg.adjustSize()
            if explicit_size.isValid() and explicit_size.width() > 0 and explicit_size.height() > 0:
                dlg.resize(explicit_size)
        except Exception:
            pass
        _fit_widget_to_screen(dlg)

    def _create_scroll_dialog(self, title: str, width: int, height: int) -> tuple[QtWidgets.QDialog, QtWidgets.QVBoxLayout]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(width, height)
        outer = QtWidgets.QVBoxLayout(dlg)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return dlg, body_layout

    def _on_worker_done(self, rc: int):
        self._worker = None
        self.status_bar.showMessage("Ready.", 3000)
        self._append_log(f"[INFO] Process finished with code {rc}")
        was_canceled = self._progress_was_canceled
        is_extraction = getattr(self, "_is_extraction_run", False)
        self._is_missing_terms_batch = False
        self._finalize_run_progress(success=(rc == 0))

        # Show toast notification only for extraction runs
        if is_extraction:
            if was_canceled:
                self._show_toast("Extraction run aborted")
            elif rc == 0:
                self._show_toast("Extraction completed successfully")
            else:
                self._show_toast(f"Extraction failed with code {rc}")

        if was_canceled:
            self._cleanup_last_run_dir()

        # Refresh UI after a scan; no post-run enrichment step
        if getattr(self, "_enrich_after_run", False):
            self._enrich_after_run = False
            try:
                be.rebuild_registry_from_run_data()
            except Exception as exc:
                self._append_log(f"[WARN] Registry rebuild failed: {exc}")
            try:
                self._refresh_run_registry()
            except Exception as exc:
                self._append_log(f"[WARN] Registry refresh failed: {exc}")
        # Update master database after successful extraction run
        if is_extraction and rc == 0 and not was_canceled:
            try:
                self._append_log("[GUI] Compiling master database after extraction run...")
                be.compile_master()
                self._show_toast("Master database updated")
            except Exception as e:
                self._append_log(f"[ERROR] Failed to compile master database: {e}")

        self._scan_refresh()
    def _act_install(self):
        self._start_worker(be.run_install_full, status_msg="Installing environment...")

    def _act_check_env(self):
        self._start_worker(be.check_environment, status_msg="Checking environment...")

    def _act_open_env(self):
        try:
            be.SCANNER_ENV_LOCAL.parent.mkdir(parents=True, exist_ok=True)
            if not be.SCANNER_ENV_LOCAL.exists():
                be.save_scanner_env({"QUIET": "1"})
            be.open_path(be.SCANNER_ENV_LOCAL)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(e))

    def _open_terms_editor(self):
        raw = (self.ed_terms.text() or "").strip()
        try:
            target = be.resolve_terms_path(Path(raw).expanduser()) if raw else be.resolve_terms_path()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Terms file", str(exc))
            return
        if not target.exists():
            resp = QtWidgets.QMessageBox.question(
                self,
                "Terms spreadsheet missing",
                f"{target} was not found.\nGenerate a fresh simple schema template now?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if resp == QtWidgets.QMessageBox.StandardButton.Yes:
                self._act_generate_terms()
            return
        try:
            dlg = TermsEditorDialog(target, self)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Unable to open editor", str(exc))
            return
        self._prepare_dialog(dlg)
        result = dlg.exec()
        # If dialog was accepted (saved successfully), show toast and sync
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            self._show_toast("Simple schema terms saved successfully")
            # Auto-sync workspace after successful save
            QtCore.QTimer.singleShot(500, lambda: self._start_workspace_sync(auto=False, show_popup=False, heading="Workspace Sync"))

    def _act_generate_terms(self):
        raw = (self.ed_terms.text() or "").strip()
        try:
            target = be.resolve_terms_path(Path(raw).expanduser()) if raw else be.resolve_terms_path()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Terms file", str(exc))
            return
        if target.exists():
            msg = (
                f"A simple schema workbook already exists:\n\n{target}\n\n"
                "Generating a new template will overwrite it.\n\nContinue?"
            )
            if (
                QtWidgets.QMessageBox.question(
                    self, "Overwrite simple schema workbook?", msg
                )
                != QtWidgets.QMessageBox.StandardButton.Yes
            ):
                return
        self._start_worker(be.generate_terms, status_msg="Generating simple schema spreadsheet...")

    def _show_extraction_options(self):
        """Show dialog with extraction options: out-of-date only, force all, or choose files."""
        dlg, layout = self._create_scroll_dialog("Extraction Options", 500, 280)
        dlg.setStyleSheet("""
            QDialog {
                background: #ffffff;
            }
            QPushButton {
                padding: 12px 20px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
        """)

        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title and description
        title = QtWidgets.QLabel("Choose Extraction Method")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827;")
        layout.addWidget(title)

        desc = QtWidgets.QLabel("Select how you want to process your documents:")
        desc.setStyleSheet("font-size: 13px; color: #6b7280; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Option 1: Extract out-of-date only
        btn_outdated = QtWidgets.QPushButton("\u26A1  Extract All Out-of-Date")
        btn_outdated.setStyleSheet("""
            QPushButton {
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                text-align: left;
                padding-left: 16px;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
        """)
        btn_outdated.setToolTip("Process only files marked as 'New' or 'Out-of-Date'")

        # Option 2: Force extract all
        btn_force = QtWidgets.QPushButton("\U0001F504  Force Extract All")
        btn_force.setStyleSheet("""
            QPushButton {
                background: #dc2626;
                color: #ffffff;
                border: 1px solid #dc2626;
                text-align: left;
                padding-left: 16px;
            }
            QPushButton:hover {
                background: #b91c1c;
            }
        """)
        btn_force.setToolTip("Re-run extraction on the entire database (may take longer)")

        # Option 3: Choose specific files
        btn_choose = QtWidgets.QPushButton("\U0001F4CB  Choose Files to Extract")
        btn_choose.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                text-align: left;
                padding-left: 16px;
            }
            QPushButton:hover {
                background: #f9fafb;
                border-color: #9ca3af;
            }
        """)
        btn_choose.setToolTip("Manually select which files to process")

        layout.addWidget(btn_outdated)
        layout.addWidget(btn_force)
        layout.addWidget(btn_choose)

        # Cancel button
        layout.addStretch()
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #6b7280;
                border: 1px solid #d1d5db;
            }
            QPushButton:hover {
                background: #f9fafb;
            }
        """)
        layout.addWidget(btn_cancel)

        # Connect buttons
        def extract_outdated():
            dlg.accept()
            # Get all out-of-date files (new or pdf_newer or terms_newer)
            details = getattr(self, "_sync_details", None) or []
            rows = [d for d in details if d.get("reason") in ("new", "pdf_newer", "terms_newer")]
            entries = [(Path(d.get("pdf")), d.get("serial_component", "")) for d in rows if d.get("pdf")]
            if not entries:
                QtWidgets.QMessageBox.information(self, "Nothing to extract", "No out-of-date files found. Run 'Sync Workspace Now' first.")
                return
            try:
                raw_terms = (self.ed_terms.text() or "").strip()
                terms = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
            except Exception:
                terms = be.resolve_terms_path()
            self._prompt_outdated_run_mode(
                entries,
                terms,
                title="Out-of-Date Extraction",
                description="Process all out-of-date EIDPs or limit the run to only the schema terms that are missing.",
                full_status="Extracting out-of-date files...",
                missing_status="Extracting missing terms for out-of-date files...",
            )

        def force_extract_all():
            dlg.accept()
            # Force run on all PDFs
            self._act_start_scan()

        def choose_files():
            dlg.accept()
            # Show the file selection dialog
            self._show_outdated_popup()

        btn_outdated.clicked.connect(extract_outdated)
        btn_force.clicked.connect(force_extract_all)
        btn_choose.clicked.connect(choose_files)
        btn_cancel.clicked.connect(dlg.reject)

        self._prepare_dialog(dlg)
        dlg.exec()

    def _prompt_outdated_run_mode(
        self,
        entries: list[tuple[Path, str]],
        terms: Path,
        *,
        title: str,
        description: str,
        full_status: str,
        missing_status: str,
    ) -> bool:
        if not entries:
            QtWidgets.QMessageBox.information(self, "Nothing to run", "No data packages selected.")
            return False
        dlg, layout = self._create_scroll_dialog(title, 420, 220)
        dlg.setStyleSheet("""
            QDialog { background: #ffffff; }
            QPushButton {
                padding: 10px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
        """)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        desc_lbl = QtWidgets.QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #4b5563; font-size: 13px;")
        layout.addWidget(desc_lbl)

        btn_missing = QtWidgets.QPushButton("\u26A1  Just Missing Terms")
        btn_missing.setStyleSheet("""
            QPushButton {
                background: #0ea5e9;
                color: #ffffff;
                border: 1px solid #0ea5e9;
                text-align: left;
                padding-left: 16px;
            }
            QPushButton:hover { background: #0284c7; }
        """)
        btn_full = QtWidgets.QPushButton("\U0001F504  Re-extract All Terms")
        btn_full.setStyleSheet("""
            QPushButton {
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
                text-align: left;
                padding-left: 16px;
            }
            QPushButton:hover { background: #1d4ed8; }
        """)
        layout.addWidget(btn_missing)
        layout.addWidget(btn_full)
        layout.addStretch()
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #6b7280;
                border: 1px solid #d1d5db;
            }
            QPushButton:hover { background: #f9fafb; }
        """)
        layout.addWidget(btn_cancel)

        result = {"run": False}

        def _run_missing():
            serials = sorted({serial.strip() for _, serial in entries if serial and serial.strip()})
            if not serials:
                QtWidgets.QMessageBox.information(dlg, "Missing serials", "Serial numbers are required to target missing terms.")
                return
            try:
                missing_count = be.count_missing_terms(serials, terms)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dlg, "Missing terms", str(exc))
                return
            if missing_count <= 0:
                QtWidgets.QMessageBox.information(dlg, "No Missing Terms", "All selected EIDPs already contain every schema term.")
                return
            self._enrich_after_run = True
            total_eidps = len(serials)
            self._start_worker(
                lambda entries=entries, terms=terms: be.run_missing_terms_for_selected_pdfs(entries, terms),
                status_msg=missing_status,
                show_run_progress=True,
                total_files=total_eidps,
            )
            result["run"] = True
            dlg.accept()

        def _run_full():
            paths = [p for p, _ in entries]
            if not paths:
                QtWidgets.QMessageBox.information(dlg, "Nothing to run", "No valid PDF paths were found.")
                return
            self._enrich_after_run = True
            total_eidps = len(paths)
            self._start_worker(
                lambda paths=paths, terms=terms: be.run_selected_pdfs(paths, terms),
                status_msg=full_status,
                show_run_progress=True,
                total_files=total_eidps,
            )
            result["run"] = True
            dlg.accept()

        btn_missing.clicked.connect(_run_missing)
        btn_full.clicked.connect(_run_full)
        btn_cancel.clicked.connect(dlg.reject)
        self._prepare_dialog(dlg)
        dlg.exec()
        return result["run"]

    def _act_start_scan(self):
        try:
            raw_terms = (self.ed_terms.text() or "").strip()
            terms = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Terms file", str(exc))
            return
        try:
            pdfs = be.resolve_pdf_root(Path(self.ed_pdfs.text()).expanduser())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "PDFs folder", str(exc))
            return
        if not terms.exists():
            QtWidgets.QMessageBox.critical(self, "Missing terms", f"Terms file not found:\n{terms}")
            return
        if not pdfs.exists():
            QtWidgets.QMessageBox.critical(self, "Missing PDFs folder", f"PDFs folder not found:\n{pdfs}")
            return
        # Only enrich registry after a scan completes
        self._enrich_after_run = True
        try:
            total_files = sum(1 for _ in pdfs.rglob("*.pdf"))
        except Exception:
            total_files = 0
        self._start_worker(
            lambda: be.run_scanner(terms, pdfs),
            status_msg="Running simple extraction...",
            show_run_progress=True,
            total_files=total_files,
        )

    def _act_start_excel_scan(self):
        """Run Excel data extraction on all Excel files in the data folder."""
        config = be.DEFAULT_EXCEL_TREND_CONFIG
        if not config.exists():
            QtWidgets.QMessageBox.critical(
                self, "Missing config",
                f"Excel trend config not found:\n{config}\n\n"
                "Create user_inputs/excel_trend_config.json first.")
            return
        try:
            data_dir = be.resolve_pdf_root(Path(self.ed_pdfs.text()).expanduser())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Data folder", str(exc))
            return
        if not data_dir.exists():
            QtWidgets.QMessageBox.critical(self, "Missing data folder", f"Data folder not found:\n{data_dir}")
            return
        self._enrich_after_run = True
        try:
            total_files = sum(
                1
                for p in data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in be.EXCEL_EXTENSIONS and not p.name.startswith("~$")
            )
        except Exception:
            total_files = 0
        if total_files == 0:
            QtWidgets.QMessageBox.information(
                self, "No Excel files",
                f"No Excel data files found under:\n{data_dir}")
            return
        self._start_worker(
            lambda: be.run_excel_scanner(data_dir),
            status_msg="Extracting Excel data...",
            show_run_progress=True,
            total_files=total_files,
        )

    def _act_stop_scan(self):
        try:
            if self._worker and self._worker.isRunning():
                self._worker.stop()
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[override]
        """Ensure background worker threads are stopped before closing the app."""
        worker = getattr(self, "_worker", None)
        if worker and worker.isRunning():
            try:
                self._append_log("[GUI] Stopping background task before exit...")
            except Exception:
                pass
            try:
                worker.stop()
            except Exception:
                pass
            try:
                worker.wait(3000)
            except Exception:
                pass
            if worker.isRunning():
                # Last-resort terminate to avoid Qt warning on shutdown.
                try:
                    worker.terminate()
                    worker.wait(1000)
                except Exception:
                    pass
        self._worker = None
        sync_worker = getattr(self, "_sync_worker", None)
        if sync_worker and sync_worker.isRunning():
            try:
                sync_worker.terminate()
                sync_worker.wait(1000)
            except Exception:
                pass
        self._sync_worker = None
        manager_worker = getattr(self, "_manager_worker", None)
        if manager_worker and manager_worker.isRunning():
            try:
                manager_worker.terminate()
                manager_worker.wait(1000)
            except Exception:
                pass
        self._manager_worker = None
        project_worker = getattr(self, "_project_worker", None)
        if project_worker and project_worker.isRunning():
            try:
                project_worker.terminate()
                project_worker.wait(1000)
            except Exception:
                pass
        self._project_worker = None
        try:
            self._progress_dialog.abort()
        except Exception:
            pass
        try:
            self._repo_scan_dialog.hide()
        except Exception:
            pass
        super().closeEvent(event)

    def _act_compile_master(self):
        self._start_worker(be.compile_master, status_msg="Compiling master workbook...")


    # NOTE: _act_generate_plot_terms, _act_generate_plots, _act_export_plot_summary were removed
    # as they were only called from the unused Plot tab UI

    def _get_registry_table_data(self) -> tuple[list[str], list[list[str]]]:
        try:
            p_csv = be.registry_csv_for_read()
            p_xlsx = be.registry_xlsx_for_read()
            rows: list[list[str]] = []
            headers: list[str] = []
            # Keep registry tidy before reading; best-effort and safe (prunes invalid rows)
            try:
                be.ensure_run_registry_consistent()
            except Exception:
                pass
            if p_csv.exists():
                import csv
                with open(p_csv, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for i, r in enumerate(reader):
                        if i == 0:
                            headers = [str(x) for x in r]
                        else:
                            rows.append([str(x) for x in r])
            elif p_xlsx.exists():
                # Convert xlsx -> csv once via backend and retry
                try:
                    be.ensure_run_registry_consistent()
                except Exception:
                    pass
                if p_csv.exists():
                    import csv
                    with open(p_csv, newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        for i, r in enumerate(reader):
                            if i == 0:
                                headers = [str(x) for x in r]
                            else:
                                rows.append([str(x) for x in r])
            else:
                headers = ["Run Registry"]
                rows = [["No registry found. Run a scan to create it."]]

            # Show file as-is (no derived columns) for a clean, authoritative view

            return headers, rows
        except Exception:
            return [], []

    def _act_view_registry(self):
        """Refresh cache and show the run registry dialog."""
        try:
            self._refresh_run_registry()
        except Exception:
            pass
        self._show_registry_popup()

    def _act_clear_old_runs(self):
        try:
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Confirm cleanup",
                "Delete old run_data_simple folders not referenced by the current registry?",
            )
            if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            # Ensure registry is consistent before cleanup
            try:
                be.ensure_run_registry_consistent()
            except Exception:
                pass
            deleted, kept = be.clear_stale_run_data()
            QtWidgets.QMessageBox.information(
                self,
                "Cleanup complete",
                f"Removed {deleted} old run folder(s). Kept {kept} current folder(s).",
            )
            # Refresh internal cache after cleanup
            self._refresh_run_registry()
        except Exception as e:
            QtWidgets.QMessageBox.information(self, "Cleanup", str(e))

    def _refresh_run_registry(self):
        try:
            self._registry_cache = self._get_registry_table_data()
        except Exception:
            self._registry_cache = ([], [])

    def _show_registry_popup(self):
        # Guard against duplicate dialogs
        if getattr(self, "_dlg_open_registry", False):
            return
        self._dlg_open_registry = True
        try:
            cache = getattr(self, "_registry_cache", None)
            if not cache:
                cache = self._get_registry_table_data()
            headers, rows = cache
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Run Registry")
            dlg.resize(900, 500)
            dlg.setObjectName("runRegistryDialog")
            dlg.setStyleSheet("""
                #runRegistryDialog {
                    background-color: #f8fafc;
                    color: #0f172a;
                }
                #runRegistryDialog QLabel {
                    color: #0f172a;
                }
                #runRegistryDialog QTableWidget {
                    background-color: #ffffff;
                    border: 1px solid #d4d4d8;
                    gridline-color: #e4e4e7;
                    selection-background-color: #d1d5db;
                    selection-color: #0f172a;
                    color: #0f172a;
                }
                #runRegistryDialog QTableWidget::item {
                    color: #0f172a;
                }
                #runRegistryDialog QTableWidget::item:selected {
                    background-color: #d1d5db;
                    color: #0f172a;
                }
                #runRegistryDialog QHeaderView::section {
                    background-color: #eef2ff;
                    color: #0f172a;
                    padding: 8px;
                    border: 1px solid #cbd5f5;
                    font-weight: 600;
                }
                #runRegistryDialog QPushButton {
                    background-color: #ffffff;
                    color: #0f172a;
                    border: 1px solid #cbd5f5;
                    border-radius: 6px;
                    padding: 8px 14px;
                }
                #runRegistryDialog QPushButton:hover {
                    background-color: #f1f5f9;
                }
            """)
            v = QtWidgets.QVBoxLayout(dlg)
            tbl = QtWidgets.QTableWidget(0, len(headers))
            tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
            tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            tbl.setAlternatingRowColors(False)
            if headers:
                tbl.setHorizontalHeaderLabels(headers)
            for r, row in enumerate(rows):
                tbl.insertRow(r)
                for c, val in enumerate(row[: len(headers) or len(row)]):
                    tbl.setItem(r, c, QtWidgets.QTableWidgetItem(val))
            tbl.resizeColumnsToContents()
            v.addWidget(tbl)
            # Controls: Delete Selected and Close
            bar = QtWidgets.QHBoxLayout()
            btn_save = QtWidgets.QPushButton("Save Changes")
            btn_delete = QtWidgets.QPushButton("Delete Selected")
            btn_close = QtWidgets.QPushButton("Close")
            bar.addStretch(1)
            bar.addWidget(btn_save)
            bar.addWidget(btn_delete)
            bar.addWidget(btn_close)
            v.addLayout(bar)

            def _delete_selected():
                if not headers:
                    return
                hdr_lc = [h.strip().lower() for h in headers]
                try:
                    idx_sc = hdr_lc.index("serial_component") if "serial_component" in hdr_lc else (
                        hdr_lc.index("serial") if "serial" in hdr_lc else -1
                    )
                except Exception:
                    idx_sc = -1
                if idx_sc < 0:
                    QtWidgets.QMessageBox.information(dlg, "Delete", "Cannot locate 'serial_component' column.")
                    return
                sels = tbl.selectionModel().selectedRows()
                if not sels:
                    QtWidgets.QMessageBox.information(dlg, "Delete", "Select one or more rows to delete.")
                    return
                serials: list[str] = []
                for mi in sels:
                    it = tbl.item(mi.row(), idx_sc)
                    if it and it.text().strip():
                        serials.append(it.text().strip())
                if not serials:
                    return
                confirm = QtWidgets.QMessageBox.question(
                    dlg,
                    "Confirm deletion",
                    f"Delete {len(serials)} entr(ies) and their run_data_simple folders?",
                )
                if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                try:
                    be.delete_registry_entries(serials)
                except Exception as e:
                    QtWidgets.QMessageBox.information(dlg, "Delete", str(e))
                try:
                    self._start_worker(be.compile_master, status_msg="Compiling master workbook...")
                except Exception:
                    pass
                dlg.accept()

            def _save_changes():
                if not headers:
                    return
                rows_out: list[dict[str, str]] = []
                for r in range(tbl.rowCount()):
                    row_map: dict[str, str] = {}
                    for c in range(len(headers)):
                        val = tbl.item(r, c).text() if tbl.item(r, c) else ""
                        row_map[headers[c]] = val
                    rows_out.append(row_map)
                try:
                    be.write_run_registry_rows(rows_out)
                except Exception as e:
                    QtWidgets.QMessageBox.information(dlg, "Save", str(e))
                    return
                try:
                    self._start_worker(be.compile_master, status_msg="Compiling master workbook...")
                except Exception:
                    pass
                dlg.accept()

            btn_save.clicked.connect(_save_changes)
            btn_delete.clicked.connect(_delete_selected)
            btn_close.clicked.connect(dlg.reject)
            self._prepare_dialog(dlg)
            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.information(self, "Registry", str(e))
        finally:
            self._dlg_open_registry = False

    def _safe_open(self, fn):
        try:
            fn()
        except Exception as e:
            QtWidgets.QMessageBox.information(self, "Open", str(e))

    # File/browser helpers
    def _browse_file(self, edit: QtWidgets.QLineEdit, initial_dir: Path, filter_str: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", str(initial_dir), filter_str)
        if path:
            if edit is getattr(self, "ed_terms", None):
                try:
                    be.resolve_terms_path(Path(path))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Terms file", str(exc))
                    return
            edit.setText(path)
        self._scan_refresh()

    def _browse_folder(self, edit: QtWidgets.QLineEdit, initial_dir: Path):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", str(initial_dir))
        if path:
            # Persist repository root when editing the repo picker
            if edit is getattr(self, "ed_repo", None) or edit is getattr(self, "ed_global_repo", None):
                try:
                    self._set_global_repo(Path(path))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Repository root", str(exc))
                    return
            edit.setText(path)
        self._scan_refresh()

    def _set_global_repo(self, path: Path) -> None:
        repo_path = Path(path).expanduser()
        be.set_repo_root(repo_path)
        try:
            payload = be.eidat_manager_init(repo_path)
            support_dir = payload.get("support_dir") or ""
            self.lbl_support_status.setText(f"Support: {support_dir}" if support_dir else "Support: initialized")
        except Exception as exc:
            self.lbl_support_status.setText(f"Support: init failed ({exc})")
        try:
            if hasattr(self, "ed_repo") and self.ed_repo is not None:
                self.ed_repo.setText(str(be.get_repo_root()))
        except Exception:
            pass
        # Node mode: auto-scan on launch so Files tab populates immediately (even if nothing is processed yet).
        try:
            if str(os.environ.get("EIDAT_UI_PROFILE") or "").strip().lower() == "node":
                self._act_manager_scan_all(auto=True, auto_process=False)
        except Exception:
            pass

    def _act_global_repo_changed(self) -> None:
        try:
            raw = (self.ed_global_repo.text() or "").strip()
            if not raw:
                return
            self._set_global_repo(Path(raw))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Global repository", str(exc))

    def _start_manager_action(
        self,
        *,
        heading: str,
        status_text: str,
        task,
        on_success,
        auto: bool = False,
        show_popup: bool = True,
    ) -> None:
        if self._manager_worker is not None and self._manager_worker.isRunning():
            if not auto:
                self._show_toast("EIDAT Manager task already running")
            return
        if getattr(self, "_sync_worker", None) is not None and self._sync_worker.isRunning():
            if not auto:
                self._show_toast("Workspace sync already running")
            return
        if getattr(self, "_sync_popup_active", False) and not auto:
            self._show_toast("Workspace sync in progress")
            return
        self._manager_popup_active = show_popup
        if show_popup:
            self._repo_scan_dialog.begin(status_text, heading=heading)
        else:
            try:
                self._repo_scan_dialog.hide()
            except Exception:
                pass
        self._manager_worker = ManagerTaskWorker(task, parent=self)
        self._manager_worker.completed.connect(lambda payload: self._on_manager_action_done(payload, on_success, heading, auto))
        self._manager_worker.failed.connect(lambda message: self._on_manager_action_error(message, heading, auto))
        self._manager_worker.start()

    def _on_manager_action_done(self, payload: dict, on_success, heading: str, auto: bool) -> None:
        self._manager_worker = None
        msg = None
        try:
            msg = on_success(payload)
        except Exception as exc:
            if not auto:
                QtWidgets.QMessageBox.warning(self, heading, str(exc))
            if self._manager_popup_active:
                self._repo_scan_dialog.finish(f"{heading} failed: {exc}", success=False)
                self._manager_popup_active = False
            return
        if self._manager_popup_active:
            self._repo_scan_dialog.finish(msg or f"{heading} complete", success=True)
            self._manager_popup_active = False

    def _on_manager_action_error(self, message: str, heading: str, auto: bool) -> None:
        self._manager_worker = None
        if not auto:
            QtWidgets.QMessageBox.warning(self, heading, message)
        if self._manager_popup_active:
            self._repo_scan_dialog.finish(f"{heading} failed: {message}", success=False)
            self._manager_popup_active = False

    def _background_processing_enabled(self) -> bool:
        try:
            env = be.load_scanner_env()
            raw = str(env.get("force_background_processes") or "").strip().lower()
            return raw in ("1", "true", "yes", "on")
        except Exception:
            return False

    def _auto_manager_tick(self) -> None:
        if not self._background_processing_enabled():
            return
        self._act_manager_scan_all(auto=True, auto_process=True)

    def _act_manager_scan_all(self, *, auto: bool = False, auto_process: bool = False) -> None:
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first.")
            repo = Path(repo_raw).expanduser()
        except Exception as exc:
            if not auto:
                QtWidgets.QMessageBox.warning(self, "EIDAT Manager", str(exc))
            return

        def _on_success(payload: dict):
            candidates = int(payload.get("candidates_count") or 0)
            file_count = int(payload.get("pdf_count") or 0)
            self._append_log(f"[EIDAT MANAGER] Scanned {file_count} file(s); {candidates} candidate(s) need processing.")
            try:
                self._refresh_files_tab()
            except Exception:
                pass
            if candidates > 0:
                if not auto:
                    QtWidgets.QMessageBox.information(
                        self,
                        "EIDAT Manager",
                        f"Detected {candidates} new/changed file(s) (PDF/Excel).\n\nUse 'Process New Files' to generate EIDAT Support artifacts.",
                    )
                if auto_process:
                    self._act_manager_process_new(auto=True)
                return f"Scan complete - {candidates} candidate(s) ready"
            if not auto:
                self._show_toast("No new files detected.")
            return "Scan complete - no new files detected"

        self._start_manager_action(
            heading="Global Repo Scan",
            status_text="Scanning global repository",
            task=lambda: be.eidat_manager_scan(repo),
            on_success=_on_success,
            auto=auto,
            show_popup=not auto,
        )

    def _act_manager_process_new(self, *, auto: bool = False) -> None:
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first.")
            repo = Path(repo_raw).expanduser()
        except Exception as exc:
            if not auto:
                QtWidgets.QMessageBox.warning(self, "EIDAT Manager", str(exc))
            return

        def _on_success(payload: dict):
            ok = int(payload.get("processed_ok") or 0)
            failed = int(payload.get("processed_failed") or 0)
            self._append_log(f"[EIDAT MANAGER] Processed: ok={ok}, failed={failed}")
            env_flags = payload.get("env_flags") or {}
            if env_flags:
                self._append_log(f"[EIDAT MANAGER] Env flags: {env_flags}")
            # Log individual errors to debug console
            results = payload.get("results") or []
            for r in results:
                if not r.get("ok") and r.get("error"):
                    self._append_log(f"[EIDAT MANAGER] FAILED: {r.get('rel_path')}: {r.get('error')}")
            if not auto:
                if failed > 0:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "EIDAT Manager",
                        f"Processed {ok} file(s), but {failed} failed.\n\nSee Debug Console for details.",
                    )
                else:
                    self._show_toast(f"Processed {ok} file(s).")
            return f"Processing complete - ok={ok}, failed={failed}"

        self._start_manager_action(
            heading="Process New Files",
            status_text="Processing new files",
            task=lambda: be.eidat_manager_process(repo),
            on_success=_on_success,
            auto=auto,
            show_popup=not auto,
        )

    def _act_manager_force_all(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Force Process All",
            "This will re-process all tracked files (PDF/Excel) and overwrite outputs. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first.")
            repo = Path(repo_raw).expanduser()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "EIDAT Manager", str(exc))
            return

        def _on_success(payload: dict):
            ok = int(payload.get("processed_ok") or 0)
            failed = int(payload.get("processed_failed") or 0)
            self._append_log(f"[EIDAT MANAGER] Force processed: ok={ok}, failed={failed}")
            env_flags = payload.get("env_flags") or {}
            if env_flags:
                self._append_log(f"[EIDAT MANAGER] Env flags: {env_flags}")
            # Log individual errors to debug console
            results = payload.get("results") or []
            for r in results:
                if not r.get("ok") and r.get("error"):
                    self._append_log(f"[EIDAT MANAGER] FAILED: {r.get('rel_path')}: {r.get('error')}")
            if failed > 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "EIDAT Manager",
                    f"Force processed {ok} file(s), but {failed} failed.\n\nSee Debug Console for details.",
                )
            else:
                self._show_toast(f"Force processed {ok} file(s).")
            return f"Force process complete - ok={ok}, failed={failed}"

        self._start_manager_action(
            heading="Force Process All",
            status_text="Processing all files",
            task=lambda: be.eidat_manager_process(repo, force=True),
            on_success=_on_success,
        )

    def _act_manager_index(self) -> None:
        try:
            repo_raw = (self.ed_global_repo.text() or "").strip()
            if not repo_raw:
                raise RuntimeError("Select a Global Repo first.")
            repo = Path(repo_raw).expanduser()
            payload = be.eidat_manager_index(repo)
            indexed = int(payload.get("indexed_count") or 0)
            groups = int(payload.get("groups_count") or 0)
            meta = int(payload.get("metadata_count") or 0)
            db_path = str(payload.get("index_db") or "")
            self.lbl_index_status.setText(f"Index: {indexed} docs, {groups} groups")
            QtWidgets.QMessageBox.information(
                self,
                "Index Summary",
                f"Indexed {indexed} document(s) from {meta} metadata file(s).\nGroups: {groups}\n\nIndex DB: {db_path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Index Summary", str(exc))

    # Workspace sync
    def _start_workspace_sync(self, *, auto: bool, show_popup: bool, heading: str):
        if self._sync_worker is not None and self._sync_worker.isRunning():
            if not auto:
                self._show_toast("Workspace scan already running")
            return
        if self._manager_worker is not None and self._manager_worker.isRunning():
            if not auto:
                self._show_toast("EIDAT Manager task already running")
            return
        try:
            repo_raw = Path(self.ed_repo.text()).expanduser() if hasattr(self, "ed_repo") else Path(self.ed_pdfs.text()).expanduser()
            repo = be.resolve_pdf_root(repo_raw)
        except Exception as exc:
            if not auto:
                self._show_toast(str(exc))
            return
        raw_terms = (self.ed_terms.text() or "").strip()
        try:
            terms = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
        except Exception as exc:
            if not auto:
                self._show_toast(str(exc))
            return
        if not auto:
            self._append_log(f"[GUI] {heading} started")
            self.status_bar.showMessage(f"{heading} in progress...")
        self._sync_popup_active = show_popup
        if show_popup:
            self._repo_scan_dialog.begin("Scanning repository", heading=heading)
        else:
            try:
                if not getattr(self, "_manager_popup_active", False):
                    self._repo_scan_dialog.hide()
            except Exception:
                pass
        self._sync_worker = WorkspaceSyncWorker(repo, terms, auto, parent=self)
        self._sync_worker.completed.connect(self._on_workspace_sync_done)
        self._sync_worker.failed.connect(self._on_workspace_sync_error)
        self._sync_worker.start()

    def _on_workspace_sync_done(self, summary: dict, details: list[dict], warnings: list[str]):
        auto = False
        if self._sync_worker is not None:
            auto = bool(getattr(self._sync_worker, "auto", False))
        self._sync_worker = None
        self._apply_sync_results(summary, details, auto=auto, warnings=warnings)
        if self._sync_popup_active:
            flagged = int(summary.get("new", 0) or 0) + int(summary.get("pdf_newer", 0) or 0) + int(summary.get("terms_newer", 0) or 0)
            if flagged > 0:
                msg = f"Scan complete - {flagged} out-of-date package(s) found"
            else:
                msg = "Scan complete - all packages up-to-date"
            self._repo_scan_dialog.finish(msg, success=True)
            self._sync_popup_active = False

    def _on_workspace_sync_error(self, message: str):
        auto = False
        if self._sync_worker is not None:
            auto = bool(getattr(self._sync_worker, "auto", False))
        self._sync_worker = None
        self._apply_sync_error(message, auto=auto)
        if self._sync_popup_active:
            self._repo_scan_dialog.finish(f"Scan failed: {message}", success=False)
            self._sync_popup_active = False

    def _apply_sync_results(self, summary: dict, details: list[dict], *, auto: bool, warnings: list[str] | None = None):
        self._sync_summary = summary
        self._sync_details = details
        if warnings:
            for warn in warnings:
                self._append_log(f"[WARN] {warn}")
        new = int(summary.get("new", 0) or 0)
        pdf_newer = int(summary.get("pdf_newer", 0) or 0)
        terms_newer = int(summary.get("terms_newer", 0) or 0)
        up_to_date = int(summary.get("up_to_date", 0) or 0)
        total = int(summary.get("total", 0) or 0)
        last = str(summary.get("last_sync", ""))
        repo_s = str(summary.get("repo_root", ""))
        txt = (
            f"Repository: {repo_s}\n"
            f"Total PDFs: {total}  |  New: {new}  |  Out-of-date (PDF): {pdf_newer}  |  Out-of-date (Terms): {terms_newer}  |  Up-to-date: {up_to_date}\n"
            f"Last sync: {last}"
        )
        self.lbl_sync_banner.setText(txt)
        if (new + pdf_newer + terms_newer) > 0:
            self.lbl_sync_banner.setStyleSheet(
                "#syncBanner { background: #fff4e5; color: #5b3100; border: 1px solid #ffd9a8; border-radius: 6px; padding: 8px 12px; }"
            )
        else:
            self.lbl_sync_banner.setStyleSheet(
                "#syncBanner { background: #e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9; border-radius: 6px; padding: 8px 12px; }"
            )
        flagged = (new + pdf_newer + terms_newer)
        if not auto:
            self._append_log("[GUI] Workspace sync complete")
            if flagged > 0:
                self._show_toast(f"Workspace synced - {flagged} out-of-date items found")
            else:
                self._show_toast("Workspace synced - all up-to-date")
            self.status_bar.showMessage("Workspace sync complete", 3000)
    def _apply_sync_error(self, message: str, *, auto: bool):
        self.lbl_sync_banner.setText(f"Sync failed: {message}")
        if not auto:
            self._append_log(f"[ERROR] Workspace sync failed: {message}")
            self._show_toast(f"Workspace sync failed: {message}")

    def _sync_workspace(self, auto: bool = False):
        try:
            repo_raw = Path(self.ed_repo.text()).expanduser() if hasattr(self, "ed_repo") else Path(self.ed_pdfs.text()).expanduser()
            repo = be.resolve_pdf_root(repo_raw)
        except Exception as exc:
            self.lbl_sync_banner.setText(str(exc))
            if not auto:
                self._show_toast(str(exc))
            return
        raw_terms = (self.ed_terms.text() or "").strip()
        try:
            terms = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
        except Exception as exc:
            self.lbl_sync_banner.setText(str(exc))
            if not auto:
                self._show_toast(str(exc))
            return
        warnings: list[str] = []
        try:
            # For manual syncs, first rebuild the registry from run_data_simple
            if not auto:
                try:
                    be.rebuild_registry_from_run_data()
                except Exception as e:
                    warnings.append(f"Registry rebuild failed: {e}")
                # Sync cell state with master.xlsx (prune orphaned terms)
                try:
                    from scripts.master_cell_state import sync_cell_state_with_master
                    sync_cell_state_with_master()
                except Exception as e:
                    warnings.append(f"Cell state sync failed: {e}")
            summary, details = be.compute_workspace_sync(repo, terms)
            self._apply_sync_results(summary, details, auto=auto, warnings=warnings)
        except Exception as e:
            self._apply_sync_error(str(e), auto=auto)

    def _act_sync_workspace(self):
        self._start_workspace_sync(auto=False, show_popup=True, heading="Workspace Sync")

    def _act_global_repo_scan(self):
        self._start_workspace_sync(auto=False, show_popup=True, heading="Global Repo Scan")

    def _act_compile_master_from_state(self):
        """
        Rebuild master.xlsx from simple run outputs.
        Shows confirmation dialog before proceeding.
        """
        # Show warning dialog
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Compile New Master Workbook",
            "This will rebuild master.xlsx from simple extraction data ONLY.\n\n"
            "All manual edits in master.xlsx will be LOST.\n\n"
            "Are you sure you want to continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Run the compilation synchronously (it's fast - just reading CSV/XLSX and writing Excel)
        try:
            self._append_log("[GUI] Compiling master workbook from simple runs...")
            self.status_bar.showMessage("Compiling new master workbook from simple runs...")

            # Call the backend function directly
            be.compile_master_from_state()

            self._append_log("[GUI] Master workbook compiled successfully!")
            self.status_bar.showMessage("Master workbook compiled successfully!", 5000)
            self._show_toast("Master workbook compiled successfully!")

        except Exception as e:
            self._append_log(f"[ERROR] Failed to compile master workbook: {e}")
            self.status_bar.showMessage(f"Failed to compile master: {e}", 5000)
            self._show_toast(f"Failed to compile master: {e}")
            import traceback
            traceback.print_exc()

    def _show_toast(self, message: str, duration: int = 5000):
        """Show a toast/cookie banner notification."""
        try:
            self._toast.show_message(message, duration)
        except Exception:
            pass

    def _show_outdated_popup(self, auto: bool = False):
        # Guard against duplicate dialogs
        if getattr(self, "_dlg_open_outdated", False):
            return
        # Always refresh workspace sync (and master workbook on manual invocations)
        try:
            # Treat button-driven opens as manual syncs so the registry/master
            # stay current. Callers can pass auto=True to suppress manual extras.
            self._sync_workspace(auto=auto)
        except Exception:
            pass
        self._dlg_open_outdated = True
        try:
            details = getattr(self, "_sync_details", None) or []
            # Out-of-date rows (existing behaviour) and full list for manual re-runs
            rows_outdated = [d for d in details if d.get("reason") in ("new", "pdf_newer", "terms_newer")]
            rows_all = list(details)
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("View Data Package List and Update EIDAT Database")
            dlg.resize(1100, 600)
            dlg.setStyleSheet("""
                QDialog {
                    background: #ffffff;
                }
            """)

            v = QtWidgets.QVBoxLayout(dlg)
            v.setContentsMargins(20, 20, 20, 20)
            v.setSpacing(16)

            # Title and description
            title = QtWidgets.QLabel("Workspace Data Packages")
            title.setStyleSheet("font-size: 16px; font-weight: 600; color: #111827;")
            v.addWidget(title)

            desc = QtWidgets.QLabel(
                "Review EIDAT data packages and select which EIDPs to (re)process.\n"
                "Out-of-date items include new PDFs, modified PDFs, or EIDPs missing terms that were added to the schema."
            )
            desc.setStyleSheet("font-size: 13px; color: #4b5563;")
            desc.setWordWrap(True)
            v.addWidget(desc)

            # Toolbar with Select All/None buttons
            toolbar = QtWidgets.QHBoxLayout()
            toolbar.setSpacing(8)

            btn_sel_all = QtWidgets.QPushButton("Select All")
            btn_sel_all.setStyleSheet("""
                QPushButton {
                    padding: 8px 16px;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background: #f9fafb;
                    border-color: #9ca3af;
                }
            """)

            btn_sel_none = QtWidgets.QPushButton("Select None")
            btn_sel_none.setStyleSheet("""
                QPushButton {
                    padding: 8px 16px;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #374151;
                    border: 1px solid #d1d5db;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background: #f9fafb;
                    border-color: #9ca3af;
                }
            """)

            toolbar.addWidget(btn_sel_all)
            toolbar.addWidget(btn_sel_none)
            toolbar.addStretch(1)
            v.addLayout(toolbar)

            # Tabs for Out-of-Date vs All EIDPs
            tabs = QtWidgets.QTabWidget()
            tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
            tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #e5e7eb;
                    border-radius: 6px;
                    margin-top: 8px;
                    background: #ffffff;
                }
                QTabBar::tab {
                    padding: 8px 16px;
                    margin-right: 4px;
                    border: 1px solid #e5e7eb;
                    border-bottom: none;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    background: #f3f4f6;
                    color: #4b5563;
                    font-size: 12px;
                    font-weight: 500;
                }
                QTabBar::tab:selected {
                    background: #2563eb;
                    color: #ffffff;
                    border-color: #2563eb;
                }
                QTabBar::tab:hover {
                    background: #e5e7eb;
                }
            """)
            tab_outdated = QtWidgets.QWidget()
            tab_all = QtWidgets.QWidget()
            tabs.addTab(tab_outdated, "Out-of-Date Only")
            tabs.addTab(tab_all, "All Data Packages")

            v.addWidget(tabs, 1)

            # Table with checkboxes (Out-of-Date)
            cols = [
                "Select",
                "Serial",
                "Status Reason",
                "Missing Terms",
                "PDF Path",
                "Last Run",
                "PDF Modified",
                "Schema Version Time",
            ]
            tbl = QtWidgets.QTableWidget(0, len(cols))
            tbl.setHorizontalHeaderLabels(cols)
            tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            tbl.setAlternatingRowColors(True)
            tbl.verticalHeader().setVisible(False)
            # Tables in this popup are read-only; selection controls which EIDPs to run.
            tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

            # Consistent styling matching Terms Editor
            tbl.setStyleSheet("""
                QTableWidget {
                    background-color: #ffffff;
                    alternate-background-color: #f9fafb;
                    selection-background-color: #dbeafe;
                    selection-color: #111827;
                    gridline-color: #e5e7eb;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                }
                QTableWidget::item {
                    padding: 10px 8px;
                    color: #374151;
                }
                QTableWidget::item:selected {
                    background-color: #dbeafe;
                    color: #111827;
                }
                QHeaderView::section {
                    background-color: #f3f4f6;
                    color: #111827;
                    padding: 12px 8px;
                    border: none;
                    border-right: 1px solid #e5e7eb;
                    border-bottom: 2px solid #d1d5db;
                    font-weight: 600;
                    font-size: 13px;
                }
                QCheckBox {
                    spacing: 0px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 4px;
                    border: 2px solid #d1d5db;
                    background: #ffffff;
                }
                QCheckBox::indicator:hover {
                    border-color: #2563eb;
                }
                QCheckBox::indicator:checked {
                    background: #2563eb;
                    border-color: #2563eb;
                    image: url(none);
                }
                QCheckBox::indicator:checked:after {
                    content: "✓";
                    color: #ffffff;
                }
            """)

            # Place out-of-date table inside first tab
            layout_outdated = QtWidgets.QVBoxLayout(tab_outdated)
            layout_outdated.setContentsMargins(0, 0, 0, 0)
            layout_outdated.addWidget(tbl)

            # Table for all EIDPs (grouped by program) with a database-style layout
            cols_all = [
                "Select",
                "Program",
                "Serial",
                "Status",
                "PDF Path",
                "Last Run",
                "PDF Modified",
                "Schema Version Time",
            ]
            tbl_all = QtWidgets.QTableWidget(0, len(cols_all))
            tbl_all.setHorizontalHeaderLabels(cols_all)
            tbl_all.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            tbl_all.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            tbl_all.setAlternatingRowColors(True)
            tbl_all.verticalHeader().setVisible(False)
            tbl_all.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            tbl_all.setStyleSheet(tbl.styleSheet())

            # All EIDPs tab header with program filter/search
            layout_all = QtWidgets.QVBoxLayout(tab_all)
            layout_all.setContentsMargins(8, 8, 8, 8)

            filter_row = QtWidgets.QHBoxLayout()
            filter_row.setSpacing(8)
            lbl_filter = QtWidgets.QLabel("Filter by Program / Serial:")
            lbl_filter.setStyleSheet("font-size: 12px; color: #4b5563;")
            ed_filter = QtWidgets.QLineEdit()
            ed_filter.setPlaceholderText("Type program name, serial, or path text...")
            ed_filter.setStyleSheet(
                "QLineEdit { padding: 6px 10px; border-radius: 4px; border: 1px solid #d1d5db; font-size: 12px; }"
            )
            filter_row.addWidget(lbl_filter)
            filter_row.addWidget(ed_filter, 1)
            layout_all.addLayout(filter_row)

            layout_all.addWidget(tbl_all, 1)

            # Pre-compute missing-term counts per serial for the current schema/master
            try:
                try:
                    raw_terms = (self.ed_terms.text() or "").strip()
                    terms_path = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
                except Exception:
                    terms_path = be.resolve_terms_path()
                serials_for_counts = sorted(
                    {d.get("serial_component", "") for d in rows_outdated if d.get("serial_component")}
                )
                missing_counts = (
                    be.count_missing_terms_per_serial(serials_for_counts, terms_path) if serials_for_counts else {}
                )
            except Exception:
                missing_counts = {}

            # Populate Out-of-Date table with checkboxes and missing-term counts
            for r, d in enumerate(rows_outdated):
                tbl.insertRow(r)

                # Create checkbox widget for Select column
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(True)
                checkbox.setStyleSheet("""
                    QCheckBox {
                        margin-left: 8px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                        border-radius: 3px;
                        border: 2px solid #d1d5db;
                        background: #ffffff;
                    }
                    QCheckBox::indicator:hover {
                        border-color: #2563eb;
                    }
                    QCheckBox::indicator:checked {
                        background: #2563eb;
                        border-color: #2563eb;
                    }
                """)

                # Center the checkbox
                checkbox_widget = QtWidgets.QWidget()
                checkbox_layout = QtWidgets.QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                tbl.setCellWidget(r, 0, checkbox_widget)

                # Add data to other columns
                tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(d.get("serial_component", "")))
                reason = d.get("reason", "")
                reason_item = QtWidgets.QTableWidgetItem(reason)
                tbl.setItem(r, 2, reason_item)

                # Missing-term count per EIDP (blank when not applicable)
                serial = d.get("serial_component", "") or ""
                count_val = missing_counts.get(serial, 0)
                missing_text = str(count_val) if count_val > 0 else ""
                missing_item = QtWidgets.QTableWidgetItem(missing_text)
                if count_val > 0:
                    missing_item.setForeground(QtGui.QBrush(QtGui.QColor("#b91c1c")))
                tbl.setItem(r, 3, missing_item)

                tbl.setItem(r, 4, QtWidgets.QTableWidgetItem(d.get("pdf", "")))
                tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(d.get("run_date", "")))
                tbl.setItem(r, 6, QtWidgets.QTableWidgetItem(d.get("pdf_mtime", "")))
                tbl.setItem(r, 7, QtWidgets.QTableWidgetItem(d.get("terms_mtime", "")))

            # Populate All-EIDP table with checkboxes (initially unchecked) and a database-style layout
            for r, d in enumerate(rows_all):
                tbl_all.insertRow(r)

                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(False)
                checkbox.setStyleSheet("""
                    QCheckBox {
                        margin-left: 8px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                        border-radius: 3px;
                        border: 2px solid #d1d5db;
                        background: #ffffff;
                    }
                    QCheckBox::indicator:hover {
                        border-color: #2563eb;
                    }
                    QCheckBox::indicator:checked {
                        background: #2563eb;
                        border-color: #2563eb;
                    }
                """)

                checkbox_widget = QtWidgets.QWidget()
                checkbox_layout = QtWidgets.QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                tbl_all.setCellWidget(r, 0, checkbox_widget)

                program = d.get("program_name", "") or "(Unknown Program)"
                tbl_all.setItem(r, 1, QtWidgets.QTableWidgetItem(program))
                tbl_all.setItem(r, 2, QtWidgets.QTableWidgetItem(d.get("serial_component", "")))

                status = d.get("reason", "")
                status_item = QtWidgets.QTableWidgetItem(status)
                if status in ("new", "pdf_newer", "terms_newer"):
                    status_item.setForeground(QtGui.QBrush(QtGui.QColor("#b45309")))
                tbl_all.setItem(r, 3, status_item)

                tbl_all.setItem(r, 4, QtWidgets.QTableWidgetItem(d.get("pdf", "")))
                tbl_all.setItem(r, 5, QtWidgets.QTableWidgetItem(d.get("run_date", "")))
                tbl_all.setItem(r, 6, QtWidgets.QTableWidgetItem(d.get("pdf_mtime", "")))
                tbl_all.setItem(r, 7, QtWidgets.QTableWidgetItem(d.get("terms_mtime", "")))

            tbl.resizeColumnsToContents()
            tbl.setColumnWidth(0, 80)  # Fixed width for checkbox column

            tbl_all.resizeColumnsToContents()
            tbl_all.setColumnWidth(0, 80)
            tbl_all.setSortingEnabled(True)
            tbl_all.sortItems(1)

            # Simple text filter for the "All Data Packages" tab
            def _apply_all_filter(text: str):
                text = text.strip().lower()
                for row in range(tbl_all.rowCount()):
                    # Combine Program, Serial, and PDF path into a searchable string
                    program_item = tbl_all.item(row, 1)
                    serial_item = tbl_all.item(row, 2)
                    pdf_item = tbl_all.item(row, 4)
                    combined = " ".join(
                        [
                            (program_item.text() if program_item else ""),
                            (serial_item.text() if serial_item else ""),
                            (pdf_item.text() if pdf_item else ""),
                        ]
                    ).lower()
                    match = (text in combined) if text else True
                    tbl_all.setRowHidden(row, not match)

            ed_filter.textChanged.connect(_apply_all_filter)

            # Update Select All/None to work with checkbox widgets on the active tab
            def _current_table_and_columns():
                # Determine which table is active and the indices of key columns
                current = tabs.currentWidget()
                if current is tab_outdated:
                    return tbl, cols.index("PDF Path"), cols.index("Serial")
                if current is tab_all:
                    return tbl_all, cols_all.index("PDF Path"), cols_all.index("Serial")
                return None, -1, -1

            def _set_all_checkboxes(checked: bool):
                tbl_active, _, _ = _current_table_and_columns()
                if not tbl_active:
                    return
                for r in range(tbl_active.rowCount()):
                    widget = tbl_active.cellWidget(r, 0)
                    if widget:
                        checkbox = widget.findChild(QtWidgets.QCheckBox)
                        if checkbox:
                            checkbox.setChecked(checked)

            btn_sel_all.clicked.connect(lambda: _set_all_checkboxes(True))
            btn_sel_none.clicked.connect(lambda: _set_all_checkboxes(False))

            # Bottom buttons with consistent styling
            btns = QtWidgets.QHBoxLayout()
            btns.setSpacing(12)

            btn_run_all = QtWidgets.QPushButton("Run All Out-of-Date")
            btn_run_all.setStyleSheet("""
                QPushButton {
                    padding: 10px 20px;
                    border-radius: 6px;
                    background: #2563eb;
                    color: #ffffff;
                    border: 1px solid #2563eb;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #1d4ed8;
                }
            """)

            btn_run = QtWidgets.QPushButton("Run Selected")
            btn_run.setStyleSheet("""
                QPushButton {
                    padding: 10px 20px;
                    border-radius: 6px;
                    background: #2563eb;
                    color: #ffffff;
                    border: 1px solid #2563eb;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #1d4ed8;
                }
            """)

            btn_close = QtWidgets.QPushButton("Close")
            btn_close.setStyleSheet("""
                QPushButton {
                    padding: 10px 20px;
                    border-radius: 6px;
                    background: #ffffff;
                    color: #6b7280;
                    border: 1px solid #d1d5db;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background: #f9fafb;
                }
            """)

            btn_pre_ocr = QtWidgets.QPushButton("Pre-OCR + Merge")
            btn_pre_ocr.setStyleSheet("""
                QPushButton {
                    padding: 10px 16px;
                    border-radius: 6px;
                    background: #10b981;
                    color: #ffffff;
                    border: 1px solid #0ea271;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #0ea271;
                }
            """)
            btn_simple = QtWidgets.QPushButton("Run Simple Extraction")
            btn_simple.setStyleSheet("""
                QPushButton {
                    padding: 10px 16px;
                    border-radius: 6px;
                    background: #f59e0b;
                    color: #111827;
                    border: 1px solid #d97706;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #d97706;
                    color: #ffffff;
                }
            """)
            btns.addStretch(1)
            btns.addWidget(btn_pre_ocr)
            btns.addWidget(btn_simple)
            btns.addWidget(btn_run_all)
            btns.addWidget(btn_run)
            btns.addWidget(btn_close)
            v.addLayout(btns)

            # Helper to collect checked entries (path + serial) from the active tab
            def _collect_checked_entries() -> list[tuple[Path, str]]:
                tbl_active, pdf_col, serial_col = _current_table_and_columns()
                entries: list[tuple[Path, str]] = []
                if not tbl_active or pdf_col < 0:
                    return entries
                for r in range(tbl_active.rowCount()):
                    widget = tbl_active.cellWidget(r, 0)
                    if widget:
                        checkbox = widget.findChild(QtWidgets.QCheckBox)
                        if checkbox and checkbox.isChecked():
                            path_item = tbl_active.item(r, pdf_col)
                            serial_item = tbl_active.item(r, serial_col) if serial_col >= 0 else None
                            p_text = path_item.text() if path_item else ""
                            if p_text:
                                entries.append((Path(p_text), serial_item.text() if serial_item else ""))
                return entries

            # Update run functions to work with tabbed tables
            def _run_selected():
                entries = _collect_checked_entries()
                if not entries:
                    QtWidgets.QMessageBox.information(dlg, "Nothing selected", "Choose at least one EIDP to run.")
                    return
                try:
                    raw_terms = (self.ed_terms.text() or "").strip()
                    terms = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
                except Exception:
                    terms = be.resolve_terms_path()
                started = self._prompt_outdated_run_mode(
                    entries,
                    terms,
                    title="Run Selected EIDPs",
                    description="Choose whether to run only missing schema terms or re-extract every term for the selected data packages.",
                    full_status="Running selected EIDPs...",
                    missing_status="Extracting missing terms for selected EIDPs...",
                )
                if started:
                    dlg.accept()

            def _run_all():
                entries = [(Path(d.get("pdf")), d.get("serial_component", "")) for d in rows_outdated if d.get("pdf")] if rows_outdated else []
                if not entries:
                    QtWidgets.QMessageBox.information(dlg, "Nothing to run", "No out-of-date EIDPs found.")
                    return
                try:
                    raw_terms = (self.ed_terms.text() or "").strip()
                    terms = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
                except Exception:
                    terms = be.resolve_terms_path()
                started = self._prompt_outdated_run_mode(
                    entries,
                    terms,
                    title="Run All Out-of-Date EIDPs",
                    description="Choose whether to re-run every term or only the missing schema terms for the out-of-date data packages.",
                    full_status="Running all out-of-date EIDPs...",
                    missing_status="Extracting missing terms for out-of-date EIDPs...",
                )
                if started:
                    dlg.accept()

            def _pre_ocr_merge():
                entries = _collect_checked_entries()
                if not entries:
                    QtWidgets.QMessageBox.information(dlg, "Nothing selected", "Choose at least one EIDP to pre-OCR.")
                    return
                paths = [p for p, _ in entries]
                if not paths:
                    QtWidgets.QMessageBox.information(dlg, "Nothing selected", "No valid PDF paths were found.")
                    return
                self._start_worker(
                    lambda paths=paths: be.pre_ocr_merge_pdfs(paths),
                    status_msg="Pre-OCR + merge (no extraction)...",
                    show_run_progress=False,
                    total_files=len(paths),
                )

            def _run_simple():
                entries = _collect_checked_entries()
                if not entries:
                    QtWidgets.QMessageBox.information(dlg, "Nothing selected", "Choose at least one EIDP to extract.")
                    return
                paths = [p for p, _ in entries]
                if not paths:
                    QtWidgets.QMessageBox.information(dlg, "Nothing selected", "No valid PDF paths were found.")
                    return
                # Run simple extraction (uses merged artifacts; will pre-OCR on demand).
                self._start_worker(
                    lambda paths=paths: be.run_simple_extraction(paths),
                    status_msg="Running simple merged extraction...",
                    show_run_progress=False,
                    total_files=len(paths),
                )

            btn_pre_ocr.clicked.connect(_pre_ocr_merge)
            btn_simple.clicked.connect(_run_simple)
            btn_run.clicked.connect(_run_selected)
            btn_run_all.clicked.connect(_run_all)
            btn_close.clicked.connect(dlg.reject)
            self._prepare_dialog(dlg)
            dlg.exec()
        finally:
            self._dlg_open_outdated = False

    def _scan_refresh(self):
        # Env status + badge
        py = be.resolve_project_python()
        env_ok = Path(py).exists()
        self.lbl_env.setText(f"Env: {'OK' if env_ok else 'Not Installed'} ({py})")
        self.lbl_ready.setText("System Ready" if env_ok else "Setup Needed")
        try:
            self.lbl_env_health.setProperty("status", "ok" if env_ok else "bad")
            self.lbl_env_health.setText("Healthy" if env_ok else "Issues")
            self.lbl_env_health.style().unpolish(self.lbl_env_health); self.lbl_env_health.style().polish(self.lbl_env_health)
        except Exception:
            pass

        # Sync settings from env
        try:
            env = be.load_scanner_env()
            mode = env.get("OCR_MODE", "fallback").strip().lower()
            display = self._ocr_value_to_display.get(mode, self._ocr_value_to_display["fallback"]) if hasattr(self, "_ocr_value_to_display") else mode
            if self.cmb_ocr_mode.currentText() != display:
                # Ensure the display choice exists (in case of dynamic mapping)
                if display not in [self.cmb_ocr_mode.itemText(i) for i in range(self.cmb_ocr_mode.count())]:
                    self.cmb_ocr_mode.addItem(display)
                self.cmb_ocr_mode.setCurrentText(display)
            # OCR row Y tolerance (pixels)
            row_eps = env.get("OCR_ROW_EPS", "15")
            try:
                eps_val = float(row_eps)
                eps_int = int(max(2, min(40, round(eps_val))))
                self.sld_ocr_row_tol.blockSignals(True)
                self.sld_ocr_row_tol.setValue(eps_int)
                self.sld_ocr_row_tol.blockSignals(False)
                self.lbl_ocr_row_tol.setText(str(eps_int))
            except Exception:
                pass
            # OCR DPI (100-1000)
            try:
                dpi = int(env.get("OCR_DPI", "500"))
                dpi = max(100, min(1000, dpi))
                self.sld_ocr_dpi.blockSignals(True)
                self.sld_ocr_dpi.setValue(dpi)
                self.sld_ocr_dpi.blockSignals(False)
                self.lbl_dpi_val.setText(str(dpi))
            except Exception:
                pass
            # Fuzzy Matching Preset
            fuzzy_preset = env.get("FUZZY_PRESET", "medium").strip().lower()
            fuzzy_display = self._fuzzy_value_to_display.get(fuzzy_preset, self._fuzzy_value_to_display["medium"]) if hasattr(self, "_fuzzy_value_to_display") else "Medium (Balanced)"
            if self.cmb_fuzzy_preset.currentText() != fuzzy_display:
                if fuzzy_display not in [self.cmb_fuzzy_preset.itemText(i) for i in range(self.cmb_fuzzy_preset.count())]:
                    self.cmb_fuzzy_preset.addItem(fuzzy_display)
                self.cmb_fuzzy_preset.blockSignals(True)
                self.cmb_fuzzy_preset.setCurrentText(fuzzy_display)
                self.cmb_fuzzy_preset.blockSignals(False)
            enable_logging = not (env.get("QUIET", "1").strip().lower() in ("1", "true", "yes"))
            self.chk_logging.blockSignals(True)
            self.chk_logging.setChecked(enable_logging)
            self.chk_logging.blockSignals(False)
        except Exception:
            pass

        # Enable/disable core actions
        raw_terms = (self.ed_terms.text() or "").strip()
        try:
            terms_path = be.resolve_terms_path(Path(raw_terms).expanduser()) if raw_terms else be.resolve_terms_path()
            has_terms = terms_path.exists()
        except Exception:
            has_terms = False
        try:
            pdfs_path = be.resolve_pdf_root(Path(self.ed_pdfs.text()).expanduser())
            has_pdfs = pdfs_path.exists()
        except Exception:
            has_pdfs = False
        worker = getattr(self, "_worker", None)
        self.btn_start.setEnabled(has_terms and has_pdfs and (worker is None or not worker.isRunning()))

        # Workspace sync banner refresh handled by periodic timer

        # Plotting selection
        self._refresh_series_catalog()
        # Inline viewer removed; popup will build data on demand

    # Upload ingestion
    def _ingest_paths(self, paths: list[str]) -> None:
        import shutil
        try:
            dest_dir = be.resolve_pdf_root(Path(self.ed_pdfs.text()).expanduser())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Data folder", str(exc))
            return
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Folder error", f"Cannot create data folder:\n{dest_dir}\n\n{e}")
            return
        copied, skipped, errors = 0, 0, 0
        to_visit: list[Path] = []
        for p in paths:
            try:
                to_visit.append(Path(p))
            except Exception:
                continue
        allowed = {".pdf", *{str(s).lower() for s in getattr(be, "EXCEL_EXTENSIONS", {".xlsx", ".xlsm", ".xls"})}}

        def is_allowed(p: Path) -> bool:
            try:
                return (p.suffix or "").lower() in allowed and not p.name.startswith("~$")
            except Exception:
                return False
        visit_files: list[Path] = []
        for p in to_visit:
            try:
                if p.is_dir():
                    for ext in sorted(allowed):
                        for sub in p.rglob(f"*{ext}"):
                            visit_files.append(sub)
                elif p.is_file():
                    visit_files.append(p)
            except Exception:
                errors += 1
        for src in visit_files:
            try:
                if not is_allowed(src):
                    skipped += 1
                    continue
                dst = self._unique_destination(dest_dir / src.name)
                shutil.copy2(str(src), str(dst))
                copied += 1
                self._append_log(f"[GUI] Added {src} -> {dst}")
            except Exception as e:
                errors += 1
                self._append_log(f"[ERROR] Copy failed for {src}: {e}")
        self._refresh_upload_list()
        msg = f"Added {copied} file(s)" + (f", skipped {skipped}" if skipped else "") + (f", errors {errors}" if errors else "")
        self.status_bar.showMessage(msg, 4000)

    def _unique_destination(self, initial: Path) -> Path:
        if not initial.exists():
            return initial
        stem = initial.stem; suf = initial.suffix; parent = initial.parent; i = 1
        while True:
            cand = parent / f"{stem} ({i}){suf}"
            if not cand.exists():
                return cand
            i += 1

    def _refresh_upload_list(self):
        folder = Path(self.ed_pdfs.text()).expanduser()
        rows: list[tuple[str, str, str]] = []
        if folder.exists():
            try:
                allowed = {".pdf", *{str(s).lower() for s in getattr(be, "EXCEL_EXTENSIONS", {".xlsx", ".xlsm", ".xls"})}}
                for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
                    if not p.is_file():
                        continue
                    if p.name.startswith("~$"):
                        continue
                    if (p.suffix or "").lower() not in allowed:
                        continue
                    size = self._fmt_size(p.stat().st_size)
                    mtime = self._fmt_mtime(p.stat().st_mtime)
                    rows.append((p.name, size, mtime))
            except Exception:
                pass
        self.list_pdfs.setRowCount(0)
        for r, (name, size, mtime) in enumerate(rows):
            self.list_pdfs.insertRow(r)
            self.list_pdfs.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self.list_pdfs.setItem(r, 1, QtWidgets.QTableWidgetItem(size))
            self.list_pdfs.setItem(r, 2, QtWidgets.QTableWidgetItem(mtime))

    def _fmt_size(self, n: int) -> str:
        size = float(n)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _fmt_mtime(self, ts: float) -> str:
        try:
            import datetime as _dt
            return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    def _selected_data_paths(self) -> list[Path]:
        folder = Path(self.ed_pdfs.text()).expanduser()
        sel = []
        for idx in self.list_pdfs.selectionModel().selectedRows():
            item = self.list_pdfs.item(idx.row(), 0)
            if item is None:
                continue
            sel.append(folder / item.text())
        return sel

    def _act_remove_selected(self):
        files = self._selected_data_paths()
        if not files:
            return
        names = "\n".join(p.name for p in files[:10])
        extra = "" if len(files) <= 10 else f"\nÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦and {len(files)-10} more"
        if (
            QtWidgets.QMessageBox.question(
                self, "Remove files", f"Delete these from data folder?\n\n{names}{extra}"
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        removed, errors = 0, 0
        for p in files:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                errors += 1
        self._append_log(f"[GUI] Removed {removed} file(s); errors {errors}")
        self._refresh_upload_list()

    def _act_remove_all(self):
        folder = Path(self.ed_pdfs.text()).expanduser()
        if not folder.exists():
            return
        if (
            QtWidgets.QMessageBox.question(
                self, "Remove all", f"Delete ALL data files in:\n{folder}?"
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        removed, errors = 0, 0
        allowed = {".pdf", *{str(s).lower() for s in getattr(be, "EXCEL_EXTENSIONS", {".xlsx", ".xlsm", ".xls"})}}
        for p in folder.iterdir():
            try:
                if not p.is_file():
                    continue
                if p.name.startswith("~$"):
                    continue
                if (p.suffix or "").lower() not in allowed:
                    continue
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                errors += 1
        self._append_log(f"[GUI] Cleared {removed} file(s); errors {errors}")
        self._refresh_upload_list()

    def _act_add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            str(Path(self.ed_pdfs.text()).expanduser()),
            "Supported files (*.pdf *.xlsx *.xlsm *.xls *.mat);;PDF files (*.pdf);;Excel/MAT files (*.xlsx *.xlsm *.xls *.mat);;All files (*.*)",
        )
        if paths:
            self._ingest_paths(paths)

    def _act_add_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder with data files", str(Path(self.ed_pdfs.text()).expanduser()))
        if path:
            self._ingest_paths([path])

    # Plot selection helpers
    def _refresh_series_catalog(self):
        try:
            options = be.list_plot_series_options()
        except Exception:
            options = []
        self._plot_series_options = options
        if not options and hasattr(self, "status_bar"):
            self.status_bar.showMessage("No plot series found. Add or restore user_inputs/plot_terms.xlsx.", 4000)


    # NOTE: _load_proposed_plots, _refresh_proposed_summary, _open_proposed_plots_dialog were removed
    # as they were only used by the unused Plot tab UI (ProposedPlotsDialog was also removed)

    def _apply_tab_widths(self) -> None:
        if not hasattr(self, "_tab_style_template"):
            return
        try:
            bar = self.tabs.tabBar()
        except Exception:
            bar = None
        if not bar:
            return
        count = max(1, bar.count())
        width = max(80, self.tabs.width() // count)
        css = self._tab_style_template.replace("{width}", str(width))
        self.tabs.setStyleSheet(css)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)

    def _build_logo_pixmap(self, size: int = 52) -> QtGui.QPixmap:
        """Load external app logo if present; otherwise draw the fallback glyph.

        Checks these paths in order and scales preserving aspect ratio:
        - ui_next/assets/app_logo.png
        - ui_next/assets/logo.png
        - user_inputs/app_logo.png
        """
        candidates = [
            (be.ROOT / "ui_next" / "assets" / "app_logo.png"),
            (be.ROOT / "ui_next" / "assets" / "logo.png"),
            (be.ROOT / "user_inputs" / "app_logo.png"),
        ]
        for path in candidates:
            try:
                if path.exists():
                    pix = QtGui.QPixmap(str(path))
                    if not pix.isNull():
                        scaled = pix.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                        try:
                            # Also set the window icon for consistency
                            self.setWindowIcon(QtGui.QIcon(scaled))
                        except Exception:
                            pass
                        return scaled
            except Exception:
                pass
        # Fallback: draw an inline polished monogram
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = QtCore.QRectF(0.5, 0.5, size - 1, size - 1)
        radius = size * 0.24

        # Base gradient block with subtle border
        gradient = QtGui.QLinearGradient(0, 0, size, size)
        gradient.setColorAt(0, QtGui.QColor("#60a5fa"))
        gradient.setColorAt(1, QtGui.QColor("#1d4ed8"))
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a"), max(2, size // 18)))
        painter.drawRoundedRect(rect, radius, radius)

        # Inner glow
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 35))
        painter.drawRoundedRect(rect.adjusted(size * 0.08, size * 0.08, -size * 0.08, -size * 0.25), radius * 0.8, radius * 0.8)

        # Accent ring
        inner = QtCore.QRectF(size * 0.22, size * 0.22, size * 0.56, size * 0.56)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), max(2, size // 25)))
        painter.drawEllipse(inner)

        # Stylized "E" glyph
        glyph_pen = QtGui.QPen(QtGui.QColor("#f8fafc"))
        glyph_pen.setWidth(max(3, size // 16))
        glyph_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(glyph_pen)
        mid_y = size * 0.5
        left_x = size * 0.28
        right_x = size * 0.72
        painter.drawLine(QtCore.QPointF(left_x, size * 0.26), QtCore.QPointF(left_x, size * 0.74))
        painter.drawLine(QtCore.QPointF(left_x, size * 0.3), QtCore.QPointF(right_x, size * 0.3))
        painter.drawLine(QtCore.QPointF(left_x, mid_y), QtCore.QPointF(size * 0.65, mid_y))
        painter.drawLine(QtCore.QPointF(left_x, size * 0.7), QtCore.QPointF(right_x, size * 0.7))

        # Small spark in corner
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 180), max(2, size // 28)))
        painter.drawPoint(QtCore.QPointF(size * 0.78, size * 0.28))

        painter.end()
        try:
            self.setWindowIcon(QtGui.QIcon(pix))
        except Exception:
            pass
        return pix

    # Settings persistence
    def _on_ocr_row_tol_slider(self, value: int):
        try:
            self.lbl_ocr_row_tol.setText(str(value))
        except Exception:
            pass
        self._persist_settings_from_panel()

    def _on_dpi_slider(self, value: int):
        try:
            snapped = max(100, min(1000, int(round(value / 50.0) * 50)))
            if snapped != value:
                self.sld_ocr_dpi.blockSignals(True)
                self.sld_ocr_dpi.setValue(snapped)
                self.sld_ocr_dpi.blockSignals(False)
                value = snapped
            self.lbl_dpi_val.setText(str(value))
        except Exception:
            pass
        self._persist_settings_from_panel()

    def _persist_settings_from_panel(self):
        try:
            env = be.parse_scanner_env(be.SCANNER_ENV_LOCAL)
            disp = self.cmb_ocr_mode.currentText()
            env["OCR_MODE"] = self._ocr_display_to_value.get(disp, "fallback") if hasattr(self, "_ocr_display_to_value") else disp.strip().lower()
            env["OCR_ROW_EPS"] = str(int(self.sld_ocr_row_tol.value()))
            env["OCR_DPI"] = str(int(self.sld_ocr_dpi.value()))
            # Fuzzy matching preset
            fuzzy_disp = self.cmb_fuzzy_preset.currentText()
            env["FUZZY_PRESET"] = self._fuzzy_display_to_value.get(fuzzy_disp, "medium") if hasattr(self, "_fuzzy_display_to_value") else "medium"
            # QUIET is inverse of logging toggle
            env["QUIET"] = "0" if self.chk_logging.isChecked() else "1"
            be.save_scanner_env(env)
            self.status_bar.showMessage("Settings saved", 2000)
        except Exception:
            pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
