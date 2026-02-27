from __future__ import annotations

import re
import json
import time
import shutil
import sys
import threading
import math
import statistics
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

    def finish(self, message: str, success: bool = True):
        self._dot_timer.stop()
        self.lbl_status.setText(message)
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
                for cb in (self.cb_cont_program, self.cb_cont_part, self.cb_cont_vendor, self.cb_cont_asset, getattr(self, "cb_cont_test_plan", None)):
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

        # Test Data projects are currently "All only" (no program/asset/group pre-filtering).
        try:
            if is_test_data:
                self.rb_all.setChecked(True)
                self.rb_all.setText("All indexed TD reports")
                for w in (self.rb_program, self.rb_asset, self.rb_group):
                    w.setEnabled(False)
                for w in (self.cb_program, self.cb_asset, self.cb_group):
                    w.setEnabled(False)
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
                for w in (self.rb_program, self.rb_asset, self.rb_group):
                    w.setEnabled(True)
                for w in (self.cb_program, self.cb_asset, self.cb_group):
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
        self.rb_group = QtWidgets.QRadioButton("Similarity group")
        self.rb_all.setChecked(True)

        for rb in (self.rb_all, self.rb_program, self.rb_asset, self.rb_group):
            rb.toggled.connect(lambda _=False: self._apply_filter_and_refresh_table(select_all=True))

        self.cb_program = QtWidgets.QComboBox()
        self.cb_asset = QtWidgets.QComboBox()
        self.cb_group = QtWidgets.QComboBox()
        self.cb_program.currentIndexChanged.connect(lambda *_: self._apply_filter_and_refresh_table(select_all=True))
        self.cb_asset.currentIndexChanged.connect(lambda *_: self._apply_filter_and_refresh_table(select_all=True))
        self.cb_group.currentIndexChanged.connect(lambda *_: self._apply_filter_and_refresh_table(select_all=True))

        top.addWidget(self.rb_all)
        top.addWidget(self.rb_program)
        top.addWidget(self.cb_program, 1)
        top.addWidget(self.rb_asset)
        top.addWidget(self.cb_asset, 1)
        top.addWidget(self.rb_group)
        top.addWidget(self.cb_group, 1)
        v.addLayout(top)

        tbl_cols = ["Select", "Program", "Serial", "Doc Type", "Asset Type", "Metadata (rel)", "Group"]
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
        self.cb_cont_test_plan = QtWidgets.QCheckBox("Acceptance Test Plan")

        self.list_cont_program = QtWidgets.QListWidget()
        self.list_cont_part = QtWidgets.QListWidget()
        self.list_cont_vendor = QtWidgets.QListWidget()
        self.list_cont_asset = QtWidgets.QListWidget()
        self.list_cont_test_plan = QtWidgets.QListWidget()

        for lst in (
            self.list_cont_program,
            self.list_cont_part,
            self.list_cont_vendor,
            self.list_cont_asset,
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
        _add_cont_row(2, 0, self.cb_cont_test_plan, self.list_cont_test_plan, col_span=2)

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

        programs = sorted({str(d.get("program_title") or "").strip() for d in docs if str(d.get("program_title") or "").strip()})
        assets = sorted({str(d.get("asset_type") or "").strip() for d in docs if str(d.get("asset_type") or "").strip()})
        vendors = sorted({str(d.get("vendor") or "").strip() for d in docs if str(d.get("vendor") or "").strip()})
        parts = sorted({str(d.get("part_number") or "").strip() for d in docs if str(d.get("part_number") or "").strip()})
        test_plans = sorted(
            {
                str(d.get("acceptance_test_plan_number") or "").strip()
                for d in docs
                if str(d.get("acceptance_test_plan_number") or "").strip()
            }
        )

        self.cb_program.clear()
        self.cb_asset.clear()
        self.cb_group.clear()

        self.cb_program.addItems(programs or ["(none)"])
        self.cb_asset.addItems(assets or ["(none)"])

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
            wanted = str(self.cb_program.currentText() or "").strip()
            return [d for d in docs if str(d.get("program_title") or "").strip() == wanted]
        if self.rb_asset.isChecked():
            wanted = str(self.cb_asset.currentText() or "").strip()
            return [d for d in docs if str(d.get("asset_type") or "").strip() == wanted]
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
            self.tbl.setItem(r, 5, QtWidgets.QTableWidgetItem(str(d.get("metadata_rel") or "")))
            self.tbl.setItem(r, 6, QtWidgets.QTableWidgetItem(str(d.get("similarity_group") or "")))

        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)

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
                item = self.tbl.item(r, 5)
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
        self.resize(1080, 720)
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
        self._mode = "curves"  # or "metrics"
        self._highlight_sn = ""
        self._auto_plots: list[dict] = []
        self._last_plot_def: dict | None = None
        self._auto_plot_path = self._project_dir / "auto_plots_test_data.json"

        self.setWindowTitle("Test Data - Trend / Analyze")
        self.resize(1180, 760)
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

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left panel: controls
        left = QtWidgets.QFrame()
        left.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }")
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        title = QtWidgets.QLabel("Test Data Trend / Analyze")
        title.setStyleSheet("font-size: 14px; font-weight: 700;")
        left_layout.addWidget(title)

        # Clear mode switcher (tabs hidden; these buttons are the main UI).
        switch_row = QtWidgets.QHBoxLayout()
        switch_lbl = QtWidgets.QLabel("Mode:")
        switch_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #334155;")
        self.btn_mode_curves = QtWidgets.QPushButton("Plot Curves")
        self.btn_mode_metrics = QtWidgets.QPushButton("Plot Metrics")
        for b in (self.btn_mode_curves, self.btn_mode_metrics):
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
        self.btn_mode_curves.clicked.connect(lambda: self._set_mode("curves"))
        self.btn_mode_metrics.clicked.connect(lambda: self._set_mode("metrics"))

        switch_row.addWidget(switch_lbl)
        switch_row.addWidget(self.btn_mode_curves)
        switch_row.addWidget(self.btn_mode_metrics)
        switch_row.addStretch(1)
        left_layout.addLayout(switch_row)

        cache_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_cache = QtWidgets.QPushButton("Build / Refresh Cache")
        self.btn_refresh_cache.setMinimumHeight(40)
        self.btn_refresh_cache.clicked.connect(lambda: self._load_cache(rebuild=True))
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
        self.lbl_cache = QtWidgets.QLabel("")
        self.lbl_cache.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_cache.setWordWrap(True)
        cache_row.addWidget(self.btn_refresh_cache)
        cache_row.addWidget(self.btn_plot)
        cache_row.addStretch(1)
        left_layout.addLayout(cache_row)
        left_layout.addWidget(self.lbl_cache)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self._runs_ex: list[dict] = []
        self._run_display_by_name: dict[str, str] = {}
        self._run_name_by_display: dict[str, str] = {}
        cfg = be.td_read_run_labeling_config(self._workbook_path)
        self._run_labeling_enabled = bool(isinstance(cfg, dict) and cfg.get("enabled"))

        self.cb_run = QtWidgets.QComboBox()
        self.cb_run.currentIndexChanged.connect(self._refresh_columns_for_run)
        form.addRow("Run:", self.cb_run)
        self.cb_show_run_labels = QtWidgets.QCheckBox("Show condition labels")
        self.cb_show_run_labels.setChecked(self._run_labeling_enabled)
        self.cb_show_run_labels.setEnabled(self._run_labeling_enabled)
        self.cb_show_run_labels.toggled.connect(lambda *_: self._refresh_run_dropdown())
        form.addRow("", self.cb_show_run_labels)
        left_layout.addLayout(form)

        tabs = QtWidgets.QTabWidget()
        try:
            tabs.tabBar().hide()
        except Exception:
            pass
        self._tabs = tabs

        # Metrics tab
        tab_metrics = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QVBoxLayout(tab_metrics)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(8)

        lbl_y_multi = QtWidgets.QLabel("Y Columns (multi-select)")
        lbl_y_multi.setStyleSheet("font-size: 12px; font-weight: 700; color: #334155;")
        metrics_layout.addWidget(lbl_y_multi)

        self.list_y_metrics = QtWidgets.QListWidget()
        self.list_y_metrics.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_y_metrics.itemSelectionChanged.connect(self._refresh_stats_preview)
        metrics_layout.addWidget(self.list_y_metrics, 1)

        y_sel_row = QtWidgets.QHBoxLayout()
        self.btn_y_select_all = QtWidgets.QPushButton("Select All")
        self.btn_y_clear_all = QtWidgets.QPushButton("Clear")
        self.btn_y_select_all.clicked.connect(lambda: self.list_y_metrics.selectAll())
        self.btn_y_clear_all.clicked.connect(lambda: self.list_y_metrics.clearSelection())
        y_sel_row.addWidget(self.btn_y_select_all)
        y_sel_row.addWidget(self.btn_y_clear_all)
        y_sel_row.addStretch(1)
        metrics_layout.addLayout(y_sel_row)

        lbl_stats = QtWidgets.QLabel("Stats (multi-select)")
        lbl_stats.setStyleSheet("font-size: 12px; font-weight: 700; color: #334155;")
        metrics_layout.addWidget(lbl_stats)

        self.list_stats = QtWidgets.QListWidget()
        self.list_stats.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_stats.setMaximumHeight(92)
        for st in ["mean", "min", "max", "std", "median", "count"]:
            self.list_stats.addItem(QtWidgets.QListWidgetItem(st))
        # Default selection: mean (matches prior single-select default behavior).
        for i in range(self.list_stats.count()):
            it = self.list_stats.item(i)
            if it.text().strip().lower() == "mean":
                it.setSelected(True)
                break
        metrics_layout.addWidget(self.list_stats)

        # Curves tab
        tab_curves = QtWidgets.QWidget()
        curves_layout = QtWidgets.QVBoxLayout(tab_curves)
        curves_layout.setContentsMargins(10, 10, 10, 10)
        curves_layout.setSpacing(8)

        row_y_curve = QtWidgets.QHBoxLayout()
        row_y_curve.addWidget(QtWidgets.QLabel("Y Column:"))
        self.cb_y_curve = QtWidgets.QComboBox()
        self.cb_y_curve.currentIndexChanged.connect(self._refresh_stats_preview)
        row_y_curve.addWidget(self.cb_y_curve, 1)
        curves_layout.addLayout(row_y_curve)

        row_curves = QtWidgets.QHBoxLayout()
        self.cb_x = QtWidgets.QComboBox()
        row_curves.addWidget(QtWidgets.QLabel("X Column:"))
        row_curves.addWidget(self.cb_x, 1)
        curves_layout.addLayout(row_curves)

        tabs.addTab(tab_metrics, "Metrics")
        tabs.addTab(tab_curves, "Curves")
        left_layout.addWidget(tabs)

        # Serial highlight (not selection/filtering for plotting)
        lbl_serials = QtWidgets.QLabel("Highlight Serial (optional)")
        lbl_serials.setStyleSheet("font-size: 12px; font-weight: 700;")
        left_layout.addWidget(lbl_serials)

        self.ed_filter_serials = QtWidgets.QLineEdit()
        self.ed_filter_serials.setPlaceholderText("Filter serials (e.g., SN0001)")
        self.ed_filter_serials.textChanged.connect(self._apply_serial_filter)
        left_layout.addWidget(self.ed_filter_serials)

        self.list_serials = QtWidgets.QListWidget()
        self.list_serials.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_serials.setMaximumHeight(140)
        self.list_serials.itemSelectionChanged.connect(self._on_highlight_changed)
        left_layout.addWidget(self.list_serials)

        splitter.addWidget(left)

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
        right_layout.addLayout(plot_header)

        self.plot_container = QtWidgets.QFrame()
        plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = None
        self._figure = None
        self._axes = None
        self._init_plot_area(plot_layout)
        right_layout.addWidget(self.plot_container, 6)

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
        stats_lay = QtWidgets.QHBoxLayout(self._stats_frame)
        stats_lay.setContentsMargins(10, 8, 10, 8)
        stats_lay.setSpacing(16)

        def _stat_pair(label: str) -> tuple[QtWidgets.QLabel, QtWidgets.QLabel]:
            k = QtWidgets.QLabel(label)
            k.setStyleSheet("color:#334155; font-size: 11px; font-weight: 800;")
            v = QtWidgets.QLabel("—")
            v.setStyleSheet("color:#0f172a; font-size: 11px; font-weight: 700;")
            return k, v

        self._stats_values: dict[str, QtWidgets.QLabel] = {}
        for key, label in (
            ("serial", "Serial"),
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
            stats_lay.addLayout(box)
            self._stats_values[key] = v_lbl
        stats_lay.addStretch(1)
        self._stats_frame.setVisible(False)
        right_layout.addWidget(self._stats_frame, 0)

        def _toggle_stats(opened: bool) -> None:
            self._stats_frame.setVisible(bool(opened))
            self.btn_stats_toggle.setText("Highlighted Serial Stats ▾" if opened else "Highlighted Serial Stats ▸")

        self.btn_stats_toggle.toggled.connect(_toggle_stats)
        _toggle_stats(False)

        # Bottom actions: saving + auto-plots (kept within plot panel; centered, not full-width)
        self.btn_add_auto_plot = QtWidgets.QPushButton("Add to Auto-Plots")
        self.btn_add_auto_plot.setEnabled(False)
        self.btn_add_auto_plot.clicked.connect(self._add_current_plot_to_autoplots)
        self.btn_save_plot_pdf = QtWidgets.QPushButton("Save Plot PDF")
        self.btn_save_plot_pdf.setEnabled(False)
        self.btn_save_plot_pdf.clicked.connect(self._save_plot_pdf)
        self.lbl_source = QtWidgets.QLabel("")
        self.lbl_source.setStyleSheet("color: #64748b; font-size: 11px;")
        self.lbl_source.setWordWrap(True)

        self.list_auto_plots = QtWidgets.QListWidget()
        self.list_auto_plots.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_auto_plots.itemDoubleClicked.connect(lambda *_: self._open_selected_auto_plot())
        self.list_auto_plots.itemSelectionChanged.connect(self._update_auto_actions)
        self.list_auto_plots.setMinimumHeight(110)
        self.list_auto_plots.setMaximumHeight(160)

        self.btn_open_auto = QtWidgets.QPushButton("Open")
        self.btn_open_auto.setEnabled(False)
        self.btn_open_auto.clicked.connect(self._open_selected_auto_plot)
        self.btn_open_all_auto = QtWidgets.QPushButton("Open All")
        self.btn_open_all_auto.setEnabled(False)
        self.btn_open_all_auto.clicked.connect(self._open_all_auto_plots_panel)
        self.btn_delete_auto = QtWidgets.QPushButton("Delete")
        self.btn_delete_auto.setEnabled(False)
        self.btn_delete_auto.clicked.connect(self._delete_selected_auto_plots)
        self.btn_save_all_auto = QtWidgets.QPushButton("Save All PDF")
        self.btn_save_all_auto.setEnabled(False)
        self.btn_save_all_auto.clicked.connect(self._save_all_auto_plots_pdf)

        self._actions_frame = QtWidgets.QFrame()
        self._actions_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; }"
        )
        self._actions_frame.setMinimumHeight(130)

        actions = QtWidgets.QVBoxLayout(self._actions_frame)
        actions.setContentsMargins(10, 8, 10, 8)
        actions.setSpacing(8)

        # Ordered buttons row (current plot actions, then auto-plot list actions).
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_add_auto_plot)
        btn_row.addWidget(self.btn_save_plot_pdf)
        btn_row.addSpacing(10)
        btn_row.addWidget(self.btn_open_auto)
        btn_row.addWidget(self.btn_open_all_auto)
        btn_row.addWidget(self.btn_delete_auto)
        btn_row.addWidget(self.btn_save_all_auto)
        btn_row.addStretch(1)
        actions.addLayout(btn_row)

        auto_lbl = QtWidgets.QLabel("Auto-Plots")
        auto_lbl.setStyleSheet("font-size: 12px; font-weight: 800; color: #0f172a;")
        actions.addWidget(auto_lbl)
        self.list_auto_plots.setMaximumHeight(120)
        self.list_auto_plots.setMinimumHeight(90)
        actions.addWidget(self.list_auto_plots, 1)

        actions.addWidget(self.lbl_source)
        right_layout.addWidget(self._actions_frame, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._load_cache(rebuild=False)
        self._load_auto_plots()
        self._set_mode("curves")

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

    def _load_cache(self, *, rebuild: bool) -> None:
        try:
            self._db_path = be.ensure_test_data_project_cache(
                self._project_dir, self._workbook_path, rebuild=bool(rebuild)
            )
            self.lbl_source.setText(str(self._db_path))
            self.lbl_cache.setText(f"Cache DB: {self._db_path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test Data Cache", str(exc))
            return
        self._refresh_from_cache()

    def _refresh_from_cache(self) -> None:
        if not self._db_path:
            return
        try:
            runs_ex = be.td_list_runs_ex(self._db_path)
            serials = be.td_list_serials(self._db_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test Data Cache", str(exc))
            return

        prev_ref = self.cb_run.currentData()
        if prev_ref is None:
            prev_ref = self.cb_run.currentText()
        prev_ref = str(prev_ref or "").strip()

        self._runs_ex = [d for d in (runs_ex or []) if isinstance(d, dict) and str(d.get("run_name") or "").strip()]
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

        has_any_display = any(bool(v) for v in self._run_display_by_name.values())
        if hasattr(self, "cb_show_run_labels"):
            self.cb_show_run_labels.setEnabled(bool(self._run_labeling_enabled) or has_any_display)
            if not (bool(self._run_labeling_enabled) or has_any_display):
                self.cb_show_run_labels.setChecked(False)

        prev_run = self._run_name_by_display.get(prev_ref, prev_ref)
        self._refresh_run_dropdown(prev_run_name=prev_run)

        self.list_serials.clear()
        for sn in serials:
            item = QtWidgets.QListWidgetItem(sn)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, sn)
            self.list_serials.addItem(item)

        # Default highlight: none (user can click one if desired).
        self._highlight_sn = ""

        self._refresh_columns_for_run()
        self._refresh_stats_preview()

    def _current_run_name(self) -> str:
        ref = self.cb_run.currentData() if hasattr(self, "cb_run") else None
        run = str(ref or (self.cb_run.currentText() if hasattr(self, "cb_run") else "") or "").strip()
        # Legacy auto-plots may have stored the display label instead of the underlying run_name.
        return str(self._run_name_by_display.get(run, run) or "").strip()

    def _run_display_text(self, run_name: str) -> str:
        rn = str(run_name or "").strip()
        if not rn:
            return ""
        dn = str((self._run_display_by_name or {}).get(rn) or "").strip()
        return dn or rn

    def _select_run_by_name(self, run_name: str) -> None:
        rn = str(run_name or "").strip()
        if not rn or not hasattr(self, "cb_run"):
            return
        idx = self.cb_run.findData(rn)
        if idx >= 0:
            self.cb_run.setCurrentIndex(idx)

    def _refresh_run_dropdown(self, prev_run_name: str | None = None) -> None:
        if not hasattr(self, "cb_run"):
            return
        if prev_run_name is None:
            prev_run_name = self._current_run_name()
        show = bool(getattr(self, "cb_show_run_labels", None) and self.cb_show_run_labels.isChecked())
        runs = [str(d.get("run_name") or "").strip() for d in (self._runs_ex or []) if str(d.get("run_name") or "").strip()]
        self.cb_run.blockSignals(True)
        self.cb_run.clear()
        for rn in runs:
            text = self._run_display_text(rn) if show else rn
            self.cb_run.addItem(text, rn)
        if prev_run_name:
            self._select_run_by_name(prev_run_name)
        self.cb_run.blockSignals(False)
        try:
            self._refresh_columns_for_run()
            self._refresh_stats_preview()
        except Exception:
            pass

    def _refresh_columns_for_run(self) -> None:
        if not self._db_path:
            return
        run = self._current_run_name()
        if not run:
            self.cb_y_curve.clear()
            self.cb_x.clear()
            self.list_y_metrics.clear()
            return
        try:
            y_cols = be.td_list_y_columns(self._db_path, run)
            x_cols = be.td_list_x_columns(self._db_path, run)
        except Exception:
            y_cols, x_cols = [], []

        prev_y = self.cb_y_curve.currentText()
        self.cb_y_curve.blockSignals(True)
        self.cb_y_curve.clear()
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
        self.cb_y_curve.addItems(y_names)
        if prev_y and prev_y in y_names:
            self.cb_y_curve.setCurrentText(prev_y)
        self.cb_y_curve.blockSignals(False)

        # Multi-select Y list for metrics mode
        prev_selected = {it.text() for it in self.list_y_metrics.selectedItems()} if hasattr(self, "list_y_metrics") else set()
        self.list_y_metrics.blockSignals(True)
        self.list_y_metrics.clear()
        for name in y_names:
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

        prev_x = self.cb_x.currentText()
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
            self.cb_x.setCurrentText(want)
        self.cb_x.blockSignals(False)

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

    def _apply_serial_filter(self) -> None:
        needle = (self.ed_filter_serials.text() or "").strip().lower()
        for i in range(self.list_serials.count()):
            item = self.list_serials.item(i)
            sn = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text() or "").lower()
            item.setHidden(bool(needle) and needle not in sn)

    def _on_highlight_changed(self) -> None:
        items = self.list_serials.selectedItems()
        sn = ""
        if items:
            sn = str(items[0].data(QtCore.Qt.ItemDataRole.UserRole) or items[0].text() or "").strip()
        self._highlight_sn = sn
        self._refresh_stats_preview()

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
        for st in ("count", "mean", "std", "min", "max"):
            val = None
            try:
                series = be.td_load_metric_series(self._db_path, run, y_col, st)
            except Exception:
                series = []
            for r in series:
                if str(r.get("serial") or "").strip() == sn:
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
            y_col = (getattr(self, "cb_y_curve", None).currentText() or "").strip() if hasattr(self, "cb_y_curve") else ""
        else:
            # Preview first selected Y column (keeps table compact).
            selected = [it.text() for it in self.list_y_metrics.selectedItems()] if hasattr(self, "list_y_metrics") else []
            y_col = (selected[0] if selected else "").strip()
        self._populate_stats_table(run, y_col, self._highlight_sn)

    def _set_mode(self, mode: str) -> None:
        m = str(mode or "").strip().lower()
        if m not in {"curves", "metrics"}:
            return
        self._mode = m
        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(0 if m == "metrics" else 1)
        self.btn_mode_curves.setChecked(m == "curves")
        self.btn_mode_metrics.setChecked(m == "metrics")
        self.btn_plot.setText("Plot Curves" if m == "curves" else "Plot Metrics")
        self._refresh_stats_preview()

    def _plot_current_mode(self) -> None:
        if self._mode == "metrics":
            self._plot_metrics()
        else:
            self._plot_curves()

    def _plot_metrics(self) -> None:
        if not self._plot_ready or not self._db_path:
            return
        run = self._current_run_name()
        stats = [it.text().strip().lower() for it in self.list_stats.selectedItems()] if hasattr(self, "list_stats") else []
        stats = [s for s in stats if s]
        if not run or not stats:
            return
        y_cols = [it.text().strip() for it in self.list_y_metrics.selectedItems()] if hasattr(self, "list_y_metrics") else []
        y_cols = [c for c in y_cols if c]
        if not y_cols:
            QtWidgets.QMessageBox.information(self, "Plot Metrics", "Select at least one Y column.")
            return

        # Always plot all serials.
        try:
            labels = be.td_list_serials(self._db_path)
        except Exception:
            labels = []
        if not labels:
            QtWidgets.QMessageBox.information(self, "Plot Metrics", "No serial numbers available.")
            return

        show = bool(getattr(self, "cb_show_run_labels", None) and self.cb_show_run_labels.isChecked())
        run_title = self._run_display_text(run) if show else run
        stats_label = "/".join(stats)
        y_label = stats[0] if len(stats) == 1 else "Metric value"
        self._axes.clear()
        self._axes.set_title(f"{run_title} — {stats_label}")
        self._axes.set_xlabel("Serial Number")
        self._axes.set_ylabel(y_label)
        x = list(range(len(labels)))
        any_plotted = False
        for y_col in y_cols:
            for stat in stats:
                try:
                    series = be.td_load_metric_series(self._db_path, run, y_col, stat)
                except Exception:
                    series = []
                vmap = {
                    str(r.get("serial") or "").strip(): r.get("value_num")
                    for r in series
                    if str(r.get("serial") or "").strip()
                }
                y = [
                    (float(vmap.get(sn)) if isinstance(vmap.get(sn), (int, float)) else float("nan"))
                    for sn in labels
                ]
                try:
                    line = self._axes.plot(x, y, marker="o", linewidth=1.4, label=f"{y_col}.{stat}")[0]
                    # Highlight the chosen serial with a larger marker (per series).
                    if self._highlight_sn and self._highlight_sn in labels:
                        idx = labels.index(self._highlight_sn)
                        self._axes.plot([x[idx]], [y[idx]], marker="o", markersize=9, color=line.get_color())
                    any_plotted = True
                except Exception:
                    continue
        if not any_plotted:
            QtWidgets.QMessageBox.information(self, "Plot Metrics", "No metric values found for this selection.")
            return
        self._axes.set_xticks(x)
        self._axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        self._axes.grid(True, alpha=0.25)
        try:
            self._axes.legend(fontsize=8, loc="best")
        except Exception:
            pass
        try:
            self._figure.tight_layout()
        except Exception:
            pass
        try:
            self._canvas.draw()
        except Exception:
            pass
        self.btn_save_plot_pdf.setEnabled(True)
        self._last_plot_def = {"mode": "metrics", "run": run, "stats": list(stats), "y": list(y_cols)}
        self.btn_add_auto_plot.setEnabled(True)
        # Keep table compact: preview first selected Y column (and highlight).
        self._refresh_stats_preview()

    def _plot_curves(self) -> None:
        if not self._plot_ready or not self._db_path:
            return
        run = self._current_run_name()
        y_col = (self.cb_y_curve.currentText() or "").strip() if hasattr(self, "cb_y_curve") else ""
        x_label = (self.cb_x.currentText() or "").strip()
        x_key = self.cb_x.currentData()
        x_col = (str(x_key or "") or x_label).strip()
        if not run or not y_col:
            return
        if not x_col:
            QtWidgets.QMessageBox.information(
                self,
                "Plot Curves",
                "No valid X column found.\n\nOnly Time / Pulse Number are supported for the X-axis (never excel_row).",
            )
            return
        try:
            # Always plot all serials.
            curves = be.td_load_curves(self._db_path, run, y_col, x_col, serials=None)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Plot Curves", str(exc))
            return

        if not curves:
            QtWidgets.QMessageBox.information(self, "Plot Curves", "No curve data found for this selection.")
            return

        self._axes.clear()
        show = bool(getattr(self, "cb_show_run_labels", None) and self.cb_show_run_labels.isChecked())
        run_title = self._run_display_text(run) if show else run
        self._axes.set_title(f"{run_title} — {y_col} vs {x_label or x_col}")
        self._axes.set_xlabel(x_label or x_col)
        self._axes.set_ylabel(y_col)
        for s in curves:
            sn = str(s.get("serial") or "").strip()
            xs = s.get("x") or []
            ys = s.get("y") or []
            if not isinstance(xs, list) or not isinstance(ys, list) or not xs or not ys:
                continue
            try:
                is_hi = bool(self._highlight_sn) and sn == self._highlight_sn
                self._axes.plot(
                    xs,
                    ys,
                    linewidth=(2.6 if is_hi else 1.1),
                    alpha=(1.0 if is_hi else 0.75),
                    label=sn or "SN",
                )
            except Exception:
                continue
        self._axes.grid(True, alpha=0.25)
        try:
            self._axes.legend(fontsize=8, loc="best")
        except Exception:
            pass
        try:
            self._figure.tight_layout()
        except Exception:
            pass
        try:
            self._canvas.draw()
        except Exception:
            pass
        self.btn_save_plot_pdf.setEnabled(True)
        self._last_plot_def = {"mode": "curves", "run": run, "x": (x_label or x_col), "y": [y_col]}
        self.btn_add_auto_plot.setEnabled(True)
        self._populate_stats_table(run, y_col, self._highlight_sn)

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
        self._refresh_auto_plots_list()

    def _refresh_auto_plots_list(self) -> None:
        if not hasattr(self, "list_auto_plots"):
            return
        self.list_auto_plots.clear()
        show = bool(getattr(self, "cb_show_run_labels", None) and self.cb_show_run_labels.isChecked())
        for d in self._auto_plots:
            name = str(d.get("name") or "").strip()
            if not name:
                mode = str(d.get("mode") or "").strip()
                run_ref = str(d.get("run") or "").strip()
                run = self._run_name_by_display.get(run_ref, run_ref)
                run_disp = self._run_display_text(run) if show else run
                if mode == "curves":
                    y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                    x = str(d.get("x") or "").strip()
                    name = f"Curves: {run_disp} {y} vs {x}".strip()
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
                    stats_label = "/".join(stats) if stats else "mean"
                    name = f"Metrics: {run_disp} {stats_label} ({y})".strip()
            item = QtWidgets.QListWidgetItem(name or "Auto-Plot")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, d)
            self.list_auto_plots.addItem(item)
        self._update_auto_actions()

    def _update_auto_actions(self) -> None:
        selected = self.list_auto_plots.selectedItems() if hasattr(self, "list_auto_plots") else []
        has = bool(selected)
        if hasattr(self, "btn_open_auto"):
            self.btn_open_auto.setEnabled(has and self._plot_ready and bool(self._db_path))
        if hasattr(self, "btn_open_all_auto"):
            self.btn_open_all_auto.setEnabled(bool(self._auto_plots) and self._plot_ready and bool(self._db_path))
        if hasattr(self, "btn_delete_auto"):
            self.btn_delete_auto.setEnabled(has)
        if hasattr(self, "btn_save_all_auto"):
            self.btn_save_all_auto.setEnabled(bool(self._auto_plots) and self._plot_ready and bool(self._db_path))

    def _add_current_plot_to_autoplots(self) -> None:
        if not self._last_plot_def:
            return
        d = dict(self._last_plot_def)
        if "name" not in d:
            show = bool(getattr(self, "cb_show_run_labels", None) and self.cb_show_run_labels.isChecked())
            if d.get("mode") == "curves":
                run = str(d.get("run") or "").strip()
                run_disp = self._run_display_text(run) if show else run
                y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                x = str(d.get("x") or "").strip()
                d["name"] = f"Curves: {run_disp} {y} vs {x}".strip()
            else:
                run = str(d.get("run") or "").strip()
                run_disp = self._run_display_text(run) if show else run
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
                stats_label = "/".join(stats) if stats else "mean"
                d["name"] = f"Metrics: {run_disp} {stats_label} ({y})".strip()
        self._auto_plots.append(d)
        try:
            self._auto_plot_path.write_text(json.dumps(self._auto_plots, indent=2), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Auto-Plots", str(exc))
            return
        self._refresh_auto_plots_list()

    def _open_selected_auto_plot(self) -> None:
        if not self._db_path or not self._plot_ready:
            return
        items = self.list_auto_plots.selectedItems() if hasattr(self, "list_auto_plots") else []
        if not items:
            return
        d = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(d, dict):
            return
        mode = str(d.get("mode") or "").strip().lower()
        if mode not in {"curves", "metrics"}:
            return
        run_ref = str(d.get("run") or "").strip()
        run = self._run_name_by_display.get(run_ref, run_ref)
        if run:
            self._select_run_by_name(run)
        self._set_mode(mode)
        if mode == "curves":
            ys = d.get("y") or []
            if isinstance(ys, list) and ys:
                self.cb_y_curve.setCurrentText(str(ys[0]))
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
                self.cb_x.setCurrentText(want)
            self._plot_curves()
        else:
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
            if want_stats and hasattr(self, "list_stats"):
                self.list_stats.clearSelection()
                for i in range(self.list_stats.count()):
                    it = self.list_stats.item(i)
                    if it.text().strip().lower() in want_stats:
                        it.setSelected(True)
            ys = d.get("y") or []
            want = {str(x).strip() for x in ys if str(x).strip()} if isinstance(ys, list) else set()
            if want and hasattr(self, "list_y_metrics"):
                self.list_y_metrics.clearSelection()
                for i in range(self.list_y_metrics.count()):
                    it = self.list_y_metrics.item(i)
                    if it.text() in want:
                        it.setSelected(True)
            self._plot_metrics()

    def _open_all_auto_plots_panel(self) -> None:
        if not self._plot_ready or not self._db_path:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "Plotting is unavailable.")
            return
        if not self._auto_plots:
            QtWidgets.QMessageBox.information(self, "Auto-Plots", "No auto-plots available.")
            return

        plots: list[tuple[str, dict]] = []
        show = bool(getattr(self, "cb_show_run_labels", None) and self.cb_show_run_labels.isChecked())
        for d in self._auto_plots:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "").strip()
            if not name:
                mode = str(d.get("mode") or "").strip()
                run_ref = str(d.get("run") or "").strip()
                run = self._run_name_by_display.get(run_ref, run_ref)
                run_disp = self._run_display_text(run) if show else run
                if mode == "curves":
                    y = ", ".join([str(x) for x in (d.get("y") or []) if str(x).strip()])
                    x = str(d.get("x") or "").strip()
                    name = f"Curves: {run_disp} {y} vs {x}".strip()
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
                    stats_label = "/".join(stats) if stats else "mean"
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

    def _delete_selected_auto_plots(self) -> None:
        items = self.list_auto_plots.selectedItems() if hasattr(self, "list_auto_plots") else []
        if not items:
            return
        to_del = []
        for it in items:
            d = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(d, dict):
                to_del.append(d)
        if not to_del:
            return
        self._auto_plots = [d for d in self._auto_plots if d not in to_del]
        try:
            self._auto_plot_path.write_text(json.dumps(self._auto_plots, indent=2), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Auto-Plots", str(exc))
            return
        self._refresh_auto_plots_list()

    def _render_plot_def_to_figure(self, d: dict):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        mode = str(d.get("mode") or "").strip().lower()
        run_ref = str(d.get("run") or "").strip()
        run = self._run_name_by_display.get(run_ref, run_ref)
        if mode == "curves":
            ys = d.get("y") or []
            y = str(ys[0] if isinstance(ys, list) and ys else "").strip()
            x_label = str(d.get("x") or "").strip()
            x = self._resolve_curve_x_key(run, x_label)
            ax.set_title(str(d.get("name") or "") or f"{run} — {y} vs {x_label or x}")
            ax.set_xlabel(x_label or x)
            ax.set_ylabel(y)
            curves = be.td_load_curves(self._db_path, run, y, x, serials=None)  # type: ignore[arg-type]
            for s in curves:
                sn = str(s.get("serial") or "").strip()
                xs = s.get("x") or []
                ys2 = s.get("y") or []
                if not isinstance(xs, list) or not isinstance(ys2, list) or not xs or not ys2:
                    continue
                ax.plot(xs, ys2, linewidth=1.1, alpha=0.85, label=sn or "SN")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        else:
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
            stats_label = "/".join(stats)
            ys = d.get("y") or []
            y_cols = [str(x).strip() for x in ys] if isinstance(ys, list) else []
            ax.set_title(str(d.get("name") or "") or f"{run} — {stats_label}")
            ax.set_xlabel("Serial Number")
            ax.set_ylabel(stats[0] if len(stats) == 1 else "Metric value")
            labels = be.td_list_serials(self._db_path)  # type: ignore[arg-type]
            x_idx = list(range(len(labels)))
            for y_col in y_cols:
                for stat in stats:
                    series = be.td_load_metric_series(self._db_path, run, y_col, stat)  # type: ignore[arg-type]
                    vmap = {
                        str(r.get("serial") or "").strip(): r.get("value_num")
                        for r in series
                        if str(r.get("serial") or "").strip()
                    }
                    yv = [
                        (float(vmap.get(sn)) if isinstance(vmap.get(sn), (int, float)) else float("nan"))
                        for sn in labels
                    ]
                    ax.plot(x_idx, yv, marker="o", linewidth=1.2, label=f"{y_col}.{stat}")
            ax.set_xticks(x_idx)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
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
        self._refresh_plot_series_after_worker = False
        self._auto_update_plot_terms_on_success = False
        self._plot_terms_pending = False
        self._plot_terms_pending_reason = ""

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
        self.btn_tab_projects = QtWidgets.QPushButton("Projects")

        for btn in [self.btn_tab_setup, self.btn_tab_files, self.btn_tab_projects]:
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
        self.tab_projects = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_setup, "Setup")
        self.tabs.addTab(self.tab_files, "Files")
        self.tabs.addTab(self.tab_projects, "Projects")

        # Connect tab buttons to switch content
        self.btn_tab_setup.clicked.connect(lambda: self._switch_tab(0))
        self.btn_tab_files.clicked.connect(lambda: self._switch_tab(1))
        self.btn_tab_projects.clicked.connect(lambda: self._switch_tab(2))

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
        self._setup_tab_projects()
        # Data Outputs tab replaced by top-level master button

        # Runtime
        self._worker: ProcWorker | None = None
        self._sync_worker: WorkspaceSyncWorker | None = None
        self._manager_worker: ManagerTaskWorker | None = None
        self._sync_popup_active = False
        self._manager_popup_active = False
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
        self.ed_files_filter.setPlaceholderText("Filter by filename, serial, program...")
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
        r.addLayout(filter_row)

        # Table
        cols = ["File Name", "Program", "Asset", "Serial", "Doc Type", "Report Date", "Test Date", "Status", "Certification"]
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

        # Action buttons
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.btn_files_open_pdf = QtWidgets.QPushButton("Open File")
        self.btn_files_open_metadata = QtWidgets.QPushButton("Open Metadata")
        self.btn_files_open_artifacts = QtWidgets.QPushButton("Open Artifacts")
        self.btn_files_open_combined = QtWidgets.QPushButton("Open combined.txt")
        self.btn_files_show_explorer = QtWidgets.QPushButton("Show in Explorer")
        self.btn_files_update_metadata = QtWidgets.QPushButton("Update Metadata")
        self.btn_files_certify_all = QtWidgets.QPushButton("Certify All")

        self.btn_files_open_combined.setToolTip("Open the merged OCR artifact combined.txt for this document (if present).")

        for b in (self.btn_files_open_pdf, self.btn_files_open_metadata,
                  self.btn_files_open_artifacts, self.btn_files_open_combined, self.btn_files_show_explorer,
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
        self.btn_files_update_metadata.clicked.connect(self._act_files_update_metadata)
        self.btn_files_certify_all.clicked.connect(self._act_files_certify_all)

        actions.addWidget(self.btn_files_open_pdf)
        actions.addWidget(self.btn_files_open_metadata)
        actions.addWidget(self.btn_files_open_artifacts)
        actions.addWidget(self.btn_files_open_combined)
        actions.addWidget(self.btn_files_show_explorer)
        actions.addWidget(self.btn_files_update_metadata)
        actions.addWidget(self.btn_files_certify_all)
        r.addLayout(actions)

        root.addWidget(left)
        root.addWidget(right, 1)

        # Store file data for filtering
        self._files_data: list[dict] = []
        self._files_filtered: list[dict] = []

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
        self.btn_project_env = QtWidgets.QPushButton("Project Env")
        self.btn_project_delete = QtWidgets.QPushButton("Delete Project")
        self.btn_project_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_project_open_workbook = QtWidgets.QPushButton("Open Workbook")
        for b in (
            self.btn_project_update,
            self.btn_project_env,
            self.btn_project_delete,
            self.btn_project_open_folder,
            self.btn_project_open_workbook,
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
        self.btn_project_env.clicked.connect(self._act_project_env)
        self.btn_project_delete.clicked.connect(self._act_delete_project)
        self.btn_project_open_folder.clicked.connect(self._act_open_project_folder)
        self.btn_project_open_workbook.clicked.connect(self._act_open_project_workbook)
        actions.addWidget(self.cb_project_overwrite)
        actions.addWidget(self.btn_project_update)
        actions.addWidget(self.btn_project_env)
        actions.addWidget(self.btn_project_delete)
        actions.addWidget(self.btn_project_open_folder)
        actions.addWidget(self.btn_project_open_workbook)
        r.addLayout(actions)

        root.addWidget(left)
        root.addWidget(right, 1)

        self._refresh_projects()

    def _refresh_projects(self) -> None:
        tbl = getattr(self, "tbl_projects", None)
        if not tbl:
            return
        repo_raw = (self.ed_global_repo.text() or "").strip() if hasattr(self, "ed_global_repo") else ""
        self.btn_project_new.setEnabled(bool(repo_raw))
        tbl.setRowCount(0)
        if not repo_raw:
            self.lbl_projects_status.setText("Select a Global Repo in Setup")
            self._update_project_actions()
            return

        repo = Path(repo_raw).expanduser()
        try:
            projects = be.list_eidat_projects(repo)
        except Exception as exc:
            self.lbl_projects_status.setText("Unable to read projects")
            self._append_log(f"[PROJECTS] Failed to load registry: {exc}")
            return

        for r, p in enumerate(projects):
            name = str(p.get("name") or "").strip()
            ptype = str(p.get("type") or "").strip()
            rel_folder = str(p.get("project_dir") or "").strip()
            rel_workbook = str(p.get("workbook") or "").strip()
            folder_abs = (repo / rel_folder).expanduser() if rel_folder and not Path(rel_folder).is_absolute() else Path(rel_folder).expanduser()
            workbook_abs = (repo / rel_workbook).expanduser() if rel_workbook and not Path(rel_workbook).is_absolute() else Path(rel_workbook).expanduser()

            tbl.insertRow(r)
            item_name = QtWidgets.QTableWidgetItem(name)
            item_type = QtWidgets.QTableWidgetItem(ptype)
            item_folder = QtWidgets.QTableWidgetItem(rel_folder)
            item_workbook = QtWidgets.QTableWidgetItem(rel_workbook)
            item_folder.setData(QtCore.Qt.ItemDataRole.UserRole, str(folder_abs))
            item_workbook.setData(QtCore.Qt.ItemDataRole.UserRole, str(workbook_abs))
            tbl.setItem(r, 0, item_name)
            tbl.setItem(r, 1, item_type)
            tbl.setItem(r, 2, item_folder)
            tbl.setItem(r, 3, item_workbook)

        tbl.resizeColumnsToContents()
        tbl.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.lbl_projects_status.setText(f"{len(projects)} project(s)")
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
        is_impl_supported = False
        is_update_supported = False
        if record:
            ptype = str(record.get("type") or "").strip()
            is_impl_supported = ptype in (
                getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"),
                getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"),
                getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"),
            )
            is_update_supported = is_impl_supported
        if hasattr(self, "btn_project_implementation"):
            self.btn_project_implementation.setEnabled(bool(record) and is_impl_supported)
        if hasattr(self, "btn_project_update"):
            self.btn_project_update.setEnabled(bool(record) and is_update_supported)
        if hasattr(self, "btn_project_env"):
            self.btn_project_env.setEnabled(bool(record))

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
            if ptype == getattr(be, "EIDAT_PROJECT_TYPE_TRENDING", "EIDP Trending"):
                payload = be.update_eidp_trending_project_workbook(repo, wb_path, overwrite=overwrite)
            elif ptype == getattr(be, "EIDAT_PROJECT_TYPE_RAW_TRENDING", "EIDP Raw File Trending"):
                payload = be.update_eidp_raw_trending_project_workbook(repo, wb_path, overwrite=overwrite)
            elif ptype == getattr(be, "EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING", "Test Data Trending"):
                payload = be.update_test_data_trending_project_workbook(repo, wb_path, overwrite=overwrite)
            else:
                raise RuntimeError(f"Unsupported project type: {ptype}")
            updated = int(payload.get("updated_cells") or 0)
            missing_src = int(payload.get("missing_source") or 0)
            missing_val = int(payload.get("missing_value") or 0)
            serials = int(payload.get("serials_in_workbook") or 0)
            have_src = int(payload.get("serials_with_source") or 0)
            serials_added = int(payload.get("serials_added") or 0)
            added_serials = payload.get("added_serials") or []
            dbg = str(payload.get("debug_json") or "").strip()
            self._append_log(
                f"[PROJECT UPDATE] updated={updated}, serials={serials}, added={serials_added}, sources={have_src}, missing_source={missing_src}, missing_value={missing_val}"
            )
            lines = [
                f"Updated cells: {updated}",
                f"Serials in workbook: {serials}",
            ]
            if serials_added:
                lines.append(f"Serials auto-added: {serials_added}")
            lines.extend(
                [
                    f"Serials with debug source: {have_src}",
                    f"Missing debug source: {missing_src}",
                    f"No value found: {missing_val}",
                    "",
                    f"Workbook: {payload.get('workbook') or wb_path}",
                ]
            )
            msg = "\n".join(lines)
            if serials_added and isinstance(added_serials, list) and added_serials:
                shown = ", ".join(str(s).strip() for s in added_serials[:20] if str(s).strip())
                if shown:
                    msg += f"\nAdded serials: {shown}" + (" ..." if len(added_serials) > 20 else "")
            if dbg:
                msg += f"\nDebug JSON: {dbg}"
            QtWidgets.QMessageBox.information(
                self,
                "Project Updated",
                msg,
            )
            self._show_toast(f"Project updated: {updated} cell(s)")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Update Project", str(exc))

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
        self.btn_tab_projects.setChecked(idx == 2)
        # Switch the actual tab
        self.tabs.setCurrentIndex(idx)

    def _on_tab_changed(self, idx: int):
        try:
            if self.tabs.widget(idx) is self.tab_files:
                self._refresh_files_tab()
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
        self._populate_files_table(self._files_data)

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

        self._populate_files_table(self._files_filtered)

    def _populate_files_table(self, files: list[dict]) -> None:
        """Populate the table with file data."""
        self.tbl_files.setSortingEnabled(False)
        self.tbl_files.setRowCount(0)

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

    def _apply_files_filter(self, text: str) -> None:
        """Filter table rows by search text."""
        text = text.strip().lower()
        for row in range(self.tbl_files.rowCount()):
            combined = " ".join(
                (self.tbl_files.item(row, c).text() if self.tbl_files.item(row, c) else "")
                for c in range(self.tbl_files.columnCount())
            ).lower()
            match = (text in combined) if text else True
            self.tbl_files.setRowHidden(row, not match)

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
            task=lambda: be.refresh_metadata_only(repo, rel_paths),
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
        act_update_metadata = menu.addAction("Update Metadata Only")
        menu.addSeparator()
        act_recertify = menu.addAction("Re-analyze Certification")

        act_open_pdf.triggered.connect(self._act_files_open_pdf)
        act_open_metadata.triggered.connect(self._act_files_open_metadata)
        act_open_artifacts.triggered.connect(self._act_files_open_artifacts)
        act_open_combined.triggered.connect(self._act_files_open_combined)
        act_show_explorer.triggered.connect(self._act_files_show_explorer)
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

    def _schedule_plot_terms_update(self, reason: str):
        label = "Refreshing available plot terms"
        if reason:
            label += f" ({reason})"
        if self._worker is not None and self._worker.isRunning():
            self._plot_terms_pending = True
            self._plot_terms_pending_reason = reason
            return
        self._plot_terms_pending = False
        self._plot_terms_pending_reason = ""
        self._refresh_plot_series_after_worker = True
        self._start_worker(be.generate_plot_terms, status_msg=label)

    def _start_worker(
        self,
        popen_factory,
        *,
        status_msg: str,
        show_run_progress: bool = False,
        refresh_plot_terms_after: bool = False,
        total_files: int | None = None,
    ):
        if self._worker is not None and self._worker.isRunning():
            return
        self._auto_update_plot_terms_on_success = bool(refresh_plot_terms_after)
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
            dlg.adjustSize()
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
        if getattr(self, "_refresh_plot_series_after_worker", False):
            self._refresh_plot_series_after_worker = False
            try:
                self._refresh_series_catalog()
            except Exception:
                pass
        auto_refresh = self._auto_update_plot_terms_on_success and not was_canceled and rc == 0
        self._auto_update_plot_terms_on_success = False
        if auto_refresh:
            self._schedule_plot_terms_update("latest scan")

        # Update master database after successful extraction run
        if is_extraction and rc == 0 and not was_canceled:
            try:
                self._append_log("[GUI] Compiling master database after extraction run...")
                be.compile_master()
                self._show_toast("Master database updated")
            except Exception as e:
                self._append_log(f"[ERROR] Failed to compile master database: {e}")

        self._scan_refresh()
        if self._plot_terms_pending and (self._worker is None or not self._worker.isRunning()):
            reason = self._plot_terms_pending_reason or "refresh"
            self._plot_terms_pending = False
            self._plot_terms_pending_reason = ""
            self._schedule_plot_terms_update(reason)

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
                refresh_plot_terms_after=True,
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
                refresh_plot_terms_after=True,
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
            refresh_plot_terms_after=True,
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
            refresh_plot_terms_after=True,
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
        self._schedule_plot_terms_update("workspace sync")

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

            # Refresh plot terms if needed
            self._schedule_plot_terms_update("master compilation")

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
            "Data files (*.pdf *.xlsx *.xlsm *.xls);;PDF files (*.pdf);;Excel files (*.xlsx *.xlsm *.xls);;All files (*.*)",
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
            self.status_bar.showMessage("No plot series found. Generate the term list first.", 4000)


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
