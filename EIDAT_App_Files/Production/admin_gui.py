#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

from . import admin_db
from .admin_runner import now_ns, run_pipeline

# Extraction V2 (new pipeline) — optional; the legacy dashboard is fully
# functional without it. Never wired to the legacy runner.
try:
    from extraction_v2 import admin_panel as _v2_panel  # type: ignore
except Exception:  # pragma: no cover - missing deps (rapidocr/cv2/...)
    _v2_panel = None


def _as_abs(p: str | Path) -> Path:
    path = Path(p).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _default_runtime_root() -> Path:
    # .../EIDAT_App_Files/Production/admin_gui.py -> runtime root is 2 levels up
    return Path(__file__).resolve().parents[2]


def _fmt_ts(epoch_ns: int | None) -> str:
    if not epoch_ns:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_ns / 1e9))
    except Exception:
        return str(epoch_ns)


class _Worker(QtCore.QThread):
    log = QtCore.Signal(str)
    status = QtCore.Signal(str)
    # Use `object` for timestamps to avoid PySide6/shiboken coercing to 32-bit C++ int.
    finished_one = QtCore.Signal(str, str, object, object, bool, str, object)  # node_id, action, started_ns, finished_ns, ok, error, summary

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: list[tuple[str, str, str, bool, str, dict[str, object]]] = []

    def set_tasks(self, tasks: list[tuple]) -> None:
        normalized: list[tuple[str, str, str, bool, str, dict[str, object]]] = []
        for task in tasks:
            if len(task) == 5:
                node_id, node_root, runtime_root, node_env_enabled, action = task
                options: dict[str, object] = {}
            else:
                node_id, node_root, runtime_root, node_env_enabled, action, raw_options = task
                options = dict(raw_options or {}) if isinstance(raw_options, dict) else {}
            normalized.append((str(node_id), str(node_root), str(runtime_root), bool(node_env_enabled), str(action), options))
        self._tasks = normalized

    def run(self) -> None:  # type: ignore[override]
        from .admin_runner import run_scan_force_candidates

        for node_id, node_root, runtime_root, node_env_enabled, action, options in self._tasks:
            def _emit(line: str) -> None:
                s = str(line or "").strip()
                if not s:
                    return
                self.log.emit(s)
                # Keep the popup status short and always reflect the most recent event.
                try:
                    self.status.emit(f"{Path(node_root).name}: {s}")
                except Exception:
                    self.status.emit(s)

            self.log.emit(f"[RUN] {node_root}")
            self.status.emit(f"{Path(node_root).name}: starting…")
            started = now_ns()
            if action == "scan_force_candidates":
                _emit("[ACTION] scan -> force candidates only")
                res = run_scan_force_candidates(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    node_env_enabled=node_env_enabled,
                    on_log=_emit,
                )
            elif action == "force_single_file":
                from .admin_runner import run_force_files

                raw_paths = options.get("file_paths")
                if isinstance(raw_paths, (list, tuple)):
                    file_paths = [str(p or "").strip() for p in raw_paths if str(p or "").strip()]
                else:
                    file_paths = []
                single = str(options.get("file_path") or "").strip()
                if single and single not in file_paths:
                    file_paths.append(single)
                _emit(f"[ACTION] force update {len(file_paths)} source file(s)")
                res = run_force_files(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    file_paths=file_paths,
                    node_env_enabled=node_env_enabled,
                    on_log=_emit,
                )
            elif action == "update_processor":
                from .admin_runner import run_update_processor

                _emit("[ACTION] update processor envs using current runtime code (keep caches)")
                res = run_update_processor(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    node_env_enabled=node_env_enabled,
                    on_log=_emit,
                )
            elif action == "update_projects":
                from .admin_runner import run_update_projects

                projects_raw = options.get("projects") or []
                projects = [dict(item or {}) for item in projects_raw if isinstance(item, dict)] if isinstance(projects_raw, list) else []
                overwrite = bool(options.get("overwrite"))
                force_project_rebuild = bool(options.get("force_project_rebuild"))
                repair_serials = bool(options.get("repair_serials"))
                rebuild = bool(options.get("rebuild"))
                auto_rebuild_on_failure = bool(options.get("auto_rebuild_on_failure", True))
                _emit(
                    "[ACTION] update projects via backend processor "
                    f"(selected={len(projects)} overwrite={overwrite} force_rebuild={force_project_rebuild} "
                    f"repair_serials={repair_serials} rebuild={rebuild} auto_rebuild_on_failure={auto_rebuild_on_failure})"
                )
                res = run_update_projects(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    projects=projects,
                    overwrite=overwrite,
                    force_project_rebuild=force_project_rebuild,
                    repair_serials=repair_serials,
                    rebuild=rebuild,
                    auto_rebuild_on_failure=auto_rebuild_on_failure,
                    on_log=_emit,
                )
            else:
                from .admin_runner import run_pipeline

                _emit("[ACTION] pipeline (scan -> process -> index)")
                res = run_pipeline(
                    node_root=node_root,
                    runtime_root=runtime_root,
                    node_env_enabled=node_env_enabled,
                    on_log=_emit,
                )
            finished = now_ns()
            summary = dict(res.outputs.get(action) or {}) if isinstance(res.outputs.get(action), dict) else {}
            if res.ok:
                if action == "update_processor":
                    _emit("[OK] update_processor")
                elif action == "update_projects":
                    try:
                        selected = int((summary or {}).get("selected_count", 0))
                        succeeded = int((summary or {}).get("succeeded_count", 0))
                        failed = int((summary or {}).get("failed_count", 0))
                        rebuilt = int((summary or {}).get("rebuilt_count", 0))
                    except Exception:
                        selected = succeeded = failed = rebuilt = 0
                    _emit(
                        f"[OK] update_projects selected={selected} succeeded={succeeded} failed={failed}"
                        + (f" rebuilt={rebuilt}" if rebuilt else "")
                    )
                elif action == "force_single_file":
                    proc = res.outputs.get("process") or {}
                    sources = res.outputs.get("source_files") or []
                    try:
                        ok = int((proc or {}).get("processed_ok", 0))
                        bad = int((proc or {}).get("processed_failed", 0))
                    except Exception:
                        ok = bad = 0
                    suffix = ""
                    try:
                        if isinstance(sources, list) and len(sources) == 1:
                            rel = str((sources[0] or {}).get("rel_path") or "").strip()
                            suffix = f" file={rel}" if rel else ""
                        elif isinstance(sources, list) and sources:
                            suffix = f" files={len(sources)}"
                    except Exception:
                        suffix = ""
                    _emit(f"[OK] force_files processed_ok={ok} processed_failed={bad}{suffix}")
                else:
                    scan = res.outputs.get("scan") or {}
                    proc = res.outputs.get("process") or {}
                    idx = res.outputs.get("index") or {}
                    try:
                        cand = int((scan or {}).get("candidates_count", 0))
                        ok = int((proc or {}).get("processed_ok", 0))
                        bad = int((proc or {}).get("processed_failed", 0))
                        groups = int((idx or {}).get("groups_count", 0))
                    except Exception:
                        cand = ok = bad = groups = 0
                    _emit(f"[OK] candidates={cand} processed_ok={ok} processed_failed={bad} groups={groups}")
                self.finished_one.emit(node_id, action, started, finished, True, "", summary)
            else:
                _emit(f"[FAIL] {res.error}")
                self.finished_one.emit(node_id, action, started, finished, False, res.error or "Unknown error", summary)


class _RunStatusDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Status")
        self.resize(760, 420)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        layout = QtWidgets.QVBoxLayout(self)
        self.lbl_heading = QtWidgets.QLabel("Running…")
        self.lbl_heading.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        layout.addWidget(self.lbl_heading)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #334155;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        layout.addWidget(self.progress)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(500)
        layout.addWidget(self.txt, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def begin(self, heading: str) -> None:
        self.lbl_heading.setText(str(heading or "Running…"))
        self.lbl_status.setText("")
        self.txt.clear()
        self.progress.setRange(0, 0)
        self.btn_close.setEnabled(False)
        self.show()
        self.raise_()
        self.activateWindow()

    def append_line(self, line: str) -> None:
        s = str(line or "").strip()
        if not s:
            return
        self.txt.appendPlainText(s)

    def set_status(self, status: str) -> None:
        self.lbl_status.setText(str(status or ""))

    def finish(self, summary: str = "") -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if summary:
            self.lbl_status.setText(summary)
        self.btn_close.setEnabled(True)


class _ProjectUpdatePickerDialog(QtWidgets.QDialog):
    def __init__(self, *, node_name: str, projects: list[dict[str, object]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Projects")
        self.resize(860, 540)
        self._projects = [dict(item or {}) for item in (projects or []) if isinstance(item, dict)]

        layout = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel(f"Select projects to update for {node_name}")
        heading.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        layout.addWidget(heading)

        hint = QtWidgets.QLabel(
            "Run project updates from the backend processor path. "
            "Force full rebuild / Rebuild from scratch apply only to Test Data Trending projects. "
            "Rebuild from scratch discards the project cache files, regenerates unreadable workbooks "
            "(damaged files are kept as *.corrupt-*.bak) and runs a full update; it is also applied "
            "automatically when an update fails on an unreadable workbook or an incomplete/corrupt cache DB."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 12px; color: #475569;")
        layout.addWidget(hint)

        self.tbl = QtWidgets.QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Name", "Type", "Workbook"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setWordWrap(False)
        self.tbl.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.tbl.setColumnWidth(0, 220)
        self.tbl.setColumnWidth(1, 220)
        self.tbl.setColumnWidth(2, 380)
        layout.addWidget(self.tbl, 1)

        opts = QtWidgets.QHBoxLayout()
        self.cb_overwrite = QtWidgets.QCheckBox("Overwrite existing cells")
        self.cb_force_rebuild = QtWidgets.QCheckBox("Force full rebuild")
        self.cb_force_rebuild.setToolTip("Applies only to Test Data Trending projects.")
        self.cb_rebuild = QtWidgets.QCheckBox("Rebuild from scratch")
        self.cb_rebuild.setToolTip(
            "Test Data Trending projects only: discard cache DBs/mirrors, regenerate any unreadable "
            "project/support workbook, then run a full update. Use when Update Project keeps failing "
            "with 'file is not a zip file' or 'cache DB is incomplete'."
        )
        self.cb_auto_rebuild = QtWidgets.QCheckBox("Auto-rebuild on file/cache errors")
        self.cb_auto_rebuild.setChecked(True)
        self.cb_auto_rebuild.setToolTip(
            "If an update fails because a workbook is unreadable or the cache DB is missing/incomplete/corrupt, "
            "retry that project once as a full rebuild instead of reporting a failure."
        )
        self.cb_repair_serials = QtWidgets.QCheckBox("Detect && repair broken serials")
        self.cb_repair_serials.setChecked(True)
        self.cb_repair_serials.setToolTip(
            "Test Data Trending projects only: before updating, scan for serials that are "
            "corrupted, stuck at 'missing data', or extracted on disk but not visible in the UI. "
            "Broken serials are purged from the project caches so this update re-creates them "
            "from scratch (corrupted extractions are re-extracted from their source documents)."
        )
        opts.addWidget(self.cb_overwrite)
        opts.addWidget(self.cb_force_rebuild)
        opts.addWidget(self.cb_rebuild)
        opts.addWidget(self.cb_auto_rebuild)
        opts.addWidget(self.cb_repair_serials)
        opts.addStretch(1)
        layout.addLayout(opts)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.btn_ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self.btn_ok is not None:
            self.btn_ok.setText("Run Update")
            self.btn_ok.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.tbl.itemSelectionChanged.connect(self._refresh_buttons)
        self._load_projects()

    def _load_projects(self) -> None:
        self.tbl.setRowCount(0)
        for row, project in enumerate(self._projects):
            self.tbl.insertRow(row)
            for col, key in enumerate(("name", "type", "workbook")):
                value = str(project.get(key) or "").strip()
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                if col == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(project))
                self.tbl.setItem(row, col, item)
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        if self.btn_ok is not None:
            self.btn_ok.setEnabled(bool(self.selected_projects()))

    def selected_projects(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        seen_rows: set[int] = set()
        for item in self.tbl.selectedItems():
            row = int(item.row())
            if row in seen_rows:
                continue
            seen_rows.add(row)
            first = self.tbl.item(row, 0)
            payload = dict(first.data(QtCore.Qt.ItemDataRole.UserRole) or {}) if first is not None else {}
            if payload:
                out.append(payload)
        return out


class _TdSerialManagerDialog(QtWidgets.QDialog):
    """Browse a Test Data Trending project's cached programs/serials and purge selected serials.

    Purging removes all cached rows for the serial from the project's
    implementation and raw-cache DBs and (optionally) deletes the serial's
    official extraction artifacts, so the next Update Projects run re-creates
    the serial completely from scratch instead of re-using cached data.
    """

    def __init__(self, *, node_name: str, node_root: str, projects: list[dict[str, object]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Data Serials")
        self.resize(980, 620)
        self._node_root = str(node_root)
        self._projects = [dict(item or {}) for item in (projects or []) if isinstance(item, dict)]
        self.purged_any = False
        self.last_purged_project: dict[str, object] | None = None

        layout = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel(f"Test Data serial numbers for {node_name}")
        heading.setStyleSheet("font-size: 14px; font-weight: 700; color: #0f172a;")
        layout.addWidget(heading)

        hint = QtWidgets.QLabel(
            "Select one or more serial numbers and delete them to purge every cached row and "
            "extraction artifact for that serial. The next Update Projects run then re-creates "
            "the serial from scratch from its source document (no cache re-use)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 12px; color: #475569;")
        layout.addWidget(hint)

        proj_row = QtWidgets.QHBoxLayout()
        proj_row.addWidget(QtWidgets.QLabel("Project:"))
        self.cmb_project = QtWidgets.QComboBox()
        for project in self._projects:
            self.cmb_project.addItem(str(project.get("name") or project.get("project_dir") or "?"), dict(project))
        proj_row.addWidget(self.cmb_project, 1)
        self.btn_reload = QtWidgets.QPushButton("Refresh List")
        proj_row.addWidget(self.btn_reload)
        layout.addLayout(proj_row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Program / Serial", "Serial Number", "Asset", "Status", "Cached Artifacts"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setColumnWidth(0, 340)
        self.tree.setColumnWidth(1, 160)
        self.tree.setColumnWidth(2, 160)
        self.tree.setColumnWidth(3, 110)
        layout.addWidget(self.tree, 1)

        self.cb_delete_artifacts = QtWidgets.QCheckBox(
            "Also delete extracted serial artifacts (Test Data File Extractions)"
        )
        self.cb_delete_artifacts.setChecked(True)
        self.cb_delete_artifacts.setToolTip(
            "Deletes the serial's folder under EIDAT Support\\Test Data File Extractions so the "
            "extraction is rebuilt from the source document, not re-materialized from cache."
        )
        layout.addWidget(self.cb_delete_artifacts)

        btn_row = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color:#475569;")
        btn_row.addWidget(self.lbl_status, 1)
        self.btn_select_all = QtWidgets.QPushButton("Select All")
        self.btn_find_problems = QtWidgets.QPushButton("Find Problem Serials")
        self.btn_find_problems.setToolTip(
            "Scan this project for serials that are corrupted, stuck at 'missing data', or "
            "extracted on disk but not visible in the UI, and select them in the list."
        )
        self.btn_delete = QtWidgets.QPushButton("Delete Selected Serials…")
        self.btn_delete.setEnabled(False)
        self.btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_find_problems)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.btn_close.clicked.connect(self.accept)
        self.btn_reload.clicked.connect(self.reload_serials)
        self.cmb_project.currentIndexChanged.connect(lambda _=0: self.reload_serials())
        self.tree.itemSelectionChanged.connect(self._refresh_buttons)
        self.btn_select_all.clicked.connect(self._act_select_all)
        self.btn_find_problems.clicked.connect(self._act_find_problems)
        self.btn_delete.clicked.connect(self._act_delete_selected)

        self.reload_serials()

    def current_project(self) -> dict[str, object]:
        payload = self.cmb_project.currentData()
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def reload_serials(self) -> None:
        from . import node_backend

        self.tree.clear()
        project = self.current_project()
        project_dir = str(project.get("project_dir") or "").strip()
        if not project_dir:
            self.lbl_status.setText("No project selected.")
            self._refresh_buttons()
            return
        try:
            rows = list(node_backend.list_project_serials(self._node_root, project_dir=project_dir) or [])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test Data Serials", str(exc))
            rows = []
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for row in rows:
            program = str(row.get("program_title") or "").strip() or "(Unknown Program)"
            parent = groups.get(program)
            if parent is None:
                parent = QtWidgets.QTreeWidgetItem([program, "", "", "", ""])
                parent.setFlags(parent.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                self.tree.addTopLevelItem(parent)
                parent.setFirstColumnSpanned(True)
                groups[program] = parent
            asset = " / ".join(
                part
                for part in (
                    str(row.get("asset_type") or "").strip(),
                    str(row.get("asset_specific_type") or "").strip(),
                )
                if part
            )
            artifacts = "on disk" if row.get("official_artifacts_exists") else "missing"
            item = QtWidgets.QTreeWidgetItem(
                [
                    str(row.get("serial") or "").strip(),
                    str(row.get("source_serial_number") or "").strip(),
                    asset,
                    str(row.get("status") or "").strip(),
                    artifacts,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, dict(row))
            item.setToolTip(0, str(row.get("serial") or "").strip())
            item.setToolTip(4, str(row.get("official_artifacts_dir") or "").strip())
            parent.addChild(item)
        self.tree.expandAll()
        count = sum(parent.childCount() for parent in groups.values())
        self.lbl_status.setText(f"{count} cached serial source(s) in {len(groups)} program(s).")
        self._refresh_buttons()

    def selected_serials(self) -> list[str]:
        out: list[str] = []
        for item in self.tree.selectedItems():
            payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict):
                serial = str(payload.get("serial") or "").strip()
                if serial and serial not in out:
                    out.append(serial)
        return out

    def _refresh_buttons(self) -> None:
        self.btn_delete.setEnabled(bool(self.selected_serials()))

    def _iter_serial_items(self):
        for top in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(top)
            if parent is None:
                continue
            for child in range(parent.childCount()):
                item = parent.child(child)
                if item is not None:
                    yield item

    def _act_select_all(self) -> None:
        self.tree.clearSelection()
        for item in self._iter_serial_items():
            item.setSelected(True)

    def _act_find_problems(self) -> None:
        from . import node_backend

        project = self.current_project()
        project_dir = str(project.get("project_dir") or "").strip()
        if not project_dir:
            return
        try:
            issues = [dict(i or {}) for i in (node_backend.list_project_serial_issues(self._node_root, project_dir=project_dir) or [])]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Find Problem Serials", str(exc))
            return
        if not issues:
            self.lbl_status.setText("No problem serials detected.")
            QtWidgets.QMessageBox.information(
                self, "Find Problem Serials", "No corrupted, stuck, or invisible serials were detected."
            )
            return

        issue_keys: set[str] = set()
        for item in issues:
            for key in (item.get("serial"), item.get("serial_number")):
                text = str(key or "").strip()
                if text:
                    issue_keys.add(text)

        self.tree.clearSelection()
        selected_count = 0
        for tree_item in self._iter_serial_items():
            payload = tree_item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if not isinstance(payload, dict):
                continue
            keys = {
                str(payload.get("serial") or "").strip(),
                str(payload.get("source_serial_number") or "").strip(),
            }
            keys.discard("")
            if keys & issue_keys:
                tree_item.setSelected(True)
                selected_count += 1

        lines: list[str] = []
        for item in issues[:25]:
            serial = str(item.get("serial") or item.get("serial_number") or "?").strip()
            lines.append(f"  [{item.get('issue')}] {serial}: {item.get('detail')}")
        if len(issues) > 25:
            lines.append(f"  … and {len(issues) - 25} more")
        not_listed = len(issues) - selected_count
        extra = ""
        if not_listed > 0:
            extra = (
                f"\n\n{not_listed} issue(s) have no row in this list (e.g. extracted on disk but never "
                "cached). Run Update Projects with 'Detect & repair broken serials' enabled to fix them."
            )
        self.lbl_status.setText(f"{len(issues)} problem serial(s) found; {selected_count} selected in the list.")
        QtWidgets.QMessageBox.information(
            self,
            "Find Problem Serials",
            (
                f"Detected {len(issues)} problem serial(s); selected {selected_count} in the list.\n\n"
                + "\n".join(lines)
                + "\n\nDelete the selected serials to purge their caches, then run Update Projects to "
                "re-create them from scratch."
                + extra
            ),
        )

    def _act_delete_selected(self) -> None:
        from . import node_backend

        project = self.current_project()
        project_dir = str(project.get("project_dir") or "").strip()
        serials = self.selected_serials()
        if not project_dir or not serials:
            return
        preview = "\n".join(f"  {serial}" for serial in serials[:20])
        if len(serials) > 20:
            preview += f"\n  … and {len(serials) - 20} more"
        artifacts_note = (
            "Their extracted serial artifacts (Test Data File Extractions) will also be deleted.\n\n"
            if self.cb_delete_artifacts.isChecked()
            else "Extracted serial artifacts are kept (cache rows only).\n\n"
        )
        resp = QtWidgets.QMessageBox.question(
            self,
            "Delete Serials",
            (
                f"Delete {len(serials)} serial(s) and ALL of their cached data from "
                f"project '{project.get('name')}'?\n\n{preview}\n\n"
                + artifacts_note
                + "Source documents (EIDPs / TD reports) are NOT deleted. The next "
                "Update Projects run re-creates these serials from scratch.\n\nContinue?"
            ),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            result = node_backend.purge_project_serials(
                self._node_root,
                project_dir=project_dir,
                serials=serials,
                delete_official_artifacts=bool(self.cb_delete_artifacts.isChecked()),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete Serials", str(exc))
            return
        self.purged_any = True
        self.last_purged_project = dict(project)
        impl_rows = int(result.get("impl_rows_deleted") or 0)
        raw_rows = int(result.get("raw_rows_deleted") or 0)
        art_dirs = list(result.get("deleted_artifact_dirs") or [])
        errors = [str(e) for e in (result.get("errors") or [])]
        summary = (
            f"Purged {len(serials)} serial(s): {impl_rows} project cache row(s), "
            f"{raw_rows} raw cache row(s), {len(art_dirs)} artifact folder(s) deleted."
        )
        if errors:
            summary += "\n\nWarnings:\n" + "\n".join(f"  {e}" for e in errors[:10])
        QtWidgets.QMessageBox.information(self, "Delete Serials", summary)
        self.reload_serials()


class AdminWindow(QtWidgets.QMainWindow):
    def __init__(self, *, runtime_root: Path, registry_path: Path):
        super().__init__()
        self._runtime_root = _as_abs(runtime_root)
        self._registry_path = _as_abs(registry_path)
        self._conn = admin_db.connect_registry(self._registry_path)

        self.setWindowTitle("EDAT Admin Dashboard (Nodes)")
        self.resize(1200, 780)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)

        header = QtWidgets.QHBoxLayout()
        self.ed_runtime = QtWidgets.QLineEdit(str(self._runtime_root))
        self.ed_runtime.setReadOnly(True)
        self.ed_registry = QtWidgets.QLineEdit(str(self._registry_path))
        self.ed_registry.setReadOnly(True)
        header.addWidget(QtWidgets.QLabel("Runtime Root:"))
        header.addWidget(self.ed_runtime, 2)
        header.addWidget(QtWidgets.QLabel("Registry:"))
        header.addWidget(self.ed_registry, 3)
        layout.addLayout(header)

        actions = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add/Deploy Node…")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_remove = QtWidgets.QPushButton("Remove From List")
        self.lbl_proc_cfg = QtWidgets.QLabel("Process config: Node .env overrides (force / limit / DPI)")
        self.lbl_proc_cfg.setStyleSheet("color:#475569;")
        self.lbl_proc_cfg.setToolTip("Open Node .env to set process force, limit, and DPI overrides.")
        actions.addWidget(self.btn_add)
        actions.addWidget(self.btn_refresh)
        actions.addWidget(self.btn_remove)
        actions.addStretch(1)
        actions.addWidget(self.lbl_proc_cfg, 1)
        layout.addLayout(actions)

        self.tbl = QtWidgets.QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["Enabled", "Node Root", "Status", "Last Run", "Env", "Actions", "Extraction V2"])
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionsMovable(True)
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        self.tbl.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.tbl.setWordWrap(False)
        self.tbl.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.tbl.verticalHeader().setDefaultSectionSize(88)
        self.tbl.setColumnWidth(0, 70)
        self.tbl.setColumnWidth(1, 260)
        self.tbl.setColumnWidth(2, 240)
        self.tbl.setColumnWidth(3, 160)
        self.tbl.setColumnWidth(4, 220)
        self.tbl.setColumnWidth(5, 620)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        layout.addWidget(self.tbl, 2)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        layout.addWidget(self.log, 1)

        self.worker = _Worker(self)
        self.worker.log.connect(self._append_log)
        self.worker.finished_one.connect(self._on_finished_one)
        self.worker.status.connect(self._on_worker_status)
        self.worker.finished.connect(self._on_worker_all_done)
        self._run_status = _RunStatusDialog(self)

        self.btn_add.clicked.connect(self._act_add_deploy)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_remove.clicked.connect(self._act_remove)
        # Per-node processing is initiated from each row's "Process" button.

        self.tbl.itemChanged.connect(self._on_item_changed)

        self.refresh()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        try:
            self._conn.close()
        except Exception:
            pass
        super().closeEvent(event)

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)
        try:
            if self._run_status.isVisible():
                self._run_status.append_line(line)
        except Exception:
            pass

    def _on_worker_status(self, status: str) -> None:
        try:
            if self._run_status.isVisible():
                self._run_status.set_status(status)
        except Exception:
            pass

    def _on_worker_all_done(self) -> None:
        try:
            if self._run_status.isVisible():
                self._run_status.finish("Run complete.")
        except Exception:
            pass

    def _default_scanner_env_path(self) -> Path:
        return self._runtime_root / "user_inputs" / "scanner.env"

    def _node_env_path(self, node_root: str | Path) -> Path:
        return _as_abs(node_root) / "EIDAT" / "UserData" / ".env"

    def _has_env_kv(self, path: Path) -> bool:
        try:
            for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith(";"):
                    continue
                if "=" in line:
                    k = line.split("=", 1)[0].strip()
                    if k:
                        return True
        except Exception:
            return False
        return False

    def _ensure_node_env_file(self, node_root: str | Path) -> Path:
        path = self._node_env_path(node_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        default_env = self._default_scanner_env_path()

        # Create as an exact copy of scanner.env so behavior matches default until user edits this file.
        # If an older placeholder file exists with no KEY=VALUE entries, upgrade it to the copied contents.
        should_seed = (not path.exists()) or (path.exists() and not self._has_env_kv(path))
        if should_seed and default_env.exists():
            try:
                path.write_bytes(default_env.read_bytes())
            except Exception:
                pass
        elif not path.exists():
            # Fallback template if scanner.env is missing.
            text = (
                "# Node-level overrides (KEY=VALUE)\n"
                "#\n"
                "# This file is the per-node processing config (KEY=VALUE).\n"
                "# It is initially seeded from the Central Runtime user_inputs/scanner.env.\n"
                "\n"
            )
            try:
                path.write_text(text, encoding="utf-8")
            except Exception:
                pass
        return path

    def _open_path(self, path: Path) -> None:
        p = _as_abs(path)
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))

    def _act_open_node_env(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        try:
            env_path = self._ensure_node_env_file(n.node_root)
            admin_db.set_node_env_enabled(self._conn, node_id=node_id, enabled=True)
            self.refresh()
            self._open_path(env_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Node .env", str(exc))

    def _selected_node_id(self) -> str | None:
        row = self.tbl.currentRow()
        if row < 0:
            return None
        item = self.tbl.item(row, 1)
        if not item:
            return None
        return str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip() or None

    def refresh(self) -> None:
        self.tbl.blockSignals(True)
        try:
            nodes = admin_db.list_nodes(self._conn)
            self.tbl.setRowCount(0)
            for r, n in enumerate(nodes):
                self.tbl.insertRow(r)

                it_enabled = QtWidgets.QTableWidgetItem("")
                it_enabled.setFlags(it_enabled.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it_enabled.setCheckState(QtCore.Qt.CheckState.Checked if n.enabled else QtCore.Qt.CheckState.Unchecked)
                self.tbl.setItem(r, 0, it_enabled)

                it_root = QtWidgets.QTableWidgetItem(n.node_root)
                it_root.setData(QtCore.Qt.ItemDataRole.UserRole, n.node_id)
                it_root.setToolTip(n.node_root)
                it_root.setFlags(it_root.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.tbl.setItem(r, 1, it_root)

                statusw = QtWidgets.QWidget()
                sl = QtWidgets.QHBoxLayout(statusw)
                sl.setContentsMargins(0, 0, 0, 0)
                sl.setSpacing(6)
                lbl = QtWidgets.QLabel((n.last_run_status or "idle").strip() or "idle")
                lbl.setStyleSheet("color:#0f172a;")
                btn_scan_force = QtWidgets.QPushButton("Scan+Force Candidates")
                btn_scan_force.setMaximumWidth(170)
                btn_scan_force.clicked.connect(lambda _=False, nid=n.node_id: self._act_scan_force_candidates(nid))
                sl.addWidget(lbl)
                sl.addWidget(btn_scan_force)
                self.tbl.setCellWidget(r, 2, statusw)

                it_last = QtWidgets.QTableWidgetItem(_fmt_ts(n.last_run_finished_epoch_ns))
                it_last.setFlags(it_last.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.tbl.setItem(r, 3, it_last)

                envw = QtWidgets.QWidget()
                el = QtWidgets.QHBoxLayout(envw)
                el.setContentsMargins(0, 0, 0, 0)
                el.setSpacing(6)

                lbl_env = QtWidgets.QLabel("Node .env")
                lbl_env.setStyleSheet("color:#0f172a;")
                lbl_env.setToolTip(str(self._node_env_path(n.node_root)))

                b_env = QtWidgets.QPushButton("Open Node .env")
                for b in (b_env,):
                    b.setMaximumWidth(130)

                b_env.clicked.connect(lambda _=False, nid=n.node_id: self._act_open_node_env(nid))

                el.addWidget(lbl_env)
                el.addWidget(b_env)
                self.tbl.setCellWidget(r, 4, envw)

                btns = QtWidgets.QWidget()
                outer = QtWidgets.QVBoxLayout(btns)
                outer.setContentsMargins(0, 0, 0, 0)
                outer.setSpacing(4)
                row1 = QtWidgets.QHBoxLayout()
                row2 = QtWidgets.QHBoxLayout()
                for rl in (row1, row2):
                    rl.setContentsMargins(0, 0, 0, 0)
                    rl.setSpacing(6)
                b_open = QtWidgets.QPushButton("Open")
                b_deploy = QtWidgets.QPushButton("Deploy/Repair")
                b_reset = QtWidgets.QPushButton("Reset")
                b_proc = QtWidgets.QPushButton("Process")
                b_update = QtWidgets.QPushButton("Update Proc")
                b_update_projects = QtWidgets.QPushButton("Update Projects")
                b_force_file = QtWidgets.QPushButton("Force Files")
                b_td_serials = QtWidgets.QPushButton("TD Serials")
                b_update.setToolTip(
                    "Refresh node runtime and node UI venv packages against the current runtime code "
                    "without deleting node caches/artifacts."
                )
                b_update_projects.setToolTip(
                    "Select one or more registered node projects and update them from the backend processor path."
                )
                b_force_file.setToolTip(
                    "Select one or more PDF, Excel, or MAT source files (or a whole folder) in this node and "
                    "rebuild only those files' EIDAT Support artifacts, metadata, source pointers when "
                    "applicable, and index entries."
                )
                b_td_serials.setToolTip(
                    "View a Test Data Trending project's programs/serial numbers and delete a serial "
                    "with ALL of its cached data, so the next Update Projects run re-creates it from scratch."
                )
                for b in (b_open, b_deploy, b_reset, b_proc, b_update, b_update_projects, b_force_file, b_td_serials):
                    b.setMinimumHeight(30)
                    b.setMinimumWidth(130)
                    b.setMaximumWidth(145)
                b_open.clicked.connect(lambda _=False, root=n.node_root: self._open_folder(root))
                b_deploy.clicked.connect(lambda _=False, root=n.node_root: self._deploy_node(root))
                b_reset.clicked.connect(lambda _=False, root=n.node_root: self._reset_node(root))
                b_proc.clicked.connect(lambda _=False, nid=n.node_id: self._process_one(nid))
                b_update.clicked.connect(lambda _=False, nid=n.node_id: self._update_processor(nid))
                b_update_projects.clicked.connect(lambda _=False, nid=n.node_id: self._act_update_projects(nid))
                b_force_file.clicked.connect(lambda _=False, nid=n.node_id: self._act_force_single_file(nid))
                b_td_serials.clicked.connect(lambda _=False, nid=n.node_id: self._act_manage_td_serials(nid))
                row1.addWidget(b_open)
                row1.addWidget(b_deploy)
                row1.addWidget(b_reset)
                row1.addWidget(b_proc)
                row2.addWidget(b_update)
                row2.addWidget(b_update_projects)
                row2.addWidget(b_force_file)
                row2.addWidget(b_td_serials)
                row1.addStretch(1)
                row2.addStretch(1)
                outer.addLayout(row1)
                outer.addLayout(row2)
                btns.setMinimumHeight(72)
                self.tbl.setCellWidget(r, 5, btns)

                # --- Extraction V2 column (independent of legacy runner) ---
                if _v2_panel is not None:
                    try:
                        v2_cell = _v2_panel.make_v2_cell(
                            self, n.node_id, n.node_root, str(self._runtime_root)
                        )
                        self.tbl.setCellWidget(r, 6, v2_cell)
                    except Exception as exc:
                        it_v2 = QtWidgets.QTableWidgetItem(f"v2 error: {exc}")
                        it_v2.setFlags(it_v2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                        self.tbl.setItem(r, 6, it_v2)
                else:
                    it_v2 = QtWidgets.QTableWidgetItem("v2 not installed")
                    it_v2.setToolTip(
                        "Extraction V2 needs: pip install rapidocr onnxruntime "
                        "opencv-python-headless pypdfium2 pdfplumber"
                    )
                    it_v2.setFlags(it_v2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    self.tbl.setItem(r, 6, it_v2)

                self.tbl.setRowHeight(r, 88)
        finally:
            self.tbl.blockSignals(False)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        try:
            row = item.row()
            col = item.column()
            node_id = str(self.tbl.item(row, 1).data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
            if not node_id:
                return
            if col == 0:
                enabled = item.checkState() == QtCore.Qt.CheckState.Checked
                admin_db.set_node_enabled(self._conn, node_id=node_id, enabled=enabled)
        except Exception as exc:
            self._append_log(f"[WARN] Update failed: {exc}")

    def _open_folder(self, root: str) -> None:
        try:
            p = _as_abs(root)
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open", str(exc))

    def _deploy_node(self, node_root: str) -> None:
        try:
            # Run deploy in-process to keep it simple.
            from .deploy import main as deploy_main

            deploy_main(["--node-root", node_root, "--runtime-root", str(self._runtime_root)])
            node_id = admin_db.upsert_node(
                self._conn,
                node_root=str(_as_abs(node_root)),
                runtime_root=str(self._runtime_root),
                enabled=True,
                notes=None,
            )
            # Processing always uses the node .env (seeded/refreshed during deploy/repair).
            admin_db.set_node_env_enabled(self._conn, node_id=node_id, enabled=True)
            self._append_log("[ENV] Processing uses: Node .env")
            self._append_log(f"[DEPLOY] {node_root}")
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Deploy", str(exc))

    def _reset_node(self, node_root: str) -> None:
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return
        root = str(_as_abs(node_root))
        msg = (
            "This will completely delete and recreate the node runtime folders:\n\n"
            f"  {Path(root) / 'EIDAT'}\n"
            f"  {Path(root) / 'EIDAT' / 'EIDAT Support'}\n"
            f"  {Path(root) / 'EIDAT Support'} (legacy, if present)\n\n"
            "Node-local venv packages are preserved (Lib/site-packages is NOT removed/replaced).\n\n"
            "This does NOT delete EIDPs.\n\n"
            "Continue?"
        )
        resp = QtWidgets.QMessageBox.question(
            self,
            "Reset Node",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            from .deploy import main as deploy_main

            deploy_main(
                [
                    "--node-root",
                    root,
                    "--runtime-root",
                    str(self._runtime_root),
                    "--reset-node",
                    "--reset-keep-site-packages",
                ]
            )
            node_id = admin_db.upsert_node(
                self._conn,
                node_root=str(_as_abs(root)),
                runtime_root=str(self._runtime_root),
                enabled=True,
                notes=None,
            )
            admin_db.set_node_env_enabled(self._conn, node_id=node_id, enabled=True)
            self._append_log(f"[RESET] {root}")
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Reset", str(exc))

    def _act_add_deploy(self) -> None:
        node_root = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Node Root (repo folder)")
        if not node_root:
            return
        self._deploy_node(node_root)

    def _act_remove(self) -> None:
        node_id = self._selected_node_id()
        if not node_id:
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Remove Node",
            "Remove this node from the admin list?\n\n(This does not delete any files in the node folder.)",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            admin_db.delete_node(self._conn, node_id=node_id)
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Remove", str(exc))

    def _process_one(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return
        # Ensure the node env file exists; processing always uses Node .env.
        self._ensure_node_env_file(n.node_root)
        self.worker.set_tasks([(n.node_id, n.node_root, n.runtime_root, True, "pipeline")])
        self._run_status.begin(f"Processing node: {Path(n.node_root).name}")
        self.worker.start()

    def _update_processor(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return
        # Ensure the node env file exists (even though update currently doesn't read it).
        self._ensure_node_env_file(n.node_root)
        self.worker.set_tasks([(n.node_id, n.node_root, n.runtime_root, True, "update_processor")])
        self._run_status.begin(f"Update Processor Envs/Runtime: {Path(n.node_root).name}")
        self.worker.start()

    def _choose_projects_for_update(
        self, node_root: str | Path
    ) -> tuple[list[dict[str, object]], bool, bool, bool, bool, bool] | None:
        from . import node_backend

        projects = list(node_backend.list_projects(node_root) or [])
        dlg = _ProjectUpdatePickerDialog(node_name=Path(str(node_root)).name, projects=projects, parent=self)
        try:
            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None
            return (
                dlg.selected_projects(),
                bool(dlg.cb_overwrite.isChecked()),
                bool(dlg.cb_force_rebuild.isChecked()),
                bool(dlg.cb_repair_serials.isChecked()),
                bool(dlg.cb_rebuild.isChecked()),
                bool(dlg.cb_auto_rebuild.isChecked()),
            )
        finally:
            dlg.deleteLater()

    def _act_update_projects(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return
        try:
            picked = self._choose_projects_for_update(n.node_root)
            if not picked:
                return
            projects, overwrite, force_project_rebuild, repair_serials, rebuild, auto_rebuild = picked
            if not projects:
                QtWidgets.QMessageBox.information(self, "Update Projects", "Select at least one project to update.")
                return
            if rebuild:
                td_count = sum(
                    1 for item in projects if str(item.get("type") or "").strip().casefold() == "test data trending"
                )
                confirm = QtWidgets.QMessageBox.question(
                    self,
                    "Rebuild Projects",
                    f"Rebuild {td_count} Test Data Trending project(s) from scratch?\n\n"
                    "Cache files are discarded and rebuilt from the TD source extractions; unreadable "
                    "workbooks are regenerated from project.json (damaged files are kept as *.corrupt-*.bak).",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            self._ensure_node_env_file(n.node_root)
            self.worker.set_tasks(
                [
                    (
                        n.node_id,
                        n.node_root,
                        n.runtime_root,
                        True,
                        "update_projects",
                        {
                            "projects": projects,
                            "overwrite": bool(overwrite),
                            "force_project_rebuild": bool(force_project_rebuild),
                            "repair_serials": bool(repair_serials),
                            "rebuild": bool(rebuild),
                            "auto_rebuild_on_failure": bool(auto_rebuild),
                        },
                    )
                ]
            )
            self._run_status.begin(
                f"{'Rebuild' if rebuild else 'Update'} Projects: {Path(n.node_root).name} ({len(projects)} selected)"
            )
            self.worker.start()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Update Projects", str(exc))

    def _act_manage_td_serials(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return
        try:
            from . import node_backend

            td_type = "Test Data Trending"
            projects = [
                p
                for p in (node_backend.list_projects(n.node_root) or [])
                if str(p.get("type") or "").strip() == td_type
            ]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Test Data Serials", str(exc))
            return
        if not projects:
            QtWidgets.QMessageBox.information(
                self, "Test Data Serials", "This node has no Test Data Trending projects."
            )
            return
        dlg = _TdSerialManagerDialog(
            node_name=Path(n.node_root).name, node_root=n.node_root, projects=projects, parent=self
        )
        try:
            dlg.exec()
            purged_project = dict(dlg.last_purged_project or {}) if dlg.purged_any else {}
        finally:
            dlg.deleteLater()
        if not purged_project:
            return
        self._append_log(
            f"[TD SERIALS] Purged serial cache in project '{purged_project.get('name')}' ({n.node_root})"
        )
        resp = QtWidgets.QMessageBox.question(
            self,
            "Re-process Project",
            (
                f"Run Update Projects now for '{purged_project.get('name')}' to re-create the "
                "deleted serial number(s) from scratch?"
            ),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if resp != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._ensure_node_env_file(n.node_root)
        self.worker.set_tasks(
            [
                (
                    n.node_id,
                    n.node_root,
                    n.runtime_root,
                    True,
                    "update_projects",
                    {
                        "projects": [purged_project],
                        "overwrite": False,
                        "force_project_rebuild": False,
                        "repair_serials": True,
                    },
                )
            ]
        )
        self._run_status.begin(f"Update Projects: {Path(n.node_root).name} (1 selected)")
        self.worker.start()

    def _act_scan_force_candidates(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return
        # Ensure the node env file exists; processing always uses Node .env.
        self._ensure_node_env_file(n.node_root)
        self.worker.set_tasks([(n.node_id, n.node_root, n.runtime_root, True, "scan_force_candidates")])
        self._run_status.begin(f"Scan+Force Candidates: {Path(n.node_root).name}")
        self.worker.start()

    def _act_force_single_file(self, node_id: str) -> None:
        nodes = {n.node_id: n for n in admin_db.list_nodes(self._conn)}
        n = nodes.get(node_id)
        if not n:
            return
        if self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Processing is already running.")
            return

        node_root = _as_abs(n.node_root)
        source_exts = {".pdf", ".xlsx", ".xls", ".xlsm", ".mat"}

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Force Files")
        box.setText("Force-reprocess which sources?")
        box.setInformativeText(
            "Select Files lets you pick one or more source files (Ctrl/Shift-click).\n"
            "Select Folder reprocesses every source file found under a folder."
        )
        btn_files = box.addButton("Select Files…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_folder = box.addButton("Select Folder…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()

        sources: list[Path] = []
        if clicked is btn_files:
            filter_str = "EIDAT source files (*.pdf *.xlsx *.xls *.xlsm *.mat);;All files (*)"
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select Source Files to Force Update",
                str(node_root),
                filter_str,
            )
            if not paths:
                return
            sources = [_as_abs(p) for p in paths]
            bad = [s for s in sources if s.suffix.lower() not in source_exts]
            if bad:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Force Files",
                    "Select only PDF, Excel, or MAT source files. Not a source file:\n"
                    + "\n".join(f"  {s.name}" for s in bad[:10]),
                )
                return
        elif clicked is btn_folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select Folder of Source Files to Force Update",
                str(node_root),
            )
            if not folder:
                return
            folder_path = _as_abs(folder)
            try:
                folder_path.resolve().relative_to(node_root.resolve())
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Force Files", "Select a folder inside the node root.")
                return
            skip_dirs = {"eidat support", "eidat", "edat support"}
            try:
                for p in sorted(folder_path.rglob("*")):
                    if not p.is_file() or p.suffix.lower() not in source_exts:
                        continue
                    rel_parts = {str(part).strip().casefold() for part in p.relative_to(folder_path).parts[:-1]}
                    if rel_parts & skip_dirs:
                        continue
                    if p.name.startswith("~$"):
                        continue
                    sources.append(p)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Force Files", f"Folder scan failed: {exc}")
                return
            if not sources:
                QtWidgets.QMessageBox.information(
                    self, "Force Files", "No PDF, Excel, or MAT source files found under that folder."
                )
                return
        else:
            return

        outside = []
        for s in sources:
            try:
                s.resolve().relative_to(node_root.resolve())
            except Exception:
                outside.append(s)
        if outside:
            QtWidgets.QMessageBox.warning(
                self,
                "Force Files",
                "All source files must be inside the node root. Outside:\n"
                + "\n".join(f"  {s}" for s in outside[:10]),
            )
            return

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique_sources: list[Path] = []
        for s in sources:
            key = str(s.resolve()) if s.exists() else str(s)
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        sources = unique_sources

        if len(sources) > 1:
            preview = "\n".join(f"  {s.name}" for s in sources[:20])
            if len(sources) > 20:
                preview += f"\n  … and {len(sources) - 20} more"
            resp = QtWidgets.QMessageBox.question(
                self,
                "Force Files",
                f"Force-reprocess {len(sources)} source file(s)?\n\n{preview}\n\nContinue?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self._ensure_node_env_file(n.node_root)
        self.worker.set_tasks(
            [
                (
                    n.node_id,
                    n.node_root,
                    n.runtime_root,
                    True,
                    "force_single_file",
                    {"file_paths": [str(s) for s in sources]},
                )
            ]
        )
        label = sources[0].name if len(sources) == 1 else f"{len(sources)} files"
        self._run_status.begin(f"Force File Update: {label}")
        self.worker.start()

    def _on_finished_one(
        self,
        node_id: str,
        action: str,
        started_ns: int,
        finished_ns: int,
        ok: bool,
        error: str,
        summary: object,
    ) -> None:
        try:
            # Record processing and project update runs in the run registry.
            if action in {"pipeline", "scan_force_candidates", "force_single_file", "update_projects"}:
                status = "ok" if ok else "failed"
                payload = dict(summary or {}) if isinstance(summary, dict) else {}
                payload["ok"] = bool(ok)
                payload["action"] = str(action)
                if error:
                    payload["error"] = str(error)
                admin_db.insert_run(
                    self._conn,
                    node_id=node_id,
                    started_epoch_ns=int(started_ns or 0),
                    finished_epoch_ns=int(finished_ns or 0),
                    status=status,
                    summary_json=json.dumps(payload, default=str),
                    error=error or None,
                )
        except Exception as exc:
            self._append_log(f"[WARN] Failed to record run: {exc}")
        finally:
            self.refresh()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EIDAT Admin Dashboard (manage nodes).")
    ap.add_argument("--runtime-root", default="", help="Central runtime root (contains EIDAT_App_Files).")
    ap.add_argument("--registry", default="", help="Registry DB path (optional).")
    args = ap.parse_args(argv)

    runtime_root = _as_abs(args.runtime_root) if str(args.runtime_root or "").strip() else _default_runtime_root()
    registry_path = Path(args.registry).expanduser() if str(args.registry or "").strip() else admin_db.default_registry_path()

    app = QtWidgets.QApplication(sys.argv)
    w = AdminWindow(runtime_root=runtime_root, registry_path=registry_path)
    w.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
