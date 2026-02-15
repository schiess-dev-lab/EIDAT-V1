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
    finished_one = QtCore.Signal(str, object, object, bool, str)  # node_id, started_ns, finished_ns, ok, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: list[tuple[str, str, str, bool, str]] = []  # (node_id, node_root, runtime_root, node_env_enabled, action)

    def set_tasks(self, tasks: list[tuple[str, str, str, bool, str]]) -> None:
        self._tasks = list(tasks)

    def run(self) -> None:  # type: ignore[override]
        from .admin_runner import run_scan_force_candidates

        for node_id, node_root, runtime_root, node_env_enabled, action in self._tasks:
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
            if res.ok:
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
                self.finished_one.emit(node_id, started, finished, True, "")
            else:
                _emit(f"[FAIL] {res.error}")
                self.finished_one.emit(node_id, started, finished, False, res.error or "Unknown error")


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


class AdminWindow(QtWidgets.QMainWindow):
    def __init__(self, *, runtime_root: Path, registry_path: Path):
        super().__init__()
        self._runtime_root = _as_abs(runtime_root)
        self._registry_path = _as_abs(registry_path)
        self._conn = admin_db.connect_registry(self._registry_path)

        self.setWindowTitle("EIDAT Admin Dashboard (Nodes)")
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
        self.lbl_proc_cfg = QtWidgets.QLabel("Process config: Node .env (EIDAT_PROCESS_FORCE/LIMIT/DPI)")
        self.lbl_proc_cfg.setStyleSheet("color:#475569;")
        self.lbl_proc_cfg.setToolTip("Open Node .env to set EIDAT_PROCESS_FORCE, EIDAT_PROCESS_LIMIT, EIDAT_PROCESS_DPI.")
        actions.addWidget(self.btn_add)
        actions.addWidget(self.btn_refresh)
        actions.addWidget(self.btn_remove)
        actions.addStretch(1)
        actions.addWidget(self.lbl_proc_cfg, 1)
        layout.addLayout(actions)

        self.tbl = QtWidgets.QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["Enabled", "Node Root", "Status", "Last Run", "Env", "Actions"])
        self.tbl.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
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
                bl = QtWidgets.QHBoxLayout(btns)
                bl.setContentsMargins(0, 0, 0, 0)
                b_open = QtWidgets.QPushButton("Open")
                b_deploy = QtWidgets.QPushButton("Deploy/Repair")
                b_proc = QtWidgets.QPushButton("Process")
                for b in (b_open, b_deploy, b_proc):
                    b.setMaximumWidth(110)
                b_open.clicked.connect(lambda _=False, root=n.node_root: self._open_folder(root))
                b_deploy.clicked.connect(lambda _=False, root=n.node_root: self._deploy_node(root))
                b_proc.clicked.connect(lambda _=False, nid=n.node_id: self._process_one(nid))
                bl.addWidget(b_open)
                bl.addWidget(b_deploy)
                bl.addWidget(b_proc)
                self.tbl.setCellWidget(r, 5, btns)
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

    def _on_finished_one(self, node_id: str, started_ns: int, finished_ns: int, ok: bool, error: str) -> None:
        try:
            status = "ok" if ok else "failed"
            summary = {"ok": ok, "error": error} if error else {"ok": ok}
            admin_db.insert_run(
                self._conn,
                node_id=node_id,
                started_epoch_ns=int(started_ns or 0),
                finished_epoch_ns=int(finished_ns or 0),
                status=status,
                summary_json=json.dumps(summary),
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
