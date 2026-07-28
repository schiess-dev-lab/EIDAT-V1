"""Mapping Review dialog (admin-side, PySide6).

Shows each pending shape with the source-image crop next to the extracted
preview, a dropdown of known canonical tables (editable for new names),
and Save / Skip. Saving creates a rule in the central library and replays
it over the whole node — back-filling every already-scanned document.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

from . import fingerprint as fp
from . import store


class MappingReviewDialog(QtWidgets.QDialog):
    def __init__(self, node_root: str, runtime_root: str, parent=None):
        super().__init__(parent)
        self._node_root = node_root
        self._runtime_root = runtime_root
        self.setWindowTitle(f"Extraction V2 — Mapping Review — {Path(node_root).name}")
        self.resize(1050, 640)

        self._node_db = store.open_node(node_root)
        self._rules_db = store.open_rules(runtime_root)

        root = QtWidgets.QHBoxLayout(self)

        # left: pending list
        left = QtWidgets.QVBoxLayout()
        self.lst = QtWidgets.QListWidget()
        self.lst.currentRowChanged.connect(self._show_item)
        left.addWidget(QtWidgets.QLabel("Unidentified shapes"))
        left.addWidget(self.lst, 1)
        root.addLayout(left, 1)

        # right: detail
        right = QtWidgets.QVBoxLayout()
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        right.addWidget(self.lbl_info)

        self.img = QtWidgets.QLabel("(no image)")
        self.img.setMinimumHeight(200)
        self.img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img.setStyleSheet("border:1px solid #cbd5e1; background:#f8fafc;")
        img_scroll = QtWidgets.QScrollArea()
        img_scroll.setWidgetResizable(True)
        img_scroll.setWidget(self.img)
        right.addWidget(img_scroll, 2)

        self.tbl_preview = QtWidgets.QTableWidget(0, 0)
        self.tbl_preview.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.tbl_preview, 2)

        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("This is:"))
        self.cmb = QtWidgets.QComboBox()
        self.cmb.setEditable(True)
        self.cmb.setMinimumWidth(320)
        form.addWidget(self.cmb, 1)
        self.btn_save = QtWidgets.QPushButton("Save mapping + replay")
        self.btn_skip = QtWidgets.QPushButton("Skip")
        self.btn_save.clicked.connect(self._save)
        self.btn_skip.clicked.connect(self._skip)
        form.addWidget(self.btn_save)
        form.addWidget(self.btn_skip)
        right.addLayout(form)

        self.lbl_status = QtWidgets.QLabel("")
        right.addWidget(self.lbl_status)
        root.addLayout(right, 3)

        self._items: list[tuple] = []
        self._reload()

    # ------------------------------------------------------------------
    def _reload(self) -> None:
        self._items = list(
            self._node_db.execute(
                "SELECT q.queue_id, q.obj_kind, q.obj_id, q.fingerprint_json,"
                " q.preview_json, q.crop_path, d.file"
                " FROM queue q JOIN docs d USING(doc_id)"
                " WHERE q.status='pending' ORDER BY q.queue_id"
            )
        )
        self.lst.clear()
        for _qid, kind, obj_id, _fpj, pvj, _crop, file in self._items:
            pv = json.loads(pvj)
            hint = pv.get("title") or ", ".join(l for l, _ in pv.get("pairs", [])[:3])
            self.lst.addItem(f"[{kind}] {Path(file).name} — {hint or obj_id}")
        labels = store.known_labels(self._rules_db)
        self.cmb.clear()
        self.cmb.addItems(labels)
        self.cmb.setCurrentText("")
        if self._items:
            self.lst.setCurrentRow(0)
        else:
            self.lbl_info.setText("Nothing pending — all shapes are mapped.")
            self.img.setText("(no image)")
            self.tbl_preview.setRowCount(0)
            self.tbl_preview.setColumnCount(0)

    def _show_item(self, row: int) -> None:
        if row < 0 or row >= len(self._items):
            return
        _qid, kind, obj_id, fpj, pvj, crop, file = self._items[row]
        pv = json.loads(pvj)
        self.lbl_info.setText(
            f"<b>{Path(file).name}</b> — {kind} <code>{obj_id}</code>"
            + (f" — title: “{pv.get('title')}”" if pv.get("title") else "")
        )
        if crop and Path(crop).exists():
            pix = QtGui.QPixmap(str(crop))
            if pix.width() > 900:
                pix = pix.scaledToWidth(
                    900, QtCore.Qt.TransformationMode.SmoothTransformation
                )
            self.img.setPixmap(pix)
        else:
            self.img.setPixmap(QtGui.QPixmap())
            self.img.setText("(no image crop)")

        rows = pv.get("rows") or [[l, v] for l, v in pv.get("pairs", [])]
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)
        self.tbl_preview.setRowCount(n_rows)
        self.tbl_preview.setColumnCount(n_cols)
        self.tbl_preview.horizontalHeader().setVisible(False)
        self.tbl_preview.verticalHeader().setVisible(False)
        for r, rvals in enumerate(rows):
            for c, val in enumerate(rvals):
                self.tbl_preview.setItem(r, c, QtWidgets.QTableWidgetItem(str(val)))
        self.tbl_preview.resizeColumnsToContents()

        # suggest closest known label
        try:
            f = json.loads(fpj)
            rule, score = fp.best_match(f, store.list_rules(self._rules_db), threshold=0.35)
            if rule:
                self.cmb.setCurrentText(rule["canonical_label"])
                self.lbl_status.setText(f"Suggestion: “{rule['canonical_label']}” (similarity {score:.2f})")
            else:
                self.cmb.setCurrentText("")
                self.lbl_status.setText("")
        except Exception:
            pass

    def _current(self):
        row = self.lst.currentRow()
        if row < 0 or row >= len(self._items):
            return None
        return self._items[row]

    def _save(self) -> None:
        item = self._current()
        if not item:
            return
        label = self.cmb.currentText().strip()
        if not label:
            QtWidgets.QMessageBox.information(
                self, "Mapping", "Enter or pick a canonical table name."
            )
            return
        _qid, kind, _obj_id, fpj, _pvj, _crop, _file = item
        store.add_rule(self._rules_db, kind, label, json.loads(fpj))
        facts, _ = store.replay_node(self._node_db, self._rules_db)
        self.lbl_status.setText(f"Saved “{label}” — replay wrote {facts} facts.")
        self._reload()

    def _skip(self) -> None:
        item = self._current()
        if not item:
            return
        self._node_db.execute(
            "UPDATE queue SET status='skipped' WHERE queue_id=?", (item[0],)
        )
        self._node_db.commit()
        self._reload()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._node_db.close()
            self._rules_db.close()
        except Exception:
            pass
        super().closeEvent(event)
