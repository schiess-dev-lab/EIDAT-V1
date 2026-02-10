#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _as_abs(p: str | Path) -> Path:
    path = Path(p).expanduser()
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


class NodeMainWindow:  # thin wrapper; delegates UI to ui_next.qt_main.MainWindow
    def __init__(self, *, node_root: Path, start_tab: str):
        from PySide6 import QtCore, QtWidgets  # type: ignore

        from ui_next.qt_main import MainWindow

        self._node_root = _as_abs(node_root)
        self._start_tab = (start_tab or "files").strip().lower()

        self._w: QtWidgets.QMainWindow = MainWindow()

        # "Drop" the Setup tab for node users by hiding its button (tab remains internal).
        try:
            btn = getattr(self._w, "btn_tab_setup", None)
            if btn is not None:
                btn.setVisible(False)
                btn.setEnabled(False)
        except Exception:
            pass

        # Bind the UI to this node root so Files/Projects behave like ui_next_debug.
        try:
            if hasattr(self._w, "ed_global_repo") and getattr(self._w, "ed_global_repo") is not None:
                self._w.ed_global_repo.setText(str(self._node_root))  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            if hasattr(self._w, "_set_global_repo"):
                self._w._set_global_repo(self._node_root)  # type: ignore[attr-defined]
        except Exception as exc:
            try:
                QtWidgets.QMessageBox.warning(self._w, "Node Root", str(exc))
            except Exception:
                pass

        def _select_tab() -> None:
            try:
                if self._start_tab == "projects" and hasattr(self._w, "_switch_tab"):
                    self._w._switch_tab(2)  # type: ignore[attr-defined]
                elif hasattr(self._w, "_switch_tab"):
                    self._w._switch_tab(1)  # type: ignore[attr-defined]
            except Exception:
                pass

        QtCore.QTimer.singleShot(0, _select_tab)

    def show(self) -> None:
        self._w.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EIDAT Node GUI (ui_next skin; Files + Projects).")
    ap.add_argument("--node-root", required=True, help="Node root directory.")
    ap.add_argument("--start-tab", choices=["files", "projects"], default="files", help="Initial tab.")
    args = ap.parse_args(argv)

    node_root = _as_abs(args.node_root)
    data_root = node_root / "EIDAT" / "UserData"

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ["EIDAT_NODE_ROOT"] = str(node_root)
    os.environ["EIDAT_DATA_ROOT"] = str(data_root)
    os.environ["EIDAT_UI_PROFILE"] = "node"

    from PySide6 import QtWidgets  # type: ignore

    app = QtWidgets.QApplication(sys.argv)
    w = NodeMainWindow(node_root=node_root, start_tab=args.start_tab)
    w.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

