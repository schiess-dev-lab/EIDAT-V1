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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch EIDAT UI bound to a node root.")
    ap.add_argument("--node-root", required=True, help="Node root directory.")
    ap.add_argument("--mode", choices=["files", "projects"], default="files", help="Starting UI tab.")
    args = ap.parse_args(argv)

    node_root = _as_abs(args.node_root)
    data_root = node_root / "EIDAT" / "UserData"

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ["EIDAT_NODE_ROOT"] = str(node_root)
    os.environ["EIDAT_DATA_ROOT"] = str(data_root)
    os.environ["EIDAT_UI_START_TAB"] = str(args.mode)

    from PySide6 import QtCore, QtWidgets  # type: ignore

    from ui_next.qt_main import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()

    try:
        if hasattr(w, "_set_global_repo"):
            w._set_global_repo(node_root)  # type: ignore[attr-defined]
    except Exception:
        pass

    def _select_tab() -> None:
        try:
            if args.mode == "projects" and hasattr(w, "_switch_tab"):
                w._switch_tab(2)  # type: ignore[attr-defined]
            elif args.mode == "files" and hasattr(w, "_switch_tab"):
                w._switch_tab(1)  # type: ignore[attr-defined]
        except Exception:
            pass

    QtCore.QTimer.singleShot(0, _select_tab)
    w.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

