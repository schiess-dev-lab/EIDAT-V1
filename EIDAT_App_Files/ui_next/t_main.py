"""
Compatibility shim.

Some older node launchers referenced `ui_next.t_main` (typo/legacy name).
The canonical entrypoint is `ui_next.qt_main`.
"""

from __future__ import annotations

from .qt_main import MainWindow, main

__all__ = ["MainWindow", "main"]

