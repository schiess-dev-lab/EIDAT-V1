import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtWidgets
    from ui_next.qt_main import MainWindow, MetadataBatchEditorDialog
except Exception:  # pragma: no cover
    QtWidgets = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]
    MetadataBatchEditorDialog = None  # type: ignore[assignment]


class _FakeMessageBox:
    class Icon:
        Warning = object()

    class ButtonRole:
        AcceptRole = 0
        DestructiveRole = 1
        RejectRole = 2

    choice_label = "Overwrite"

    def __init__(self, *_args, **_kwargs):
        self._buttons = {}
        self._clicked = None

    def setWindowTitle(self, _value):
        return None

    def setIcon(self, _value):
        return None

    def setText(self, _value):
        return None

    def setInformativeText(self, _value):
        return None

    def addButton(self, text, _role):
        btn = object()
        self._buttons[str(text)] = btn
        return btn

    def setDefaultButton(self, _button):
        return None

    def exec(self):
        self._clicked = self._buttons.get(self.choice_label)
        return 0

    def clickedButton(self):
        return self._clicked


@unittest.skipIf(QtWidgets is None or MainWindow is None or MetadataBatchEditorDialog is None, "PySide6 is required")
class TestQtMainFilesMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _make_window(self) -> MainWindow:
        with patch.object(MainWindow, "_scan_refresh", lambda self: None), patch.object(
            MainWindow, "_start_workspace_sync", lambda self, *, auto, show_popup, heading: None
        ):
            return MainWindow()

    def test_metadata_batch_editor_returns_only_touched_fields(self) -> None:
        dlg = MetadataBatchEditorDialog(
            [
                {"program_title": "Program Alpha", "vendor": "Vendor A"},
                {"program_title": "Program Alpha", "vendor": "Vendor B"},
            ],
            choices={"vendor": ["Vendor A", "Vendor B", "Vendor C"]},
        )
        try:
            self.assertEqual(dlg.field_updates(), {})
            dlg._widgets["vendor"].setCurrentText("Vendor C")
            self.assertEqual(dlg.field_updates(), {"vendor": "Vendor C"})
        finally:
            dlg.close()

    def test_update_metadata_prompt_can_overwrite_manual_fields(self) -> None:
        window = self._make_window()
        calls: list[dict] = []
        try:
            window.ed_global_repo.setText("C:/repo")

            def _start_now(self, *, heading, status_text, task, on_success, auto=False, show_popup=True):
                payload = task()
                calls.append({"heading": heading, "status_text": status_text, "payload": payload})
                on_success(payload)

            with patch.object(MainWindow, "_selected_files_info", return_value=[{
                "rel_path": "source/doc1.pdf",
                "metadata_source": "manual_override",
                "has_manual_override": True,
            }]), patch.object(MainWindow, "_refresh_files_tab", lambda self: None), patch.object(
                MainWindow, "_show_toast", lambda self, text: None
            ), patch.object(
                MainWindow, "_append_log", lambda self, text: None
            ), patch.object(
                MainWindow, "_start_manager_action", _start_now
            ), patch(
                "ui_next.qt_main.be.refresh_metadata_only",
                side_effect=lambda repo, rel_paths, overwrite_manual_fields=False: {
                    "updated": 1,
                    "failed": 0,
                    "results": [],
                    "overwrite_manual_fields": overwrite_manual_fields,
                },
            ) as refresh_mock, patch("ui_next.qt_main.QtWidgets.QMessageBox", _FakeMessageBox):
                _FakeMessageBox.choice_label = "Overwrite"
                window._act_files_update_metadata()

            self.assertEqual(refresh_mock.call_count, 1)
            self.assertTrue(bool(refresh_mock.call_args.kwargs.get("overwrite_manual_fields")))
            self.assertEqual(len(calls), 1)
        finally:
            window.close()

    def test_update_metadata_prompt_cancel_skips_refresh(self) -> None:
        window = self._make_window()
        try:
            window.ed_global_repo.setText("C:/repo")
            with patch.object(MainWindow, "_selected_files_info", return_value=[{
                "rel_path": "source/doc1.pdf",
                "metadata_source": "manual_override",
                "has_manual_override": True,
            }]), patch.object(MainWindow, "_start_manager_action") as start_mock, patch(
                "ui_next.qt_main.be.refresh_metadata_only"
            ) as refresh_mock, patch("ui_next.qt_main.QtWidgets.QMessageBox", _FakeMessageBox):
                _FakeMessageBox.choice_label = "Cancel"
                window._act_files_update_metadata()

            start_mock.assert_not_called()
            refresh_mock.assert_not_called()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
