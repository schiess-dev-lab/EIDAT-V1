import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


try:
    from PySide6 import QtWidgets
    from ui_next.qt_main import TestDataTrendDialog
except Exception:  # pragma: no cover - optional dependency guard for local runs
    QtWidgets = None  # type: ignore[assignment]
    TestDataTrendDialog = None  # type: ignore[assignment]


@unittest.skipIf(QtWidgets is None or TestDataTrendDialog is None, "PySide6 is required")
class TestQtMainFacetedGlobalFilters(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _make_window(self) -> TestDataTrendDialog:
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        workbook_path = project_dir / "project.xlsx"
        workbook_path.write_text("", encoding="utf-8")
        with patch.object(TestDataTrendDialog, "_load_cache", lambda self, *, rebuild: None), patch.object(
            TestDataTrendDialog, "_load_auto_plots", lambda self: None
        ):
            window = TestDataTrendDialog(project_dir, workbook_path)
        window._test_tmpdir = tmpdir  # type: ignore[attr-defined]
        return window

    @staticmethod
    def _close_window(window: TestDataTrendDialog) -> None:
        window.close()
        tmpdir = getattr(window, "_test_tmpdir", "")
        if tmpdir:
            shutil.rmtree(str(tmpdir), ignore_errors=True)

    @staticmethod
    def _seed_filters(window: TestDataTrendDialog) -> None:
        window._available_program_filters = ["Program A", "Program B"]
        window._checked_program_filters = ["Program A", "Program B"]
        window._available_serial_filter_rows = [
            {"serial": "SN-001", "program_title": "Program A", "document_type": "TD"},
            {"serial": "SN-002", "program_title": "Program A", "document_type": "TD"},
            {"serial": "SN-101", "program_title": "Program B", "document_type": "TD"},
        ]
        window._checked_serial_filters = ["SN-001", "SN-002", "SN-101"]
        window._available_control_period_filters = ["10", "20", "30"]
        window._checked_control_period_filters = ["10", "20", "30"]
        window._available_suppression_voltage_filters = ["5", "7", "9"]
        window._checked_suppression_voltage_filters = ["5", "7", "9"]
        window._available_valve_voltage_filters = ["28", "30", "32"]
        window._checked_valve_voltage_filters = ["28", "30", "32"]
        window._serial_source_rows = list(window._available_serial_filter_rows)
        window._serial_source_by_serial = {
            row["serial"]: dict(row) for row in window._available_serial_filter_rows
        }
        window._global_filter_rows = [
            {
                "serial": "SN-001",
                "program_title": "Program A",
                "source_run_name": "Seq A1",
                "control_period": 10.0,
                "suppression_voltage": 5.0,
                "valve_voltage": 28.0,
            },
            {
                "serial": "SN-002",
                "program_title": "Program A",
                "source_run_name": "Seq A2",
                "control_period": 20.0,
                "suppression_voltage": 7.0,
                "valve_voltage": 30.0,
            },
            {
                "serial": "SN-101",
                "program_title": "Program B",
                "source_run_name": "Seq B1",
                "control_period": 30.0,
                "suppression_voltage": 9.0,
                "valve_voltage": 32.0,
            },
        ]

    def test_program_filter_facets_serials_and_condition_options(self) -> None:
        window = self._make_window()
        try:
            self._seed_filters(window)
            state = window._current_auto_plot_filter_state()
            state["programs"] = ["Program A"]

            self.assertEqual(window._faceted_available_filter_values("programs", filter_state=state), ["Program A", "Program B"])
            self.assertEqual(window._faceted_available_filter_values("serials", filter_state=state), ["SN-001", "SN-002"])
            self.assertEqual(window._faceted_available_filter_values("control_periods", filter_state=state), ["10", "20"])
            self.assertEqual(window._faceted_available_filter_values("suppression_voltages", filter_state=state), ["5", "7"])
            self.assertEqual(window._faceted_available_filter_values("valve_voltages", filter_state=state), ["28", "30"])
            self.assertIn("Serials: All (2)", window._auto_plot_filter_summary_text(state))
        finally:
            self._close_window(window)

    def test_serial_and_condition_filters_facet_other_categories(self) -> None:
        window = self._make_window()
        try:
            self._seed_filters(window)
            state = window._current_auto_plot_filter_state()
            state["serials"] = ["SN-001"]

            self.assertEqual(window._faceted_available_filter_values("programs", filter_state=state), ["Program A"])
            self.assertEqual(window._faceted_available_filter_values("control_periods", filter_state=state), ["10"])
            self.assertEqual(window._faceted_available_filter_values("suppression_voltages", filter_state=state), ["5"])
            self.assertEqual(window._faceted_available_filter_values("valve_voltages", filter_state=state), ["28"])

            state = window._current_auto_plot_filter_state()
            state["control_periods"] = ["10"]

            self.assertEqual(window._faceted_available_filter_values("programs", filter_state=state), ["Program A"])
            self.assertEqual(window._faceted_available_filter_values("serials", filter_state=state), ["SN-001"])
            self.assertEqual(window._faceted_available_filter_values("suppression_voltages", filter_state=state), ["5"])
            self.assertEqual(window._faceted_available_filter_values("valve_voltages", filter_state=state), ["28"])
        finally:
            self._close_window(window)

    def test_empty_filter_is_recoverable_and_reset_restores_static_universe(self) -> None:
        window = self._make_window()
        try:
            self._seed_filters(window)
            state = window._current_auto_plot_filter_state()
            state["programs"] = []

            self.assertEqual(
                [entry["value"] for entry in window._filter_checklist_entries("programs", filter_state=state)],
                ["Program A", "Program B"],
            )
            self.assertEqual(window._faceted_available_filter_values("serials", filter_state=state), [])
            self.assertEqual(window._faceted_available_filter_values("control_periods", filter_state=state), [])

            window._checked_program_filters = ["Program A"]
            window._checked_serial_filters = ["SN-001"]
            window._checked_control_period_filters = ["10"]
            window._checked_suppression_voltage_filters = ["5"]
            window._checked_valve_voltage_filters = ["28"]
            with patch.object(window, "_on_global_filters_changed", return_value=None):
                window._reset_global_filters()

            self.assertEqual(window._checked_program_filters, ["Program A", "Program B"])
            self.assertEqual(window._checked_serial_filters, ["SN-001", "SN-002", "SN-101"])
            self.assertEqual(window._checked_control_period_filters, ["10", "20", "30"])
            self.assertEqual(window._checked_suppression_voltage_filters, ["5", "7", "9"])
            self.assertEqual(window._checked_valve_voltage_filters, ["28", "30", "32"])
        finally:
            self._close_window(window)


if __name__ == "__main__":
    unittest.main()
