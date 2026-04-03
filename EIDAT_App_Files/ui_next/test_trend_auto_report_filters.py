from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from ui_next import trend_auto_report as tar


class TestTrendAutoReportFilters(unittest.TestCase):
    def test_filter_rows_respects_explicit_empty_filter_lists(self) -> None:
        rows = [
            {
                "serial": "SN-001",
                "program_title": "Program A",
                "control_period": 10.0,
                "suppression_voltage": 5.0,
            }
        ]
        self.assertEqual(tar._filter_rows_for_filter_state(rows, {"serials": []}), [])
        self.assertEqual(tar._filter_rows_for_filter_state(rows, {"programs": []}), [])

    def test_resolve_filtered_serials_honors_filter_state_and_run_scope(self) -> None:
        fake_be = SimpleNamespace(
            td_read_observation_filter_rows_from_cache=lambda _db_path: [
                {
                    "serial": "SN-001",
                    "program_title": "Program A",
                    "source_run_name": "Seq A",
                    "control_period": 10.0,
                    "suppression_voltage": 5.0,
                },
                {
                    "serial": "SN-002",
                    "program_title": "Program B",
                    "source_run_name": "Seq B",
                    "control_period": 20.0,
                    "suppression_voltage": 7.0,
                },
            ]
        )
        options = {
            "filter_state": {
                "programs": ["Program A"],
                "serials": ["SN-001"],
                "control_periods": ["10"],
                "suppression_voltages": ["5"],
            },
            "run_selections": [
                {
                    "member_sequences": ["Seq A"],
                    "member_programs": ["Program A"],
                }
            ],
        }

        resolved = tar._resolve_filtered_serials(
            fake_be,
            Path("cache.sqlite3"),
            ["SN-001", "SN-002"],
            options,
        )
        self.assertEqual(resolved, ["SN-001"])


if __name__ == "__main__":
    unittest.main()
