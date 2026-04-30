import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402


class TestTDProjectParameterUnits(unittest.TestCase):
    def _project_dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_build_units_rows_uses_project_units_as_fallback_only(self) -> None:
        project_dir = self._project_dir()
        backend.save_td_project_parameter_normalization(
            project_dir,
            {
                "groups": [
                    {
                        "id": "display:pulsepressure",
                        "display_name": "Pulse Pressure",
                        "preferred_units": "global-u",
                    }
                ]
            },
        )

        rows = backend.td_build_project_parameter_units_rows(
            project_dir,
            [
                {
                    "program_title": "Program A",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "PulsePressure",
                    "default_display_parameter": "Pulse Pressure",
                    "displayed_parameter": "Pulse Pressure",
                    "preferred_units": "",
                }
            ],
            {
                "inventory": [
                    {
                        "program_title": "Program A",
                        "asset_type": "Valve",
                        "asset_specific_type": "Main",
                        "raw_name": "PulsePressure",
                        "displayed_parameter": "Pulse Pressure",
                        "default_display_parameter": "Pulse Pressure",
                        "units": ["inventory-u"],
                    }
                ]
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0].get("preferred_units") or ""), "global-u")
        self.assertFalse(bool(rows[0].get("unit_conflict")))

    def test_build_units_rows_do_not_let_stale_project_group_display_override_repo_row(self) -> None:
        project_dir = self._project_dir()
        backend.save_td_project_parameter_normalization(
            project_dir,
            {
                "groups": [
                    {
                        "id": backend._td_program_parameter_canonical_id("Chamber Temp"),
                        "display_name": "ChamberTemp",
                        "preferred_units": "degC",
                    }
                ]
            },
        )

        rows = backend.td_build_project_parameter_units_rows(
            project_dir,
            [
                {
                    "program_title": "Program A",
                    "asset_type": "Chamber",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "TC",
                    "default_display_parameter": "Chamber Temp",
                    "displayed_parameter": "Chamber Temp",
                    "preferred_units": "degC",
                }
            ],
            {
                "inventory": [
                    {
                        "program_title": "Program A",
                        "asset_type": "Chamber",
                        "asset_specific_type": "Main",
                        "raw_name": "TC",
                        "displayed_parameter": "Chamber Temp",
                        "default_display_parameter": "Chamber Temp",
                        "units": ["degC"],
                    }
                ]
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0].get("displayed_parameter") or ""), "Chamber Temp")

    def test_build_units_rows_leaves_blank_when_units_conflict(self) -> None:
        project_dir = self._project_dir()

        rows = backend.td_build_project_parameter_units_rows(
            project_dir,
            [
                {
                    "program_title": "Program A",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "PulsePressure",
                    "default_display_parameter": "Pulse Pressure",
                    "displayed_parameter": "Pulse Pressure",
                    "preferred_units": "",
                },
                {
                    "program_title": "Program B",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "PulsePressureAvg",
                    "default_display_parameter": "Pulse Pressure",
                    "displayed_parameter": "Pulse Pressure",
                    "preferred_units": "",
                },
            ],
            {
                "inventory": [
                    {
                        "program_title": "Program A",
                        "asset_type": "Valve",
                        "asset_specific_type": "Main",
                        "raw_name": "PulsePressure",
                        "displayed_parameter": "Pulse Pressure",
                        "default_display_parameter": "Pulse Pressure",
                        "units": ["psi"],
                    },
                    {
                        "program_title": "Program B",
                        "asset_type": "Valve",
                        "asset_specific_type": "Main",
                        "raw_name": "PulsePressureAvg",
                        "displayed_parameter": "Pulse Pressure",
                        "default_display_parameter": "Pulse Pressure",
                        "units": ["bar"],
                    },
                ]
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0].get("preferred_units") or ""), "")
        self.assertTrue(bool(rows[0].get("unit_conflict")))

    def test_save_units_mirrors_rows_and_project_catalog(self) -> None:
        project_dir = self._project_dir()
        repo_rows = [
            {
                "program_title": "Program A",
                "asset_type": "Valve",
                "asset_specific_type": "Main",
                "ingested_parameter": "PulsePressure",
                "default_display_parameter": "Pulse Pressure",
                "displayed_parameter": "Pulse Pressure",
                "preferred_units": "",
                "default_preferred_units": "",
            },
            {
                "program_title": "Program B",
                "asset_type": "Valve",
                "asset_specific_type": "Main",
                "ingested_parameter": "PulsePressureAvg",
                "default_display_parameter": "Pulse Pressure",
                "displayed_parameter": "Pulse Pressure",
                "preferred_units": "",
                "default_preferred_units": "",
            },
        ]
        unit_rows = [
            {
                "canonical_id": "display:pulsepressure",
                "displayed_parameter": "Pulse Pressure",
                "preferred_units": "psi",
            }
        ]

        with patch(
            "ui_next.backend.save_td_repo_parameter_mappings",
            side_effect=lambda _project_dir, rows: [dict(row) for row in rows],
        ) as save_repo_mock:
            payload = backend.td_save_project_parameter_units(project_dir, repo_rows, unit_rows)

        save_repo_mock.assert_called_once()
        saved_rows = [dict(row) for row in (payload.get("saved_rows") or [])]
        self.assertEqual({str(row.get("preferred_units") or "") for row in saved_rows}, {"psi"})
        self.assertEqual({str(row.get("default_preferred_units") or "") for row in saved_rows}, {"psi"})

        normalization = backend.load_td_project_parameter_normalization(project_dir)
        group = dict((normalization.get("groups_by_id") or {}).get("display:pulsepressure") or {})
        self.assertEqual(str(group.get("preferred_units") or ""), "psi")

    def test_rebuild_catalog_drops_orphaned_groups_after_rename(self) -> None:
        project_dir = self._project_dir()
        backend.save_td_project_parameter_normalization(
            project_dir,
            {
                "groups": [
                    {
                        "id": "display:oldname",
                        "display_name": "Old Name",
                        "preferred_units": "psi",
                    }
                ]
            },
        )

        backend.td_rebuild_project_parameter_units_catalog(
            project_dir,
            [
                {
                    "program_title": "Program A",
                    "asset_type": "Valve",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "PulsePressure",
                    "default_display_parameter": "New Name",
                    "displayed_parameter": "New Name",
                    "preferred_units": "bar",
                }
            ],
        )

        normalization = backend.load_td_project_parameter_normalization(project_dir)
        self.assertNotIn("display:oldname", dict(normalization.get("groups_by_id") or {}))
        self.assertEqual(
            str(((normalization.get("groups_by_id") or {}).get("display:newname") or {}).get("preferred_units") or ""),
            "bar",
        )

    def test_program_parameter_merge_rows_preserve_existing_exact_mapping(self) -> None:
        rows = backend._td_program_parameter_merge_rows(
            [
                {
                    "program_title": "Program A",
                    "asset_type": "Chamber",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "TC",
                    "default_display_parameter": "TC",
                    "displayed_parameter": "Chamber Temp",
                    "preferred_units": "degC",
                    "enabled": True,
                    "edited": True,
                }
            ],
            [
                {
                    "program_title": "Program A",
                    "asset_type": "Chamber",
                    "asset_specific_type": "Main",
                    "ingested_parameter": "TC",
                    "default_display_parameter": "Thermocouple",
                    "displayed_parameter": "Thermocouple",
                    "preferred_units": "celsius",
                    "enabled": True,
                    "edited": False,
                }
            ],
            program_title="Program A",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0].get("default_display_parameter") or ""), "Chamber Temp")
        self.assertEqual(str(rows[0].get("displayed_parameter") or ""), "Chamber Temp")
        self.assertEqual(str(rows[0].get("preferred_units") or ""), "degC")
        self.assertFalse(bool(rows[0].get("edited")))

    def test_selector_options_clear_conflict_when_group_units_exist(self) -> None:
        normalization = backend._td_normalize_parameter_normalization(
            {
                "groups": [
                    {
                        "id": "display:pulsepressure",
                        "display_name": "Pulse Pressure",
                        "preferred_units": "psi",
                    }
                ],
                "mappings": [
                    {
                        "canonical_id": "display:pulsepressure",
                        "raw_name": "PulsePressure",
                        "program_titles": ["Program A"],
                        "asset_types": ["Valve"],
                        "asset_specific_types": ["Main"],
                        "default_display_parameter": "Pulse Pressure",
                        "displayed_parameter": "Pulse Pressure",
                        "preferred_units": "psi",
                        "enabled": True,
                        "edited": False,
                    },
                    {
                        "canonical_id": "display:pulsepressure",
                        "raw_name": "PulsePressureAvg",
                        "program_titles": ["Program B"],
                        "asset_types": ["Valve"],
                        "asset_specific_types": ["Main"],
                        "default_display_parameter": "Pulse Pressure",
                        "displayed_parameter": "Pulse Pressure",
                        "preferred_units": "psi",
                        "enabled": True,
                        "edited": False,
                    },
                ],
            }
        )

        options = backend.td_build_parameter_selector_options(
            {
                "normalization": normalization,
                "entries": [
                    {
                        "surface": "metrics",
                        "run_name": "Run-1",
                        "raw_name": "PulsePressure",
                        "units": "bar",
                        "program_title": "Program A",
                        "asset_type": "Valve",
                        "asset_specific_type": "Main",
                        "source_run_name": "Run-1",
                        "source_key": "A",
                    },
                    {
                        "surface": "metrics",
                        "run_name": "Run-2",
                        "raw_name": "PulsePressureAvg",
                        "units": "psi",
                        "program_title": "Program B",
                        "asset_type": "Valve",
                        "asset_specific_type": "Main",
                        "source_run_name": "Run-2",
                        "source_key": "B",
                    },
                ],
            },
            surface="metrics",
        )

        self.assertEqual(len(options), 1)
        self.assertEqual(str(options[0].get("preferred_units") or ""), "psi")
        self.assertFalse(bool(options[0].get("unit_conflict")))


if __name__ == "__main__":
    unittest.main()
