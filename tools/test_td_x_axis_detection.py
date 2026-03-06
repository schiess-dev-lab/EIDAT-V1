import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _have_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_have_openpyxl(), "openpyxl not installed")
class TestTDXAxisDetection(unittest.TestCase):
    def _write_config(self, path: Path, *, x_axis: dict) -> None:
        cfg = {
            "description": "test",
            "data_group": "Excel Data",
            "columns": [{"name": "thrust", "units": "lbf", "range_min": None, "range_max": None}],
            "statistics": ["mean", "min", "max", "std"],
            "statistics_ignore_first_n": 0,
            "performance_plotters": [],
            "sheet_name": None,
            "header_row": 0,
            "x_axis": x_axis,
        }
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _write_workbook(self, path: Path, *, source_sqlite: Path) -> None:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        ws_cfg = wb.active
        ws_cfg.title = "Config"
        ws_cfg.append(["name", "units"])
        ws_cfg.append(["thrust", "lbf"])

        ws_sources = wb.create_sheet("Sources")
        ws_sources.append(["serial_number", "excel_sqlite_rel"])
        ws_sources.append(["SN0001", str(source_sqlite)])

        wb.save(str(path))
        try:
            wb.close()
        except Exception:
            pass

    def _read_single_curve_x(self, db_path: Path, *, run: str, y: str, x_name: str, serial: str) -> list[float]:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                """
                SELECT x_json
                FROM td_curves
                WHERE run_name=? AND y_name=? AND x_name=? AND serial=?
                LIMIT 1
                """,
                (run, y, x_name, serial),
            ).fetchone()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        self.assertIsNotNone(row, "expected a td_curves row")
        return json.loads(row[0])

    def test_seq_time_header_maps_to_time(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            out_db = root / "td_cache.sqlite3"
            wb_path = root / "td.xlsx"
            cfg_path = root / "excel_trend_config.json"

            # Source sqlite: note `Seq_Time_sec` is the safe-ified header.
            conn = sqlite3.connect(str(src_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE "sheet__Run1" (
                        excel_row INTEGER NOT NULL,
                        "Seq_Time_sec" REAL,
                        thrust REAL
                    )
                    """
                )
                rows = [(i + 1, float(i), float(i) * 10.0) for i in range(20)]
                conn.executemany('INSERT INTO "sheet__Run1"(excel_row,"Seq_Time_sec",thrust) VALUES(?,?,?)', rows)
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            self._write_workbook(wb_path, source_sqlite=src_db)
            self._write_config(
                cfg_path,
                x_axis={
                    "replace_defaults": True,
                    "time_aliases": ["Seq Time (sec)"],
                    "pulse_aliases": ["Pulse Number", "cycle"],
                    "fuzzy_match": {"enabled": True, "min_ratio": 0.82},
                    "sequential_validation": {"enabled": True, "max_probe_rows": 250, "min_samples": 6, "pulse_min_run": 5},
                    "fallback_mode": "alias_only",
                    "default_x": "Time",
                },
            )

            old = be.DEFAULT_EXCEL_TREND_CONFIG
            be.DEFAULT_EXCEL_TREND_CONFIG = cfg_path
            try:
                be.rebuild_test_data_project_cache(out_db, wb_path)
            finally:
                be.DEFAULT_EXCEL_TREND_CONFIG = old

            conn = sqlite3.connect(str(out_db))
            try:
                xcols = conn.execute(
                    "SELECT name FROM td_columns WHERE run_name='Run1' AND kind='x' ORDER BY name"
                ).fetchall()
                default_x = conn.execute(
                    "SELECT default_x FROM td_runs WHERE run_name='Run1' LIMIT 1"
                ).fetchone()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            self.assertEqual([r[0] for r in xcols], ["Time"])
            self.assertEqual(default_x[0], "Time")

            xs = self._read_single_curve_x(out_db, run="Run1", y="thrust", x_name="Time", serial="SN0001")
            self.assertEqual(xs, [float(i) for i in range(20)])

    def test_cycle_header_maps_to_pulse_number(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            out_db = root / "td_cache.sqlite3"
            wb_path = root / "td.xlsx"
            cfg_path = root / "excel_trend_config.json"

            conn = sqlite3.connect(str(src_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE "sheet__Run1" (
                        excel_row INTEGER NOT NULL,
                        cycle REAL,
                        thrust REAL
                    )
                    """
                )
                rows = [(i + 1, float(i + 1), float(i) * 10.0) for i in range(20)]
                conn.executemany('INSERT INTO "sheet__Run1"(excel_row,cycle,thrust) VALUES(?,?,?)', rows)
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            self._write_workbook(wb_path, source_sqlite=src_db)
            self._write_config(
                cfg_path,
                x_axis={
                    "replace_defaults": True,
                    "time_aliases": ["Time (s)", "Seq Time (sec)"],
                    "pulse_aliases": ["cycle"],
                    "fuzzy_match": {"enabled": True, "min_ratio": 0.82},
                    "sequential_validation": {"enabled": True, "max_probe_rows": 250, "min_samples": 6, "pulse_min_run": 5},
                    "fallback_mode": "alias_only",
                    "default_x": "Time",
                },
            )

            old = be.DEFAULT_EXCEL_TREND_CONFIG
            be.DEFAULT_EXCEL_TREND_CONFIG = cfg_path
            try:
                be.rebuild_test_data_project_cache(out_db, wb_path)
            finally:
                be.DEFAULT_EXCEL_TREND_CONFIG = old

            conn = sqlite3.connect(str(out_db))
            try:
                xcols = conn.execute(
                    "SELECT name FROM td_columns WHERE run_name='Run1' AND kind='x' ORDER BY name"
                ).fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            self.assertEqual([r[0] for r in xcols], ["Pulse Number"])

            xs = self._read_single_curve_x(out_db, run="Run1", y="thrust", x_name="Pulse Number", serial="SN0001")
            self.assertEqual(xs, [float(i + 1) for i in range(20)])

    def test_non_sequential_time_like_metric_is_rejected(self) -> None:
        from EIDAT_App_Files.ui_next import backend as be  # type: ignore

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.sqlite3"
            out_db = root / "td_cache.sqlite3"
            wb_path = root / "td.xlsx"
            cfg_path = root / "excel_trend_config.json"

            conn = sqlite3.connect(str(src_db))
            try:
                conn.execute(
                    """
                    CREATE TABLE "sheet__Run1" (
                        excel_row INTEGER NOT NULL,
                        "Open_Stroke_Time" REAL,
                        "Seq_Time_sec" REAL,
                        thrust REAL
                    )
                    """
                )
                # Open_Stroke_Time is constant (time-like header but not sequential index).
                rows = [(i + 1, 1.23, float(i), float(i) * 10.0) for i in range(20)]
                conn.executemany(
                    'INSERT INTO "sheet__Run1"(excel_row,"Open_Stroke_Time","Seq_Time_sec",thrust) VALUES(?,?,?,?)',
                    rows,
                )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            self._write_workbook(wb_path, source_sqlite=src_db)
            self._write_config(
                cfg_path,
                x_axis={
                    "replace_defaults": True,
                    # Include a broad alias that could match "Open Stroke Time".
                    "time_aliases": ["time", "Seq Time (sec)"],
                    "pulse_aliases": ["cycle"],
                    "fuzzy_match": {"enabled": True, "min_ratio": 0.82},
                    "sequential_validation": {"enabled": True, "max_probe_rows": 250, "min_samples": 6, "pulse_min_run": 5},
                    "fallback_mode": "alias_only",
                    "default_x": "Time",
                },
            )

            old = be.DEFAULT_EXCEL_TREND_CONFIG
            be.DEFAULT_EXCEL_TREND_CONFIG = cfg_path
            try:
                be.rebuild_test_data_project_cache(out_db, wb_path)
            finally:
                be.DEFAULT_EXCEL_TREND_CONFIG = old

            xs = self._read_single_curve_x(out_db, run="Run1", y="thrust", x_name="Time", serial="SN0001")
            self.assertEqual(xs, [float(i) for i in range(20)])


if __name__ == "__main__":
    unittest.main()
