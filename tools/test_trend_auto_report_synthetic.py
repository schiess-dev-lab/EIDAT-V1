import json
import math
import tempfile
import unittest
from pathlib import Path

import sqlite3


def _have_numpy() -> bool:
    try:
        import numpy as _np  # noqa: F401
    except Exception:
        return False
    return True


class TestTrendAutoReportSynthetic(unittest.TestCase):
    def _make_td_db(self, root: Path) -> Path:
        db = root / "td_test.sqlite3"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                CREATE TABLE td_sources (
                    serial TEXT PRIMARY KEY,
                    sqlite_path TEXT,
                    mtime_ns INTEGER,
                    size_bytes INTEGER,
                    status TEXT,
                    last_ingested_epoch_ns INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_runs (
                    run_name TEXT PRIMARY KEY,
                    default_x TEXT,
                    display_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_columns (
                    run_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    units TEXT,
                    kind TEXT NOT NULL,
                    PRIMARY KEY (run_name, name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_curves (
                    run_name TEXT NOT NULL,
                    y_name TEXT NOT NULL,
                    x_name TEXT NOT NULL,
                    serial TEXT NOT NULL,
                    x_json TEXT NOT NULL,
                    y_json TEXT NOT NULL,
                    n_points INTEGER NOT NULL,
                    source_mtime_ns INTEGER,
                    computed_epoch_ns INTEGER NOT NULL,
                    PRIMARY KEY (run_name, y_name, x_name, serial)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_metrics (
                    serial TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    value_num REAL,
                    computed_epoch_ns INTEGER NOT NULL,
                    source_mtime_ns INTEGER,
                    PRIMARY KEY (serial, run_name, column_name, stat)
                )
                """
            )
            conn.execute("INSERT INTO td_sources(serial,status) VALUES ('SN1','ok'),('SN2','ok')")
            conn.execute("INSERT INTO td_runs(run_name,default_x,display_name) VALUES ('Run1','Time','Run 1')")
            conn.execute(
                "INSERT INTO td_columns(run_name,name,units,kind) VALUES ('Run1','thrust','lbf','y')"
            )

            x = [0.0, 1.0, 2.0, 3.0]
            y1 = [0.0, 1.0, 2.0, 3.0]
            y2 = [0.0, 2.0, 4.0, 6.0]
            conn.execute(
                """
                INSERT INTO td_curves(run_name,y_name,x_name,serial,x_json,y_json,n_points,computed_epoch_ns)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                ("Run1", "thrust", "Time", "SN1", json.dumps(x), json.dumps(y1), len(x), 1),
            )
            conn.execute(
                """
                INSERT INTO td_curves(run_name,y_name,x_name,serial,x_json,y_json,n_points,computed_epoch_ns)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                ("Run1", "thrust", "Time", "SN2", json.dumps(x), json.dumps(y2), len(x), 1),
            )

            # Metric min/max for autofill ranges.
            conn.execute(
                """
                INSERT INTO td_metrics(serial,run_name,column_name,stat,value_num,computed_epoch_ns)
                VALUES
                    ('SN1','Run1','thrust','min',0.0,1),
                    ('SN1','Run1','thrust','max',3.0,1),
                    ('SN2','Run1','thrust','min',0.0,1),
                    ('SN2','Run1','thrust','max',6.0,1)
                """
            )
        return db

    def _make_perf_db(self, root: Path) -> Path:
        db = root / "td_perf.sqlite3"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                CREATE TABLE td_sources (
                    serial TEXT PRIMARY KEY,
                    sqlite_path TEXT,
                    mtime_ns INTEGER,
                    size_bytes INTEGER,
                    status TEXT,
                    last_ingested_epoch_ns INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_runs (
                    run_name TEXT PRIMARY KEY,
                    default_x TEXT,
                    display_name TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_columns (
                    run_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    units TEXT,
                    kind TEXT NOT NULL,
                    PRIMARY KEY (run_name, name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE td_metrics (
                    serial TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    value_num REAL,
                    computed_epoch_ns INTEGER NOT NULL,
                    source_mtime_ns INTEGER,
                    PRIMARY KEY (serial, run_name, column_name, stat)
                )
                """
            )
            conn.execute("INSERT INTO td_sources(serial,status) VALUES ('SN1','ok'),('SN2','ok')")
            conn.execute(
                "INSERT INTO td_runs(run_name,default_x,display_name) VALUES ('Run1','Time','Run 1'),('Run2','Time','Run 2')"
            )
            for rn in ("Run1", "Run2"):
                conn.execute("INSERT INTO td_columns(run_name,name,units,kind) VALUES (?,?,?,?)", (rn, "isp", "s", "y"))
                conn.execute(
                    "INSERT INTO td_columns(run_name,name,units,kind) VALUES (?,?,?,?)", (rn, "thrust", "lbf", "y")
                )

            # Metrics: 2 runs x 2 serials x 2 cols x 3 stats.
            rows = []
            # Run1
            rows.extend(
                [
                    ("SN1", "Run1", "isp", "mean", 200.0),
                    ("SN1", "Run1", "isp", "min", 195.0),
                    ("SN1", "Run1", "isp", "max", 205.0),
                    ("SN1", "Run1", "thrust", "mean", 10.0),
                    ("SN1", "Run1", "thrust", "min", 9.5),
                    ("SN1", "Run1", "thrust", "max", 10.5),
                    ("SN2", "Run1", "isp", "mean", 210.0),
                    ("SN2", "Run1", "isp", "min", 205.0),
                    ("SN2", "Run1", "isp", "max", 215.0),
                    ("SN2", "Run1", "thrust", "mean", 11.0),
                    ("SN2", "Run1", "thrust", "min", 10.5),
                    ("SN2", "Run1", "thrust", "max", 11.5),
                ]
            )
            # Run2
            rows.extend(
                [
                    ("SN1", "Run2", "isp", "mean", 220.0),
                    ("SN1", "Run2", "isp", "min", 215.0),
                    ("SN1", "Run2", "isp", "max", 225.0),
                    ("SN1", "Run2", "thrust", "mean", 12.0),
                    ("SN1", "Run2", "thrust", "min", 11.5),
                    ("SN1", "Run2", "thrust", "max", 12.5),
                    ("SN2", "Run2", "isp", "mean", 230.0),
                    ("SN2", "Run2", "isp", "min", 225.0),
                    ("SN2", "Run2", "isp", "max", 235.0),
                    ("SN2", "Run2", "thrust", "mean", 13.0),
                    ("SN2", "Run2", "thrust", "min", 12.5),
                    ("SN2", "Run2", "thrust", "max", 13.5),
                ]
            )
            conn.executemany(
                """
                INSERT INTO td_metrics(serial,run_name,column_name,stat,value_num,computed_epoch_ns)
                VALUES (?,?,?,?,?,?)
                """,
                [(sn, rn, col, st, v, 1) for (sn, rn, col, st, v) in rows],
            )
        return db

    def test_autofill_excel_trend_config_from_cache(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_td_db(root)
            cfg = root / "excel_trend_config.json"
            cfg.write_text(
                json.dumps(
                    {
                        "description": "x",
                        "data_group": "Excel Data",
                        "columns": [{"name": "thrust", "units": "", "range_min": None, "range_max": None}],
                        "statistics": ["mean", "min", "max"],
                        "header_row": 0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            updated, summary = tar.autofill_excel_trend_config_from_td_cache(db, cfg, fill_units=True, fill_ranges=True)
            self.assertIn("Backup created:", summary)
            cols = updated.get("columns") or []
            thrust = [c for c in cols if isinstance(c, dict) and str(c.get("name")) == "thrust"][0]
            self.assertEqual(thrust.get("units"), "lbf")
            self.assertEqual(float(thrust.get("range_min")), 0.0)
            self.assertEqual(float(thrust.get("range_max")), 6.0)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_master_curve_and_poly_fit(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_td_db(root)
            with sqlite3.connect(str(db)) as conn:
                series = tar._load_curves(conn, "Run1", "thrust", "Time")
                self.assertEqual(len(series), 2)
                x_grid = [0.0, 1.0, 2.0, 3.0]
                ys = [tar._interp_linear(s.x, s.y, x_grid) for s in series]
                master = tar._nan_median(ys)
                # With y1=x and y2=2x, median with 2 series is the midpoint: 1.5x
                self.assertTrue(math.isclose(master[2], 3.0, rel_tol=1e-9))
                poly = tar._poly_fit(x_grid, master, 3, normalize_x=True)
                self.assertEqual(int(poly.get("degree")), 3)
                self.assertEqual(len(poly.get("coeffs") or []), 4)

    @unittest.skipUnless(_have_numpy(), "numpy not installed")
    def test_performance_pooled_points_and_poly_fit(self):
        from EIDAT_App_Files.ui_next import trend_auto_report as tar

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            db = self._make_perf_db(root)
            with sqlite3.connect(str(db)) as conn:
                serials = [r[0] for r in conn.execute("SELECT serial FROM td_sources ORDER BY serial").fetchall()]
                runs = [r[0] for r in conn.execute("SELECT run_name FROM td_runs ORDER BY run_name").fetchall()]

                def metric_map(run: str, col: str, stat: str) -> dict[str, float]:
                    rows = conn.execute(
                        "SELECT serial, value_num FROM td_metrics WHERE run_name=? AND column_name=? AND stat=?",
                        (run, col, stat),
                    ).fetchall()
                    return {str(sn): float(v) for sn, v in rows if sn is not None and v is not None}

                pooled_x: list[float] = []
                pooled_y: list[float] = []
                curves: dict[str, list[tuple[float, float]]] = {}
                for sn in serials:
                    pts = []
                    for rn in runs:
                        xmap = metric_map(rn, "isp", "mean")
                        ymap = metric_map(rn, "thrust", "mean")
                        if sn in xmap and sn in ymap:
                            pts.append((xmap[sn], ymap[sn]))
                    pts.sort(key=lambda t: t[0])
                    curves[sn] = pts
                    pooled_x.extend([p[0] for p in pts])
                    pooled_y.extend([p[1] for p in pts])

                self.assertEqual(len(curves.get("SN1") or []), 2)
                self.assertEqual(len(curves.get("SN2") or []), 2)
                self.assertEqual(len(pooled_x), 4)
                self.assertEqual(len(pooled_y), 4)

                poly = tar._poly_fit(pooled_x, pooled_y, 2, normalize_x=True)
                self.assertEqual(int(poly.get("degree")), 2)
                self.assertEqual(len(poly.get("coeffs") or []), 3)
                eqn = tar._fmt_equation(poly)
                self.assertTrue(bool(eqn))


if __name__ == "__main__":
    unittest.main()
