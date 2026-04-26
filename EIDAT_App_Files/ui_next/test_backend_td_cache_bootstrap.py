import sys
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover - optional dependency guard for local runs
    Workbook = None  # type: ignore[assignment]


TEST_TD_STATS = ["mean", "max"]
TEST_TD_COLUMNS = [{"name": "Pressure", "units": "psi"}]
TEST_METRICS_LONG_HEADER = [
    "observation_id",
    "serial",
    "condition_key",
    "condition_display",
    "program_title",
    "source_run_name",
    "parameter_name",
    "stat",
    "value_num",
    "source_mtime_ns",
]
TEST_RAW_CACHE_LONG_HEADER = [
    "observation_id",
    "serial",
    "program_title",
    "source_run_name",
    "condition_key",
    "condition_display",
    "x_axis_kind",
    "run_type",
    "pulse_width",
    "control_period",
    "source_mtime_ns",
]


def _td_validation_patchers():
    return (
        patch.object(
            backend,
            "_load_project_td_trend_config",
            return_value={"statistics": list(TEST_TD_STATS), "columns": list(TEST_TD_COLUMNS)},
        ),
        patch.object(
            backend,
            "_load_excel_trend_config",
            return_value={"x_axis": {}},
        ),
    )


def _td_validation_context():
    stack = ExitStack()
    for patcher in _td_validation_patchers():
        stack.enter_context(patcher)
    return stack


def _td_update_context(project_dir: Path, workbook_path: Path):
    stack = ExitStack()
    impl_db = project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB
    raw_db = project_dir / backend.EIDAT_PROJECT_TD_RAW_CACHE_DB
    for patcher in _td_validation_patchers():
        stack.enter_context(patcher)
    stack.enter_context(
        patch.object(
            backend,
            "sync_test_data_project_cache",
            return_value={
                "db_path": str(impl_db),
                "raw_db_path": str(raw_db),
                "mode": "noop",
                "counts": {},
                "reason": "",
            },
        )
    )
    stack.enter_context(patch.object(backend, "_sync_td_support_workbook_program_sheets", return_value=None))
    stack.enter_context(patch.object(backend, "_refresh_td_support_run_conditions_sheet", return_value=None))
    stack.enter_context(patch.object(backend, "_sync_project_workbook_metadata_inplace", return_value=None))
    stack.enter_context(patch.object(backend, "read_eidat_index_documents", return_value=[]))
    stack.enter_context(
        patch.object(
            backend,
            "td_perf_refresh_saved_equation_store",
            return_value={"refreshed_count": 0, "failed_count": 0, "errors": [], "path": str(project_dir / "saved.json")},
        )
    )
    return stack


def _create_ready_td_project_fixture(project_dir: Path) -> tuple[Path, Path, Path]:
    if Workbook is None:
        raise RuntimeError("openpyxl is required for TD readiness tests")

    project_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = project_dir / "project.xlsx"
    impl_db = project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB
    raw_db = project_dir / backend.EIDAT_PROJECT_TD_RAW_CACHE_DB
    source_db = project_dir / "source.sqlite3"

    with closing(sqlite3.connect(str(source_db))) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sample (id INTEGER PRIMARY KEY)")
        conn.commit()

    wb = Workbook()
    ws_sources = wb.active
    ws_sources.title = "Sources"
    ws_sources.append(
        ["serial_number", "program_title", "document_type", "metadata_rel", "artifacts_rel", "excel_sqlite_rel"]
    )
    source_row = {
        "serial": "SN-001",
        "serial_number": "SN-001",
        "program_title": "Program Alpha",
        "document_type": "TD",
        "metadata_rel": "",
        "artifacts_rel": "",
        "excel_sqlite_rel": source_db.name,
    }
    ws_sources.append(
        [
            source_row["serial"],
            source_row["program_title"],
            source_row["document_type"],
            source_row["metadata_rel"],
            source_row["artifacts_rel"],
            source_row["excel_sqlite_rel"],
        ]
    )

    ws_data_calc = wb.create_sheet("Data_calc")
    ws_data_calc.append(["Metric", "SN-001"])
    ws_data_calc.append(["RunA", ""])
    ws_data_calc.append(["RunA.Pressure.mean", 1.23])
    ws_data_calc.append(["RunA.Pressure.max", 1.45])
    ws_data_calc.append(["", ""])

    ws_metrics_long = wb.create_sheet("Metrics_long")
    ws_metrics_long.append(TEST_METRICS_LONG_HEADER)
    ws_metrics_long.append(["obs1", "SN-001", "RunA", "RunA", "Program Alpha", "RunA", "Pressure", "mean", 1.23, 1])
    ws_metrics_long.append(["obs1", "SN-001", "RunA", "RunA", "Program Alpha", "RunA", "Pressure", "max", 1.45, 1])

    ws_metrics_long_sequences = wb.create_sheet("Metrics_long_sequences")
    ws_metrics_long_sequences.append(TEST_METRICS_LONG_HEADER)
    ws_metrics_long_sequences.append(["obs1", "SN-001", "RunA", "RunA", "Program Alpha", "RunA", "Pressure", "mean", 1.23, 1])
    ws_metrics_long_sequences.append(["obs1", "SN-001", "RunA", "RunA", "Program Alpha", "RunA", "Pressure", "max", 1.45, 1])

    ws_raw_cache_long = wb.create_sheet("RawCache_long")
    ws_raw_cache_long.append(TEST_RAW_CACHE_LONG_HEADER)
    ws_raw_cache_long.append(["obs1", "SN-001", "Program Alpha", "RunA", "RunA", "RunA", "time", "", None, None, 1])

    wb.save(str(workbook_path))
    wb.close()

    with patch.object(backend, "_load_excel_trend_config", return_value={"x_axis": {}}):
        project_raw_signature = backend._td_build_project_raw_signature(workbook_path, raw_columns_csv="Pressure")
    runtime_state = backend._td_source_runtime_state(
        workbook_path,
        source_row,
        project_raw_signature=project_raw_signature,
    )

    with closing(sqlite3.connect(str(impl_db))) as conn:
        backend._ensure_test_data_impl_tables(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO td_meta(key, value) VALUES (?, ?)",
            [
                ("workbook_path", str(workbook_path)),
                ("node_root", str(backend._infer_node_root_from_workbook_path(workbook_path))),
                ("statistics", ",".join(TEST_TD_STATS)),
                ("raw_columns", "Pressure"),
                ("support_workbook_mtime_ns", "0"),
                ("project_raw_signature", project_raw_signature),
                ("cache_schema_version", backend.TD_PROJECT_CACHE_SCHEMA_VERSION),
            ],
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_sources(serial, sqlite_path, mtime_ns, size_bytes, status, last_ingested_epoch_ns, raw_fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SN-001",
                str(source_db),
                int(runtime_state.get("mtime_ns") or 0),
                int(runtime_state.get("size_bytes") or 0),
                str(runtime_state.get("status") or "ok"),
                1,
                str(runtime_state.get("fingerprint") or ""),
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_source_diagnostics(
                serial, resolved_sqlite_path, status, run_name, x_axis_kind, matched_y_json, curves_written, metrics_written, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("SN-001", str(source_db), "ok", "", "", "[]", 1, 2, ""),
        )
        conn.execute(
            "INSERT OR REPLACE INTO td_runs(run_name, default_x, display_name, run_type, control_period, pulse_width) VALUES (?, ?, ?, ?, ?, ?)",
            ("RunA", "time", "RunA", "", None, None),
        )
        conn.execute(
            "INSERT OR REPLACE INTO td_columns_calc(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
            ("RunA", "Pressure", "psi", "y"),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_metrics_calc(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("obs1", "SN-001", "RunA", "Pressure", "mean", 1.23, 1, 1, "Program Alpha", "RunA"),
                ("obs1", "SN-001", "RunA", "Pressure", "max", 1.45, 1, 1, "Program Alpha", "RunA"),
            ],
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_condition_observations_sequences(
                observation_id, serial, run_name, program_title, source_run_name, run_type, pulse_width, control_period, suppression_voltage, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("obs1", "SN-001", "RunA", "Program Alpha", "RunA", "", None, None, None, 1, 1),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO td_metrics_calc_sequences(
                observation_id, serial, run_name, column_name, stat, value_num, computed_epoch_ns, source_mtime_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("obs1", "SN-001", "RunA", "Pressure", "mean", 1.23, 1, 1, "Program Alpha", "RunA"),
                ("obs1", "SN-001", "RunA", "Pressure", "max", 1.45, 1, 1, "Program Alpha", "RunA"),
            ],
        )
        conn.commit()

    with closing(sqlite3.connect(str(raw_db))) as conn:
        backend._ensure_test_data_raw_cache_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO td_raw_sequences(
                run_name, display_name, x_axis_kind, source_run_name, pulse_width, run_type, control_period, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RunA", "RunA", "time", "RunA", None, "", None, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_raw_condition_observations(
                observation_id, run_name, serial, program_title, source_run_name, run_type, pulse_width, control_period, source_mtime_ns, computed_epoch_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("obs1", "RunA", "SN-001", "Program Alpha", "RunA", "", None, None, 1, 1),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_curves_raw(
                run_name, y_name, x_name, observation_id, serial, x_json, y_json, n_points, source_mtime_ns, computed_epoch_ns, program_title, source_run_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RunA", "Pressure", "time", "obs1", "SN-001", "[0.0, 1.0]", "[1.0, 1.2]", 2, 1, 1, "Program Alpha", "RunA"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO td_columns_raw(run_name, name, units, kind) VALUES (?, ?, ?, ?)",
            ("RunA", "Pressure", "psi", "y"),
        )
        conn.commit()

    return workbook_path, impl_db, raw_db


def _append_excluded_td_source_to_fixture(
    project_dir: Path,
    workbook_path: Path,
    *,
    serial: str = "SN-002",
    program_title: str = "Program Beta",
    document_type: str = "TD",
    metadata_rel: str = "docs/sn002.json",
    artifacts_rel: str = "debug/ocr/sn002",
    excel_sqlite_rel: str = "missing_source.sqlite3",
) -> dict[str, str]:
    if Workbook is None:
        raise RuntimeError("openpyxl is required for TD readiness tests")
    from openpyxl import load_workbook  # type: ignore

    wb = load_workbook(str(workbook_path))
    try:
        ws_sources = wb["Sources"]
        ws_sources.append([serial, program_title, document_type, metadata_rel, artifacts_rel, excel_sqlite_rel])
        wb.save(str(workbook_path))
    finally:
        wb.close()

    source_row = {
        "serial": serial,
        "serial_number": serial,
        "program_title": program_title,
        "document_type": document_type,
        "metadata_rel": metadata_rel,
        "artifacts_rel": artifacts_rel,
        "excel_sqlite_rel": excel_sqlite_rel,
    }
    with patch.object(backend, "_load_excel_trend_config", return_value={"x_axis": {}}):
        project_raw_signature = backend._td_build_project_raw_signature(workbook_path, raw_columns_csv="Pressure")
    runtime_state = backend._td_source_runtime_state(
        workbook_path,
        source_row,
        project_raw_signature=project_raw_signature,
    )
    reason = str((runtime_state.get("source_resolution") or {}).get("reason") or "Source excluded from compilation.")
    with closing(sqlite3.connect(str(project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB))) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO td_sources(serial, sqlite_path, mtime_ns, size_bytes, status, last_ingested_epoch_ns, raw_fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                serial,
                str(runtime_state.get("sqlite_path") or ""),
                int(runtime_state.get("mtime_ns") or 0),
                int(runtime_state.get("size_bytes") or 0),
                str(runtime_state.get("status") or "missing"),
                1,
                str(runtime_state.get("fingerprint") or ""),
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_source_metadata(
                serial, program_title, document_type, metadata_rel, artifacts_rel, excel_sqlite_rel, metadata_mtime_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (serial, program_title, document_type, metadata_rel, artifacts_rel, excel_sqlite_rel, 0),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_source_diagnostics(
                serial, resolved_sqlite_path, status, run_name, x_axis_kind, matched_y_json, curves_written, metrics_written, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                serial,
                str(runtime_state.get("sqlite_path") or ""),
                str(runtime_state.get("status") or "missing"),
                "",
                "",
                "[]",
                0,
                0,
                reason,
            ),
        )
        conn.commit()
    return source_row


def _set_fixture_source_missing(project_dir: Path, workbook_path: Path, *, serial: str = "SN-001") -> None:
    if Workbook is None:
        raise RuntimeError("openpyxl is required for TD readiness tests")
    from openpyxl import load_workbook  # type: ignore

    wb = load_workbook(str(workbook_path))
    try:
        ws_sources = wb["Sources"]
        headers = {
            str(ws_sources.cell(1, col).value or "").strip().lower(): col
            for col in range(1, (ws_sources.max_column or 0) + 1)
            if str(ws_sources.cell(1, col).value or "").strip()
        }
        for row in range(2, (ws_sources.max_row or 0) + 1):
            if str(ws_sources.cell(row, headers["serial_number"]).value or "").strip() != serial:
                continue
            ws_sources.cell(row, headers["excel_sqlite_rel"]).value = "missing_source.sqlite3"
            break
        wb.save(str(workbook_path))
    finally:
        wb.close()

    source_row = {
        "serial": serial,
        "serial_number": serial,
        "program_title": "Program Alpha",
        "document_type": "TD",
        "metadata_rel": "",
        "artifacts_rel": "",
        "excel_sqlite_rel": "missing_source.sqlite3",
    }
    with patch.object(backend, "_load_excel_trend_config", return_value={"x_axis": {}}):
        project_raw_signature = backend._td_build_project_raw_signature(workbook_path, raw_columns_csv="Pressure")
    runtime_state = backend._td_source_runtime_state(
        workbook_path,
        source_row,
        project_raw_signature=project_raw_signature,
    )
    reason = str((runtime_state.get("source_resolution") or {}).get("reason") or "Source SQLite not found.")
    with closing(sqlite3.connect(str(project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB))) as conn:
        conn.execute(
            """
            UPDATE td_sources
            SET sqlite_path=?, mtime_ns=?, size_bytes=?, status=?, last_ingested_epoch_ns=?, raw_fingerprint=?
            WHERE serial=?
            """,
            (
                str(runtime_state.get("sqlite_path") or ""),
                int(runtime_state.get("mtime_ns") or 0),
                int(runtime_state.get("size_bytes") or 0),
                str(runtime_state.get("status") or "missing"),
                1,
                str(runtime_state.get("fingerprint") or ""),
                serial,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO td_source_diagnostics(
                serial, resolved_sqlite_path, status, run_name, x_axis_kind, matched_y_json, curves_written, metrics_written, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                serial,
                str(runtime_state.get("sqlite_path") or ""),
                str(runtime_state.get("status") or "missing"),
                "",
                "",
                "[]",
                0,
                0,
                reason,
            ),
        )
        conn.execute(
            "UPDATE td_source_metadata SET excel_sqlite_rel=? WHERE serial=?",
            ("missing_source.sqlite3", serial),
        )
        conn.commit()


class TestBackendTdCacheBootstrap(unittest.TestCase):
    def test_source_runtime_state_fingerprint_uses_healed_official_links(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workbook_path = root / "project.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Sources"
            ws.append(
                ["serial_number", "program_title", "document_type", "metadata_rel", "artifacts_rel", "excel_sqlite_rel"]
            )
            ws.append(["SN-001", "Program Alpha", "TD", "", "", "stale\\SN-001.sqlite3"])
            wb.save(str(workbook_path))
            wb.close()

            source_row = {
                "serial": "SN-001",
                "serial_number": "SN-001",
                "program_title": "Program Alpha",
                "document_type": "TD",
                "asset_type": "Thruster",
                "asset_specific_type": "Valve",
                "metadata_rel": "",
                "artifacts_rel": "",
                "excel_sqlite_rel": "stale\\SN-001.sqlite3",
            }
            official_sqlite = backend._td_official_sqlite_path_for_source_like(root, source_row)
            self.assertIsNotNone(official_sqlite)
            assert official_sqlite is not None
            official_sqlite.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(str(official_sqlite))) as conn:
                conn.commit()

            runtime_state = backend._td_source_runtime_state(
                workbook_path,
                source_row,
                project_raw_signature="synthetic-signature",
            )
            source_resolution = dict(runtime_state.get("source_resolution") or {})

            self.assertEqual(str(runtime_state.get("artifacts_rel") or ""), str(source_resolution.get("healed_artifacts_rel") or ""))
            self.assertEqual(str(runtime_state.get("excel_sqlite_rel") or ""), str(source_resolution.get("healed_excel_sqlite_rel") or ""))

            legacy_fingerprint = backend._td_hash_payload(
                {
                    "serial": "SN-001",
                    "serial_number": "SN-001",
                    "status": str(runtime_state.get("status") or ""),
                    "sqlite_path": str(official_sqlite),
                    "mtime_ns": int(runtime_state.get("mtime_ns") or 0),
                    "size_bytes": int(runtime_state.get("size_bytes") or 0),
                    "metadata_rel": "",
                    "artifacts_rel": "",
                    "excel_sqlite_rel": "stale\\SN-001.sqlite3",
                    "metadata_mtime_ns": 0,
                    "project_raw_signature": "synthetic-signature",
                }
            )
            self.assertNotEqual(str(runtime_state.get("fingerprint") or ""), legacy_fingerprint)

    def test_export_debug_excels_ensures_cache_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("stub workbook", encoding="utf-8")

            impl_db = project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB
            raw_db = project_dir / backend.EIDAT_PROJECT_TD_RAW_CACHE_DB
            raw_points = project_dir / backend.EIDAT_PROJECT_TD_RAW_POINTS_XLSX
            progress: list[str] = []

            def fake_sync(project_dir_arg: Path, workbook_path_arg: Path, *, rebuild: bool = False, progress_cb=None) -> dict[str, object]:
                self.assertEqual(Path(project_dir_arg), project_dir)
                self.assertEqual(Path(workbook_path_arg), workbook_path)
                self.assertFalse(rebuild)
                impl_db.write_text("impl", encoding="utf-8")
                raw_db.write_text("raw", encoding="utf-8")
                raw_points.write_text("points", encoding="utf-8")
                if progress_cb is not None:
                    progress_cb("cache refreshed")
                return {
                    "db_path": str(impl_db),
                    "raw_db_path": str(raw_db),
                }

            def fake_validate(project_dir_arg: Path, workbook_path_arg: Path) -> Path:
                self.assertEqual(Path(project_dir_arg), project_dir)
                self.assertEqual(Path(workbook_path_arg), workbook_path)
                self.assertTrue(impl_db.exists())
                self.assertTrue(raw_db.exists())
                return impl_db

            def fake_sync_mirror(db_path: Path, *, force: bool = False) -> Path:
                self.assertTrue(force)
                out_path = Path(db_path).with_suffix(".xlsx")
                out_path.write_text(f"mirror:{Path(db_path).name}", encoding="utf-8")
                return out_path

            with patch.object(backend, "sync_test_data_project_cache", side_effect=fake_sync) as sync_mock, patch.object(
                backend,
                "validate_existing_test_data_project_cache",
                side_effect=fake_validate,
            ) as validate_mock, patch.object(
                backend,
                "_sync_sqlite_excel_mirror",
                side_effect=fake_sync_mirror,
            ) as mirror_mock:
                generated = backend.export_test_data_project_debug_excels(
                    project_dir,
                    workbook_path,
                    force=True,
                    progress_cb=progress.append,
                )

            sync_mock.assert_called_once()
            validate_mock.assert_called_once()
            self.assertEqual(mirror_mock.call_count, 2)
            self.assertEqual(generated["implementation_excel"], impl_db.with_suffix(".xlsx"))
            self.assertEqual(generated["raw_cache_excel"], raw_db.with_suffix(".xlsx"))
            self.assertEqual(generated["raw_points_excel"], raw_points)
            self.assertIn("Ensuring project cache", progress)
            self.assertIn("Exporting implementation cache Excel mirror", progress)
            self.assertIn("Exporting raw cache Excel mirror", progress)

    def test_export_debug_excels_falls_back_to_existing_sqlite_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir(parents=True, exist_ok=True)
            workbook_path = project_dir / "project.xlsx"
            workbook_path.write_text("stub workbook", encoding="utf-8")

            impl_db = project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB
            raw_db = project_dir / backend.EIDAT_PROJECT_TD_RAW_CACHE_DB
            impl_db.write_text("impl", encoding="utf-8")
            raw_db.write_text("raw", encoding="utf-8")

            progress: list[str] = []

            def fake_sync_mirror(db_path: Path, *, force: bool = False) -> Path:
                self.assertTrue(force)
                out_path = Path(db_path).with_suffix(".xlsx")
                out_path.write_text(f"mirror:{Path(db_path).name}", encoding="utf-8")
                return out_path

            with patch.object(
                backend,
                "sync_test_data_project_cache",
                side_effect=RuntimeError("cache rebuild failed"),
            ) as sync_mock, patch.object(
                backend,
                "validate_existing_test_data_project_cache",
                side_effect=RuntimeError("cache incomplete"),
            ) as validate_mock, patch.object(
                backend,
                "_sync_sqlite_excel_mirror",
                side_effect=fake_sync_mirror,
            ) as mirror_mock:
                generated = backend.export_test_data_project_debug_excels(
                    project_dir,
                    workbook_path,
                    force=True,
                    progress_cb=progress.append,
                )

            sync_mock.assert_called_once()
            validate_mock.assert_called_once()
            self.assertEqual(mirror_mock.call_count, 2)
            self.assertEqual(generated["implementation_excel"], impl_db.with_suffix(".xlsx"))
            self.assertEqual(generated["raw_cache_excel"], raw_db.with_suffix(".xlsx"))
            self.assertTrue(
                any("Cache refresh failed; exporting existing SQLite files anyway" in msg for msg in progress)
            )
            self.assertTrue(
                any("Cache validation failed; exporting existing SQLite files anyway" in msg for msg in progress)
            )

    def test_create_test_data_project_primes_cache_sqlite_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir(parents=True, exist_ok=True)

            selected_rel = "docs/doc1.json"
            source_doc = {
                "metadata_rel": selected_rel,
                "serial_number": "SN-001",
                "document_type": "TD",
                "document_type_acronym": "TD",
                "document_type_status": "confirmed",
                "document_type_review_required": False,
                "excel_sqlite_rel": "excel_sqlite/doc1.sqlite3",
                "artifacts_rel": "debug/ocr/doc1",
                "program_title": "Program Alpha",
            }
            sync_calls: list[tuple[Path, Path]] = []

            def fake_write_workbook(
                workbook_path: Path,
                *,
                global_repo: Path,
                serials: list[str],
                docs: list[dict],
                config: dict,
            ) -> None:
                self.assertEqual(Path(global_repo), repo)
                self.assertEqual(serials, ["SN-001"])
                self.assertEqual(len(docs), 1)
                self.assertIn("columns", config)
                workbook_path.write_text("td workbook", encoding="utf-8")

            def fake_write_support_workbook(
                support_workbook_path: Path,
                *,
                sequence_names: list[str],
                param_defs: list[dict],
                program_titles: list[str],
                sequences_by_program: dict[str, list[str]],
            ) -> None:
                self.assertEqual(sequence_names, ["Run1"])
                self.assertEqual(program_titles, ["Program Alpha"])
                self.assertEqual(sequences_by_program, {"Program Alpha": ["Run1"]})
                self.assertEqual(param_defs, [{"name": "RPM"}])
                support_workbook_path.write_text("support workbook", encoding="utf-8")

            def fake_sync(project_dir_arg: Path, workbook_path_arg: Path, *, rebuild: bool = False, progress_cb=None) -> dict[str, object]:
                self.assertFalse(rebuild)
                project_dir_path = Path(project_dir_arg)
                workbook_path_path = Path(workbook_path_arg)
                sync_calls.append((project_dir_path, workbook_path_path))
                impl_db = project_dir_path / backend.EIDAT_PROJECT_IMPLEMENTATION_DB
                raw_db = project_dir_path / backend.EIDAT_PROJECT_TD_RAW_CACHE_DB
                impl_db.write_text("impl", encoding="utf-8")
                raw_db.write_text("raw", encoding="utf-8")
                return {
                    "db_path": str(impl_db),
                    "raw_db_path": str(raw_db),
                }

            with patch.object(backend, "read_eidat_index_documents", return_value=[source_doc]), patch.object(
                backend,
                "_resolve_td_source_sqlite_for_node",
                return_value={"status": "ok"},
            ), patch.object(
                backend,
                "_load_excel_trend_config",
                return_value={"columns": [{"name": "RPM"}]},
            ), patch.object(
                backend,
                "_write_test_data_trending_workbook",
                side_effect=fake_write_workbook,
            ), patch.object(
                backend,
                "_write_td_support_workbook",
                side_effect=fake_write_support_workbook,
            ), patch.object(
                backend,
                "_discover_td_runs_by_program_for_docs",
                return_value={"Program Alpha": ["Run1"]},
            ), patch.object(
                backend,
                "_discover_td_runs_for_docs",
                return_value=["Run1"],
            ), patch.object(
                backend,
                "sync_test_data_project_cache",
                side_effect=fake_sync,
            ):
                meta = backend.create_eidat_project(
                    repo,
                    project_parent_dir=Path("projects"),
                    project_name="TD Project",
                    project_type=backend.EIDAT_PROJECT_TYPE_TEST_DATA_TRENDING,
                    selected_metadata_rel=[selected_rel],
                )

            project_dir = Path(str(meta["project_dir"]))
            workbook_path = Path(str(meta["workbook"]))
            impl_db = project_dir / backend.EIDAT_PROJECT_IMPLEMENTATION_DB
            raw_db = project_dir / backend.EIDAT_PROJECT_TD_RAW_CACHE_DB

            self.assertEqual(sync_calls, [(project_dir, workbook_path)])
            self.assertTrue(workbook_path.exists())
            self.assertTrue((project_dir / str(meta["support_workbook"])).exists())
            self.assertTrue(impl_db.exists())
            self.assertTrue(raw_db.exists())
            self.assertEqual(meta["cache_db_path"], str(impl_db))
            self.assertEqual(meta["raw_cache_db_path"], str(raw_db))

    def test_ready_validator_accepts_fully_built_td_project(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            with _td_validation_context():
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )
                self.assertEqual(
                    backend._validate_test_data_project_cache_for_update(project_dir, workbook_path),
                    impl_db,
                )

    def test_open_validator_defers_generated_workbook_output_scan(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            with _td_validation_context(), patch.object(
                backend,
                "_td_validate_generated_workbook_outputs",
                side_effect=AssertionError("workbook output scan should be deferred on open"),
            ) as validate_outputs_mock:
                with patch.object(
                    backend,
                    "inspect_test_data_project_cache_state",
                    side_effect=AssertionError("open validator should not inspect source cache state"),
                ), patch.object(
                    backend,
                    "td_read_sources_metadata",
                    side_effect=AssertionError("open validator should not read workbook sources"),
                ), patch.object(
                    backend,
                    "_read_td_support_workbook",
                    side_effect=AssertionError("open validator should not read support workbook"),
                ):
                    self.assertEqual(
                        backend.validate_test_data_project_cache_for_open(project_dir, workbook_path),
                        impl_db,
                    )
                readiness = backend._td_collect_project_readiness(
                    project_dir,
                    workbook_path,
                    validate_workbook_outputs=False,
                )
            validate_outputs_mock.assert_not_called()
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(bool((readiness.get("summary") or {}).get("workbook_outputs_deferred")))

    def test_open_validator_fails_fast_when_raw_cache_db_missing(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, _impl_db, raw_db = _create_ready_td_project_fixture(project_dir)
            raw_db.unlink()
            with self.assertRaises(RuntimeError) as ctx:
                backend.validate_test_data_project_cache_for_open(project_dir, workbook_path)
            self.assertIn("Project raw cache DB not found", str(ctx.exception))
            self.assertIn("Update Project", str(ctx.exception))

    def test_open_validator_fails_fast_when_required_impl_table_missing(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            with closing(sqlite3.connect(str(impl_db))) as conn:
                conn.execute("DROP TABLE td_metrics_calc_sequences")
                conn.commit()
            with self.assertRaises(RuntimeError) as ctx:
                backend.validate_test_data_project_cache_for_open(project_dir, workbook_path)
            self.assertIn("missing required tables", str(ctx.exception))
            self.assertIn("td_metrics_calc_sequences", str(ctx.exception))

    def test_read_sources_metadata_from_cache_uses_cache_metadata(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            project_dir = Path(tmpdir) / "project"
            _workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            with closing(sqlite3.connect(str(impl_db))) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO td_source_metadata(
                        serial, program_title, document_type, metadata_rel, artifacts_rel, excel_sqlite_rel, metadata_mtime_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("SN-001", "Program Alpha", "TD", "meta/doc1.json", "artifacts/doc1", "source.sqlite3", 1),
                )
                conn.commit()
            rows = backend.td_read_sources_metadata_from_cache(impl_db)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["serial"], "SN-001")
            self.assertEqual(rows[0]["program_title"], "Program Alpha")
            self.assertEqual(rows[0]["document_type"], "TD")
            self.assertEqual(rows[0]["metadata_rel"], "meta/doc1.json")

    def test_ready_validator_accepts_excluded_source_rows_for_update_and_open(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            _append_excluded_td_source_to_fixture(project_dir, workbook_path)
            with _td_validation_context():
                for validator in (
                    backend.validate_existing_test_data_project_cache,
                    backend._validate_test_data_project_cache_for_update,
                ):
                    self.assertEqual(validator(project_dir, workbook_path), impl_db)
                readiness = backend._td_collect_project_readiness(project_dir, workbook_path)
            self.assertEqual(readiness["problems"], [])
            self.assertEqual(readiness["compiled_serials"], ["SN-001"])
            self.assertEqual(len(readiness["excluded_sources"]), 1)
            self.assertEqual(readiness["excluded_sources"][0]["serial"], "SN-002")
            self.assertIn("ignored until fixed", str(readiness.get("warning_summary") or ""))

    def test_update_succeeds_with_excluded_sources_and_rewrites_compiled_serial_headers(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, raw_db = _create_ready_td_project_fixture(project_dir)
            _append_excluded_td_source_to_fixture(project_dir, workbook_path)
            from openpyxl import load_workbook  # type: ignore

            wb_existing = load_workbook(str(workbook_path))
            try:
                ws_data_calc = wb_existing["Data_calc"]
                ws_data_calc.cell(1, 3).value = "SN-002"
                wb_existing.save(str(workbook_path))
            finally:
                wb_existing.close()

            with _td_update_context(project_dir, workbook_path):
                payload = backend.update_test_data_trending_project_workbook(
                    project_dir,
                    workbook_path,
                )

            self.assertTrue(payload["cache_validation_ok"])
            self.assertEqual(payload["db_path"], str(impl_db))
            self.assertEqual(payload["compiled_serials"], ["SN-001"])
            self.assertEqual(payload["compiled_serials_count"], 1)
            self.assertEqual(payload["serials_with_source"], 1)
            self.assertEqual(payload["excluded_sources_count"], 1)
            self.assertEqual(payload["excluded_sources"][0]["serial"], "SN-002")
            self.assertIn("ignored until fixed", str(payload.get("warning_summary") or ""))
            self.assertEqual(payload["cache_state"]["compiled_serials_count"], 1)
            self.assertEqual(payload["cache_state"]["excluded_sources_count"], 1)

            wb_check = load_workbook(str(workbook_path), read_only=True, data_only=True)
            try:
                ws_data_calc = wb_check["Data_calc"]
                header = [str(value or "").strip() for value in next(ws_data_calc.iter_rows(min_row=1, max_row=1, values_only=True))]
            finally:
                wb_check.close()
            self.assertEqual(header[:2], ["Metric", "SN-001"])
            self.assertNotIn("SN-002", header)

            with _td_validation_context():
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )

    def test_update_fails_when_all_sources_are_excluded(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, _impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            _set_fixture_source_missing(project_dir, workbook_path)
            with _td_update_context(project_dir, workbook_path):
                with self.assertRaises(RuntimeError) as ctx:
                    backend.update_test_data_trending_project_workbook(project_dir, workbook_path)
            self.assertIn("Project cache has no compiled Test Data sources", str(ctx.exception))

    def test_update_succeeds_when_post_build_validation_has_only_nonfatal_diagnostics(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, _impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            tolerated_readiness = {
                "summary": {
                    "mode": "calc",
                    "reason": "selected statistics changed",
                    "impl_complete": True,
                    "raw_complete": True,
                    "workbook_outputs": {
                        "ok": False,
                        "problems": ["Project workbook outputs are incomplete: RawCache_long has no data rows."],
                    },
                },
                "compiled_serials": ["SN-001"],
                "excluded_sources": [],
                "warning_summary": "",
                "warnings": [
                    "Project cache is stale: selected statistics changed.",
                    "Project workbook outputs are incomplete: RawCache_long has no data rows.",
                ],
                "problems": [],
            }
            with _td_update_context(project_dir, workbook_path), patch.object(
                backend,
                "_td_collect_project_readiness",
                return_value=tolerated_readiness,
            ):
                payload = backend.update_test_data_trending_project_workbook(
                    project_dir,
                    workbook_path,
                    require_existing_cache=False,
                )
            self.assertTrue(payload["cache_validation_ok"])
            self.assertEqual(str(payload.get("cache_validation_error") or ""), "")
            self.assertEqual(str((payload.get("cache_state") or {}).get("mode") or ""), "calc")
            warnings = [str(value).strip() for value in (payload.get("cache_validation_warnings") or []) if str(value).strip()]
            self.assertTrue(any("selected statistics changed" in warning for warning in warnings))
            self.assertTrue(any("RawCache_long" in warning for warning in warnings))

    def test_ready_validator_allows_workbook_output_mismatch_as_warning(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            from openpyxl import load_workbook  # type: ignore

            wb_existing = load_workbook(str(workbook_path))
            try:
                ws_raw_cache_long = wb_existing["RawCache_long"]
                ws_raw_cache_long.delete_rows(2, ws_raw_cache_long.max_row or 1)
                wb_existing.save(str(workbook_path))
            finally:
                wb_existing.close()

            with _td_validation_context():
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )
                readiness = backend._td_collect_project_readiness(project_dir, workbook_path)
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(any("RawCache_long" in warning for warning in readiness["warnings"]))

    def test_ready_validator_allows_missing_metrics_long_sheet_as_warning(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            from openpyxl import load_workbook  # type: ignore

            wb_existing = load_workbook(str(workbook_path))
            try:
                del wb_existing["Metrics_long"]
                wb_existing.save(str(workbook_path))
            finally:
                wb_existing.close()

            with _td_validation_context():
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )
                readiness = backend._td_collect_project_readiness(project_dir, workbook_path)
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(any("Metrics_long" in warning for warning in readiness["warnings"]))

    def test_ready_validator_allows_missing_metrics_long_sequences_sheet_as_warning(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            from openpyxl import load_workbook  # type: ignore

            wb_existing = load_workbook(str(workbook_path))
            try:
                del wb_existing["Metrics_long_sequences"]
                wb_existing.save(str(workbook_path))
            finally:
                wb_existing.close()

            with _td_validation_context():
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )
                readiness = backend._td_collect_project_readiness(project_dir, workbook_path)
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(any("Metrics_long_sequences" in warning for warning in readiness["warnings"]))

    def test_ready_validator_allows_data_calc_without_generated_metric_rows_as_warning(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            from openpyxl import load_workbook  # type: ignore

            wb_existing = load_workbook(str(workbook_path))
            try:
                ws_data_calc = wb_existing["Data_calc"]
                ws_data_calc.delete_rows(2, max(0, (ws_data_calc.max_row or 0) - 1))
                ws_data_calc.append(["RunA", ""])
                wb_existing.save(str(workbook_path))
            finally:
                wb_existing.close()

            with _td_validation_context():
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )
                readiness = backend._td_collect_project_readiness(project_dir, workbook_path)
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(any("Data_calc" in warning for warning in readiness["warnings"]))

    def test_ready_validator_allows_calc_only_staleness_as_warning(self) -> None:
        if Workbook is None:
            self.skipTest("openpyxl is required for TD readiness tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            workbook_path, impl_db, _raw_db = _create_ready_td_project_fixture(project_dir)
            calc_stale_cfg = {"statistics": ["mean"], "columns": list(TEST_TD_COLUMNS)}

            with _td_validation_context(), patch.object(
                backend,
                "_load_project_td_trend_config",
                return_value=calc_stale_cfg,
            ):
                self.assertEqual(
                    backend.validate_existing_test_data_project_cache(project_dir, workbook_path),
                    impl_db,
                )
                readiness = backend._td_collect_project_readiness(project_dir, workbook_path)
            self.assertEqual(readiness["problems"], [])
            self.assertTrue(any("selected statistics changed" in warning for warning in readiness["warnings"]))


if __name__ == "__main__":
    unittest.main()
