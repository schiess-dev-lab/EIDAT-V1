from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SupportPaths:
    global_repo: Path
    support_dir: Path
    db_path: Path
    logs_dir: Path
    staging_dir: Path


def support_paths(global_repo: Path) -> SupportPaths:
    repo = Path(global_repo).expanduser()
    support_dir = repo / "EIDAT Support"
    return SupportPaths(
        global_repo=repo,
        support_dir=support_dir,
        db_path=support_dir / "eidat_support.sqlite3",
        logs_dir=support_dir / "logs",
        staging_dir=support_dir / "staging",
    )


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_epoch_ns INTEGER NOT NULL,
  finished_epoch_ns INTEGER NOT NULL,
  global_repo TEXT NOT NULL,
  pdf_count INTEGER NOT NULL,
  candidates_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rel_path TEXT NOT NULL UNIQUE,
  file_fingerprint TEXT,
  content_sha1 TEXT,
  eidat_uuid TEXT,
  pointer_token TEXT,
  size_bytes INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  first_seen_epoch_ns INTEGER NOT NULL,
  last_seen_epoch_ns INTEGER NOT NULL,
  last_processed_epoch_ns INTEGER,
  last_processed_mtime_ns INTEGER,
  needs_processing INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_files_needs_processing ON files(needs_processing);
CREATE INDEX IF NOT EXISTS idx_files_last_seen ON files(last_seen_epoch_ns);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _ensure_column(conn, "files", "file_fingerprint", "file_fingerprint TEXT")
    _ensure_column(conn, "files", "content_sha1", "content_sha1 TEXT")
    _ensure_column(conn, "files", "eidat_uuid", "eidat_uuid TEXT")
    _ensure_column(conn, "files", "pointer_token", "pointer_token TEXT")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(r[1]) for r in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
    except Exception:
        pass


def get_meta_int(conn: sqlite3.Connection, key: str, default: int = 0) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return int(str(row["value"]).strip())
    except Exception:
        return default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
