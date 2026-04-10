from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


_EIDAT_CONTAINER_DIR_NAMES = ("EIDAT", "EDAT")
_EIDAT_SUPPORT_DIR_NAMES = ("EIDAT Support", "EDAT Support")
_EIDAT_SUPPORT_MAX_DEPTH = 5


@dataclass(frozen=True)
class SupportPaths:
    global_repo: Path
    support_dir: Path
    db_path: Path
    logs_dir: Path
    staging_dir: Path


def _candidate_support_dirs(global_repo: Path) -> list[Path]:
    repo = Path(global_repo).expanduser()
    candidates: list[Path] = []
    seen: set[str] = set()

    def _append(path: Path) -> None:
        key = str(path).casefold()
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for support_name in _EIDAT_SUPPORT_DIR_NAMES:
        _append(repo / support_name)

    chain_bases = [repo]
    for _ in range(_EIDAT_SUPPORT_MAX_DEPTH):
        next_bases: list[Path] = []
        for base in chain_bases:
            for dirname in _EIDAT_CONTAINER_DIR_NAMES:
                child = base / dirname
                next_bases.append(child)
                for support_name in _EIDAT_SUPPORT_DIR_NAMES:
                    _append(child / support_name)
        chain_bases = next_bases
    return candidates


def _default_support_dir(global_repo: Path) -> Path:
    repo = Path(global_repo).expanduser()
    deepest_base = repo
    chain_bases = [repo]
    for _ in range(_EIDAT_SUPPORT_MAX_DEPTH):
        next_bases: list[Path] = []
        for base in chain_bases:
            for dirname in _EIDAT_CONTAINER_DIR_NAMES:
                child = base / dirname
                try:
                    if child.is_dir():
                        next_bases.append(child)
                except Exception:
                    continue
        if not next_bases:
            break
        deepest_base = max(next_bases, key=lambda path: (len(path.parts), str(path).casefold()))
        chain_bases = next_bases
    return deepest_base / _EIDAT_SUPPORT_DIR_NAMES[0]


def _support_dir_score(path: Path) -> tuple[int, int]:
    score = 0
    markers = (
        ("eidat_index.sqlite3", 12),
        ("eidat_support.sqlite3", 12),
        ("projects/projects_registry.sqlite3", 10),
        ("projects", 3),
        ("debug/ocr", 3),
        ("excel_sqlite", 2),
        ("logs", 1),
        ("staging", 1),
    )
    for rel_path, weight in markers:
        try:
            if (path / rel_path).exists():
                score += weight
        except Exception:
            continue
    return score, len(path.parts)


def support_paths(global_repo: Path) -> SupportPaths:
    repo = Path(global_repo).expanduser()
    existing: list[Path] = []
    for candidate in _candidate_support_dirs(repo):
        try:
            if candidate.is_dir():
                existing.append(candidate)
        except Exception:
            continue
    support_dir = max(existing, key=_support_dir_score) if existing else _default_support_dir(repo)
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
