from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


def default_registry_path() -> Path:
    override = (os.environ.get("EIDAT_ADMIN_REGISTRY_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "EIDAT" / "Admin" / "admin_registry.sqlite3"
    return Path.home() / ".local" / "share" / "eidat" / "admin_registry.sqlite3"


def connect_registry(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else default_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA busy_timeout=2500;")
    except Exception:
        pass
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes (
          node_id TEXT PRIMARY KEY,
          node_root TEXT NOT NULL UNIQUE,
          runtime_root TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          node_env_enabled INTEGER NOT NULL DEFAULT 0,
          notes TEXT,
          added_epoch_ns INTEGER NOT NULL,
          updated_epoch_ns INTEGER NOT NULL,
          last_run_status TEXT,
          last_run_finished_epoch_ns INTEGER
        );

        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          node_id TEXT NOT NULL,
          started_epoch_ns INTEGER NOT NULL,
          finished_epoch_ns INTEGER,
          status TEXT NOT NULL,
          summary_json TEXT,
          error TEXT,
          FOREIGN KEY(node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_enabled ON nodes(enabled);
        CREATE INDEX IF NOT EXISTS idx_runs_node_started ON runs(node_id, started_epoch_ns);
        """
    )

    # Lightweight migration for older registries.
    try:
        cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(nodes);").fetchall()}
        if "node_env_enabled" not in cols:
            conn.execute("ALTER TABLE nodes ADD COLUMN node_env_enabled INTEGER NOT NULL DEFAULT 0;")
    except Exception:
        pass
    conn.commit()


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    node_root: str
    runtime_root: str
    enabled: bool
    node_env_enabled: bool
    notes: str | None
    added_epoch_ns: int
    updated_epoch_ns: int
    last_run_status: str | None = None
    last_run_finished_epoch_ns: int | None = None


def list_nodes(conn: sqlite3.Connection) -> list[NodeRecord]:
    rows = conn.execute(
        """
        SELECT node_id, node_root, runtime_root, enabled, node_env_enabled, notes,
               added_epoch_ns, updated_epoch_ns, last_run_status, last_run_finished_epoch_ns
        FROM nodes
        ORDER BY enabled DESC, updated_epoch_ns DESC
        """
    ).fetchall()
    out: list[NodeRecord] = []
    for r in rows:
        out.append(
            NodeRecord(
                node_id=str(r["node_id"]),
                node_root=str(r["node_root"]),
                runtime_root=str(r["runtime_root"]),
                enabled=bool(int(r["enabled"] or 0)),
                node_env_enabled=bool(int(r["node_env_enabled"] or 0)),
                notes=(str(r["notes"]) if r["notes"] is not None else None),
                added_epoch_ns=int(r["added_epoch_ns"] or 0),
                updated_epoch_ns=int(r["updated_epoch_ns"] or 0),
                last_run_status=(str(r["last_run_status"]) if r["last_run_status"] is not None else None),
                last_run_finished_epoch_ns=(int(r["last_run_finished_epoch_ns"]) if r["last_run_finished_epoch_ns"] is not None else None),
            )
        )
    return out


def upsert_node(conn: sqlite3.Connection, *, node_root: str, runtime_root: str, enabled: bool = True, notes: str | None = None) -> str:
    now = time.time_ns()
    row = conn.execute("SELECT node_id FROM nodes WHERE node_root = ?", (node_root,)).fetchone()
    if row:
        node_id = str(row["node_id"])
        conn.execute(
            """
            UPDATE nodes
            SET runtime_root = ?, enabled = ?, notes = ?, updated_epoch_ns = ?
            WHERE node_id = ?
            """,
            (runtime_root, 1 if enabled else 0, notes, now, node_id),
        )
    else:
        node_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO nodes(node_id, node_root, runtime_root, enabled, notes, added_epoch_ns, updated_epoch_ns)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (node_id, node_root, runtime_root, 1 if enabled else 0, notes, now, now),
        )
    conn.commit()
    return node_id


def set_node_enabled(conn: sqlite3.Connection, *, node_id: str, enabled: bool) -> None:
    conn.execute(
        "UPDATE nodes SET enabled = ?, updated_epoch_ns = ? WHERE node_id = ?",
        (1 if enabled else 0, time.time_ns(), node_id),
    )
    conn.commit()


def set_node_env_enabled(conn: sqlite3.Connection, *, node_id: str, enabled: bool) -> None:
    conn.execute(
        "UPDATE nodes SET node_env_enabled = ?, updated_epoch_ns = ? WHERE node_id = ?",
        (1 if enabled else 0, time.time_ns(), node_id),
    )
    conn.commit()


def delete_node(conn: sqlite3.Connection, *, node_id: str) -> None:
    conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
    conn.commit()


def set_node_notes(conn: sqlite3.Connection, *, node_id: str, notes: str | None) -> None:
    conn.execute(
        "UPDATE nodes SET notes = ?, updated_epoch_ns = ? WHERE node_id = ?",
        (notes, time.time_ns(), node_id),
    )
    conn.commit()


def insert_run(
    conn: sqlite3.Connection,
    *,
    node_id: str,
    started_epoch_ns: int,
    finished_epoch_ns: int | None,
    status: str,
    summary_json: str | None = None,
    error: str | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO runs(run_id, node_id, started_epoch_ns, finished_epoch_ns, status, summary_json, error)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, node_id, int(started_epoch_ns), finished_epoch_ns, status, summary_json, error),
    )
    conn.execute(
        """
        UPDATE nodes
        SET last_run_status = ?, last_run_finished_epoch_ns = ?, updated_epoch_ns = ?
        WHERE node_id = ?
        """,
        (status, finished_epoch_ns, time.time_ns(), node_id),
    )
    conn.commit()
    return run_id
