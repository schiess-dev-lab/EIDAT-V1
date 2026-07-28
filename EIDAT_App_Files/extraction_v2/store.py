"""V2 persistence.

Two databases, both SQLite WAL (same discipline as the rest of EIDAT):

* Central rule library  — `<runtime_root>/extraction_v2_rules.sqlite3`
  Mapping decisions are made once and shared by every node.
* Per-node store        — `<node_root>/EIDAT/EIDAT Support/extract_v2/v2.sqlite3`
  Scanned docs, extracted facts (long format), and the mapping queue.

Facts are written only through a mapping rule, so the fact store contains
canonical labels exclusively. Unmapped shapes wait in the queue; resolving
a rule replays stored document.json files — no re-OCR.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import fingerprint as fp

RULES_DDL = """
CREATE TABLE IF NOT EXISTS rules(
  rule_id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,                -- 'table' | 'kv'
  canonical_label TEXT NOT NULL,
  fingerprint_json TEXT NOT NULL,
  created_ts REAL NOT NULL
);
"""

NODE_DDL = """
CREATE TABLE IF NOT EXISTS docs(
  doc_id INTEGER PRIMARY KEY,
  file TEXT UNIQUE NOT NULL,
  sha256 TEXT NOT NULL,
  json_path TEXT NOT NULL,
  program TEXT, serial TEXT, test_date TEXT,
  scanned_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts(
  doc_id INT NOT NULL REFERENCES docs(doc_id),
  canonical_label TEXT NOT NULL,
  row_key TEXT, col_name TEXT,
  text TEXT, num REAL, unit TEXT, kind TEXT,
  page INT, conf REAL,
  obj_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts ON facts(canonical_label, row_key, col_name);
CREATE TABLE IF NOT EXISTS queue(
  queue_id INTEGER PRIMARY KEY,
  doc_id INT NOT NULL REFERENCES docs(doc_id),
  obj_kind TEXT NOT NULL,            -- 'table' | 'kv'
  obj_id TEXT NOT NULL,              -- table_id / group_id in document.json
  fingerprint_json TEXT NOT NULL,
  preview_json TEXT NOT NULL,        -- rows / pairs for display
  crop_path TEXT,
  status TEXT NOT NULL DEFAULT 'pending',   -- pending | resolved | skipped
  resolved_label TEXT,
  UNIQUE(doc_id, obj_id)
);
"""


def open_rules(runtime_root: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(Path(runtime_root) / "extraction_v2_rules.sqlite3"))
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(RULES_DDL)
    return db


def node_store_dir(node_root: str | Path) -> Path:
    return Path(node_root) / "EIDAT" / "EIDAT Support" / "extract_v2"


def open_node(node_root: str | Path) -> sqlite3.Connection:
    d = node_store_dir(node_root)
    d.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(d / "v2.sqlite3"))
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(NODE_DDL)
    return db


# ---------------------------------------------------------------- rules ---
def list_rules(rules_db: sqlite3.Connection) -> list[dict]:
    return [
        {
            "rule_id": r[0],
            "kind": r[1],
            "canonical_label": r[2],
            "fingerprint": json.loads(r[3]),
        }
        for r in rules_db.execute(
            "SELECT rule_id, kind, canonical_label, fingerprint_json FROM rules"
        )
    ]


def known_labels(rules_db: sqlite3.Connection, kind: str | None = None) -> list[str]:
    q = "SELECT DISTINCT canonical_label FROM rules"
    args: tuple = ()
    if kind:
        q += " WHERE kind=?"
        args = (kind,)
    return [r[0] for r in rules_db.execute(q + " ORDER BY canonical_label", args)]


def add_rule(
    rules_db: sqlite3.Connection, kind: str, canonical_label: str, fingerprint: dict
) -> int:
    cur = rules_db.execute(
        "INSERT INTO rules(kind, canonical_label, fingerprint_json, created_ts)"
        " VALUES(?,?,?,?)",
        (kind, canonical_label, fp.dumps(fingerprint), time.time()),
    )
    rules_db.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------- facts ---
def write_table_facts(
    node_db: sqlite3.Connection, doc_id: int, table: dict, canonical_label: str
) -> int:
    header = table["header"]
    rows = table["rows"]
    cells = {(c["r"], c["c"]): c for c in table["cells"]}
    n = 0
    for r_idx in range(1, len(rows)):
        row_key = (rows[r_idx][0] or "").strip()
        for c_idx in range(len(rows[r_idx])):
            cell = cells.get((r_idx, c_idx))
            if cell is None:
                continue
            v = cell["value"]
            node_db.execute(
                "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    canonical_label,
                    row_key,
                    (header[c_idx] or f"col{c_idx}").strip() if c_idx < len(header) else f"col{c_idx}",
                    cell["text"],
                    v.get("num", v.get("lo")),
                    v.get("unit"),
                    v["kind"],
                    table["page"],
                    cell["conf"],
                    table["table_id"],
                ),
            )
            n += 1
    return n


def write_kv_facts(
    node_db: sqlite3.Connection, doc_id: int, group: dict, canonical_label: str
) -> int:
    n = 0
    for p in group["pairs"]:
        v = p["value_typed"]
        node_db.execute(
            "INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                doc_id,
                canonical_label,
                p["label"],
                "Value",
                p["value"],
                v.get("num", v.get("lo")),
                v.get("unit"),
                v["kind"],
                group["page"],
                p["conf"],
                group["group_id"],
            ),
        )
        n += 1
    return n


# ---------------------------------------------------------------- mapping ---
def map_document(
    node_db: sqlite3.Connection,
    rules_db: sqlite3.Connection,
    doc_id: int,
    doc: dict,
    crop_fn=None,
) -> tuple[int, int]:
    """Map every table/kv group in `doc`; write facts for matches, queue the
    rest. Returns (facts_written, queued). Idempotent per (doc, object).
    """
    rules = list_rules(rules_db)
    facts_written = 0
    queued = 0

    def _mapped_obj_ids() -> set[str]:
        return {
            r[0]
            for r in node_db.execute(
                "SELECT DISTINCT obj_id FROM facts WHERE doc_id=?", (doc_id,)
            )
        }

    done = _mapped_obj_ids()

    for table in doc.get("tables", []):
        if table["table_id"] in done:
            continue
        f = fp.table_fingerprint(table)
        rule, _score = fp.best_match(f, rules)
        if rule:
            facts_written += write_table_facts(
                node_db, doc_id, table, rule["canonical_label"]
            )
            node_db.execute(
                "UPDATE queue SET status='resolved', resolved_label=? "
                "WHERE doc_id=? AND obj_id=?",
                (rule["canonical_label"], doc_id, table["table_id"]),
            )
        else:
            crop = crop_fn(table["page"], table["bbox"]) if crop_fn else None
            node_db.execute(
                "INSERT OR IGNORE INTO queue"
                "(doc_id, obj_kind, obj_id, fingerprint_json, preview_json, crop_path)"
                " VALUES(?,?,?,?,?,?)",
                (
                    doc_id,
                    "table",
                    table["table_id"],
                    fp.dumps(f),
                    json.dumps(
                        {"title": table.get("title", ""), "rows": table["rows"][:8]}
                    ),
                    crop,
                ),
            )
            queued += 1

    for group in doc.get("kv_groups", []):
        if group["group_id"] in done:
            continue
        f = fp.kv_fingerprint(group)
        rule, _score = fp.best_match(f, rules)
        if rule:
            facts_written += write_kv_facts(
                node_db, doc_id, group, rule["canonical_label"]
            )
            node_db.execute(
                "UPDATE queue SET status='resolved', resolved_label=? "
                "WHERE doc_id=? AND obj_id=?",
                (rule["canonical_label"], doc_id, group["group_id"]),
            )
        else:
            crop = crop_fn(group["page"], group["bbox"]) if crop_fn else None
            node_db.execute(
                "INSERT OR IGNORE INTO queue"
                "(doc_id, obj_kind, obj_id, fingerprint_json, preview_json, crop_path)"
                " VALUES(?,?,?,?,?,?)",
                (
                    doc_id,
                    "kv",
                    group["group_id"],
                    fp.dumps(f),
                    json.dumps(
                        {"pairs": [[p["label"], p["value"]] for p in group["pairs"]][:10]}
                    ),
                    crop,
                ),
            )
            queued += 1

    node_db.commit()
    return facts_written, queued


def replay_node(
    node_db: sqlite3.Connection, rules_db: sqlite3.Connection
) -> tuple[int, int]:
    """Re-run mapping over every stored document.json (no re-OCR).

    Called after a new rule is saved so the decision back-fills history.
    """
    total_facts = 0
    still_queued = 0
    for doc_id, json_path in node_db.execute("SELECT doc_id, json_path FROM docs"):
        try:
            doc = json.loads(Path(json_path).read_text())
        except Exception:
            continue
        f, q = map_document(node_db, rules_db, doc_id, doc)
        total_facts += f
        still_queued += q
    return total_facts, still_queued


def pending_count(node_db: sqlite3.Connection) -> int:
    return int(
        node_db.execute(
            "SELECT COUNT(*) FROM queue WHERE status='pending'"
        ).fetchone()[0]
    )
