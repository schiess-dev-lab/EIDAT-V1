from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


_PAGE_MARKER = re.compile(r"^=== Page (\d+) ===\s*$")
_LABEL_INSTANCE = re.compile(r"\s*\((\d+)\)\s*$")


def _clean_cell_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00a0", " ")
    s = s.strip()
    s = s.strip("|").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_table_border(ln: str) -> bool:
    s = str(ln or "").strip()
    return bool(s.startswith("+") and s.endswith("+") and (("-" in s) or ("=" in s)))


def _split_label_instance(label: str) -> tuple[str, int | None]:
    raw = str(label or "").strip()
    if not raw:
        return "", None
    m = _LABEL_INSTANCE.search(raw)
    if not m:
        return raw, None
    base = raw[: m.start()].strip()
    if not base:
        return raw, None
    try:
        idx = int(m.group(1))
    except Exception:
        return raw, None
    if idx <= 0:
        return raw, None
    return base, idx


@dataclass(frozen=True)
class LabeledAsciiTable:
    table_index: int
    page: int | None
    heading: str
    label: str
    label_base: str
    label_instance: int | None
    rows: list[list[str]]
    raw_table_text: str

    @property
    def n_rows(self) -> int:
        return int(len(self.rows))

    @property
    def n_cols(self) -> int:
        if not self.rows:
            return 0
        return int(max((len(r) for r in self.rows), default=0))


def parse_labeled_ascii_tables_from_combined_lines(
    lines: list[str],
    *,
    marker: str = "TABLE_LABEL",
) -> list[LabeledAsciiTable]:
    """
    Parse labeled ASCII tables from combined.txt lines.

    Only tables immediately preceded by a label block are returned:
      - f"[{marker}]"
      - next non-empty line is the label text
    """
    marker = str(marker or "TABLE_LABEL").strip() or "TABLE_LABEL"
    marker_line = f"[{marker}]"

    current_page: int | None = None
    current_heading = ""
    pending_label = ""

    out: list[LabeledAsciiTable] = []
    export_idx = 0

    i = 0
    while i < len(lines):
        ln = str(lines[i] or "").rstrip("\n")

        m = _PAGE_MARKER.match(ln.strip())
        if m:
            try:
                current_page = int(m.group(1))
            except Exception:
                current_page = None
            i += 1
            continue

        if ln.strip() in ("[Table/Chart Title]", "[STRING]"):
            j = i + 1
            while j < len(lines) and not str(lines[j] or "").strip():
                j += 1
            if j < len(lines):
                current_heading = _clean_cell_text(lines[j])
            i = j + 1
            continue

        if ln.strip() == marker_line:
            j = i + 1
            while j < len(lines) and not str(lines[j] or "").strip():
                j += 1
            if j < len(lines):
                pending_label = str(lines[j] or "").strip()
            i = j + 1
            continue

        if _looks_like_table_border(ln) and pending_label.strip():
            # Collect the contiguous ASCII table block.
            tbl_lines: list[str] = []
            j = i
            while j < len(lines):
                s = str(lines[j] or "").rstrip("\n")
                if not s.strip():
                    break
                st = s.strip()
                if not (st.startswith(("+", "|"))):
                    break
                tbl_lines.append(s)
                j += 1

            # Parse rows from "|" lines.
            rows: list[list[str]] = []
            for tln in tbl_lines:
                if not str(tln).strip().startswith("|"):
                    continue
                parts = [p.strip() for p in str(tln).strip().strip("|").split("|")]
                parts = [_clean_cell_text(p) for p in parts]
                while parts and not parts[-1]:
                    parts.pop()
                if parts:
                    rows.append(parts)

            export_idx += 1
            label = str(pending_label).strip()
            label_base, label_instance = _split_label_instance(label)
            out.append(
                LabeledAsciiTable(
                    table_index=export_idx,
                    page=current_page,
                    heading=current_heading,
                    label=label,
                    label_base=label_base,
                    label_instance=label_instance,
                    rows=rows,
                    raw_table_text="\n".join(tbl_lines),
                )
            )
            pending_label = ""
            i = j + 1
            continue

        i += 1

    return out


_SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tables (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_index INTEGER NOT NULL,
  page INTEGER,
  heading TEXT,
  label TEXT NOT NULL,
  label_base TEXT NOT NULL,
  label_instance INTEGER,
  n_rows INTEGER NOT NULL,
  n_cols INTEGER NOT NULL,
  raw_table_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cells (
  table_id INTEGER NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
  row_index INTEGER NOT NULL,
  col_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  PRIMARY KEY (table_id, row_index, col_index)
);

CREATE INDEX IF NOT EXISTS idx_tables_label ON tables(label_base, label_instance);
CREATE INDEX IF NOT EXISTS idx_tables_page ON tables(page);
"""


def export_labeled_tables_db(
    *,
    artifacts_dir: Path,
    combined_txt_path: Path,
    marker: str = "TABLE_LABEL",
    db_name: str = "labeled_tables.db",
) -> Path | None:
    """
    Export labeled ASCII tables from `combined.txt` into an artifacts-local SQLite DB.

    If no labeled tables are found, deletes the DB if present and returns None.
    """
    artifacts_dir = Path(artifacts_dir).expanduser()
    combined_txt_path = Path(combined_txt_path).expanduser()
    db_path = artifacts_dir / str(db_name or "labeled_tables.db")

    if not combined_txt_path.exists():
        try:
            db_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            pass
        return None

    combined_bytes = combined_txt_path.read_bytes()
    combined_sha1 = hashlib.sha1(combined_bytes).hexdigest()
    lines = combined_bytes.decode("utf-8", errors="ignore").splitlines(True)
    tables = parse_labeled_ascii_tables_from_combined_lines(lines, marker=marker)

    if not tables:
        try:
            db_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            pass
        return None

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        db_path.unlink(missing_ok=True)  # type: ignore[call-arg]
    except TypeError:
        if db_path.exists():
            db_path.unlink()
    except Exception:
        pass

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("schema_version", "1"))
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("combined_sha1", combined_sha1))
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            ("exported_epoch_ns", str(time.time_ns())),
        )

        for t in tables:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO tables(
                  table_index, page, heading, label, label_base, label_instance,
                  n_rows, n_cols, raw_table_text
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(t.table_index),
                    (None if t.page is None else int(t.page)),
                    str(t.heading or ""),
                    str(t.label),
                    str(t.label_base),
                    (None if t.label_instance is None else int(t.label_instance)),
                    int(t.n_rows),
                    int(t.n_cols),
                    str(t.raw_table_text or ""),
                ),
            )
            table_id = int(cur.lastrowid)

            n_cols = int(t.n_cols)
            for r_idx, row in enumerate(t.rows):
                padded = list(row) + ([""] * max(0, n_cols - len(row)))
                for c_idx in range(n_cols):
                    conn.execute(
                        "INSERT INTO cells(table_id, row_index, col_index, text) VALUES(?, ?, ?, ?)",
                        (table_id, int(r_idx), int(c_idx), str(padded[c_idx])),
                    )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return db_path
