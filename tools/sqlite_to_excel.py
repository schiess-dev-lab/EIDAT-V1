from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "EIDAT_App_Files" / "Application"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from eidat_manager_excel_to_sqlite import export_sqlite_excel_mirror  # type: ignore


def _iter_sqlite_paths(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            out.extend(sorted(x for x in p.rglob("*.sqlite3") if x.is_file()))
            continue
        out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one or more SQLite files to .xlsx workbooks next to each source DB."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more .sqlite3 files or directories to scan recursively.",
    )
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        default=None,
        help="Optional row cap per table in the output workbook.",
    )
    args = parser.parse_args(argv)

    sqlite_paths = _iter_sqlite_paths(list(args.paths))
    if not sqlite_paths:
        print("No SQLite files found.", file=sys.stderr)
        return 1

    exit_code = 0
    for db_path in sqlite_paths:
        if not db_path.exists() or not db_path.is_file():
            print(f"Missing SQLite file: {db_path}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            out_path = export_sqlite_excel_mirror(
                db_path,
                max_rows_per_table=args.max_rows_per_table,
            )
            print(out_path)
        except Exception as exc:
            print(f"Failed to export {db_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
