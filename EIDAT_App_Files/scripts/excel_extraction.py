#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Excel trend config must be a JSON object.")
    cols = data.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("Excel trend config must include a non-empty `columns` list.")
    stats = data.get("statistics") or ["mean", "min", "max", "std", "median", "count"]
    if not isinstance(stats, list) or not all(isinstance(s, str) and s for s in stats):
        raise ValueError("Excel trend config `statistics` must be a list of strings.")
    data["statistics"] = [str(s).strip().lower() for s in stats]
    if "header_row" not in data:
        data["header_row"] = 0
    return data


_SN_RE = re.compile(r"^(sn|s\/n)[\s_\-]*([0-9a-z]+)$", re.IGNORECASE)


def derive_file_identity(excel_path: Path) -> Tuple[str, str, str]:
    """
    Heuristic identity from filename like: Program_Vehicle_SN1001.xlsx
    Returns (program, vehicle, serial).
    """
    stem = excel_path.stem
    parts = [p for p in stem.split("_") if p]
    program = parts[0] if parts else ""
    vehicle = parts[1] if len(parts) >= 2 else ""
    serial = ""
    for p in reversed(parts):
        m = _SN_RE.match(p.strip())
        if m:
            serial = f"SN{m.group(2)}"
            break
        if p.strip().lower().startswith("sn") and len(p.strip()) > 2:
            serial = p.strip().upper()
            break
    return program, vehicle, serial


def _read_dataframe(excel_path: Path, *, sheet_name: Optional[str], header_row: int):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for Excel extraction in this repo.") from exc

    sheet = sheet_name if sheet_name not in ("", None) else 0
    try:
        return pd.read_excel(excel_path, sheet_name=sheet, header=int(header_row))
    except ValueError:
        # sheet name missing, fall back to first sheet
        return pd.read_excel(excel_path, sheet_name=0, header=int(header_row))


def _normalize_col_name(name: Any) -> str:
    return str(name).strip().lower()


def _stat_value(series, stat: str):
    stat = stat.lower().strip()
    if stat == "mean":
        return float(series.mean())
    if stat == "min":
        return float(series.min())
    if stat == "max":
        return float(series.max())
    if stat == "std":
        val = series.std()
        return float(val) if val == val else None  # NaN guard
    if stat == "median":
        return float(series.median())
    if stat == "count":
        return int(series.count())
    raise ValueError(f"Unknown statistic: {stat}")


def extract_from_excel(excel_path: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    sheet_name = config.get("sheet_name", None)
    header_row = int(config.get("header_row", 0) or 0)
    df = _read_dataframe(excel_path, sheet_name=sheet_name, header_row=header_row)

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for Excel extraction in this repo.") from exc

    cols_cfg = config.get("columns") or []
    stats = [str(s).strip().lower() for s in (config.get("statistics") or [])]
    if not stats:
        stats = ["mean", "min", "max", "std", "median", "count"]

    col_map = {_normalize_col_name(c): c for c in list(df.columns)}
    program, vehicle, serial = derive_file_identity(excel_path)

    rows: List[Dict[str, Any]] = []
    for col in cols_cfg:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name") or "").strip()
        if not name:
            continue
        units = col.get("units")
        range_min = col.get("range_min")
        range_max = col.get("range_max")
        df_col = col_map.get(name.strip().lower())
        if df_col is None:
            rows.append(
                {
                    "source_file": str(excel_path),
                    "program": program,
                    "vehicle": vehicle,
                    "serial": serial,
                    "column": name,
                    "units": units,
                    "stat": "error",
                    "value": None,
                    "error": f"Missing column: {name}",
                    "range_min": range_min,
                    "range_max": range_max,
                }
            )
            continue
        numeric = pd.to_numeric(df[df_col], errors="coerce")
        numeric = numeric.dropna()
        if numeric.empty:
            rows.append(
                {
                    "source_file": str(excel_path),
                    "program": program,
                    "vehicle": vehicle,
                    "serial": serial,
                    "column": name,
                    "units": units,
                    "stat": "error",
                    "value": None,
                    "error": f"No numeric data in column: {name}",
                    "range_min": range_min,
                    "range_max": range_max,
                }
            )
            continue
        for stat in stats:
            try:
                val = _stat_value(numeric, stat)
            except Exception as exc:
                rows.append(
                    {
                        "source_file": str(excel_path),
                        "program": program,
                        "vehicle": vehicle,
                        "serial": serial,
                        "column": name,
                        "units": units,
                        "stat": stat,
                        "value": None,
                        "error": str(exc),
                        "range_min": range_min,
                        "range_max": range_max,
                    }
                )
                continue
            rows.append(
                {
                    "source_file": str(excel_path),
                    "program": program,
                    "vehicle": vehicle,
                    "serial": serial,
                    "column": name,
                    "units": units,
                    "stat": stat,
                    "value": val,
                    "error": None,
                    "range_min": range_min,
                    "range_max": range_max,
                }
            )
    return rows


def _output_root(global_repo: Optional[Path]) -> Path:
    env_root = os.environ.get("MERGED_OCR_ROOT") or os.environ.get("OCR_ROOT") or ""
    if str(env_root).strip():
        return Path(env_root)
    if global_repo is not None:
        return global_repo / "global_run_mirror" / "debug" / "ocr"
    return Path.cwd() / "global_run_mirror" / "debug" / "ocr"


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    (output_dir / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # Optional human-readable "combined" text for downstream metadata and quick inspection.
    lines: List[str] = []
    if rows:
        src = rows[0].get("source_file") or ""
        lines.append(f"Excel file: {src}")
        for key in ("program", "vehicle", "serial"):
            val = (rows[0].get(key) or "").strip()
            if val:
                lines.append(f"{key}: {val}")
        lines.append("")
        for r in rows:
            col = r.get("column")
            stat = r.get("stat")
            val = r.get("value")
            if stat == "error":
                err = r.get("error") or ""
                lines.append(f"{col}: ERROR: {err}")
            else:
                units = r.get("units") or ""
                suffix = f" {units}".rstrip()
                lines.append(f"{col}.{stat} = {val}{suffix}")
    (output_dir / "combined.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _iter_excel_paths(values: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in values:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Excel file not found: {p}")
        out.append(p)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract configured trend stats from Excel files.")
    parser.add_argument("--global-repo", type=str, default="", help="Path to the global repo root (output fallback).")
    parser.add_argument("--excel", action="append", default=[], help="Excel file path (repeatable).")
    parser.add_argument("--config", type=str, required=True, help="Path to user_inputs/excel_trend_config.json")
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Override output root. Defaults to MERGED_OCR_ROOT or <global-repo>/global_run_mirror/debug/ocr.",
    )
    args = parser.parse_args(argv)

    excel_paths = _iter_excel_paths(args.excel)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)

    global_repo = Path(args.global_repo) if str(args.global_repo).strip() else None
    out_root = Path(args.out_root) if str(args.out_root).strip() else _output_root(global_repo)
    out_root.mkdir(parents=True, exist_ok=True)

    for excel_path in excel_paths:
        out_dir = out_root / f"{excel_path.stem}__excel"
        rows = extract_from_excel(excel_path, cfg)
        write_outputs(rows, out_dir)
        print(f"[DONE] {excel_path.name} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

