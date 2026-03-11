#!/usr/bin/env python3
"""
Convert MATLAB `.mat` files into Excel workbooks for the existing EIDAT TD pipeline.

Common cases supported:
- Numeric vector/matrix -> one sheet with columns/rows preserved
- Dict-like structs -> one or more sheets with flattened key paths
- MATLAB cell/object arrays -> flattened into rows where possible

Usage:
  python tools/mat_to_excel.py input.mat
  python tools/mat_to_excel.py input.mat --output input.xlsx
  python tools/mat_to_excel.py a.mat b.mat --out-dir converted_excels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    clean = "".join("_" if ch in r'[]:*?/\\' else ch for ch in str(name or "sheet"))
    clean = clean.strip() or "sheet"
    clean = clean[:31]
    base = clean
    i = 2
    lowered = {x.lower() for x in used}
    while clean.lower() in lowered:
        suffix = f"_{i}"
        clean = (base[: max(0, 31 - len(suffix))] + suffix).strip() or f"sheet{i}"
        i += 1
    used.add(clean)
    return clean


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required to read MATLAB `.mat` files.") from exc

    try:
        data = loadmat(str(path), simplify_cells=True)
        if isinstance(data, dict):
            return data
    except NotImplementedError:
        pass
    except ValueError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Failed to read MAT file {path}: {exc}") from exc

    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "This looks like an HDF5-based MATLAB file. Install `h5py` to read MATLAB v7.3 files."
        ) from exc

    try:
        with h5py.File(str(path), "r") as handle:
            return {k: _from_h5(v) for k, v in handle.items()}
    except Exception as exc:
        raise RuntimeError(f"Failed to read HDF5-based MAT file {path}: {exc}") from exc


def _from_h5(node: Any) -> Any:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("h5py and numpy are required for MATLAB v7.3 files.") from exc

    if isinstance(node, h5py.Dataset):
        data = node[()]
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return repr(data)
        if isinstance(data, np.ndarray):
            return data
        return data
    if isinstance(node, h5py.Group):
        return {k: _from_h5(v) for k, v in node.items()}
    return node


def _is_scalar(value: Any) -> bool:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if np is not None and isinstance(value, np.generic):
        return True
    return False


def _coerce_scalar(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return repr(value)
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _flatten_record(value: Any, prefix: str = "") -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    out: dict[str, Any] = {}
    if _is_scalar(value):
        out[prefix or "value"] = _coerce_scalar(value)
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if _is_scalar(v):
                out[key] = _coerce_scalar(v)
            else:
                out.update(_flatten_record(v, key))
        return out

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            key = f"{prefix}[{i}]"
            if _is_scalar(item):
                out[key] = _coerce_scalar(item)
            else:
                out.update(_flatten_record(item, key))
        return out

    if np is not None and isinstance(value, np.ndarray):
        if value.ndim == 0:
            out[prefix or "value"] = _coerce_scalar(value.item())
            return out
        if value.ndim == 1 and all(_is_scalar(x) for x in value.tolist()):
            for i, item in enumerate(value.tolist()):
                out[f"{prefix}[{i}]"] = _coerce_scalar(item)
            return out
        if value.ndim == 2 and value.shape[0] == 1 and all(_is_scalar(x) for x in value[0].tolist()):
            for i, item in enumerate(value[0].tolist()):
                out[f"{prefix}[{i}]"] = _coerce_scalar(item)
            return out
        if value.dtype.names:
            names = list(value.dtype.names)
            for idx in range(int(value.size)):
                item = value.reshape(-1)[idx]
                row_prefix = f"{prefix}[{idx}]"
                for name in names:
                    key = f"{row_prefix}.{name}"
                    out.update(_flatten_record(item[name], key))
            return out
        out[prefix or "value"] = repr(value.shape)
        return out

    out[prefix or "value"] = str(value)
    return out


def _to_dataframes(data: dict[str, Any]) -> list[tuple[str, Any]]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas and numpy are required to write Excel output.") from exc

    sheets: list[tuple[str, Any]] = []
    for raw_key, raw_value in sorted(data.items()):
        key = str(raw_key or "").strip()
        if not key or key.startswith("__"):
            continue
        value = raw_value

        if isinstance(value, dict):
            row = _flatten_record(value)
            sheets.append((key, pd.DataFrame([row])))
            continue

        if isinstance(value, (list, tuple)):
            if value and all(isinstance(x, dict) for x in value):
                rows = [_flatten_record(x) for x in value]
                sheets.append((key, pd.DataFrame(rows)))
            elif value and all(_is_scalar(x) for x in value):
                sheets.append((key, pd.DataFrame({key: [_coerce_scalar(x) for x in value]})))
            else:
                rows = [_flatten_record(x) for x in value]
                sheets.append((key, pd.DataFrame(rows)))
            continue

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                sheets.append((key, pd.DataFrame([{key: _coerce_scalar(value.item())}])))
            elif value.dtype.names:
                rows = []
                for item in value.reshape(-1):
                    row: dict[str, Any] = {}
                    for name in value.dtype.names:
                        row.update(_flatten_record(item[name], str(name)))
                    rows.append(row)
                sheets.append((key, pd.DataFrame(rows)))
            elif value.ndim == 1:
                sheets.append((key, pd.DataFrame({key: [_coerce_scalar(x) for x in value.tolist()]})))
            elif value.ndim == 2:
                sheets.append((key, pd.DataFrame(value)))
            else:
                flat = value.reshape(value.shape[0], -1) if value.shape else value.reshape(1, -1)
                sheets.append((key, pd.DataFrame(flat)))
            continue

        if _is_scalar(value):
            sheets.append((key, pd.DataFrame([{key: _coerce_scalar(value)}])))
            continue

        sheets.append((key, pd.DataFrame([_flatten_record(value)])))

    if not sheets:
        sheets.append(("data", pd.DataFrame([{"message": "No user variables found in MAT file"}])))
    return sheets


def convert_mat_to_excel(mat_path: Path, out_path: Path) -> Path:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to write Excel output.") from exc

    payload = _load_mat(mat_path)
    sheets = _to_dataframes(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for raw_name, df in sheets:
            name = _sanitize_sheet_name(raw_name, used_names)
            df.to_excel(writer, sheet_name=name, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert MATLAB `.mat` files into Excel workbooks.")
    parser.add_argument("mat", nargs="+", help="MATLAB file path(s).")
    parser.add_argument("--output", help="Explicit output .xlsx path for a single input file.")
    parser.add_argument("--out-dir", help="Directory for converted .xlsx files.")
    args = parser.parse_args(argv)

    mat_paths = [Path(p).expanduser() for p in args.mat]
    for mat_path in mat_paths:
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

    if args.output and len(mat_paths) != 1:
        raise ValueError("--output can only be used with a single input MAT file.")

    out_dir = Path(args.out_dir).expanduser() if str(args.out_dir or "").strip() else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for mat_path in mat_paths:
        if args.output:
            out_path = Path(args.output).expanduser()
        elif out_dir is not None:
            out_path = out_dir / f"{mat_path.stem}.xlsx"
        else:
            out_path = mat_path.with_suffix(".xlsx")
        result = convert_mat_to_excel(mat_path, out_path)
        print(f"[DONE] {mat_path} -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
