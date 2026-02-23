from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1] / "EIDAT_App_Files"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from extraction.table_labeler import load_table_label_rules, label_combined_lines  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a fake EIDP combined.txt with repeating tables and label it.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="EIDAT Support/debug/ocr/Fake_EIDP_Repeating_Tables",
        help="Output artifacts folder (relative to repo root by default).",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        default="Synthetic_EIDPs/Fake_EIDP_Repeating_Tables__combined.txt",
        help="Fixture combined.txt (relative to repo root by default).",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="user_inputs/table_label_rules.json",
        help="Rules JSON (relative to repo root by default).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture = Path(args.fixture).expanduser()
    if not fixture.is_absolute():
        fixture = root / fixture
    if not fixture.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture}")

    rules_path = Path(args.rules).expanduser()
    if not rules_path.is_absolute():
        rules_path = root / rules_path
    rules_cfg = load_table_label_rules(rules_path)
    if not rules_cfg:
        raise FileNotFoundError(f"Rules missing/invalid: {rules_path}")

    combined_path = out_dir / "combined.txt"
    raw_lines = fixture.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    combined_path.write_text("".join(raw_lines), encoding="utf-8")

    labeled = label_combined_lines(raw_lines, rules_cfg)
    combined_path.write_text("".join(labeled), encoding="utf-8")

    print(f"[DONE] Wrote labeled combined.txt: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
