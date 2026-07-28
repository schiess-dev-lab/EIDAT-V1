"""Shape fingerprints for tables and KV groups + similarity matching.

A fingerprint captures *presentation shape*, so a mapping decision made
once ("this shape is the Coil Test table") replays on every future
document with the same shape — ask once, apply everywhere.
"""
from __future__ import annotations

import json
import re
from typing import Any

_norm_re = re.compile(r"[^a-z0-9 ]+")


def norm(s: str) -> str:
    return _norm_re.sub("", (s or "").lower()).strip()


def norm_tokens(strings) -> list[str]:
    out: set[str] = set()
    for s in strings:
        out.update(w for w in norm(s).split() if w)
    return sorted(out)


def table_fingerprint(table: dict[str, Any]) -> dict[str, Any]:
    """table: an entry from document.json['tables']."""
    return {
        "kind": "table",
        "n_cols": table["n_cols"],
        "header": norm_tokens(table.get("header", [])),
        "title": norm(table.get("title", "")),
    }


def kv_fingerprint(group: dict[str, Any]) -> dict[str, Any]:
    """group: an entry from document.json['kv_groups']."""
    return {
        "kind": "kv",
        "labels": norm_tokens(p["label"] for p in group["pairs"]),
    }


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def similarity(fp_a: dict, fp_b: dict) -> float:
    if fp_a.get("kind") != fp_b.get("kind"):
        return 0.0
    if fp_a["kind"] == "table":
        if fp_a["n_cols"] != fp_b["n_cols"]:
            return 0.0
        h = _jaccard(fp_a["header"], fp_b["header"])
        t = _jaccard(fp_a["title"].split(), fp_b["title"].split())
        return 0.65 * h + 0.35 * t
    return _jaccard(fp_a["labels"], fp_b["labels"])


def best_match(
    fp: dict, rules: list[dict], threshold: float = 0.72
) -> tuple[dict | None, float]:
    """rules: [{rule_id, canonical_label, fingerprint(dict)}…] -> (rule, score)."""
    best, score = None, 0.0
    for rule in rules:
        s = similarity(fp, rule["fingerprint"])
        if s > score:
            best, score = rule, s
    if score >= threshold:
        return best, score
    return None, score


def dumps(fp: dict) -> str:
    return json.dumps(fp, sort_keys=True)
