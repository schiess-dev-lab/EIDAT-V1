"""Loose key/value pair detection from free (non-table) text.

Vendors often print values with no table at all:

    Part Number        VLV-42A-X
    Coil Resistance    12.4 ohm

A line is a KV candidate when its tokens split into exactly two clusters
separated by a wide horizontal gap, the left cluster looks like a label,
and the right parses as a value. Vertically adjacent KV lines group into a
KvGroup — the "virtual table" that mapping treats exactly like a bordered
table (canonical tables are defined by content, not presentation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engines import Token
from .values import parse_value


@dataclass
class KvPair:
    label: str
    value: str
    value_typed: dict[str, Any]
    bbox: tuple[float, float, float, float]
    conf: float


@dataclass
class KvGroup:
    page: int
    pairs: list[KvPair] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(p.bbox[0] for p in self.pairs),
            min(p.bbox[1] for p in self.pairs),
            max(p.bbox[2] for p in self.pairs),
            max(p.bbox[3] for p in self.pairs),
        )

    @property
    def labels(self) -> list[str]:
        return [p.label for p in self.pairs]


def _split_on_gap(line: list[Token], min_gap: float) -> list[list[Token]]:
    clusters: list[list[Token]] = [[line[0]]]
    for t in line[1:]:
        if t.x0 - clusters[-1][-1].x1 > min_gap:
            clusters.append([t])
        else:
            clusters[-1].append(t)
    return clusters


def _looks_like_label(text: str) -> bool:
    words = text.split()
    if not (1 <= len(words) <= 6):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    return not text.rstrip().endswith((".", ",", ";"))


def detect_kv_pairs(
    lines: list[list[Token]], page: int, page_width: float
) -> list[KvGroup]:
    """lines: token lines (already grouped, x-sorted), outside any table."""
    min_gap = max(40.0, page_width * 0.03)
    pairs: list[tuple[float, KvPair]] = []  # (line y, pair)
    for line in lines:
        if len(line) < 2:
            continue
        clusters = _split_on_gap(line, min_gap)
        if len(clusters) != 2:
            continue
        label = " ".join(t.text for t in clusters[0]).strip().rstrip(":").strip()
        value = " ".join(t.text for t in clusters[1]).strip()
        if not label or not value or not _looks_like_label(label):
            continue
        typed = parse_value(value)
        bbox = (
            min(t.x0 for t in line),
            min(t.y0 for t in line),
            max(t.x1 for t in line),
            max(t.y1 for t in line),
        )
        conf = min(t.conf for t in line)
        pairs.append((bbox[1], KvPair(label, value, typed, bbox, conf)))

    # group vertically adjacent pairs
    pairs.sort(key=lambda x: x[0])
    groups: list[KvGroup] = []
    for y, p in pairs:
        h = p.bbox[3] - p.bbox[1]
        if groups and (p.bbox[1] - groups[-1].pairs[-1].bbox[3]) < 2.5 * max(h, 10):
            groups[-1].pairs.append(p)
        else:
            groups.append(KvGroup(page=page, pairs=[p]))
    def _worth_keeping(g: KvGroup) -> bool:
        if len(g.pairs) >= 2:
            return True
        # single pair: only keep if the value actually looks like a value
        p = g.pairs[0]
        return p.value_typed.get("kind") in {"number", "limit", "range", "verdict"} or any(
            ch.isdigit() for ch in p.value
        )

    return [g for g in groups if _worth_keeping(g)]
