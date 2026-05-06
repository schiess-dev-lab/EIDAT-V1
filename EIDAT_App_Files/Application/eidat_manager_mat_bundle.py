from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_SERIAL_TOKEN_RE = re.compile(r"^sn[0-9a-z]+$", re.IGNORECASE)
_SEQ_INLINE_DIGITS_RE = re.compile(r"^(seq|sequence)(\d+)$", re.IGNORECASE)
_SEQ_INLINE_WORD_RE = re.compile(r"^(seq|sequence)([a-z]+)$", re.IGNORECASE)
_SEQ_ORDINAL_DIGITS_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$", re.IGNORECASE)

_SEQ_MARKERS = {"seq", "sequence"}
_SEQ_CARDINAL_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_SEQ_ORDINAL_WORDS = {
    "zeroth": 0,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
}
_SEQ_TENS_WORDS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SEQ_TENS_ORDINAL_WORDS = {
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
}


@dataclass(frozen=True)
class MatBundleMember:
    file_path: Path
    serial_number: str
    sequence_name: str
    sequence_number: int
    group_key: str
    folder_hash: str

    @property
    def bundle_stem(self) -> str:
        return f"{self.serial_number}__matseq_{self.folder_hash}"


def _repo_rel_parent_key(repo_root: Path | None, file_path: Path) -> str:
    parent = Path(file_path).expanduser().parent
    if repo_root is not None:
        try:
            return parent.resolve().relative_to(Path(repo_root).expanduser().resolve()).as_posix().casefold()
        except Exception:
            pass
    try:
        return str(parent.resolve()).replace("\\", "/").casefold()
    except Exception:
        return str(parent).replace("\\", "/").casefold()


def _parse_sequence_word_tokens(tokens: list[str]) -> int | None:
    parts = [str(token or "").strip().casefold() for token in tokens if str(token or "").strip()]
    if not parts:
        return None
    if len(parts) == 1:
        token = parts[0]
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                return None
        match = _SEQ_ORDINAL_DIGITS_RE.match(token)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        if token in _SEQ_CARDINAL_WORDS:
            return int(_SEQ_CARDINAL_WORDS[token])
        if token in _SEQ_ORDINAL_WORDS:
            return int(_SEQ_ORDINAL_WORDS[token])
        if token in _SEQ_TENS_WORDS:
            return int(_SEQ_TENS_WORDS[token])
        if token in _SEQ_TENS_ORDINAL_WORDS:
            return int(_SEQ_TENS_ORDINAL_WORDS[token])
        return None
    if len(parts) == 2:
        tens = _SEQ_TENS_WORDS.get(parts[0])
        if tens is None:
            return None
        ones = _SEQ_CARDINAL_WORDS.get(parts[1])
        if ones is not None and 0 < ones < 10:
            return int(tens + ones)
        ordinal_ones = _SEQ_ORDINAL_WORDS.get(parts[1])
        if ordinal_ones is not None and 0 < ordinal_ones < 10:
            return int(tens + ordinal_ones)
    return None


def _parse_sequence_candidate(tokens: list[str], *, token_idx: int) -> tuple[str, int] | None:
    if token_idx < 0 or token_idx >= len(tokens):
        return None
    token = str(tokens[token_idx] or "").strip().casefold()
    if not token:
        return None
    inline_digits = _SEQ_INLINE_DIGITS_RE.match(token)
    if inline_digits:
        try:
            number = int(inline_digits.group(2))
        except Exception:
            number = -1
        if number >= 0:
            return (f"seq{number}", int(number))
    inline_word = _SEQ_INLINE_WORD_RE.match(token)
    if inline_word:
        number = _parse_sequence_word_tokens([str(inline_word.group(2) or "")])
        if number is not None and number >= 0:
            return (f"seq{number}", int(number))
    if token not in _SEQ_MARKERS:
        return None
    max_end = min(len(tokens), token_idx + 3)
    for end_idx in range(max_end, token_idx + 1, -1):
        number = _parse_sequence_word_tokens(tokens[token_idx + 1 : end_idx])
        if number is not None and number >= 0:
            return (f"seq{number}", int(number))
    return None


def _parse_sequence_after_serial(tokens: list[str], *, serial_idx: int) -> tuple[str, int] | None:
    if not tokens or serial_idx < 0:
        return None
    for token_idx in range(int(serial_idx) + 1, len(tokens)):
        sequence = _parse_sequence_candidate(tokens, token_idx=token_idx)
        if sequence is not None:
            return sequence
    return None


def detect_mat_bundle_member(file_path: Path, *, repo_root: Path | None = None) -> MatBundleMember | None:
    path = Path(file_path).expanduser()
    if path.suffix.lower() != ".mat":
        return None
    stem = str(path.stem or "").strip()
    if not stem:
        return None
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", stem) if t]
    if not tokens:
        return None

    serial = ""
    serial_idx = -1
    for idx, token in enumerate(tokens):
        if not serial and _SERIAL_TOKEN_RE.match(token):
            serial = token.upper()
            serial_idx = idx
    sequence = _parse_sequence_after_serial(tokens, serial_idx=serial_idx)
    seq_name = str(sequence[0] or "") if sequence is not None else ""
    seq_number = int(sequence[1]) if sequence is not None else -1
    if not serial or not seq_name or seq_number < 0:
        return None

    parent_key = _repo_rel_parent_key(repo_root, path)
    group_key = f"{parent_key}|{serial.casefold()}"
    folder_hash = hashlib.sha1(group_key.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return MatBundleMember(
        file_path=path,
        serial_number=serial,
        sequence_name=seq_name,
        sequence_number=seq_number,
        group_key=group_key,
        folder_hash=folder_hash,
    )


def list_mat_bundle_members(file_path: Path, *, repo_root: Path | None = None) -> list[MatBundleMember]:
    seed = detect_mat_bundle_member(file_path, repo_root=repo_root)
    if seed is None:
        return []
    members: list[MatBundleMember] = []
    try:
        siblings = list(seed.file_path.parent.iterdir())
    except Exception:
        siblings = []
    for sibling in siblings:
        try:
            if not sibling.is_file() or sibling.suffix.lower() != ".mat":
                continue
        except Exception:
            continue
        info = detect_mat_bundle_member(sibling, repo_root=repo_root)
        if info is None or info.group_key != seed.group_key:
            continue
        members.append(info)
    members.sort(key=lambda item: (int(item.sequence_number), str(item.file_path.name).lower()))
    return members


def mat_bundle_artifacts_dir(support_dir: Path, member: MatBundleMember) -> Path:
    return Path(support_dir) / "debug" / "ocr" / f"{member.bundle_stem}__excel"


def mat_bundle_sqlite_path(support_dir: Path, member: MatBundleMember) -> Path:
    return mat_bundle_artifacts_dir(support_dir, member) / f"{member.bundle_stem}.sqlite3"
