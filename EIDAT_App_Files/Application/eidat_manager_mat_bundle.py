from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_SERIAL_TOKEN_RE = re.compile(r"^sn[0-9a-z]+$", re.IGNORECASE)
_SEQ_TOKEN_RE = re.compile(r"^seq(\d+)$", re.IGNORECASE)


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
    seq_name = ""
    seq_number = -1
    for token in tokens:
        if not serial and _SERIAL_TOKEN_RE.match(token):
            serial = token.upper()
    last = tokens[-1]
    m_seq = _SEQ_TOKEN_RE.match(last)
    if m_seq:
        seq_name = last.lower()
        try:
            seq_number = int(m_seq.group(1))
        except Exception:
            seq_number = -1
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

