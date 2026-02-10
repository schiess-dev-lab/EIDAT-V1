from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodeLayout:
    node_root: Path

    @property
    def eidat_root(self) -> Path:
        return self.node_root / "EIDAT"

    @property
    def runtime_dir(self) -> Path:
        return self.eidat_root / "Runtime"

    @property
    def venv_dir(self) -> Path:
        return self.runtime_dir / ".venv"

    @property
    def user_data_root(self) -> Path:
        return self.eidat_root / "UserData"

    @property
    def extraction_node_dir(self) -> Path:
        return self.eidat_root / "ExtractionNode"

    @property
    def support_dir(self) -> Path:
        return self.node_root / "EIDAT Support"

    @property
    def support_projects_dir(self) -> Path:
        return self.support_dir / "projects"


def node_layout(node_root: str | Path) -> NodeLayout:
    return NodeLayout(node_root=Path(node_root).expanduser())


def venv_python(venv_dir: Path) -> Path:
    if __import__("os").name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"
