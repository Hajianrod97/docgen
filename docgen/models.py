"""Data models shared across the doc generator pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChunkType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    API_ENDPOINT = "api_endpoint"


@dataclass
class Chunk:
    """A single unit of code we might generate documentation for."""

    id: str  # stable id, e.g. "app/utils.py::parse_config"
    name: str
    type: ChunkType
    file_path: str
    start_line: int
    end_line: int
    source: str  # raw source text of this chunk
    signature: str = ""
    docstring: str | None = None  # existing docstring, if any
    calls: list[str] = field(default_factory=list)  # names this chunk calls
    parent: str | None = None  # e.g. class name for a method
    http_method: str | None = None  # for API_ENDPOINT chunks (GET/POST/etc)
    route: str | None = None  # for API_ENDPOINT chunks (/users/{id})

    def to_embedding_text(self) -> str:
        """Text representation used for embedding + retrieval."""
        header = f"{self.type.value} {self.name}"
        if self.parent:
            header += f" (in class {self.parent})"
        if self.route:
            header += f" [{self.http_method} {self.route}]"
        return f"{header}\nFile: {self.file_path}\n\n{self.source}"
