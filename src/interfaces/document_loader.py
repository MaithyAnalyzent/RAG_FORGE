from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, runtime_checkable


@dataclass
class Document:
    """A piece of loaded content with provenance metadata."""

    content: str
    source: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Document(source={self.source!r}, length={len(self.content)})"


@runtime_checkable
class DocumentLoader(Protocol):
    """Structural interface for document loaders.

    Any class that implements ``load`` and exposes ``supported_extensions``
    satisfies this protocol — no inheritance required.
    """

    supported_extensions: List[str]

    def load(self, path: Path) -> List[Document]:
        """Load and return one or more Documents from *path*."""
        ...
