from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


@dataclass
class Chunk:
    """A retrieved document fragment with relevance score and provenance."""

    content: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Format the chunk for inclusion in an LLM prompt."""
        return f"[Source: {self.source}]\n{self.content}"


@runtime_checkable
class Retriever(Protocol):
    """Structural interface for retrieval strategies.

    Implementations can use dense, sparse, or hybrid approaches
    and must support lazy indexing via ``index``.
    """

    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """Return the *top_k* most relevant chunks for *query*."""
        ...

    def index(self, chunks: List[dict]) -> None:
        """Build or update internal retrieval structures from *chunks*."""
        ...
