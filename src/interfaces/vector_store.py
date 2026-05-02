from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np


@dataclass
class SearchResult:
    """A single hit returned by a vector store search."""

    content: str
    metadata: dict
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Structural interface for vector storage backends.

    Implementations may use FAISS, ChromaDB, Pinecone, etc.
    Only ``upsert``, ``search``, ``persist``, and ``count`` are required.
    """

    @property
    def count(self) -> int:
        """Number of vectors currently stored."""
        ...

    def upsert(self, vectors: np.ndarray, records: List[dict]) -> None:
        """Add vectors with associated metadata records."""
        ...

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        """Return the *top_k* nearest neighbours for *query_vector*."""
        ...

    def persist(self, directory: str) -> None:
        """Write the index and metadata to disk."""
        ...
