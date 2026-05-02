from __future__ import annotations

from typing import List

from src.interfaces.retriever import Chunk


class DenseRetriever:
    """
    Pure semantic retriever using cosine similarity over a FAISS index.

    Suitable for queries where meaning matters more than exact keywords.
    Use HybridRetriever for production workloads that also need keyword recall.
    """

    def __init__(self, store, embedder, config) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = config

    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        qv = self._embedder.encode_single(query)
        results = self._store.search(qv, top_k)
        return [
            Chunk(
                content=r.content,
                source=r.metadata.get("source", "unknown"),
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

    def index(self, chunks: List[dict]) -> None:
        texts = [c["content"] for c in chunks]
        vectors = self._embedder.encode(texts)
        self._store.upsert(vectors, chunks)
