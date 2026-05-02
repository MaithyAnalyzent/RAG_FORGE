from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np

from src.interfaces.vector_store import SearchResult

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """
    FAISS inner-product index with pickle-backed metadata.

    Vectors are L2-normalised before insertion so that inner-product
    search is equivalent to cosine similarity. This allows exact nearest-
    neighbour search without the overhead of an approximate index.
    """

    def __init__(self, embedder, index_dir: str) -> None:
        self._embedder = embedder
        self.index_dir = Path(index_dir)
        self._index = None
        self._records: List[dict] = []

        if self._index_files_exist():
            self._load()

    # ── VectorStore protocol ──────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return int(self._index.ntotal) if self._index else 0

    def upsert(self, vectors: np.ndarray, records: List[dict]) -> None:
        import faiss

        if len(vectors) == 0:
            return

        vecs = vectors.astype(np.float32)
        faiss.normalize_L2(vecs)

        if self._index is None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])

        self._index.add(vecs)
        self._records.extend(records)
        logger.debug("Upserted %d vectors. Total: %d.", len(vecs), self.count)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        import faiss

        if self._index is None or self._index.ntotal == 0:
            return []

        qv = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(qv, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._records):
                rec = self._records[idx]
                results.append(
                    SearchResult(
                        content=rec.get("content", ""),
                        metadata=rec,
                        score=float(score),
                    )
                )
        return results

    def persist(self, directory: str) -> None:
        import faiss

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "records.pkl", "wb") as fh:
            pickle.dump(self._records, fh)

        logger.info("Persisted %d vectors to %s.", self.count, directory)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _index_files_exist(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (
            self.index_dir / "records.pkl"
        ).exists()

    def _load(self) -> None:
        import faiss

        self._index = faiss.read_index(str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "records.pkl", "rb") as fh:
            self._records = pickle.load(fh)
        logger.info("Loaded %d vectors from %s.", self.count, self.index_dir)
