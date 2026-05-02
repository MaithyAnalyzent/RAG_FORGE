from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.core.config import RetrievalConfig
from src.interfaces.retriever import Chunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, dict]]],
    k: int = 60,
) -> List[Tuple[str, dict, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for item i = Σ 1 / (rank_i + k) across all lists.
    A higher k reduces the influence of top ranks; 60 is the standard default.

    Returns items sorted by descending RRF score.
    """
    scores: Dict[str, float] = defaultdict(float)
    items: Dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, (item_id, meta) in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (rank + k)
            items[item_id] = meta

    return [
        (item_id, items[item_id], score)
        for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


class HybridRetriever:
    """
    Combines dense (embedding) and sparse (BM25) retrieval via RRF.

    Dense retrieval captures semantic similarity; BM25 handles exact
    keyword matches. Fusing them with RRF improves recall compared to
    either signal alone, without requiring score normalisation.

    The BM25 index is built lazily from store records on the first
    ``retrieve`` call, so it reflects any vectors added before retrieval.
    """

    def __init__(self, store, embedder, config: RetrievalConfig) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = config
        self._bm25: Optional[object] = None
        self._bm25_corpus: List[dict] = []

    # ── Retriever protocol ────────────────────────────────────────────────────

    def index(self, chunks: List[dict]) -> None:
        texts = [c["content"] for c in chunks]
        vectors = self._embedder.encode(texts)
        self._store.upsert(vectors, chunks)
        self._bm25 = None  # force rebuild on next retrieve

    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        self._ensure_bm25()

        dense = self._dense_ranked(query, top_k)

        if self._bm25 is None:
            return self._chunks_from_ranked(dense, top_k)

        sparse = self._bm25_ranked(query, top_k)
        fused = reciprocal_rank_fusion([dense, sparse], k=self._cfg.rrf_k)

        return [
            Chunk(
                content=meta.get("content", ""),
                source=meta.get("source", "unknown"),
                score=rrf_score,
                metadata=meta,
            )
            for _, meta, rrf_score in fused[:top_k]
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None or self._store.count == 0:
            return
        try:
            from rank_bm25 import BM25Okapi

            corpus = self._store._records
            tokenized = [r.get("content", "").lower().split() for r in corpus]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_corpus = corpus
            logger.debug("Built BM25 index over %d records.", len(corpus))
        except ImportError:
            logger.warning("rank-bm25 not installed — using dense-only retrieval.")

    def _dense_ranked(self, query: str, top_k: int) -> List[Tuple[str, dict]]:
        qv = self._embedder.encode_single(query)
        results = self._store.search(qv, top_k)
        return [(f"d{i}", r.metadata) for i, r in enumerate(results)]

    def _bm25_ranked(self, query: str, top_k: int) -> List[Tuple[str, dict]]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(f"b{idx}", self._bm25_corpus[idx]) for idx in indices]

    def _chunks_from_ranked(
        self, ranked: List[Tuple[str, dict]], top_k: int
    ) -> List[Chunk]:
        return [
            Chunk(
                content=meta.get("content", ""),
                source=meta.get("source", "unknown"),
                score=1.0 / (i + 1),
                metadata=meta,
            )
            for i, (_, meta) in enumerate(ranked[:top_k])
        ]
