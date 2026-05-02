from __future__ import annotations

import logging
from typing import List, Optional

from src.core.config import RetrievalConfig, RerankingConfig
from src.interfaces.retriever import Chunk

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Wraps a retriever and applies optional cross-encoder re-ranking.

    Re-ranking is kept in this service layer rather than inside the
    retriever so that any retrieval strategy can benefit from it
    without code duplication.
    """

    def __init__(
        self,
        retriever,
        retrieval_cfg: RetrievalConfig,
        rerank_cfg: RerankingConfig,
    ) -> None:
        self._retriever = retriever
        self._retrieval_cfg = retrieval_cfg
        self._rerank_cfg = rerank_cfg
        self._reranker: Optional[object] = None

        if rerank_cfg.enabled:
            self._reranker = self._load_reranker(rerank_cfg.model)

    def _load_reranker(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading re-ranker: %s", model_name)
            return CrossEncoder(model_name)
        except Exception as exc:
            logger.warning("Re-ranker unavailable (%s). Skipping re-ranking.", exc)
            return None

    def run(self, query: str) -> List[Chunk]:
        """Retrieve and optionally re-rank chunks for *query*."""
        chunks = self._retriever.retrieve(query, top_k=self._retrieval_cfg.top_k)

        if self._reranker and len(chunks) > self._retrieval_cfg.final_top_k:
            pairs = [(query, c.content) for c in chunks]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            top = ranked[: self._retrieval_cfg.final_top_k]
            result = []
            for chunk, score in top:
                chunk.score = float(score)
                result.append(chunk)
            return result

        return chunks[: self._retrieval_cfg.final_top_k]
