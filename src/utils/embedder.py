from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps a sentence-transformers model for batch text encoding.

    The model is loaded lazily on first use to avoid slow startup when
    the index already exists and no new documents need to be embedded.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._config.model)
            self._model = SentenceTransformer(self._config.model)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts. Returns a float32 array of shape (N, D)."""
        vectors = self.model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(vectors, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single string. Returns a 1-D float32 array."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
