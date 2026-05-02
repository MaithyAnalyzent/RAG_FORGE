from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.core.config import AppConfig
from src.utils.text_splitter import TextSplitter

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Handles the full document → chunk → embed → store pipeline.

    Uses the component registry to dispatch loading by file extension,
    making it trivial to add support for new formats without modifying
    this class.
    """

    def __init__(self, registry, embedder, store, config: AppConfig) -> None:
        self._registry = registry
        self._embedder = embedder
        self._store = store
        self._splitter = TextSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=config.chunking.separators,
        )

    def run(self, paths: List[Path]) -> int:
        """
        Ingest all documents at *paths* into the vector store.

        Returns the total number of chunks added.
        """
        documents = []
        for path in paths:
            ext = path.suffix.lower()
            try:
                loader_cls = self._registry.get_loader(ext)
            except KeyError:
                logger.warning("No loader for '%s' — skipping %s.", ext, path.name)
                continue

            try:
                docs = loader_cls().load(path)
                documents.extend(docs)
                logger.info("Loaded %d doc(s) from %s.", len(docs), path.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", path.name, exc)

        if not documents:
            logger.warning("No documents were loaded.")
            return 0

        chunks = self._splitter.split(documents)
        logger.info("Split into %d chunks.", len(chunks))

        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        vectors = self._embedder.encode(texts)
        self._store.upsert(vectors, chunks)
        self._store.persist(str(self._store.index_dir))

        return len(chunks)
