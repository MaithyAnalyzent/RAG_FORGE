from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from src.core.config import AppConfig
from src.interfaces.llm_provider import Message, Role
from src.services.ingestion_service import IngestionService
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.session_service import SessionService

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """The result of a single pipeline query."""

    answer: str
    sources: List[str] = field(default_factory=list)
    session_id: str = ""
    context_chunks: int = 0


class RAGPipeline:
    """
    Orchestrates the full RAG workflow: ingest → retrieve → generate.

    Designed around dependency injection — every service is supplied
    via the constructor, making each component independently testable
    and fully swappable without touching this class.

    Quick start::

        config = AppConfig.from_yaml("configs/default.yaml")
        pipeline = RAGPipeline.from_config(config)
        pipeline.ingest(["docs/manual.pdf"])
        response = pipeline.query("How do I reset my password?")
        print(response.answer)
    """

    def __init__(
        self,
        ingestion: IngestionService,
        retrieval: RetrievalService,
        generation: GenerationService,
        sessions: SessionService,
    ) -> None:
        self._ingest = ingestion
        self._retrieve = retrieval
        self._generate = generation
        self._sessions = sessions

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: AppConfig) -> "RAGPipeline":
        """Assemble the pipeline from a configuration object."""
        from src.core.registry import get_registry
        from src.utils.embedder import Embedder

        registry = get_registry()
        embedder = Embedder(config.embedding)

        store_cls = registry.get_store("faiss")
        store = store_cls(embedder, config.storage.index_dir)

        retriever_cls = registry.get_retriever(config.retrieval.strategy)
        retriever = retriever_cls(store, embedder, config.retrieval)

        llm_cls = registry.get_llm(config.llm.provider)
        llm = llm_cls(config.llm)

        return cls(
            ingestion=IngestionService(registry, embedder, store, config),
            retrieval=RetrievalService(retriever, config.retrieval, config.reranking),
            generation=GenerationService(llm, config.llm),
            sessions=SessionService(config.storage.db_path),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, paths: List[Union[str, Path]]) -> int:
        """
        Ingest documents into the knowledge base.

        Returns the total number of chunks indexed.
        """
        resolved = [Path(p) for p in paths]
        count = self._ingest.run(resolved)
        logger.info("Indexed %d chunks from %d file(s).", count, len(resolved))
        return count

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> QueryResponse:
        """
        Answer *question* using context retrieved from the knowledge base.

        Conversation history is maintained per session. A new session is
        created automatically when *session_id* is None.
        """
        if session_id is None:
            session_id = self._sessions.create_session()

        history = self._sessions.get_history(session_id, max_turns=3)
        chunks = self._retrieve.run(question)
        context = "\n\n".join(c.to_context_string() for c in chunks)

        messages = [
            *[Message(role=Role(m["role"]), content=m["content"]) for m in history],
            Message(role=Role.USER, content=question),
        ]

        completion = self._generate.run(messages, context)
        sources = sorted({c.source for c in chunks})

        self._sessions.record(session_id, question, completion.text, sources)

        return QueryResponse(
            answer=completion.text,
            sources=sources,
            session_id=session_id,
            context_chunks=len(chunks),
        )

    def new_session(self, name: Optional[str] = None) -> str:
        """Create and return a fresh session ID."""
        return self._sessions.create_session(name=name)

    def list_sessions(self) -> List[dict]:
        """Return metadata for all stored sessions."""
        return self._sessions.list_all()
