from __future__ import annotations

import logging
from typing import List

from src.core.config import LLMConfig
from src.interfaces.llm_provider import Completion, Message, Role

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Use only the provided context to answer questions. "
    "If the context does not contain enough information to answer, say so honestly. "
    "Be concise, accurate, and cite the source document when relevant."
)

_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer based solely on the context above. If the answer is not in the context, \
say "I don't have enough information to answer this question.\""""


class GenerationService:
    """
    Formats a grounded prompt and delegates completion to the LLM provider.

    Inserts retrieved context into a structured template, prepends conversation
    history, then calls the provider. The system prompt is kept here — not
    inside the provider — so it can be overridden per deployment without
    touching provider code.
    """

    def __init__(self, provider, config: LLMConfig) -> None:
        self._provider = provider
        self._cfg = config

    def run(self, messages: List[Message], context: str) -> Completion:
        """Generate a grounded completion for the last user message in *messages*."""
        if not messages:
            raise ValueError("messages must contain at least one entry.")

        last = messages[-1]
        history = messages[:-1]

        grounded_content = _PROMPT_TEMPLATE.format(
            context=context or "No relevant context available.",
            question=last.content,
        )

        full_messages = [
            *history,
            Message(role=Role.USER, content=grounded_content),
        ]

        return self._provider.complete(
            messages=full_messages,
            system_prompt=_SYSTEM_PROMPT,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
