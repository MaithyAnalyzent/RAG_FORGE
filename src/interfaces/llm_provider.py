from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single turn in a conversation thread."""

    role: Role
    content: str


@dataclass
class Completion:
    """The output of a language model call."""

    text: str
    model: str
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface for language model providers.

    Concrete implementations may target Ollama, OpenAI, Anthropic, etc.
    Only ``complete`` and ``is_available`` are required.
    """

    @property
    def model_name(self) -> str:
        """Human-readable identifier for the underlying model."""
        ...

    def is_available(self) -> bool:
        """Return True if the provider endpoint is reachable."""
        ...

    def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Completion:
        """Generate a completion for the given message thread."""
        ...
