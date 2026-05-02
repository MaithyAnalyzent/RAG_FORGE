from __future__ import annotations

import logging
from typing import List, Optional

import requests

from src.core.config import LLMConfig
from src.interfaces.llm_provider import Completion, Message, Role

logger = logging.getLogger(__name__)


class OllamaProvider:
    """LLM provider backed by a locally running Ollama server."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._generate_url = f"{config.endpoint}/api/generate"

    @property
    def model_name(self) -> str:
        return self._config.model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self._config.endpoint}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Completion:
        prompt = self._to_prompt(messages, system_prompt)
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(self._generate_url, json=payload, timeout=120)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            return Completion(text=text, model=self._config.model)
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return Completion(
                text="I'm unable to generate a response right now.",
                model=self._config.model,
                finish_reason="error",
            )

    @staticmethod
    def _to_prompt(messages: List[Message], system_prompt: Optional[str]) -> str:
        lines: List[str] = []
        if system_prompt:
            lines.append(f"System: {system_prompt}")
        for msg in messages:
            label = "Human" if msg.role == Role.USER else "Assistant"
            lines.append(f"{label}: {msg.content}")
        lines.append("Assistant:")
        return "\n\n".join(lines)
