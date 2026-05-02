from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.core.config import LLMConfig
from src.interfaces.llm_provider import Completion, Message, Role

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """LLM provider backed by the OpenAI Chat Completions API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        env_var = config.api_key_env or "OPENAI_API_KEY"
        self._api_key = os.getenv(env_var)
        if not self._api_key:
            logger.warning(
                "OpenAI API key not found in env var '%s'. Set it before querying.", env_var
            )

    @property
    def model_name(self) -> str:
        return self._config.model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def complete(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Completion:
        try:
            from openai import OpenAI
        except ImportError:
            return Completion(
                text="The 'openai' package is not installed. Run: pip install openai",
                model=self._config.model,
                finish_reason="error",
            )

        client = OpenAI(api_key=self._api_key)

        oai_msgs: List[dict] = []
        if system_prompt:
            oai_msgs.append({"role": "system", "content": system_prompt})
        for msg in messages:
            oai_msgs.append({"role": msg.role.value, "content": msg.content})

        try:
            resp = client.chat.completions.create(
                model=self._config.model,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
            return Completion(
                text=text,
                model=self._config.model,
                finish_reason=resp.choices[0].finish_reason,
                usage=usage,
            )
        except Exception as exc:
            logger.error("OpenAI request failed: %s", exc)
            return Completion(
                text="I'm unable to generate a response right now.",
                model=self._config.model,
                finish_reason="error",
            )
