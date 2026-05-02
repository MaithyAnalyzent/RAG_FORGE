from __future__ import annotations

from typing import Any, Dict, Optional


class ComponentRegistry:
    """Central registry for pluggable pipeline components.

    Register implementations by string alias so the pipeline can be
    configured entirely from a YAML file without changing source code.

    Example::

        registry = get_registry()
        registry.register_llm("my_provider", MyLLMProvider)
        # Then set llm.provider: "my_provider" in default.yaml
    """

    def __init__(self) -> None:
        self._loaders: Dict[str, Any] = {}
        self._stores: Dict[str, Any] = {}
        self._retrievers: Dict[str, Any] = {}
        self._llms: Dict[str, Any] = {}

    # ── Registration ────────────────────────────────────────────────────────

    def register_loader(self, extension: str, cls: Any) -> None:
        self._loaders[extension.lower()] = cls

    def register_store(self, name: str, cls: Any) -> None:
        self._stores[name] = cls

    def register_retriever(self, name: str, cls: Any) -> None:
        self._retrievers[name] = cls

    def register_llm(self, name: str, cls: Any) -> None:
        self._llms[name] = cls

    # ── Retrieval ────────────────────────────────────────────────────────────

    def get_loader(self, extension: str) -> Any:
        key = extension.lower()
        if key not in self._loaders:
            raise KeyError(
                f"No loader for '{key}'. Registered: {sorted(self._loaders)}"
            )
        return self._loaders[key]

    def get_store(self, name: str) -> Any:
        if name not in self._stores:
            raise KeyError(
                f"No store '{name}'. Registered: {sorted(self._stores)}"
            )
        return self._stores[name]

    def get_retriever(self, name: str) -> Any:
        if name not in self._retrievers:
            raise KeyError(
                f"No retriever '{name}'. Registered: {sorted(self._retrievers)}"
            )
        return self._retrievers[name]

    def get_llm(self, name: str) -> Any:
        if name not in self._llms:
            raise KeyError(
                f"No LLM provider '{name}'. Registered: {sorted(self._llms)}"
            )
        return self._llms[name]


_registry: Optional[ComponentRegistry] = None


def get_registry() -> ComponentRegistry:
    """Return the singleton registry, populating it with built-in providers."""
    global _registry
    if _registry is None:
        _registry = ComponentRegistry()
        _register_builtins(_registry)
    return _registry


def _register_builtins(registry: ComponentRegistry) -> None:
    from src.providers.loaders.docx_loader import DocxLoader
    from src.providers.loaders.pdf_loader import PdfLoader
    from src.providers.loaders.txt_loader import TxtLoader
    from src.providers.stores.faiss_store import FaissVectorStore
    from src.providers.retrievers.hybrid_retriever import HybridRetriever
    from src.providers.llms.ollama_provider import OllamaProvider
    from src.providers.llms.openai_provider import OpenAIProvider

    registry.register_loader(".docx", DocxLoader)
    registry.register_loader(".pdf", PdfLoader)
    registry.register_loader(".txt", TxtLoader)
    registry.register_loader(".md", TxtLoader)

    registry.register_store("faiss", FaissVectorStore)

    registry.register_retriever("hybrid", HybridRetriever)
    registry.register_retriever("dense", HybridRetriever)

    registry.register_llm("ollama", OllamaProvider)
    registry.register_llm("openai", OpenAIProvider)
