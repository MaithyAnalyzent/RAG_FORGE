from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]


class RetrievalConfig(BaseModel):
    strategy: Literal["dense", "bm25", "hybrid"] = "hybrid"
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)


class RerankingConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    model: str = "llama2"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    endpoint: Optional[str] = "http://localhost:11434"
    api_key_env: Optional[str] = None


class StorageConfig(BaseModel):
    index_dir: str = "data/index"
    db_path: str = "data/sessions.db"


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load from the path in RAG_CONFIG_PATH, or use defaults."""
        config_path = os.getenv("RAG_CONFIG_PATH", "configs/default.yaml")
        if Path(config_path).exists():
            return cls.from_yaml(config_path)
        return cls()
