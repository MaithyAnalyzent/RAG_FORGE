# ragforge

**A production-ready, provider-agnostic Retrieval-Augmented Generation (RAG) framework.**

ragforge lets you point at any collection of documents, build a searchable knowledge base, and query it with natural language — with no domain-specific logic baked in. Swap the LLM, the vector store, or the retrieval strategy entirely from the config file.

---

## Problem Statement

Many teams need a system that can answer questions grounded in internal documents (policies, manuals, FAQs, contracts) without hallucination. The challenge is doing this in a way that is:

- **Accurate** — answers must come from the source documents, not model priors.
- **Maintainable** — adding a new document type, vector store, or LLM should not require touching core logic.
- **Portable** — the same codebase must work for a legal team, a support desk, or a developer portal with zero code changes.

ragforge solves this by separating every concern into its own layer and connecting them through narrow, swappable interfaces.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Python API                      │
│                          main.py                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │       RAGPipeline         │  core/pipeline.py
              │  (dependency injection)   │
              └──┬──────────┬──────────┬──┘
                 │          │          │
    ┌────────────▼──┐ ┌─────▼──────┐ ┌▼───────────────┐
    │  Ingestion    │ │ Retrieval  │ │  Generation    │
    │  Service      │ │ Service    │ │  Service       │
    └────────────┬──┘ └─────┬──────┘ └┬───────────────┘
                 │          │          │
    ┌────────────▼──────────▼──────────▼───────────────┐
    │                   Providers                       │
    │  Loaders: DocxLoader | PdfLoader | TxtLoader      │
    │  Stores:  FaissVectorStore                        │
    │  Retrievers: HybridRetriever (dense + BM25 + RRF) │
    │  LLMs: OllamaProvider | OpenAIProvider            │
    └───────────────────────────────────────────────────┘
                 │
    ┌────────────▼───────────────────────────────────────┐
    │              Session Service (SQLite)               │
    └────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| **Component interfaces** | Python `Protocol` (structural typing) | No inheritance required — any class that quacks is accepted |
| **Configuration** | Pydantic v2 + YAML | Type-safe, IDE-friendly, zero-friction overrides |
| **Retrieval strategy** | Hybrid: dense + BM25 fused via RRF | Neither signal dominates — keywords and semantics both win |
| **Re-ranking** | Cross-encoder (separate service layer) | Any retriever benefits without code duplication |
| **Session persistence** | SQLite with WAL mode + context managers | Lightweight, zero-ops, safe under concurrent reads |
| **LLM coupling** | Provider protocol + factory registry | Switch from Ollama to OpenAI in one config line |

---

## How It Differs from a Naive RAG

| Aspect | Traditional approach | ragforge approach |
|---|---|---|
| Retrieval | Single embedding search | Hybrid dense + BM25, fused with Reciprocal Rank Fusion |
| Component coupling | Hard-coded class imports | Provider registry + dependency injection |
| Configuration | Scattered magic numbers | Single validated YAML file |
| LLM integration | One provider baked in | Factory pattern; add a provider in ~30 lines |
| Session storage | Ad-hoc or none | Repository-pattern SQLite with WAL and FK constraints |
| Extensibility | Fork the code | Register a new provider, change one config value |

---

## Project Structure

```
ragforge/
├── src/
│   ├── core/
│   │   ├── config.py          # Pydantic config models
│   │   ├── pipeline.py        # RAGPipeline orchestrator
│   │   └── registry.py        # Component registry & factory
│   ├── interfaces/
│   │   ├── document_loader.py # DocumentLoader Protocol + Document dataclass
│   │   ├── vector_store.py    # VectorStore Protocol + SearchResult
│   │   ├── retriever.py       # Retriever Protocol + Chunk dataclass
│   │   └── llm_provider.py    # LLMProvider Protocol + Message/Completion
│   ├── services/
│   │   ├── ingestion_service.py   # load → split → embed → store
│   │   ├── retrieval_service.py   # retrieve → re-rank
│   │   ├── generation_service.py  # prompt formatting + LLM call
│   │   └── session_service.py     # SQLite conversation persistence
│   ├── providers/
│   │   ├── loaders/   DocxLoader, PdfLoader, TxtLoader
│   │   ├── stores/    FaissVectorStore
│   │   ├── retrievers/ DenseRetriever, HybridRetriever
│   │   └── llms/      OllamaProvider, OpenAIProvider
│   └── utils/
│       ├── embedder.py        # Lazy-loading SentenceTransformer wrapper
│       ├── text_splitter.py   # Recursive separator-cascade chunker
│       └── logger.py          # Logging configuration
├── configs/
│   └── default.yaml           # All tunable parameters
├── data/                      # Created at runtime
│   ├── index/                 # FAISS index + metadata pickle
│   └── sessions.db            # SQLite conversation store
├── main.py                    # CLI entry point
├── .env.example
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally (or an OpenAI API key)

### Install

```bash
# Clone or copy the project
cd ragforge

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY if using OpenAI
```

To change the LLM provider, edit `configs/default.yaml`:

```yaml
llm:
  provider: "openai"        # was "ollama"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
```

---

## Usage

### 1. Ingest documents

```bash
python main.py ingest path/to/manual.pdf path/to/faq.docx path/to/notes.txt
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### 2. Ask a question

```bash
python main.py query "What are the eligibility requirements?"
```

Output:

```
Answer: Based on the documentation, eligibility requires...
Sources: manual.pdf#p3, faq.docx
Session: 3f2a1b...  |  Chunks used: 5
```

### 3. Interactive chat

```bash
python main.py chat
```

Resume a previous session:

```bash
python main.py chat --session 3f2a1b8c-...
```

### 4. List sessions

```bash
python main.py sessions
```

### Use as a Python library

```python
from src.core.config import AppConfig
from src.core.pipeline import RAGPipeline

config = AppConfig.from_yaml("configs/default.yaml")
pipeline = RAGPipeline.from_config(config)

pipeline.ingest(["data/my_doc.pdf"])

response = pipeline.query("What does section 3 cover?")
print(response.answer)
print(response.sources)
```

---

## Extending ragforge

### Add a new document loader

```python
# src/providers/loaders/html_loader.py
from pathlib import Path
from typing import List
from src.interfaces.document_loader import Document

class HtmlLoader:
    supported_extensions = [".html", ".htm"]

    def load(self, path: Path) -> List[Document]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(), "html.parser")
        return [Document(content=soup.get_text(), source=path.name)]
```

Then register it:

```python
from src.core.registry import get_registry
from src.providers.loaders.html_loader import HtmlLoader

get_registry().register_loader(".html", HtmlLoader)
```

### Add a new LLM provider

Implement the `LLMProvider` protocol (no inheritance needed):

```python
class AnthropicProvider:
    @property
    def model_name(self) -> str: ...
    def is_available(self) -> bool: ...
    def complete(self, messages, system_prompt=None, temperature=0.1, max_tokens=512): ...
```

Register and select it in `default.yaml`:

```yaml
llm:
  provider: "anthropic"
  model: "claude-3-5-haiku-20241022"
```

### Add a new vector store

Implement the `VectorStore` protocol (FAISS, ChromaDB, Pinecone, etc.) and register it:

```python
get_registry().register_store("chroma", ChromaVectorStore)
```

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `embedding.model` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `chunking.chunk_size` | `512` | Max characters per chunk |
| `chunking.chunk_overlap` | `64` | Overlap between adjacent chunks |
| `retrieval.strategy` | `hybrid` | `dense`, `bm25`, or `hybrid` |
| `retrieval.top_k` | `10` | Candidates retrieved before re-ranking |
| `retrieval.final_top_k` | `5` | Chunks kept after re-ranking |
| `retrieval.rrf_k` | `60` | RRF fusion constant |
| `reranking.enabled` | `true` | Enable cross-encoder re-ranking |
| `reranking.model` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `llm.provider` | `ollama` | LLM backend |
| `llm.model` | `llama2` | Model name passed to the provider |
| `llm.temperature` | `0.1` | Sampling temperature |
| `llm.max_tokens` | `512` | Max tokens in the response |
| `storage.index_dir` | `data/index` | FAISS index directory |
| `storage.db_path` | `data/sessions.db` | SQLite session database |

---

## License

MIT — free to use, modify, and distribute.
