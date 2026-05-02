from __future__ import annotations

from pathlib import Path
from typing import List

from src.interfaces.document_loader import Document


class TxtLoader:
    """Loads plain-text and Markdown files as a single Document."""

    supported_extensions = [".txt", ".md"]

    def load(self, path: Path) -> List[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                content=content,
                source=path.name,
                metadata={"path": str(path), "format": path.suffix.lstrip(".")},
            )
        ]
