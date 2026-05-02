from __future__ import annotations

from typing import List

from src.interfaces.document_loader import Document


class TextSplitter:
    """
    Splits documents into overlapping chunks using a separator cascade.

    The algorithm tries each separator in order. If a piece produced by
    splitting on the current separator is still larger than *chunk_size*,
    it recurses with the next separator. A sliding overlap is added when
    merging pieces so context is preserved across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, documents: List[Document]) -> List[dict]:
        """Return a flat list of chunk dicts ready for embedding."""
        all_chunks: List[dict] = []
        for doc in documents:
            raw = self._split_text(doc.content)
            for i, text in enumerate(raw):
                text = text.strip()
                if len(text) > 20:
                    all_chunks.append(
                        {
                            "content": text,
                            "source": doc.source,
                            "chunk_index": i,
                            **doc.metadata,
                        }
                    )
        return all_chunks

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _split_text(self, text: str) -> List[str]:
        pieces = self._recursive_split(text, self.separators)
        return self._merge_with_overlap(pieces)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, remaining = separators[0], separators[1:]
        parts = text.split(sep) if sep else list(text)

        result: List[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, remaining))
        return result

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        if not pieces:
            return []

        chunks: List[str] = []
        current = ""

        for piece in pieces:
            candidate = (current + " " + piece).strip() if current else piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    tail = current[max(0, len(current) - self.chunk_overlap) :]
                    current = (tail + " " + piece).strip()
                else:
                    current = piece

        if current:
            chunks.append(current)

        return chunks
