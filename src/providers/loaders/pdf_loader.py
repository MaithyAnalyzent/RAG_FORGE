from __future__ import annotations

from pathlib import Path
from typing import List

from src.interfaces.document_loader import Document


class PdfLoader:
    """Extracts text from PDF files, one Document per page."""

    supported_extensions = [".pdf"]

    def load(self, path: Path) -> List[Document]:
        import PyPDF2

        documents: List[Document] = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            content=text,
                            source=f"{path.name}#p{page_num}",
                            metadata={
                                "path": str(path),
                                "format": "pdf",
                                "page": page_num,
                            },
                        )
                    )
        return documents
