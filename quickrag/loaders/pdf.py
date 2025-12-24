"""PDF file loader for QuickRAG."""

from pathlib import Path

from quickrag.loaders.base import BaseLoader, LoadedDocument


class PDFLoader(BaseLoader):
    """Loader for PDF files using pypdf."""

    EXTENSIONS = {".pdf"}

    def supports(self, source: str | Path) -> bool:
        """Check if source is a PDF file."""
        path = Path(source)
        return path.suffix.lower() in self.EXTENSIONS

    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load a PDF file.

        Args:
            source: Path to the PDF file.

        Returns:
            List of LoadedDocuments (one per page or combined).
        """
        from pypdf import PdfReader

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        reader = PdfReader(path)
        documents = []

        # Extract text from each page
        all_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                all_text.append(text)

        # Combine into single document
        if all_text:
            combined_text = "\n\n".join(all_text)
            documents.append(
                LoadedDocument(
                    content=combined_text,
                    source=str(path.absolute()),
                    metadata={
                        "filename": path.name,
                        "extension": path.suffix,
                        "num_pages": len(reader.pages),
                        "size_bytes": path.stat().st_size,
                    },
                )
            )

        return documents

