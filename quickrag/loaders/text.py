"""Text file loader for QuickRAG."""

from pathlib import Path

from quickrag.loaders.base import BaseLoader, LoadedDocument


class TextLoader(BaseLoader):
    """Loader for plain text and markdown files."""

    EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}

    def supports(self, source: str | Path) -> bool:
        """Check if source is a supported text file."""
        path = Path(source)
        return path.suffix.lower() in self.EXTENSIONS

    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load a text file.

        Args:
            source: Path to the text file.

        Returns:
            List with single LoadedDocument.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")

        return [
            LoadedDocument(
                content=content,
                source=str(path.absolute()),
                metadata={
                    "filename": path.name,
                    "extension": path.suffix,
                    "size_bytes": path.stat().st_size,
                },
            )
        ]

