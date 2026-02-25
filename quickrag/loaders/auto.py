"""Auto-detecting document loader for QuickRAG."""

from pathlib import Path
from typing import Union

from quickrag.loaders.base import BaseLoader, LoadedDocument
from quickrag.loaders.text import TextLoader
from quickrag.loaders.pdf import PDFLoader
from quickrag.loaders.web import WebLoader
from quickrag.loaders.csv_loader import CSVLoader
from quickrag.loaders.json_loader import JSONLoader
from quickrag.loaders.docx_loader import DocxLoader
from quickrag.logging import get_logger

logger = get_logger(__name__)


class AutoLoader(BaseLoader):
    """Automatically detect and use the appropriate loader.

    Supports:
    - Text files (.txt, .md, .markdown, .rst)
    - PDF files (.pdf)
    - CSV/TSV files (.csv, .tsv)
    - JSON/JSONL files (.json, .jsonl)
    - Word documents (.docx)
    - Web pages (http://, https://)
    - Directories (recursively loads all supported files)
    """

    def __init__(self):
        """Initialize with all available loaders."""
        self.loaders = [
            WebLoader(),
            PDFLoader(),
            CSVLoader(),
            JSONLoader(),
            DocxLoader(),
            TextLoader(),
        ]

    def supports(self, source: str | Path) -> bool:
        """Check if any loader supports this source."""
        return any(loader.supports(source) for loader in self.loaders)

    def _get_loader(self, source: str | Path) -> BaseLoader | None:
        """Get the appropriate loader for a source."""
        for loader in self.loaders:
            if loader.supports(source):
                return loader
        return None

    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load documents from any supported source.

        Args:
            source: Path, URL, or directory to load from.

        Returns:
            List of loaded documents.
        """
        path = Path(source) if not str(source).startswith(("http://", "https://")) else None

        # Handle directories
        if path and path.is_dir():
            return self._load_directory(path)

        # Get appropriate loader
        loader = self._get_loader(source)
        if loader is None:
            raise ValueError(f"No loader found for: {source}")

        return loader.load(source)

    def _load_directory(self, directory: Path) -> list[LoadedDocument]:
        """Recursively load all supported files from a directory."""
        documents = []

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                loader = self._get_loader(file_path)
                if loader:
                    try:
                        docs = loader.load(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", file_path, e)

        return documents


# Convenience function
def load(source: Union[str, Path, list[str], list[Path]]) -> list[LoadedDocument]:
    """Load documents from one or more sources.

    Args:
        source: Single source or list of sources (paths, URLs, directories).

    Returns:
        List of loaded documents.

    Example:
        >>> docs = load("./documents")  # Directory
        >>> docs = load("file.pdf")  # Single file
        >>> docs = load("https://example.com")  # Web page
        >>> docs = load(["file1.pdf", "file2.md"])  # Multiple files
    """
    loader = AutoLoader()

    if isinstance(source, (list, tuple)):
        documents = []
        for s in source:
            documents.extend(loader.load(s))
        return documents

    return loader.load(source)

