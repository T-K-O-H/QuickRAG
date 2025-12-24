"""Base loader class for QuickRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadedDocument:
    """A loaded document with content and metadata."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Add source to metadata
        self.metadata["source"] = self.source


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str | Path) -> list[LoadedDocument]:
        """Load documents from a source.

        Args:
            source: Path or URL to load from.

        Returns:
            List of loaded documents.
        """
        ...

    @abstractmethod
    def supports(self, source: str | Path) -> bool:
        """Check if this loader supports the given source.

        Args:
            source: Path or URL to check.

        Returns:
            True if supported.
        """
        ...

