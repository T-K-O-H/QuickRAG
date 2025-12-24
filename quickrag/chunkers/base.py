"""Base chunker class for QuickRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0  # Position in original document

    def __len__(self) -> int:
        return len(self.content)


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of chunks.
        """
        ...

