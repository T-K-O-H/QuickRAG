"""Base store class for QuickRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self):
        if self.id is None:
            import uuid

            self.id = str(uuid.uuid4())


@dataclass
class SearchResult:
    """A search result with document and score."""

    document: Document
    score: float
    sparse_score: float | None = None  # BM25 score for hybrid search
    dense_score: float | None = None  # Semantic score for hybrid search


class BaseStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        """Add documents with their embeddings.

        Args:
            documents: List of documents to add.
            embeddings: List of embedding vectors.

        Returns:
            List of document IDs.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of search results.
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...

