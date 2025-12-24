"""Base embedding class for QuickRAG."""

from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
        return self.embed([text])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            documents: List of document strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embed(documents)

