"""OpenAI embedding models."""

from quickrag.embeddings.base import BaseEmbeddings
from quickrag.config import settings


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings via API.

    Higher quality but requires API key and has latency.
    Recommended models:
    - text-embedding-3-small: Fast, cheap, good quality (1536 dims)
    - text-embedding-3-large: Best quality, more expensive (3072 dims)
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        """Initialize OpenAI embeddings.

        Args:
            model_name: OpenAI embedding model name.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        self.model_name = model_name
        self.api_key = api_key or settings.openai_api_key
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

    @property
    def _client(self):
        """Get OpenAI client."""
        from openai import OpenAI

        return OpenAI(api_key=self.api_key)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        # OpenAI API accepts batch requests
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

