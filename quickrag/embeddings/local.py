"""Local embedding models using sentence-transformers."""

from functools import cached_property

from quickrag.embeddings.base import BaseEmbeddings


class LocalEmbeddings(BaseEmbeddings):
    """Local embeddings using sentence-transformers/fastembed.

    Fast, free, and works offline. Recommended models:
    - bge-small-en-v1.5: Good balance of speed and quality (384 dims)
    - all-MiniLM-L6-v2: Fastest option (384 dims)
    - bge-base-en-v1.5: Higher quality, slower (768 dims)
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "bge-small-en-v1.5": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "bge-base-en-v1.5": 768,
        "BAAI/bge-base-en-v1.5": 768,
        "all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(self, model_name: str = "bge-small-en-v1.5"):
        """Initialize local embeddings.

        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        self.model_name = model_name
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 384)

    @cached_property
    def _model(self):
        """Lazy load the embedding model."""
        from fastembed import TextEmbedding

        # Map common names to fastembed format
        model_map = {
            "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        }
        model_id = model_map.get(self.model_name, self.model_name)
        return TextEmbedding(model_name=model_id)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local model.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        # fastembed returns a generator, convert to list
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]

