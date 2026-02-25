"""Qdrant vector store with hybrid search support."""

from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from quickrag.stores.base import BaseStore, Document, SearchResult
from quickrag.config import settings
from quickrag.logging import get_logger

logger = get_logger(__name__)


class QdrantStore(BaseStore):
    """Qdrant vector store with hybrid search (semantic + BM25).

    Supports both dense (semantic) and sparse (BM25) vectors for
    optimal retrieval quality.
    """

    def __init__(
        self,
        collection: str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedding_dim: int = 384,
        enable_hybrid: bool = True,
    ):
        """Initialize Qdrant store.

        Args:
            collection: Collection name. Falls back to DEFAULT_COLLECTION.
            host: Qdrant host. Falls back to QDRANT_HOST.
            port: Qdrant port. Falls back to QDRANT_PORT.
            embedding_dim: Dimension of dense embeddings.
            enable_hybrid: Enable hybrid search with BM25.
        """
        self.collection = collection or settings.default_collection
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.embedding_dim = embedding_dim
        self.enable_hybrid = enable_hybrid

        self._client = QdrantClient(host=self.host, port=self.port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection for c in collections)

        if not exists:
            vectors_config = {
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                )
            }

            sparse_vectors_config = None
            if self.enable_hybrid:
                sparse_vectors_config = {
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False),
                    )
                }

            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

    def _compute_sparse_vector(
        self, text: str, avg_dl: float = 256.0, k1: float = 1.5, b: float = 0.75
    ) -> tuple[list[int], list[float]]:
        """Compute BM25-weighted sparse vector for text.

        Uses BM25 term-frequency normalization:
            tf_bm25 = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))

        Combined with Qdrant's RRF fusion this provides proper hybrid retrieval.

        Args:
            text: The input text.
            avg_dl: Assumed average document length (in tokens).
            k1: BM25 term saturation parameter.
            b: BM25 length normalization parameter.

        Returns:
            Tuple of (indices, values) for a sparse vector.
        """
        from collections import Counter, defaultdict
        import re

        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [], []

        dl = len(tokens)
        counts = Counter(tokens)

        index_values: defaultdict[int, float] = defaultdict(float)
        for token, tf in counts.items():
            tf_bm25 = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            idx = abs(hash(token)) % 1_000_000
            index_values[idx] += tf_bm25

        sorted_items = sorted(index_values.items())
        indices = [idx for idx, _ in sorted_items]
        values = [val for _, val in sorted_items]

        return indices, values

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> list[str]:
        """Add documents with embeddings to Qdrant."""
        points = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = doc.id or str(uuid.uuid4())

            vectors = {"dense": embedding}

            if self.enable_hybrid:
                indices, values = self._compute_sparse_vector(doc.content)
                vectors["sparse"] = models.SparseVector(indices=indices, values=values)

            point = models.PointStruct(
                id=doc_id,
                vector=vectors,
                payload={
                    "content": doc.content,
                    "metadata": doc.metadata,
                },
            )
            points.append(point)

        self._client.upsert(collection_name=self.collection, points=points)

        return [p.id for p in points]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        query_text: str | None = None,
        hybrid_weight: float = 0.7,
    ) -> list[SearchResult]:
        """Search using hybrid (semantic + sparse) retrieval.

        Args:
            query_embedding: Dense query embedding.
            top_k: Number of results.
            filter: Metadata filter.
            query_text: Original query text for sparse search.
            hybrid_weight: Weight for dense vs sparse (0-1, higher = more dense).

        Returns:
            List of search results.
        """
        # Build filter if provided
        qdrant_filter = None
        if filter:
            conditions = [
                models.FieldCondition(
                    key=f"metadata.{k}",
                    match=models.MatchValue(value=v),
                )
                for k, v in filter.items()
            ]
            qdrant_filter = models.Filter(must=conditions)

        # Hybrid search with prefetch
        if self.enable_hybrid and query_text:
            sparse_indices, sparse_values = self._compute_sparse_vector(query_text)

            query_kwargs = {
                "collection_name": self.collection,
                "prefetch": [
                    models.Prefetch(
                        query=query_embedding,
                        using="dense",
                        limit=top_k * 2,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                        using="sparse",
                        limit=top_k * 2,
                    ),
                ],
                "query": models.FusionQuery(fusion=models.Fusion.RRF),
                "limit": top_k,
                "with_payload": True,
            }
            if qdrant_filter:
                query_kwargs["query_filter"] = qdrant_filter

            results = self._client.query_points(**query_kwargs)
        else:
            # Dense-only search
            query_kwargs = {
                "collection_name": self.collection,
                "query": query_embedding,
                "using": "dense",
                "limit": top_k,
                "with_payload": True,
            }
            if qdrant_filter:
                query_kwargs["query_filter"] = qdrant_filter

            results = self._client.query_points(**query_kwargs)

        # Convert to SearchResults
        search_results = []
        for point in results.points:
            doc = Document(
                content=point.payload.get("content", ""),
                metadata=point.payload.get("metadata", {}),
                id=str(point.id),
            )
            search_results.append(
                SearchResult(
                    document=doc,
                    score=point.score or 0.0,
                )
            )

        return search_results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self._client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=ids),
        )

    def clear(self) -> None:
        """Clear all documents from collection."""
        self._client.delete_collection(self.collection)
        self._ensure_collection()

    def count(self) -> int:
        """Return document count."""
        info = self._client.get_collection(self.collection)
        return info.points_count or 0

