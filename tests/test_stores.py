"""Tests for the vector store implementations."""

import pytest

from quickrag.stores.base import Document, SearchResult
from quickrag.stores.qdrant import QdrantStore


class TestDocument:
    """Tests for the Document dataclass."""

    def test_auto_id(self):
        doc = Document(content="hello")
        assert doc.id is not None
        assert isinstance(doc.id, str)
        assert len(doc.id) > 0

    def test_custom_id(self):
        doc = Document(content="hello", id="custom-id")
        assert doc.id == "custom-id"

    def test_metadata_default(self):
        doc = Document(content="hello")
        assert doc.metadata == {}

    def test_metadata_custom(self):
        doc = Document(content="hello", metadata={"key": "value"})
        assert doc.metadata == {"key": "value"}


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_basic(self):
        doc = Document(content="test")
        result = SearchResult(document=doc, score=0.95)
        assert result.score == 0.95
        assert result.document.content == "test"
        assert result.sparse_score is None
        assert result.dense_score is None

    def test_with_scores(self):
        doc = Document(content="test")
        result = SearchResult(document=doc, score=0.9, sparse_score=0.3, dense_score=0.8)
        assert result.sparse_score == 0.3
        assert result.dense_score == 0.8


class TestQdrantStoreSparseVector:
    """Tests for the BM25 sparse vector computation (no Qdrant connection needed)."""

    def test_basic_sparse_vector(self):
        store = QdrantStore.__new__(QdrantStore)
        store.enable_hybrid = True
        indices, values = store._compute_sparse_vector("hello world hello")

        assert len(indices) == len(values)
        assert len(indices) >= 1
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)
        assert all(v > 0 for v in values)

    def test_empty_text(self):
        store = QdrantStore.__new__(QdrantStore)
        store.enable_hybrid = True
        indices, values = store._compute_sparse_vector("")
        assert indices == []
        assert values == []

    def test_indices_sorted(self):
        store = QdrantStore.__new__(QdrantStore)
        store.enable_hybrid = True
        indices, _ = store._compute_sparse_vector("the quick brown fox jumps over the lazy dog")
        assert indices == sorted(indices)

    def test_bm25_saturation(self):
        """Term frequency should saturate — doubling occurrences should not double the score."""
        store = QdrantStore.__new__(QdrantStore)
        store.enable_hybrid = True

        # Text with low TF
        _, values_low = store._compute_sparse_vector("apple")
        # Text with high TF
        _, values_high = store._compute_sparse_vector("apple " * 20)

        # With BM25, 20x repetition should yield less than 20x the value
        # Both should have just one unique token so one value each
        assert len(values_low) == 1
        assert len(values_high) == 1
        assert values_high[0] < values_low[0] * 20

    def test_large_hash_space(self):
        """Indices should use the 1M hash space."""
        store = QdrantStore.__new__(QdrantStore)
        store.enable_hybrid = True
        indices, _ = store._compute_sparse_vector("unique_token_xyz")
        assert all(0 <= i < 1_000_000 for i in indices)
