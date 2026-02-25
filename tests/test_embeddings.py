"""Tests for embedding model implementations."""

import pytest

from quickrag.embeddings.base import BaseEmbeddings
from quickrag.embeddings.local import LocalEmbeddings
from quickrag.embeddings.openai import OpenAIEmbeddings


class TestLocalEmbeddings:
    """Tests for local embedding model configuration (no model download)."""

    def test_default_model(self):
        emb = LocalEmbeddings()
        assert emb.model_name == "bge-small-en-v1.5"
        assert emb.dimension == 384

    def test_custom_model_dimension(self):
        emb = LocalEmbeddings("bge-base-en-v1.5")
        assert emb.dimension == 768

    def test_unknown_model_fallback(self):
        emb = LocalEmbeddings("some-unknown-model")
        assert emb.dimension == 384  # fallback default

    def test_dimension_mapping(self):
        assert LocalEmbeddings.MODEL_DIMENSIONS["bge-small-en-v1.5"] == 384
        assert LocalEmbeddings.MODEL_DIMENSIONS["BAAI/bge-small-en-v1.5"] == 384
        assert LocalEmbeddings.MODEL_DIMENSIONS["all-MiniLM-L6-v2"] == 384


class TestOpenAIEmbeddings:
    """Tests for OpenAI embedding model configuration."""

    def test_dimension_mapping(self):
        assert OpenAIEmbeddings.MODEL_DIMENSIONS["text-embedding-3-small"] == 1536
        assert OpenAIEmbeddings.MODEL_DIMENSIONS["text-embedding-3-large"] == 3072

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="API key required"):
            OpenAIEmbeddings(api_key=None)

    def test_custom_api_key(self):
        emb = OpenAIEmbeddings(api_key="test-key")
        assert emb.api_key == "test-key"
        assert emb.dimension == 1536
