"""Embedding models for QuickRAG."""

from quickrag.embeddings.base import BaseEmbeddings
from quickrag.embeddings.local import LocalEmbeddings
from quickrag.embeddings.openai import OpenAIEmbeddings

__all__ = ["BaseEmbeddings", "LocalEmbeddings", "OpenAIEmbeddings"]

