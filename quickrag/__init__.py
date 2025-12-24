"""QuickRAG - Fast, plug-and-play RAG framework built on LangGraph."""

from quickrag.pipeline import RAGPipeline
from quickrag.stores.qdrant import QdrantStore
from quickrag.embeddings.local import LocalEmbeddings
from quickrag.embeddings.openai import OpenAIEmbeddings
from quickrag.llms.ollama import OllamaLLM
from quickrag.llms.openai import OpenAILLM

__version__ = "0.1.0"
__all__ = [
    "RAGPipeline",
    "QdrantStore",
    "LocalEmbeddings",
    "OpenAIEmbeddings",
    "OllamaLLM",
    "OpenAILLM",
]

