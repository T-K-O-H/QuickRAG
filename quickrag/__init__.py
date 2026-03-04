"""QuickRAG - Fast, plug-and-play RAG framework built on LangGraph."""

from quickrag.config import FeatureToggles
from quickrag.pipeline import RAGPipeline, RAGResponse, Citation
from quickrag.stores.qdrant import QdrantStore
from quickrag.stores.base import Document, SearchResult
from quickrag.embeddings.local import LocalEmbeddings
from quickrag.embeddings.openai import OpenAIEmbeddings
from quickrag.llms.ollama import OllamaLLM
from quickrag.llms.openai import OpenAILLM
from quickrag.loaders.csv_loader import CSVLoader
from quickrag.loaders.json_loader import JSONLoader
from quickrag.loaders.docx_loader import DocxLoader

__version__ = "0.2.0"
__all__ = [
    "FeatureToggles",
    "RAGPipeline",
    "RAGResponse",
    "Citation",
    "QdrantStore",
    "Document",
    "SearchResult",
    "LocalEmbeddings",
    "OpenAIEmbeddings",
    "OllamaLLM",
    "OpenAILLM",
    "CSVLoader",
    "JSONLoader",
    "DocxLoader",
]

