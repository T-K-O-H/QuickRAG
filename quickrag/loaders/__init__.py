"""Document loaders for QuickRAG."""

from quickrag.loaders.base import BaseLoader, LoadedDocument
from quickrag.loaders.text import TextLoader
from quickrag.loaders.pdf import PDFLoader
from quickrag.loaders.web import WebLoader
from quickrag.loaders.auto import AutoLoader, load

__all__ = [
    "BaseLoader",
    "LoadedDocument",
    "TextLoader",
    "PDFLoader",
    "WebLoader",
    "AutoLoader",
    "load",
]

