"""FastAPI dependencies for QuickRAG."""

import os
from functools import lru_cache

from quickrag import RAGPipeline


@lru_cache()
def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline singleton.

    Uses environment variables to determine configuration:
    - QUICKRAG_MODE: "local", "cloud", or "hybrid" (default: "local")
    - OPENAI_API_KEY: Required for cloud/hybrid modes
    - QUICKRAG_COLLECTION: Collection name (default: "documents")
    """
    mode = os.getenv("QUICKRAG_MODE", "local")
    collection = os.getenv("QUICKRAG_COLLECTION", "documents")

    if mode == "cloud":
        return RAGPipeline.cloud(collection=collection)
    elif mode == "hybrid":
        return RAGPipeline.hybrid(collection=collection)
    else:
        return RAGPipeline.local(collection=collection)

