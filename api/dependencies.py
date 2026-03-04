"""FastAPI dependencies for QuickRAG."""

import os
from functools import lru_cache

from quickrag import RAGPipeline
from quickrag.config import FeatureToggles
from quickrag.logging import get_logger

logger = get_logger(__name__)

# Cache of pipelines keyed by collection name
_pipelines: dict[str, RAGPipeline] = {}


@lru_cache()
def get_pipeline() -> RAGPipeline:
    """Get or create the default RAG pipeline singleton.

    Uses environment variables to determine configuration:
    - QUICKRAG_MODE: "local", "cloud", or "hybrid" (default: "local")
    - OPENAI_API_KEY: Required for cloud/hybrid modes
    - QUICKRAG_COLLECTION: Collection name (default: "documents")

    Feature toggles are loaded from QUICKRAG_* environment variables
    (see FeatureToggles.from_settings).
    """
    mode = os.getenv("QUICKRAG_MODE", "local")
    collection = os.getenv("QUICKRAG_COLLECTION", "documents")
    toggles = FeatureToggles.from_settings()
    return _create_pipeline(mode, collection, toggles)


def get_pipeline_for_collection(collection: str) -> RAGPipeline:
    """Get or create a pipeline for a specific collection.

    Enables multi-collection / multi-tenant support by maintaining a cache
    of pipeline instances keyed by collection name.

    Args:
        collection: The Qdrant collection name.

    Returns:
        A RAGPipeline bound to the given collection.
    """
    if collection not in _pipelines:
        mode = os.getenv("QUICKRAG_MODE", "local")
        toggles = FeatureToggles.from_settings()
        _pipelines[collection] = _create_pipeline(mode, collection, toggles)
        logger.info("Created pipeline for collection: %s", collection)
    return _pipelines[collection]


def _create_pipeline(
    mode: str, collection: str, toggles: FeatureToggles | None = None
) -> RAGPipeline:
    """Create a pipeline with the given mode and collection."""
    if mode == "cloud":
        return RAGPipeline.cloud(collection=collection, toggles=toggles)
    elif mode == "hybrid":
        return RAGPipeline.hybrid(collection=collection, toggles=toggles)
    else:
        return RAGPipeline.local(collection=collection, toggles=toggles)
