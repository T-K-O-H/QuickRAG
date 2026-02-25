"""FastAPI backend for QuickRAG."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, ingest, collections, documents
from api.dependencies import get_pipeline
from quickrag.config import settings
from quickrag.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup/shutdown."""
    logger.info("API starting up...")
    pipeline = get_pipeline()
    logger.info("Connected to collection: %s", pipeline.store.collection)
    logger.info("Documents in store: %d", pipeline.count())
    auth_enabled = bool(settings.get_api_keys())
    logger.info("API key authentication: %s", "enabled" if auth_enabled else "disabled")
    yield
    logger.info("API shutting down...")


app = FastAPI(
    title="QuickRAG API",
    description="Fast, plug-and-play RAG API built on LangGraph",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware — origins from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(ingest.router, prefix="/api", tags=["Ingest"])
app.include_router(collections.router, prefix="/api", tags=["Collections"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "QuickRAG API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "collection": pipeline.store.collection,
        "document_count": pipeline.count(),
    }
