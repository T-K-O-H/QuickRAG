"""FastAPI backend for QuickRAG."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, ingest, collections, documents
from api.dependencies import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize pipeline
    print("[QuickRAG] API starting up...")
    pipeline = get_pipeline()
    print(f"[QuickRAG] Connected to collection: {pipeline.store.collection}")
    print(f"[QuickRAG] Documents in store: {pipeline.count()}")
    yield
    # Shutdown
    print("[QuickRAG] API shutting down...")


app = FastAPI(
    title="QuickRAG API",
    description="Fast, plug-and-play RAG API built on LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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

