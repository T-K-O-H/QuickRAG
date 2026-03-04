"""Query endpoints for QuickRAG API."""

import json
from typing import AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import verify_api_key
from api.dependencies import get_pipeline, get_pipeline_for_collection
from quickrag.config import FeatureToggles
from quickrag.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ToggleOverrides(BaseModel):
    """Per-query feature toggle overrides."""

    search_mode: Literal["hybrid", "dense", "keyword"] | None = None
    citations: bool | None = None
    debug: bool | None = None
    score_threshold: float | None = None
    routing: bool | None = None


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    query: str
    stream: bool = False
    top_k: int | None = None
    collection: str | None = None
    filter: dict | None = None
    conversational: bool = False
    toggles: ToggleOverrides | None = None


class SourceDocument(BaseModel):
    """A source document from retrieval."""

    content: str
    score: float
    metadata: dict


class CitationRef(BaseModel):
    """A citation reference linking a [n] marker to a source."""

    ref: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    content_preview: str = ""
    document_id: str | None = None


class QueryResponse(BaseModel):
    """Response from query endpoint."""

    answer: str
    sources: list[SourceDocument]
    citations: list[CitationRef]
    query: str


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, _key: str | None = Depends(verify_api_key)):
    """Query the RAG pipeline.

    Returns an answer based on retrieved documents.  Supports optional
    metadata filtering, per-collection targeting, conversational mode,
    and per-query feature toggle overrides.
    """
    pipeline = (
        get_pipeline_for_collection(request.collection)
        if request.collection
        else get_pipeline()
    )

    if request.top_k:
        pipeline.top_k = request.top_k

    # Build per-query toggles from overrides
    query_toggles = None
    if request.toggles:
        query_toggles = pipeline.toggles.merge(
            request.toggles.model_dump(exclude_none=True)
        )

    try:
        if request.conversational:
            response = pipeline.query_conversational(request.query)
        else:
            response = await pipeline.aquery(
                request.query, filter=request.filter, toggles=query_toggles
            )

        sources = [
            SourceDocument(
                content=s.document.content,
                score=s.score,
                metadata=s.document.metadata,
            )
            for s in response.sources
        ]

        citations = [
            CitationRef(
                ref=c.ref,
                source=c.source,
                page=c.page,
                chunk_index=c.chunk_index,
                score=c.score,
                content_preview=c.content_preview,
                document_id=c.document_id,
            )
            for c in response.citations
        ]

        return QueryResponse(
            answer=response.answer,
            sources=sources,
            citations=citations,
            query=response.query,
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest, _key: str | None = Depends(verify_api_key)):
    """Stream a response from the RAG pipeline.

    Returns Server-Sent Events with response chunks.
    """
    pipeline = (
        get_pipeline_for_collection(request.collection)
        if request.collection
        else get_pipeline()
    )

    if request.top_k:
        pipeline.top_k = request.top_k

    # Build per-query toggles from overrides
    query_toggles = None
    if request.toggles:
        query_toggles = pipeline.toggles.merge(
            request.toggles.model_dump(exclude_none=True)
        )

    async def generate() -> AsyncIterator[str]:
        try:
            results = pipeline._retrieve(
                request.query, filter=request.filter, toggles=query_toggles
            )
            sources = [
                {
                    "content": s.document.content,
                    "score": s.score,
                    "metadata": s.document.metadata,
                }
                for s in results
            ]
            citations = [
                {
                    "ref": f"[{i}]",
                    "source": (
                        s.document.metadata.get("original_filename")
                        or s.document.metadata.get("filename")
                        or s.document.metadata.get("source", "Unknown")
                    ),
                    "page": s.document.metadata.get("page") or s.document.metadata.get("page_number"),
                    "chunk_index": s.document.metadata.get("chunk_index"),
                    "score": s.score,
                    "content_preview": s.document.content[:120].replace("\n", " "),
                    "document_id": s.document.metadata.get("document_id"),
                }
                for i, s in enumerate(results, 1)
            ]

            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"

            async for chunk in pipeline.astream(request.query, filter=request.filter):
                yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error("Stream failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/query/clear-history")
async def clear_history(
    collection: str | None = None,
    _key: str | None = Depends(verify_api_key),
):
    """Clear conversation history for a pipeline."""
    pipeline = (
        get_pipeline_for_collection(collection)
        if collection
        else get_pipeline()
    )
    pipeline.clear_history()
    return {"success": True, "message": "Conversation history cleared"}


@router.get("/toggles")
async def get_toggles(
    collection: str | None = None,
    _key: str | None = Depends(verify_api_key),
):
    """Return the active feature toggles for the pipeline."""
    pipeline = (
        get_pipeline_for_collection(collection)
        if collection
        else get_pipeline()
    )
    t = pipeline.toggles
    return {
        "search_mode": t.search_mode,
        "citations": t.citations,
        "debug": t.debug,
        "chunking_strategy": t.chunking_strategy,
        "score_threshold": t.score_threshold,
        "routing": t.routing,
    }
