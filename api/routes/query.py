"""Query endpoints for QuickRAG API."""

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import verify_api_key
from api.dependencies import get_pipeline, get_pipeline_for_collection
from quickrag.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    query: str
    stream: bool = False
    top_k: int | None = None
    collection: str | None = None
    filter: dict | None = None
    conversational: bool = False


class SourceDocument(BaseModel):
    """A source document from retrieval."""

    content: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    """Response from query endpoint."""

    answer: str
    sources: list[SourceDocument]
    query: str


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, _key: str | None = Depends(verify_api_key)):
    """Query the RAG pipeline.

    Returns an answer based on retrieved documents.  Supports optional
    metadata filtering, per-collection targeting, and conversational mode.
    """
    pipeline = (
        get_pipeline_for_collection(request.collection)
        if request.collection
        else get_pipeline()
    )

    if request.top_k:
        pipeline.top_k = request.top_k

    try:
        if request.conversational:
            response = pipeline.query_conversational(request.query)
        else:
            response = await pipeline.aquery(request.query, filter=request.filter)

        sources = [
            SourceDocument(
                content=s.document.content,
                score=s.score,
                metadata=s.document.metadata,
            )
            for s in response.sources
        ]

        return QueryResponse(
            answer=response.answer,
            sources=sources,
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

    async def generate() -> AsyncIterator[str]:
        try:
            results = pipeline._retrieve(request.query, filter=request.filter)
            sources = [
                {
                    "content": s.document.content,
                    "score": s.score,
                    "metadata": s.document.metadata,
                }
                for s in results
            ]

            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

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
