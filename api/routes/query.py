"""Query endpoints for QuickRAG API."""

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.dependencies import get_pipeline


router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    query: str
    stream: bool = False
    top_k: int | None = None


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
async def query(request: QueryRequest):
    """Query the RAG pipeline.

    Returns an answer based on retrieved documents.
    """
    pipeline = get_pipeline()

    if request.top_k:
        pipeline.top_k = request.top_k

    try:
        response = await pipeline.aquery(request.query)

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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream a response from the RAG pipeline.

    Returns Server-Sent Events with response chunks.
    """
    pipeline = get_pipeline()

    if request.top_k:
        pipeline.top_k = request.top_k

    async def generate() -> AsyncIterator[str]:
        try:
            # First, get sources
            results = pipeline._retrieve(request.query)
            sources = [
                {
                    "content": s.document.content,
                    "score": s.score,
                    "metadata": s.document.metadata,
                }
                for s in results
            ]

            # Send sources first
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            # Stream the answer
            async for chunk in pipeline.astream(request.query):
                yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

