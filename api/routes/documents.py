"""Document management endpoints for QuickRAG API."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import verify_api_key
from api.dependencies import get_pipeline, get_pipeline_for_collection
from quickrag.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class DocumentInfo(BaseModel):
    """Information about a document."""

    document_id: str
    source: str
    source_type: str
    filename: str | None = None
    chunk_count: int
    created_at: str | None = None


class DocumentListResponse(BaseModel):
    """Response listing documents with pagination metadata."""

    documents: list[DocumentInfo]
    total_chunks: int
    total_documents: int
    page: int
    page_size: int
    has_more: bool


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Documents per page"),
    source_type: str | None = Query(None, description="Filter by source type"),
    collection: str | None = Query(None, description="Target collection"),
    _key: str | None = Depends(verify_api_key),
):
    """List documents with pagination and optional filtering.

    Groups chunks by document_id to show individual documents.
    """
    pipeline = (
        get_pipeline_for_collection(collection) if collection else get_pipeline()
    )

    try:
        documents_map: dict[str, DocumentInfo] = {}

        offset = None
        while True:
            results, offset = pipeline.store._client.scroll(
                collection_name=pipeline.store.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                metadata = point.payload.get("metadata", {})
                doc_id = metadata.get("document_id", "unknown")

                if doc_id not in documents_map:
                    source = metadata.get("source", "Unknown")
                    st = metadata.get("source_type", "unknown")
                    filename = metadata.get("original_filename") or metadata.get("filename")
                    created_at = metadata.get("created_at")

                    if source_type and st != source_type:
                        # Skip documents that don't match the filter — but
                        # we still need to count their chunks so keep a
                        # sentinel in the map that we'll strip later.
                        pass
                    else:
                        display_source = source if st == "url" else (filename or source)
                        documents_map[doc_id] = DocumentInfo(
                            document_id=doc_id,
                            source=display_source,
                            source_type=st,
                            filename=filename,
                            chunk_count=0,
                            created_at=created_at,
                        )

                if doc_id in documents_map:
                    documents_map[doc_id].chunk_count += 1

            if offset is None:
                break

        documents = sorted(
            documents_map.values(),
            key=lambda d: d.created_at or "",
            reverse=True,
        )

        total_documents = len(documents)
        total_chunks = sum(d.chunk_count for d in documents)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_docs = documents[start:end]

        return DocumentListResponse(
            documents=page_docs,
            total_chunks=total_chunks,
            total_documents=total_documents,
            page=page,
            page_size=page_size,
            has_more=end < total_documents,
        )
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection: str | None = Query(None),
    _key: str | None = Depends(verify_api_key),
):
    """Delete a specific document and all its chunks."""
    pipeline = (
        get_pipeline_for_collection(collection) if collection else get_pipeline()
    )

    try:
        points_to_delete = []

        offset = None
        while True:
            results, offset = pipeline.store._client.scroll(
                collection_name=pipeline.store.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                metadata = point.payload.get("metadata", {})
                if metadata.get("document_id") == document_id:
                    points_to_delete.append(point.id)

            if offset is None:
                break

        if not points_to_delete:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        pipeline.store._client.delete(
            collection_name=pipeline.store.collection,
            points_selector=points_to_delete,
        )

        logger.info("Deleted document %s (%d chunks)", document_id, len(points_to_delete))
        return {
            "success": True,
            "message": f"Deleted {len(points_to_delete)} chunks from document",
            "chunks_deleted": len(points_to_delete),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=str(e))
