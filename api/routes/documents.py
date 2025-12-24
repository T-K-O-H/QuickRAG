"""Document management endpoints for QuickRAG API."""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline


router = APIRouter()


class DocumentInfo(BaseModel):
    """Information about a document."""

    document_id: str
    source: str
    source_type: str  # file, url, text
    filename: str | None = None
    chunk_count: int
    created_at: str | None = None


class DocumentListResponse(BaseModel):
    """Response listing all documents."""

    documents: list[DocumentInfo]
    total_chunks: int


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents in the current collection.
    
    Groups chunks by document_id to show individual documents.
    """
    pipeline = get_pipeline()
    
    try:
        # Scroll through all points to get unique documents
        documents_map: dict[str, DocumentInfo] = {}
        
        # Use scroll to get all points
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
                    source_type = metadata.get("source_type", "unknown")
                    filename = metadata.get("original_filename") or metadata.get("filename")
                    created_at = metadata.get("created_at")
                    
                    # Determine display name
                    if source_type == "url":
                        display_source = source
                    elif filename:
                        display_source = filename
                    else:
                        display_source = source
                    
                    documents_map[doc_id] = DocumentInfo(
                        document_id=doc_id,
                        source=display_source,
                        source_type=source_type,
                        filename=filename,
                        chunk_count=0,
                        created_at=created_at,
                    )
                
                documents_map[doc_id].chunk_count += 1
            
            if offset is None:
                break
        
        documents = list(documents_map.values())
        # Sort by created_at (newest first)
        documents.sort(key=lambda d: d.created_at or "", reverse=True)
        
        total_chunks = sum(d.chunk_count for d in documents)
        
        return DocumentListResponse(
            documents=documents,
            total_chunks=total_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document and all its chunks."""
    pipeline = get_pipeline()
    
    try:
        # Find all points with this document_id
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
        
        # Delete the points
        pipeline.store._client.delete(
            collection_name=pipeline.store.collection,
            points_selector=points_to_delete,
        )
        
        return {
            "success": True,
            "message": f"Deleted {len(points_to_delete)} chunks from document",
            "chunks_deleted": len(points_to_delete),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

