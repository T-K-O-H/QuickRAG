"""Collection management endpoints for QuickRAG API."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import verify_api_key
from api.dependencies import get_pipeline, get_pipeline_for_collection
from quickrag.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class CollectionInfo(BaseModel):
    """Information about a collection."""

    name: str
    document_count: int
    embedding_dim: int


class CollectionListResponse(BaseModel):
    """Response listing all collections."""

    collections: list[CollectionInfo]
    current: str


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections(_key: str | None = Depends(verify_api_key)):
    """List all collections."""
    pipeline = get_pipeline()

    try:
        current = CollectionInfo(
            name=pipeline.store.collection,
            document_count=pipeline.count(),
            embedding_dim=pipeline.embeddings.dimension,
        )

        collections_response = pipeline.store._client.get_collections()
        collections = []

        for col in collections_response.collections:
            if col.name == pipeline.store.collection:
                collections.append(current)
            else:
                info = pipeline.store._client.get_collection(col.name)
                collections.append(
                    CollectionInfo(
                        name=col.name,
                        document_count=info.points_count or 0,
                        embedding_dim=0,
                    )
                )

        return CollectionListResponse(
            collections=collections,
            current=pipeline.store.collection,
        )
    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}", response_model=CollectionInfo)
async def get_collection(name: str, _key: str | None = Depends(verify_api_key)):
    """Get information about a specific collection."""
    pipeline = get_pipeline()

    try:
        if name == pipeline.store.collection:
            return CollectionInfo(
                name=name,
                document_count=pipeline.count(),
                embedding_dim=pipeline.embeddings.dimension,
            )

        info = pipeline.store._client.get_collection(name)
        return CollectionInfo(
            name=name,
            document_count=info.points_count or 0,
            embedding_dim=0,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")


@router.post("/collections/{name}/switch")
async def switch_collection(name: str, _key: str | None = Depends(verify_api_key)):
    """Get or create a pipeline bound to a specific collection.

    This enables multi-tenant workflows where different tenants operate on
    separate collections.
    """
    try:
        pipeline = get_pipeline_for_collection(name)
        logger.info("Switched to collection: %s", name)
        return {
            "success": True,
            "collection": name,
            "document_count": pipeline.count(),
        }
    except Exception as e:
        logger.error("Failed to switch to collection %s: %s", name, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str, _key: str | None = Depends(verify_api_key)):
    """Delete a collection and all its documents."""
    pipeline = get_pipeline()

    try:
        if name == pipeline.store.collection:
            pipeline.clear()
            return {"success": True, "message": f"Cleared collection: {name}"}

        pipeline.store._client.delete_collection(name)
        logger.info("Deleted collection: %s", name)
        return {"success": True, "message": f"Deleted collection: {name}"}
    except Exception as e:
        logger.error("Failed to delete collection %s: %s", name, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/clear")
async def clear_collection(name: str, _key: str | None = Depends(verify_api_key)):
    """Clear all documents from a collection without deleting it."""
    pipeline = get_pipeline()

    try:
        if name == pipeline.store.collection:
            pipeline.clear()
            return {
                "success": True,
                "message": f"Cleared all documents from collection: {name}",
            }

        raise HTTPException(
            status_code=400,
            detail="Can only clear the current collection. Switch collections first.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to clear collection %s: %s", name, e)
        raise HTTPException(status_code=500, detail=str(e))
