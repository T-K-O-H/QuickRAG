"""Collection management endpoints for QuickRAG API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline


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
async def list_collections():
    """List all collections."""
    pipeline = get_pipeline()

    try:
        # Get current collection info
        current = CollectionInfo(
            name=pipeline.store.collection,
            document_count=pipeline.count(),
            embedding_dim=pipeline.embeddings.dimension,
        )

        # Get all collections from Qdrant
        collections_response = pipeline.store._client.get_collections()
        collections = []

        for col in collections_response.collections:
            if col.name == pipeline.store.collection:
                collections.append(current)
            else:
                # Get info for other collections
                info = pipeline.store._client.get_collection(col.name)
                collections.append(
                    CollectionInfo(
                        name=col.name,
                        document_count=info.points_count or 0,
                        embedding_dim=0,  # Can't easily get this for other collections
                    )
                )

        return CollectionListResponse(
            collections=collections,
            current=pipeline.store.collection,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}", response_model=CollectionInfo)
async def get_collection(name: str):
    """Get information about a specific collection."""
    pipeline = get_pipeline()

    try:
        if name == pipeline.store.collection:
            return CollectionInfo(
                name=name,
                document_count=pipeline.count(),
                embedding_dim=pipeline.embeddings.dimension,
            )

        # Get info for other collection
        info = pipeline.store._client.get_collection(name)
        return CollectionInfo(
            name=name,
            document_count=info.points_count or 0,
            embedding_dim=0,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a collection and all its documents."""
    pipeline = get_pipeline()

    try:
        if name == pipeline.store.collection:
            # Clear current collection
            pipeline.clear()
            return {"success": True, "message": f"Cleared collection: {name}"}

        # Delete other collection
        pipeline.store._client.delete_collection(name)
        return {"success": True, "message": f"Deleted collection: {name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/clear")
async def clear_collection(name: str):
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
        raise HTTPException(status_code=500, detail=str(e))

