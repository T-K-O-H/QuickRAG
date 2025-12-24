"""Document ingestion endpoints for QuickRAG API."""

import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from api.dependencies import get_pipeline


router = APIRouter()


class IngestResponse(BaseModel):
    """Response from ingest endpoint."""

    success: bool
    chunks_indexed: int
    message: str
    document_id: str | None = None


class IngestURLRequest(BaseModel):
    """Request body for URL ingestion."""

    url: str
    metadata: dict | None = None


def generate_document_metadata(
    source_type: str,
    source: str,
    filename: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Generate standard document metadata for tracking."""
    meta = {
        "document_id": str(uuid.uuid4()),
        "source_type": source_type,
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
    }
    if filename:
        meta["original_filename"] = filename
        meta["filename"] = filename
    if extra:
        meta.update(extra)
    return meta


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
):
    """Ingest a single file.

    Supports: PDF, TXT, MD, and other text files.
    """
    pipeline = get_pipeline()

    # Parse extra metadata if provided
    extra_meta = {}
    if metadata:
        import json
        try:
            extra_meta = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    # Save uploaded file temporarily
    suffix = Path(file.filename or "document").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Generate document metadata
        meta = generate_document_metadata(
            source_type="file",
            source=file.filename or "uploaded_file",
            filename=file.filename,
            extra=extra_meta,
        )

        # Ingest the file
        chunks = pipeline.ingest(tmp_path, metadata=meta)

        return IngestResponse(
            success=True,
            chunks_indexed=chunks,
            message=f"Successfully indexed {chunks} chunks from {file.filename}",
            document_id=meta["document_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    metadata: str | None = Form(None),
):
    """Ingest multiple files at once.
    
    Each file gets its own document_id for individual management.
    """
    pipeline = get_pipeline()

    # Parse extra metadata if provided
    extra_meta = {}
    if metadata:
        import json
        try:
            extra_meta = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    total_chunks = 0
    errors = []
    document_ids = []

    for file in files:
        suffix = Path(file.filename or "document").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Each file gets its own document_id
            meta = generate_document_metadata(
                source_type="file",
                source=file.filename or "uploaded_file",
                filename=file.filename,
                extra=extra_meta,
            )
            
            chunks = pipeline.ingest(tmp_path, metadata=meta)
            total_chunks += chunks
            document_ids.append(meta["document_id"])
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if errors:
        message = f"Indexed {total_chunks} chunks with {len(errors)} errors: {'; '.join(errors)}"
    else:
        message = f"Successfully indexed {total_chunks} chunks from {len(files)} files"

    return IngestResponse(
        success=len(errors) == 0,
        chunks_indexed=total_chunks,
        message=message,
        document_id=document_ids[0] if len(document_ids) == 1 else None,
    )


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest):
    """Ingest content from a URL."""
    pipeline = get_pipeline()

    try:
        # Generate document metadata
        meta = generate_document_metadata(
            source_type="url",
            source=request.url,
            extra=request.metadata,
        )
        
        chunks = pipeline.ingest(request.url, metadata=meta)

        return IngestResponse(
            success=True,
            chunks_indexed=chunks,
            message=f"Successfully indexed {chunks} chunks from {request.url}",
            document_id=meta["document_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    text: str = Form(...),
    source: str = Form("manual"),
    metadata: str | None = Form(None),
):
    """Ingest raw text content."""
    pipeline = get_pipeline()

    # Parse extra metadata if provided
    extra_meta = {}
    if metadata:
        import json
        try:
            extra_meta = json.loads(metadata)
        except json.JSONDecodeError:
            pass

    try:
        # Generate document metadata
        meta = generate_document_metadata(
            source_type="text",
            source=source,
            extra=extra_meta,
        )
        
        # Create a temporary file with the text
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        chunks = pipeline.ingest(tmp_path, metadata=meta)
        Path(tmp_path).unlink(missing_ok=True)

        return IngestResponse(
            success=True,
            chunks_indexed=chunks,
            message=f"Successfully indexed {chunks} chunks",
            document_id=meta["document_id"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
