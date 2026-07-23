"""
Ingest Router
Handles document upload and ingestion into ChromaDB.
"""

import tempfile
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from loguru import logger

from src.api.schemas import IngestResponse
from src.ingestion.ingestion_pipeline import IngestionPipeline

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Module-level pipeline (created once, reused)
_pipeline: IngestionPipeline = None

def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a legal document",
    description="Upload a PDF, DOCX, or TXT file to ingest into the knowledge base."
)
async def ingest_document(
    file: UploadFile = File(..., description="Legal document to ingest"),
    doc_title: Optional[str] = Form(None),
):
    """
    Upload and ingest a legal document.

    - Accepts PDF, DOCX, or TXT files
    - Saves to a temporary file
    - Runs the full ingestion pipeline
    - Returns doc_id and chunk statistics
    """
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    logger.info(f"Ingest request: {file.filename} ({file.content_type})")

    # Save uploaded file to temp location
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Saved upload to temp: {tmp_path}")

        # Run ingestion pipeline
        pipeline = get_pipeline()
        result = pipeline.ingest(
            tmp_path,
            doc_title=doc_title or None,
            original_filename=file.filename,
        )

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {result.error}"
            )

        logger.info(f"Ingestion complete: {result.summary()}")

        return IngestResponse(
            success=True,
            doc_id=result.doc_id,
            file_name=file.filename,
            doc_title=result.metadata.get("doc_title", ""),
            chunks_created=result.chunks_created,
            articles_found=result.articles_found,
            total_chars=result.total_chars,
            message=f"Document '{file.filename}' ingested successfully"
        )

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug(f"Cleaned up temp file: {tmp_path}")


@router.put(
    "/documents/{doc_id}/reingest",
    response_model=IngestResponse,
    summary="Re-ingest an updated document",
    description="Replace an existing document with a new version."
)
async def reingest_document(
    doc_id: str,
    file: UploadFile = File(...),
    doc_title: Optional[str] = Form(None),
):
    """
    Upload a new version of an existing document.
    Deletes old chunks and ingests the new file.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}"
        )

    logger.info(f"Re-ingest request: {doc_id} → {file.filename}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        pipeline = get_pipeline()
        result = pipeline.reingest(
            file_path=tmp_path,
            doc_id=doc_id,
            doc_title=doc_title or None,
            original_filename=file.filename,
        )

        if not result.success:
            raise HTTPException(500, detail=result.error)

        return IngestResponse(
            success=True,
            doc_id=result.doc_id,
            file_name=file.filename,
            doc_title=result.metadata.get("doc_title", ""),
            chunks_created=result.chunks_created,
            articles_found=result.articles_found,
            total_chars=result.total_chars,
            message=f"Document re-ingested successfully"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
