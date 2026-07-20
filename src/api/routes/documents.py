"""
Documents Router
Lists all documents currently ingested in ChromaDB.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import DocumentListResponse, DocumentInfo

router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List ingested documents",
    description="Returns all documents currently in the knowledge base."
)
async def list_documents():
    try:
        from src.ingestion.vector_store import VectorStore
        store = VectorStore()
        stats = store.get_stats()

        documents = [
            DocumentInfo(
                doc_id=doc["doc_id"],
                file_name=doc.get("file_name", doc["doc_id"]),
                chunk_count=doc["chunk_count"],
            )
            for doc in stats.documents
        ]

        return DocumentListResponse(
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
            documents=documents,
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


@router.delete(
    "/documents/{doc_id}",
    summary="Delete a document",
)
async def delete_document(doc_id: str):
    try:
        from src.ingestion.vector_store import VectorStore
        store = VectorStore()
        deleted = store.delete_document(doc_id)
        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found"
            )
        return {"success": True, "deleted_chunks": deleted, "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
