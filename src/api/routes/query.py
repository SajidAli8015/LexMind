"""
Query Router
Handles legal queries through the LexMind agent graph.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import QueryRequest, QueryResponse
from src.graph.graph import run_query

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a legal question",
    description="Run a question through the LexMind agent pipeline."
)
async def query_document(request: QueryRequest):
    """
    Ask a legal question about ingested documents.

    Runs the full agent pipeline:
    Orchestrator → Retrieval → Reasoning → Critic

    Returns a grounded, cited answer with quality scores.
    """
    logger.info(
        f"Query request: '{request.query[:80]}' "
        f"doc_id={request.doc_id or 'all'}"
    )

    try:
        result = run_query(
            query=request.query,
            doc_id=request.doc_id,
        )

        final_answer = result.get("final_answer") or result.get("answer") or ""
        chunks_used = len(result.get("retrieved_chunks") or [])

        if not final_answer:
            raise HTTPException(
                status_code=500,
                detail="Agent pipeline did not produce an answer"
            )

        logger.info(
            f"Query complete | "
            f"type={result.get('query_type')} | "
            f"chunks={chunks_used} | "
            f"passed={result.get('critique_passed')}"
        )

        return QueryResponse(
            success=True,
            query=request.query,
            query_type=result.get("query_type") or "factual",
            final_answer=final_answer,
            citations=result.get("citations") or [],
            groundedness_score=result.get("groundedness_score"),
            citation_score=result.get("citation_score"),
            relevance_score=result.get("relevance_score"),
            critique_passed=result.get("critique_passed"),
            regeneration_count=result.get("regeneration_count") or 0,
            chunks_used=chunks_used,
            error=result.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
