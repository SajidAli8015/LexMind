"""
API Schemas for LexMind
Pydantic models for all request and response payloads.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Ingest Schemas ───────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response returned after document ingestion."""
    success: bool
    doc_id: str
    file_name: str
    doc_title: str = ""
    chunks_created: int
    articles_found: int
    total_chars: int
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "doc_id": "contract_abc_a1b2c3d4",
                "file_name": "nda_2024.pdf",
                "doc_title": "Non-Disclosure Agreement",
                "chunks_created": 23,
                "articles_found": 8,
                "total_chars": 34500,
                "message": "Document ingested successfully"
            }
        }


# ─── Query Schemas ────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the query endpoint."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The legal question to answer"
    )
    doc_id: Optional[str] = Field(
        None,
        description="Restrict search to this document. None = search all."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the termination conditions?",
                "doc_id": "contract_abc_a1b2c3d4"
            }
        }


class QueryResponse(BaseModel):
    """Response returned after running the agent graph."""
    success: bool
    query: str
    query_type: str
    final_answer: str
    citations: List[str]
    groundedness_score: Optional[float]
    citation_score: Optional[float]
    relevance_score: Optional[float]
    critique_passed: Optional[bool]
    regeneration_count: Optional[int]
    chunks_used: int
    error: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "What are the termination conditions?",
                "query_type": "factual",
                "final_answer": "Either party may terminate upon 30 days written notice [Article 47].",
                "citations": ["Article 47", "Article 47.1"],
                "groundedness_score": 0.91,
                "citation_score": 0.88,
                "relevance_score": 0.95,
                "critique_passed": True,
                "regeneration_count": 0,
                "chunks_used": 5,
                "error": None
            }
        }


# ─── Document List Schemas ────────────────────────────────────

class DocumentInfo(BaseModel):
    """Information about one ingested document."""
    doc_id: str
    file_name: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response for listing all ingested documents."""
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]


# ─── Health Schema ────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for the health check endpoint."""
    status: str
    version: str
    documents_ingested: int
    total_chunks: int


# ─── Session Schemas ──────────────────────────────────────────

class MessageResponse(BaseModel):
    """A single message in a session."""
    id:                 str
    session_id:         str
    role:               str
    content:            str
    query_type:         Optional[str]     = None
    citations:          List[str]         = []
    groundedness_score: Optional[float]   = None
    citation_score:     Optional[float]   = None
    relevance_score:    Optional[float]   = None
    critique_passed:    Optional[bool]    = None
    regeneration_count: Optional[int]     = None
    chunks_used:        Optional[int]     = None
    created_at:         str


class SessionResponse(BaseModel):
    """A research session with metadata."""
    id:            str
    title:         str
    doc_id:        Optional[str]  = None
    doc_name:      Optional[str]  = None
    created_at:    str
    updated_at:    str
    message_count: int


class SessionDetailResponse(BaseModel):
    """A session with all its messages."""
    id:            str
    title:         str
    doc_id:        Optional[str]  = None
    doc_name:      Optional[str]  = None
    created_at:    str
    updated_at:    str
    message_count: int
    messages:      List[MessageResponse] = []


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    title:    str  = "New research session"
    doc_id:   Optional[str] = None
    doc_name: Optional[str] = None


class SendMessageRequest(BaseModel):
    """Request to send a message in a session."""
    content: str = Field(..., min_length=3, max_length=2000)
