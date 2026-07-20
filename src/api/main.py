"""
LexMind FastAPI Application
Main entry point for the REST API backend.

Run with:
    uvicorn src.api.main:app --reload --port 8000

Docs at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.api.routes.documents import router as documents_router
from src.api.routes.sessions import router as sessions_router
from src.api.schemas import HealthResponse
from src.ingestion.vector_store import VectorStore
from src.db.database import init_db


# ─── App Setup ────────────────────────────────────────────────

app = FastAPI(
    title="LexMind API",
    description="Multi-agent legal research assistant API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── CORS ─────────────────────────────────────────────────────
# Allow Next.js frontend (localhost:3000) to call the API

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routers ──────────────────────────────────────────────────

app.include_router(ingest_router, prefix="/api", tags=["Ingestion"])
app.include_router(query_router, prefix="/api", tags=["Query"])
app.include_router(documents_router, prefix="/api", tags=["Documents"])
app.include_router(sessions_router, prefix="/api", tags=["Sessions"])


# ─── Health Endpoint ──────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check"
)
async def health_check():
    """Check API status and knowledge base statistics."""
    try:
        store = VectorStore()
        stats = store.get_stats()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            documents_ingested=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            version="0.1.0",
            documents_ingested=0,
            total_chunks=0,
        )


# ─── Startup Event ────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("LexMind API starting up...")
    logger.info("API ready. Docs at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("LexMind API shutting down...")
