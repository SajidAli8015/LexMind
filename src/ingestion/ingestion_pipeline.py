"""
Ingestion Pipeline for LexMind
Orchestrates all four ingestion stations in sequence:
  parser → chunker → embedder → vector_store

Single entry point for the entire ingestion workflow.
Call ingest_document("contract.pdf") and the full
pipeline runs automatically.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from loguru import logger

from src.config import settings
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import LegalChunker
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.vector_store import VectorStore


# ─── Result Dataclass ─────────────────────────────────────────

@dataclass
class IngestionResult:
    """
    Summary of a completed ingestion run.

    Returned by ingest_document() so the caller knows
    exactly what was processed and stored.

    Example:
        IngestionResult(
            doc_id="contract_001_abc123",
            file_name="contract.pdf",
            chunks_created=23,
            articles_found=8,
            total_chars=34500,
            success=True
        )
    """
    doc_id: str
    file_name: str
    chunks_created: int = 0
    articles_found: int = 0
    total_chars: int = 0
    success: bool = False
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        if not self.success:
            return f"FAILED {self.file_name}: {self.error}"
        return (
            f"OK {self.file_name} | "
            f"chunks={self.chunks_created} "
            f"articles={self.articles_found} "
            f"chars={self.total_chars} "
            f"doc_id={self.doc_id}"
        )


# ─── Pipeline Class ───────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full document ingestion workflow.

    Station 1 — parser:       raw file → structured elements
    Station 2 — chunker:      elements → legal chunks
    Station 3 — embedder:     chunks → 1024-dim vectors
    Station 4 — vector_store: vectors → ChromaDB on disk

    All four stations are initialized once and reused for
    multiple documents, avoiding repeated model loading.

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.ingest("contract.pdf")
        print(result.summary())
    """

    def __init__(
        self,
        persist_path: str = None,
        embedding_model: str = None,
    ):
        """
        Initialize all four pipeline stations.

        Args:
            persist_path:    ChromaDB storage path.
                            Defaults to settings.CHROMA_DB_PATH
            embedding_model: Override embedding model name.
                            Defaults to settings.EMBEDDING_MODEL

        Note: The embedding model loads lazily on first use,
              not at __init__ time.
        """
        logger.info("Initializing IngestionPipeline...")

        self.parser = DocumentParser()
        self.chunker = LegalChunker()
        self.embedder = DocumentEmbedder(
            model_name=embedding_model
        )
        self.vector_store = VectorStore(
            persist_path=persist_path
        )

        logger.info("IngestionPipeline ready — all stations initialized")

    def ingest(
        self,
        file_path: str,
        replace_existing: bool = True,
    ) -> IngestionResult:
        """
        Run the full ingestion pipeline on one document.

        Calls all four stations in sequence:
          parse → chunk → embed → store

        Args:
            file_path:        Path to PDF, DOCX, or TXT file
            replace_existing: If True and document was already
                             ingested, replace old chunks with new.
                             If False and document exists, skip it.

        Returns:
            IngestionResult with counts and doc_id

        Example:
            pipeline = IngestionPipeline()
            result = pipeline.ingest("data/contract.pdf")
            # result.chunks_created = 23
            # result.doc_id = "contract_abc12345"
        """
        path = Path(file_path)
        file_name = path.name

        logger.info(
            f"Starting ingestion | file: {file_name}"
        )

        try:
            # ── Station 1: Parse ──────────────────────────────
            logger.info(f"Station 1/4 — Parsing {file_name}")
            parsed_doc = self.parser.parse(file_path)
            logger.info(
                f"  Parsed: {parsed_doc.element_count()} elements, "
                f"{len(parsed_doc.get_articles())} articles"
            )

            # ── Station 2: Chunk ──────────────────────────────
            logger.info(f"Station 2/4 — Chunking {file_name}")
            chunking_result = self.chunker.chunk(parsed_doc)
            logger.info(
                f"  Chunked: {chunking_result.total_chunks} chunks, "
                f"{chunking_result.articles_found} article chunks"
            )

            # ── Check for existing document ───────────────────
            doc_id = chunking_result.doc_id
            if not replace_existing:
                if self.vector_store.document_exists(doc_id):
                    logger.info(
                        f"Document already exists (doc_id={doc_id}), "
                        f"skipping — set replace_existing=True to update"
                    )
                    return IngestionResult(
                        doc_id=doc_id,
                        file_name=file_name,
                        chunks_created=0,
                        success=True,
                        metadata={"skipped": True, "reason": "already_exists"}
                    )

            # ── Station 3: Embed ──────────────────────────────
            logger.info(f"Station 3/4 — Embedding {file_name}")
            embedding_result = self.embedder.embed_chunks(
                chunking_result
            )
            logger.info(
                f"  Embedded: {embedding_result.total_chunks} chunks "
                f"at dim={embedding_result.embedding_dim}"
            )

            # ── Station 4: Store ──────────────────────────────
            logger.info(f"Station 4/4 — Storing {file_name}")
            stored_count = self.vector_store.add_documents(
                embedding_result,
                replace_existing=replace_existing
            )
            logger.info(
                f"  Stored: {stored_count} chunks in ChromaDB"
            )

            result = IngestionResult(
                doc_id=doc_id,
                file_name=file_name,
                chunks_created=stored_count,
                articles_found=chunking_result.articles_found,
                total_chars=chunking_result.total_chars,
                success=True,
                metadata={
                    "embedding_dim": embedding_result.embedding_dim,
                    "embedding_model": self.embedder.model_name,
                    "parser": parsed_doc.metadata.get("parser", "unknown"),
                }
            )

            logger.info(f"Ingestion complete | {result.summary()}")
            return result

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            return IngestionResult(
                doc_id="",
                file_name=file_name,
                success=False,
                error=str(e)
            )

        except ValueError as e:
            logger.error(f"Invalid file: {e}")
            return IngestionResult(
                doc_id="",
                file_name=file_name,
                success=False,
                error=str(e)
            )

        except Exception as e:
            logger.error(
                f"Ingestion failed for {file_name}: {e}"
            )
            return IngestionResult(
                doc_id="",
                file_name=file_name,
                success=False,
                error=str(e)
            )

    def ingest_many(
        self,
        file_paths: list,
        replace_existing: bool = True,
    ) -> list:
        """
        Ingest multiple documents in sequence.

        Reuses the same pipeline instance for all files —
        the embedding model loads only once, not per file.

        Args:
            file_paths:       List of file paths to ingest
            replace_existing: Passed to each ingest() call

        Returns:
            List of IngestionResult, one per file

        Example:
            pipeline = IngestionPipeline()
            results = pipeline.ingest_many([
                "contracts/contract_a.pdf",
                "contracts/contract_b.pdf",
                "contracts/contract_c.pdf",
            ])
            for r in results:
                print(r.summary())
        """
        results = []
        total = len(file_paths)

        logger.info(
            f"Batch ingestion started | {total} files"
        )

        for i, file_path in enumerate(file_paths, 1):
            logger.info(
                f"Processing file {i}/{total}: {file_path}"
            )
            result = self.ingest(
                file_path,
                replace_existing=replace_existing
            )
            results.append(result)

        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        logger.info(
            f"Batch ingestion complete | "
            f"{passed} succeeded, {failed} failed"
        )
        return results


# ─── Convenience Function ─────────────────────────────────────

def ingest_document(
    file_path: str,
    replace_existing: bool = True,
    persist_path: str = None,
) -> IngestionResult:
    """
    Convenience function — ingest one document in one line.

    Creates a fresh pipeline, runs ingestion, returns result.
    Use IngestionPipeline() directly if ingesting multiple
    files (avoids reloading the embedding model each time).

    Usage:
        from src.ingestion.ingestion_pipeline import ingest_document
        result = ingest_document("contract.pdf")
        print(result.chunks_created)   # 23
        print(result.doc_id)           # "contract_abc12345"
        print(result.success)          # True

    Args:
        file_path:        Path to the document to ingest
        replace_existing: Replace if document already ingested
        persist_path:     ChromaDB path (defaults to settings)

    Returns:
        IngestionResult with ingestion summary
    """
    pipeline = IngestionPipeline(persist_path=persist_path)
    return pipeline.ingest(
        file_path,
        replace_existing=replace_existing
    )
