"""
Vector Store for LexMind
Manages persistent storage and similarity search of
document chunk embeddings using ChromaDB.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from src.config import settings
from src.ingestion.embedder import EmbeddedChunk, EmbeddingResult


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    One result from a similarity search.

    Returned by VectorStore.search() — contains the
    chunk text, similarity score, and all metadata.

    Example:
        SearchResult(
            chunk_id="contract_001_chunk_3",
            text="ARTICLE 3 PAYMENT TERMS...",
            score=0.91,
            article_ref="Article 3"
        )
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float                    # cosine similarity 0-1
    chunk_type: str = "article"
    article_ref: str = ""
    page_number: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SearchResult("
            f"article='{self.article_ref}', "
            f"score={self.score:.4f}, "
            f"chars={self.char_count})"
        )


@dataclass
class StoreStats:
    """
    Statistics about the current vector store state.
    Used by the UI to show document library info.
    """
    total_chunks: int = 0
    total_documents: int = 0
    documents: List[Dict[str, Any]] = field(default_factory=list)
    collection_name: str = ""
    persist_path: str = ""


# ─── Vector Store Class ──────────────────────────────────────

class VectorStore:
    """
    Manages ChromaDB for persistent vector storage and search.

    ChromaDB stores embeddings on disk so documents only need
    to be embedded once. Subsequent queries load from disk
    instantly without re-embedding.

    Storage location: CHROMA_DB_PATH from settings (.env)
    Collection name:  "legal_documents"
    """

    COLLECTION_NAME = "legal_documents"

    def __init__(self, persist_path: str = None):
        """
        Initialize ChromaDB client with persistent storage.

        Args:
            persist_path: Directory for ChromaDB files.
                         Defaults to settings.CHROMA_DB_PATH
        """
        self.persist_path = persist_path or settings.CHROMA_DB_PATH
        self._client = None
        self._collection = None

        # Ensure storage directory exists
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"VectorStore initialized | "
            f"path: {self.persist_path}"
        )

    def _get_collection(self):
        """
        Get or create the ChromaDB collection lazily.

        Lazy initialization means ChromaDB only connects
        when actually needed, not at import time.
        """
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        chunk_count = self._collection.count()
        logger.info(
            f"ChromaDB collection ready | "
            f"name: {self.COLLECTION_NAME} | "
            f"existing chunks: {chunk_count}"
        )
        return self._collection

    def add_documents(
        self,
        embedding_result: EmbeddingResult,
        replace_existing: bool = True
    ) -> int:
        """
        Store embedded chunks from an EmbeddingResult in ChromaDB.

        If replace_existing=True (default) and doc_id already
        exists in the store, old chunks are deleted first.
        This prevents duplicate chunks when re-ingesting.

        Args:
            embedding_result: Output from DocumentEmbedder
            replace_existing: Delete existing chunks for this
                            doc_id before adding new ones

        Returns:
            Number of chunks successfully stored
        """
        collection = self._get_collection()
        embedded_chunks = embedding_result.embedded_chunks

        if not embedded_chunks:
            logger.warning(
                f"No chunks to store for {embedding_result.file_name}"
            )
            return 0

        doc_id = embedding_result.doc_id

        # Delete existing chunks for this document if requested
        if replace_existing:
            existing = self._get_chunks_by_doc_id(doc_id)
            if existing:
                self.delete_document(doc_id)
                logger.info(
                    f"Replaced {len(existing)} existing chunks "
                    f"for doc_id: {doc_id}"
                )

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in embedded_chunks:
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)

            # ChromaDB only supports str, int, float, bool values
            meta = {
                "doc_id": str(chunk.doc_id),
                "file_name": str(
                    chunk.metadata.get("file_name", "")
                ),
                "article_ref": str(chunk.article_ref),
                "chunk_type": str(chunk.chunk_type),
                "chunk_index": int(chunk.chunk_index),
                "total_chunks": int(chunk.total_chunks),
                "page_number": int(chunk.page_number),
                "char_count": int(chunk.char_count),
            }
            metadatas.append(meta)

        # Add to ChromaDB in batches of 100
        batch_size = 100
        total_added = 0

        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end]
            )
            total_added += end - start

        logger.info(
            f"Stored {total_added} chunks | "
            f"doc_id: {doc_id} | "
            f"file: {embedding_result.file_name}"
        )
        return total_added

    def search(
        self,
        query_embedding: List[float],
        n_results: int = None,
        filter_doc_id: str = None,
        filter_article_ref: str = None,
        min_score: float = None,
    ) -> List[SearchResult]:
        """
        Find the most similar chunks to a query vector.

        Args:
            query_embedding: 1024-dim query vector from embedder
            n_results:       Number of results to return
                            Defaults to settings.TOP_K_FINAL
            filter_doc_id:   Only search within this document
            filter_article_ref: Only return chunks from this article
            min_score:       Minimum similarity threshold
                            Defaults to settings.SIMILARITY_THRESHOLD

        Returns:
            List of SearchResult sorted by score descending
        """
        collection = self._get_collection()

        n_results = n_results or settings.TOP_K_FINAL
        min_score = (
            min_score
            if min_score is not None
            else settings.SIMILARITY_THRESHOLD
        )

        # Build metadata filter if specified
        where_filter = self._build_filter(
            filter_doc_id, filter_article_ref
        )

        # Ensure n_results does not exceed collection size
        total_in_collection = collection.count()
        if total_in_collection == 0:
            logger.warning("Vector store is empty — no results")
            return []

        actual_n = min(n_results, total_in_collection)

        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": actual_n,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter:
                query_params["where"] = where_filter

            results = collection.query(**query_params)

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

        # Convert ChromaDB results to SearchResult objects
        # ChromaDB returns distances not similarities
        # For cosine: similarity = 1 - distance
        search_results = []

        if not results["ids"] or not results["ids"][0]:
            return []

        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = float(1.0 - distance)

            if score < min_score:
                continue

            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]

            search_results.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                score=score,
                chunk_type=meta.get("chunk_type", ""),
                article_ref=meta.get("article_ref", ""),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                total_chunks=int(meta.get("total_chunks", 0)),
                char_count=int(meta.get("char_count", 0)),
                metadata=meta,
            ))

        search_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Search returned {len(search_results)} results "
            f"(n_results={actual_n}, min_score={min_score})"
        )
        return search_results

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a specific document.

        Args:
            doc_id: Document identifier to delete

        Returns:
            Number of chunks deleted
        """
        collection = self._get_collection()

        existing = self._get_chunks_by_doc_id(doc_id)
        if not existing:
            logger.warning(
                f"No chunks found for doc_id: {doc_id}"
            )
            return 0

        chunk_ids = [item["id"] for item in existing]
        collection.delete(ids=chunk_ids)

        logger.info(
            f"Deleted {len(chunk_ids)} chunks "
            f"for doc_id: {doc_id}"
        )
        return len(chunk_ids)

    def get_stats(self) -> StoreStats:
        """
        Get statistics about the current vector store.

        Returns:
            StoreStats with chunk count, document list, etc.
        """
        collection = self._get_collection()
        total_chunks = collection.count()

        if total_chunks == 0:
            return StoreStats(
                total_chunks=0,
                total_documents=0,
                collection_name=self.COLLECTION_NAME,
                persist_path=self.persist_path
            )

        all_items = collection.get(include=["metadatas"])

        doc_map = {}
        for meta in all_items["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": meta.get("file_name", ""),
                    "chunk_count": 0
                }
            doc_map[doc_id]["chunk_count"] += 1

        documents = list(doc_map.values())

        return StoreStats(
            total_chunks=total_chunks,
            total_documents=len(documents),
            documents=documents,
            collection_name=self.COLLECTION_NAME,
            persist_path=self.persist_path
        )

    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document is already in the vector store.

        Args:
            doc_id: Document identifier to check

        Returns:
            True if document exists, False otherwise
        """
        existing = self._get_chunks_by_doc_id(doc_id)
        return len(existing) > 0

    def _get_chunks_by_doc_id(
        self, doc_id: str
    ) -> List[Dict]:
        """
        Get all chunk IDs for a specific document.
        Internal helper for delete and deduplication.
        """
        collection = self._get_collection()

        try:
            results = collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            items = []
            for i, chunk_id in enumerate(results["ids"]):
                items.append({
                    "id": chunk_id,
                    "metadata": results["metadatas"][i]
                })
            return items
        except Exception:
            return []

    def _build_filter(
        self,
        doc_id: Optional[str],
        article_ref: Optional[str]
    ) -> Optional[Dict]:
        """
        Build a ChromaDB metadata filter dict.

        ChromaDB filter syntax:
            Single condition: {"field": "value"}
            Multiple conditions: {"$and": [
                {"field1": "value1"},
                {"field2": "value2"}
            ]}
        """
        conditions = []

        if doc_id:
            conditions.append({"doc_id": {"$eq": doc_id}})
        if article_ref:
            conditions.append(
                {"article_ref": {"$eq": article_ref}}
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def reset(self) -> None:
        """
        Delete ALL data from the vector store.
        Use with caution — this cannot be undone.
        """
        collection = self._get_collection()
        count = collection.count()

        if count > 0:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
            logger.warning(
                f"Vector store RESET — deleted {count} chunks"
            )
        else:
            logger.info("Vector store already empty")


# ─── Convenience Function ────────────────────────────────────

def get_vector_store(persist_path: str = None) -> VectorStore:
    """
    Get a VectorStore instance.

    Usage:
        from src.ingestion.vector_store import get_vector_store
        store = get_vector_store()
        results = store.search(query_vector)
    """
    return VectorStore(persist_path=persist_path)
