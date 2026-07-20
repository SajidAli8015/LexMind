"""
Retrieval Agent for LexMind
Finds the most relevant document chunks for a query
using hybrid dense + BM25 search with cross-encoder reranking.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

from src.config import settings
from src.graph.state import LexMindState
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.vector_store import VectorStore, SearchResult


# ─── BM25 Index Manager ──────────────────────────────────────

class BM25Index:
    """
    Manages a BM25 keyword search index over document chunks.

    BM25 is a classic keyword ranking algorithm. It finds
    chunks containing the exact words from the query,
    weighted by term frequency and document length.

    The index is saved to disk at BM25_INDEX_PATH so it
    persists between runs. It must be rebuilt whenever
    new documents are ingested.

    Why BM25 alongside dense search?
    - Dense search finds "terminate" when user asks "cancel"
    - BM25 finds "Article 47" or "PKR 5000000" exactly
    - Together they catch what either alone would miss
    """

    def __init__(self, index_path: str = None):
        self.index_path = index_path or settings.BM25_INDEX_PATH
        self._index = None
        self._corpus = []      # list of chunk texts
        self._chunk_ids = []   # parallel list of chunk IDs
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

    def build_from_vector_store(
        self, vector_store: VectorStore
    ) -> int:
        """
        Build BM25 index from all chunks currently in ChromaDB.

        Called after ingestion to keep BM25 in sync with
        the vector store. Overwrites any existing index.

        Returns:
            Number of chunks indexed
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 not installed. "
                "Run: pip install rank-bm25"
            )

        logger.info("Building BM25 index from vector store...")

        # Get all chunks from ChromaDB
        collection = vector_store._get_collection()
        all_items = collection.get(include=["documents", "metadatas"])

        if not all_items["ids"]:
            logger.warning("Vector store is empty — BM25 index not built")
            return 0

        self._corpus = all_items["documents"]
        self._chunk_ids = all_items["ids"]

        # Tokenise: lowercase and split on whitespace
        tokenised = [doc.lower().split() for doc in self._corpus]
        self._index = BM25Okapi(tokenised)

        # Save to disk
        self._save()

        logger.info(
            f"BM25 index built: {len(self._corpus)} chunks indexed"
        )
        return len(self._corpus)

    def search(
        self,
        query: str,
        n_results: int = None,
        filter_doc_id: str = None,
    ) -> List[Tuple[str, float]]:
        """
        Search BM25 index for chunks matching query keywords.

        Args:
            query:         The search query
            n_results:     Number of results (default MAX_CHUNKS_BM25)
            filter_doc_id: Only return chunks from this document

        Returns:
            List of (chunk_id, bm25_score) tuples sorted by score
        """
        if not self._load_if_needed():
            logger.warning("BM25 index not available — skipping BM25 search")
            return []

        n_results = n_results or settings.MAX_CHUNKS_BM25
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        # Pair scores with chunk IDs and sort
        scored = list(zip(self._chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply doc_id filter if specified
        if filter_doc_id:
            scored = [
                (cid, score) for cid, score in scored
                if doc_id_from_chunk_id(cid) == filter_doc_id
            ]

        return scored[:n_results]

    def _save(self):
        """Save index to disk."""
        data = {
            "index": self._index,
            "corpus": self._corpus,
            "chunk_ids": self._chunk_ids,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {self.index_path}")

    def _load_if_needed(self) -> bool:
        """Load index from disk if not already in memory."""
        if self._index is not None:
            return True
        if not os.path.exists(self.index_path):
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._index = data["index"]
            self._corpus = data["corpus"]
            self._chunk_ids = data["chunk_ids"]
            logger.info(
                f"BM25 index loaded: {len(self._corpus)} chunks"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """True if index exists and can be loaded."""
        return self._load_if_needed()


# ─── Cross-Encoder Reranker ───────────────────────────────────

class CrossEncoderReranker:
    """
    Reranks candidate chunks using a cross-encoder model.

    Unlike E5 which encodes query and document separately,
    a cross-encoder reads both together. This is much more
    accurate but slower — so we only use it to rerank
    the top candidates (dense + BM25 merged), not all chunks.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (from settings)
    This model is ~85MB and runs in < 1 second for 40 pairs.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        logger.info(
            f"CrossEncoderReranker initialized | "
            f"model: {self.model_name}"
        )

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=512
            )
            logger.info(
                f"Cross-encoder loaded: {self.model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise

    def rerank(
        self,
        query: str,
        chunks: List[SearchResult],
        top_k: int = None,
    ) -> List[SearchResult]:
        """
        Rerank chunks by relevance to query.

        Args:
            query:  The user's question
            chunks: Candidate chunks to rerank
            top_k:  How many to return (default TOP_K_FINAL)

        Returns:
            Top-k chunks sorted by cross-encoder score descending
        """
        if not chunks:
            return []

        top_k = top_k or settings.TOP_K_FINAL
        self._load_model()

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]

        # Score all pairs — cross-encoder reads both together
        scores = self._model.predict(pairs)

        # Attach cross-encoder scores and sort
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Update the score field on each SearchResult
        result = []
        for chunk, ce_score in scored_chunks[:top_k]:
            chunk.score = float(ce_score)
            result.append(chunk)

        logger.info(
            f"Reranked {len(chunks)} chunks → top {len(result)} | "
            f"best score: {result[0].score:.4f}"
        )
        return result


# ─── Helper Functions ─────────────────────────────────────────

def doc_id_from_chunk_id(chunk_id: str) -> str:
    """
    Extract doc_id from a chunk_id.

    Chunk IDs have format: "{doc_id}_chunk_{n}"
    Example: "contract_abc_a1b2_chunk_3" → "contract_abc_a1b2"
    """
    parts = chunk_id.split("_chunk_")
    return parts[0] if len(parts) >= 2 else chunk_id


def merge_and_deduplicate(
    dense_results: List[SearchResult],
    bm25_chunk_ids: List[Tuple[str, float]],
    vector_store: VectorStore,
) -> List[SearchResult]:
    """
    Merge dense and BM25 results, removing duplicates.

    Dense results are SearchResult objects (with text).
    BM25 results are (chunk_id, score) tuples — we need
    to fetch their text from ChromaDB.

    Returns:
        Combined list of unique SearchResult objects
    """
    seen_ids = set()
    merged = []

    # Add dense results first (already have text)
    for chunk in dense_results:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            merged.append(chunk)

    # Add BM25 results not already in dense results
    if bm25_chunk_ids:
        new_ids = [
            cid for cid, _ in bm25_chunk_ids
            if cid not in seen_ids
        ]
        if new_ids:
            try:
                collection = vector_store._get_collection()
                bm25_items = collection.get(
                    ids=new_ids,
                    include=["documents", "metadatas"]
                )
                for i, chunk_id in enumerate(bm25_items["ids"]):
                    meta = bm25_items["metadatas"][i]
                    text = bm25_items["documents"][i]
                    merged.append(SearchResult(
                        chunk_id=chunk_id,
                        doc_id=meta.get("doc_id", ""),
                        text=text,
                        score=0.0,  # will be set by reranker
                        chunk_type=meta.get("chunk_type", ""),
                        article_ref=meta.get("article_ref", ""),
                        page_number=int(meta.get("page_number", 0)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        total_chunks=int(meta.get("total_chunks", 0)),
                        char_count=int(meta.get("char_count", 0)),
                        metadata=meta,
                    ))
                    seen_ids.add(chunk_id)
            except Exception as e:
                logger.warning(f"Failed to fetch BM25 chunks: {e}")

    return merged


# ─── Retrieval Agent ─────────────────────────────────────────

class RetrievalAgent:
    """
    Hybrid retrieval agent for LexMind.

    Combines dense vector search (E5) with BM25 keyword
    search, then reranks with a cross-encoder for precision.

    Usage as a LangGraph node:
        agent = RetrievalAgent()
        state = agent.run(state)
    """

    def __init__(
        self,
        embedder: DocumentEmbedder = None,
        vector_store: VectorStore = None,
        bm25_index: BM25Index = None,
        reranker: CrossEncoderReranker = None,
    ):
        self.embedder = embedder or DocumentEmbedder()
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or CrossEncoderReranker()
        logger.info("RetrievalAgent initialized")

    def run(self, state: LexMindState) -> LexMindState:
        """
        LangGraph node function.

        Reads:  state['query'], state['doc_id'], state['query_type']
        Writes: state['retrieved_chunks'], state['retrieval_scores'],
                state['search_strategy']

        Args:
            state: Current LexMindState

        Returns:
            Updated LexMindState with retrieved_chunks filled in
        """
        query = state["query"]
        doc_id = state.get("doc_id")
        query_type = state.get("query_type", "factual")

        logger.info(
            f"RetrievalAgent running | "
            f"query_type={query_type} | "
            f"doc_id={doc_id or 'all'}"
        )

        try:
            chunks = self._retrieve(query, doc_id, query_type)

            state["retrieved_chunks"] = chunks
            state["retrieval_scores"] = [c.score for c in chunks]
            state["search_strategy"] = (
                "hybrid" if self.bm25_index.is_available
                else "dense"
            )

            logger.info(
                f"Retrieval complete | "
                f"{len(chunks)} chunks | "
                f"strategy={state['search_strategy']}"
            )

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state["retrieved_chunks"] = []
            state["retrieval_scores"] = []
            state["search_strategy"] = "failed"
            state["error"] = f"Retrieval failed: {str(e)}"

        return state

    def _retrieve(
        self,
        query: str,
        doc_id: Optional[str],
        query_type: str,
    ) -> List[SearchResult]:
        """
        Core retrieval logic: dense + BM25 + rerank.
        """
        # Step 1: Dense search
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=settings.MAX_CHUNKS_DENSE,
            filter_doc_id=doc_id,
        )
        logger.info(f"Dense search: {len(dense_results)} results")

        # Step 2: BM25 search (if index available)
        bm25_results = []
        if self.bm25_index.is_available:
            bm25_results = self.bm25_index.search(
                query=query,
                n_results=settings.MAX_CHUNKS_BM25,
                filter_doc_id=doc_id,
            )
            logger.info(f"BM25 search: {len(bm25_results)} results")

        # Step 3: Merge and deduplicate
        candidates = merge_and_deduplicate(
            dense_results=dense_results,
            bm25_chunk_ids=bm25_results,
            vector_store=self.vector_store,
        )
        logger.info(
            f"Merged candidates: {len(candidates)} unique chunks"
        )

        if not candidates:
            logger.warning("No candidates found — returning empty")
            return []

        # Step 4: Cross-encoder reranking
        top_chunks = self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=settings.TOP_K_FINAL,
        )

        return top_chunks


# ─── LangGraph Node Function ──────────────────────────────────

# Module-level agent instance (created once, reused)
_retrieval_agent: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    """Get or create the module-level RetrievalAgent instance."""
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = RetrievalAgent()
    return _retrieval_agent


def retrieval_node(state: LexMindState) -> LexMindState:
    """
    LangGraph node function for the Retrieval Agent.

    This is the function passed to graph.add_node().
    It delegates to the RetrievalAgent instance.

    Usage in graph:
        graph.add_node("retrieval_agent", retrieval_node)
    """
    return get_retrieval_agent().run(state)
