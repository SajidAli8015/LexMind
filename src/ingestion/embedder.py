"""
Document Embedder for LexMind
Converts legal text chunks into dense vector embeddings
using E5-large multilingual model for semantic search.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from loguru import logger

from src.ingestion.chunker import Chunk, ChunkingResult


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """
    A Chunk with its embedding vector attached.

    This is what gets stored in ChromaDB — the text,
    the vector, and the metadata all together.

    Example:
        EmbeddedChunk(
            chunk_id="contract_001_chunk_3",
            text="ARTICLE 3 PAYMENT TERMS...",
            embedding=[0.23, -0.87, 0.45, ...],
            metadata={"article_ref": "Article 3", ...}
        )
    """
    chunk_id: str
    doc_id: str
    text: str
    embedding: List[float]
    chunk_type: str = "article"
    article_ref: str = ""
    page_number: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vector."""
        return len(self.embedding)


@dataclass
class EmbeddingResult:
    """
    Result of embedding an entire document's chunks.
    Contains all EmbeddedChunks plus statistics.
    """
    doc_id: str
    file_name: str
    embedded_chunks: List[EmbeddedChunk] = field(default_factory=list)
    total_chunks: int = 0
    embedding_dim: int = 0
    model_name: str = ""

    def __post_init__(self):
        self.total_chunks = len(self.embedded_chunks)
        if self.embedded_chunks:
            self.embedding_dim = self.embedded_chunks[0].embedding_dim


# ─── Embedder Class ──────────────────────────────────────────

class DocumentEmbedder:
    """
    Generates dense vector embeddings for legal text chunks.

    Uses intfloat/multilingual-e5-large which:
    - Supports 93 languages including Arabic and English
    - Uses asymmetric encoding (query: vs passage: prefixes)
    - Runs locally after initial download (~1.3GB)
    - Produces 1024-dimensional vectors

    The model is loaded once at initialization and reused
    for all subsequent embedding calls to avoid the overhead
    of reloading it each time.
    """

    PASSAGE_PREFIX = "passage: "
    QUERY_PREFIX = "query: "

    def __init__(
        self,
        model_name: str = None,
        batch_size: int = None,
        device: str = None,
    ):
        """
        Initialize the embedder and load the E5 model.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of chunks to embed at once
            device:     'cpu', 'cuda', or None (auto-detect)

        Note: First run downloads ~1.3GB model from HuggingFace.
              Subsequent runs load from local cache instantly.
        """
        from src.config import settings
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None

        if device is not None:
            self.device = device
        elif settings.EMBEDDING_DEVICE != "cpu":
            self.device = settings.EMBEDDING_DEVICE
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif (hasattr(torch.backends, "mps")
                      and torch.backends.mps.is_available()):
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        logger.info(
            f"DocumentEmbedder initialized | "
            f"model: {model_name} | "
            f"device: {self.device} | "
            f"batch_size: {batch_size}"
        )

    def _load_model(self):
        """
        Load the E5 model lazily on first use.

        Lazy loading means the model is only downloaded/loaded
        when actually needed, not at import time.
        First call: downloads model (~1.3GB) and loads it.
        Subsequent calls: loads from local cache instantly.
        """
        if self._model is not None:
            return

        logger.info(
            f"Loading embedding model: {self.model_name} "
            f"(first run downloads ~1.3GB — please wait...)"
        )

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            test_emb = self._model.encode(
                ["test"],
                show_progress_bar=False
            )
            dim = test_emb.shape[1]
            logger.info(
                f"Model loaded successfully | "
                f"embedding dim: {dim} | "
                f"device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_chunks(
        self,
        chunking_result: ChunkingResult
    ) -> EmbeddingResult:
        """
        Embed all chunks from a ChunkingResult.

        This is the main method called by the pipeline.
        Takes the output of chunker.py and adds embeddings.

        Args:
            chunking_result: Output from LegalChunker.chunk()

        Returns:
            EmbeddingResult with all chunks embedded
        """
        self._load_model()

        chunks = chunking_result.chunks
        if not chunks:
            logger.warning(
                f"No chunks to embed for {chunking_result.file_name}"
            )
            return EmbeddingResult(
                doc_id=chunking_result.doc_id,
                file_name=chunking_result.file_name,
                model_name=self.model_name
            )

        logger.info(
            f"Embedding {len(chunks)} chunks for "
            f"{chunking_result.file_name}..."
        )

        texts = [
            self.PASSAGE_PREFIX + chunk.text
            for chunk in chunks
        ]

        embeddings = self._encode_batch(texts)

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                embedding=embedding.tolist(),
                chunk_type=chunk.chunk_type,
                article_ref=chunk.article_ref,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                char_count=chunk.char_count,
                metadata=chunk.metadata,
            ))

        result = EmbeddingResult(
            doc_id=chunking_result.doc_id,
            file_name=chunking_result.file_name,
            embedded_chunks=embedded_chunks,
            model_name=self.model_name,
        )

        logger.info(
            f"Embedded {result.total_chunks} chunks | "
            f"dim={result.embedding_dim} | "
            f"doc={chunking_result.file_name}"
        )
        return result

    def embed_query(self, query_text: str) -> List[float]:
        """
        Embed a user query for similarity search.

        Uses "query: " prefix — different from document chunks
        which use "passage: ". This asymmetry is required by
        the E5 model and improves retrieval accuracy.

        Args:
            query_text: The user's search query

        Returns:
            1024-dimensional embedding vector as list of floats
        """
        self._load_model()

        prefixed = self.QUERY_PREFIX + query_text
        embedding = self._model.encode(
            [prefixed],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of raw texts as documents (passage: prefix).

        Utility method for embedding arbitrary texts.
        Used in testing and by the retrieval agent.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each 1024 floats)
        """
        self._load_model()

        prefixed = [self.PASSAGE_PREFIX + t for t in texts]
        embeddings = self._encode_batch(prefixed)
        return [e.tolist() for e in embeddings]

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts in batches for memory efficiency.

        Processing in batches prevents out-of-memory errors
        when embedding large documents with many chunks.

        Args:
            texts: List of prefixed texts to encode

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = texts[start:end]

            batch_embeddings = self._model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=self.batch_size,
            )
            all_embeddings.append(batch_embeddings)

            if total > self.batch_size:
                logger.debug(f"Embedded batch {end}/{total}")

        return np.vstack(all_embeddings)

    def get_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Returns a score between -1 and 1:
            1.0  = identical meaning
            0.0  = unrelated
           -1.0  = opposite meaning

        Since embeddings are normalized, this equals dot product.
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))


# ─── Convenience Function ────────────────────────────────────

def embed_chunks(
    chunking_result: ChunkingResult,
    model_name: str = None,
) -> EmbeddingResult:
    """
    Convenience function to embed chunks in one line.

    Usage:
        from src.ingestion.embedder import embed_chunks
        result = embed_chunks(chunking_result)
        print(result.total_chunks)
    """
    embedder = DocumentEmbedder(model_name=model_name)
    return embedder.embed_chunks(chunking_result)
