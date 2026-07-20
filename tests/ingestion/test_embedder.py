"""
Test suite for document embedder.
Uses a session-scoped shared_embedder fixture from conftest.py
to load the model only once across all tests.

Run with: python -m pytest tests/ingestion/test_embedder.py -v
"""

import tempfile
import os
from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import DocumentEmbedder


SAMPLE_TEXT = """ARTICLE 1 DEFINITIONS
In this Agreement terms shall have the following meanings:
1.1 Agreement means this contract dated January 2024.
1.2 Party means ABC Corporation or XYZ Limited.

ARTICLE 2 PAYMENT TERMS
XYZ Limited shall pay PKR 5000000 within 30 days.
2.1 Late payments attract 2 percent monthly interest.
2.2 All payments in Pakistani Rupees unless agreed otherwise.

ARTICLE 3 TERMINATION
Either party may terminate this Agreement upon notice.
3.1 Termination requires 30 days written notice.
3.2 Immediate termination allowed for material breach.
"""


def create_chunked_doc(content: str = None):
    """Helper: parse and chunk a temporary text document."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt",
        delete=False, encoding="utf-8"
    ) as f:
        f.write(content or SAMPLE_TEXT)
        tmp = f.name
    parsed = parse_document(tmp)
    chunked = chunk_document(parsed)
    os.unlink(tmp)
    return chunked


def test_embedder_initializes():
    """Test DocumentEmbedder initializes with correct defaults."""
    embedder = DocumentEmbedder()
    assert embedder.model_name == "intfloat/multilingual-e5-large"
    assert embedder.device in ("cpu", "cuda", "mps")
    assert embedder.batch_size == 32
    assert embedder._model is None, "Model should not load at init"
    print("  test_embedder_initializes PASSED")


def test_embed_chunks_returns_result(shared_embedder):
    """Test that embed_chunks returns EmbeddingResult."""
    chunked = create_chunked_doc()
    result = shared_embedder.embed_chunks(chunked)
    assert result is not None
    assert result.total_chunks == chunked.total_chunks
    assert result.total_chunks > 0
    assert result.doc_id == chunked.doc_id
    print(
        f"  test_embed_chunks_returns_result PASSED "
        f"({result.total_chunks} chunks)"
    )


def test_embedding_dimension(shared_embedder):
    """Test that all embeddings have exactly 1024 dimensions."""
    chunked = create_chunked_doc()
    result = shared_embedder.embed_chunks(chunked)
    assert result.embedding_dim == 1024, (
        f"Expected 1024 dims, got {result.embedding_dim}"
    )
    for ec in result.embedded_chunks:
        assert len(ec.embedding) == 1024
    print(f"  test_embedding_dimension PASSED (dim=1024)")


def test_embedding_values_are_floats(shared_embedder):
    """Test that embedding values are valid floats."""
    chunked = create_chunked_doc()
    result = shared_embedder.embed_chunks(chunked)
    for ec in result.embedded_chunks:
        assert all(isinstance(v, float) for v in ec.embedding)
    print("  test_embedding_values_are_floats PASSED")


def test_query_embedding(shared_embedder):
    """Test that embed_query returns a 1024-dim vector."""
    vec = shared_embedder.embed_query("What are the payment terms?")
    assert isinstance(vec, list)
    assert len(vec) == 1024
    assert all(isinstance(v, float) for v in vec)
    print(f"  test_query_embedding PASSED (dim={len(vec)})")


def test_semantic_similarity(shared_embedder):
    """
    Test that payment query scores higher on payment chunk
    than on termination chunk.
    """
    chunked = create_chunked_doc()
    result = shared_embedder.embed_chunks(chunked)

    payment_chunk = next(
        (c for c in result.embedded_chunks
         if "payment" in c.text.lower()[:80]),
        None
    )
    termination_chunk = next(
        (c for c in result.embedded_chunks
         if "terminat" in c.text.lower()[:80]),
        None
    )

    assert payment_chunk is not None, "Should find payment chunk"
    assert termination_chunk is not None, "Should find termination chunk"

    query_vec = shared_embedder.embed_query(
        "payment amount and due date PKR"
    )

    sim_payment = shared_embedder.get_similarity(
        query_vec, payment_chunk.embedding
    )
    sim_termination = shared_embedder.get_similarity(
        query_vec, termination_chunk.embedding
    )

    assert sim_payment > sim_termination, (
        f"Payment chunk ({sim_payment:.4f}) should score higher "
        f"than termination chunk ({sim_termination:.4f})"
    )
    print(
        f"  test_semantic_similarity PASSED | "
        f"payment={sim_payment:.4f} "
        f"termination={sim_termination:.4f}"
    )


def test_embeddings_are_normalized(shared_embedder):
    """Test that embedding vectors have unit length (norm = 1.0)."""
    import numpy as np
    vec = shared_embedder.embed_query("test query for normalization")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01, (
        f"Vector should be normalized (norm=1.0), got {norm:.4f}"
    )
    print(
        f"  test_embeddings_are_normalized PASSED "
        f"(norm={norm:.4f})"
    )


def test_different_texts_different_vectors(shared_embedder):
    """Test that different texts produce different vectors."""
    import numpy as np
    vec1 = shared_embedder.embed_query("payment terms and amount due")
    vec2 = shared_embedder.embed_query("termination notice period")
    similarity = shared_embedder.get_similarity(vec1, vec2)
    assert similarity < 0.99, (
        "Different texts should not produce identical vectors"
    )
    print(
        f"  test_different_texts_different_vectors PASSED "
        f"(similarity={similarity:.4f})"
    )
