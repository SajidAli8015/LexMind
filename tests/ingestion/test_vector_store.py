"""
Test suite for vector store.
Uses a session-scoped shared_embedder fixture from conftest.py
to load the model only once across all tests.

Run with: python -m pytest tests/ingestion/test_vector_store.py -v
"""

import tempfile
import os
import shutil
import time
import gc
from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.vector_store import VectorStore, SearchResult


SAMPLE_TEXT = """ARTICLE 1 DEFINITIONS
In this Agreement terms shall have the following meanings:
1.1 Agreement means this contract dated January 2024.
1.2 Party means ABC Corporation or XYZ Limited.

ARTICLE 2 PAYMENT TERMS
XYZ Limited shall pay PKR 5000000 within 30 days.
2.1 Late payments attract 2 percent monthly interest.
2.2 All payments in Pakistani Rupees unless agreed.

ARTICLE 3 TERMINATION
Either party may terminate this Agreement upon notice.
3.1 Termination requires 30 days written notice.
3.2 Immediate termination allowed for material breach.
"""


def create_test_store():
    """Helper: create a temp vector store for testing."""
    tmp_dir = tempfile.mkdtemp()
    store = VectorStore(persist_path=tmp_dir)
    return store, tmp_dir


def create_embedded_doc(embedder, content: str = None):
    """Helper: parse, chunk, and embed a temp document."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt",
        delete=False, encoding="utf-8"
    ) as f:
        f.write(content or SAMPLE_TEXT)
        tmp = f.name
    parsed = parse_document(tmp)
    chunked = chunk_document(parsed)
    embedded = embedder.embed_chunks(chunked)
    os.unlink(tmp)
    return embedded


def safe_cleanup(path: str) -> None:
    """
    Safely remove a temp ChromaDB directory on Windows.
    ChromaDB holds file handles open so we give it time to release.
    """
    gc.collect()
    time.sleep(0.2)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def test_store_initializes():
    """Test VectorStore initializes and creates directory."""
    store, tmp_dir = create_test_store()
    try:
        assert os.path.exists(tmp_dir)
        print("  test_store_initializes PASSED")
    finally:
        safe_cleanup(tmp_dir)


def test_add_documents(shared_embedder):
    """Test that chunks are stored in ChromaDB."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        count = store.add_documents(embedded)
        assert count == embedded.total_chunks
        assert count > 0
        print(f"  test_add_documents PASSED ({count} chunks stored)")
    finally:
        safe_cleanup(tmp_dir)


def test_search_returns_results(shared_embedder):
    """Test that search returns relevant results."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        query_vec = shared_embedder.embed_query(
            "What are the payment terms?"
        )
        results = store.search(query_vec, n_results=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0 for r in results)
        assert all(r.text for r in results)
        print(
            f"  test_search_returns_results PASSED "
            f"({len(results)} results)"
        )
    finally:
        safe_cleanup(tmp_dir)


def test_search_scores_sorted(shared_embedder):
    """Test that results are sorted by score descending."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        query_vec = shared_embedder.embed_query("payment terms PKR")
        results = store.search(query_vec, n_results=3)
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
        print("  test_search_scores_sorted PASSED")
    finally:
        safe_cleanup(tmp_dir)


def test_semantic_search_relevance(shared_embedder):
    """Test that payment query returns payment chunk first."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        query_vec = shared_embedder.embed_query(
            "payment amount PKR 5000000 due date"
        )
        results = store.search(query_vec, n_results=3)
        assert len(results) > 0
        top_result = results[0]
        assert "payment" in top_result.text.lower() or \
               "payment" in top_result.article_ref.lower(), (
            f"Top result should be payment-related, "
            f"got: {top_result.article_ref}"
        )
        print(
            f"  test_semantic_search_relevance PASSED | "
            f"top: {top_result.article_ref} "
            f"score={top_result.score:.4f}"
        )
    finally:
        safe_cleanup(tmp_dir)


def test_filter_by_doc_id(shared_embedder):
    """Test that filter_doc_id restricts results to one doc."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        doc_id = embedded.doc_id
        query_vec = shared_embedder.embed_query("agreement terms")
        results = store.search(
            query_vec, n_results=5, filter_doc_id=doc_id
        )
        assert all(r.doc_id == doc_id for r in results)
        print(
            f"  test_filter_by_doc_id PASSED "
            f"({len(results)} results from doc {doc_id})"
        )
    finally:
        safe_cleanup(tmp_dir)


def test_delete_document(shared_embedder):
    """Test that delete removes all chunks for a document."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        doc_id = embedded.doc_id
        assert store.document_exists(doc_id)
        deleted = store.delete_document(doc_id)
        assert deleted > 0
        assert not store.document_exists(doc_id)
        print(
            f"  test_delete_document PASSED "
            f"({deleted} chunks deleted)"
        )
    finally:
        safe_cleanup(tmp_dir)


def test_replace_existing(shared_embedder):
    """Test that adding same doc_id replaces old chunks."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        first_count = store.get_stats().total_chunks
        store.add_documents(embedded, replace_existing=True)
        second_count = store.get_stats().total_chunks
        assert first_count == second_count, (
            f"Re-adding same doc should not increase chunks: "
            f"{first_count} vs {second_count}"
        )
        print(
            f"  test_replace_existing PASSED "
            f"({first_count} chunks, no duplicates)"
        )
    finally:
        safe_cleanup(tmp_dir)


def test_get_stats(shared_embedder):
    """Test that get_stats returns correct information."""
    store, tmp_dir = create_test_store()
    try:
        embedded = create_embedded_doc(shared_embedder)
        store.add_documents(embedded)
        stats = store.get_stats()
        assert stats.total_chunks == embedded.total_chunks
        assert stats.total_documents == 1
        assert len(stats.documents) == 1
        assert stats.documents[0]["doc_id"] == embedded.doc_id
        print(
            f"  test_get_stats PASSED | "
            f"chunks={stats.total_chunks} "
            f"docs={stats.total_documents}"
        )
    finally:
        safe_cleanup(tmp_dir)


def test_empty_store_search(shared_embedder):
    """Test that searching empty store returns empty list."""
    store, tmp_dir = create_test_store()
    try:
        query_vec = shared_embedder.embed_query("test query")
        results = store.search(query_vec)
        assert results == []
        print("  test_empty_store_search PASSED")
    finally:
        safe_cleanup(tmp_dir)
