"""
Test suite for ingestion pipeline.
Uses session-scoped shared_embedder fixture from conftest.py
so the embedding model loads only once across all tests.

Run with: python -m pytest tests/ingestion/test_pipeline.py -v
"""

import tempfile
import os
import shutil
import time
import gc
from src.ingestion.ingestion_pipeline import IngestionPipeline, IngestionResult


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


def create_temp_file(content: str = None, suffix: str = ".txt") -> str:
    """Helper: write content to a temp file and return path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix,
        delete=False, encoding="utf-8"
    ) as f:
        f.write(content or SAMPLE_TEXT)
        return f.name


def create_temp_store_path() -> str:
    """Helper: create a temp directory for ChromaDB."""
    return tempfile.mkdtemp()


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


def make_pipeline(shared_embedder, persist_path=None):
    """
    Helper: create an IngestionPipeline that reuses
    the session-scoped shared embedder instead of
    loading a new model instance.
    """
    pipeline = IngestionPipeline(persist_path=persist_path)
    pipeline.embedder = shared_embedder  # inject shared embedder
    return pipeline


def test_ingest_document_returns_result(shared_embedder):
    """Test that ingest returns an IngestionResult."""
    tmp_file = create_temp_file()
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        result = pipeline.ingest(tmp_file)
        assert isinstance(result, IngestionResult)
        assert result.success is True
        assert result.doc_id != ""
        assert result.file_name != ""
        print(
            f"  test_ingest_document_returns_result PASSED | "
            f"{result.summary()}"
        )
    finally:
        os.unlink(tmp_file)
        safe_cleanup(tmp_store)


def test_ingest_creates_chunks(shared_embedder):
    """Test that ingestion produces chunks in ChromaDB."""
    tmp_file = create_temp_file()
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        result = pipeline.ingest(tmp_file)
        assert result.chunks_created > 0
        assert result.articles_found > 0
        print(
            f"  test_ingest_creates_chunks PASSED | "
            f"chunks={result.chunks_created} "
            f"articles={result.articles_found}"
        )
    finally:
        os.unlink(tmp_file)
        safe_cleanup(tmp_store)


def test_ingest_and_search(shared_embedder):
    """Test that ingested document is searchable."""
    tmp_file = create_temp_file()
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        result = pipeline.ingest(tmp_file)
        assert result.success

        query_vec = shared_embedder.embed_query(
            "What are the payment terms?"
        )
        search_results = pipeline.vector_store.search(
            query_vec, n_results=3
        )
        assert len(search_results) > 0
        assert search_results[0].score > 0
        print(
            f"  test_ingest_and_search PASSED | "
            f"top result: {search_results[0].article_ref} "
            f"score={search_results[0].score:.4f}"
        )
    finally:
        os.unlink(tmp_file)
        safe_cleanup(tmp_store)


def test_replace_existing_true(shared_embedder):
    """Test that re-ingesting replaces old chunks (no duplicates)."""
    tmp_file = create_temp_file()
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        pipeline.ingest(tmp_file, replace_existing=True)
        count_after_first = pipeline.vector_store.get_stats().total_chunks
        pipeline.ingest(tmp_file, replace_existing=True)
        count_after_second = pipeline.vector_store.get_stats().total_chunks
        assert count_after_first == count_after_second
        print(
            f"  test_replace_existing_true PASSED | "
            f"{count_after_first} chunks, no duplicates"
        )
    finally:
        os.unlink(tmp_file)
        safe_cleanup(tmp_store)


def test_replace_existing_false_skips(shared_embedder):
    """Test that replace_existing=False skips already-ingested docs."""
    tmp_file = create_temp_file()
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        result1 = pipeline.ingest(tmp_file, replace_existing=True)
        assert result1.chunks_created > 0
        result2 = pipeline.ingest(tmp_file, replace_existing=False)
        assert result2.success is True
        assert result2.chunks_created == 0
        assert result2.metadata.get("skipped") is True
        print(
            f"  test_replace_existing_false_skips PASSED | "
            f"second run skipped correctly"
        )
    finally:
        os.unlink(tmp_file)
        safe_cleanup(tmp_store)


def test_ingest_file_not_found(shared_embedder):
    """Test that missing file returns failed result."""
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        result = pipeline.ingest("nonexistent_file.pdf")
        assert result.success is False
        assert result.error is not None
        assert result.chunks_created == 0
        print(
            f"  test_ingest_file_not_found PASSED | "
            f"error: {result.error}"
        )
    finally:
        safe_cleanup(tmp_store)


def test_ingest_many(shared_embedder):
    """Test batch ingestion of multiple files."""
    tmp_files = [create_temp_file() for _ in range(3)]
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        results = pipeline.ingest_many(tmp_files)
        assert len(results) == 3
        assert all(r.success for r in results)
        total_chunks = sum(r.chunks_created for r in results)
        assert total_chunks > 0
        print(
            f"  test_ingest_many PASSED | "
            f"3 files, {total_chunks} total chunks"
        )
    finally:
        for f in tmp_files:
            os.unlink(f)
        safe_cleanup(tmp_store)


def test_result_summary(shared_embedder):
    """Test that IngestionResult.summary() returns a string."""
    tmp_file = create_temp_file()
    tmp_store = create_temp_store_path()
    try:
        pipeline = make_pipeline(shared_embedder, persist_path=tmp_store)
        result = pipeline.ingest(tmp_file)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        print(
            f"  test_result_summary PASSED | "
            f"summary: {summary}"
        )
    finally:
        os.unlink(tmp_file)
        safe_cleanup(tmp_store)
