"""
Test suite for legal chunker.
Tests grouping of elements into legal chunks.

Run with: python -m tests.test_chunker
      or: pytest tests/test_chunker.py
"""

import tempfile
import os
from src.ingestion.parser import parse_document
from src.ingestion.chunker import LegalChunker, chunk_document


SAMPLE_LEGAL_TEXT = """PART I - GENERAL PROVISIONS

ARTICLE 1 DEFINITIONS
In this Agreement the following terms shall have meanings:

1.1 Agreement means this contract dated January 2024.
1.2 Party means either ABC Corporation or XYZ Limited.

ARTICLE 2 OBLIGATIONS
ABC Corporation shall perform the following:

2.1 Deliver services within 30 days of the Effective Date.
2.2 Maintain confidentiality of all shared information.

ARTICLE 3 PAYMENT TERMS
XYZ Limited shall pay PKR 5000000 within 30 days.

3.1 Late payments attract 2 percent monthly interest.
3.2 Disputes must be raised within 14 days of invoice.
"""


def create_temp_doc(content: str = None):
    """Helper: parse a temporary text document."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt",
        delete=False, encoding="utf-8"
    ) as f:
        f.write(content or SAMPLE_LEGAL_TEXT)
        tmp = f.name
    doc = parse_document(tmp)
    os.unlink(tmp)
    return doc


def test_chunker_initializes():
    """Test LegalChunker initializes correctly."""
    chunker = LegalChunker()
    assert chunker.max_chunk_size == 1500
    assert chunker.chunk_overlap == 150
    print("  test_chunker_initializes PASSED")
    return True


def test_chunks_created():
    """Test that chunking produces chunks."""
    doc = create_temp_doc()
    result = chunk_document(doc)
    assert result.total_chunks > 0, "Should create chunks"
    assert len(result.chunks) == result.total_chunks
    print(f"  test_chunks_created PASSED ({result.total_chunks} chunks)")
    return True


def test_articles_chunked_separately():
    """Test that each article becomes its own chunk."""
    doc = create_temp_doc()
    result = chunk_document(doc)
    assert result.articles_found == 3, (
        f"Should find 3 article chunks, found {result.articles_found}"
    )
    print(f"  test_articles_chunked_separately PASSED "
          f"({result.articles_found} article chunks)")
    return True


def test_chunk_contains_full_article():
    """Test that article chunk contains heading AND sub-clauses."""
    doc = create_temp_doc()
    result = chunk_document(doc)

    article_3 = next(
        (c for c in result.chunks if "Article 3" in c.article_ref),
        None
    )
    assert article_3 is not None, "Should find Article 3 chunk"
    assert "PAYMENT" in article_3.text.upper(), \
        "Article 3 chunk should contain PAYMENT"
    assert "3.1" in article_3.text, \
        "Article 3 chunk should contain sub-clause 3.1"
    assert "3.2" in article_3.text, \
        "Article 3 chunk should contain sub-clause 3.2"
    print("  test_chunk_contains_full_article PASSED")
    return True


def test_chunk_metadata():
    """Test that chunks have required metadata fields."""
    doc = create_temp_doc()
    result = chunk_document(doc)

    for chunk in result.chunks:
        assert chunk.chunk_id, "Chunk should have an ID"
        assert chunk.doc_id, "Chunk should have a doc_id"
        assert chunk.text, "Chunk should have text"
        assert chunk.chunk_type, "Chunk should have a type"
        assert chunk.chunk_index > 0, "Chunk index should be positive"
        assert "doc_id" in chunk.metadata
        assert "file_name" in chunk.metadata
        assert "article_ref" in chunk.metadata

    print(f"  test_chunk_metadata PASSED "
          f"(all {result.total_chunks} chunks have valid metadata)")
    return True


def test_chunk_ids_unique():
    """Test that all chunk IDs are unique."""
    doc = create_temp_doc()
    result = chunk_document(doc)

    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids)), "All chunk IDs should be unique"
    print("  test_chunk_ids_unique PASSED")
    return True


def test_long_article_splits():
    """Test that very long articles are split into sub-chunks."""
    long_article = "ARTICLE 1 VERY LONG ARTICLE\n"
    long_article += "This is a very long paragraph. " * 100

    doc = create_temp_doc(long_article)
    result = chunk_document(doc, max_chunk_size=200)

    assert result.total_chunks > 1, \
        "Long article should be split into multiple chunks"
    print(f"  test_long_article_splits PASSED "
          f"({result.total_chunks} sub-chunks created)")
    return True


def run_all_tests():
    """Run all chunker tests and report results."""
    print("=" * 50)
    print("CHUNKER TESTS")
    print("=" * 50)
    print()

    tests = [
        test_chunker_initializes,
        test_chunks_created,
        test_articles_chunked_separately,
        test_chunk_contains_full_article,
        test_chunk_metadata,
        test_chunk_ids_unique,
        test_long_article_splits,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
