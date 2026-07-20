"""
Test suite for document parser.
Tests parsing of TXT files and element detection.

Run with: python -m tests.test_parser
      or: pytest tests/test_parser.py
"""

import tempfile
import os
from src.ingestion.parser import DocumentParser, parse_document


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


def create_temp_file(content: str, suffix: str = ".txt") -> str:
    """Helper: create a temporary file with given content."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix,
        delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        return f.name


def test_parser_initializes():
    """Test DocumentParser initializes without errors."""
    parser = DocumentParser()
    assert parser is not None
    print("  test_parser_initializes PASSED")
    return True


def test_parse_txt_returns_document():
    """Test that parsing a TXT file returns a ParsedDocument."""
    tmp = create_temp_file(SAMPLE_LEGAL_TEXT)
    try:
        doc = parse_document(tmp)
        assert doc is not None, "Should return a document"
        assert doc.file_type == "txt", "File type should be txt"
        assert doc.element_count() > 0, "Should have elements"
        print(f"  test_parse_txt_returns_document PASSED "
              f"({doc.element_count()} elements)")
        return True
    finally:
        os.unlink(tmp)


def test_headings_detected():
    """Test that article headings are correctly detected."""
    tmp = create_temp_file(SAMPLE_LEGAL_TEXT)
    try:
        doc = parse_document(tmp)
        headings = doc.get_headings()
        assert len(headings) > 0, "Should detect headings"

        heading_texts = [h.text for h in headings]
        found_article = any(
            "ARTICLE" in t.upper() for t in heading_texts
        )
        assert found_article, "Should detect ARTICLE headings"
        print(f"  test_headings_detected PASSED "
              f"({len(headings)} headings found)")
        return True
    finally:
        os.unlink(tmp)


def test_articles_detected():
    """Test that Article-level headings (level 2) are detected."""
    tmp = create_temp_file(SAMPLE_LEGAL_TEXT)
    try:
        doc = parse_document(tmp)
        articles = doc.get_articles()
        assert len(articles) == 3, (
            f"Should find 3 articles, found {len(articles)}"
        )
        print(f"  test_articles_detected PASSED "
              f"({len(articles)} articles)")
        return True
    finally:
        os.unlink(tmp)


def test_element_levels():
    """Test that heading levels are correctly assigned."""
    tmp = create_temp_file(SAMPLE_LEGAL_TEXT)
    try:
        doc = parse_document(tmp)

        level_1 = [e for e in doc.elements if e.level == 1]
        level_2 = [e for e in doc.elements if e.level == 2]
        level_3 = [e for e in doc.elements if e.level == 3]

        assert len(level_1) >= 1, "Should have at least 1 Part heading"
        assert len(level_2) == 3, f"Should have 3 Article headings, got {len(level_2)}"
        assert len(level_3) >= 6, f"Should have 6+ sub-clauses, got {len(level_3)}"

        print(f"  test_element_levels PASSED "
              f"(L1:{len(level_1)} L2:{len(level_2)} L3:{len(level_3)})")
        return True
    finally:
        os.unlink(tmp)


def test_file_not_found_raises():
    """Test that missing file raises FileNotFoundError."""
    parser = DocumentParser()
    try:
        parser.parse("nonexistent_file.txt")
        print("  FAILED — should have raised FileNotFoundError")
        return False
    except FileNotFoundError:
        print("  test_file_not_found_raises PASSED")
        return True


def test_unsupported_type_raises():
    """Test that unsupported file type raises ValueError."""
    tmp = create_temp_file("some content", suffix=".xyz")
    try:
        parser = DocumentParser()
        parser.parse(tmp)
        print("  FAILED — should have raised ValueError")
        return False
    except ValueError:
        print("  test_unsupported_type_raises PASSED")
        return True
    finally:
        os.unlink(tmp)


def run_all_tests():
    """Run all parser tests and report results."""
    print("=" * 50)
    print("PARSER TESTS")
    print("=" * 50)
    print()

    tests = [
        test_parser_initializes,
        test_parse_txt_returns_document,
        test_headings_detected,
        test_articles_detected,
        test_element_levels,
        test_file_not_found_raises,
        test_unsupported_type_raises,
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
