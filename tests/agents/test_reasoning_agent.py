"""
Tests for the Reasoning Agent.
Run with: python -m pytest tests/agents/test_reasoning_agent.py -v
"""

from src.agents.reasoning_agent import (
    extract_citations,
    format_chunks_for_prompt,
    get_prompt_template,
)
from src.ingestion.vector_store import SearchResult


def make_chunk(article_ref, text, chunk_id="chunk_1"):
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="test_doc",
        text=text,
        score=0.9,
        article_ref=article_ref,
        chunk_index=1,
    )


def test_extract_citations_article():
    answer = "Either party may terminate [Article 47]."
    citations = extract_citations(answer)
    assert "Article 47" in citations


def test_extract_citations_multiple():
    answer = "Notice is 30 days [Article 47]. Payment is monthly [Article 3]."
    citations = extract_citations(answer)
    assert "Article 47" in citations
    assert "Article 3" in citations


def test_extract_citations_sub_clause():
    answer = "Written notice required [Article 47.1]."
    citations = extract_citations(answer)
    assert "Article 47.1" in citations


def test_extract_citations_section():
    answer = "Governed by Pakistani law [Section 12]."
    citations = extract_citations(answer)
    assert "Section 12" in citations


def test_extract_citations_none():
    answer = "This answer has no citations."
    citations = extract_citations(answer)
    assert citations == []


def test_extract_citations_unique():
    answer = "Terminate [Article 47]. Also see [Article 47] again."
    citations = extract_citations(answer)
    assert citations.count("Article 47") == 1


def test_format_chunks_empty():
    result = format_chunks_for_prompt([])
    assert "No relevant excerpts" in result


def test_format_chunks_single():
    chunk = make_chunk("Article 47", "ARTICLE 47 TERMINATION\nEither party may terminate.")
    result = format_chunks_for_prompt([chunk])
    assert "Article 47" in result
    assert "TERMINATION" in result
    assert "Excerpt 1" in result


def test_format_chunks_multiple():
    chunks = [
        make_chunk("Article 3", "ARTICLE 3 PAYMENT\nPay PKR 5000000.", "chunk_1"),
        make_chunk("Article 47", "ARTICLE 47 TERMINATION\nTerminate with notice.", "chunk_2"),
    ]
    result = format_chunks_for_prompt(chunks)
    assert "Excerpt 1" in result
    assert "Excerpt 2" in result
    assert "Article 3" in result
    assert "Article 47" in result


def test_get_prompt_template_factual():
    template = get_prompt_template("factual")
    assert "{query}" in template
    assert "{chunks}" in template


def test_get_prompt_template_analytical():
    template = get_prompt_template("analytical")
    assert "{query}" in template
    assert "ANALYSIS" in template


def test_get_prompt_template_unknown_falls_back():
    template = get_prompt_template("unknown_type")
    assert "{query}" in template
