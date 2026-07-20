"""
Tests for LexMindState and create_initial_state.
Run with: python -m pytest tests/agents/test_state.py -v
"""

from src.graph.state import LexMindState, create_initial_state


def test_initial_state_has_query():
    state = create_initial_state("What is the notice period?")
    assert state["query"] == "What is the notice period?"


def test_initial_state_doc_id_default_none():
    state = create_initial_state("test query")
    assert state["doc_id"] is None


def test_initial_state_doc_id_set():
    state = create_initial_state("test query", doc_id="contract_abc")
    assert state["doc_id"] == "contract_abc"


def test_initial_state_regeneration_count_zero():
    state = create_initial_state("test query")
    assert state["regeneration_count"] == 0


def test_initial_state_all_agent_fields_none():
    state = create_initial_state("test query")
    agent_fields = [
        "query_type", "query_intent",
        "retrieved_chunks", "retrieval_scores", "search_strategy",
        "answer", "citations", "reasoning_prompt",
        "groundedness_score", "citation_score", "relevance_score",
        "critique_feedback", "critique_passed",
        "final_answer", "error",
    ]
    for field in agent_fields:
        assert state[field] is None, f"Expected {field} to be None"
