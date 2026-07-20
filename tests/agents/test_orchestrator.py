"""
Tests for the Orchestrator Agent.
Run with: python -m pytest tests/agents/test_orchestrator.py -v
"""

from src.agents.orchestrator import parse_orchestrator_response, QUERY_TYPES
import json


def test_all_query_types_defined():
    assert "factual" in QUERY_TYPES
    assert "analytical" in QUERY_TYPES
    assert "comparison" in QUERY_TYPES
    assert "summarisation" in QUERY_TYPES


def test_parse_factual_response():
    response = json.dumps({
        "query_type": "factual",
        "query_intent": "Find the notice period"
    })
    result = parse_orchestrator_response(response)
    assert result["query_type"] == "factual"
    assert result["query_intent"] == "Find the notice period"


def test_parse_analytical_response():
    response = json.dumps({
        "query_type": "analytical",
        "query_intent": "Analyse liability risks"
    })
    result = parse_orchestrator_response(response)
    assert result["query_type"] == "analytical"


def test_parse_comparison_response():
    response = json.dumps({
        "query_type": "comparison",
        "query_intent": "Compare Article 3 and Article 7"
    })
    result = parse_orchestrator_response(response)
    assert result["query_type"] == "comparison"


def test_parse_summarisation_response():
    response = json.dumps({
        "query_type": "summarisation",
        "query_intent": "Summarise payment terms"
    })
    result = parse_orchestrator_response(response)
    assert result["query_type"] == "summarisation"


def test_parse_fallback_on_invalid_json():
    result = parse_orchestrator_response("not valid json at all")
    assert result["query_type"] == "factual"


def test_parse_fallback_on_unknown_type():
    response = json.dumps({
        "query_type": "unknown_type",
        "query_intent": "something"
    })
    result = parse_orchestrator_response(response)
    assert result["query_type"] == "factual"


def test_parse_handles_markdown_codeblock():
    response = '```json\n{"query_type": "analytical", "query_intent": "test"}\n```'
    result = parse_orchestrator_response(response)
    assert result["query_type"] == "analytical"
