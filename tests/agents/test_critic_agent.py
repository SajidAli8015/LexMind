"""
Tests for the Critic Agent.
Run with: python -m pytest tests/agents/test_critic_agent.py -v
"""

import json
from src.agents.critic_agent import parse_critic_response, compose_critique_feedback
from src.config import settings


def make_scores(g=0.9, c=0.9, r=0.9):
    return {
        "groundedness_score": g,
        "citation_score": c,
        "relevance_score": r,
        "groundedness_feedback": "OK",
        "citation_feedback": "OK",
        "relevance_feedback": "OK",
    }


def test_parse_valid_response():
    response = json.dumps(make_scores())
    result = parse_critic_response(response)
    assert result["groundedness_score"] == 0.9
    assert result["citation_score"] == 0.9
    assert result["relevance_score"] == 0.9


def test_parse_handles_markdown():
    response = '```json\n' + json.dumps(make_scores()) + '\n```'
    result = parse_critic_response(response)
    assert result["groundedness_score"] == 0.9


def test_parse_clamps_above_one():
    scores = make_scores(g=1.5, c=2.0, r=1.1)
    response = json.dumps(scores)
    result = parse_critic_response(response)
    assert result["groundedness_score"] == 1.0
    assert result["citation_score"] == 1.0
    assert result["relevance_score"] == 1.0


def test_parse_clamps_below_zero():
    scores = make_scores(g=-0.5, c=-1.0, r=-0.1)
    response = json.dumps(scores)
    result = parse_critic_response(response)
    assert result["groundedness_score"] == 0.0
    assert result["citation_score"] == 0.0
    assert result["relevance_score"] == 0.0


def test_parse_fallback_on_invalid_json():
    result = parse_critic_response("not valid json")
    assert "groundedness_score" in result
    assert result["groundedness_score"] == 0.5


def test_compose_feedback_all_pass():
    scores = make_scores(g=0.9, c=0.9, r=0.9)
    feedback = compose_critique_feedback(scores)
    assert feedback == ""


def test_compose_feedback_groundedness_fails():
    scores = make_scores(g=0.3, c=0.9, r=0.9)
    scores["groundedness_feedback"] = "Claim about notice period not in excerpts"
    feedback = compose_critique_feedback(scores)
    assert "Groundedness" in feedback
    assert "notice period" in feedback


def test_compose_feedback_citation_fails():
    scores = make_scores(g=0.9, c=0.3, r=0.9)
    scores["citation_feedback"] = "Article 48 cited but not in excerpts"
    feedback = compose_critique_feedback(scores)
    assert "Citation" in feedback


def test_compose_feedback_relevance_fails():
    scores = make_scores(g=0.9, c=0.9, r=0.3)
    scores["relevance_feedback"] = "Answer discusses payment, not termination"
    feedback = compose_critique_feedback(scores)
    assert "Relevance" in feedback


def test_compose_feedback_multiple_failures():
    scores = make_scores(g=0.3, c=0.3, r=0.3)
    scores["groundedness_feedback"] = "Not grounded"
    scores["citation_feedback"] = "Wrong citations"
    scores["relevance_feedback"] = "Off topic"
    feedback = compose_critique_feedback(scores)
    assert "Groundedness" in feedback
    assert "Citation" in feedback
    assert "Relevance" in feedback
