"""
Tests for graph routing logic.
Run with: python -m pytest tests/agents/test_graph.py -v
"""

from src.graph.graph import route_after_critic, build_graph
from src.graph.state import create_initial_state
from src.config import settings


def test_route_passes_when_critique_passed():
    state = {"critique_passed": True, "regeneration_count": 0}
    assert route_after_critic(state) == "END"


def test_route_loops_when_critique_failed():
    state = {"critique_passed": False, "regeneration_count": 0}
    assert route_after_critic(state) == "reasoning_agent"


def test_route_ends_when_max_regenerations_reached():
    state = {
        "critique_passed": False,
        "regeneration_count": settings.MAX_REGENERATIONS + 1
    }
    assert route_after_critic(state) == "END"


def test_route_still_loops_at_max_minus_one():
    state = {
        "critique_passed": False,
        "regeneration_count": settings.MAX_REGENERATIONS - 1
    }
    assert route_after_critic(state) == "reasoning_agent"


def test_graph_builds_successfully():
    graph = build_graph()
    assert graph is not None


def test_graph_has_correct_nodes():
    graph = build_graph()
    node_names = list(graph.nodes.keys())
    assert "orchestrator" in node_names
    assert "retrieval_agent" in node_names
    assert "reasoning_agent" in node_names
    assert "critic_agent" in node_names


def test_initial_state_valid_for_graph():
    state = create_initial_state(
        query="What are the termination conditions?",
        doc_id="test_doc"
    )
    assert state["query"] == "What are the termination conditions?"
    assert state["doc_id"] == "test_doc"
    assert state["regeneration_count"] == 0
