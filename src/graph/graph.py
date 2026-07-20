"""
LexMind Agent Graph
Wires Orchestrator → Retrieval → Reasoning → Critic
into a LangGraph StateGraph with conditional routing.
"""

from typing import Optional
from loguru import logger

from src.graph.state import LexMindState, create_initial_state
from src.agents.orchestrator import orchestrator_node
from src.agents.retrieval_agent import retrieval_node
from src.agents.reasoning_agent import reasoning_node
from src.agents.critic_agent import critic_node


# ─── Routing Logic ────────────────────────────────────────────

def route_after_critic(state: LexMindState) -> str:
    """
    Conditional edge function called after the Critic Agent.

    Decides whether to:
    - Return the answer to the user (END)
    - Loop back to the Reasoning Agent for regeneration

    Routing rules:
    1. If critique_passed → END (answer is good)
    2. If regeneration_count > MAX_REGENERATIONS → END
       (tried enough times — accept best answer)
    3. Otherwise → reasoning_agent (try again with feedback)

    Args:
        state: Current LexMindState after Critic ran

    Returns:
        "END" or "reasoning_agent"
    """
    from src.config import settings

    critique_passed = state.get("critique_passed", False)
    regeneration_count = state.get("regeneration_count", 0)

    if critique_passed:
        logger.info(
            f"Routing: critique passed → END "
            f"(regenerations: {regeneration_count})"
        )
        return "END"

    if regeneration_count > settings.MAX_REGENERATIONS:
        logger.warning(
            f"Routing: max regenerations reached "
            f"({settings.MAX_REGENERATIONS}) → END"
        )
        return "END"

    logger.info(
        f"Routing: critique failed → reasoning_agent "
        f"(attempt {regeneration_count + 1} of "
        f"{settings.MAX_REGENERATIONS})"
    )
    return "reasoning_agent"


# ─── Graph Builder ────────────────────────────────────────────

def build_graph():
    """
    Build and compile the LexMind LangGraph graph.

    Graph structure:
        START
          ↓
        orchestrator      (classifies query type)
          ↓
        retrieval_agent   (hybrid search + reranking)
          ↓
        reasoning_agent   (generates grounded answer)
          ↓
        critic_agent      (scores answer quality)
          ↓ (conditional)
        ┌─ critique passed  → END
        └─ critique failed  → reasoning_agent (loop)

    Returns:
        Compiled LangGraph graph ready to invoke

    Usage:
        graph = build_graph()
        result = graph.invoke(create_initial_state(
            query="What are the termination conditions?",
            doc_id="contract_abc_a1b2c3d4"
        ))
        print(result['final_answer'])
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError(
            "langgraph not installed. "
            "Run: pip install langgraph"
        )

    logger.info("Building LexMind agent graph...")

    graph = StateGraph(LexMindState)

    # ── Add nodes ─────────────────────────────────────────────
    graph.add_node("orchestrator",    orchestrator_node)
    graph.add_node("retrieval_agent", retrieval_node)
    graph.add_node("reasoning_agent", reasoning_node)
    graph.add_node("critic_agent",    critic_node)

    # ── Set entry point ───────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # ── Add linear edges ──────────────────────────────────────
    # Orchestrator always goes to Retrieval
    graph.add_edge("orchestrator", "retrieval_agent")
    # Retrieval always goes to Reasoning
    graph.add_edge("retrieval_agent", "reasoning_agent")
    # Reasoning always goes to Critic
    graph.add_edge("reasoning_agent", "critic_agent")

    # ── Add conditional edge after Critic ─────────────────────
    # route_after_critic decides: END or loop to reasoning_agent
    graph.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {
            "reasoning_agent": "reasoning_agent",
            "END": END,
        }
    )

    compiled = graph.compile()
    logger.info("LexMind graph compiled successfully")
    return compiled


# ─── Module-Level Graph Instance ─────────────────────────────

_graph = None


def get_graph():
    """
    Get or create the module-level compiled graph.
    Graph is built once and reused for all queries.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─── Public Query Function ────────────────────────────────────

def run_query(
    query: str,
    doc_id: Optional[str] = None,
    conversation_history: Optional[str] = None,
) -> LexMindState:
    """
    Run a query through the full LexMind agent pipeline.
    conversation_history is passed to the Reasoning Agent
    for coherent follow-up answers but NOT used for retrieval.
    """
    logger.info(
        f"run_query called | "
        f"query='{query[:80]}' | "
        f"doc_id={doc_id or 'all'} | "
        f"has_history={bool(conversation_history)}"
    )

    initial_state = create_initial_state(
        query=query,
        doc_id=doc_id,
    )

    # Inject history into state — Reasoning Agent reads it
    # Retrieval Agent ignores it (only reads query + doc_id)
    if conversation_history:
        initial_state["conversation_history"] = conversation_history

    graph = get_graph()
    result = graph.invoke(initial_state)

    logger.info(
        f"run_query complete | "
        f"query_type={result.get('query_type')} | "
        f"chunks={len(result.get('retrieved_chunks') or [])} | "
        f"critique_passed={result.get('critique_passed')}"
    )

    return result
