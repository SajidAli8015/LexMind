"""
LexMind Agent State
Defines the shared TypedDict that flows through every
node in the LangGraph multi-agent graph.

Every agent reads what it needs and writes what it
produces. The graph router reads control fields to
decide which node runs next.
"""

from typing import TypedDict, List, Optional
from src.ingestion.vector_store import SearchResult


class LexMindState(TypedDict):
    """
    Shared state passed between all LexMind agents.

    Fields are grouped by which agent produces them.
    Every field is Optional because the state starts
    empty and gets populated as agents run.

    Flow:
        User input → Orchestrator → Retrieval →
        Reasoning → Critic → Final Answer
    """

    # ── User Input ────────────────────────────────────────────
    query:              str
    # The user's original question exactly as typed.
    # Example: "What are the termination conditions?"

    doc_id:             Optional[str]
    # Which document to search. None = search all documents.
    # Example: "contract_abc_a1b2c3d4"

    # ── Orchestrator Output ───────────────────────────────────
    query_type:         Optional[str]
    # One of: 'factual' | 'analytical' | 'comparison' | 'summarisation'
    # factual      = single direct answer (what is the notice period?)
    # analytical   = reasoning required (what are the risks in clause 3?)
    # comparison   = compare two things (how do articles 3 and 7 differ?)
    # summarisation= summarise a section (summarise the payment terms)

    query_intent:       Optional[str]
    # One-line description of what the user wants.
    # Example: "Find the required notice period for contract termination"

    # ── Retrieval Agent Output ────────────────────────────────
    retrieved_chunks:   Optional[List[SearchResult]]
    # Top-N chunks from hybrid search + reranking.
    # These are the only chunks the Reasoning Agent is
    # allowed to use when generating its answer.

    retrieval_scores:   Optional[List[float]]
    # Similarity/relevance score for each retrieved chunk.
    # Parallel list to retrieved_chunks.

    search_strategy:    Optional[str]
    # Which search strategy was used.
    # One of: 'dense' | 'hybrid' | 'filtered'

    # ── Reasoning Agent Output ────────────────────────────────
    answer:             Optional[str]
    # The generated answer. Must be grounded in
    # retrieved_chunks only — no external knowledge.
    # Example: "According to Article 47, either party may
    #           terminate upon 30 days written notice."

    citations:          Optional[List[str]]
    # Article references cited in the answer.
    # Example: ["Article 47", "Article 48"]

    reasoning_prompt:   Optional[str]
    # The full prompt sent to the LLM (stored for debugging
    # and evaluation purposes).

    # ── Critic Agent Output ───────────────────────────────────
    groundedness_score: Optional[float]
    # 0.0-1.0: Is every claim in the answer supported by
    # a retrieved chunk? 1.0 = fully grounded.
    # Threshold: settings.GROUNDEDNESS_THRESHOLD (default 0.75)

    citation_score:     Optional[float]
    # 0.0-1.0: Are the cited article numbers correct?
    # 1.0 = all citations verified against retrieved chunks.
    # Threshold: settings.CITATION_THRESHOLD (default 0.85)

    relevance_score:    Optional[float]
    # 0.0-1.0: Does the answer address what was asked?
    # 1.0 = answer directly and completely addresses query.
    # Threshold: settings.RELEVANCE_THRESHOLD (default 0.70)

    critique_feedback:  Optional[str]
    # Specific feedback from the Critic explaining what failed.
    # Passed back to the Reasoning Agent for regeneration.
    # Example: "The claim about 30 days notice is not supported
    #           by the retrieved chunks. Article 47 says 14 days."

    critique_passed:    Optional[bool]
    # True if all three scores are above their thresholds.
    # False triggers regeneration (up to MAX_REGENERATIONS).

    # ── Pipeline Control ──────────────────────────────────────
    regeneration_count: Optional[int]
    # How many times the Critic has rejected and asked for
    # regeneration. Starts at 0. Max = settings.MAX_REGENERATIONS.
    # When regeneration_count >= MAX_REGENERATIONS, the graph
    # accepts the current answer regardless of critique scores.

    final_answer:       Optional[str]
    # The answer approved by the Critic (or the best available
    # answer after MAX_REGENERATIONS attempts).
    # This is what gets returned to the user.

    error:              Optional[str]
    # Error message if any agent fails. None if all agents
    # succeeded. The graph should handle this gracefully
    # rather than crashing.

    conversation_history: Optional[str]
    # Formatted previous messages from the session.
    # Used by Reasoning Agent for coherent answers.
    # NOT used by Retrieval Agent — retrieval always
    # uses the raw query for unbiased chunk selection.


def create_initial_state(
    query: str,
    doc_id: Optional[str] = None,
    conversation_history: Optional[str] = None,
) -> LexMindState:
    """
    Create an initial state with only the user input filled in.
    All agent output fields start as None.

    Args:
        query:  The user's question
        doc_id: Optional document ID to restrict search scope

    Returns:
        LexMindState ready to be passed into the graph

    Example:
        state = create_initial_state(
            query="What are the termination conditions?",
            doc_id="contract_abc_a1b2c3d4"
        )
        result = graph.invoke(state)
        print(result['final_answer'])
    """
    return LexMindState(
        # User input
        query=query,
        doc_id=doc_id,

        # Orchestrator (not yet set)
        query_type=None,
        query_intent=None,

        # Retrieval (not yet set)
        retrieved_chunks=None,
        retrieval_scores=None,
        search_strategy=None,

        # Reasoning (not yet set)
        answer=None,
        citations=None,
        reasoning_prompt=None,

        # Critic (not yet set)
        groundedness_score=None,
        citation_score=None,
        relevance_score=None,
        critique_feedback=None,
        critique_passed=None,

        # Pipeline control
        regeneration_count=0,
        final_answer=None,
        error=None,

        # Session context
        conversation_history=conversation_history,
    )
