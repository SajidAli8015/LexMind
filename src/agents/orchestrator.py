"""
Orchestrator Agent for LexMind
Classifies user queries before retrieval so downstream
agents can apply the right strategy for each query type.
"""

import re
from typing import Optional
from loguru import logger

from src.config import settings
from src.graph.state import LexMindState
from src.llm_client import get_llm


# ─── Query Types ──────────────────────────────────────────────

QUERY_TYPES = {
    "factual": (
        "A direct question with a single specific answer. "
        "Examples: 'What is the notice period?', "
        "'What is the payment amount?', "
        "'When does this contract expire?'"
    ),
    "analytical": (
        "Requires reasoning or analysis across multiple clauses. "
        "Examples: 'What are the risks in the liability clause?', "
        "'Is this termination clause fair?', "
        "'What obligations does ABC Corp have?'"
    ),
    "comparison": (
        "Asks to compare two or more things within the contract. "
        "Examples: 'How do Articles 3 and 7 differ?', "
        "'Compare the rights of each party', "
        "'What changed between versions?'"
    ),
    "summarisation": (
        "Asks for a summary of a section or the whole document. "
        "Examples: 'Summarise the payment terms', "
        "'Give me an overview of Article 47', "
        "'What are the main obligations?'"
    ),
}


# ─── Classification Prompt ────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are a legal query classifier.
Classify the following question into exactly ONE of these types:

- factual: A direct question with a specific answer
  (e.g. notice periods, payment amounts, dates, names)

- analytical: Requires reasoning or analysis across clauses
  (e.g. risks, fairness, obligations, implications)

- comparison: Compares two or more things in the contract
  (e.g. comparing articles, parties, or provisions)

- summarisation: Asks for a summary of a section
  (e.g. summarise, overview, main points)

QUESTION: {query}

Respond with ONLY a JSON object in this exact format:
{{
  "query_type": "<factual|analytical|comparison|summarisation>",
  "query_intent": "<one sentence describing what the user wants>"
}}"""


# ─── Response Parser ──────────────────────────────────────────

def parse_orchestrator_response(response_text: str) -> dict:
    """
    Parse the LLM's classification response.

    Handles markdown code blocks and extra text around JSON.
    Falls back to 'factual' type on any parse failure.

    Args:
        response_text: Raw LLM response string

    Returns:
        Dict with query_type and query_intent fields

    Example return:
        {
            "query_type": "factual",
            "query_intent": "Find the required notice period for termination"
        }
    """
    import json

    text = response_text.strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        data = json.loads(text)
        query_type = str(data.get("query_type", "factual")).lower()

        # Validate query type
        if query_type not in QUERY_TYPES:
            logger.warning(
                f"Unknown query type '{query_type}' — "
                f"defaulting to 'factual'"
            )
            query_type = "factual"

        return {
            "query_type": query_type,
            "query_intent": str(data.get("query_intent", "")),
        }

    except Exception as e:
        logger.warning(
            f"Failed to parse orchestrator response: {e} — "
            f"defaulting to factual"
        )
        return {
            "query_type": "factual",
            "query_intent": "",
        }


# ─── Orchestrator Agent ───────────────────────────────────────

class OrchestratorAgent:
    """
    Classifies user queries before retrieval.

    Determines query type so downstream agents use the
    right search strategy and prompt template:

    factual      → direct dense search, concise answer prompt
    analytical   → broad search, analytical prompt
    comparison   → multi-article search, comparison prompt
    summarisation→ article-filtered search, summary prompt

    Usage as a LangGraph node:
        agent = OrchestratorAgent()
        state = agent.run(state)
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain chat model. If None, uses get_llm()
        """
        self._llm = llm
        logger.info("OrchestratorAgent initialized")

    @property
    def llm(self):
        """Lazy-load LLM on first use."""
        if self._llm is None:
            self._llm = get_llm()
            logger.info(
                f"LLM loaded for OrchestratorAgent: "
                f"{settings.LLM_PROVIDER}"
            )
        return self._llm

    def run(self, state: LexMindState) -> LexMindState:
        """
        LangGraph node function.

        Reads:  state['query']
        Writes: state['query_type'], state['query_intent']

        Args:
            state: Current LexMindState

        Returns:
            Updated LexMindState with query_type filled in
        """
        query = state["query"]
        logger.info(f"OrchestratorAgent classifying: '{query[:80]}'")

        try:
            result = self._classify(query)

            state["query_type"] = result["query_type"]
            state["query_intent"] = result["query_intent"]

            logger.info(
                f"Query classified as '{result['query_type']}' | "
                f"intent: {result['query_intent']}"
            )

        except Exception as e:
            logger.error(f"Orchestrator failed: {e} — defaulting to factual")
            state["query_type"] = "factual"
            state["query_intent"] = query
            state["error"] = f"Orchestrator failed: {str(e)}"

        return state

    def _classify(self, query: str) -> dict:
        """
        Call LLM to classify the query.

        Returns:
            Dict with query_type and query_intent
        """
        prompt = ORCHESTRATOR_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        return parse_orchestrator_response(response.content)

    def classify(self, query: str) -> dict:
        """
        Public method for classifying a query directly.
        Useful for testing without full state setup.

        Args:
            query: The user's question

        Returns:
            Dict with query_type and query_intent

        Example:
            agent = OrchestratorAgent()
            result = agent.classify("What is the notice period?")
            # result = {"query_type": "factual",
            #           "query_intent": "Find notice period"}
        """
        return self._classify(query)


# ─── LangGraph Node Function ──────────────────────────────────

_orchestrator_agent: Optional[OrchestratorAgent] = None


def get_orchestrator_agent() -> OrchestratorAgent:
    """Get or create the module-level OrchestratorAgent instance."""
    global _orchestrator_agent
    if _orchestrator_agent is None:
        _orchestrator_agent = OrchestratorAgent()
    return _orchestrator_agent


def orchestrator_node(state: LexMindState) -> LexMindState:
    """
    LangGraph node function for the Orchestrator Agent.

    Usage in graph:
        graph.add_node("orchestrator", orchestrator_node)
    """
    return get_orchestrator_agent().run(state)
