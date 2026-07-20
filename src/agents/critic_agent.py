"""
Critic Agent for LexMind
Quality gate that scores answers on groundedness,
citation accuracy, and relevance before returning
them to the user.
"""

import json
import re
from typing import Optional
from loguru import logger

from src.config import settings
from src.graph.state import LexMindState
from src.llm_client import get_llm
from src.agents.reasoning_agent import format_chunks_for_prompt


# ─── Critic Prompt ────────────────────────────────────────────

CRITIC_PROMPT = """You are a rigorous legal answer quality evaluator.
Evaluate the answer below against the contract excerpts provided.

Score the answer on THREE dimensions (0.0 to 1.0 each):

1. GROUNDEDNESS (threshold: {groundedness_threshold})
   Is every factual claim in the answer explicitly supported
   by the provided excerpts?
   - 1.0 = every claim is directly supported
   - 0.5 = some claims are supported, some are not
   - 0.0 = claims are not supported or contradict the excerpts

2. CITATION_ACCURACY (threshold: {citation_threshold})
   Are the article numbers cited in the answer correct?
   Do they match the actual articles in the excerpts?
   - 1.0 = all citations are correct
   - 0.5 = some citations are correct
   - 0.0 = citations are wrong or missing

3. RELEVANCE (threshold: {relevance_threshold})
   Does the answer directly address what was asked?
   - 1.0 = answer fully addresses the question
   - 0.5 = answer partially addresses the question
   - 0.0 = answer does not address the question

CONTRACT EXCERPTS:
{chunks}

QUESTION: {query}

ANSWER TO EVALUATE:
{answer}

Respond with ONLY a JSON object in this exact format:
{{
  "groundedness_score": <float 0.0-1.0>,
  "citation_score": <float 0.0-1.0>,
  "relevance_score": <float 0.0-1.0>,
  "groundedness_feedback": "<what is wrong with groundedness, or 'OK'>",
  "citation_feedback": "<what citations are wrong, or 'OK'>",
  "relevance_feedback": "<why answer misses the question, or 'OK'>"
}}"""


# ─── Score Parser ─────────────────────────────────────────────

def parse_critic_response(response_text: str) -> dict:
    """
    Parse the LLM's JSON critique response.

    Handles cases where the LLM wraps JSON in markdown
    code blocks or adds extra text around it.

    Args:
        response_text: Raw LLM response string

    Returns:
        Dict with score fields, or default scores on parse failure

    Example return:
        {
            "groundedness_score": 0.9,
            "citation_score": 0.85,
            "relevance_score": 1.0,
            "groundedness_feedback": "OK",
            "citation_feedback": "OK",
            "relevance_feedback": "OK"
        }
    """
    # Strip markdown code blocks if present
    text = response_text.strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Find JSON object in the text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        data = json.loads(text)

        def safe_float(val, default=0.5):
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                return default

        return {
            "groundedness_score": safe_float(
                data.get("groundedness_score", 0.5)
            ),
            "citation_score": safe_float(
                data.get("citation_score", 0.5)
            ),
            "relevance_score": safe_float(
                data.get("relevance_score", 0.5)
            ),
            "groundedness_feedback": str(
                data.get("groundedness_feedback", "")
            ),
            "citation_feedback": str(
                data.get("citation_feedback", "")
            ),
            "relevance_feedback": str(
                data.get("relevance_feedback", "")
            ),
        }

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"Failed to parse critic JSON response: {e} | "
            f"raw: {response_text[:200]}"
        )
        # Return conservative defaults so the pipeline doesn't crash.
        # Scores of 0.5 will likely trigger regeneration, which is
        # safer than accepting a potentially bad answer.
        return {
            "groundedness_score": 0.5,
            "citation_score": 0.5,
            "relevance_score": 0.5,
            "groundedness_feedback": "Could not parse critic response.",
            "citation_feedback": "Could not parse critic response.",
            "relevance_feedback": "Could not parse critic response.",
        }


# ─── Feedback Composer ────────────────────────────────────────

def compose_critique_feedback(scores: dict) -> str:
    """
    Combine per-dimension feedback into a single actionable
    string passed back to the Reasoning Agent for regeneration.

    Only includes feedback for dimensions that failed.

    Args:
        scores: Dict returned by parse_critic_response

    Returns:
        Multi-line string summarising what needs to improve,
        or empty string if all dimensions passed.
    """
    issues = []

    if scores["groundedness_score"] < settings.GROUNDEDNESS_THRESHOLD:
        fb = scores["groundedness_feedback"]
        if fb and fb.upper() != "OK":
            issues.append(f"Groundedness: {fb}")

    if scores["citation_score"] < settings.CITATION_THRESHOLD:
        fb = scores["citation_feedback"]
        if fb and fb.upper() != "OK":
            issues.append(f"Citation: {fb}")

    if scores["relevance_score"] < settings.RELEVANCE_THRESHOLD:
        fb = scores["relevance_feedback"]
        if fb and fb.upper() != "OK":
            issues.append(f"Relevance: {fb}")

    return "\n".join(issues)


# ─── Critic Agent ─────────────────────────────────────────────

class CriticAgent:
    """
    LLM-as-judge quality gate for LexMind answers.

    Scores the Reasoning Agent's answer on:
      - Groundedness: every claim supported by retrieved chunks
      - Citation accuracy: cited article numbers are correct
      - Relevance: answer addresses what was actually asked

    If any score falls below its threshold, sets
    critique_passed=False and provides specific feedback
    so the Reasoning Agent can improve on regeneration.

    Usage as a LangGraph node:
        agent = CriticAgent()
        state = agent.run(state)
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain chat model. If None, uses get_llm()
                 from llm_client.py (reads provider from settings).
        """
        self._llm = llm
        logger.info("CriticAgent initialized")

    @property
    def llm(self):
        """Lazy-load LLM on first use."""
        if self._llm is None:
            self._llm = get_llm()
            logger.info(
                f"LLM loaded for CriticAgent: "
                f"{settings.LLM_PROVIDER}"
            )
        return self._llm

    def run(self, state: LexMindState) -> LexMindState:
        """
        LangGraph node function.

        Reads:  state['query'], state['answer'],
                state['retrieved_chunks'], state['regeneration_count']
        Writes: state['groundedness_score'], state['citation_score'],
                state['relevance_score'], state['critique_feedback'],
                state['critique_passed'], state['final_answer']

        Args:
            state: Current LexMindState

        Returns:
            Updated LexMindState with critique scores filled in
        """
        query = state["query"]
        answer = state.get("answer") or ""
        chunks = state.get("retrieved_chunks") or []
        regeneration_count = state.get("regeneration_count") or 0

        logger.info(
            f"CriticAgent running | "
            f"regeneration={regeneration_count} | "
            f"answer_length={len(answer)}"
        )

        if not answer:
            logger.warning("No answer to critique — marking as failed")
            state["groundedness_score"] = 0.0
            state["citation_score"] = 0.0
            state["relevance_score"] = 0.0
            state["critique_feedback"] = "No answer was generated."
            state["critique_passed"] = False
            return state

        try:
            scores = self._evaluate(query, answer, chunks)

            state["groundedness_score"] = scores["groundedness_score"]
            state["citation_score"] = scores["citation_score"]
            state["relevance_score"] = scores["relevance_score"]

            passed = self._all_scores_pass(scores)
            state["critique_passed"] = passed

            if passed:
                state["critique_feedback"] = None
                state["final_answer"] = answer
                logger.info(
                    f"Critique PASSED | "
                    f"G={scores['groundedness_score']:.2f} "
                    f"C={scores['citation_score']:.2f} "
                    f"R={scores['relevance_score']:.2f}"
                )
            else:
                feedback = compose_critique_feedback(scores)
                state["critique_feedback"] = feedback
                logger.info(
                    f"Critique FAILED | "
                    f"G={scores['groundedness_score']:.2f} "
                    f"C={scores['citation_score']:.2f} "
                    f"R={scores['relevance_score']:.2f} | "
                    f"feedback: {feedback[:100]}"
                )

                # Accept the answer anyway if max regenerations reached
                if regeneration_count >= settings.MAX_REGENERATIONS:
                    logger.warning(
                        f"Max regenerations ({settings.MAX_REGENERATIONS}) "
                        f"reached — accepting answer despite failed critique"
                    )
                    state["final_answer"] = answer

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            state["groundedness_score"] = 0.0
            state["citation_score"] = 0.0
            state["relevance_score"] = 0.0
            state["critique_feedback"] = f"Critique error: {str(e)}"
            state["critique_passed"] = False
            state["error"] = f"Critique failed: {str(e)}"

        return state

    def _evaluate(
        self,
        query: str,
        answer: str,
        chunks: list,
    ) -> dict:
        """
        Call LLM to score the answer.

        Returns:
            Dict from parse_critic_response
        """
        chunks_text = format_chunks_for_prompt(chunks)

        prompt = CRITIC_PROMPT.format(
            query=query,
            answer=answer,
            chunks=chunks_text,
            groundedness_threshold=settings.GROUNDEDNESS_THRESHOLD,
            citation_threshold=settings.CITATION_THRESHOLD,
            relevance_threshold=settings.RELEVANCE_THRESHOLD,
        )

        response = self.llm.invoke(prompt)
        return parse_critic_response(response.content)

    def _all_scores_pass(self, scores: dict) -> bool:
        """
        Return True if all three scores meet their thresholds.
        """
        return (
            scores["groundedness_score"] >= settings.GROUNDEDNESS_THRESHOLD
            and scores["citation_score"] >= settings.CITATION_THRESHOLD
            and scores["relevance_score"] >= settings.RELEVANCE_THRESHOLD
        )


# ─── LangGraph Node Function ──────────────────────────────────

_critic_agent: Optional[CriticAgent] = None


def get_critic_agent() -> CriticAgent:
    """Get or create the module-level CriticAgent instance."""
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent


def critic_node(state: LexMindState) -> LexMindState:
    """
    LangGraph node function for the Critic Agent.

    Usage in graph:
        graph.add_node("critic_agent", critic_node)
    """
    return get_critic_agent().run(state)
