"""
Reasoning Agent for LexMind
Generates grounded, cited answers from retrieved chunks
using the configured LLM provider.
"""

import re
from typing import List, Optional
from loguru import logger

from src.config import settings
from src.graph.state import LexMindState
from src.llm_client import get_llm
from src.ingestion.vector_store import SearchResult


# ─── Prompt Templates ─────────────────────────────────────────

FACTUAL_PROMPT = """You are a precise legal research assistant.
Answer the question using ONLY the contract excerpts provided below.

STRICT RULES:
1. Use ONLY information from the provided excerpts.
2. Do NOT use any external legal knowledge or assumptions.
3. Cite BOTH the article number AND document name in square brackets.
   The document name is shown in parentheses in each excerpt header.
   Example: "Notice must be 30 days [Article 75 — Saudi Labor Law]."
   Example: "A company needs 2 directors [Article 12 — Companies Law]."
4. If the excerpts do not contain enough information to answer,
   say: "The provided excerpts do not contain information about this."
5. Be concise and direct.

CONTRACT EXCERPTS:
{chunks}

QUESTION: {query}

ANSWER (with citations):"""


ANALYTICAL_PROMPT = """You are a careful legal analyst.
Analyse the question using ONLY the contract excerpts below.

STRICT RULES:
1. Use ONLY information from the provided excerpts.
2. Do NOT use any external legal knowledge or assumptions.
3. Cite BOTH the article number AND document name in square brackets.
   Example: "The employer is liable [Article 34 — Labor Law]."
4. Structure your analysis with clear points.
5. If evidence is insufficient, state this explicitly.

CONTRACT EXCERPTS:
{chunks}

QUESTION: {query}

ANALYSIS (with citations):"""


COMPARISON_PROMPT = """You are a meticulous legal researcher.
Compare the elements described in the question using ONLY
the contract excerpts provided below.

STRICT RULES:
1. Use ONLY information from the provided excerpts.
2. Do NOT use any external legal knowledge or assumptions.
3. Cite BOTH the article number AND document name in square brackets.
   Example: "Under the Labor Law [Article 75], notice is 30 days."
4. Structure your comparison clearly (similarities / differences).
5. Only compare what the excerpts explicitly state.

CONTRACT EXCERPTS:
{chunks}

QUESTION: {query}

COMPARISON (with citations):"""


SUMMARISATION_PROMPT = """You are a thorough legal summariser.
Summarise the requested section using ONLY the contract
excerpts provided below.

STRICT RULES:
1. Use ONLY information from the provided excerpts.
2. Do NOT add interpretation beyond what is stated.
3. Cite BOTH the article number AND document name in square brackets.
   Example: "Workers get annual leave [Article 109 — Labor Law]."
4. Cover all key points from the excerpts.
5. Use clear, structured bullet points.

CONTRACT EXCERPTS:
{chunks}

REQUEST: {query}

SUMMARY (with citations):"""


REGENERATION_PROMPT = """You are a precise legal research assistant.
Your previous answer had quality issues. Please improve it.

PREVIOUS ANSWER:
{previous_answer}

QUALITY ISSUES IDENTIFIED:
{critique_feedback}

CONTRACT EXCERPTS (use ONLY these):
{chunks}

STRICT RULES:
1. Use ONLY information from the provided excerpts.
2. Do NOT use any external legal knowledge.
3. Cite the article number in square brackets for EVERY claim.
4. Directly address the quality issues listed above.

QUESTION: {query}

IMPROVED ANSWER (with citations):"""


# ─── Prompt Selector ─────────────────────────────────────────

PROMPT_TEMPLATES = {
    "factual":       FACTUAL_PROMPT,
    "analytical":    ANALYTICAL_PROMPT,
    "comparison":    COMPARISON_PROMPT,
    "summarisation": SUMMARISATION_PROMPT,
}


def get_prompt_template(query_type: str) -> str:
    """Select prompt template based on query type.
    Falls back to factual prompt for unknown types.
    """
    return PROMPT_TEMPLATES.get(query_type, FACTUAL_PROMPT)


# ─── Citation Parser ──────────────────────────────────────────

def extract_citations(answer_text: str) -> List[str]:
    """
    Parse citations from answer text.
    Handles both formats:
      [Article 75]
      [Article 75 - Saudi Labor Law]
      [Article 75 — Saudi Labor Law]
    """
    full_pattern = (
        r'\[((?:Article|Section|Clause|Schedule|Part)'
        r'\s+[\dIVXivx]+(?:\.\d+)*'
        r'(?:\s*[—–-]+\s*[^\]]+)?)\]'
    )
    matches = re.findall(full_pattern, answer_text, re.IGNORECASE)

    citations = []
    for match in matches:
        match = match.strip()
        for sep in [' — ', ' – ', ' - ', '—', '–']:
            if sep in match:
                parts = match.split(sep, 1)
                article_part = parts[0].strip()
                doc_part = parts[1].strip()
                words = article_part.split()
                if words:
                    words[0] = words[0].capitalize()
                citations.append(f"{' '.join(words)} — {doc_part}")
                break
        else:
            words = match.split()
            if words:
                words[0] = words[0].capitalize()
            citations.append(' '.join(words))

    return sorted(list(set(citations)))


# ─── Chunk Formatter ─────────────────────────────────────────

def format_chunks_for_prompt(chunks: List[SearchResult]) -> str:
    """
    Format retrieved chunks into a readable context block.
    Each chunk is labelled with its article ref AND document title
    so the LLM can cite both in its answer.

    Example label: [Excerpt 1 — Article 75 (Saudi Labor Law)]
    """
    if not chunks:
        return "No relevant excerpts found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        article = chunk.article_ref or f"Chunk {chunk.chunk_index}"
        doc_title = chunk.metadata.get('doc_title', '') if chunk.metadata else ''

        if doc_title:
            label = f"{article} ({doc_title})"
        else:
            label = article

        parts.append(f"[Excerpt {i} — {label}]\n{chunk.text}")

    return "\n\n".join(parts)


# ─── Reasoning Agent ─────────────────────────────────────────

class ReasoningAgent:
    """
    Generates grounded, cited answers from retrieved chunks.

    Uses the LLM configured in settings (Google/OpenAI/
    Anthropic/Azure) with prompt templates that enforce
    grounding and citation requirements.

    On regeneration (when Critic rejects), includes the
    previous answer and critique feedback in the prompt
    so the LLM can specifically address the issues.

    Usage as a LangGraph node:
        agent = ReasoningAgent()
        state = agent.run(state)
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain chat model. If None, uses get_llm()
                 from llm_client.py (reads provider from settings).
        """
        self._llm = llm
        logger.info("ReasoningAgent initialized")

    @property
    def llm(self):
        """Lazy-load LLM on first use."""
        if self._llm is None:
            self._llm = get_llm()
            logger.info(
                f"LLM loaded for ReasoningAgent: "
                f"{settings.LLM_PROVIDER}"
            )
        return self._llm

    def run(self, state: LexMindState) -> LexMindState:
        """
        LangGraph node function.

        Reads:  state['query'], state['retrieved_chunks'],
                state['query_type'], state['regeneration_count'],
                state['critique_feedback'] (on regeneration),
                state['conversation_history'] (from session)
        Writes: state['answer'], state['citations'],
                state['reasoning_prompt']

        Args:
            state: Current LexMindState

        Returns:
            Updated LexMindState with answer filled in
        """
        query = state["query"]
        conversation_history = state.get("conversation_history") or ""
        chunks = state.get("retrieved_chunks") or []
        query_type = state.get("query_type") or "factual"
        regeneration_count = state.get("regeneration_count") or 0
        critique_feedback = state.get("critique_feedback")
        previous_answer = state.get("answer")

        logger.info(
            f"ReasoningAgent running | "
            f"query_type={query_type} | "
            f"chunks={len(chunks)} | "
            f"regeneration={regeneration_count}"
        )

        if not chunks:
            logger.warning("No chunks provided — cannot generate answer")
            state["answer"] = (
                "I was unable to find relevant information in the "
                "document to answer this question."
            )
            state["citations"] = []
            state["reasoning_prompt"] = ""
            return state

        try:
            answer, prompt = self._generate(
                query=query,
                chunks=chunks,
                query_type=query_type,
                is_regeneration=regeneration_count > 0,
                previous_answer=previous_answer,
                critique_feedback=critique_feedback,
                conversation_history=conversation_history,
            )

            citations = extract_citations(answer)

            state["answer"] = answer
            state["citations"] = citations
            state["reasoning_prompt"] = prompt

            logger.info(
                f"Answer generated | "
                f"length={len(answer)} chars | "
                f"citations={citations}"
            )

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            state["answer"] = f"Error generating answer: {str(e)}"
            state["citations"] = []
            state["reasoning_prompt"] = ""
            state["error"] = f"Reasoning failed: {str(e)}"

        return state

    def _generate(
        self,
        query: str,
        chunks: List[SearchResult],
        query_type: str,
        is_regeneration: bool,
        previous_answer: Optional[str],
        critique_feedback: Optional[str],
        conversation_history: str = "",
    ) -> tuple[str, str]:
        """
        Build prompt and call LLM.

        Returns:
            Tuple of (answer_text, prompt_used)
        """
        chunks_text = format_chunks_for_prompt(chunks)

        if is_regeneration and previous_answer and critique_feedback:
            prompt = REGENERATION_PROMPT.format(
                query=query,
                chunks=chunks_text,
                previous_answer=previous_answer,
                critique_feedback=critique_feedback,
            )
        elif conversation_history:
            template = get_prompt_template(query_type)
            base_prompt = template.format(
                query=query,
                chunks=chunks_text,
            )
            prompt = (
                "CONVERSATION CONTEXT (for coherence only — "
                "do NOT treat as source of facts):\n"
                f"{conversation_history}\n\n"
                "=" * 50 + "\n\n"
                f"{base_prompt}"
            )
        else:
            template = get_prompt_template(query_type)
            prompt = template.format(
                query=query,
                chunks=chunks_text,
            )

        response = self.llm.invoke(prompt)
        answer = response.content.strip()
        return answer, prompt


# ─── LangGraph Node Function ──────────────────────────────────

_reasoning_agent: Optional[ReasoningAgent] = None


def get_reasoning_agent() -> ReasoningAgent:
    """Get or create the module-level ReasoningAgent instance."""
    global _reasoning_agent
    if _reasoning_agent is None:
        _reasoning_agent = ReasoningAgent()
    return _reasoning_agent


def reasoning_node(state: LexMindState) -> LexMindState:
    """
    LangGraph node function for the Reasoning Agent.

    Usage in graph:
        graph.add_node("reasoning_agent", reasoning_node)
    """
    return get_reasoning_agent().run(state)
