"""
agents/router.py – Query router: classify query → select agent.

Two-stage routing:
  1. Rule-based fast path  (zero latency, handles obvious patterns)
  2. LLM classifier        (handles ambiguous queries)

The router populates state["selected_agent"] with one of:
  "qa" | "summarizer" | "comparator"
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from graph.state import ResearchState
from prompts.agent_prompts import ROUTER_HUMAN_PROMPT, ROUTER_SYSTEM_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)

AgentLabel = Literal["qa", "summarizer", "comparator"]

# ─── Rule-based fast path ─────────────────────────────────────────────────────

_SUMMARIZER_PATTERNS = [
    r"\b(summarize|summarise|summary|overview|outline|abstract)\b",
    r"\bkey (contributions?|findings?|points?|takeaways?)\b",
    r"\bwhat (is|are) (this|the) paper (about|on)\b",
    r"\bgive me (an? )?(summary|overview|digest)\b",
]

_COMPARATOR_PATTERNS = [
    r"\b(compare|contrast|difference|versus|vs\.?|between)\b",
    r"\bhow (do|does|did) .{3,40} differ\b",
    r"\bwhich (paper|approach|method|model) (is|performs|works)\b",
]

_SUMMARIZER_RE = re.compile(
    "|".join(_SUMMARIZER_PATTERNS), re.IGNORECASE
)
_COMPARATOR_RE = re.compile(
    "|".join(_COMPARATOR_PATTERNS), re.IGNORECASE
)


def _rule_based_route(query: str) -> AgentLabel | None:
    """
    Fast rule-based routing.  Returns None if no rule matches clearly,
    deferring to the LLM classifier.
    """
    if _COMPARATOR_RE.search(query):
        return "comparator"
    if _SUMMARIZER_RE.search(query):
        return "summarizer"
    return None  # ambiguous – escalate to LLM


def _llm_route(query: str) -> AgentLabel:
    """LLM-based routing for ambiguous queries."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
        # BUG FIX: max_tokens=5 is dangerously low. The model is expected to
        # return one of "qa", "summarizer", or "comparator".  "summarizer" is
        # 10 characters / ~2-3 tokens, but some tokenisers may split it into
        # more tokens depending on BPE merges. Raised to 16 to ensure the full
        # label always fits, preventing silent truncation to an empty or partial
        # string that falls through to the "qa" fallback.
        max_tokens=16,
    )
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=ROUTER_HUMAN_PROMPT.format(query=query)),
    ]
    response = llm.invoke(messages)

    # BUG FIX: response.content may contain surrounding whitespace, newlines,
    # or punctuation (e.g. '"summarizer"'). Strip non-alpha characters to
    # safely extract the label.
    raw = response.content.strip().lower()
    # Remove any surrounding quotes or punctuation
    label = re.sub(r"[^a-z]", "", raw)

    if label in ("qa", "summarizer", "comparator"):
        return label  # type: ignore[return-value]

    log.warning("router_llm_unexpected_label", label=raw, fallback="qa")
    return "qa"


# ─── Public node function ─────────────────────────────────────────────────────

def run_router(state: ResearchState) -> ResearchState:
    """
    LangGraph node: classify the query and set state["selected_agent"].

    Routing priority:
      1. Rule-based (if confident match)
      2. LLM classifier (fallback)
    """
    query = state.get("query", "")

    # BUG FIX: If the query is empty, skip routing and default to "qa" rather
    # than calling the LLM with an empty string, which wastes a token budget
    # and may return an unexpected label.
    if not query.strip():
        log.warning("router_empty_query", fallback="qa")
        return {**state, "selected_agent": "qa"}

    label = _rule_based_route(query)
    source = "rules"

    if label is None:
        label = _llm_route(query)
        source = "llm"

    log.info(
        "routing_decision",
        query=query[:80],
        agent=label,
        source=source,
    )

    return {**state, "selected_agent": label}
