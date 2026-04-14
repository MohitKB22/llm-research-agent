"""
graph/orchestrator.py – LangGraph multi-agent graph definition.

Graph topology:
  [rewrite_query]
       ↓
  [retrieve_docs]
       ↓
  [route_query]
       ↓ (conditional edge)
  ┌────┴──────────────┐
  [qa]  [summarizer]  [comparator]
  └────┬──────────────┘
       ↓
     END

Every node is a pure function (ResearchState → ResearchState).
The graph is compiled once and reused across all requests.
"""
from __future__ import annotations

import time
from typing import Optional

from langgraph.graph import END, StateGraph

from agents.comparator import run_comparator_agent
from agents.qa_agent import run_qa_agent
from agents.router import run_router
from agents.summarizer import run_summarizer_agent
from graph.state import ResearchState
from pipeline.query_rewriter import RewriteMode, rewrite_query
from retriever.hybrid_retriever import get_retriever
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Intermediate nodes ───────────────────────────────────────────────────────

def node_rewrite_query(state: ResearchState) -> ResearchState:
    """Rewrite the raw query for better retrieval."""
    query = state.get("query", "")
    history = state.get("conversation_history", [])

    context = ""
    if history:
        recent = history[-4:]  # last 2 turns
        # BUG FIX: `m.content` assumes all items are LangChain BaseMessage
        # objects. When `conversation_history` is populated from raw dicts
        # (e.g. during testing or direct run_pipeline calls), this raises
        # AttributeError. Added a safe accessor that handles both dicts and
        # message objects.
        lines = []
        for i, m in enumerate(recent):
            if hasattr(m, "content"):
                content = m.content[:150]
            else:
                content = str(m.get("content", ""))[:150]
            role = "User" if i % 2 == 0 else "AI"
            lines.append(f"{role}: {content}")
        context = "\n".join(lines)

    rewritten = rewrite_query(
        query=query,
        mode=RewriteMode.SIMPLE,
        conversation_context=context,
    )
    return {**state, "rewritten_query": rewritten}


def node_retrieve_docs(state: ResearchState) -> ResearchState:
    """Run hybrid retrieval for the (rewritten) query."""
    query = state.get("rewritten_query") or state.get("query", "")
    doc_ids = state.get("doc_ids")

    t0 = time.perf_counter()
    docs = get_retriever().retrieve(
        query=query,
        filter_doc_ids=doc_ids,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    log.info(
        "retrieval_node_complete",
        docs_retrieved=len(docs),
        latency_ms=latency_ms,
    )

    meta = {**(state.get("metadata") or {}), "retrieval_latency_ms": latency_ms}
    return {**state, "retrieved_docs": docs, "metadata": meta}


def _route_to_agent(state: ResearchState) -> str:
    """Conditional edge: return the name of the next node."""
    # BUG FIX: If selected_agent is an empty string (initial state value) the
    # conditional edge raises a KeyError because "" is not in the path_map.
    # Defaulting to "qa" explicitly guards against this.
    agent = state.get("selected_agent") or "qa"
    if agent not in ("qa", "summarizer", "comparator"):
        log.warning("unknown_agent_label", label=agent, fallback="qa")
        return "qa"
    return agent


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns a compiled graph ready for .invoke() calls.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve_docs", node_retrieve_docs)
    graph.add_node("route_query", run_router)
    graph.add_node("qa", run_qa_agent)
    graph.add_node("summarizer", run_summarizer_agent)
    graph.add_node("comparator", run_comparator_agent)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "route_query")

    graph.add_conditional_edges(
        source="route_query",
        path=_route_to_agent,
        path_map={
            "qa": "qa",
            "summarizer": "summarizer",
            "comparator": "comparator",
        },
    )

    graph.add_edge("qa", END)
    graph.add_edge("summarizer", END)
    graph.add_edge("comparator", END)

    return graph.compile()


# BUG FIX: Typed the singleton as Optional[...] so type checkers understand
# it starts as None and becomes a compiled graph after first call.
_graph: Optional[object] = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("langgraph_compiled")
    return _graph


# ─── Public entry point ───────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    doc_ids: Optional[list[str]] = None,
    conversation_history: Optional[list] = None,
) -> ResearchState:
    """
    Execute the full multi-agent pipeline for a user query.

    Args:
        query: Raw user question.
        doc_ids: Optional list of doc_ids to scope retrieval.
        conversation_history: Prior conversation as LangChain messages.

    Returns:
        Final ResearchState with agent_response, citations, metadata.
    """
    # BUG FIX: Validate that query is not empty before running the pipeline.
    # An empty query causes the rewriter and retriever to silently return
    # useless results instead of an actionable error.
    if not query or not query.strip():
        return {
            "query": query,
            "rewritten_query": "",
            "doc_ids": doc_ids,
            "selected_agent": "",
            "retrieved_docs": [],
            "agent_response": "Please provide a non-empty query.",
            "citations": [],
            "conversation_history": conversation_history or [],
            "error": "Empty query",
            "metadata": {},
        }

    initial_state: ResearchState = {
        "query": query,
        "rewritten_query": "",
        "doc_ids": doc_ids,
        "selected_agent": "",
        "retrieved_docs": [],
        "agent_response": "",
        "citations": [],
        "conversation_history": conversation_history or [],
        "error": None,
        "metadata": {"query": query},
    }

    t0 = time.perf_counter()
    try:
        final_state = get_graph().invoke(initial_state)
    except Exception as exc:
        log.error("pipeline_error", error=str(exc), query=query[:80])
        final_state = {
            **initial_state,
            "agent_response": (
                "An error occurred while processing your query. "
                f"Details: {exc}"
            ),
            "error": str(exc),
        }

    total_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "pipeline_complete",
        agent=final_state.get("selected_agent"),
        total_ms=total_ms,
    )

    meta = final_state.get("metadata") or {}
    meta["total_latency_ms"] = total_ms
    final_state["metadata"] = meta

    return final_state
