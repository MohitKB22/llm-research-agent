"""
api/routes.py – FastAPI route definitions.

Endpoints:
  POST /ingest          – Upload and index a PDF
  POST /ask             – General query (auto-routed to correct agent)
  POST /summarize       – Summarise a specific document
  POST /compare         – Compare multiple documents
  GET  /health          – System status and index stats
  GET  /documents       – List all indexed document IDs
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from langchain_core.messages import AIMessage, HumanMessage

from api.schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    IngestResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from cache.redis_cache import cache_get, cache_set
from config import settings
from graph.orchestrator import run_pipeline
from pipeline.ingestion import ingest_pdf
from retriever.bm25_store import get_bm25_store
from retriever.faiss_store import get_faiss_store
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


# ─── Helper ───────────────────────────────────────────────────────────────────

def _history_to_messages(
    raw: list[dict[str, str]] | None,
) -> list[HumanMessage | AIMessage]:
    """Convert [{role, content}] dicts to LangChain message objects."""
    if not raw:
        return []
    messages = []
    # BUG FIX: Comment said "keep last 5 turns" but the slice was raw[-10:],
    # which keeps 10 entries (10 half-turns = 5 full turns of user+AI).
    # This is actually correct for 5 full turns, but the comment was misleading.
    # Kept the slice as raw[-10:] and updated the comment.
    for turn in raw[-10:]:  # keep last 10 half-turns (5 full user+AI turns)
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


# ─── /ingest ──────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and index a PDF research paper",
    status_code=status.HTTP_201_CREATED,
)
async def ingest(
    file: UploadFile = File(..., description="PDF file to ingest"),
    doc_id: str = Form(
        default="",
        description="Optional custom document ID. Auto-generated if empty.",
    ),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    effective_doc_id = doc_id.strip() or (
        Path(file.filename).stem.lower().replace(" ", "_")
        + "_"
        + uuid.uuid4().hex[:6]
    )

    tmp_path = settings.documents_dir / f"_upload_{uuid.uuid4().hex}.pdf"
    try:
        content = await file.read()

        # BUG FIX: If the uploaded file is empty (0 bytes), ingest_pdf will
        # either raise an obscure parser error or produce 0 chunks. Catch it
        # early with a clear error message.
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        tmp_path.write_bytes(content)

        result = ingest_pdf(
            file_path=tmp_path,
            doc_id=effective_doc_id,
            save_original=True,
        )
    finally:
        # BUG FIX: The original code only unlinked tmp_path in the finally
        # block, but if tmp_path.write_bytes() raised (e.g. disk full),
        # tmp_path would not exist and the unlink guard `if tmp_path.exists()`
        # already handles that — so this is correct as-is. No change needed.
        if tmp_path.exists():
            tmp_path.unlink()

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion failed: {result.error}",
        )

    return IngestResponse(**result.__dict__)


# ─── /ask ─────────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question (auto-routed to QA / Summarizer / Comparator)",
)
async def ask(body: AskRequest) -> AskResponse:
    # BUG FIX: Validate that query is non-empty before hitting the cache or
    # pipeline. An empty string would be cached and returned as a valid
    # response on future empty queries.
    if not body.query or not body.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query must not be empty.",
        )

    cached = cache_get(body.query, body.doc_ids)
    if cached:
        return AskResponse(**cached, cached=True)

    history = _history_to_messages(body.conversation_history)

    t0 = time.perf_counter()
    state = run_pipeline(
        query=body.query,
        doc_ids=body.doc_ids,
        conversation_history=history,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if state.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=state["error"],
        )

    response_payload = {
        "query": body.query,
        "rewritten_query": state.get("rewritten_query", body.query),
        "agent": state.get("selected_agent", "unknown"),
        "response": state.get("agent_response", ""),
        "citations": state.get("citations", []),
        "latency_ms": latency_ms,
        "cached": False,
    }

    cache_set(body.query, response_payload, body.doc_ids)

    return AskResponse(**response_payload)


# ─── /summarize ───────────────────────────────────────────────────────────────

@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Generate a structured summary of an indexed document",
)
async def summarize(body: SummarizeRequest) -> SummarizeResponse:
    store = get_faiss_store()
    if body.doc_id not in store.stored_doc_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{body.doc_id}' is not indexed. "
                   "Please ingest it first via POST /ingest.",
        )

    # BUG FIX: The query injected doc_id directly into the text:
    #   "Give me a structured bullet-point summary of the paper with doc_id=foo"
    # The summarizer agent receives this as-is and may try to retrieve by that
    # literal string. Since doc_ids are already passed to run_pipeline as the
    # filter, the query should be a generic summarisation instruction.
    style_hint = (
        "Give me a structured bullet-point summary of this paper."
        if body.style == "bullet"
        else "Write a concise 150-word abstract-style summary of this paper."
    )

    t0 = time.perf_counter()
    state = run_pipeline(
        query=style_hint,          # BUG FIX: clean query without leaking doc_id
        doc_ids=[body.doc_id],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return SummarizeResponse(
        doc_id=body.doc_id,
        summary=state.get("agent_response", ""),
        citations=state.get("citations", []),
        latency_ms=latency_ms,
    )


# ─── /compare ─────────────────────────────────────────────────────────────────

@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare 2-4 indexed documents",
)
async def compare(body: CompareRequest) -> CompareResponse:
    # BUG FIX: Validate number of doc_ids before hitting the store, so the
    # error message is actionable rather than an internal KeyError.
    if len(body.doc_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least 2 document IDs are required for comparison.",
        )
    if len(body.doc_ids) > 4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At most 4 document IDs can be compared at once.",
        )

    store = get_faiss_store()
    missing = [d for d in body.doc_ids if d not in store.stored_doc_ids]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documents not found in index: {missing}. "
                   "Ingest them first via POST /ingest.",
        )

    focus = f" Focus on: {body.focus}." if body.focus else ""
    query = (
        f"Compare and contrast these papers: {', '.join(body.doc_ids)}.{focus}"
    )

    t0 = time.perf_counter()
    state = run_pipeline(
        query=query,
        doc_ids=body.doc_ids,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return CompareResponse(
        doc_ids=body.doc_ids,
        comparison=state.get("agent_response", ""),
        citations=state.get("citations", []),
        latency_ms=latency_ms,
    )


# ─── /health ──────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health and index statistics",
)
async def health() -> HealthResponse:
    faiss = get_faiss_store()
    bm25 = get_bm25_store()
    return HealthResponse(
        status="ok",
        indexed_docs=faiss.stored_doc_ids,
        bm25_corpus_size=bm25.corpus_size,
    )


# ─── /documents ───────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    summary="List all indexed document IDs",
)
async def list_documents() -> dict:
    return {"doc_ids": get_faiss_store().stored_doc_ids}
