"""
app.py – Streamlit frontend for PaperMind.

Tabs:
  💬 Chat     – Conversational interface (auto-routes to correct agent)
  📄 Ingest   – Upload PDF research papers
  📝 Summarize – One-click structured summaries
  ⚖️  Compare  – Side-by-side paper comparison
  📊 Status   – Index health and document list
"""
from __future__ import annotations

# BUG FIX: `import time` was present but never used — removed unused import.
from typing import Optional

import requests
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 PaperMind")
    st.caption("Multi-Agent RAG System")
    st.divider()

    # BUG FIX: @st.cache_data must NOT be defined inside a `with` block or
    # any conditional/loop scope — Streamlit re-defines it on every rerun,
    # causing a DuplicateWidgetID / cache key collision. Moved to module scope
    # (defined once below), and only called here.
    doc_ids = []  # populated after the function is defined at module scope

    st.subheader("📚 Indexed Documents")
    # Populated after get_indexed_docs() is defined below
    st.divider()
    st.caption("Powered by LangGraph + FAISS + GPT-4o")


# ─── Cached helper (module-level — must NOT be nested) ────────────────────────

# BUG FIX: Moved @st.cache_data decorated function out of the `with st.sidebar`
# block. Defining a cached function inside a context manager causes Streamlit to
# re-register it on every script rerun, leaking memory and raising warnings.
@st.cache_data(ttl=10)
def get_indexed_docs() -> list[str]:
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=3)
        r.raise_for_status()  # BUG FIX: was silently ignoring HTTP errors
        return r.json().get("doc_ids", [])
    except Exception:
        return []


# Populate sidebar doc list now that the function exists
with st.sidebar:
    doc_ids = get_indexed_docs()
    if doc_ids:
        for d in doc_ids:
            st.success(f"✅ {d}")
    else:
        st.info("No documents indexed yet. Use the Ingest tab.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict, timeout: int = 60) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE}/{endpoint}",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Is `uvicorn api.main:app` running?")
    except requests.exceptions.HTTPError as e:
        # BUG FIX: e.response can be None (e.g. on 5xx with no JSON body),
        # which causes AttributeError. Added a guard.
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API Error: {detail}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def render_response_card(data: dict) -> None:
    """Render agent response with metadata badges."""
    agent = data.get("agent", data.get("selected_agent", "unknown"))
    latency = data.get("latency_ms", 0)
    cached = data.get("cached", False)

    badge_color = {"qa": "🟦", "summarizer": "🟩", "comparator": "🟨"}.get(
        agent, "⬜"
    )
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.caption(f"{badge_color} Agent: **{agent.upper()}**")
    with col2:
        st.caption(f"⏱ {latency}ms")
    with col3:
        if cached:
            st.caption("⚡ Cached")

    response_text = data.get("response") or data.get("summary") or data.get("comparison", "")
    st.markdown(response_text)

    citations = data.get("citations", [])
    if citations:
        with st.expander(f"📎 Sources ({len(citations)})"):
            for c in citations:
                st.caption(c)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_chat, tab_ingest, tab_summarize, tab_compare, tab_status = st.tabs(
    ["💬 Chat", "📄 Ingest", "📝 Summarize", "⚖️ Compare", "📊 Status"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 Research Chat")
    st.caption(
        "Ask any question. The system automatically routes to the correct agent."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    all_docs = get_indexed_docs()
    scoped_docs = st.multiselect(
        "Scope to documents (optional — leave empty to search all)",
        options=all_docs,
        default=[],
        key="chat_scope",
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_response_card(msg["data"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your research papers…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # BUG FIX: The history slice `[:-1]` excluded the message just appended
        # above, which is correct — but assistant messages stored as {"data": …}
        # have no "content" key, so `.get("content")` returns None and the
        # fallback `.get("data", {}).get("response", "")` is needed. This was
        # already handled, but added an explicit str() cast to guard against
        # None leaking into the API payload if response is None.
        history = [
            {
                "role": m["role"],
                "content": str(
                    m.get("content") or m.get("data", {}).get("response", "")
                ),
            }
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                data = api_post(
                    "ask",
                    {
                        "query": prompt,
                        # BUG FIX: Pass None (not empty list) when no scope is
                        # selected so the API correctly searches all documents.
                        "doc_ids": scoped_docs if scoped_docs else None,
                        "conversation_history": history,
                    },
                )
            if data:
                render_response_card(data)
                st.session_state.messages.append(
                    {"role": "assistant", "data": data}
                )

    if st.button("🗑️ Clear conversation", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("📄 Ingest Research Papers")
    st.caption("Upload PDFs to add them to the knowledge base.")

    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
    )

    custom_id = st.text_input(
        "Custom document ID (optional, single file only)",
        placeholder="e.g. attention_is_all_you_need",
    )

    if st.button("🚀 Ingest", disabled=not uploaded_files):
        for uploaded in uploaded_files:
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    # BUG FIX: uploaded.read() is only valid once per rerun.
                    # Store bytes in a variable before building the files dict
                    # so the same buffer is not read twice if the dict is
                    # reconstructed (e.g. during retry logic).
                    pdf_bytes = uploaded.read()
                    files = {"file": (uploaded.name, pdf_bytes, "application/pdf")}
                    data_form = {
                        "doc_id": custom_id if custom_id and len(uploaded_files) == 1 else ""
                    }
                    r = requests.post(
                        f"{API_BASE}/ingest",
                        files=files,
                        data=data_form,
                        timeout=120,
                    )
                    r.raise_for_status()
                    result = r.json()
                    st.success(
                        f"✅ **{result['filename']}** ingested: "
                        f"{result['pages_parsed']} pages, "
                        f"{result['chunks_indexed']} chunks, "
                        f"{result['latency_ms']}ms"
                    )
                except requests.exceptions.HTTPError as e:
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = str(e)
                    st.error(f"Failed: {detail}")
                except Exception as e:
                    st.error(f"Error: {e}")
        get_indexed_docs.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_summarize:
    st.header("📝 Structured Summary")
    st.caption("Generate a detailed structured summary for any indexed paper.")

    sum_docs = get_indexed_docs()
    if not sum_docs:
        st.warning("No documents indexed. Please ingest papers first.")
    else:
        selected_doc = st.selectbox("Select document", sum_docs, key="sum_doc")
        style = st.radio(
            "Summary style",
            ["bullet", "abstract"],
            horizontal=True,
            help="Bullet = structured sections; Abstract = prose paragraph",
        )

        if st.button("📝 Generate Summary"):
            with st.spinner("Summarising…"):
                data = api_post(
                    "summarize",
                    {"doc_id": selected_doc, "style": style},
                )
            if data:
                st.subheader(f"Summary: {selected_doc}")
                render_response_card(
                    {
                        "agent": "summarizer",
                        "response": data.get("summary", ""),
                        "citations": data.get("citations", []),
                        "latency_ms": data.get("latency_ms", 0),
                    }
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("⚖️ Compare Papers")
    st.caption("Select 2-4 papers for a structured side-by-side comparison.")

    cmp_docs = get_indexed_docs()
    if len(cmp_docs) < 2:
        st.warning("Ingest at least 2 papers to use comparison.")
    else:
        selected_cmp = st.multiselect(
            "Select 2-4 papers",
            cmp_docs,
            max_selections=4,
            key="cmp_docs",
        )
        focus = st.text_input(
            "Focus area (optional)",
            placeholder="e.g. methodology, results, efficiency",
        )

        if st.button("⚖️ Compare", disabled=len(selected_cmp) < 2):
            with st.spinner("Comparing papers…"):
                data = api_post(
                    "compare",
                    {
                        "doc_ids": selected_cmp,
                        "focus": focus or None,
                    },
                )
            if data:
                render_response_card(
                    {
                        "agent": "comparator",
                        "response": data.get("comparison", ""),
                        "citations": data.get("citations", []),
                        "latency_ms": data.get("latency_ms", 0),
                    }
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – STATUS
# ══════════════════════════════════════════════════════════════════════════════
with tab_status:
    st.header("📊 System Status")

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        # BUG FIX: After clearing cache the page must rerun for the fresh data
        # to render. Without st.rerun() the stale (now-cleared) data is shown.
        st.rerun()

    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success("🟢 API is running")

        col1, col2, col3 = st.columns(3)
        col1.metric("Indexed Documents", len(health.get("indexed_docs", [])))
        col2.metric("BM25 Corpus Size", health.get("bm25_corpus_size", 0))
        col3.metric("Version", health.get("version", "—"))

        if health.get("indexed_docs"):
            st.subheader("Indexed Document IDs")
            for doc in health["indexed_docs"]:
                st.code(doc)
    except Exception:
        st.error(
            "🔴 API is not reachable. Start it with: `uvicorn api.main:app --reload`"
        )
