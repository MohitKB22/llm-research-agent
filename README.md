# 🧠 LLM-Powered Research Co-Pilot

> An LLM-powered AI research co-pilot that autonomously gathers, analyzes, and summarizes information from uploaded research papers to assist in efficient knowledge discovery — grounded entirely in your documents with zero hallucination.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![GPT-4o](https://img.shields.io/badge/LLM-GPT--4o-purple)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🤖 How LLMs Power This System

PaperMind is built **entirely around Large Language Models**. Every intelligent action in the system — from understanding your question to writing the final answer — is driven by an LLM. Here is exactly where and how:

```
                        ┌─────────────────────────────┐
                        │        User Query            │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  1. LLM Query Rewriter       │  ← GPT rewrites query
                        │     (better retrieval)       │    for semantic clarity
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  2. Hybrid Retrieval         │  ← OpenAI Embeddings
                        │     FAISS + BM25 + Reranker  │    power dense search
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  3. LLM Router               │  ← GPT classifies intent
                        │     qa / summarizer /        │    (rule-based first,
                        │     comparator               │     LLM fallback)
                        └──────┬──────────────┬────────┘
                               │              │
               ┌───────────────▼─┐  ┌─────────▼───────┐  ┌──────────────────┐
               │ 4a. QA Agent    │  │ 4b. Summarizer  │  │ 4c. Comparator  │
               │ GPT-4o answers  │  │ GPT-4o produces │  │ GPT-4o builds   │
               │ factual queries │  │ 7-section report│  │ comparison table│
               └───────────────┬─┘  └─────────┬───────┘  └──────────┬──────┘
                               └──────────────┴──────────────────────┘
                                                      │
                                       ┌──────────────▼──────────────┐
                                       │  5. Cited Response           │
                                       │     Every claim linked to    │
                                       │     source chunk + page      │
                                       └─────────────────────────────┘
```

### LLM Usage Breakdown

| Step | Model | Role | Temperature |
|---|---|---|---|
| Query Rewriting | GPT-4o | Expands and clarifies raw user queries for better retrieval | 0.0 |
| Intent Routing | GPT-4o | Classifies query as `qa`, `summarizer`, or `comparator` | 0.0 |
| Dense Embedding | `text-embedding-3-small` | Converts paper chunks into 1536-dim vectors for FAISS | — |
| Reranking | `ms-marco-MiniLM-L-6-v2` | Cross-encoder scores query–document relevance | — |
| QA Agent | GPT-4o | Answers factual questions strictly from retrieved context | 0.0 |
| Summarizer Agent | GPT-4o | Produces structured 7-section paper summaries | 0.3 |
| Comparator Agent | GPT-4o | Builds side-by-side markdown comparison tables | 0.1 |

> **Zero hallucination guarantee:** All three agents operate under strict system prompts instructing the LLM to respond only using retrieved context chunks. If information is not in the documents, the model replies: *"Not available in the provided documents."*

---

## 📌 What It Does

| Feature | Description |
|---|---|
| 📄 **Ingest** | Upload PDFs — parsed, chunked, embedded, and indexed automatically |
| 💬 **Q&A** | Ask any question; LLM answers using only your documents |
| 📝 **Summarize** | LLM produces a structured 7-section summary or prose abstract |
| ⚖️ **Compare** | LLM generates a side-by-side comparison table for 2–4 papers |
| 🔗 **Citations** | Every LLM claim is linked to the exact source chunk and page number |
| ⚡ **Caching** | Redis (prod) / diskcache (dev) avoids redundant LLM API calls |

---

## 🏗️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (app.py)                 │
│   [Chat] [Ingest] [Summarize] [Compare] [Status]               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP  (FastAPI)
┌──────────────────────────▼──────────────────────────────────────┐
│                     API Layer  (api/)                           │
│   POST /ingest    POST /ask    POST /summarize    POST /compare │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              LangGraph Multi-Agent Orchestrator (graph/)        │
│                                                                 │
│  [rewrite_query] → [retrieve_docs] → [route_query]             │
│                                            │                    │
│                          ┌─────────────────┼──────────────┐    │
│                   [qa_agent]        [summarizer]  [comparator] │
│                    GPT-4o            GPT-4o        GPT-4o      │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Hybrid Retriever  (retriever/)                │
│                                                                 │
│   Dense (FAISS + OpenAI Embeddings)                            │
│   +  Sparse (BM25 keyword)                                     │
│   →  RRF Fusion  →  Cross-Encoder Reranker                     │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Vector Store  (data/)                         │
│          FAISS Index  +  BM25 Pickle  +  Redis Cache           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline — Step by Step

```
User uploads PDF
      │
      ▼
1. Parse        PyMuPDF → pdfplumber fallback
      │         Returns: PageContent(page_number, text)
      ▼
2. Chunk        RecursiveCharacterTextSplitter(size=512, overlap=64)
      │         Metadata preserved: doc_id, filename, page_number
      ▼
3. Embed        OpenAI text-embedding-3-small         ◄── LLM (Embeddings)
      │         Each chunk → 1536-dim vector
      ▼
4. Index        FAISS (dense) + BM25 (sparse) — persisted to disk

─ Query time ──────────────────────────────────────────────────────
      │
      ▼
5. Rewrite      GPT-4o cleans and expands raw query  ◄── LLM
      │
      ▼
6. Retrieve     FAISS top-10 + BM25 top-10
      │         → RRF fusion
      │         → MiniLM cross-encoder reranks to top-5
      ▼
7. Route        GPT-4o classifies intent             ◄── LLM
      │         → qa / summarizer / comparator
      ▼
8. Generate     Agent-specific prompt + context      ◄── LLM (GPT-4o)
      │         → grounded, cited answer
      ▼
9. Cite         Map LLM claims to [1],[2]… source chunks + page numbers
```

---

## 🤖 Agent Design

All agents are LangGraph **nodes** sharing a single `ResearchState`. Each agent has a dedicated LLM system prompt and temperature tuned to its task.

```python
class ResearchState(TypedDict):
    query: str
    rewritten_query: str
    doc_ids: Optional[list[str]]
    selected_agent: str              # "qa" | "summarizer" | "comparator"
    retrieved_docs: list[Document]
    agent_response: str
    citations: list[str]
    conversation_history: list[BaseMessage]
    metadata: dict                   # latency, agent name, etc.
```

### LLM System Prompts

#### 🟦 QA Agent — `temperature=0.0` (fully deterministic)
```
You are a precise research assistant.
Answer ONLY using the provided numbered context chunks [1],[2]…
Cite the chunk number inline after every claim you make.
If the answer is not present in the context, respond with:
"Not available in the provided documents."
Never speculate. Never use prior knowledge.
```

#### 🟩 Summarizer Agent — `temperature=0.3`
```
You are an expert research summarizer.
Output EXACTLY this 7-section template:

## Title
## Authors & Year
## Problem Statement
## Key Contributions  (5 bullet points)
## Methodology
## Results & Findings
## Limitations & Future Work

Use only information from the provided context chunks.
```

#### 🟨 Comparator Agent — `temperature=0.1`
```
You are a research comparison specialist.
Produce a Markdown comparison table with columns:
Problem | Methodology | Dataset | Key Result |
Evaluation Metric | Limitation | Future Work

Follow with:
### Similarities
### Key Differences
### Recommendation

Ground every cell in the provided context. Cite [doc_id] inline.
```

### Routing Logic

```
Query
  │
  ├─ Stage 1 — Rule-based regex (0ms, no LLM cost):
  │     "compare / vs / difference / between"  →  comparator
  │     "summarize / overview / key points"    →  summarizer
  │     (no match)                             →  Stage 2
  │
  └─ Stage 2 — LLM classifier (GPT-4o, ~200ms):
        Returns exactly: "qa" | "summarizer" | "comparator"
```

---

## 📂 Project Structure

```
papermind/
├── config.py                   # Central config (pydantic-settings)
├── app.py                      # Streamlit UI (5 tabs)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── agents/
│   ├── router.py               # Rule-based + LLM query classifier
│   ├── qa_agent.py             # Strict factual QA — GPT-4o, temp 0.0
│   ├── summarizer.py           # 7-section summaries — GPT-4o, temp 0.3
│   └── comparator.py           # Multi-paper comparison — GPT-4o, temp 0.1
│
├── retriever/
│   ├── embeddings.py           # OpenAI text-embedding-3-small singleton
│   ├── faiss_store.py          # Dense vector store + persistence
│   ├── bm25_store.py           # Sparse BM25 keyword index
│   ├── hybrid_retriever.py     # RRF fusion + cross-encoder reranking
│   └── reranker.py             # ms-marco-MiniLM-L-6-v2 reranker
│
├── pipeline/
│   ├── ingestion.py            # PDF → parse → chunk → embed → index
│   ├── chunking.py             # RecursiveCharacterTextSplitter
│   └── query_rewriter.py       # LLM-powered query expansion
│
├── graph/
│   ├── state.py                # ResearchState TypedDict
│   └── orchestrator.py         # LangGraph compiled graph
│
├── api/
│   ├── main.py                 # FastAPI app + middleware + lifespan
│   ├── routes.py               # /ingest /ask /summarize /compare /health
│   └── schemas.py              # Pydantic v2 request/response models
│
├── prompts/
│   └── agent_prompts.py        # All LLM system prompts — edit here to tune
│
├── cache/
│   └── redis_cache.py          # Redis + diskcache fallback
│
├── evaluation/
│   └── evaluator.py            # Retrieval accuracy + LLM-as-judge scoring
│
├── utils/
│   ├── parsing.py              # PyMuPDF + pdfplumber PDF parser
│   ├── citation.py             # Citation building + formatting
│   ├── token_counter.py        # tiktoken token counting/truncation
│   └── logger.py               # structlog JSON logger
│
├── scripts/
│   └── cli.py                  # CLI: ingest / query / evaluate / status
│
└── tests/
    ├── test_router.py          # Router unit tests (15 cases)
    ├── test_chunking.py        # Chunking unit tests (6 cases)
    ├── test_retriever.py       # RRF + rewriter tests (10 cases)
    └── test_integration.py     # Full pipeline integration tests
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- OpenAI API key (`OPENAI_API_KEY=sk-...`)
- Docker + Docker Compose (optional)

### Local Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/papermind.git
cd papermind

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and set: OPENAI_API_KEY=sk-...

# 5. Create required data directories
mkdir -p data/faiss_index data/documents data/diskcache

# 6. Start the API backend (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 7. Start the Streamlit frontend (Terminal 2)
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Docker Setup (Recommended for Production)

```bash
cp .env.example .env          # set OPENAI_API_KEY inside
docker-compose up --build

# Services started:
#   Streamlit UI  →  http://localhost:8501
#   FastAPI docs  →  http://localhost:8000/docs
#   Redis cache   →  localhost:6379
```

---

## 📦 Dependencies

| Category | Package | Version | Purpose |
|---|---|---|---|
| **LLM** | `langchain-openai` | 0.2.5 | GPT-4o + embeddings client |
| **Orchestration** | `langgraph` | 0.2.28 | Multi-agent state graph |
| **LangChain** | `langchain-core` | 0.3.15 | Messages, Documents, chains |
| **Vector Search** | `faiss-cpu` | 1.8.0 | Dense ANN index |
| **Keyword Search** | `rank-bm25` | 0.2.2 | Sparse BM25 index |
| **Reranker** | `sentence-transformers` | 3.1.1 | Cross-encoder reranking |
| **PDF (primary)** | `pymupdf` | 1.24.10 | Fast PDF text extraction |
| **PDF (fallback)** | `pdfplumber` | 0.11.4 | Complex layout extraction |
| **API** | `fastapi` | 0.115.0 | REST backend |
| **Server** | `uvicorn[standard]` | 0.30.6 | ASGI server |
| **UI** | `streamlit` | ≥1.35.0 | Chat interface |
| **Cache** | `redis` | 5.1.0 | Production response cache |
| **Cache (dev)** | `diskcache` | 5.6.3 | Local fallback cache |
| **Config** | `pydantic-settings` | ≥2.5.0 | Type-safe env config |
| **File Uploads** | `python-multipart` | 0.0.12 | FastAPI UploadFile support |
| **Tokenisation** | `tiktoken` | ≥0.7.0 | Token counting/truncation |
| **Logging** | `structlog` | 24.4.0 | Structured JSON logs |
| **Evaluation** | `rouge-score` | 0.1.2 | ROUGE-L scoring |
| **Testing** | `pytest` + `httpx` | ≥8.3.0 | Unit + integration tests |

---

## 🚀 Usage

### Via the Streamlit UI
1. **Ingest tab** → drag and drop PDFs → click **Ingest**
2. **Chat tab** → type any question — the LLM router auto-selects the right agent
3. **Summarize tab** → pick a document → click **Generate Summary**
4. **Compare tab** → select 2–4 papers → click **Compare**

### Via CLI

```bash
# Ingest a paper
python scripts/cli.py ingest papers/attention_is_all_you_need.pdf

# Ask a question (LLM answers with citations)
python scripts/cli.py query "What optimizer did the authors use?"

# Check which documents are indexed
python scripts/cli.py status

# Run evaluation against a labeled dataset
python scripts/cli.py evaluate eval_dataset_sample.json
```

### Via REST API

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@paper.pdf" -F "doc_id=my_paper"

# Ask a question (LLM auto-routes to correct agent)
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the key contribution of this paper?"}'

# Generate a structured LLM summary
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "my_paper", "style": "bullet"}'

# LLM comparison of multiple papers
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"doc_ids": ["paper_a", "paper_b"], "focus": "methodology"}'
```

---

## 📊 Evaluation

PaperMind uses a **GPT-4o judge** to evaluate LLM response quality — the same model that powers the agents also independently evaluates their outputs.

```bash
python scripts/cli.py evaluate eval_dataset_sample.json
```

| Metric | How It Is Measured |
|---|---|
| **Retrieval Accuracy** | Is the gold-standard chunk present in the top-k retrieved results? |
| **Answer Faithfulness** | GPT-4o judge scores 1–5: does the LLM answer stay grounded in context? |
| **Latency** | End-to-end pipeline time (rewrite → retrieve → generate) in ms |
| **ROUGE-L** | F1 overlap between LLM summary output and human reference summary |

> A faithfulness score of **≥ 4 / 5** is considered a trusted, grounded response.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run pure logic tests (no OpenAI API key required)
pytest tests/test_router.py tests/test_chunking.py tests/test_retriever.py -v

# Run with HTML coverage report
pytest tests/ --cov=. --cov-report=html
```

---

## 🔧 Tuning LLM Behaviour

All system prompts are centralised in `prompts/agent_prompts.py`. To change how any agent responds, edit the constants in that single file — no other code changes are needed.

| Variable | Effect |
|---|---|
| `QA_SYSTEM_PROMPT` | Controls strictness of citation grounding in QA answers |
| `SUMMARIZER_SYSTEM_PROMPT` | Controls the 7-section template structure and detail level |
| `COMPARATOR_SYSTEM_PROMPT` | Controls comparison table columns and synthesis sections |
| `ROUTER_SYSTEM_PROMPT` | Controls how the LLM classifies ambiguous queries |
| `settings.openai_model` | Switch between `gpt-4o`, `gpt-4o-mini`, or any OpenAI model |

---

## ☁️ Deployment

### AWS EC2
```bash
git clone <repo> && cd papermind
echo "OPENAI_API_KEY=sk-..." > .env
docker-compose up -d
```

### GCP Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/papermind
gcloud run deploy papermind \
  --image gcr.io/PROJECT_ID/papermind \
  --platform managed \
  --set-env-vars OPENAI_API_KEY=sk-...
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | GPT-4o | QA, summarization, comparison, routing, query rewriting |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic dense vector search |
| **Orchestration** | LangGraph 0.2 | Stateful multi-agent graph |
| **Dense Index** | FAISS | Approximate nearest-neighbour vector search |
| **Sparse Index** | BM25 (rank-bm25) | Keyword overlap retrieval |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder candidate reranking |
| **PDF Parser** | PyMuPDF + pdfplumber | Text extraction from research PDFs |
| **Backend** | FastAPI | REST API |
| **Frontend** | Streamlit | Chat + ingest UI |
| **Cache** | Redis + diskcache | Avoid redundant LLM API calls |
| **Logging** | structlog | Structured JSON logs |
| **Config** | pydantic-settings | Type-safe environment configuration |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
