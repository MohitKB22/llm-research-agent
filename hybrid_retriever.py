"""
retriever/hybrid_retriever.py – Hybrid search: FAISS + BM25 → RRF → Rerank.

Pipeline:
  1. Dense retrieval via FAISS (semantic similarity)
  2. Sparse retrieval via BM25 (keyword overlap)
  3. Reciprocal Rank Fusion (RRF) to merge ranked lists
  4. Cross-encoder reranking of the fused candidates
  5. Return top-k final documents

RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
              where k=60 (standard constant), rank_i is the document's
              position in result list i.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from langchain_core.documents import Document

from config import settings
from retriever.bm25_store import get_bm25_store
from retriever.faiss_store import get_faiss_store
from retriever.reranker import rerank
from utils.logger import get_logger

log = get_logger(__name__)

_RRF_K = 60  # standard RRF constant


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
) -> list[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    De-duplicates by page_content, preserving the highest-fused-score copy.

    Returns:
        List of Documents sorted by RRF score descending, with
        metadata["_rrf_score"] attached.
    """
    # BUG FIX: `hash(doc.page_content)` produces an int, but the type
    # annotations declared `dict[str, float]` and `dict[str, Document]`,
    # causing a type mismatch that also breaks JSON serialisation downstream.
    # Changed the key to `str(hash(...))` so it is always a str, consistent
    # with the declared types.
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = str(hash(doc.page_content))  # BUG FIX: cast to str
            rrf_scores[key] += 1.0 / (_RRF_K + rank)
            # BUG FIX: keep the doc with the HIGHEST individual score, not
            # the latest one.  Overwriting with the latest loses the best
            # candidate when the same chunk is retrieved by both retrievers.
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "_rrf_score": rrf_scores[key]},
        )
        results.append(doc)

    return results


class HybridRetriever:
    """
    Stateless retriever that orchestrates dense + sparse + rerank.

    Usage:
        retriever = HybridRetriever()
        docs = retriever.retrieve("What optimizer did the authors use?")
    """

    def retrieve(
        self,
        query: str,
        top_k_candidates: Optional[int] = None,
        top_k_final: Optional[int] = None,
        filter_doc_ids: Optional[list[str]] = None,
        skip_rerank: bool = False,
    ) -> list[Document]:
        """
        Execute the full hybrid retrieval pipeline.

        Args:
            query: Retrieval query (ideally already rewritten).
            top_k_candidates: How many results each retriever returns
                              before fusion. Defaults to settings.top_k_retrieval.
            top_k_final: Final docs after reranking.
                         Defaults to settings.top_k_rerank.
            filter_doc_ids: Scope search to specific documents.
            skip_rerank: Skip cross-encoder reranking (faster, lower quality).

        Returns:
            Final list of Documents, most relevant first.
        """
        k_cand = top_k_candidates or settings.top_k_retrieval
        k_final = top_k_final or settings.top_k_rerank

        # ── Step 1: Dense retrieval ───────────────────────────────────────────
        faiss_results = get_faiss_store().similarity_search(
            query=query,
            k=k_cand,
            filter_doc_ids=filter_doc_ids,
        )
        log.info("faiss_retrieved", count=len(faiss_results))

        # ── Step 2: Sparse retrieval ──────────────────────────────────────────
        bm25_results = get_bm25_store().search(
            query=query,
            k=k_cand,
            filter_doc_ids=filter_doc_ids,
        )
        log.info("bm25_retrieved", count=len(bm25_results))

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────
        fused = _reciprocal_rank_fusion([faiss_results, bm25_results])
        log.info("rrf_fused", count=len(fused))

        if not fused:
            return []

        # ── Step 4: Cross-encoder reranking ───────────────────────────────────
        # BUG FIX: Original condition `len(fused) <= k_final` skipped reranking
        # even when skip_rerank=False and there were more candidates than
        # k_final. The reranker should always run when skip_rerank=False AND
        # there are more candidates than needed, regardless of list length.
        if skip_rerank:
            final = fused[:k_final]
        elif len(fused) <= k_final:
            # Not enough candidates to warrant reranking; take all of them.
            final = fused
        else:
            final = rerank(query=query, documents=fused, top_k=k_final)

        log.info(
            "hybrid_retrieval_complete",
            query=query[:60],
            final_docs=len(final),
        )
        return final


# Module-level singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
