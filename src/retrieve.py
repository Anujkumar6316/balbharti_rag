"""
retrieve.py — Hybrid retriever orchestrator (BM25 + Dense → RRF).

Combines:
  1. BM25 (sparse) — fast, good for keyword overlap (numbers, proper nouns)
  2. Dense (MuRIL) — paraphrase-aware, semantic match
  3. Reciprocal Rank Fusion — combines their rankings

Returns top-K (default 3) candidate QA pairs ready for the LLM generator.

This module is stateful (holds the BM25 + dense indices) but thread-safe
for reads. On a serial Pi kiosk, concurrency is not a concern.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bm25_index import BM25Index
from .config import get_config
from .dense_index import DenseIndex
from .fusion import FusionResult, reciprocal_rank_fusion, score_weighted_selection
from .kb import KBArticle, articles_by_id
from .query_expand import expand_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Output of the retriever."""
    candidates: List[KBArticle]   # top-K articles, best first
    fused_scores: List[float]     # parallel to candidates
    fusion: FusionResult          # full fusion metadata
    bm25_top_ids: List[str]
    dense_top_ids: List[str]
    latency_s: float
    bm25_scores: Dict[str, float]     # raw BM25 scores for top docs
    dense_scores: Dict[str, float]    # raw dense scores for top docs
    weighted_scores: Dict[str, float] = field(default_factory=dict)  # score-weighted fusion results
    intent_rerank_log: list = field(default_factory=list)  # debug: (qa_id, intent, old_score, new_score, action)
    query_intent: str = "UNKNOWN"


class HybridRetriever:
    """BM25 + Dense + RRF retriever."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        articles_by_id_map: Dict[str, KBArticle],
        kb_vocab: set = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.articles_by_id = articles_by_id_map
        self._kb_vocab = kb_vocab or set()

    @classmethod
    def build(cls, articles: List[KBArticle]) -> "HybridRetriever":
        """Build both indices from a list of KB articles."""
        bm25 = BM25Index.from_config()
        dense = DenseIndex.from_config()

        doc_ids = [a.qa_id for a in articles]
        docs = [a.doc_text for a in articles]
        variants_per_doc = [a.variant_list for a in articles]

        bm25.add_docs(doc_ids, docs)
        bm25.build()
        dense.build_from_variants(doc_ids, variants_per_doc)

        # Build KB vocab for fuzzy spell correction
        from .query_expand import build_kb_vocab
        kb_vocab = build_kb_vocab(articles)

        return cls(
            bm25=bm25,
            dense=dense,
            articles_by_id_map=articles_by_id(articles),
            kb_vocab=kb_vocab,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Run hybrid retrieval for a single query.

        Args:
            query: Normalized + fuzzy-corrected user query (Devanagari Marathi).
                   Fuzzy correction is applied upstream in pipeline.py so that
                   cache, retrieve, and reranker all see the same corrected text.
            top_k: Override config default (3).

        Returns:
            RetrievalResult with top-K candidates.
        """
        import time
        cfg = get_config()
        if top_k is None:
            top_k = cfg["retrieval"]["top_k"]

        t0 = time.perf_counter()

        # 1. BM25 retrieval — expand query, merge results from all forms
        bm25_scores_raw: Dict[str, float] = {}
        for q in expand_query(query):
            doc_ids, scores = self.bm25.score(q)
            for doc_id, score in zip(doc_ids, scores):
                if doc_id not in bm25_scores_raw or score > bm25_scores_raw[doc_id]:
                    bm25_scores_raw[doc_id] = score
        # NOTE: previously capped at min(n_docs, 30) before fusion. At this KB
        # size (hundreds of docs, not millions) that cap is pure recall loss
        # for no latency benefit — a correct candidate ranked just outside the
        # top 30 by one retriever is invisible to RRF fusion and to intent
        # rerank, no matter how well-tuned those stages are. Rank the full
        # corpus instead; truncation happens later via pool_size / top_k.
        bm25_top = sorted(bm25_scores_raw.items(), key=lambda x: x[1], reverse=True)[:self.bm25.n_docs]

        # 2. Dense retrieval
        dense_top = self.dense.top_k(query, k=self.dense.n_docs)

        # 3. RRF fusion
        fusion = reciprocal_rank_fusion(
            bm25_ranking=bm25_top,
            dense_ranking=dense_top,
            k=cfg["retrieval"]["fusion"]["k"],
            bm25_weight=cfg["retrieval"]["fusion"]["bm25_weight"],
            dense_weight=cfg["retrieval"]["fusion"]["dense_weight"],
            top_k_shortlist=top_k,
        )

        # 4. Pick top-K candidates (use a wider pool so intent rerank has room)
        pool_mult = cfg["retrieval"]["intent"]["pool_multiplier"]
        pool_size = min(top_k * pool_mult, len(fusion.ranked_doc_ids))
        pool_doc_ids = fusion.ranked_doc_ids[:pool_size]
        pool_scores = fusion.fused_scores[:pool_size]
        pool_candidates = [self.articles_by_id[did] for did in pool_doc_ids if did in self.articles_by_id]

        # 5. Intent-aware reranking
        from .query_intent import extract_intent, rerank_by_intent
        query_intent = extract_intent(query)
        intent_log = []
        if query_intent != "UNKNOWN" and pool_candidates:
            pool_candidates, pool_scores, intent_log = rerank_by_intent(
                pool_candidates,
                pool_scores,
                query_intent,
                match_boost=cfg["retrieval"]["intent"]["match_boost"],
                mismatch_penalty=cfg["retrieval"]["intent"]["mismatch_penalty"],
            )
            logger.info(
                "Intent rerank: query_intent=%s, top=%s",
                query_intent,
                getattr(pool_candidates[0], "qa_id", "?") if pool_candidates else "NONE",
            )

        # 6. Keep top-K after intent rerank
        candidates = pool_candidates[:top_k]
        top_scores = pool_scores[:top_k]

        latency = time.perf_counter() - t0

        return RetrievalResult(
            candidates=candidates,
            fused_scores=top_scores,
            fusion=fusion,
            bm25_top_ids=fusion.bm25_top_ids,
            dense_top_ids=fusion.dense_top_ids,
            latency_s=latency,
            bm25_scores=dict(bm25_top),
            dense_scores=dict(dense_top),
            intent_rerank_log=intent_log,
            query_intent=query_intent,
        )
