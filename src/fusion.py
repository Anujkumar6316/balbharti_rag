"""
fusion.py — RRF fusion, score-weighted selection, and CRAG-lite confidence gating.

Two fusion methods:
  1. RRF (Reciprocal Rank Fusion) — rank-based, good when scores are on
     different scales. Default for LLM-assisted selection.
  2. Score-weighted selection — uses raw BM25 and Dense scores directly
      with min-max normalization. Used for --skip-llm mode (no LLM needed).

CRAG-lite (Corrective RAG, lite version):
  - No extra LLM call.
  - Just use the top-1 fused score as confidence.
  - Below threshold_low  → immediate "no answer" fallback (skip LLM).
  - Between low and high  → still call LLM but instruct "may not be relevant".
  - Above high            → normal LLM call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Output of the fusion step."""
    ranked_doc_ids: List[str]      # all docs, ranked by fused score (desc)
    fused_scores: List[float]      # parallel to ranked_doc_ids
    top_doc_id: str                # convenience: ranked_doc_ids[0]
    top_score: float               # convenience: fused_scores[0]
    confidence_tier: str           # "high" | "medium" | "low"
    bm25_top_ids: List[str]        # for debugging
    dense_top_ids: List[str]


def reciprocal_rank_fusion(
    bm25_ranking: List[Tuple[str, float]],
    dense_ranking: List[Tuple[str, float]],
    k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_k_shortlist: int = 3,
) -> FusionResult:
    """Combine BM25 and dense rankings via RRF.

    Args:
        bm25_ranking: List of (doc_id, score) sorted descending.
        dense_ranking: Same.
        k: RRF constant (default 60).
        bm25_weight / dense_weight: optional weighting.
        top_k_shortlist: how many top docs to return for LLM context.

    Returns:
        FusionResult with all docs ranked by fused score.
    """
    cfg = get_config()
    conf = cfg["retrieval"]["confidence"]
    threshold_low = conf["threshold_low"]
    threshold_high = conf["threshold_high"]

    # Compute RRF scores
    rrf_scores: Dict[str, float] = {}

    for rank, (doc_id, _score) in enumerate(bm25_ranking, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + bm25_weight / (k + rank)

    for rank, (doc_id, _score) in enumerate(dense_ranking, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_weight / (k + rank)

    # Sort by fused score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_doc_ids = [doc_id for doc_id, _ in ranked]
    fused_scores = [score for _, score in ranked]

    # CRAG-lite confidence tier
    top_score = fused_scores[0] if fused_scores else 0.0
    if top_score >= threshold_high:
        tier = "high"
    elif top_score >= threshold_low:
        tier = "medium"
    else:
        tier = "low"

    return FusionResult(
        ranked_doc_ids=ranked_doc_ids,
        fused_scores=fused_scores,
        top_doc_id=ranked_doc_ids[0] if ranked_doc_ids else "",
        top_score=top_score,
        confidence_tier=tier,
        bm25_top_ids=[d for d, _ in bm25_ranking[:top_k_shortlist]],
        dense_top_ids=[d for d, _ in dense_ranking[:top_k_shortlist]],
    )


def score_weighted_selection(
    bm25_ranking: List[Tuple[str, float]],
    dense_ranking: List[Tuple[str, float]],
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    normalize_top_k: int = 10,
    min_score: float = 0.2,
) -> Tuple[str, float, Dict[str, float]]:
    """Select the best doc using score-level weighted fusion.

    BM25 scores are unbounded → softmax normalization within the union
    of top-K docs from both retrievers.
    Dense scores are already cosine similarity in [0, 1] → used directly.

    Args:
        bm25_ranking: List of (doc_id, raw_bm25_score) sorted descending.
        dense_ranking: List of (doc_id, cosine_sim) sorted descending.
        bm25_weight: Weight for BM25 score (default 0.4).
        dense_weight: Weight for dense score (default 0.6).
        normalize_top_k: Consider top-N from each retriever for normalization.
        min_score: Minimum weighted score to accept. Below this → fallback.

    Returns:
        (best_doc_id, weighted_score, all_weighted_scores).
        best_doc_id is "" if no doc meets min_score.
    """
    bm25_scores: Dict[str, float] = {d: s for d, s in bm25_ranking[:normalize_top_k]}
    dense_scores: Dict[str, float] = {d: s for d, s in dense_ranking[:normalize_top_k]}

    all_docs = set(bm25_scores) | set(dense_scores)
    if not all_docs:
        return ("", 0.0)

    # Min-max normalize BM25 scores within the union set
    bm25_vals = [bm25_scores.get(d) for d in all_docs]
    valid = [v for v in bm25_vals if v is not None]
    bm25_min, bm25_max = min(valid), max(valid) if valid else (0.0, 1.0)
    bm25_range = bm25_max - bm25_min or 1.0
    bm25_norm: Dict[str, float] = {}
    for d in all_docs:
        v = bm25_scores.get(d)
        bm25_norm[d] = (v - bm25_min) / bm25_range if v is not None else 0.0

    # Weighted combination
    weighted: Dict[str, float] = {}
    for d in all_docs:
        w = bm25_weight * bm25_norm.get(d, 0.0) + dense_weight * dense_scores.get(d, 0.0)
        weighted[d] = w

    best_id = max(weighted, key=weighted.get)
    best_score = weighted[best_id]

    if best_score < min_score:
        return ("", best_score, weighted)

    return (best_id, best_score, weighted)
