"""
reranker.py — Cross-encoder reranker for context selection.

Replaces score-weighted fusion with neural pairwise scoring.
~150-250ms for 3 candidates on Pi 5 CPU.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from .config import get_config
from .kb import KBArticle

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker that selects the best context for a query."""

    def __init__(self, model_id: Optional[str] = None, threshold: Optional[float] = None):
        cfg = get_config()
        rc = cfg.get("reranker", {})
        model_id = model_id or rc.get("model_id", "Alibaba-NLP/gte-multilingual-reranker-base")
        threshold = threshold if threshold is not None else rc.get("threshold", 0.1)
        logger.info("Loading CrossEncoder model: %s", model_id)
        self._model = CrossEncoder(model_id, trust_remote_code=True)
        self.threshold = threshold

    def select(
        self, query: str, candidates: List[KBArticle]
    ) -> Tuple[Optional[str], float, List[float]]:
        """Score all candidates with cross-encoder and return best.

        Args:
            query: Normalized user query.
            candidates: List of KBArticle candidates from retrieval.

        Returns:
            (best_qa_id, best_score, all_scores).
            best_qa_id is None if no candidate meets threshold.
        """
        if not candidates:
            return None, 0.0, []

        pairs = [
            (query, f"प्रश्न: {c.question} | उत्तर: {c.answer_mr}")
            for c in candidates
        ]
        scores: List[float] = self._model.predict(pairs).tolist()

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        if best_score < self.threshold:
            return None, best_score, scores

        return candidates[best_idx].qa_id, best_score, scores
