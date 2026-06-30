"""
pipeline.py — End-to-end RAG pipeline orchestrator.

Flow:
  raw STT text
    ↓ normalize.normalize_stt_text()  — clean Whisper noise, NFC, nuktas
    ↓ (if garbage) → return fallback
    ↓ (cache lookup) — if normalized query in LRU cache, return cached answer
    ↓ retrieve.retrieve() — BM25 + Dense + RRF → top-3 candidates
    ↓ (CRAG-lite gate) — if top_score < threshold_low, return fallback (skip LLM)
    ↓ generate.generate_answer() — ONE LLM call = rerank + answer + self-grade
    ↓ (if LLM signaled don't-know) → return fallback
    ↓ cache + return

All stages log structured timing for kiosk monitoring.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import get_config
from .generate import GenerationResult, generate_answer
from .normalize import NormalizationResult, normalize_stt_text
from .reranker import Reranker
from .retrieve import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ---------- LRU Cache ----------

class LRUCache:
    """Simple OrderedDict-based LRU cache."""

    def __init__(self, max_size: int = 256):
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)  # evict oldest

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ---------- Result ----------

@dataclass
class PipelineResult:
    """End-to-end pipeline output."""
    answer_mr: str                  # final Marathi answer (or fallback)
    is_fallback: bool               # True if we couldn't answer
    fallback_reason: str            # "" if not fallback, else one of:
                                    #   "garbage_input"
                                    #   "low_confidence_crag"
                                    #   "llm_dont_know"
                                    #   "llm_error"
                                    #   "no_candidates"
    chosen_qa_id: str               # qa_id of chosen context ("" if fallback)
    chosen_context_idx: int         # 0 = no context, 1-3 = which top-K
    confidence_tier: str            # "high" | "medium" | "low" | "unknown"
    top_fused_score: float
    cached: bool                    # True if served from LRU cache
    latency_s: float
    query_intent: str = "UNKNOWN"   # rule-based intent (fast path)
    stage_latencies_s: Dict[str, float] = field(default_factory=dict)
    normalization: Optional[NormalizationResult] = None
    retrieval: Optional[RetrievalResult] = None
    generation: Optional[GenerationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view (excludes heavy nested objects)."""
        return {
            "answer_mr": self.answer_mr,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "chosen_qa_id": self.chosen_qa_id,
            "chosen_context_idx": self.chosen_context_idx,
            "confidence_tier": self.confidence_tier,
            "top_fused_score": self.top_fused_score,
            "cached": self.cached,
            "latency_s": round(self.latency_s, 3),
            "stage_latencies_s": {k: round(v, 3) for k, v in self.stage_latencies_s.items()},
            "normalization_transformations": (
                self.normalization.transformations if self.normalization else []
            ),
            "normalization_is_garbage": (
                self.normalization.is_garbage if self.normalization else None
            ),
            "retrieval_top_doc_ids": (
                [a.qa_id for a in self.retrieval.candidates] if self.retrieval else []
            ),
            "retrieval_bm25_top": (
                self.retrieval.bm25_top_ids if self.retrieval else []
            ),
            "retrieval_dense_top": (
                self.retrieval.dense_top_ids if self.retrieval else []
            ),
            "generation_finish_reason": (
                self.generation.finish_reason if self.generation else None
            ),
            "generation_prompt_tokens": (
                self.generation.prompt_tokens if self.generation else 0
            ),
            "generation_completion_tokens": (
                self.generation.completion_tokens if self.generation else 0
            ),
        }


# ---------- Pipeline ----------

class RAGPipeline:
    """Top-level orchestrator. Load once, call .answer() per query."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._reranker: Optional[Reranker] = None
        cfg = get_config()
        self._cache_enabled = cfg["cache"]["enabled"]
        self._cache = LRUCache(max_size=cfg["cache"]["max_size"])
        self._fallback_msg = cfg["pipeline"]["fallback_message_mr"]
        self._threshold_low = cfg["retrieval"]["confidence"]["threshold_low"]
        self._threshold_high = cfg["retrieval"]["confidence"]["threshold_high"]
        self._respect_llm_dont_know = cfg["pipeline"]["respect_llm_dont_know"]
        self._llm_selection_enabled = cfg["pipeline"].get("llm_selection_enabled", False)
        self._latency_budget_s = cfg["pipeline"]["latency_budget_s"]

    @classmethod
    def from_kb(cls, kb_path: Optional[str] = None) -> "RAGPipeline":
        """Build the full pipeline from the KB."""
        from .kb import load_kb
        cfg = get_config()
        kb_path = kb_path or cfg["kb"]["path"]
        articles, _meta = load_kb(kb_path)
        retriever = HybridRetriever.build(articles)
        return cls(retriever=retriever)

    def answer(self, raw_stt_text: str, skip_llm: bool = False) -> PipelineResult:
        """Process one STT output end-to-end.

        Args:
            raw_stt_text: Raw text from STT model.
            skip_llm: If True, use score-weighted fusion instead of LLM
                      to select the best context. ~0ms generation.

        Returns:
            PipelineResult.
        """
        t_start = time.perf_counter()
        stages: Dict[str, float] = {}

        # ---- 1. Normalize ----
        t0 = time.perf_counter()
        norm = normalize_stt_text(raw_stt_text)
        stages["normalize"] = time.perf_counter() - t0

        if norm.is_garbage:
            return self._fallback(
                reason="garbage_input",
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                top_fused_score=0.0,
                query_intent="UNKNOWN",
            )

        # ---- 1b. Fuzzy spell correction ----
        t0 = time.perf_counter()
        from .query_expand import fuzzy_correct
        corrected_text = fuzzy_correct(norm.cleaned, self.retriever._kb_vocab, max_dist=1)
        if corrected_text != norm.cleaned:
            logger.info(
                "Query fuzzy-corrected: %r → %r",
                norm.cleaned, corrected_text,
            )
            norm.cleaned = corrected_text
        stages["fuzzy_correct"] = time.perf_counter() - t0

        # ---- 1c. Intent extraction (rule-based, ~5ms) ----
        # Fast path. The LLM will also output intent as part of the combined
        # call, but rules give us instant intent for retrieval-stage reranking.
        t0 = time.perf_counter()
        from .query_intent import extract_intent
        query_intent = extract_intent(norm.cleaned)
        stages["intent_extract"] = time.perf_counter() - t0
        logger.info("Query intent (rules): %s for %r", query_intent, norm.cleaned[:80])

        # ---- 1d. Structural validation gate (Stage 1, pre-retrieval) ----
        t0 = time.perf_counter()
        from .query_validate import validate_structure, validate_scores
        cfg = get_config()
        val_cfg = cfg.get("query_validation", {})
        is_valid, reject_reason = validate_structure(
            norm.cleaned, val_cfg, kb_vocab=self.retriever._kb_vocab,
        )
        stages["validate_structure"] = time.perf_counter() - t0
        if not is_valid:
            logger.info("Query rejected (structural): %s for %r", reject_reason, norm.cleaned[:80])
            return self._fallback(
                reason=reject_reason,
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                query_intent=query_intent,
            )

        # ---- 2. Cache lookup ----
        cache_key = norm.cleaned
        if self._cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached.cached = True
                cached.latency_s = time.perf_counter() - t_start
                cached.stage_latencies_s = {**stages, "cache_hit": 0.0}
                logger.info("Pipeline cache hit", extra={"query": cache_key[:80]})
                return cached

        # ---- 3. Retrieve ----
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(norm.cleaned)
        stages["retrieve"] = time.perf_counter() - t0

        if not retrieval.candidates:
            return self._fallback(
                reason="no_candidates",
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                top_fused_score=0.0,
                retrieval=retrieval,
                query_intent=query_intent,
            )

        # ---- 4. CRAG-lite confidence gate ----
        # NOTE: retrieval.fusion.top_score / confidence_tier are computed
        # BEFORE intent-aware reranking (see retrieve.py step 3 vs step 5)
        # and are kept only for debug display of the raw RRF ranking.
        # Grading confidence on that stale score is wrong: intent rerank can
        # (and routinely does) demote the pre-rerank #1 candidate by 3x+ for
        # an intent mismatch, or promote a candidate that was nowhere near
        # the top. If we gate on the stale score we can answer "high
        # confidence" using a candidate that intent rerank just penalized
        # into irrelevance (e.g. answering a YES_NO-shaped QA pair for a WHY
        # question). The gate must use the score of the candidate that will
        # actually be served, i.e. the post-rerank top of retrieval.candidates.
        top_score = retrieval.fused_scores[0] if retrieval.fused_scores else 0.0
        if top_score >= self._threshold_high:
            tier = "high"
        elif top_score >= self._threshold_low:
            tier = "medium"
        else:
            tier = "low"

        if tier == "low":
            return self._fallback(
                reason="low_confidence_crag",
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                retrieval=retrieval,
                top_fused_score=top_score,
                query_intent=query_intent,
            )

        # ---- 5. Score-weighted selection (skip LLM) ----
        if skip_llm:
            if self._reranker is None:
                self._reranker = Reranker()
            best_id, best_score, all_scores = self._reranker.select(norm.cleaned, retrieval.candidates)
            retrieval.weighted_scores = {c.qa_id: s for c, s in zip(retrieval.candidates, all_scores)}

            # ---- 5a. Score-based validation (Stage 2, post-retrieval) ----
            # Use raw scores (not RRF) as the gate — they carry absolute signal.
            dense_top = max(retrieval.dense_scores.values()) if retrieval.dense_scores else 0.0
            bm25_top = max(retrieval.bm25_scores.values()) if retrieval.bm25_scores else 0.0
            from .query_validate import compute_token_coverage
            token_coverage = compute_token_coverage(norm.cleaned, retrieval.candidates)
            t0 = time.perf_counter()
            is_valid, reject_reason = validate_scores(
                dense_top_score=dense_top,
                bm25_top_score=bm25_top,
                reranker_score=best_score,
                token_coverage=token_coverage,
                cfg=val_cfg,
            )
            stages["validate_scores"] = time.perf_counter() - t0
            if not is_valid:
                logger.info(
                    "Query rejected (scores): %s | dense=%.3f bm25=%.3f reranker=%.3f coverage=%.2f",
                    reject_reason, dense_top, bm25_top, best_score, token_coverage,
                )
                return self._fallback(
                    reason=reject_reason,
                    stage_latencies_s=stages,
                    t_start=t_start,
                    normalization=norm,
                    retrieval=retrieval,
                    top_fused_score=top_score,
                    query_intent=query_intent,
                )

            if not best_id or best_id not in self.retriever.articles_by_id:
                return self._fallback(
                    reason="low_confidence_crag",
                    stage_latencies_s=stages,
                    t_start=t_start,
                    normalization=norm,
                    retrieval=retrieval,
                    top_fused_score=top_score,
                    query_intent=query_intent,
                )
            best_article = self.retriever.articles_by_id[best_id]
            stages["generate"] = 0.0
            result = PipelineResult(
                answer_mr=best_article.answer_mr,
                is_fallback=False,
                fallback_reason="",
                chosen_qa_id=best_id,
                chosen_context_idx=1,
                confidence_tier=tier,
                top_fused_score=top_score,
                cached=False,
                latency_s=time.perf_counter() - t_start,
                query_intent=query_intent,
                stage_latencies_s=stages,
                normalization=norm,
                retrieval=retrieval,
                generation=None,
            )
            if self._cache_enabled:
                self._cache.put(cache_key, result)
            logger.info(
                "Pipeline success (skip_llm)",
                extra={
                    "query": norm.cleaned[:80],
                    "qa_id": best_id,
                    "tier": tier,
                    "top_score": round(top_score, 4),
                    "latency_s": round(result.latency_s, 3),
                },
            )
            return result

        # ---- 6. Generate (single LLM call) ----
        # Off by default (see config.yaml: pipeline.llm_selection_enabled).
        # Skips straight to serving retrieval.candidates[0] verbatim, which
        # is now a reliable pick thanks to the widened intent-rerank pool
        # and the post-rerank confidence gate above — without the 6-11s
        # LLM round trip this step previously cost on every single query.
        if not self._llm_selection_enabled:
            best_article = retrieval.candidates[0]
            stages["generate"] = 0.0
            result = PipelineResult(
                answer_mr=best_article.answer_mr,
                is_fallback=False,
                fallback_reason="",
                chosen_qa_id=best_article.qa_id,
                chosen_context_idx=1,
                confidence_tier=tier,
                top_fused_score=top_score,
                cached=False,
                latency_s=time.perf_counter() - t_start,
                query_intent=query_intent,
                stage_latencies_s=stages,
                normalization=norm,
                retrieval=retrieval,
                generation=None,
            )
            if self._cache_enabled:
                self._cache.put(cache_key, result)
            logger.info(
                "Pipeline success (direct, no LLM)",
                extra={
                    "query": norm.cleaned[:80],
                    "qa_id": best_article.qa_id,
                    "tier": tier,
                    "top_score": round(top_score, 4),
                    "latency_s": round(result.latency_s, 3),
                },
            )
            return result

        t0 = time.perf_counter()
        try:
            gen = generate_answer(
                query=norm.cleaned,
                top_k_candidates=retrieval.candidates,
                confidence_tier=tier,
            )
        except Exception as e:
            logger.error("LLM generation failed", extra={"error": str(e)[:200]})
            return self._fallback(
                reason="llm_error",
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                retrieval=retrieval,
                top_fused_score=top_score,
                query_intent=query_intent,
            )
        stages["generate"] = time.perf_counter() - t0

        # ---- 7. Check LLM "don't know" signal ----
        if gen.llm_signaled_dont_know and self._respect_llm_dont_know:
            return self._fallback(
                reason="llm_dont_know",
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                retrieval=retrieval,
                top_fused_score=top_score,
                generation=gen,
                query_intent=query_intent,
            )

        # ---- 8. Success — build result ----
        chosen_qa_id = ""
        if 1 <= gen.chosen_context_idx <= len(retrieval.candidates):
            chosen_qa_id = retrieval.candidates[gen.chosen_context_idx - 1].qa_id

        result = PipelineResult(
            answer_mr=gen.answer_mr,
            is_fallback=False,
            fallback_reason="",
            chosen_qa_id=chosen_qa_id,
            chosen_context_idx=gen.chosen_context_idx,
            confidence_tier=tier,
            top_fused_score=top_score,
            cached=False,
            latency_s=time.perf_counter() - t_start,
            query_intent=query_intent,
            stage_latencies_s=stages,
            normalization=norm,
            retrieval=retrieval,
            generation=gen,
        )

        if self._cache_enabled:
            self._cache.put(cache_key, result)

        logger.info(
            "Pipeline success",
            extra={
                "query": norm.cleaned[:80],
                "qa_id": chosen_qa_id,
                "ctx_idx": gen.chosen_context_idx,
                "tier": tier,
                "top_score": round(top_score, 4),
                "latency_s": round(result.latency_s, 3),
                "stage_latencies_s": {k: round(v, 3) for k, v in stages.items()},
            },
        )
        return result


    # Fallback reasons that map to "rejected" (definitive score/structural rejection)
    _SCORE_FALLBACKS = frozenset({
        "low_dense_cosine", "low_bm25", "low_reranker",
        "low_token_coverage", "vocab_mismatch", "entity_mismatch",
    })

    def _fallback(
        self,
        reason: str,
        stage_latencies_s: Dict[str, float],
        t_start: float,
        normalization: Optional[NormalizationResult] = None,
        retrieval: Optional[RetrievalResult] = None,
        generation: Optional[GenerationResult] = None,
        top_fused_score: float = 0.0,
        query_intent: str = "UNKNOWN",
    ) -> PipelineResult:
        """Build a fallback result with correct confidence tier.

        Never copies the pre-rerank RRF confidence tier when falling back.
        Instead assigns a meaningful tier based on the rejection reason:
          - Score/validation failures → "rejected"
          - Low confidence / errors → "low"
          - Structural rejections → "rejected"
        """
        if reason in self._SCORE_FALLBACKS:
            confidence_tier = "rejected"
        elif reason in ("low_confidence_crag", "no_candidates", "llm_dont_know", "llm_error"):
            confidence_tier = "low"
        else:
            confidence_tier = "rejected"

        result = PipelineResult(
            answer_mr=self._fallback_msg,
            is_fallback=True,
            fallback_reason=reason,
            chosen_qa_id="",
            chosen_context_idx=0,
            confidence_tier=confidence_tier,
            top_fused_score=top_fused_score,
            cached=False,
            latency_s=time.perf_counter() - t_start,
            query_intent=query_intent,
            stage_latencies_s=stage_latencies_s,
            normalization=normalization,
            retrieval=retrieval,
            generation=generation,
        )
        logger.info(
            "Pipeline fallback",
            extra={
                "reason": reason,
                "top_score": round(top_fused_score, 4),
                "latency_s": round(result.latency_s, 3),
            },
        )
        return result

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
