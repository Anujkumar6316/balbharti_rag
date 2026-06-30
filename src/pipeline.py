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
        self._respect_llm_dont_know = cfg["pipeline"]["respect_llm_dont_know"]
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
            )

        # ---- 1b. Fuzzy spell correction ----
        # Fix STT/typo drift against KB vocab BEFORE any downstream use.
        # This ensures cache, retrieve, reranker all see the same corrected query.
        t0 = time.perf_counter()
        from .query_expand import fuzzy_correct
        corrected_text = fuzzy_correct(norm.cleaned, self.retriever._kb_vocab, max_dist=1)
        if corrected_text != norm.cleaned:
            logger.info(
                "Query fuzzy-corrected: %r → %r",
                norm.cleaned, corrected_text,
            )
            # Update norm so the entire downstream pipeline uses the corrected text
            norm.cleaned = corrected_text
        stages["fuzzy_correct"] = time.perf_counter() - t0

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
            )

        # ---- 4. CRAG-lite confidence gate ----
        top_score = retrieval.fusion.top_score
        tier = retrieval.fusion.confidence_tier

        if tier == "low":
            return self._fallback(
                reason="low_confidence_crag",
                stage_latencies_s=stages,
                t_start=t_start,
                normalization=norm,
                retrieval=retrieval,
                top_fused_score=top_score,
            )

        # ---- 5. Score-weighted selection (skip LLM) ----
        if skip_llm:
            if self._reranker is None:
                self._reranker = Reranker()
            best_id, best_score, all_scores = self._reranker.select(norm.cleaned, retrieval.candidates)
            retrieval.weighted_scores = {c.qa_id: s for c, s in zip(retrieval.candidates, all_scores)}

            if not best_id or best_id not in self.retriever.articles_by_id:
                return self._fallback(
                    reason="low_confidence_crag",
                    stage_latencies_s=stages,
                    t_start=t_start,
                    normalization=norm,
                    retrieval=retrieval,
                    top_fused_score=top_score,
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


    def _fallback(
        self,
        reason: str,
        stage_latencies_s: Dict[str, float],
        t_start: float,
        normalization: Optional[NormalizationResult] = None,
        retrieval: Optional[RetrievalResult] = None,
        generation: Optional[GenerationResult] = None,
        top_fused_score: float = 0.0,
    ) -> PipelineResult:
        """Build a fallback result."""
        result = PipelineResult(
            answer_mr=self._fallback_msg,
            is_fallback=True,
            fallback_reason=reason,
            chosen_qa_id="",
            chosen_context_idx=0,
            confidence_tier=retrieval.fusion.confidence_tier if retrieval else "unknown",
            top_fused_score=top_fused_score,
            cached=False,
            latency_s=time.perf_counter() - t_start,
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
