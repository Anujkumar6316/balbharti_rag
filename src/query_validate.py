"""
query_validate.py — Score-based query validation (zero hardcoded words).

Two-stage gate:
  1. Structural checks (pre-retrieval): length, token repetition
  2. Score checks (post-retrieval): dense cosine, BM25 raw, reranker logit

No word lists, no keyword blocklists. The retrieval scores themselves are
the validation signal. Adapts automatically as KB grows.

This is the production-scalable replacement for hardcoded keyword filtering.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

logger = __import__("logging").getLogger(__name__)

_TOKEN_SPLIT_RE = re.compile(r"[\s,।;:!?'\"\.\(\)\[\]\{\}<>\/\\|@#$%\^&\*\-=\+~`_…\u0964\u0965]+")


# ─────────────────────────────────────────────────────────────────
# Stage 1: Structural validation (pre-retrieval)
# ─────────────────────────────────────────────────────────────────

def validate_structure(text: str, cfg: dict, kb_vocab: set = None) -> Tuple[bool, str]:
    """Structural checks before retrieval. ~0.5ms. No word lists.

    Catches:
      - Too short / too long queries
      - Repetitive tokens ("hello hello hello")
      - Empty after normalization
      - Queries with tokens absent from the entire KB (impossible facts)

    Args:
        text: Normalized query text.
        cfg: Config dict (query_validation section).
        kb_vocab: Full KB vocabulary set for vocab coverage check.
                  If None, skips the vocab check.

    Returns:
        (is_valid, reason)
    """
    if not cfg.get("enabled", True):
        return (True, "")

    if not text or not text.strip():
        return (False, "empty")

    text = text.strip()
    struct_cfg = cfg.get("structural", {})

    # Length bounds
    min_len = struct_cfg.get("min_length", 3)
    max_len = struct_cfg.get("max_length", 200)
    if len(text) < min_len:
        return (False, "too_short")
    if len(text) > max_len:
        return (False, "too_long")

    # Token repetition (structural — language-agnostic)
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text.lower()) if t]
    if not tokens:
        return (False, "no_tokens")

    max_repeat = struct_cfg.get("max_token_repeat_ratio", 0.6)
    if len(tokens) >= 3:
        token_counts = Counter(tokens)
        most_common_count = token_counts.most_common(1)[0][1]
        repeat_ratio = most_common_count / len(tokens)
        if repeat_ratio > max_repeat:
            return (False, "repetitive")

    # Vocab coverage (pre-retrieval): reject if many query words don't exist
    # anywhere in the KB. Uses the same word-extraction regex as build_kb_vocab
    # (not aksara-level tokens) so vocab is directly comparable.
    # Catches anachronisms ("aircraft", "rocket") and out-of-domain terms
    # without any hardcoded word lists — the KB itself defines what's valid.
    if kb_vocab is not None:
        max_missing = struct_cfg.get("max_missing_token_ratio", 0.4)
        # Match build_kb_vocab extraction: whole Devanagari/Latin words
        q_words = [t.lower() for t in re.findall(r"[\u0900-\u097FA-Za-z]+", text)
                   if len(t) >= 4]
        if q_words:
            missing = sum(1 for t in q_words if t not in kb_vocab)
            missing_ratio = missing / len(q_words)
            if missing_ratio > max_missing:
                return (False, f"vocab_mismatch ({missing}/{len(q_words)} words unknown to KB)")

    return (True, "")


# ─────────────────────────────────────────────────────────────────
# Stage 2: Score-based validation (post-retrieval)
# ─────────────────────────────────────────────────────────────────

def validate_scores(
    dense_top_score: float,
    bm25_top_score: float,
    reranker_score: float = None,
    token_coverage: float = 1.0,
    cfg: dict = None,
) -> Tuple[bool, str]:
    """Score-based checks after retrieval. ~0ms (scores already computed).

    Catches:
      - Gibberish (low dense cosine = no semantic match)
      - Out-of-scope queries (low BM25 = no keyword overlap)
      - Wrong-topic matches (low reranker = cross-encoder not confident)
      - Entity mismatch (low token coverage = query entities absent from candidates)

    No word lists — thresholds calibrate to your actual KB score distribution.
    As KB grows, score distributions stay valid (they're relative to KB content).

    Args:
        dense_top_score: Top dense cosine similarity (0.0-1.0).
        bm25_top_score: Top raw BM25 score (unbounded, typically 0-15).
        reranker_score: Cross-encoder logit (optional, None if skip-llm mode).
        token_coverage: Fraction (0-1) of non-stopword query tokens present in
                        candidate texts. 1.0 = all tokens found.
        cfg: config dict (query_validation.score section).

    Returns:
        (is_valid, reason)
    """
    if cfg is None or not cfg.get("enabled", True):
        return (True, "")

    score_cfg = cfg.get("score", {})

    # Dense cosine gate — catches gibberish + OOS English
    min_dense = score_cfg.get("min_dense_cosine", 0.35)
    if dense_top_score < min_dense:
        return (False, f"low_dense_cosine ({dense_top_score:.3f} < {min_dense})")

    # BM25 raw gate — catches queries with no keyword overlap
    min_bm25 = score_cfg.get("min_bm25_raw", 3.0)
    if bm25_top_score < min_bm25:
        return (False, f"low_bm25 ({bm25_top_score:.3f} < {min_bm25})")

    # Reranker gate (optional — only in skip-llm mode)
    if reranker_score is not None:
        min_reranker = score_cfg.get("min_reranker_score", 0.35)
        if reranker_score < min_reranker:
            return (False, f"low_reranker ({reranker_score:.3f} < {min_reranker})")

    # Token coverage gate — catches entity mismatch (e.g. "Sambhaji" vs "Shivaji")
    # and impossible facts (e.g. "aircraft" absent from all candidate texts).
    min_coverage = score_cfg.get("min_token_coverage", 0.5)
    if token_coverage < min_coverage:
        return (False, f"low_token_coverage ({token_coverage:.2f} < {min_coverage})")

    return (True, "")


def compute_token_coverage(query: str, candidates: List) -> float:
    """Fraction of query content words (≥4 chars) present in candidate texts.

    Score-based entity/domain consistency check:
      - If a query mentions "Sambhaji" but all candidates are about "Shivaji",
        coverage drops → entity mismatch.
      - If a query mentions "aircraft" and no candidate contains that word,
        coverage drops → impossible fact / out of domain.

    Uses whole-word extraction (not aksara tokens) to match entity names
    at the word level. No word lists needed — the KB defines what's valid.

    Args:
        query: Normalized user query.
        candidates: List of KBArticle objects from retrieval.

    Returns:
        Fraction (0.0-1.0) of long query words present in candidate texts.
    """
    q_words = [t.lower() for t in re.findall(r"[\u0900-\u097FA-Za-z]+", query)
               if len(t) >= 4]
    if not q_words:
        return 1.0
    candidate_text = " ".join(
        f"{c.question} {c.answer_mr}"
        for c in candidates
    ).lower()
    matched = sum(1 for t in q_words if t in candidate_text)
    return matched / len(q_words)
