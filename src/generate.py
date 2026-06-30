"""
generate.py — Single LLM call: intent classification + context selection.

The LLM outputs BOTH the query intent (WHY/WHAT/HOW/...) AND the best context
index (1/2/3/0) in ONE response. This avoids two separate LLM calls and keeps
total LLM latency at ~300-500ms (15 tokens at ~30 tok/s on Pi 5).

Output format from LLM (max 15 tokens):
  हेतू: WHY
  संदर्भ: 1

The final answer is served VERBATIM from the KB — no generation, no hallucination.

CRITICAL: max_tokens MUST stay ≤ 20. Setting it to 220 (old default) causes
8-10 second latency. The LLM only needs to output 2 short lines.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import get_config
from .llm_client import LLMClient, LLMResponse, get_llm_client
from .query_intent import parse_intent_from_llm, parse_context_from_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Prompts — combined intent + selection
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """तुम्ही बालभारती सहाय्यक आहात. खालील प्रश्नाचा हेतू ओळखा आणि सर्वात योग्य संदर्भ निवडा.

फॉरमॅट (काटेकोरपणे पाळा):
हेतू: <एकच शब्द>
संदर्भ: <एकच अंक>

हेतू शब्द: WHY, WHAT, HOW, WHEN, WHO, WHERE, HOW_MUCH, YES_NO, किंवा UNKNOWN
संदर्भ अंक: 1, 2, 3 (सर्वात योग्य संदर्भ), किंवा 0 (एकही योग्य नाही)

इतर काहीही लिहू नका."""

USER_TEMPLATE = """प्रश्न: {query}

संदर्भ:
[1] {ctx1}
[2] {ctx2}
[3] {ctx3}

हेतू आणि संदर्भ द्या:"""


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Output of the generator."""
    answer_mr: str
    chosen_context_idx: int        # 0 = no context, 1-3 = which top-K
    llm_intent: str                # intent extracted from LLM response
    llm_signaled_dont_know: bool
    finish_reason: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    raw_response: Optional[LLMResponse] = None
    raw_text: str = ""             # raw LLM output text, for debug


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _format_context(qa_pair) -> str:
    """Format a QA pair: 'प्रश्न: <Q> | उत्तर: <A>' (Devanagari variant preferred)."""
    if hasattr(qa_pair, "question"):
        q = qa_pair.question
        variants = qa_pair.variants
        answer_mr = qa_pair.answer_mr
    else:
        q = qa_pair.get("question", "")
        variants = qa_pair.get("variants", [])
        answer_mr = qa_pair.get("answer", {}).get("mr", "")

    devanagari_variants = [
        v for v in variants
        if any("\u0900" <= ch <= "\u097F" for ch in v)
    ]
    q_display = devanagari_variants[0] if devanagari_variants else q
    return f"प्रश्न: {q_display} | उत्तर: {answer_mr}"


def _get_answer(candidate) -> str:
    if hasattr(candidate, "answer_mr"):
        return candidate.answer_mr
    return candidate.get("answer", {}).get("mr", "")


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    top_k_candidates: List,
    confidence_tier: str = "high",
    llm_client: Optional[LLMClient] = None,
) -> GenerationResult:
    """Combined LLM call: intent + context selection. Answer from KB verbatim.

    Args:
        query: Normalized + fuzzy-corrected user query.
        top_k_candidates: Top-K QA pairs from retrieval (post intent-rerank).
        confidence_tier: "high" | "medium" | "low" (logged only).
        llm_client: Optional injected client (for testing).

    Returns:
        GenerationResult with answer_mr from chosen KB article.
    """
    cfg = get_config()
    client = llm_client or get_llm_client()

    # No candidates → immediate fallback (no LLM call)
    if not top_k_candidates:
        return GenerationResult(
            answer_mr=cfg["pipeline"]["fallback_message_mr"],
            chosen_context_idx=0,
            llm_intent="UNKNOWN",
            llm_signaled_dont_know=True,
            finish_reason="no_candidates",
            latency_s=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    # Pad to exactly 3
    filler = {"question": "(रिक्त)", "variants": [], "answer": {"mr": "(कोणताही संदर्भ नाही)"}}
    candidates = list(top_k_candidates)
    while len(candidates) < 3:
        candidates.append(filler)

    ctx1 = _format_context(candidates[0])
    ctx2 = _format_context(candidates[1])
    ctx3 = _format_context(candidates[2])

    user_msg = USER_TEMPLATE.format(query=query, ctx1=ctx1, ctx2=ctx2, ctx3=ctx3)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # CRITICAL: max_tokens ≤ 20 for combined call
    # 15 tokens = enough for "हेतू: WHY\nसंदर्भ: 1"
    # 220 tokens = 8-10s latency (NEVER do this for selection mode)
    llm_resp = client.chat(
        messages=messages,
        temperature=0.0,           # deterministic
        max_tokens=15,             # combined intent + selection
        stop=["\n\n", "[4]"],
        timeout_s=cfg["llm"]["timeout_s"],
    )

    # Parse both fields from single LLM response
    llm_intent = parse_intent_from_llm(llm_resp.text)
    chosen_idx = parse_context_from_llm(llm_resp.text)

    # If LLM said context 0 → don't know
    dont_know = chosen_idx == 0

    if dont_know or not (1 <= chosen_idx <= len(candidates)):
        final_answer = cfg["pipeline"]["fallback_message_mr"]
        dont_know = True
    else:
        final_answer = _get_answer(candidates[chosen_idx - 1])
        if not final_answer:
            final_answer = cfg["pipeline"]["fallback_message_mr"]
            chosen_idx = 0
            dont_know = True

    return GenerationResult(
        answer_mr=final_answer,
        chosen_context_idx=chosen_idx,
        llm_intent=llm_intent,
        llm_signaled_dont_know=dont_know,
        finish_reason=llm_resp.finish_reason,
        latency_s=llm_resp.latency_s,
        prompt_tokens=llm_resp.prompt_tokens,
        completion_tokens=llm_resp.completion_tokens,
        raw_response=llm_resp,
        raw_text=llm_resp.text,
    )
