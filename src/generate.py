"""
generate.py — LLM-based context selector, answer served from KB verbatim.

The LLM only outputs a context index (1, 2, 3, or 0 for none).
The final answer is the KB answer from the chosen candidate — no generation needed.

This cuts generation latency from ~10s to ~150ms on Pi CPU.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from .config import get_config
from .llm_client import LLMClient, LLMResponse, get_llm_client

logger = logging.getLogger(__name__)


# ---------- Prompts (selection-only) ----------

SYSTEM_PROMPT = """तुम्ही बालभारतीचे सहायक आहात. खाली दिलेल्या तीन संदर्भांपैकी प्रश्नाला सर्वात योग्य उत्तर देणारा संदर्भ निवडा.

फक्त एक संख्या लिहा: 1, 2, किंवा 3. जर एकही संदर्भ योग्य नसेल तर 0 लिहा."""

USER_TEMPLATE = """प्रश्न: {query}

[1] {ctx1}
[2] {ctx2}
[3] {ctx3}

सर्वात योग्य संदर्भ क्रमांक:"""


# ---------- Response parsing ----------

# Extract first number from LLM output
_NUM_RE = re.compile(r"(\d+)")


@dataclass
class GenerationResult:
    """Output of the generator."""
    answer_mr: str                 # final Marathi answer (may be fallback msg)
    chosen_context_idx: int        # 0 = no context, 1-3 = which top-K was used
    llm_signaled_dont_know: bool   # LLM explicitly said "don't know"
    finish_reason: str             # from LLM
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    raw_response: Optional[LLMResponse] = None


def _format_context(qa_pair) -> str:
    """Format a QA pair as context for the LLM.

    Accepts either a KBArticle (from retriever) or a dict (for testing).
    Shows: the Devanagari question (preferred) + the Marathi answer.
    """
    # Handle both KBArticle (dataclass) and dict
    if hasattr(qa_pair, "question"):
        q = qa_pair.question
        variants = qa_pair.variants
        answer_mr = qa_pair.answer_mr
    else:
        q = qa_pair.get("question", "")
        variants = qa_pair.get("variants", [])
        answer_mr = qa_pair.get("answer", {}).get("mr", "")

    # Pick the Devanagari variant if available, else canonical question
    devanagari_variants = [
        v for v in variants
        if any("\u0900" <= ch <= "\u097F" for ch in v)
    ]
    q_display = devanagari_variants[0] if devanagari_variants else q
    return f"प्रश्न: {q_display} | उत्तर: {answer_mr}"


def _parse_llm_response(text: str) -> tuple[int, bool]:
    """Parse LLM output to extract selected context index.

    Returns:
        (chosen_context_idx, llm_signaled_dont_know)
    """
    if not text:
        return (0, True)

    match = _NUM_RE.search(text)
    if not match:
        return (0, True)

    idx = int(match.group(1))
    if idx not in (1, 2, 3):
        return (0, True)

    return (idx, False)


def _get_answer(candidate) -> str:
    """Extract answer_mr from a KBArticle or dict candidate."""
    if hasattr(candidate, "answer_mr"):
        return candidate.answer_mr
    return candidate.get("answer", {}).get("mr", "")


def generate_answer(
    query: str,
    top_k_candidates: List,
    confidence_tier: str = "high",
    llm_client: Optional[LLMClient] = None,
) -> GenerationResult:
    """Select best context via LLM, serve answer verbatim from KB.

    Args:
        query: Normalized user query (Devanagari Marathi).
        top_k_candidates: List of QA-pair objects (top-K from retrieval).
            Each must have .question, .variants, .answer_mr (KBArticle)
            or be a dict with 'question', 'answer.mr'.
        confidence_tier: "high" | "medium" | "low" (logged for analysis).
        llm_client: Optional injected client (for testing).

    Returns:
        GenerationResult with answer_mr from the chosen KB article.
    """
    cfg = get_config()
    client = llm_client or get_llm_client()

    if not top_k_candidates:
        return GenerationResult(
            answer_mr=cfg["pipeline"]["fallback_message_mr"],
            chosen_context_idx=0,
            llm_signaled_dont_know=True,
            finish_reason="no_candidates",
            latency_s=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    # Pad candidates to exactly 3
    filler = {"question": "(रिक्त)", "variants": [], "answer": {"mr": "(कोणताही संदर्भ नाही)"}}
    while len(top_k_candidates) < 3:
        top_k_candidates.append(filler)

    ctx1 = _format_context(top_k_candidates[0])
    ctx2 = _format_context(top_k_candidates[1])
    ctx3 = _format_context(top_k_candidates[2])

    user_msg = USER_TEMPLATE.format(query=query, ctx1=ctx1, ctx2=ctx2, ctx3=ctx3)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    llm_resp = client.chat(
        messages=messages,
        temperature=cfg["llm"]["temperature"],
        max_tokens=5,
        timeout_s=cfg["llm"]["timeout_s"],
    )

    chosen_idx, dont_know = _parse_llm_response(llm_resp.text)

    if dont_know or not (1 <= chosen_idx <= len(top_k_candidates)):
        final_answer = cfg["pipeline"]["fallback_message_mr"]
        dont_know = True
    else:
        final_answer = _get_answer(top_k_candidates[chosen_idx - 1])

    return GenerationResult(
        answer_mr=final_answer,
        chosen_context_idx=chosen_idx,
        llm_signaled_dont_know=dont_know,
        finish_reason=llm_resp.finish_reason,
        latency_s=llm_resp.latency_s,
        prompt_tokens=llm_resp.prompt_tokens,
        completion_tokens=llm_resp.completion_tokens,
        raw_response=llm_resp,
    )


# ---------- Smoke test (requires llama-server running) ----------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python generate.py '<query>'")
        sys.exit(1)

    query = sys.argv[1]
    # Fake top-3 from KB
    fake_candidates = [
        {
            "question": "fees kiti aahe",
            "variants": ["फी किती आहे", "शुल्क किती आहे"],
            "answer": {"mr": "वार्षिक शुल्क ५००० रुपये आहे. ते दोन हप्त्यांमध्ये भरता येते."},
        },
        {
            "question": "pariksha kadhi aahe",
            "variants": ["परीक्षा कधी आहे"],
            "answer": {"mr": "पहिली परीक्षा ऑक्टोबरमध्ये आणि दुसरी परीक्षा मार्चमध्ये होते."},
        },
        {
            "question": "pustak kuthe miltat",
            "variants": ["पुस्तक कुठे मिळतात"],
            "answer": {"mr": "बालभारतीची पुस्तके शाळेमध्ये मिळतात."},
        },
    ]
    result = generate_answer(query, fake_candidates)
    print(f"\n--- RESULT ---")
    print(f"Answer (mr): {result.answer_mr}")
    print(f"Chosen ctx : {result.chosen_context_idx}")
    print(f"Dont-know : {result.llm_signaled_dont_know}")
    print(f"Latency   : {result.latency_s:.2f}s")
    print(f"Tokens    : {result.prompt_tokens} in / {result.completion_tokens} out")
