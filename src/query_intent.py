"""
query_intent.py — Intent extraction + intent-aware candidate re-ranking.

Two-layer approach (NO separate LLM call for intent):
  1. extract_intent() — rule-based, ~5ms, handles ~70% of queries
  2. parse_intent_from_llm() — extracts intent from the COMBINED LLM
     response (intent + context selection in one call). No extra LLM call.

The LLM's combined output looks like:
  हेतू: WHY
  संदर्भ: 1

We parse both fields from that single response.

Used by:
  - kb.py at index time → tag each KBArticle.intent (rule-based)
  - retrieve.py at query time → rerank top-K candidates by intent match
  - generate.py at query time → parse LLM's intent from combined response
"""
from __future__ import annotations

import re
from typing import List, Tuple

logger = __import__("logging").getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Rule-based intent extraction (fast path, ~5ms)
# ─────────────────────────────────────────────────────────────────

# Ordered: phrases first (longest match), then single words.
# YES_NO phrases must be checked before bare "ka" (WHY).
_INTENT_KEYWORDS: List[Tuple[str, str]] = [
    # YES/NO — check FIRST
    ("hote ka", "YES_NO"),
    ("aahe ka", "YES_NO"),
    ("ahe ka", "YES_NO"),
    ("hoteka", "YES_NO"),
    ("होते का", "YES_NO"),
    ("आहे का", "YES_NO"),
    ("होतेका", "YES_NO"),

    # WHY — का
    ("ka", "WHY"),
    ("kaa", "WHY"),
    ("karun", "WHY"),
    ("karan", "WHY"),
    ("का", "WHY"),
    ("कारण", "WHY"),

    # WHAT — काय
    ("kaya", "WHAT"),
    ("kay", "WHAT"),
    ("mhaneje", "WHAT"),
    ("mhanaje", "WHAT"),
    ("काय", "WHAT"),
    ("म्हणजे", "WHAT"),

    # HOW — कसा/कशी/कसे
    ("kasha", "HOW"),
    ("kashi", "HOW"),
    ("kase", "HOW"),
    ("kasa", "HOW"),
    ("kashe", "HOW"),
    ("कसा", "HOW"),
    ("कशी", "HOW"),
    ("कसे", "HOW"),

    # WHEN — कधी
    ("kadhi", "WHEN"),
    ("kada", "WHEN"),
    ("कधी", "WHEN"),

    # WHO — कोण
    ("kona", "WHO"),
    ("kon", "WHO"),
    ("koni", "WHO"),
    ("कोण", "WHO"),
    ("कोणी", "WHO"),

    # WHERE — कुठे
    ("kuthe", "WHERE"),
    ("kuth", "WHERE"),
    ("कुठे", "WHERE"),

    # HOW MUCH — किती
    ("kiti", "HOW_MUCH"),
    ("किती", "HOW_MUCH"),
]


def extract_intent(text: str) -> str:
    """Extract intent from Marathi question via rules. ~5ms.

    Returns: WHY/WHAT/HOW/WHEN/WHO/WHERE/HOW_MUCH/YES_NO/UNKNOWN
    """
    if not text or not text.strip():
        return "UNKNOWN"

    lower = text.lower()
    lower = re.sub(r"\s+", " ", lower)

    for keyword, intent in _INTENT_KEYWORDS:
        if len(keyword) <= 4:
            # Word-boundary match for short keywords
            pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
            if re.search(pattern, lower):
                return intent
        else:
            if keyword in lower:
                return intent

    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────
# LLM output parser (for combined intent + selection call)
# ─────────────────────────────────────────────────────────────────

# Matches: "हेतू: WHY" or "intent: WHY" or "हेतू:WHY"
_INTENT_RE = re.compile(
    r"(?:हेतू|intent)\s*:\s*([A-Za-z_]+)",
    re.IGNORECASE,
)
# Matches: "संदर्भ: 1" or "context: 1" or "संदर्भ:1"
_CONTEXT_RE = re.compile(
    r"(?:संदर्भ|context|reference)\s*:\s*([0-3])",
    re.IGNORECASE,
)


def parse_intent_from_llm(text: str) -> str:
    """Extract intent from LLM's combined response.

    Expected format: "हेतू: WHY\nसंदर्भ: 1"

    Returns: WHY/WHAT/HOW/.../UNKNOWN
    """
    if not text:
        return "UNKNOWN"
    m = _INTENT_RE.search(text)
    if not m:
        return "UNKNOWN"
    intent = m.group(1).upper().strip()
    # Validate against known intents
    valid = {"WHY", "WHAT", "HOW", "WHEN", "WHO", "WHERE",
             "HOW_MUCH", "YES_NO", "UNKNOWN"}
    if intent in valid:
        return intent
    # Try to fix common LLM variations
    intent_lower = intent.lower()
    if "why" in intent_lower or "का" in intent:
        return "WHY"
    if "what" in intent_lower or "काय" in intent:
        return "WHAT"
    if "how" in intent_lower or "कस" in intent:
        return "HOW"
    if "when" in intent_lower or "कधी" in intent:
        return "WHEN"
    if "who" in intent_lower or "कोण" in intent:
        return "WHO"
    if "where" in intent_lower or "कुठ" in intent:
        return "WHERE"
    return "UNKNOWN"


def parse_context_from_llm(text: str) -> int:
    """Extract context index from LLM's combined response.

    Returns: 0 (none), 1, 2, or 3
    """
    if not text:
        return 0
    m = _CONTEXT_RE.search(text)
    if not m:
        # Fallback: find first standalone digit 0-3
        m2 = re.search(r"(?<!\w)([0-3])(?!\w)", text)
        if m2:
            return int(m2.group(1))
        return 0
    return int(m.group(1))


# ─────────────────────────────────────────────────────────────────
# Intent-aware reranking
# ─────────────────────────────────────────────────────────────────

def rerank_by_intent(
    candidates: List,
    fused_scores: List[float],
    query_intent: str,
    match_boost: float = 1.5,
    mismatch_penalty: float = 0.3,
) -> Tuple[List, List[float], List[Tuple[str, str, float, float, str]]]:
    """Re-rank candidates by intent match.

    Returns:
        (reordered_candidates, reordered_scores, debug_log)
        debug_log = list of (qa_id, cand_intent, original_score, adjusted_score, action)
    """
    if query_intent == "UNKNOWN":
        return candidates, fused_scores, []

    adjusted: List[Tuple[float, int]] = []
    debug_log: List[Tuple[str, str, float, float, str]] = []

    for i, (cand, score) in enumerate(zip(candidates, fused_scores)):
        cand_intent = _get_intent(cand)
        if cand_intent == query_intent:
            adj = score * match_boost
            action = "boost"
        elif cand_intent == "UNKNOWN":
            adj = score
            action = "neutral"
        else:
            adj = score * mismatch_penalty
            action = "penalize"

        qa_id = _get_qa_id(cand)
        debug_log.append((qa_id, cand_intent, score, adj, action))
        adjusted.append((adj, i))

    adjusted.sort(key=lambda x: (-x[0], x[1]))

    reordered_cands = [candidates[i] for _, i in adjusted]
    reordered_scores = [adjusted[j][0] for j in range(len(adjusted))]

    return reordered_cands, reordered_scores, debug_log


def _get_intent(candidate) -> str:
    if hasattr(candidate, "intent"):
        return candidate.intent or "UNKNOWN"
    return candidate.get("intent", "UNKNOWN")


def _get_qa_id(candidate) -> str:
    if hasattr(candidate, "qa_id"):
        return candidate.qa_id
    return candidate.get("qa_id", "?")
