"""
query_expand.py — Query expansion via Devanagari↔Roman transliteration
and fuzzy spell correction.

Expands the user query with transliterated variants to improve BM25 recall
when the KB stores names/questions in a different script. Also corrects
spelling drift from STT (Whisper) or typos via edit-distance matching
against the KB vocabulary.
"""
from __future__ import annotations

import re
from typing import List, Set

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

logger = __import__("logging").getLogger(__name__)

_DEVANAGARI_CHARS = re.compile(r"[\u0900-\u097F]")
_LATIN_CHARS = re.compile(r"[a-zA-Z]")


# ─────────────────────────────────────────────────────────────────
# Fuzzy spell correction (edit-distance against KB vocab)
# ─────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Bounded Levenshtein distance. Returns 999 if exceeds max_dist."""
    if abs(len(a) - len(b)) > max_dist:
        return 999
    if a == b:
        return 0
    # Standard DP with early exit
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + cost))
        if min(cur) > max_dist:
            return 999
        prev = cur
    return prev[-1]


def build_kb_vocab(articles) -> Set[str]:
    """Build vocabulary set from all KB questions, variants, and answers.

    Used for fuzzy spell correction. Includes both Roman and Devanagari words.
    """
    vocab: Set[str] = set()
    for art in articles:
        # KBArticle has: question, variants (list), answer_mr
        for text in [art.question, art.answer_mr] + list(art.variants):
            # Split on non-alphanumeric (keeps Devanagari letters together)
            for word in re.findall(r"[\u0900-\u097FA-Za-z]+", text):
                if len(word) >= 3:        # skip very short noise
                    vocab.add(word.lower())
    logger.info("KB vocab built: %d words", len(vocab))
    return vocab


def fuzzy_correct(query: str, vocab: Set[str], max_dist: int = 1) -> str:
    """Correct spelling drift in query using KB vocab.

    For each query word:
      - If already in vocab → keep as-is
      - If not in vocab → find nearest vocab word within edit distance
        max_dist, replace if found

    Conservative: only corrects words ≥4 chars, only edit distance 1 by default.
    Edit distance 2 is risky (too many false positives).

    Args:
        query: Normalized user query.
        vocab: KB vocabulary set (from build_kb_vocab).
        max_dist: Maximum edit distance for correction (default 1).

    Returns:
        Corrected query string.
    """
    if not vocab or not query.strip():
        return query

    words = re.findall(r"[\u0900-\u097FA-Za-z]+", query)
    if not words:
        return query

    corrected_words = []
    changes = 0
    for word in words:
        wl = word.lower()
        # Skip very short words — too many false positives
        if len(word) < 4:
            corrected_words.append(word)
            continue
        # Already in vocab? Keep as-is
        if wl in vocab:
            corrected_words.append(word)
            continue
        # Find nearest vocab word
        best_match = None
        best_dist = max_dist + 1
        for vw in vocab:
            if abs(len(vw) - len(wl)) > max_dist:
                continue  # quick prune
            d = _levenshtein(wl, vw, max_dist)
            if d < best_dist:
                best_dist = d
                best_match = vw
        if best_match and best_dist <= max_dist:
            corrected_words.append(best_match)
            changes += 1
        else:
            corrected_words.append(word)

    if changes > 0:
        logger.debug("Fuzzy corrected %d words: %r → %r", changes, words, corrected_words)

    # Reconstruct query with original separators (best-effort)
    # Simple approach: join with spaces
    return " ".join(corrected_words)


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_CHARS.search(text))


def has_latin(text: str) -> bool:
    return bool(_LATIN_CHARS.search(text))


def _normalize_roman(text: str) -> str:
    """Normalize HK transliteration to plain ASCII lowercase."""
    text = text.lower()
    replacements = {
        "ā": "a", "ī": "i", "ū": "u", "ṃ": "m", "ṛ": "r", "ṝ": "r",
        "ḥ": "h", "ñ": "n", "ṭ": "t", "ḍ": "d", "ṇ": "n", "ś": "s",
        "ṣ": "s", "ṅ": "n", "ḷ": "l", "ṁ": "m",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def expand_query(query: str) -> List[str]:
    """Generate transliterated variants for broader BM25 recall.

    Always includes the original query. Returns deduplicated list.
    """
    queries: List[str] = [query]

    if has_devanagari(query):
        try:
            roman = transliterate(query, sanscript.DEVANAGARI, sanscript.HK)
            roman = _normalize_roman(roman)
            if roman != query.lower():
                queries.append(roman)
        except Exception as e:
            logger.debug("DN→Roman transliteration failed: %s", e)

    if has_latin(query):
        try:
            dev = transliterate(query, sanscript.HK, sanscript.DEVANAGARI)
            if dev != query:
                queries.append(dev)
        except Exception as e:
            logger.debug("Roman→DN transliteration failed: %s", e)

    seen: set = set()
    deduped: List[str] = []
    for q in queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    if len(queries) > 1:
        logger.debug("Query expanded: %s -> %s", query, deduped)

    return deduped
