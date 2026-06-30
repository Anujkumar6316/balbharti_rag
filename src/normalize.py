"""
normalize.py — STT output normalization for Devanagari Marathi.

This module sits BETWEEN the STT model and the retrieval pipeline. Its job is
to clean up the messy text that real STT models (especially Whisper) emit:

1. Whisper hallucinations — "Thank you for watching.", "Please subscribe.",
   "Aaj tak", "[Music]", "[Applause]" — these appear because Whisper was
   trained on YouTube and emits them on silence / noise.

2. Background noise / VAD artifacts — "(background noise)", "[__NOISE__]",
   "  ".

3. Unicode normalization — Devanagari has combining forms; NFC ensures
   consistent representation (matra + consonant vs pre-composed).

4. Devanagari-specific cleanup —
     - Strip chandrabindu/anusvara inconsistencies (lightly — keep them as
       they carry meaning, but normalize to canonical form).
     - Normalize nukta variants (ज़ → ज, etc. — Marathi doesn't use nuktas,
       they leak in from Hindi training data).
     - Normalize ZWJ/ZWNJ (zero-width joiners) — STT sometimes emits them.

5. Punctuation cleanup — collapse runs of "?", "!", "," into single tokens.

6. Trimming — strip leading/trailing whitespace, repeated internal whitespace.

7. Empty / garbage detection — if after cleaning the text is too short or
   is only punctuation, return None so the pipeline can fire the fallback.

This module is PURE (no model calls). It runs in microseconds on a Pi.

NOTE: We intentionally DO NOT do Roman→Devanagari transliteration here.
That requires the AI4Bharat Xlit model (~30MB) which we can add later.
For now, if STT occasionally emits Romanized chunks, the tokenizer handles
them — BM25 still matches Roman variants in the KB, and dense embeddings
(MuRIL) are cross-script.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# ---------- Whisper hallucination patterns ----------
# These are the most common English hallucinations Whisper emits on silence/noise.
# Source: common observation from Whisper Marathi/Hindi evals.
_WHISPER_HALLUCINATIONS_EN = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe for more",
    "like and subscribe",
    "don't forget to subscribe",
    "thanks for watching, please subscribe",
    "thank you for watching, please subscribe",
    "music",
    "applause",
    "laughter",
    "[music]",
    "[applause]",
    "[laughter]",
    "(music)",
    "(applause)",
    "(laughter)",
    "[background music]",
    "[__noise__]",
    "[noise]",
    "[silence]",
    "[inaudible]",
    "[crosstalk]",
    "aaj tak",
    "you",
    "so,",
]

# Build a single regex for fast matching (longest-first to avoid partials)
_HALLUCINATION_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(_WHISPER_HALLUCINATIONS_EN, key=len, reverse=True)),
    flags=re.IGNORECASE,
)

# Background noise / non-speech markers (brackets, parentheses, square brackets)
_BRACKET_NOISE_RE = re.compile(
    r"[\(\[][\s\w_]*?(?:noise|silence|inaudible|crosstalk|music|applause|laughter|background|static|beep)[\s\w_]*?[\)\]]",
    flags=re.IGNORECASE,
)
_SQUARE_BRACKET_ALL_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")

# Pure ASCII filler words to drop (single tokens, longer phrases handled above)
_FILLER_TOKENS = {"uh", "um", "ah", "er", "hmm", "mm", "mmm", "uhh", "umm", "yeah", "yep", "nope"}

# Marathi STT fillers (often emitted on hesitation)
_FILLER_TOKENS_MR = {"हम्म", "म्हम", "म्म", "अरेराव", "अहो"}

# Repeated character noise — STT sometimes emits "aaaaaaa" or "............"
_REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")
_REPEAT_PUNCT_RE = re.compile(r"([।?!,\.\-])\1{1,}")

# ZWJ/ZWNJ (zero-width joiners) — strip, they don't carry meaning in modern Marathi
_ZW_RE = re.compile(r"[\u200C\u200D\uFEFF]")

# Nukta — Marathi doesn't use these, they leak from Hindi training data
# (ज़ → ज, ख़ → ख, ग़ → ग, ड़ → ड, ढ़ → ढ, फ़ → फ, क़ → क)
# str.maketrans only accepts single-char keys, so:
#   - combining nukta U+093C  → strip via translate (single char)
#   - nukta-ed consonant pairs → replace via dict (multi-char)
_NUKTA_STRIP = str.maketrans({"़": ""})  # U+093C combining nukta
_NUKTA_PAIR_REPLACEMENTS = {
    "ज़": "ज",
    "ख़": "ख",
    "ग़": "ग",
    "ड़": "ड",
    "ढ़": "ढ",
    "फ़": "फ",
    "क़": "क",
}

# Devanagari punctuation to normalize
_DEV_PUNCT_NORMALIZE = {
    "।।": "।",   # double danda → single
    "॥": "।",     # double danda (verse) → single
}

# Collapse whitespace
_WS_RE = re.compile(r"\s+")

# Minimum acceptable cleaned length (chars)
_MIN_CLEAN_LEN = 2
# Minimum acceptable token count after tokenization
_MIN_TOKENS = 1


@dataclass
class NormalizationResult:
    """Output of the normalizer."""
    raw: str               # original STT text
    cleaned: str           # cleaned text (may be empty if garbage)
    is_garbage: bool       # True if input was effectively empty/noise
    reason: str            # if is_garbage, why ("empty", "hallucination_only", etc.)
    transformations: list  # list of cleanup steps applied (for debugging)


def _strip_hallucinations(text: str) -> str:
    """Remove Whisper hallucination phrases."""
    return _HALLUCINATION_RE.sub(" ", text)


def _strip_bracket_noise(text: str) -> str:
    """Remove [noise]/(music)/[silence] markers."""
    text = _BRACKET_NOISE_RE.sub(" ", text)
    # Also strip any remaining square-bracket tags like [__BLANK__]
    text = _SQUARE_BRACKET_ALL_RE.sub(" ", text)
    return text


def _normalize_devanagari(text: str) -> str:
    """Devanagari-specific normalization."""
    # Strip ZWJ/ZWNJ
    text = _ZW_RE.sub("", text)
    # Strip combining nuktas (Marathi doesn't use them; Hindi STT leakage)
    text = text.translate(_NUKTA_STRIP)
    # Replace nukta-ed consonant pairs (ज़ → ज, etc.)
    for src, dst in _NUKTA_PAIR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Normalize Devanagari punctuation
    for src, dst in _DEV_PUNCT_NORMALIZE.items():
        text = text.replace(src, dst)
    return text


def _collapse_noise(text: str) -> str:
    """Collapse repeated chars/punctuation and whitespace."""
    text = _REPEAT_CHAR_RE.sub(r"\1", text)       # aaaaa → a
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)       # ??? → ?
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _drop_filler_tokens(text: str) -> str:
    """Drop standalone filler tokens (uh, um, हम्म, etc.)."""
    if not text:
        return text
    words = text.split(" ")
    kept = []
    for w in words:
        wl = w.lower().strip("।?,.!\"'")
        if wl in _FILLER_TOKENS or wl in _FILLER_TOKENS_MR:
            continue
        if w:  # skip empty (can happen if double space)
            kept.append(w)
    return " ".join(kept)


def normalize_stt_text(raw: str) -> NormalizationResult:
    """Normalize STT (Whisper) output for downstream retrieval.

    Pipeline (all pure-Python, microsecond-scale):
      1. NFC unicode normalization
      2. Strip bracketed noise markers
      3. Strip Whisper hallucinations (English)
      4. Devanagari normalization (nukta, ZWJ, punctuation)
      5. Drop filler tokens (uh, um, हम्म)
      6. Collapse repeated chars / punctuation / whitespace
      7. Garbage detection — empty or too short

    Args:
        raw: Raw text from STT model.

    Returns:
        NormalizationResult with cleaned text + metadata.
    """
    transformations = []

    if raw is None:
        return NormalizationResult(
            raw="", cleaned="", is_garbage=True,
            reason="null_input", transformations=[]
        )

    text = raw.strip()
    if not text:
        return NormalizationResult(
            raw=raw, cleaned="", is_garbage=True,
            reason="empty_input", transformations=[]
        )

    # 1. NFC normalization (combines combining chars into canonical form)
    text = unicodedata.normalize("NFC", text)
    transformations.append("nfc")

    # 2. Strip bracketed noise
    before = text
    text = _strip_bracket_noise(text)
    if text != before:
        transformations.append("strip_bracket_noise")

    # 3. Strip Whisper hallucinations
    before = text
    text = _strip_hallucinations(text)
    if text != before:
        transformations.append("strip_hallucination")

    # 4. Devanagari normalization
    before = text
    text = _normalize_devanagari(text)
    if text != before:
        transformations.append("normalize_devanagari")

    # 5. Drop filler tokens
    before = text
    text = _drop_filler_tokens(text)
    if text != before:
        transformations.append("drop_fillers")

    # 6. Collapse noise
    text = _collapse_noise(text)
    transformations.append("collapse_noise")

    # 7. Garbage detection
    is_garbage = False
    reason = ""
    if not text:
        is_garbage = True
        reason = "empty_after_cleanup"
    elif len(text.strip()) < _MIN_CLEAN_LEN:
        is_garbage = True
        reason = "too_short_after_cleanup"
    elif not any(c.isalpha() for c in text):
        # All punctuation/numbers/whitespace
        is_garbage = True
        reason = "no_alpha_chars"

    return NormalizationResult(
        raw=raw,
        cleaned=text,
        is_garbage=is_garbage,
        reason=reason,
        transformations=transformations,
    )


def normalize(raw: str) -> Optional[str]:
    """Convenience wrapper: return cleaned text or None if garbage."""
    result = normalize_stt_text(raw)
    if result.is_garbage:
        return None
    return result.cleaned


# ---------- Smoke test ----------
if __name__ == "__main__":
    samples = [
        # Clean Devanagari
        "प्रवेश कशी घ्यायची",
        # Whisper hallucination prefix
        "Thank you for watching. प्रवेश कशी घ्यायची",
        # Background noise tag
        "[noise] fees किती आहे [music]",
        # Filler-heavy
        "uh, um, हम्म, मला वाटतं admission प्रक्रिया काय आहे",
        # Repeated chars
        "aaa शिवाजी महाराज.........",
        # Nukta (Hindi leakage)
        "राज़ा आणि राणी",
        # Pure hallucination
        "Thank you for watching. Please subscribe.",
        # Empty
        "",
        # Only punctuation
        "?, ।।  ...",
    ]
    for s in samples:
        r = normalize_stt_text(s)
        print(f"  IN : {s!r}")
        print(f"  OUT: {r.cleaned!r}  (garbage={r.is_garbage}, reason={r.reason!r})")
        print(f"  X  : {r.transformations}")
        print()
