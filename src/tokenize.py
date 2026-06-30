"""
tokenize.py — Indic-aware tokenizer for Marathi BM25.

CRITICAL: Plain whitespace tokenization cripples Devanagari recall because:
  1. Marathi words are agglutinative — "प्रवेशासाठी" = प्रवेश + साठी (joined).
  2. Matras (vowel signs) attach to consonants; splitting on whitespace keeps
     whole-word forms that won't match paraphrased variants.
  3. Conjunct consonants (ज्ञ, क्ष, त्र, ज्ञ) need consistent handling.

Our tokenizer:
  - Splits Devanagari into syllable-level units (aksara) — robust to
    morphological variation ("प्रवेश" matches "प्रवेशासाठी" via syllable overlap).
  - Splits Latin (Romanized Marathi / English loanwords) on word boundaries
    and lowercases.
  - Strips Devanagari punctuation and STT filler tokens.
  - Drops pure-number tokens and 1-char Latin tokens (noise).

This is the lightweight approach — no model download required. For higher
quality you could swap in AI4Bharat IndicTokenize, but that pulls in heavy
deps; on a Pi, this syllable splitter is the right tradeoff.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

# ---------- Regexes (compiled once) ----------

# Devanagari Unicode ranges
_DEVANAGARI = r"\u0900-\u097F"
_DEV_NUMBERS = r"\u0966-\u096F"  # ०-९

# A Devanagari "aksara" (syllable) = optional consonant cluster + vowel sign
# Consonant = U+0915–U+0939 + optional nukta + optional virama + next consonant
# We approximate with: (consonant [+virama]*consonant*) + (vowel_sign | virama | nothing)
# This regex is the standard Sanskrit/Hindi/Marathi syllable splitter.
_AKSARA_RE = re.compile(
    rf"""
    (
      [{_DEVANAGARI}]{{1,}}   # one or more Devanagari chars (greedy — captures conjuncts)
      (?:
        \u094D[{_DEVANAGARI}]   # virama (halant) + consonant — i.e. conjunct continuation
      )*
    )
    | \d+ | [a-zA-Z]+
    """,
    re.VERBOSE,
)

# Cleaner syllable splitter (preferred): walk the string, group consonants
# joined by virama into one aksara, attach following vowel sign / matra.
_CONSONANT = lambda c: "\u0915" <= c <= "\u0939" or c in "\u093C\u0921\u0922\u0926\u0927"  # incl. nukta-ed
_VOWEL_SIGN = lambda c: c in "\u093E\u093F\u0940\u0941\u0942\u0943\u0944\u0946\u0947\u0948\u0949\u094A\u094B\u094C\u094E\u094F"
_VIRAMA = "\u094D"
_INDEPENDENT_VOWEL = lambda c: "\u0904" <= c <= "\u0914"
_DEV_PUNCT = "\u0964\u0965"  # danda, double danda

# Latin / general punctuation to strip
_PUNCT_RE = re.compile(r"[\s,।;:!?'\"\.\(\)\[\]\{\}<>\/\\|@#\$\%\^&\*\-=\+~`_…\u2013\u2014\u2018\u2019\u201C\u201D\u0964\u0965]+")

# STT filler tokens to drop (English Whisper hallucinations + Marathi fillers)
_STT_FILLERS = {
    "uh", "um", "ah", "er", "hmm", "mm", "hmm", "mmm",
    "thank you", "thanks", "please", "okay", "ok", "yeah", "yes", "no",
    "हम्म", "म्ह्म", "अरे", "अहो",
}

# Common Marathi stopwords (light list — we don't want to over-prune for FAQ matching)
# Kept intentionally small: only articles, be-verbs, and obvious fillers.
_MARATHI_STOPWORDS = {
    "आहे", "आहेत", "आहो", "आहात", "होता", "होती", "होते", "होत्या", "होतो",
    "काय", "कसे", "कशी", "कसा", "कसला", "कसली", "कसले",
    "हा", "ही", "हे", "तो", "ती", "ते", "त्या", "या",
    "आणि", "किंवा", "पण", "परंतु", "तर",
    "म्हणजे", "म्हणून", "त्यामुळे",
    "नाही", "नाहीत",
}


def _split_devanagari_aksaras(text: str) -> List[str]:
    """Split Devanagari text into aksara (syllable) units.

    An aksara = (consonant [virama consonant]*) + optional vowel sign + optional nukta.
    Independent vowels form their own aksara.
    """
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if not ("\u0900" <= c <= "\u097F"):
            i += 1
            continue

        # Start of an aksara
        start = i
        # Consume consonants joined by virama
        while i < n:
            ch = text[i]
            if "\u0915" <= ch <= "\u0939" or ch == "\u093C":
                i += 1
                # If next is virama, consume it; the next consonant joins this aksara
                if i < n and text[i] == _VIRAMA:
                    i += 1
                    continue
                else:
                    # Consume following vowel sign(s) and anusvara/visarga/chandrabindu
                    while i < n and (
                        text[i] in "\u093E\u093F\u0940\u0941\u0942\u0943\u0944\u0946\u0947\u0948\u0949\u094A\u094B\u094C\u094E\u094F\u093C\u0900\u0901\u0902\u0903"
                    ):
                        i += 1
                    break
            elif _INDEPENDENT_VOWEL(ch) or ch in "\u0900\u0901\u0902\u0903":
                # Independent vowel — own aksara (with following vowel sign if any — rare)
                i += 1
                while i < n and text[i] in "\u093E\u093F\u0940\u0941\u0942\u0943\u0944\u0946\u0947\u0948\u0949\u094A\u094B\u094C":
                    i += 1
                break
            else:
                # Other Devanagari (numbers, signs) — own token
                i += 1
                break

        aksara = text[start:i]
        if aksara:
            out.append(aksara)
    return out


def _split_latin_tokens(text: str) -> List[str]:
    """Split Latin text into lowercase word tokens, drop 1-char noise."""
    tokens = re.findall(r"[A-Za-z]+", text)
    return [t.lower() for t in tokens if len(t) > 1]


def tokenize(text: str, drop_stopwords: bool = False) -> List[str]:
    """Indic-aware tokenizer for Marathi BM25.

    Args:
        text: Input text (Devanagari, Latin, or mixed).
        drop_stopwords: If True, drop common Marathi stopwords. Default False
            (for retrieval we keep stopwords because query phrasing varies).

    Returns:
        List of tokens: Devanagari aksaras + Latin words, all lowercased for Latin.
    """
    if not text or not text.strip():
        return []

    # NFC normalize first — important for Devanagari (combining forms)
    text = unicodedata.normalize("NFC", text)

    # Strip STT filler phrases (longest-first to catch multi-word fillers)
    lower_for_fillers = text.lower()
    for filler in _STT_FILLERS:
        if filler in lower_for_fillers:
            # case-insensitive replace
            text = re.sub(re.escape(filler), " ", text, flags=re.IGNORECASE)

    # Strip all Devanagari + Latin punctuation, replace with space
    text = _PUNCT_RE.sub(" ", text)

    # Now walk the cleaned text and split into tokens
    tokens: List[str] = []
    # Find runs of Devanagari and runs of Latin, in order
    for m in re.finditer(rf"[{_DEVANAGARI}]+|[A-Za-z]+", text):
        chunk = m.group(0)
        if "\u0900" <= chunk[0] <= "\u097F":
            tokens.extend(_split_devanagari_aksaras(chunk))
        else:
            tokens.extend(_split_latin_tokens(chunk))

    if drop_stopwords:
        tokens = [t for t in tokens if t not in _MARATHI_STOPWORDS]

    return tokens


def tokenize_for_index(text: str) -> List[str]:
    """Tokenize a KB document (question + variants). Same as tokenize()."""
    return tokenize(text, drop_stopwords=False)


def tokenize_for_query(text: str) -> List[str]:
    """Tokenize a user query. Same as tokenize() — kept separate for future
    tuning (e.g., query expansion, stopword removal)."""
    return tokenize(text, drop_stopwords=False)


# ---------- Smoke test ----------
if __name__ == "__main__":
    samples = [
        "प्रवेश कशी घ्यायची",
        "admission kashi ghyaychi",
        "fees किती आहे आणि परीक्षा कधी आहे",
        "namaskar, मला शाळेबद्दल माहिती हवी आहे",
        "शिवाजी महाराज कोणत्या काळात झाले?",
        "uh, हम्म, मला वाटतं fees किती आहे",
        "",
    ]
    for s in samples:
        print(f"  IN : {s!r}")
        print(f"  OUT: {tokenize(s)}")
        print()
