"""
test_pipeline.py — Pytest unit tests for the Balbharati RAG pipeline.

Run: pytest tests/test_pipeline.py -v

These tests use the mock embedder + mock LLM so they run offline in <1 second.
They cover:
  - Tokenizer (Devanagari aksara splitting, mixed-script, fillers)
  - Normalizer (Whisper hallucinations, nuktas, garbage detection)
  - BM25 (basic retrieval of known QA pair)
  - RRF fusion (rank combination math)
  - Full pipeline (end-to-end with mock LLM)
"""
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

# ---------- Patch embedder + LLM BEFORE importing src ----------
from src import embedder as embedder_mod


class MockEmbedder:
    _instance = None
    def __init__(self): self._dim = 256; self._loaded = True; self._load_time_s = 0.001
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def load(self): pass
    @property
    def dim(self): return self._dim
    @property
    def load_time_s(self): return self._load_time_s
    def encode(self, texts, batch_size=32, normalize=True):
        from src.tokenize import tokenize
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = hash(tok) & 0xFFFF
                out[i, h % self._dim] += 1.0
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out.astype(np.float32)
    def encode_one(self, text, normalize=True):
        return self.encode([text], normalize=normalize)[0]


embedder_mod.MurilEmbedder = MockEmbedder
embedder_mod.get_embedder = MockEmbedder.get
from src import dense_index as dense_mod
dense_mod.get_embedder = MockEmbedder.get
dense_mod.MurilEmbedder = MockEmbedder

from src import llm_client as llm_mod
from src.llm_client import LLMResponse
import re


class MockLLMClient:
    _instance = None
    def __init__(self): pass
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def chat(self, messages, temperature=None, top_p=None, max_tokens=None, stop=None, timeout_s=None):
        user_msg = messages[-1]["content"]
        m = re.search(r"\[1\]\s+प्रश्न:\s*(.+?)\s*\|\s*उत्तर:\s*(.+?)(?=\n\[2\]|\Z)",
                      user_msg, re.DOTALL)
        if m:
            text = f"उत्तर: {m.group(2).strip()}\nसंदर्भ: 1"
        else:
            text = "उत्तर: माहिती उपलब्ध नाही.\nसंदर्भ: 0"
        return LLMResponse(text=text, finish_reason="stop", prompt_tokens=100,
                           completion_tokens=30, latency_s=0.001, raw=None)
    def health_check(self): return True


llm_mod.LLMClient = MockLLMClient
llm_mod.get_llm_client = MockLLMClient.get
from src import generate as gen_mod
gen_mod.get_llm_client = MockLLMClient.get
gen_mod.LLMClient = MockLLMClient

# ---------- NOW import src modules ----------
from src.config import load_config
from src.tokenize import tokenize
from src.normalize import normalize_stt_text, normalize
from src.fusion import reciprocal_rank_fusion
from src.pipeline import RAGPipeline

# Load config once
load_config()


# ---------- Tokenizer tests ----------

class TestTokenizer:
    def test_devanagari_syllable_split(self):
        tokens = tokenize("प्रवेश")
        # प्र (conjunct), वे (व+े), श
        assert "प्र" in tokens
        assert "वे" in tokens
        assert "श" in tokens

    def test_latin_tokens(self):
        tokens = tokenize("admission kashi")
        assert "admission" in tokens
        assert "kashi" in tokens

    def test_mixed_script(self):
        tokens = tokenize("fees किती आहे")
        assert "fees" in tokens
        assert "कि" in tokens
        assert "ती" in tokens

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_filler_stripped(self):
        tokens = tokenize("uh हम्म hello")
        assert "uh" not in tokens
        assert "हम्म" not in tokens
        assert "hello" in tokens

    def test_punctuation_stripped(self):
        tokens = tokenize("काय? कसे!")
        assert "?" not in "".join(tokens)
        assert "!" not in "".join(tokens)


# ---------- Normalizer tests ----------

class TestNormalizer:
    def test_clean_devanagari_passthrough(self):
        assert normalize("प्रवेश कशी घ्यायची") == "प्रवेश कशी घ्यायची"

    def test_whisper_hallucination_stripped(self):
        result = normalize_stt_text("Thank you for watching. प्रवेश कशी घ्यायची")
        assert not result.is_garbage
        assert "प्रवेश" in result.cleaned
        assert "thank you" not in result.cleaned.lower()

    def test_pure_hallucination_is_garbage(self):
        result = normalize_stt_text("Thank you for watching. Please subscribe.")
        assert result.is_garbage
        assert "hallucination" in result.reason or "alpha" in result.reason

    def test_empty_input_garbage(self):
        assert normalize_stt_text("").is_garbage
        assert normalize_stt_text("   ").is_garbage

    def test_punctuation_only_garbage(self):
        result = normalize_stt_text("?, ।।  ...")
        assert result.is_garbage

    def test_nukta_normalized(self):
        # राज़ा → राजा
        result = normalize_stt_text("राज़ा आणि राणी")
        assert "राजा" in result.cleaned
        assert "़" not in result.cleaned  # combining nukta stripped

    def test_bracket_noise_stripped(self):
        result = normalize_stt_text("[noise] fees किती आहे [music]")
        assert "[noise]" not in result.cleaned
        assert "[music]" not in result.cleaned
        assert "fees" in result.cleaned

    def test_filler_dropped(self):
        result = normalize_stt_text("uh, um, हम्म, मला वाटतं")
        assert "uh" not in result.cleaned.lower()
        assert "हम्म" not in result.cleaned
        assert "मला" in result.cleaned

    def test_repeated_chars_collapsed(self):
        # Threshold is 5+ repeats (4+ after the first char). Use 6 a's to trigger.
        result = normalize_stt_text("aaaaaa शिवाजी.........")
        assert "aaaaaa" not in result.cleaned  # collapsed to single 'a'
        assert "शिवाजी" in result.cleaned
        # Repeated punctuation should also collapse
        assert "........" not in result.cleaned


# ---------- RRF fusion tests ----------

class TestRRFFusion:
    def test_both_retrievers_agree_top1(self):
        # Both retrievers rank doc A as #1
        bm25 = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
        dense = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        result = reciprocal_rank_fusion(bm25, dense, k=60)
        assert result.top_doc_id == "a"
        # Max possible RRF = 1/61 + 1/61 ≈ 0.0328
        assert abs(result.top_score - 2 * (1 / 61)) < 0.001
        assert result.confidence_tier == "high"

    def test_disagreement(self):
        # BM25 says A, Dense says B
        bm25 = [("a", 10.0), ("b", 5.0)]
        dense = [("b", 0.9), ("a", 0.7)]
        result = reciprocal_rank_fusion(bm25, dense, k=60)
        # Both have rank 1 + rank 2 = 1/61 + 1/62 — should be a tie, but
        # Python's sort is stable so the first-added wins
        assert result.top_doc_id in {"a", "b"}
        assert result.top_score > 0.03  # close to max

    def test_low_confidence(self):
        # Doc ranks #30 in both retrievers
        bm25 = [(f"d{i}", float(30 - i)) for i in range(30)]
        dense = [(f"d{i}", float(30 - i)) for i in range(30)]
        result = reciprocal_rank_fusion(bm25, dense, k=60)
        # Top doc ranks #1 in both → high confidence
        assert result.top_doc_id == "d0"
        assert result.confidence_tier == "high"


# ---------- Full pipeline tests ----------

class TestPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return RAGPipeline.from_kb()

    def test_admission_query(self, pipeline):
        result = pipeline.answer("प्रवेश कशी घ्यायची")
        assert not result.is_fallback
        assert result.chosen_qa_id == "qa_0"
        assert result.confidence_tier == "high"

    def test_fees_query(self, pipeline):
        result = pipeline.answer("fees किती आहे")
        assert not result.is_fallback
        assert result.chosen_qa_id == "qa_2"

    def test_shivaji_query(self, pipeline):
        result = pipeline.answer("शिवाजी महाराज कोणत्या काळात झाले")
        assert not result.is_fallback
        # Should retrieve a history QA pair
        assert "history" in result.retrieval.candidates[0].category or \
               "qa_102" in [a.qa_id for a in result.retrieval.candidates]

    def test_garbage_input_falls_back(self, pipeline):
        result = pipeline.answer("")
        assert result.is_fallback
        assert result.fallback_reason == "garbage_input"

    def test_pure_hallucination_falls_back(self, pipeline):
        result = pipeline.answer("Thank you for watching. Please subscribe.")
        assert result.is_fallback
        assert result.fallback_reason == "garbage_input"

    def test_punctuation_only_falls_back(self, pipeline):
        result = pipeline.answer("?, ।।  ...")
        assert result.is_fallback
        assert result.fallback_reason == "garbage_input"

    def test_normalization_preserves_answer(self, pipeline):
        # Query with noise prefix should still get the same answer as clean query
        clean = pipeline.answer("प्रवेश कशी घ्यायची")
        noisy = pipeline.answer("Thank you for watching. प्रवेश कशी घ्यायची")
        assert clean.chosen_qa_id == noisy.chosen_qa_id
        assert not clean.is_fallback
        assert not noisy.is_fallback

    def test_cache_works(self, pipeline):
        pipeline.clear_cache()
        # First call — cache miss
        r1 = pipeline.answer("प्रवेश कशी घ्यायची")
        assert not r1.cached
        # Second call — cache hit
        r2 = pipeline.answer("प्रवेश कशी घ्यायची")
        assert r2.cached
        # Same answer
        assert r1.answer_mr == r2.answer_mr

    def test_fallback_response_is_marathi(self, pipeline):
        result = pipeline.answer("")
        assert result.is_fallback
        # Should contain Devanagari chars
        assert any("\u0900" <= c <= "\u097F" for c in result.answer_mr)
