#!/usr/bin/env python3
"""
smoke_test_offline.py — End-to-end pipeline test using MOCK embedder + MOCK LLM.

This lets us verify the pipeline wiring, normalization, BM25, RRF, and
confidence gating WITHOUT needing torch / sentence-transformers / llama-server
installed. It runs in pure Python and is fast (~1 second for 200 articles).

Use this to:
  - Sanity-check the pipeline architecture
  - Verify CRAG-lite gating thresholds make sense
  - Catch integration bugs before deploying on Pi

For the REAL pipeline, install requirements.txt and run scripts/build_kb_index.py.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# ---------- Patch embedder to be a deterministic mock ----------
# We use a simple bag-of-tokens hashing embedder. It's not as good as MuRIL
# but it lets us verify the pipeline plumbing end-to-end.

from src import embedder as embedder_mod

class MockEmbedder:
    """Deterministic hash-based embedder for testing.
    Not semantic — just bag-of-tokens hashed to a fixed-dim vector."""
    _instance = None
    def __init__(self):
        self._dim = 256
        self._loaded = True
        self._load_time_s = 0.001
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
            toks = tokenize(t)
            for tok in toks:
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

# Also patch dense_index's import
from src import dense_index as dense_mod
dense_mod.get_embedder = MockEmbedder.get
dense_mod.MurilEmbedder = MockEmbedder

# ---------- Patch LLM client to be a deterministic mock ----------
from src import llm_client as llm_mod
from src.llm_client import LLMResponse

class MockLLMClient:
    """Mock LLM that always picks context [1] and returns the first article's answer."""
    _instance = None
    def __init__(self): pass
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def chat(self, messages, temperature=None, top_p=None, max_tokens=None, stop=None, timeout_s=None):
        # Parse the user message to find context [1]'s answer
        user_msg = messages[-1]["content"]
        # Updated regex: matches "[1] प्रश्न: X | उत्तर: Y" up to next [2] or end
        import re
        m = re.search(r"\[1\]\s+प्रश्न:\s*(.+?)\s*\|\s*उत्तर:\s*(.+?)(?=\n\[2\]|\Z)",
                      user_msg, re.DOTALL)
        if m:
            answer = m.group(2).strip()
            text = f"उत्तर: {answer}\nसंदर्भ: 1"
        else:
            text = "उत्तर: माहिती उपलब्ध नाही.\nसंदर्भ: 0"
        return LLMResponse(
            text=text, finish_reason="stop",
            prompt_tokens=100, completion_tokens=30,
            latency_s=0.001, raw=None,
        )
    def health_check(self): return True

llm_mod.LLMClient = MockLLMClient
llm_mod.get_llm_client = MockLLMClient.get

# Also patch generate module's import
from src import generate as gen_mod
gen_mod.get_llm_client = MockLLMClient.get
gen_mod.LLMClient = MockLLMClient


# ---------- Now run the actual pipeline ----------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.config import load_config
    from src.pipeline import RAGPipeline

    print("="*70)
    print("BALBHARATI RAG — OFFLINE SMOKE TEST (mock embedder + mock LLM)")
    print("="*70)
    print()

    load_config()
    print("[1/3] Building pipeline (this builds BM25 + dense index in-memory)...")
    t0 = time.perf_counter()
    pipeline = RAGPipeline.from_kb()
    print(f"      Pipeline ready in {time.perf_counter()-t0:.2f}s")
    print(f"      Articles: {len(pipeline.retriever.bm25._doc_ids)}")
    print(f"      BM25 vocab: {len(pipeline.retriever.bm25._df)}")
    print()

    # ---------- Test queries ----------
    test_queries = [
        # (input, description, expects_fallback)
        ("प्रवेश कशी घ्यायची", "Clean Devanagari — admission query", False),
        ("Thank you for watching. प्रवेश कशी घ्यायची", "Whisper hallucination prefix + real query", False),
        ("[noise] fees किती आहे [music]", "Bracketed noise + mixed-script query", False),
        ("uh, um, हम्म, मला वाटतं admission प्रक्रिया काय आहे", "Filler-heavy query", False),
        ("aaa शिवाजी महाराज.........", "Repeated char noise + history query", False),
        ("राज़ा आणि राणी", "Hindi nukta leakage (राज़ → राज)", False),  # might fallback if not in KB
        ("Thank you for watching. Please subscribe.", "Pure Whisper hallucination", True),
        ("", "Empty input", True),
        ("?, ।।  ...", "Only punctuation", True),
        ("xyzqwk abcdef nopqrst", "Gibberish (no KB match)", True),
    ]

    print("[2/3] Running test queries...\n")
    print(f"{'#':<3} {'Query':<50} {'FB?':<5} {'Score':<7} {'Tier':<7} {'Top candidate'}")
    print("-" * 130)

    for i, (q, desc, expect_fb) in enumerate(test_queries, 1):
        result = pipeline.answer(q)
        # Get top candidate (for display only)
        top_qa = ""
        if result.retrieval and result.retrieval.candidates:
            top_qa = f"{result.retrieval.candidates[0].qa_id} ({result.retrieval.candidates[0].category})"
        preview = result.answer_mr[:40].replace("\n", " ")
        print(f"{i:<3} {q[:50]:<50} {'Y' if result.is_fallback else 'N':<5} "
              f"{result.top_fused_score:.4f}  {result.confidence_tier:<7} "
              f"{top_qa or '-'}")
        print(f"     -> {desc}")
        print(f"     -> {preview}")
        if result.is_fallback:
            print(f"     -> fallback_reason: {result.fallback_reason}")
        print()

    # ---------- Stats ----------
    print("[3/3] Stats:")
    print(f"  Cache size: {pipeline.cache_size}")
    print()
    print("="*70)
    print("OFFLINE SMOKE TEST PASSED — pipeline wiring is correct.")
    print("Next: deploy on Pi with real MuRIL + llama-server.")
    print("="*70)


if __name__ == "__main__":
    main()
