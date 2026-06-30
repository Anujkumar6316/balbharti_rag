#!/usr/bin/env python3
"""
eval.py — Evaluation harness for the RAG pipeline.

Two modes:
  1. Retrieval-only eval (no LLM needed) — measures BM25/Dense/RRF quality.
     Runs in seconds, useful for tuning BM25 params and RRF k.
     Usage: python scripts/eval.py --mode retrieval

  2. Full pipeline eval (needs llama-server running) — measures end-to-end
     accuracy, fallback rate, latency.
     Usage: python scripts/eval.py --mode full

The golden set is in data/golden_eval.jsonl (one JSON per line):
  {"query": "...", "expected_qa_id": "qa_42", "note": "..."}

If the golden set doesn't exist, we generate one from the KB by:
  - For each QA pair, take the Devanagari variant as the "query"
  - Mark the QA's own qa_id as the expected match
This gives a quick upper-bound sanity check (retrieval should be ~perfect
when querying with a known variant).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, load_config
from src.kb import load_kb
from src.retrieve import HybridRetriever
from src.normalize import normalize_stt_text


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def generate_golden_from_kb(articles, output_path: Path) -> List[dict]:
    """Generate a held-out golden set from the KB.

    Strategy: for each article, pick one random Devanagari variant as the
    query. Mark that article's qa_id as the expected match.
    """
    import random
    random.seed(42)  # reproducible

    golden = []
    for art in articles:
        # Prefer Devanagari variants (matches our STT output assumption)
        dev_vars = [v for v in art.variants if any("\u0900" <= c <= "\u097F" for c in v)]
        if dev_vars:
            query = random.choice(dev_vars)
        elif art.variants:
            query = random.choice(art.variants)
        else:
            query = art.question
        golden.append({
            "query": query,
            "expected_qa_id": art.qa_id,
            "category": art.category,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in golden:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(golden)} golden eval items at {output_path}")
    return golden


def load_golden(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def eval_retrieval(retriever: HybridRetriever, golden: List[dict], top_k: int = 3) -> dict:
    """Evaluate retrieval-only (no LLM)."""
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    fallback_low = 0
    fallback_medium = 0
    fallback_high = 0
    latencies = []

    per_query_results = []

    for item in golden:
        query = item["query"]
        expected = item["expected_qa_id"]

        # Run through normalizer first (real pipeline does this)
        norm = normalize_stt_text(query)
        if norm.is_garbage:
            print(f"  WARN: garbage query: {query!r}")
            continue

        t0 = time.perf_counter()
        result = retriever.retrieve(norm.cleaned, top_k=5)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        top_ids = [a.qa_id for a in result.candidates]
        if expected in top_ids[:1]:
            hits_at_1 += 1
        if expected in top_ids[:3]:
            hits_at_3 += 1
        if expected in top_ids[:5]:
            hits_at_5 += 1

        if result.fusion.confidence_tier == "low":
            fallback_low += 1
        elif result.fusion.confidence_tier == "medium":
            fallback_medium += 1
        else:
            fallback_high += 1

        per_query_results.append({
            "query": query,
            "expected": expected,
            "got_top1": top_ids[0] if top_ids else "",
            "top5": top_ids,
            "hit_at_1": expected in top_ids[:1],
            "hit_at_3": expected in top_ids[:3],
            "tier": result.fusion.confidence_tier,
            "top_score": result.fusion.top_score,
            "latency_ms": lat * 1000,
        })

    n = len(per_query_results)
    summary = {
        "n": n,
        "hit_at_1": hits_at_1,
        "hit_at_3": hits_at_3,
        "hit_at_5": hits_at_5,
        "hit_at_1_rate": hits_at_1 / n if n else 0,
        "hit_at_3_rate": hits_at_3 / n if n else 0,
        "hit_at_5_rate": hits_at_5 / n if n else 0,
        "avg_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else 0,
        "p95_latency_ms": (sorted(latencies)[int(len(latencies) * 0.95)] * 1000) if latencies else 0,
        "tier_distribution": {
            "low": fallback_low,
            "medium": fallback_medium,
            "high": fallback_high,
        },
    }
    return summary, per_query_results


def eval_full(pipeline, golden: List[dict]) -> dict:
    """Full pipeline eval (with LLM). Slower — rate-limit if needed."""
    correct = 0
    fallback_count = 0
    latencies = []
    results = []

    for i, item in enumerate(golden):
        query = item["query"]
        expected = item["expected_qa_id"]

        result = pipeline.answer(query)
        latencies.append(result.latency_s)

        if result.is_fallback:
            fallback_count += 1
        elif result.chosen_qa_id == expected:
            correct += 1

        results.append({
            "query": query[:60],
            "expected": expected,
            "chosen": result.chosen_qa_id,
            "correct": result.chosen_qa_id == expected,
            "is_fallback": result.is_fallback,
            "fallback_reason": result.fallback_reason,
            "latency_s": result.latency_s,
            "answer_preview": result.answer_mr[:80],
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(golden)} — acc so far: {correct/(i+1):.2%}")

    n = len(results)
    summary = {
        "n": n,
        "correct": correct,
        "fallback": fallback_count,
        "accuracy": correct / n if n else 0,
        "fallback_rate": fallback_count / n if n else 0,
        "avg_latency_s": sum(latencies) / n if n else 0,
        "p95_latency_s": sorted(latencies)[int(n * 0.95)] if n else 0,
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--golden", default=None, help="Path to golden_eval.jsonl")
    parser.add_argument("--regenerate-golden", action="store_true",
                        help="Regenerate golden set from KB")
    parser.add_argument("--limit", type=int, default=None, help="Limit N queries (for quick test)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None, help="Write detailed results JSON here")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock embedder + mock LLM (offline testing)")
    args = parser.parse_args()

    setup_logging("INFO")
    load_config(args.config)

    # If --mock, patch the embedder + LLM before any retriever/pipeline code runs
    if args.mock:
        print("[mock mode] Patching embedder + LLM with offline mocks...")
        import numpy as np
        from src import embedder as embedder_mod

        class MockEmbedder:
            _instance = None
            def __init__(self): self._dim=256; self._loaded=True; self._load_time_s=0.001
            @classmethod
            def get(cls):
                if cls._instance is None: cls._instance = cls()
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
                if cls._instance is None: cls._instance = cls()
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

    cfg = get_config()

    project_root = Path(cfg["_project_root"])
    golden_path = Path(args.golden) if args.golden else project_root / "data" / "golden_eval.jsonl"

    # Load KB + build retriever
    print("Loading KB and building indices...")
    articles, _ = load_kb(cfg["kb"]["path"])
    retriever = HybridRetriever.build(articles)
    print(f"  {len(articles)} articles indexed")

    # Generate or load golden
    if args.regenerate_golden or not golden_path.exists():
        golden = generate_golden_from_kb(articles, golden_path)
    else:
        golden = load_golden(golden_path)
        print(f"Loaded {len(golden)} golden items from {golden_path}")

    if args.limit:
        golden = golden[:args.limit]
        print(f"Limited to first {len(golden)} items")

    if args.mode == "retrieval":
        print(f"\n=== RETRIEVAL-ONLY EVAL ({len(golden)} queries) ===")
        summary, per_query = eval_retrieval(retriever, golden)
        print("\n--- Summary ---")
        print(f"  Hit@1 : {summary['hit_at_1']}/{summary['n']}  ({summary['hit_at_1_rate']:.2%})")
        print(f"  Hit@3 : {summary['hit_at_3']}/{summary['n']}  ({summary['hit_at_3_rate']:.2%})")
        print(f"  Hit@5 : {summary['hit_at_5']}/{summary['n']}  ({summary['hit_at_5_rate']:.2%})")
        print(f"  Avg latency : {summary['avg_latency_ms']:.1f}ms")
        print(f"  P95 latency : {summary['p95_latency_ms']:.1f}ms")
        print(f"  Tier distribution: {summary['tier_distribution']}")

        # Show first 10 failures (hit@1 misses)
        failures = [r for r in per_query if not r["hit_at_1"]][:10]
        if failures:
            print(f"\n--- First 10 Hit@1 misses ---")
            for r in failures:
                print(f"  Q: {r['query'][:60]}")
                print(f"    Expected: {r['expected']}, Got: {r['got_top1']}, Top5: {r['top5']}")
                print(f"    Tier: {r['tier']}, Score: {r['top_score']:.4f}")

    elif args.mode == "full":
        print(f"\n=== FULL PIPELINE EVAL ({len(golden)} queries) ===")
        print("  (Requires llama-server running at http://127.0.0.1:8080)")
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline(retriever=retriever)
        summary, results = eval_full(pipeline, golden)
        print("\n--- Summary ---")
        print(f"  N         : {summary['n']}")
        print(f"  Correct   : {summary['correct']}  ({summary['accuracy']:.2%})")
        print(f"  Fallback  : {summary['fallback']}  ({summary['fallback_rate']:.2%})")
        print(f"  Avg latency : {summary['avg_latency_s']:.2f}s")
        print(f"  P95 latency : {summary['p95_latency_s']:.2f}s")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_query": per_query if args.mode == "retrieval" else results},
                      f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()
