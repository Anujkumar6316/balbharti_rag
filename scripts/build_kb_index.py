#!/usr/bin/env python3
"""
build_kb_index.py — Build the BM25 + Dense indices from the KB JSON.

This is a one-time operation (run once at install, or when KB changes).
On a Raspberry Pi 5 8GB with MuRIL small, building the dense index for
~200 QA pairs (with ~10 variants each, ~2000 total embeddings) takes
about 60-120 seconds. After that, the index lives in RAM and queries
are sub-millisecond.

Usage:
    python scripts/build_kb_index.py
    python scripts/build_kb_index.py --rebuild    # force rebuild
    python scripts/build_kb_index.py --smoke      # build + run 3 sample queries
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, load_config
from src.kb import load_kb
from src.retrieve import HybridRetriever
from src.bm25_index import BM25Index
from src.dense_index import DenseIndex


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_indices():
    """Build and return the hybrid retriever. Also prints diagnostics."""
    cfg = get_config()
    kb_path = cfg["kb"]["path"]
    print(f"[1/4] Loading KB from: {kb_path}")
    t0 = time.perf_counter()
    articles, meta = load_kb(kb_path)
    print(f"      Loaded {len(articles)} articles in {time.perf_counter()-t0:.2f}s")
    print(f"      KB version: {meta.get('version')}, language: {meta.get('language')}")

    # Quick category summary
    cats = {}
    for a in articles:
        cats[a.category] = cats.get(a.category, 0) + 1
    print(f"      Categories: {cats}")

    print(f"[2/4] Building BM25 index (Indic syllable tokenizer)...")
    t0 = time.perf_counter()
    bm25 = BM25Index.from_config()
    bm25.add_docs([a.qa_id for a in articles], [a.doc_text for a in articles])
    bm25.build()
    print(f"      BM25 built in {time.perf_counter()-t0:.2f}s — vocab size: {len(bm25._df)}")
    print(f"      Avg doc length: {bm25._avg_dl:.1f} tokens")

    print(f"[3/4] Building Dense index (MiniLM small)...")
    print(f"      This is the slow step on Pi — encoding ~{sum(len(a.variant_list) for a in articles)} variants...")
    t0 = time.perf_counter()
    dense = DenseIndex.from_config()
    dense.build_from_variants(
        [a.qa_id for a in articles],
        [a.variant_list for a in articles],
    )
    print(f"      Dense built in {time.perf_counter()-t0:.2f}s — dim: {dense._dim}")

    print(f"[4/4] Assembling hybrid retriever...")
    retriever = HybridRetriever(
        bm25=bm25,
        dense=dense,
        articles_by_id_map={a.qa_id: a for a in articles},
    )
    print(f"      Done.")
    print()
    print(f"  Total QA pairs indexed : {len(articles)}")
    print(f"  BM25 vocab size        : {len(bm25._df)}")
    print(f"  Dense embedding dim    : {dense._dim}")
    print(f"  Embedder load time     : {dense._embedder.load_time_s:.2f}s")
    return retriever, articles


def smoke_test(retriever, articles):
    """Run 3 sample queries to verify retrieval works."""
    print("\n--- Smoke test (3 sample queries) ---")
    samples = [
        "प्रवेश कशी घ्यायची",       # admission
        "fees किती आहे",              # fees (mixed script)
        "शिवाजी महाराज कोणत्या काळात झाले",  # history
    ]
    for q in samples:
        result = retriever.retrieve(q, top_k=3)
        print(f"\nQuery: {q!r}")
        print(f"  Latency: {result.latency_s*1000:.1f}ms")
        print(f"  Tier: {result.fusion.confidence_tier} (top score: {result.top_fused_score:.4f})")
        for i, (art, score) in enumerate(zip(result.candidates, result.fused_scores)):
            print(f"    [{i+1}] {art.qa_id} (score={score:.4f}, cat={art.category})")
            print(f"        Q: {art.question[:80]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test after build")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging("INFO")
    load_config(args.config)

    retriever, articles = build_indices()

    if args.smoke:
        smoke_test(retriever, articles)

    print("\n[OK] Index build complete. The pipeline will reuse these in-memory indices.")


if __name__ == "__main__":
    main()
