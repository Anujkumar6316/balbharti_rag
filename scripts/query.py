#!/usr/bin/env python3
"""
query.py — Interactive CLI for the Balbharati RAG pipeline.

Usage:
    python scripts/query.py "प्रवेश कशी घ्यायची"
    python scripts/query.py --repl
    python scripts/query.py --suite tests/rag_suite.yaml

In REPL mode, type queries one per line. Empty line = quit.
Suite mode runs a batch of queries from a YAML file and reports pass/fail.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pipeline import RAGPipeline


def print_result(result, verbose=False):
    """Pretty-print pipeline result. Verbose mode shows full debug trace."""
    print()
    if not verbose:
        # Simple output for non-verbose mode
        print("─" * 70)
        print(f"  Answer  : {result.answer_mr}")
        print(
            f"  Fallback: {result.is_fallback}"
            + (f" ({result.fallback_reason})" if result.is_fallback else "")
        )
        print(f"  Tier    : {result.confidence_tier}   (score: {result.top_fused_score:.4f})")
        print(f"  Ctx idx : {result.chosen_context_idx}   (qa_id: {result.chosen_qa_id or '-'})")
        print(f"  Cached  : {result.cached}")
        print(f"  Latency : {result.latency_s*1000:.0f} ms")
        print("─" * 70)
        return

    # ═══════════════════════════════════════════════════════════════
    # VERBOSE MODE — full debug trace
    # ═══════════════════════════════════════════════════════════════
    print("═" * 70)
    print("  QUERY DEBUG")
    print("═" * 70)

    # ── Input processing ──
    print("─" * 70)
    print("  INPUT PROCESSING")
    print("─" * 70)
    if result.normalization:
        print(f"  Raw input       : {result.normalization.raw!r}")
        print(f"  Normalized      : {result.normalization.cleaned!r}")
        print(f"  Transformations : {', '.join(result.normalization.transformations) or 'none'}")
        print(f"  Is garbage      : {result.normalization.is_garbage}"
              + (f" ({result.normalization.reason})" if result.normalization.is_garbage else ""))
    print(f"  Rule intent     : {result.query_intent}")

    # ── Retrieval ──
    if result.retrieval:
        print()
        print("─" * 70)
        print("  RETRIEVAL")
        print("─" * 70)
        print(f"  Latency: {result.retrieval.latency_s*1000:.1f}ms")
        print()

        # BM25 top-5
        print("  BM25 top-5:")
        bm25_sorted = sorted(
            result.retrieval.bm25_scores.items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        for i, (doc_id, score) in enumerate(bm25_sorted, 1):
            art = _find_article(result, doc_id)
            q_preview = art.question[:60] if art else "?"
            print(f"    {i}. {doc_id:8s}  score={score:6.3f}  Q: {q_preview}")

        print()
        print("  Dense top-5:")
        dense_sorted = sorted(
            result.retrieval.dense_scores.items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        for i, (doc_id, score) in enumerate(dense_sorted, 1):
            art = _find_article(result, doc_id)
            q_preview = art.question[:60] if art else "?"
            print(f"    {i}. {doc_id:8s}  score={score:6.3f}  Q: {q_preview}")

        print()
        print(f"  RRF fused top-5 (query_intent={result.retrieval.query_intent}):")
        for i, (art, score) in enumerate(zip(result.retrieval.candidates[:5],
                                              result.retrieval.fused_scores[:5]), 1):
            intent_tag = f" [{art.intent}]" if hasattr(art, 'intent') else ""
            print(f"    {i}. {art.qa_id:8s}  rrf={score:6.4f}{intent_tag}  Q: {art.question[:50]}")

        # Intent rerank log
        if result.retrieval.intent_rerank_log:
            print()
            print(f"  Intent rerank (query={result.retrieval.query_intent}):")
            for qa_id, cand_intent, old_s, new_s, action in result.retrieval.intent_rerank_log[:8]:
                symbol = "↑" if action == "boost" else ("↓" if action == "penalize" else "~")
                print(f"    {qa_id:8s}  [{cand_intent:8s}]  {old_s:.4f} → {new_s:.4f}  {symbol} {action}")

        # Reranker scores (skip_llm mode)
        if result.retrieval.weighted_scores:
            print()
            print("  Cross-encoder reranker scores:")
            rr_sorted = sorted(
                result.retrieval.weighted_scores.items(),
                key=lambda x: x[1], reverse=True
            )
            for i, (doc_id, score) in enumerate(rr_sorted[:5], 1):
                art = _find_article(result, doc_id)
                q_preview = art.question[:50] if art else "?"
                marker = " ⭐" if i == 1 else ""
                print(f"    {i}. {doc_id:8s}  score={score:6.4f}  Q: {q_preview}{marker}")

    # ── Generation ──
    if result.generation:
        print()
        print("─" * 70)
        print("  GENERATION (LLM)")
        print("─" * 70)
        g = result.generation
        print(f"  LLM intent      : {g.llm_intent}")
        print(f"  Context selected: {g.chosen_context_idx}")
        print(f"  Finish reason   : {g.finish_reason}")
        print(f"  Latency         : {g.latency_s*1000:.0f}ms")
        print(f"  Tokens          : {g.prompt_tokens} in / {g.completion_tokens} out")
        print(f"  Raw LLM output  : {g.raw_text!r}")

    # ── Final answer ──
    print()
    print("═" * 70)
    print("  FINAL ANSWER")
    print("═" * 70)
    print(f"  Answer    : {result.answer_mr}")
    print(f"  Source    : {result.chosen_qa_id or '(fallback)'}")
    print(f"  Fallback  : {result.is_fallback}"
          + (f" ({result.fallback_reason})" if result.is_fallback else ""))
    print(f"  Confidence: {result.confidence_tier} (score: {result.top_fused_score:.4f})")
    print(f"  Cached    : {result.cached}")
    print(f"  Total latency: {result.latency_s*1000:.0f}ms")
    print()
    print("  Stage breakdown:")
    for stage, t in result.stage_latencies_s.items():
        bar = "█" * int(t * 1000 / 50)  # 1 char per 50ms
        print(f"    {stage:20s} {t*1000:>7.1f}ms  {bar}")
    print("═" * 70)


def _find_article(result, qa_id):
    """Helper: find article by qa_id in retrieval candidates."""
    if result.retrieval:
        for art in result.retrieval.candidates:
            if art.qa_id == qa_id:
                return art
    return None


def load_suite(path: str) -> List[Dict[str, Any]]:
    """Load a YAML test suite.

    Expected schema:
        queries:
          - query: "प्रवेश कशी घ्यायची"
            expected_qa_id: "qa_0"        # optional
            expect_fallback: false         # optional
            description: "Admission process"  # optional
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "queries" not in data:
        raise ValueError(f"Suite file {path} must have a 'queries:' list")
    return data["queries"]


def run_suite(pipeline: RAGPipeline, suite_path: str, verbose=False, skip_llm=False) -> int:
    """Run a YAML test suite. Returns exit code (0=pass, 1=failures)."""
    queries = load_suite(suite_path)
    print(f"Loaded {len(queries)} queries from {suite_path}")
    print()

    passes = 0
    fails = 0
    no_expect = 0
    latencies = []

    for i, item in enumerate(queries, 1):
        query = item.get("query", "")
        # Note: we intentionally do NOT strip here, because empty/whitespace
        # queries are valid test cases (they should trigger garbage fallback).
        # The pipeline's normalizer handles them.
        expected_qa_id = item.get("expected_qa_id")
        expect_fallback = item.get("expect_fallback")
        description = item.get("description", "")

        t0 = time.perf_counter()
        result = pipeline.answer(query, skip_llm=skip_llm)
        lat = time.perf_counter() - t0
        latencies.append(lat)

        passed = None
        reason = ""
        if expected_qa_id is not None:
            if result.chosen_qa_id == expected_qa_id:
                passed = True
            else:
                passed = False
                reason = f"expected qa_id={expected_qa_id}, got {result.chosen_qa_id or '(fallback)'}"
        elif expect_fallback is not None:
            if result.is_fallback == expect_fallback:
                passed = True
            else:
                passed = False
                reason = f"expected fallback={expect_fallback}, got {result.is_fallback}"
        else:
            no_expect += 1

        if passed is True:
            passes += 1
            status = "PASS"
        elif passed is False:
            fails += 1
            status = "FAIL"
        else:
            status = "  - "  # no expectation

        query_display = query if query.strip() else "(empty)"
        print(f"[{i:>3}/{len(queries)}] {status}  {lat*1000:>5.0f}ms  {query_display[:50]}")
        if description:
            print(f"          -> {description}")
        if passed is False:
            print(f"          -> {reason}")
            print(f"          -> answer: {result.answer_mr[:80]}")
        elif verbose:
            print(f"          -> answer: {result.answer_mr[:80]}")

    print()
    print("─" * 70)
    print(f"  Suite   : {suite_path}")
    print(f"  Total   : {len(queries)}")
    print(f"  PASS    : {passes}")
    print(f"  FAIL    : {fails}")
    print(f"  No-expect: {no_expect}")
    if latencies:
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        print(f"  Latency : avg {avg*1000:.0f}ms / p95 {p95*1000:.0f}ms")
    print("─" * 70)

    return 0 if fails == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Balbharati RAG — interactive CLI / REPL / suite runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python scripts/query.py "प्रवेश कशी घ्यायची"\n'
            "  python scripts/query.py --repl\n"
            "  python scripts/query.py --repl --verbose\n"
            "  python scripts/query.py --suite tests/rag_suite.yaml\n"
        ),
    )
    parser.add_argument("query", nargs="?", help="One-shot query (Devanagari Marathi)")
    parser.add_argument(
        "--repl",
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Interactive REPL mode (default if no query and no suite)",
    )
    parser.add_argument(
        "--suite",
        help="Run queries from a YAML suite file (batch mode)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show stage latencies + retrieval debug",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM call, use score-weighted fusion to select best context",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override config.yaml path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
    load_config(args.config)

    print("Loading pipeline (first run takes ~3 s for MuRIL)...")
    t0 = time.perf_counter()
    pipeline = RAGPipeline.from_kb()
    print(f"Ready in {time.perf_counter()-t0:.2f}s. Cache size: {pipeline.cache_size}")

    skip_llm = args.skip_llm

    if skip_llm:
        print("  --skip-llm mode: using score-weighted fusion (no LLM call)")

    # Mode priority: --suite > one-shot query > --repl
    if args.suite:
        sys.exit(run_suite(pipeline, args.suite, verbose=args.verbose, skip_llm=skip_llm))

    if args.query:
        result = pipeline.answer(args.query, skip_llm=skip_llm)
        print_result(result, verbose=args.verbose)
        return

    # Default: REPL mode
    print("\nInteractive REPL. Type a query and press Enter. Empty line to quit.\n")
    while True:
        try:
            query = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            print("Bye.")
            break
        result = pipeline.answer(query, skip_llm=skip_llm)
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
