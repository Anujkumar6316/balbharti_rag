#!/usr/bin/env python3
"""
serve.py — FastAPI server exposing the RAG pipeline over HTTP.

Single endpoint for serial kiosk mode:
  POST /answer   { "text": "<STT output>" }
  → 200 OK       { "answer_mr": "...", "is_fallback": false, ... }

Also exposes:
  GET  /health   — Pi + llama-server health
  GET  /stats    — cache + latency stats
  POST /cache/clear  — clear LRU cache

Run:
    python scripts/serve.py --port 8000

Then test:
    curl -X POST http://localhost:8000/answer \
         -H "Content-Type: application/json" \
         -d '{"text":"प्रवेश कशी घ्यायची"}'
"""
# from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, load_config
from src.pipeline import RAGPipeline
from src.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Global pipeline (loaded once)
_pipeline: Optional[RAGPipeline] = None
_start_time: float = 0.0
_request_count: int = 0
_fallback_count: int = 0
_total_latency_s: float = 0.0


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


def create_app():
    """Create FastAPI app. We do this inside a function so the lifespan
    handler can initialize the pipeline (loads MuRIL model — slow)."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class AnswerRequest(BaseModel):
        text: str = Field(..., description="Raw STT output to answer")
        # Future: session_id, language_hint, etc.

    class AnswerResponse(BaseModel):
        answer_mr: str
        is_fallback: bool
        fallback_reason: str = ""
        confidence_tier: str
        top_fused_score: float
        latency_s: float
        cached: bool
        chosen_qa_id: str = ""

    app = FastAPI(
        title="Balbharati Marathi Voice-Agent RAG",
        version="1.0.0",
        description="Hybrid BM25+Dense+RRF RAG pipeline for Marathi (Devanagari).",
    )

    @app.on_event("startup")
    async def startup():
        global _pipeline, _start_time
        _start_time = time.perf_counter()
        logger.info("Starting RAG pipeline initialization...")
        _pipeline = RAGPipeline.from_kb()
        elapsed = time.perf_counter() - _start_time
        logger.info(f"Pipeline ready in {elapsed:.2f}s")

    @app.get("/health")
    async def health():
        """Health check — pipeline + llama-server reachability."""
        llm_ok = get_llm_client().health_check()
        uptime_s = time.perf_counter() - _start_time
        return {
            "status": "ok" if (_pipeline is not None and llm_ok) else "degraded",
            "pipeline_loaded": _pipeline is not None,
            "llama_server_reachable": llm_ok,
            "uptime_s": round(uptime_s, 1),
            "cache_size": _pipeline.cache_size if _pipeline else 0,
        }

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(req: AnswerRequest):
        """Answer a user query (from STT)."""
        global _request_count, _fallback_count, _total_latency_s
        if _pipeline is None:
            raise HTTPException(503, "Pipeline not ready")

        _request_count += 1
        result = _pipeline.answer(req.text)
        _total_latency_s += result.latency_s
        if result.is_fallback:
            _fallback_count += 1

        return AnswerResponse(
            answer_mr=result.answer_mr,
            is_fallback=result.is_fallback,
            fallback_reason=result.fallback_reason,
            confidence_tier=result.confidence_tier,
            top_fused_score=result.top_fused_score,
            latency_s=result.latency_s,
            cached=result.cached,
            chosen_qa_id=result.chosen_qa_id,
        )

    @app.get("/stats")
    async def stats():
        """Pipeline stats for kiosk monitoring."""
        avg_latency = (_total_latency_s / _request_count) if _request_count > 0 else 0.0
        return {
            "request_count": _request_count,
            "fallback_count": _fallback_count,
            "fallback_rate": (_fallback_count / _request_count) if _request_count > 0 else 0.0,
            "avg_latency_s": round(avg_latency, 3),
            "cache_size": _pipeline.cache_size if _pipeline else 0,
            "uptime_s": round(time.perf_counter() - _start_time, 1),
        }

    @app.post("/cache/clear")
    async def cache_clear():
        if _pipeline:
            _pipeline.clear_cache()
            return {"status": "cleared", "cache_size": 0}
        raise HTTPException(503, "Pipeline not ready")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_config(args.config)

    # Import here so config is loaded first
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
