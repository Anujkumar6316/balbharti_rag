#!/bin/bash
# =============================================================================
# Balbharati Marathi RAG Pipeline — Run Script
# Hybrid BM25 + Dense (MuRIL) + RRF + Qwen3-0.8B | Raspberry Pi 5
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make project root visible to child Python processes so config.py resolves
# relative paths (./kb, ./models, ./logs) against the project root, not the
# current working directory.
export RAG_HOME="${RAG_HOME:-$SCRIPT_DIR}"

# ── llama-server endpoint (optional override) ──────────────────────────────
# By default the pipeline talks to a local llama-server at http://127.0.0.1:8080.
# Override with env vars if you run llama-server on a different host/port:
#   export LLM_BASE_URL="http://192.168.1.10:8080/v1"
#   export LLM_MODEL="qwen3-0.8b-instruct"
# These are read by config.yaml's ${LLM_BASE_URL} / ${LLM_MODEL} placeholders
# (if you edit config.yaml to use them) or by setting them in the YAML directly.

MODE="${1:-help}"

case "$MODE" in
    rag|query|repl)
        echo "Starting RAG query / REPL..."
        shift
        python3 scripts/query.py "$@"
        ;;
    build-index|build)
        echo "Building hybrid KB index (BM25 + Dense MuRIL)..."
        shift
        python3 scripts/build_kb_index.py "$@"
        ;;
    eval|eval-kb)
        echo "Running RAG evaluation harness..."
        shift
        python3 scripts/eval.py "$@"
        ;;
    serve|api)
        echo "Starting FastAPI server..."
        shift
        PORT="${PORT:-8000}"
        python3 scripts/serve.py --host 0.0.0.0 --port "$PORT" "$@"
        ;;
    smoke|smoke-test)
        echo "Running offline smoke test (mock embedder + mock LLM)..."
        shift
        python3 scripts/smoke_test_offline.py "$@"
        ;;
    test|pytest)
        echo "Running pytest unit tests..."
        shift
        python3 -m pytest tests/ "$@"
        ;;
    help|*)
        echo ""
        echo "Balbharati Marathi RAG Pipeline"
        echo "================================"
        echo ""
        echo "Usage: ./run.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  rag          Interactive RAG REPL (type Marathi queries)"
        echo "  build-index  Build hybrid (BM25 + MuRIL dense) KB index"
        echo "  eval         Run retrieval / full-pipeline evaluation"
        echo "  serve        Start FastAPI HTTP server (/answer endpoint)"
        echo "  smoke        Offline smoke test (mock embedder + mock LLM)"
        echo "  test         Run pytest unit tests"
        echo "  help         Show this help message"
        echo ""
        echo "Options (for rag):"
        echo "  \"<query>\"              One-shot query, print answer and exit"
        echo "  --repl                  Interactive REPL mode (default if no query)"
        echo "  --suite <path>          Run queries from a YAML suite (batch mode)"
        echo "  --verbose, -v           Show stage latencies + retrieval debug"
        echo "  --config <path>         Override config.yaml path"
        echo ""
        echo "Options (for build-index):"
        echo "  --rebuild               Force rebuild indices"
        echo "  --smoke                 Run 3 sample queries after build"
        echo "  --config <path>         Override config.yaml path"
        echo ""
        echo "Options (for eval):"
        echo "  --mode retrieval        Retrieval-only (no LLM needed) [default]"
        echo "  --mode full             Full pipeline (requires llama-server running)"
        echo "  --mock                  Use mock embedder + mock LLM (offline test)"
        echo "  --limit N               Limit to first N queries"
        echo "  --regenerate-golden     Regenerate golden eval set from KB"
        echo "  --golden <path>         Path to golden_eval.jsonl"
        echo "  --output <path>         Write detailed results JSON"
        echo ""
        echo "Options (for serve):"
        echo "  --port N                HTTP port (default: 8000, override with PORT env)"
        echo "  --host <addr>           Bind address (default: 0.0.0.0)"
        echo "  --log-level <level>     DEBUG | INFO | WARNING | ERROR (default: INFO)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh rag                                                  # Interactive REPL"
        echo "  ./run.sh rag \"प्रवेश कशी घ्यायची\"                                # One-shot query"
        echo "  ./run.sh rag --repl --verbose                                 # Verbose REPL"
        echo "  ./run.sh rag --suite tests/rag_suite.yaml                     # Run regression suite"
        echo "  ./run.sh build-index --smoke                                  # Build + smoke test"
        echo "  ./run.sh build-index --rebuild                                # Force rebuild"
        echo "  ./run.sh eval --mode retrieval --mock                         # Offline retrieval eval"
        echo "  ./run.sh eval --mode full                                     # Full pipeline eval"
        echo "  ./run.sh eval --mock --limit 30                               # Quick 30-query test"
        echo "  ./run.sh eval --regenerate-golden                             # Rebuild golden set"
        echo "  PORT=9000 ./run.sh serve                                      # Start API on port 9000"
        echo "  ./run.sh serve --log-level DEBUG                              # Debug logging"
        echo "  ./run.sh smoke                                                # Offline smoke test"
        echo "  ./run.sh test                                                 # Pytest unit tests"
        echo "  ./run.sh test -v                                              # Verbose pytest"
        echo ""
        echo "Configuration:"
        echo "  config.yaml              All tunable params (BM25, RRF, LLM, cache)"
        echo "  RAG_HOME env var         Project root (default: script directory)"
        echo "  PORT env var             HTTP port for 'serve' command (default: 8000)"
        echo ""
        echo "Pipeline:"
        echo "  STT text → Normalize → [BM25 + Dense MuRIL] → RRF fusion"
        echo "           → CRAG-lite gate → Qwen3-0.8B → Devanagari Marathi answer"
        echo ""
        echo "Prerequisites (one-time setup):"
        echo "  1. pip install -r requirements.txt"
        echo "  2. Download Qwen3-0.8B-Instruct GGUF to ~/models/"
        echo "  3. Start llama-server on port 8080"
        echo "  4. ./run.sh build-index --smoke    # build + verify"
        echo "  See README.md for full setup instructions."
        echo ""
        ;;
esac
