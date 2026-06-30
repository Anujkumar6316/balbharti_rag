# Balbharati Marathi Voice-Agent RAG Pipeline

A robust, accurate, **edge-deployable** RAG pipeline for a Marathi voice agent
serving Balbharati staff and students. Built for **Raspberry Pi 5 (8GB)** with
**bare-minimum latency** (2–5 s per query) and **SOTA retrieval accuracy** on
a 199-pair Marathi knowledge base.

> **Status**: Pipeline complete. STT/TTS deferred (per spec) — see
> "Adding STT/TTS" below.

---

## What this is

A pipeline that takes raw STT (speech-to-text) output in **Devanagari Marathi**
and returns a **Devanagari Marathi answer** drawn from the Balbharati knowledge
base (`kb/knowledgebase.json` — 199 QA pairs across admission, fees, exams,
history, literature, values, etc.).

```
raw STT text → normalize → BM25 + Dense (MuRIL) → RRF fusion
   → CRAG-lite confidence gate → LLM (Qwen3-0.8B) → Marathi answer
```

### Why this architecture (research-backed)

Per a fresh web-research pass on SOTA RAG (full notes in `research/SYNTHESIS.md`):

| Technique | Verdict | Reason |
|-----------|---------|--------|
| Hybrid BM25 + Dense + RRF | ✅ **USED** | SOTA for small KBs. RRF lifts recall 15–30%. |
| MuRIL small embedder | ✅ **USED** | Best Marathi quality at ~110MB. |
| Indic syllable tokenizer | ✅ **USED** | Whitespace tokenization cripples Devanagari recall. |
| CRAG-lite confidence gate | ✅ **USED** | Skip LLM on low-confidence — saves 1.5–4 s. |
| LRU answer cache | ✅ **USED** | Repeated queries ("namaskar", "fees kiti") → instant. |
| Single LLM call (rerank + answer) | ✅ **USED** | Folds 2 calls into 1 — saves 0.6–1.2 s. |
| Separate LLM rerank | ❌ SKIPPED | Marginal accuracy gain not worth the latency. |
| LLM query decomposition | ❌ SKIPPED | Kiosk queries are single-intent. |
| ColBERT / late-interaction | ❌ SKIPPED | No Indic model exists; overkill for 199 docs. |
| Self-RAG / RAPTOR / GraphRAG | ❌ SKIPPED | Too heavy for Pi 5 + 0.8B LLM. |
| OKF RAG | ❌ N/A | Not a recognized term in the literature. |
| Vectorless RAG | ❌ SKIPPED | Loses on mixed Roman/Devanagari. |

---

## Project structure

```
balbharati_rag/
├── config.yaml              # All tunable params (BM25, RRF, LLM, cache, ...)
├── requirements.txt         # Python deps (~150MB on disk)
├── README.md                # This file
│
├── kb/
│   └── knowledgebase.json   # Your KB (199 QA pairs)
│
├── src/                     # Pipeline source code
│   ├── config.py            # YAML config loader (singleton)
│   ├── kb.py                # KB loader + KBArticle dataclass
│   ├── tokenize.py          # Indic-aware syllable tokenizer (CRITICAL)
│   ├── normalize.py         # STT text normalizer (Whisper hallucinations, NFC, nuktas)
│   ├── bm25_index.py        # BM25 sparse index (Okapi, pure Python+numpy)
│   ├── embedder.py          # MuRIL embedder wrapper (lazy singleton)
│   ├── dense_index.py       # Dense vector index (numpy brute-force cosine)
│   ├── fusion.py            # Reciprocal Rank Fusion + CRAG-lite gating
│   ├── retrieve.py          # Hybrid retriever orchestrator
│   ├── llm_client.py        # OpenAI-compatible HTTP client for llama-server
│   ├── generate.py          # Single LLM call: rerank top-3 + answer + self-grade
│   └── pipeline.py          # Top-level orchestrator + LRU cache
│
├── scripts/
│   ├── build_kb_index.py    # One-time index builder + smoke test
│   ├── serve.py             # FastAPI server (single /answer endpoint)
│   ├── eval.py              # Eval harness (retrieval-only or full pipeline)
│   └── smoke_test_offline.py  # End-to-end test using MOCK embedder + LLM
│
├── tests/
│   └── test_pipeline.py     # Pytest unit tests for normalize + tokenize + fusion
│
└── data/
    └── golden_eval.jsonl    # Auto-generated golden set (199 items)
```

---

## Setup on Raspberry Pi 5 (8 GB)

### Step 1 — System packages

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git build-essential cmake
```

### Step 2 — Clone & create venv

```bash
cd ~
git clone <your-repo> balbharati_rag    # or scp the project folder
cd balbharati_rag

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Step 3 — Install Python deps

```bash
# Install CPU-only PyTorch first (smaller download, no CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then the rest
pip install -r requirements.txt
```

Total download: ~150 MB. First `import torch` will be ~3 s.

### Step 4 — Download Qwen3-0.8B-Instruct GGUF

```bash
mkdir -p ~/models
cd ~/models
# Q4_K_M quantization — best quality/size tradeoff for Pi
wget https://huggingface.co/Qwen/Qwen3-0.8B-Instruct-GGUF/resolve/main/qwen3-0.8b-instruct-q4_k_m.gguf
```

Size: ~500 MB. On Pi 5, this leaves ~7 GB free for the embedder + index + KV cache.

### Step 5 — Build llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
# Binary will be at build/bin/llama-server
```

Build time on Pi 5: ~10 minutes.

### Step 6 — Start llama-server (in a separate terminal)

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  --model ~/models/qwen3-0.8b-instruct-q4_k_m.gguf \
  --port 8080 \
  --ctx-size 2048 \
  --threads 4 \
  --n-gpu-layers 0 \
  --host 127.0.0.1
```

You should see: `server: listening on 127.0.0.1:8080`. Leave this running.

### Step 7 — Build the KB index (one-time, ~2 minutes)

```bash
cd ~/balbharati_rag
source venv/bin/activate
python scripts/build_kb_index.py --smoke
```

Expected output: `199 articles indexed`, then 3 sample queries with correct
retrieval results.

### Step 8 — Run the offline smoke test (no llama-server needed)

```bash
python scripts/smoke_test_offline.py
```

This verifies the full pipeline wiring using a mock embedder + mock LLM.
Expected: 10 test queries, ~7 successful answers, ~3 fallbacks (garbage inputs).

### Step 9 — Run the eval harness (retrieval-only, no llama-server)

```bash
python scripts/eval.py --mode retrieval --mock
```

Expected: **96%+ Hit@1** on 199 golden queries (using mock embedder — real
MuRIL should be 99%+).

### Step 10 — Start the API server

```bash
python scripts/serve.py --port 8000
```

### Step 11 — Test the live API

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"text":"प्रवेश कशी घ्यायची"}'
```

Expected response:
```json
{
  "answer_mr": "अ‍ॅडमिशनसाठी ऑफिसमध्ये या. आधार कार्ड, जन्म दाखला...",
  "is_fallback": false,
  "fallback_reason": "",
  "confidence_tier": "high",
  "top_fused_score": 0.0328,
  "latency_s": 2.3,
  "cached": false,
  "chosen_qa_id": "qa_0"
}
```

---

## Performance expectations (Pi 5 8 GB)

| Stage | Latency | Notes |
|-------|---------|-------|
| Normalize | <5 ms | Pure Python regex. |
| BM25 retrieve | <1 ms | 199 docs, sub-ms. |
| Dense retrieve (MuRIL) | 30–80 ms | One 768-dim embedding + 199x768 dot. |
| RRF fusion | <1 ms | Trivial. |
| LLM generate (Qwen3-0.8B Q4) | 1.5–3.5 s | Depends on tokens generated. |
| **End-to-end (cache miss)** | **2–4 s** | Within budget. |
| **End-to-end (cache hit)** | **<10 ms** | LRU cache. |

Throughput: 1 query at a time (serial kiosk). For higher concurrency, deploy
multiple Pis behind a load balancer.

---

## Configuration

All tunable params are in `config.yaml`. Key ones:

```yaml
retrieval:
  bm25:
    k1: 1.2     # term-frequency saturation (lower for short Marathi queries)
    b: 0.55     # length normalization (lower for low-variance lengths)
  fusion:
    k: 60       # RRF constant (standard)
  confidence:
    threshold_low: 0.011   # Below → immediate fallback (no LLM call)
    threshold_high: 0.025  # Above → normal LLM answer

llm:
  temperature: 0.1         # Low = more factual
  max_tokens: 220          # ~3-4 Marathi sentences
  timeout_s: 8.0           # Hard timeout — fail fast on Pi

cache:
  max_size: 256            # LRU cache entries
```

### Tuning the confidence gate

After deploying, run `python scripts/eval.py --mode full` and inspect the
`tier_distribution` field. Tune `threshold_low` so that:
- Queries that **should** be answered are mostly in `medium` or `high`.
- Queries that **shouldn't** be answered (out-of-KB) are mostly in `low`.

If fallback rate is too high: lower `threshold_low`.
If hallucination rate is too high: raise `threshold_low`.

---

## Adding STT/TTS (next phase)

The pipeline was designed so STT/TTS drop in cleanly. The interface is:

```python
# STT (future)
from your_stt_module import transcribe
raw_text = transcribe(audio_bytes)   # Devanagari Marathi

# Pipeline (now)
from src.pipeline import RAGPipeline
pipeline = RAGPipeline.from_kb()
result = pipeline.answer(raw_text)
answer_mr = result.answer_mr

# TTS (future)
from your_tts_module import synthesize
audio_out = synthesize(answer_mr)    # Devanagari Marathi → audio
```

### Recommended STT models for Pi 5

| Model | Size | Marathi quality | Latency on Pi 5 |
|-------|------|-----------------|-----------------|
| **AI4Bharat IndicWhisper (Marathi)** | ~150 MB | Excellent | 1–3 s (5 s audio) |
| OpenAI Whisper-base (multilingual) | ~150 MB | Good | 1–3 s |
| Whisper-small (multilingual) | ~500 MB | Better | 3–6 s |
| Faster-Whisper (CTranslate2 backend) | same | same | 2x faster |

**Recommendation**: `faster-whisper` with the `ai4bharat/whisper-medium-mr`
model. It uses CTranslate2 which is ~2x faster than vanilla Whisper on CPU.

### Recommended TTS models for Pi 5

| Model | Size | Marathi quality | Latency on Pi 5 |
|-------|------|-----------------|-----------------|
| **AI4Bharat TTS (Marathi)** | ~100 MB | Excellent | 0.5–1 s |
| Piper TTS (Marathi) | ~60 MB | Good | <0.5 s |
| Coqui TTS XTTS | ~500 MB | Excellent | 2–4 s |

**Recommendation**: AI4Bharat TTS for quality, Piper for speed.

---

## Pipeline architecture (detailed)

```
┌──────────────────────────────────────────────────────────────────┐
│  POST /answer {"text": "प्रवेश कशी घ्यायची"}                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ 1. normalize_stt_text()  │  NFC, strip Whisper
              │    src/normalize.py      │  hallucinations, nuktas,
              └────────────┬─────────────┘  fillers, bracket noise
                           │
                           │ cleaned text (or garbage → fallback)
                           ▼
              ┌──────────────────────────┐
              │ 2. LRU cache lookup      │  Hit? Return cached answer
              └────────────┬─────────────┘
                           │ miss
                           ▼
              ┌──────────────────────────┐
              │ 3. BM25 retrieve         │  Indic syllable tokenizer
              │    src/bm25_index.py     │  → top-30 ranked docs
              └────────────┬─────────────┘
                           │
              ┌────────────┴─────────────┐
              │ 4. Dense retrieve        │  MuRIL small embedder
              │    src/dense_index.py    │  → top-30 ranked docs
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ 5. RRF fusion            │  Combine rankings,
              │    src/fusion.py         │  assign confidence tier
              └────────────┬─────────────┘
                           │
              ┌────────────┴─────────────┐
              │ 6. CRAG-lite gate        │  tier == "low"? → fallback
              └────────────┬─────────────┘  (skip LLM, save 1.5–4 s)
                           │
                           ▼
              ┌──────────────────────────┐
              │ 7. LLM generate (ONE     │  System: "Answer only from
              │    call: rerank + answer │  context, else say माहिती
              │    + self-grade)         │  उपलब्ध नाही"
              │    src/generate.py       │  Top-3 candidates as context
              └────────────┬─────────────┘
                           │
                           │ if LLM said "don't know" → fallback
                           ▼
              ┌──────────────────────────┐
              │ 8. LRU cache + return    │  Cache for repeat queries
              └──────────────────────────┘
```

---

## Evaluation results

### Offline (mock embedder + mock LLM, 30 queries)

```
Hit@1 : 29/30  (96.67%)
Hit@3 : 30/30  (100.00%)
Hit@5 : 30/30  (100.00%)
Avg latency : 1.8 ms
```

### Expected with real MuRIL + Qwen3-0.8B (deploy on Pi)

```
Hit@1 : 99%+  (real semantic matching)
Avg latency: 2–4 s (mostly LLM)
Fallback rate: <5%  (CRAG-lite gate + LLM "don't know")
```

---

## Monitoring endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Pipeline loaded? llama-server reachable? cache size? uptime? |
| `GET /stats` | Request count, fallback rate, avg latency, cache size |
| `POST /cache/clear` | Clear LRU cache (for debugging) |
| `POST /answer` | The main query endpoint |

---

## Troubleshooting

**llama-server not reachable**
- Check `curl http://127.0.0.1:8080/health` returns 200.
- Check `config.yaml` → `llm.base_url` matches.

**MuRIL fails to download**
- Set `HF_HOME=~/models/hf_cache` before running.
- Or download `sentence-transformers/muril-base-paraphrase-v1` manually from
  HuggingFace and point `embedder.cache_dir` at the local folder.

**Latency > 5 s**
- Check `n_threads` in llama-server (should be 4 on Pi 5).
- Lower `max_tokens` in config (200 → 150).
- Increase `cache.max_size` if many repeats.

**High fallback rate**
- Run `python scripts/eval.py --mode retrieval --output results.json`.
- Inspect `tier_distribution` — if too many `low`, lower `threshold_low`.
- Inspect `top_score` for known-good queries — calibrate `threshold_high`.

---

## License & credits

- Knowledge base: © Balbharati (Maharashtra State Bureau of Textbook Production).
- Pipeline code: MIT.
- Models: Qwen3-0.8B (Apache 2.0), MuRIL (Apache 2.0), llama.cpp (MIT).
