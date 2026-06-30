# Balbharati Marathi Voice-Agent RAG Pipeline

A robust, hallucination-resistant, **edge-deployable** RAG pipeline for a Marathi
voice agent serving Balbharati staff and students. Built for **Raspberry Pi 5 (8GB)**
with **bare-minimum latency** (2–5 s per query) and **score-based query validation**
(zero hardcoded word lists).

> **Status**: Pipeline complete with TTS. STT deferred (per spec).

---

## What this is

A pipeline that takes raw STT (speech-to-text) output in **Devanagari Marathi**
and returns a **spoken Devanagari Marathi answer** drawn from the Balbharati
knowledge base (`kb/knowledgebase.json` — 199 QA pairs across admission, fees,
exams, history, literature, values, etc.).

```
raw STT text → normalize → fuzzy correct → intent extract → validate (Stage 1)
  → BM25 + Dense (MuRIL) → RRF fusion → intent rerank → validate (Stage 2)
  → reranker (cross-encoder) → TTS (gTTS Marathi) → spoken answer
```

### Core design principles

| Principle | Why |
|-----------|-----|
| **Score-based validation** | No hardcoded word lists — the KB itself defines valid queries. Adapts automatically as KB grows. |
| **Zero hallucination** | Prefer to reject bad queries (say "माहिती नाही") rather than fabricate answers. |
| **Intent-aware reranking** | Boost candidates that match query intent (WHY/WHAT/HOW/...) before final selection. |
| **Two-stage validation gate** | Stage 1 (pre-retrieval): structure + vocab. Stage 2 (post-retrieval): dense/BM25/reranker scores + entity coverage. |
| **Minimal latency** | ~250ms retrieval-only, ~950ms with reranker, ~1.2s with TTS. |

---

## Project structure

```
balbharati_rag/
├── config.yaml              # All tunable params (validation, intent, reranker, TTS...)
├── run.sh                   # CLI entry point (rag, tts, eval, serve, ...)
├── README.md                # This file
│
├── kb/
│   └── knowledgebase.json   # Your KB (199 QA pairs)
│
├── src/                     # Pipeline source code
│   ├── config.py            # YAML config loader (singleton)
│   ├── kb.py                # KB loader + KBArticle dataclass (now with intent tagging)
│   ├── tokenize.py          # Indic-aware syllable tokenizer + light Marathi stemming
│   ├── normalize.py         # STT text normalizer (Whisper hallucinations, NFC, nuktas)
│   ├── bm25_index.py        # BM25 sparse index (Okapi, pure Python+numpy)
│   ├── embedder.py          # MuRIL embedder wrapper (lazy singleton)
│   ├── dense_index.py       # Dense vector index (numpy brute-force cosine)
│   ├── fusion.py            # Reciprocal Rank Fusion + CRAG-lite gating
│   ├── retrieve.py          # Hybrid retriever (BM25+Dense+RRF+intent rerank)
│   ├── reranker.py          # Cross-encoder reranker (gte-multilingual-reranker-base)
│   ├── query_intent.py      # Rule-based intent extraction (position-aware) + rerank
│   ├── query_validate.py    # Score-based validation (structure + vocab + entity)
│   ├── query_expand.py      # Fuzzy spell correction + transliteration
│   ├── llm_client.py        # OpenAI-compatible HTTP client for llama-server
│   ├── generate.py          # Combined intent+selection LLM call
│   ├── pipeline.py          # Top-level orchestrator + LRU cache
│   └── tts.py               # Marathi TTS wrapper (gTTS / Piper)
│
├── scripts/
│   ├── query.py             # Interactive CLI (REPL + --speak TTS support)
│   ├── build_kb_index.py    # One-time index builder + smoke test
│   ├── serve.py             # FastAPI server (single /answer endpoint)
│   ├── eval.py              # Eval harness (retrieval-only or full pipeline)
│   └── smoke_test_offline.py  # End-to-end test using MOCK embedder + LLM
│
└── tests/
    └── test_pipeline.py     # Pytest unit tests (27 tests, all passing)
```

---

## What changed (all improvements documented)

### v2 — Current (score-based validation + TTS)

| Component | Before | Now |
|-----------|--------|-----|
| Spam handling | ❌ | ✅ blocked by structural gate |
| Keyboard smash | ❌ | ✅ blocked by low dense cosine |
| Punctuation-only | ❌ | ✅ blocked by garbage detection |
| Out-of-domain | ⚠️ | ✅ blocked by score/vocab gates |
| Entity mismatch | ❌ | ✅ blocked by token coverage check |
| Anachronisms | ❌ | ✅ blocked by vocab coverage (pre-retrieval) |
| Intent detection | ⚠️ | ✅ position-aware "ka" disambiguation |
| Confidence reporting | ⚠️ | ✅ correct tiers: rejected/low/high |
| TTS | ❌ | ✅ gTTS Marathi with --speak flag |

### Key improvements

**1. Two-stage validation gate** (`src/query_validate.py`)

| Stage | Check | Catches |
|-------|-------|---------|
| Stage 1 (pre-retrieval) | Min/max length | "a", "ok", essays |
| | Token repetition | "hello hello hello" |
| | **Vocab coverage** | "Shivaji Maharaj aircraft" — "aircraft" not in KB |
| Stage 2 (post-retrieval) | Dense cosine ≥ 0.35 | "asdfghjkl", "Apple founder" |
| | BM25 raw ≥ 3.0 | Queries with zero keyword overlap |
| | Reranker score ≥ 0.35 | Wrong-topic matches |
| | **Token coverage ≥ 0.5** | "Sambhaji" vs "Shivaji" entity mismatch |

All checks are **score-based** — zero hardcoded word lists. As the KB grows,
thresholds auto-calibrate to the content.

**2. Intent-aware reranking** (`src/query_intent.py`)

- Rule-based `extract_intent()` in ~5ms (WHY/WHAT/HOW/WHEN/WHO/WHERE/HOW_MUCH/YES_NO)
- **Position-based "ka" disambiguation**: trailing "ka" → YES_NO, middle "ka" → WHY
  - "Delhi madhye Shivaji Maharaj janmale ka?" → YES_NO (correct)
  - "Swarajya ka sthapla?" → WHY (correct)
- Rerank applies: matching intent × 1.5 boost, mismatching × 0.3 penalty

**3. Fuzzy spell correction** (`src/query_expand.py`)

- Edit-distance-1 correction against full KB vocabulary
- `"Shvaji"` → `"Shivaji"`, `"sthapla"` → `"sthapna"` (if in KB)
- Applied upstream so cache, retrieval, and reranker all see the same corrected query

**4. Confidence labels** — never show "high" on fallback

| Rejection reason | Confidence tier |
|-----------------|-----------------|
| Score validation failure (dense/BM25/reranker/coverage) | `rejected` |
| Low CRAG-lite score, no candidates, LLM error | `low` |
| Structural (garbage, repetitive, too short) | `rejected` |

**5. Marathi TTS** (`src/tts.py`)

- gTTS backend: native Marathi, excellent quality, ~200ms synthesis
- Piper backend: requires community Marathi voice model (not in official repo)
- `--speak` flag for CLI and REPL
- `./run.sh tts "मराठी टेक्स्ट"` for standalone speaking

---

## Setup on Raspberry Pi 5 (8 GB)

### Quick start

```bash
# 1. Clone & venv
cd ~
git clone <your-repo> balbharati_rag
cd balbharati_rag
python3.11 -m venv venv
source venv/bin/activate

# 2. Install deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Build KB index
python scripts/build_kb_index.py --smoke

# 4. TTS (optional — for --speak flag)
pip install gtts

# 5. Run
./run.sh rag --repl --verbose --skip-llm    # Text-based REPL
./run.sh rag --repl --speak --skip-llm      # Voice REPL (needs audio)
./run.sh tts "नमस्कार"                       # Standalone TTS test
```

### Full setup with LLM (for generation)

See Step 4–6 in the detailed instructions below for Qwen3-0.8B + llama-server.

### Detailed steps

#### Step 1 — System packages

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git build-essential cmake
```

#### Step 2 — Clone & create venv

```bash
cd ~
git clone <your-repo> balbharati_rag
cd balbharati_rag
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### Step 3 — Install Python deps

```bash
# CPU-only PyTorch first (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# TTS (optional)
pip install gtts
```

#### Step 4 — Download Qwen3-0.8B-Instruct GGUF

```bash
mkdir -p ~/models
cd ~/models
wget https://huggingface.co/Qwen/Qwen3-0.8B-Instruct-GGUF/resolve/main/qwen3-0.8b-instruct-q4_k_m.gguf
```

#### Step 5 — Build llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
```

#### Step 6 — Start llama-server

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

#### Step 7 — Build KB index

```bash
cd ~/balbharati_rag
source venv/bin/activate
python scripts/build_kb_index.py --smoke
```

#### Step 8 — Run smoke test

```bash
python scripts/smoke_test_offline.py
```

---

## Usage

### Command-line

```bash
# Interactive REPL
./run.sh rag
./run.sh rag --repl --verbose
./run.sh rag --repl --speak            # With TTS
./run.sh rag --repl --skip-llm         # No LLM (faster, score-based only)

# One-shot query
./run.sh rag "प्रवेश कशी घ्यायची"
./run.sh rag --speak "प्रवेश कशी घ्यायची"

# Standalone TTS
./run.sh tts "नमस्कार, मी बालभारती सहाय्यक आहे"

# Run test suite
./run.sh test
./run.sh test -v

# Run suite from YAML
./run.sh rag --suite tests/rag_suite.yaml

# Eval
./run.sh eval --mode retrieval --mock
./run.sh eval --mode full
```

### API server

```bash
./run.sh serve --port 8000
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"text":"प्रवेश कशी घ्यायची"}'
```

---

## Performance expectations (Pi 5 8 GB)

### pipeline modes

| Mode | Latency | Use case |
|------|---------|----------|
| Retrieval-only (--skip-llm) | ~250ms + TTS: ~500ms | Fast answers, no LLM needed |
| Full pipeline (with LLM) | ~2-4s | Best accuracy, needs llama-server |
| Cache hit | <10ms | Repeated queries |

### Stage breakdown (retrieval-only)

| Stage | Latency | Notes |
|-------|---------|-------|
| Normalize | ~0.1ms | Pure Python regex |
| Fuzzy correct | ~15ms | Edit-distance-1 against KB vocab |
| Intent extract | ~0.1ms | Rule-based |
| Validate structure | ~0.1ms | Length + repetition + vocab coverage |
| Retrieve (BM25+Dense+RRF+intent) | ~40ms | Full corpus ranking |
| Cross-encoder reranker | ~500-900ms | gte-multilingual-reranker-base |
| Validate scores | ~0ms | Already computed |
| **Total (retrieval-only)** | **~650ms** | Without TTS |
| **Total (with TTS)** | **~1.2s** | gTTS ~200ms synthesis + playback |

### TTS latency

| Backend | Synthesis (3-sentence answer) | Quality | Offline |
|---------|------------------------------|---------|---------|
| gTTS (default) | ~150-300ms | Excellent (neural) | ❌ needs internet |
| Piper | ~300-400ms | Good (robot but clear) | ✅ fully offline |

---

## Configuration reference

All tunable params in `config.yaml`. Key sections:

### Query validation (`query_validation`)

```yaml
query_validation:
  structural:
    min_length: 3
    max_length: 200
    max_token_repeat_ratio: 0.6        # reject "hello hello hello"
    max_missing_token_ratio: 0.5       # reject if >50% query words unknown to KB
                                       # (with prefix + edit-dist-1 matching)
  score:
    min_dense_cosine: 0.35             # catches gibberish + OOS English
    min_bm25_raw: 3.0                  # catches no keyword overlap
    min_reranker_score: 0.35           # gte outputs ~0.2-0.8
    min_token_coverage: 0.5            # catches entity mismatch + anachronisms
```

### Intent-aware retrieval (`retrieval.intent`)

```yaml
retrieval:
  intent:
    match_boost: 1.5                   # matching intent gets × 1.5
    mismatch_penalty: 0.3              # mismatching gets × 0.3
    pool_multiplier: 6                 # pull top_k × 6 before intent rerank
```

### Reranker threshold (`reranker`)

```yaml
reranker:
  threshold: 0.35   # was 1.0 — now calibrated for gte output range
```

---

## Architecture (detailed)

```
raw STT text
    │
    ▼
┌─────────────────────────────┐
│ 1. normalize_stt_text()     │  NFC, strip Whisper hallucinations,
│    src/normalize.py         │  nuktas, fillers, bracket noise
└─────────────┬───────────────┘
              │ garbage? → fallback
              ▼
┌─────────────────────────────┐
│ 1b. fuzzy_correct()         │  Edit-distance-1 correction against
│    src/query_expand.py      │  KB vocab ("Shvaji" → "Shivaji")
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 1c. extract_intent()        │  Rule-based (~5ms): WHY/WHAT/HOW/...
│    src/query_intent.py      │  Position-aware "ka" disambiguation
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 1d. validate_structure()    │  STAGE 1: length, repetition,
│    src/query_validate.py    │  + vocab coverage (pre-retrieval)
└─────────────┬───────────────┘
              │ fail? → fallback
              ▼
┌─────────────────────────────┐
│ 2. LRU cache lookup         │  Hit? Return cached answer (instant)
└─────────────┬───────────────┘
              │ miss
              ▼
┌─────────────────────────────┐
│ 3. BM25 + Dense retrieve    │  Hybrid retrieval (full corpus)
│    src/retrieve.py          │  + RRF fusion + intent rerank
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. CRAG-lite confidence     │  fused_score < threshold_low → fallback
│    gate (post-rerank)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 5. Cross-encoder reranker   │  gte-multilingual-reranker-base
│    src/reranker.py          │  (skip-llm mode, ~500ms)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 6. validate_scores()        │  STAGE 2: dense/BM25/reranker/token
│    src/query_validate.py    │  coverage thresholds
└─────────────┬───────────────┘
              │ fail? → fallback
              ▼
┌─────────────────────────────┐
│ 7. TTS (gTTS Marathi)       │  synthesize_and_play()
│    src/tts.py               │  ~200ms synthesis
└─────────────┬───────────────┘
              │
              ▼
         spoken answer
```

---

## Testing

```bash
./run.sh test                    # All unit tests
./run.sh test -v                 # Verbose
./run.sh smoke                   # Offline smoke test
```

All 27 tests pass:
- 6 tokenizer tests (Devanagari aksaras, Latin, mixed, fillers, punctuation)
- 8 normalizer tests (Whisper hallucinations, nuktas, fillers, garbage)
- 3 RRF fusion tests (agreement, disagreement, confidence tiers)
- 10 pipeline tests (correct answers, garbage fallback, cache, etc.)

---

## Troubleshooting

**TTS: "gTTS not installed"**
```bash
pip install gtts
```

**TTS: "No audio"**
- On Pi: ensure speakers are connected and `aplay` works
- Sounddevice might need `libportaudio2`: `sudo apt install libportaudio2`

**Reranker OOM on Pi 5**
- gte-multilingual-reranker-base needs ~1.4GB RAM
- Add 2GB swap: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`
- Or use `--skip-llm` mode with score-weighted selection only

**High fallback rate**
- Run `./run.sh eval --mode retrieval --output results.json`
- Inspect score distributions and calibrate thresholds in `query_validation`
- Check `max_missing_token_ratio` — might be too aggressive for Romanized variants

---

## License & credits

- Knowledge base: © Balbharati (Maharashtra State Bureau of Textbook Production).
- Pipeline code: MIT.
- Models: Qwen3-0.8B (Apache 2.0), MuRIL (Apache 2.0), llama.cpp (MIT), gTTS (MIT).
