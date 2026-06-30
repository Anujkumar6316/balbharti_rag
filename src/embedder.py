"""
embedder.py — MuRIL-small embedder wrapper for Marathi retrieval.

MuRIL (Multilingual Representations for Indian Languages) by Google Research
is trained on 17 Indic languages + English, with cross-lingual alignment.
For Marathi retrieval, it significantly outperforms generic multilingual
MiniLM models.

Model: sentence-transformers/muril-base-paraphrase-v1 (~110MB on disk)
Output dim: 768

On a Raspberry Pi 5 8GB:
  - First load: ~3-5 seconds (model into RAM)
  - Per-embedding: ~30-80ms for a short sentence (1-2 CPU threads)
  - Batch of 32: ~400-800ms

We use sentence-transformers because it handles pooling + normalization.
If you want to avoid the sentence-transformers dep, you can swap in
HuggingFace transformers directly (see _encode_with_transformers below).

CRITICAL for Pi: load model ONCE at startup, reuse forever. Never reload
per request — that costs 3-5s each time.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)


class MurilEmbedder:
    """Singleton MuRIL embedder. Lazily loads model on first use."""

    _instance: Optional["MurilEmbedder"] = None

    def __init__(self):
        self._model = None
        self._model_id: str = ""
        self._dim: int = 768
        self._loaded = False
        self._load_time_s: float = 0.0

    @classmethod
    def get(cls) -> "MurilEmbedder":
        """Return singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Load the MuRIL model. Idempotent."""
        if self._loaded:
            return

        cfg = get_config()
        emb_cfg = cfg["embedder"]
        self._model_id = emb_cfg["model_id"]
        cache_dir = emb_cfg.get("cache_dir")
        if cache_dir and not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        t0 = time.perf_counter()
        try:
            # Prefer sentence-transformers (handles pooling + normalization)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_id,
                device=emb_cfg.get("device", "cpu"),
                cache_folder=cache_dir,
            )
            self._model.max_seq_length = emb_cfg.get("max_seq_length", 128)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e

        self._load_time_s = time.perf_counter() - t0
        self._loaded = True
        logger.info(
            "MuRIL embedder loaded",
            extra={
                "model": self._model_id,
                "dim": self._dim,
                "load_s": round(self._load_time_s, 2),
            },
        )

    @property
    def dim(self) -> int:
        if not self._loaded:
            self.load()
        return self._dim

    @property
    def load_time_s(self) -> float:
        return self._load_time_s

    def encode(
        self,
        texts: List[str] | str,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to dense vectors.

        Args:
            texts: One text or list of texts.
            batch_size: Batch size for encoding.
            normalize: L2-normalize outputs (for cosine via dot product).

        Returns:
            np.ndarray of shape (N, dim). If single text input, returns (1, dim).
        """
        if not self._loaded:
            self.load()

        cfg = get_config()
        emb_cfg = cfg["embedder"]
        # Override normalize from config if not specified
        if normalize is None:
            normalize = emb_cfg.get("normalize", True)

        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        # Use config batch_size if not specified
        if batch_size is None:
            batch_size = emb_cfg.get("batch_size", 32)

        # sentence-transformers handles batching internally
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Ensure float32 + contiguous for FAISS/numpy
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        return embeddings

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text → 1D vector of shape (dim,)."""
        vec = self.encode([text], batch_size=1, normalize=normalize)
        return vec[0]


def get_embedder() -> MurilEmbedder:
    """Convenience: get the singleton embedder (auto-loaded on first call)."""
    e = MurilEmbedder.get()
    if not e._loaded:
        e.load()
    return e


# ---------- Smoke test ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = get_embedder()
    print(f"Dim: {e.dim}, load time: {e.load_time_s:.2f}s")
    samples = [
        "प्रवेश कशी घ्यायची",
        "admission kashi ghyaychi",
        "fees किती आहे",
        "नमस्कार",
    ]
    vecs = e.encode(samples)
    print(f"Encoded {len(samples)} texts → shape {vecs.shape}")
    # Cosine similarity (vectors are L2-normalized so dot = cosine)
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sim = float(vecs[i] @ vecs[j])
            print(f"  sim({samples[i]!r}, {samples[j]!r}) = {sim:.3f}")
