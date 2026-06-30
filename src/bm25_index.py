"""
bm25_index.py — BM25 sparse retrieval index with Indic-aware tokenizer.

BM25 is the SOTA sparse retrieval method. For Marathi (agglutinative, rich
morphology), the choice of TOKENIZER matters more than BM25 hyperparameters.
We use the syllable-level Indic tokenizer in tokenize.py.

Params (from config.yaml):
  k1 = 1.2  (smaller than default 1.5 — Marathi questions are short, we don't
             want term frequency saturation to dominate)
  b  = 0.55 (slightly smaller than default 0.75 — Marathi questions have
             low variance in length, don't over-penalize long queries)

Implementation: pure Python + numpy. No external dep (no rank-bm25 needed).
199 docs × ~30 tokens each = trivially fast (<1ms per query on Pi).
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .config import get_config
from .tokenize import tokenize_for_index, tokenize_for_query

logger = logging.getLogger(__name__)


class BM25Index:
    """Okapi BM25 index over a fixed document collection.

    Documents are tokenized once at build time. Query scoring is O(N * avg_dl)
    per query — for 200 docs this is sub-millisecond.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.55):
        self.k1 = k1
        self.b = b
        self._doc_ids: List[str] = []          # external IDs (e.g., "qa_0", "qa_1")
        self._doc_tokens: List[List[str]] = []  # tokenized docs
        self._doc_freqs: List[Counter] = []     # per-doc term frequencies
        self._doc_len: np.ndarray = np.zeros(0, dtype=np.int32)
        self._avg_dl: float = 0.0
        self._df: Dict[str, int] = {}           # document frequency per term
        self._N: int = 0
        self._idf: Dict[str, float] = {}        # precomputed IDF
        self._built = False

    @classmethod
    def from_config(cls) -> "BM25Index":
        cfg = get_config()
        bcfg = cfg["retrieval"]["bm25"]
        return cls(k1=bcfg["k1"], b=bcfg["b"])

    def add_docs(self, doc_ids: List[str], docs: List[str]) -> None:
        """Add documents to the index. Call build() once after all docs added."""
        assert len(doc_ids) == len(docs), "doc_ids and docs length mismatch"
        for did, text in zip(doc_ids, docs):
            tokens = tokenize_for_index(text)
            self._doc_ids.append(did)
            self._doc_tokens.append(tokens)
            self._doc_freqs.append(Counter(tokens))

    def build(self) -> None:
        """Compute IDF, avg_dl, and finalize the index."""
        self._N = len(self._doc_ids)
        if self._N == 0:
            raise ValueError("Cannot build empty index")

        self._doc_len = np.array([len(toks) for toks in self._doc_tokens], dtype=np.int32)
        self._avg_dl = float(self._doc_len.mean())

        # Document frequency per term
        self._df = {}
        for tf in self._doc_freqs:
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1

        # Okapi IDF: idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )
        # The "+1" inside ln prevents negative IDF (Lucene/standard variant)
        for term, df in self._df.items():
            self._idf[term] = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

        self._built = True
        logger.info(
            "BM25 index built",
            extra={
                "n_docs": self._N,
                "avg_dl": round(self._avg_dl, 1),
                "vocab_size": len(self._df),
                "k1": self.k1,
                "b": self.b,
            },
        )

    def score(self, query: str) -> Tuple[List[str], np.ndarray]:
        """Score all docs against a query.

        Args:
            query: Raw query string (will be tokenized).

        Returns:
            Tuple of (doc_ids, scores) where scores[i] is BM25 score for doc_ids[i].
        """
        if not self._built:
            raise RuntimeError("BM25 index not built — call build() first")

        q_tokens = tokenize_for_query(query)
        if not q_tokens:
            return self._doc_ids, np.zeros(self._N, dtype=np.float32)

        scores = np.zeros(self._N, dtype=np.float32)
        q_token_set = set(q_tokens)  # we count each unique term once (standard BM25)

        k1, b = self.k1, self.b
        for term in q_token_set:
            if term not in self._idf:
                continue  # OOV term — no doc has it
            idf = self._idf[term]
            for doc_idx in range(self._N):
                tf = self._doc_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_len[doc_idx]
                # Okapi BM25 term score
                denom = tf + k1 * (1 - b + b * dl / self._avg_dl)
                scores[doc_idx] += idf * (tf * (k1 + 1)) / denom

        return self._doc_ids, scores

    def top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-K (doc_id, score) pairs sorted by score descending."""
        doc_ids, scores = self.score(query)
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        # argpartition for top-K, then sort
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._doc_ids[i], float(scores[i])) for i in top_idx]

    @property
    def n_docs(self) -> int:
        return self._N

    @property
    def is_built(self) -> bool:
        return self._built
