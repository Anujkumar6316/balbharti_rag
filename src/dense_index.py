"""
dense_index.py — Dense retrieval index (cosine similarity via dot product).

For 199 docs, FAISS is overkill. Brute-force numpy dot product is faster
(no index build overhead, no memory overhead) and gives exact results.

Per-QA-pair representation strategy (config: dense.pool = "max"):
  - Each QA pair has 1 canonical question + N variants (N≈10).
  - We embed each variant separately, then MAX-pool across variants per
    dimension. This gives one "paraphrase envelope" vector per QA pair.
    A query close to ANY variant scores high — exactly what we want for
    retrieval recall.

L2-normalized embeddings → cosine similarity = simple dot product.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .config import get_config
from .embedder import MurilEmbedder, get_embedder

logger = logging.getLogger(__name__)


class DenseIndex:
    """Brute-force dense retrieval index (numpy)."""

    def __init__(self, pool: str = "max", dim: int = 768):
        self.pool = pool  # "max" | "mean" | "first"
        self._dim = dim
        self._doc_ids: List[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)  # (N, D)
        self._built = False
        self._embedder: Optional[MurilEmbedder] = None

    @classmethod
    def from_config(cls) -> "DenseIndex":
        cfg = get_config()
        dcfg = cfg["retrieval"]["dense"]
        return cls(pool=dcfg["pool"])

    def _get_embedder(self) -> MurilEmbedder:
        if self._embedder is None:
            self._embedder = get_embedder()
            self._dim = self._embedder.dim
        return self._embedder

    def build_from_variants(
        self, doc_ids: List[str], variants_per_doc: List[List[str]]
    ) -> None:
        """Build index from per-doc variant lists.

        Args:
            doc_ids: External IDs (e.g., "qa_0").
            variants_per_doc: For each doc, list of variant strings (canonical + variants).
        """
        assert len(doc_ids) == len(variants_per_doc)
        emb = self._get_embedder()

        # Flatten for batch encode, keep doc boundaries
        all_variants: List[str] = []
        boundaries: List[Tuple[int, int]] = []  # (start, end) per doc
        for variants in variants_per_doc:
            # Filter empty variants
            variants = [v for v in variants if v and v.strip()]
            if not variants:
                # Should not happen, but be defensive
                variants = [" "]  # placeholder
            start = len(all_variants)
            all_variants.extend(variants)
            boundaries.append((start, start + len(variants)))

        # Batch encode all variants at once (faster than per-doc)
        logger.info(
            "Encoding variants for dense index",
            extra={"n_variants": len(all_variants), "n_docs": len(doc_ids)},
        )
        all_vecs = emb.encode(all_variants, batch_size=32, normalize=True)  # (V, D)

        # Pool per doc
        doc_vecs = np.zeros((len(doc_ids), self._dim), dtype=np.float32)
        for i, (start, end) in enumerate(boundaries):
            chunk = all_vecs[start:end]  # (n_variants_i, D)
            if self.pool == "max":
                doc_vecs[i] = chunk.max(axis=0)
            elif self.pool == "mean":
                doc_vecs[i] = chunk.mean(axis=0)
            elif self.pool == "first":
                doc_vecs[i] = chunk[0]
            else:
                raise ValueError(f"Unknown pool strategy: {self.pool}")

        # Re-normalize after pooling (max/mean breaks L2 normalization)
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vecs = doc_vecs / norms
        doc_vecs = doc_vecs.astype(np.float32, copy=False)

        self._doc_ids = list(doc_ids)
        self._matrix = np.ascontiguousarray(doc_vecs)
        self._built = True
        logger.info(
            "Dense index built",
            extra={
                "n_docs": len(self._doc_ids),
                "dim": self._dim,
                "pool": self.pool,
            },
        )

    def score(self, query: str) -> Tuple[List[str], np.ndarray]:
        """Score all docs against a query via cosine similarity.

        Args:
            query: Raw query string.

        Returns:
            Tuple of (doc_ids, scores) where scores[i] is cosine sim for doc_ids[i].
        """
        if not self._built:
            raise RuntimeError("Dense index not built")
        emb = self._get_embedder()
        q_vec = emb.encode_one(query, normalize=True)  # (D,)
        # Brute-force dot product (vectors are L2-normalized → dot = cosine)
        scores = self._matrix @ q_vec  # (N,)
        return self._doc_ids, scores.astype(np.float32, copy=False)

    def top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-K (doc_id, score) pairs sorted by score descending."""
        doc_ids, scores = self.score(query)
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._doc_ids[i], float(scores[i])) for i in top_idx]

    @property
    def n_docs(self) -> int:
        return len(self._doc_ids)

    @property
    def is_built(self) -> bool:
        return self._built
