"""
kb.py — Knowledge base loader + per-QA-pair document representation.

The KB JSON has this structure (see kb/knowledgebase.json):
  {
    "metadata": {...},
    "qa_pairs": [
      {
        "question": "admission kashi ghyaychi",          # canonical (often Roman)
        "variants": [ ... 10 variants in Roman + Devanagari ... ],
        "answer": { "mr": "अ‍ॅडमिशनसाठी ऑफिसमध्ये या..." },
        "category": "admission"
      },
      ...
    ]
  }

For retrieval, each QA pair becomes ONE document. The document text is:
  canonical_question + "\n" + "\n".join(variants)

This maximizes recall — any paraphrase will match SOMETHING in the doc.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KBArticle:
    """A single QA pair from the knowledge base."""
    qa_id: str              # e.g., "qa_0"
    question: str           # canonical question
    variants: List[str]     # all variants (Roman + Devanagari)
    answer_mr: str          # Devanagari Marathi answer
    category: str
    # Precomputed for retrieval:
    doc_text: str           # canonical + variants joined
    variant_list: List[str] # for dense embedder (canonical + variants)


def load_kb(kb_path: str | Path) -> Tuple[List[KBArticle], dict]:
    """Load knowledge base JSON.

    Returns:
        (articles, metadata)
    """
    kb_path = Path(kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    qa_pairs = data.get("qa_pairs", [])

    articles: List[KBArticle] = []
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "").strip()
        variants = [v.strip() for v in qa.get("variants", []) if v and v.strip()]
        answer_mr = qa.get("answer", {}).get("mr", "").strip()
        category = qa.get("category", "unknown")

        # All variants (canonical + variants) for dense encoder
        all_variants = [question] + variants
        all_variants = [v for v in all_variants if v]  # drop empties
        if not all_variants:
            logger.warning(f"Empty QA pair at index {i}, skipping")
            continue

        # Doc text = canonical + all variants joined with newline
        doc_text = "\n".join(all_variants)

        articles.append(
            KBArticle(
                qa_id=f"qa_{i}",
                question=question,
                variants=variants,
                answer_mr=answer_mr,
                category=category,
                doc_text=doc_text,
                variant_list=all_variants,
            )
        )

    logger.info(
        "KB loaded",
        extra={
            "path": str(kb_path),
            "n_articles": len(articles),
            "version": metadata.get("version"),
        },
    )
    return articles, metadata


def articles_by_id(articles: List[KBArticle]) -> Dict[str, KBArticle]:
    """Build a qa_id -> KBArticle lookup map."""
    return {a.qa_id: a for a in articles}
