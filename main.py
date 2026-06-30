def main():
    print("Hello from balbharati-rag!")


if __name__ == "__main__":
    main()
Here are all files changed across the entire conversation:
#	File	Action
1	src/reranker.py	Created — Cross-encoder reranker replacing score-weighted fusion
2	src/query_expand.py	Created — Devanagari↔Roman transliteration for BM25 expansion
3	config.yaml	Edited — added reranker: block, reranker.threshold: 0.1→0.05, top_k: 3→5
4	src/pipeline.py	Edited — replaced score_weighted_selection() with Reranker.select(), added self._reranker
5	src/retrieve.py	Edited — BM25 now runs on all expanded query forms (merged by max score per doc)
6	scripts/query.py	Edited — verbose label "Selection scores (weighted)" → "Reranker scores"
7	src/__init__.py	Edited — added Reranker and expand_query to exports
8	kb/knowledgebase.json	Edited — added qa_199 "Dadoji konddev kon hote?" article
