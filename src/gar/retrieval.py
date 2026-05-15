from __future__ import annotations

from gar.documents import SearchResult
from gar.fusion import reciprocal_rank_fusion
from gar.settings import Settings


class HybridRetriever:
    def __init__(self, settings: Settings, vector_store, sparse_store, reranker):
        self.settings = settings
        self.vector_store = vector_store
        self.sparse_store = sparse_store
        self.reranker = reranker

    def retrieve(self, corpus: str, query: str) -> list[SearchResult]:
        dense = self.vector_store.search(
            corpus=corpus,
            query=query,
            top_k=self.settings.retrieval.dense_top_k,
        )
        sparse = self.sparse_store.search(
            corpus=corpus,
            query=query,
            top_k=self.settings.retrieval.sparse_top_k,
        )
        fused = reciprocal_rank_fusion(
            [dense, sparse],
            rrf_k=self.settings.retrieval.rrf_k,
            limit=self.settings.retrieval.fused_top_k,
        )
        return self.reranker.rerank(
            query=query,
            results=fused,
            limit=self.settings.retrieval.final_top_k,
        )
