from __future__ import annotations

from gar.documents import SearchResult
from gar.settings import Settings


class CrossEncoderReranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.settings.reranker.model_name,
                device=self.settings.reranker.device,
            )
        return self._model

    def rerank(
        self, query: str, results: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        if not self.settings.reranker.enabled or not results:
            return results[:limit]
        pairs = [(query, result.text) for result in results]
        scores = self.model.predict(pairs)
        reranked = [
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata={**result.metadata, "hybrid_score": result.score},
                score=float(score),
                source="reranked",
            )
            for result, score in zip(results, scores, strict=True)
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)[:limit]
