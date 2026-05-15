from __future__ import annotations

from gar.documents import SearchResult


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]], rrf_k: int, limit: int
) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            by_id.setdefault(result.chunk_id, result)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            sources.setdefault(result.chunk_id, set()).add(result.source)

    fused: list[SearchResult] = []
    for chunk_id, result in by_id.items():
        metadata = {**result.metadata, "retrieval_sources": sorted(sources[chunk_id])}
        fused.append(
            SearchResult(
                chunk_id=chunk_id,
                text=result.text,
                metadata=metadata,
                score=scores[chunk_id],
                source="hybrid",
            )
        )

    return sorted(fused, key=lambda result: result.score, reverse=True)[:limit]
