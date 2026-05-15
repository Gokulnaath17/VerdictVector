from gar.documents import SearchResult
from gar.fusion import reciprocal_rank_fusion


def result(chunk_id: str, source: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        metadata={"chunk_id": chunk_id},
        score=1.0,
        source=source,
    )


def test_reciprocal_rank_fusion_dedupes_and_combines_sources():
    fused = reciprocal_rank_fusion(
        [
            [result("a", "dense"), result("b", "dense")],
            [result("b", "sparse"), result("c", "sparse")],
        ],
        rrf_k=60,
        limit=3,
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].metadata["retrieval_sources"] == ["dense", "sparse"]
