from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from gar.documents import SearchResult
from gar.progress import ProgressCallback, ProgressEvent
from gar.settings import Settings
from gar.utils import safe_collection_name, tokenize_for_sparse


@dataclass
class SparseRecord:
    chunk_id: str
    text: str
    metadata: dict


class SparseIndexStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def upsert_documents(
        self,
        corpus: str,
        documents: list[Document],
        ids: list[str],
        progress: ProgressCallback | None = None,
    ) -> None:
        records = {record.chunk_id: record for record in self.load_records(corpus)}
        total = len(documents)
        for index, (document, chunk_id) in enumerate(
            zip(documents, ids, strict=True), start=1
        ):
            records[chunk_id] = SparseRecord(
                chunk_id=chunk_id,
                text=document.page_content,
                metadata=document.metadata,
            )
            if progress and (index == total or index % 100 == 0):
                progress(ProgressEvent("sparse index", index, total, "records"))
        self._write_records(corpus, list(records.values()))
        if progress:
            progress(ProgressEvent("sparse index", total, total, "written"))

    def search(self, corpus: str, query: str, top_k: int) -> list[SearchResult]:
        records = self.load_records(corpus)
        if not records:
            return []

        tokenized_corpus = [tokenize_for_sparse(record.text) for record in records]
        query_tokens = tokenize_for_sparse(query)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]

        return [
            SearchResult(
                chunk_id=records[index].chunk_id,
                text=records[index].text,
                metadata=records[index].metadata,
                score=float(scores[index]),
                source="sparse",
            )
            for index in ranked_indexes
            if query_token_set.intersection(tokenized_corpus[index])
        ]

    def load_records(self, corpus: str) -> list[SparseRecord]:
        path = self._path(corpus)
        if not path.exists():
            return []
        records: list[SparseRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(SparseRecord(**payload))
        return records

    def _write_records(self, corpus: str, records: list[SparseRecord]) -> None:
        path = self._path(corpus)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            "\n".join(json.dumps(asdict(record), ensure_ascii=True) for record in records),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _path(self, corpus: str) -> Path:
        return self.settings.paths.sparse_dir / f"{safe_collection_name(corpus)}.jsonl"
