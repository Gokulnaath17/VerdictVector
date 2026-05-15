from __future__ import annotations

from pathlib import Path

from gar.progress import ProgressCallback, ProgressEvent
from gar.schemas import IngestResponse


class IngestionService:
    def __init__(self, pdf_loader, chunker, vector_store, sparse_store):
        self.pdf_loader = pdf_loader
        self.chunker = chunker
        self.vector_store = vector_store
        self.sparse_store = sparse_store

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        corpus: str,
        progress: ProgressCallback | None = None,
    ) -> IngestResponse:
        path = Path(pdf_path)
        if progress:
            progress(ProgressEvent("ingest", 0, 4, f"reading {path.name}"))
        pages = self.pdf_loader.load(path, corpus=corpus, progress=progress)
        if progress:
            progress(ProgressEvent("ingest", 1, 4, f"{len(pages)} pages"))
        documents, ids = self.chunker.split_pages(pages, progress=progress)
        if progress:
            progress(ProgressEvent("ingest", 2, 4, f"{len(documents)} chunks"))
        self.vector_store.upsert_documents(corpus, documents, ids, progress=progress)
        if progress:
            progress(ProgressEvent("ingest", 3, 4, "dense index ready"))
        self.sparse_store.upsert_documents(corpus, documents, ids, progress=progress)
        if progress:
            progress(ProgressEvent("ingest", 4, 4, "complete"))
        return IngestResponse(
            corpus=corpus,
            document=path.name,
            pages=len(pages),
            chunks=len(documents),
            chunk_ids=ids,
        )
