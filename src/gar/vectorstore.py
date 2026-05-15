from __future__ import annotations

from langchain_core.documents import Document

from gar.documents import SearchResult
from gar.progress import ProgressCallback, ProgressEvent
from gar.settings import Settings
from gar.utils import safe_collection_name


class ChromaStore:
    def __init__(self, settings: Settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self.settings.paths.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.settings.paths.chroma_dir)
            )
        return self._client

    def upsert_documents(
        self,
        corpus: str,
        documents: list[Document],
        ids: list[str],
        progress: ProgressCallback | None = None,
    ) -> None:
        if not documents:
            return
        if progress:
            progress(ProgressEvent("chroma", 0, 1, "opening persistent store"))
        collection = self.client.get_or_create_collection(safe_collection_name(corpus))
        if progress:
            progress(ProgressEvent("chroma", 1, 1, "ready"))
        batch_size = self.settings.ingestion.dense_upsert_batch_size
        total = len(documents)
        if progress and getattr(self.embeddings, "is_loaded", True) is False:
            progress(
                ProgressEvent(
                    "embedding model",
                    0,
                    1,
                    f"loading {self.settings.embedding.model_name}",
                )
            )
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            document_batch = documents[start:end]
            id_batch = ids[start:end]
            embeddings = self.embeddings.embed_documents(
                [document.page_content for document in document_batch]
            )
            if progress and start == 0:
                progress(ProgressEvent("embedding model", 1, 1, "ready"))
            collection.upsert(
                ids=id_batch,
                documents=[document.page_content for document in document_batch],
                metadatas=[document.metadata for document in document_batch],
                embeddings=embeddings,
            )
            if progress:
                progress(ProgressEvent("dense index", end, total, f"batch {end}/{total}"))

    def search(self, corpus: str, query: str, top_k: int) -> list[SearchResult]:
        collection = self.client.get_or_create_collection(safe_collection_name(corpus))
        query_embedding = self.embeddings.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        search_results: list[SearchResult] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            similarity = 1.0 / (1.0 + float(distance))
            search_results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    text=text or "",
                    metadata=metadata or {},
                    score=similarity,
                    source="dense",
                )
            )
        return search_results
