from __future__ import annotations

from functools import cached_property

from gar.chat import LegalChatService
from gar.chunking import TokenChunker
from gar.embeddings import StellaEmbeddings
from gar.ingestion import IngestionService
from gar.pdf_loader import PDFLoader
from gar.reranker import CrossEncoderReranker
from gar.retrieval import HybridRetriever
from gar.settings import Settings, get_settings
from gar.sparse import SparseIndexStore
from gar.vectorstore import ChromaStore


class AppContainer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

    @cached_property
    def embeddings(self) -> StellaEmbeddings:
        return StellaEmbeddings(self.settings)

    @cached_property
    def pdf_loader(self) -> PDFLoader:
        return PDFLoader(self.settings)

    @cached_property
    def chunker(self) -> TokenChunker:
        return TokenChunker(self.settings)

    @cached_property
    def vector_store(self) -> ChromaStore:
        return ChromaStore(self.settings, self.embeddings)

    @cached_property
    def sparse_store(self) -> SparseIndexStore:
        return SparseIndexStore(self.settings)

    @cached_property
    def reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker(self.settings)

    @cached_property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(
            self.settings, self.vector_store, self.sparse_store, self.reranker
        )

    @cached_property
    def ingestion_service(self) -> IngestionService:
        return IngestionService(
            self.pdf_loader, self.chunker, self.vector_store, self.sparse_store
        )

    @cached_property
    def chat_service(self) -> LegalChatService:
        return LegalChatService(self.settings, self.retriever)
