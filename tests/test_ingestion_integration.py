from pathlib import Path

import fitz

from gar.chunking import TokenChunker
from gar.ingestion import IngestionService
from gar.pdf_loader import PDFLoader
from gar.settings import Settings
from gar.sparse import SparseIndexStore
from gar.vectorstore import ChromaStore


class FakeTokenizer:
    def __init__(self):
        self.words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False):
        self.words = text.split()
        return list(range(len(self.words)))

    def decode(self, tokens, skip_special_tokens: bool = True):
        return " ".join(self.words[token] for token in tokens)


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]):
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str):
        return self._embed(text)

    def _embed(self, text: str):
        lowered = text.lower()
        return [
            1.0 if "relief" in lowered else 0.0,
            1.0 if "contract" in lowered else 0.0,
        ]


def make_legal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "The court reviewed the contract and granted relief to the claimant.",
    )
    doc.save(path)
    doc.close()


def test_ingestion_persists_to_chroma_and_sparse_index(tmp_path: Path):
    pdf = tmp_path / "case.pdf"
    make_legal_pdf(pdf)
    settings = Settings.model_validate(
        {
            "ocr": {"enabled": False},
            "paths": {
                "chroma_dir": str(tmp_path / "chroma"),
                "sparse_dir": str(tmp_path / "sparse"),
                "upload_dir": str(tmp_path / "uploads"),
            },
            "chunking": {"chunk_tokens": 20, "chunk_overlap_tokens": 2},
        }
    )
    chunker = TokenChunker(settings)
    chunker._tokenizer = FakeTokenizer()
    vector_store = ChromaStore(settings, FakeEmbeddings())
    sparse_store = SparseIndexStore(settings)
    service = IngestionService(
        PDFLoader(settings), chunker, vector_store, sparse_store
    )

    response = service.ingest_pdf(pdf, corpus="case-a")
    dense_results = vector_store.search("case-a", "contract relief", top_k=1)
    sparse_results = sparse_store.search("case-a", "contract relief", top_k=1)

    assert response.pages == 1
    assert response.chunks == 1
    assert dense_results[0].chunk_id == response.chunk_ids[0]
    assert sparse_results[0].chunk_id == response.chunk_ids[0]
