from pathlib import Path

from langchain_core.documents import Document

from gar.settings import Settings
from gar.sparse import SparseIndexStore


def test_sparse_index_upserts_and_searches(tmp_path: Path):
    settings = Settings.model_validate(
        {"paths": {"sparse_dir": str(tmp_path), "chroma_dir": "x", "upload_dir": "y"}}
    )
    store = SparseIndexStore(settings)
    documents = [
        Document(page_content="contract breach remedy damages", metadata={"page": 1}),
        Document(page_content="criminal appeal evidence", metadata={"page": 2}),
    ]

    store.upsert_documents("Case A", documents, ["one", "two"])
    results = store.search("Case A", "breach damages", top_k=2)

    assert results[0].chunk_id == "one"
    assert results[0].metadata["page"] == 1
