from pathlib import Path

from fastapi.testclient import TestClient

from gar.api import create_app
from gar.schemas import ChatResponse, Citation, IngestResponse
from gar.settings import Settings


class FakeIngestionService:
    def ingest_pdf(self, pdf_path: Path, corpus: str) -> IngestResponse:
        return IngestResponse(
            corpus=corpus,
            document=pdf_path.name,
            pages=1,
            chunks=1,
            chunk_ids=["case:abc:p1:c0"],
        )


class FakeChatService:
    def answer(self, corpus: str, message: str, session_id=None, debug=False):
        return ChatResponse(
            answer=f"answer for {message}",
            citations=[
                Citation(
                    document="case.pdf",
                    page=1,
                    chunk_id="case:abc:p1:c0",
                    snippet="sample",
                )
            ],
            debug={"ok": True} if debug else None,
        )


class FakeContainer:
    def __init__(self, tmp_path: Path):
        self.settings = Settings.model_validate(
            {
                "app": {"name": "test-gar"},
                "paths": {
                    "upload_dir": str(tmp_path / "uploads"),
                    "sparse_dir": str(tmp_path / "sparse"),
                    "chroma_dir": str(tmp_path / "chroma"),
                },
            }
        )
        self.ingestion_service = FakeIngestionService()
        self.chat_service = FakeChatService()


def test_health_ingest_and_chat_endpoints(tmp_path: Path):
    client = TestClient(create_app(FakeContainer(tmp_path)))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["app"] == "test-gar"

    ingest = client.post(
        "/v1/corpora/case-a/ingest",
        files={"file": ("case.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] == 1

    chat = client.post(
        "/v1/corpora/case-a/chat",
        json={"message": "What happened?", "debug": True},
    )
    assert chat.status_code == 200
    assert chat.json()["citations"][0]["page"] == 1
    assert chat.json()["debug"] == {"ok": True}
