from langchain_core.messages import AIMessage

from gar.chat import LegalChatService
from gar.documents import SearchResult
from gar.settings import Settings


class FakeRetriever:
    def retrieve(self, corpus: str, query: str):
        return [
            SearchResult(
                chunk_id="case:abc:p3:c0",
                text="The court granted relief.",
                metadata={"document": "case.pdf", "page": 3},
                score=1.0,
                source="reranked",
            )
        ]


class FakeLLM:
    def invoke(self, messages):
        return AIMessage(content="The court granted relief. [case:abc:p3:c0]")


def test_chat_service_returns_answer_and_citations(monkeypatch):
    settings = Settings()
    monkeypatch.setenv(settings.llm.api_key_env, "secret")
    service = LegalChatService(settings, FakeRetriever())
    service._llm = FakeLLM()

    response = service.answer("case", "What did the court do?", debug=True)

    assert "granted relief" in response.answer
    assert response.citations[0].page == 3
    assert response.debug is not None
