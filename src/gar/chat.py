from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from gar.schemas import ChatResponse, Citation
from gar.settings import Settings
from gar.utils import short_snippet


class LegalChatService:
    def __init__(self, settings: Settings, retriever):
        self.settings = settings
        self.retriever = retriever
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            api_key = self.settings.llm.api_key
            if not api_key:
                raise RuntimeError(
                    f"Missing API key. Set {self.settings.llm.api_key_env} in .env "
                    "or your environment."
                )
            self._llm = ChatOpenAI(
                model=self.settings.llm.model,
                api_key=api_key,
                base_url=self.settings.llm.base_url,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
                timeout=self.settings.llm.timeout_seconds,
                max_retries=self.settings.llm.max_retries,
            )
        return self._llm

    def answer(
        self, corpus: str, message: str, session_id: str | None = None, debug: bool = False
    ) -> ChatResponse:
        results = self.retriever.retrieve(corpus, message)
        context = self._format_context(results)
        prompt = (
            f"Corpus: {corpus}\n"
            f"Session: {session_id or 'ad-hoc'}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {message}"
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=self.settings.prompts.system),
                HumanMessage(content=prompt),
            ]
        )
        answer_text = getattr(response, "content", str(response))
        citations = [
            Citation(
                document=str(result.metadata.get("document", "unknown")),
                page=int(result.metadata.get("page", 0)),
                chunk_id=result.chunk_id,
                snippet=short_snippet(result.text),
            )
            for result in results
        ]
        debug_payload = None
        if debug or self.settings.retrieval.include_debug:
            debug_payload = {
                "retrieved": [
                    {
                        "chunk_id": result.chunk_id,
                        "score": result.score,
                        "source": result.source,
                        "metadata": result.metadata,
                    }
                    for result in results
                ]
            }
        return ChatResponse(answer=answer_text, citations=citations, debug=debug_payload)

    def _format_context(self, results) -> str:
        if not results:
            return "No retrieved context."
        blocks = []
        for result in results:
            document = result.metadata.get("document", "unknown")
            page = result.metadata.get("page", "unknown")
            blocks.append(
                f"[{result.chunk_id}] document={document} page={page}\n{result.text}"
            )
        return "\n\n".join(blocks)
