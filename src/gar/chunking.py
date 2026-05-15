from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document

from gar.documents import PageText
from gar.progress import ProgressCallback, ProgressEvent
from gar.settings import Settings


class TokenChunker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.embedding.model_name,
                trust_remote_code=self.settings.embedding.trust_remote_code,
            )
        return self._tokenizer

    def split_pages(
        self,
        pages: Iterable[PageText],
        progress: ProgressCallback | None = None,
    ) -> tuple[list[Document], list[str]]:
        page_list = list(pages)
        documents: list[Document] = []
        ids: list[str] = []
        if page_list and self._tokenizer is None and progress:
            progress(
                ProgressEvent(
                    "tokenizer",
                    0,
                    1,
                    f"loading {self.settings.embedding.model_name}",
                )
            )
        if page_list:
            _ = self.tokenizer
            if progress:
                progress(ProgressEvent("tokenizer", 1, 1, "ready"))
        for index, page in enumerate(page_list, start=1):
            if not page.text.strip():
                if progress:
                    progress(ProgressEvent("chunk pages", index, len(page_list), "empty"))
                continue
            page_documents, page_ids = self._split_page(page)
            documents.extend(page_documents)
            ids.extend(page_ids)
            if progress:
                progress(
                    ProgressEvent(
                        "chunk pages",
                        index,
                        len(page_list),
                        f"{len(page_documents)} chunks",
                    )
                )
        return documents, ids

    def _split_page(self, page: PageText) -> tuple[list[Document], list[str]]:
        tokens = self.tokenizer.encode(page.text, add_special_tokens=False)
        chunk_size = self.settings.chunking.chunk_tokens
        overlap = self.settings.chunking.chunk_overlap_tokens
        step = chunk_size - overlap
        documents: list[Document] = []
        ids: list[str] = []

        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue
            text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
            if not text:
                continue
            chunk_id = (
                f"{page.metadata['corpus']}:"
                f"{page.metadata['content_hash']}:"
                f"p{page.metadata['page']}:c{chunk_index}"
            )
            metadata = {
                **page.metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": min(end, len(tokens)),
            }
            documents.append(Document(page_content=text, metadata=metadata))
            ids.append(chunk_id)

            if end >= len(tokens):
                break

        return documents, ids
