from gar.chunking import TokenChunker
from gar.documents import PageText
from gar.settings import Settings


class FakeTokenizer:
    def __init__(self):
        self.words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False):
        self.words = text.split()
        return list(range(len(self.words)))

    def decode(self, tokens, skip_special_tokens: bool = True):
        return " ".join(self.words[token] for token in tokens)


def test_chunker_preserves_metadata_and_stable_ids():
    settings = Settings.model_validate(
        {"chunking": {"chunk_tokens": 4, "chunk_overlap_tokens": 1}}
    )
    chunker = TokenChunker(settings)
    chunker._tokenizer = FakeTokenizer()
    page = PageText(
        text="alpha beta gamma delta epsilon zeta",
        metadata={
            "source": "case.pdf",
            "document": "case.pdf",
            "corpus": "matter-a",
            "page": 2,
            "content_hash": "abc123",
            "extraction_method": "text",
        },
    )

    documents, ids = chunker.split_pages([page])

    assert ids == ["matter-a:abc123:p2:c0", "matter-a:abc123:p2:c1"]
    assert documents[0].metadata["page"] == 2
    assert documents[0].metadata["chunk_id"] == ids[0]
    assert documents[0].page_content == "alpha beta gamma delta"
