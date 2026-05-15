from __future__ import annotations

from langchain_core.embeddings import Embeddings

from gar.settings import Settings


class StellaEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.settings.embedding.model_name,
                trust_remote_code=self.settings.embedding.trust_remote_code,
                device=self.settings.embedding.device,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.settings.embedding.batch_size,
            normalize_embeddings=self.settings.embedding.normalize_embeddings,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        encode_kwargs = {
            "batch_size": 1,
            "normalize_embeddings": self.settings.embedding.normalize_embeddings,
        }
        if self.settings.embedding.query_prompt_name:
            encode_kwargs["prompt_name"] = self.settings.embedding.query_prompt_name
        vectors = self.model.encode([text], **encode_kwargs)
        return vectors[0].tolist()
