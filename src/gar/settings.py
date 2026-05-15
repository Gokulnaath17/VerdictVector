from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


class AppSettings(BaseModel):
    name: str = "gar"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


class PathSettings(BaseModel):
    chroma_dir: Path = Path("data/chroma")
    sparse_dir: Path = Path("data/sparse")
    upload_dir: Path = Path("data/uploads")


class LLMSettings(BaseModel):
    provider: str = "openai-compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_seconds: int = 60
    max_retries: int = 2

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class EmbeddingSettings(BaseModel):
    model_name: str = "NovaSearch/stella_en_400M_v5"
    query_prompt_name: str | None = "s2p_query"
    dimension: int = 1024
    device: str = "cpu"
    batch_size: int = 16
    trust_remote_code: bool = True
    normalize_embeddings: bool = True


class PDFSettings(BaseModel):
    min_text_chars_for_ocr: int = 80
    render_dpi: int = 220


class OCRSettings(BaseModel):
    enabled: bool = True
    language: str = "eng"
    tesseract_cmd: str | None = None


class IngestionSettings(BaseModel):
    show_progress: bool = True
    dense_upsert_batch_size: int = 64

    @field_validator("dense_upsert_batch_size")
    @classmethod
    def dense_batch_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("dense_upsert_batch_size must be at least 1")
        return value


class ChunkingSettings(BaseModel):
    chunk_tokens: int = 480
    chunk_overlap_tokens: int = 80

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def overlap_must_be_smaller_than_chunk(cls, value: int, info: Any) -> int:
        chunk_tokens = info.data.get("chunk_tokens", 480)
        if value >= chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_tokens")
        return value


class RetrievalSettings(BaseModel):
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fused_top_k: int = 12
    final_top_k: int = 5
    rrf_k: int = 60
    include_debug: bool = False


class RerankerSettings(BaseModel):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"


class PromptSettings(BaseModel):
    system: str = (
        "You are a careful legal research assistant. Answer only from the provided PDF context."
    )


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)

    def ensure_directories(self) -> None:
        self.paths.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.paths.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.paths.upload_dir.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv()
    path = Path(config_path or os.getenv("GAR_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
