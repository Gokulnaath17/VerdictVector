from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    page: int
    chunk_id: str
    snippet: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    debug: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    debug: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    corpus: str
    document: str
    pages: int
    chunks: int
    chunk_ids: list[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
