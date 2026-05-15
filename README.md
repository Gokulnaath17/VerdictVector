# VerdictVector: A Hybrid RAG System for Legal Document Intelligence

GAR ingests legal PDFs, indexes them with hybrid retrieval, and answers questions
through an OpenAI-compatible chat API with citations.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Set your API key in `.env`. All runtime tunables live in
`config/settings.yaml`, including the chat model, OpenAI-compatible base URL,
Stella embedding model, Chroma path, chunk sizes, OCR settings, retrieval
counts, and reranker.

Tesseract must be installed separately if OCR fallback is enabled.

## CLI

```powershell
uv run gar ingest --pdf .\examples\case.pdf --corpus case-a
uv run gar chat --corpus case-a
uv run gar serve
```

Ingest shows progress for PDF extraction, tokenizer/model loading, dense Chroma
indexing, and sparse indexing. To check the active configuration and common
latency traps:

```powershell
uv run gar doctor
```

To clear all persisted pipeline data and process caches:

```powershell
uv run gar reset --yes
```

Use `uv run gar reset --corpus case-a --yes` to reset only one corpus.

## API

- `GET /health`
- `POST /v1/corpora/{corpus}/ingest` with a PDF file upload named `file`
- `POST /v1/corpora/{corpus}/chat`

Example chat body:

```json
{
  "message": "What relief did the court grant?",
  "session_id": "research-session-1",
  "debug": false
}
```

Responses include an answer and citations with document, page, chunk id, and
snippet.
