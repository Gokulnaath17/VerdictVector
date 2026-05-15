from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, UploadFile

from gar.container import AppContainer
from gar.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from gar.utils import safe_collection_name


def create_app(container: AppContainer | None = None) -> FastAPI:
    active_container = container or AppContainer()
    app = FastAPI(title="GAR Legal RAG", version="0.1.0")

    def get_container() -> AppContainer:
        return active_container

    @app.get("/health", response_model=HealthResponse)
    def health(service: AppContainer = Depends(get_container)) -> HealthResponse:
        return HealthResponse(app=service.settings.app.name)

    @app.post("/v1/corpora/{corpus}/ingest", response_model=IngestResponse)
    def ingest_pdf(
        corpus: str,
        file: UploadFile = File(...),
        service: AppContainer = Depends(get_container),
    ) -> IngestResponse:
        upload_dir = service.settings.paths.upload_dir / safe_collection_name(corpus)
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / Path(file.filename or "upload.pdf").name
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        return service.ingestion_service.ingest_pdf(target, corpus)

    @app.post("/v1/corpora/{corpus}/chat", response_model=ChatResponse)
    def chat(
        corpus: str,
        request: ChatRequest,
        service: AppContainer = Depends(get_container),
    ) -> ChatResponse:
        return service.chat_service.answer(
            corpus=corpus,
            message=request.message,
            session_id=request.session_id,
            debug=bool(request.debug),
        )

    return app


app = create_app()
