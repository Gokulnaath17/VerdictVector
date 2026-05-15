from __future__ import annotations

from pathlib import Path

import typer

from gar.container import AppContainer
from gar.maintenance import reset_pipeline_data
from gar.progress import ConsoleProgress
from gar.settings import load_settings


app = typer.Typer(help="Hybrid legal RAG for PDF ingestion and chat.")


def build_container(config: Path | None = None) -> AppContainer:
    return AppContainer(load_settings(config))


@app.command()
def ingest(
    pdf: Path = typer.Option(..., "--pdf", exists=True, dir_okay=False, readable=True),
    corpus: str = typer.Option(..., "--corpus", help="Named legal corpus or matter."),
    config: Path | None = typer.Option(None, "--config", help="Path to settings YAML."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bars."),
) -> None:
    """Ingest one legal PDF into a named corpus."""
    container = build_container(config)
    progress = ConsoleProgress(
        enabled=container.settings.ingestion.show_progress and not no_progress
    )
    response = container.ingestion_service.ingest_pdf(pdf, corpus, progress=progress)
    typer.echo(
        f"Ingested {response.document} into {response.corpus}: "
        f"{response.pages} pages, {response.chunks} chunks."
    )


@app.command()
def reset(
    corpus: str | None = typer.Option(
        None, "--corpus", help="Reset only one named corpus."
    ),
    config: Path | None = typer.Option(None, "--config", help="Path to settings YAML."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation."),
) -> None:
    """Reset pipeline data on disk and clear process caches."""
    settings = load_settings(config)
    target = f"corpus {corpus!r}" if corpus else "all corpora and uploads"
    if not yes:
        typer.confirm(f"Reset {target}?", abort=True)

    result = reset_pipeline_data(settings, corpus=corpus)
    for path in result.removed:
        typer.echo(f"removed {path}")
    if not result.removed:
        typer.echo("No persisted data needed removal.")
    if result.memory_cleared:
        typer.echo("Cleared Python process caches.")


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config", help="Path to settings YAML."),
) -> None:
    """Check config, storage paths, and latency-sensitive settings."""
    settings = load_settings(config)
    settings.ensure_directories()

    typer.echo(f"app: {settings.app.name} on {settings.app.host}:{settings.app.port}")
    typer.echo(
        f"embedding: {settings.embedding.model_name} "
        f"device={settings.embedding.device} batch={settings.embedding.batch_size}"
    )
    typer.echo(
        f"chunking: {settings.chunking.chunk_tokens} tokens "
        f"overlap={settings.chunking.chunk_overlap_tokens}"
    )
    typer.echo(
        f"ingestion: dense_upsert_batch_size="
        f"{settings.ingestion.dense_upsert_batch_size} "
        f"progress={settings.ingestion.show_progress}"
    )
    typer.echo(
        f"retrieval: dense={settings.retrieval.dense_top_k} "
        f"sparse={settings.retrieval.sparse_top_k} "
        f"fused={settings.retrieval.fused_top_k} final={settings.retrieval.final_top_k}"
    )

    for label, path in (
        ("chroma", settings.paths.chroma_dir),
        ("sparse", settings.paths.sparse_dir),
        ("uploads", settings.paths.upload_dir),
    ):
        marker = path / ".write-test"
        try:
            marker.write_text("ok", encoding="utf-8")
            marker.unlink()
            typer.echo(f"{label}: writable at {path}")
        except OSError as exc:
            typer.echo(f"{label}: not writable at {path} ({exc})")

    lower_model = settings.embedding.model_name.lower()
    if settings.embedding.device == "cpu" and any(
        marker in lower_model for marker in ("bge-m3", "stella", "large")
    ):
        typer.echo("warning: embedding model is large for CPU; first ingest may be slow.")
    if settings.reranker.enabled:
        typer.echo("warning: reranker is enabled; this adds query latency.")
    if settings.ocr.enabled:
        typer.echo("warning: OCR is enabled; scanned PDFs will ingest much slower.")


@app.command()
def chat(
    corpus: str = typer.Option(..., "--corpus", help="Named legal corpus or matter."),
    config: Path | None = typer.Option(None, "--config", help="Path to settings YAML."),
) -> None:
    """Start an interactive chat against a named corpus."""
    container = build_container(config)
    typer.echo("Enter a question, or type 'exit' to quit.")
    while True:
        message = typer.prompt("question")
        if message.strip().lower() in {"exit", "quit"}:
            break
        response = container.chat_service.answer(corpus=corpus, message=message)
        typer.echo(response.answer)
        if response.citations:
            typer.echo("\nCitations:")
            for citation in response.citations:
                typer.echo(
                    f"- {citation.document} p.{citation.page} "
                    f"[{citation.chunk_id}]: {citation.snippet}"
                )


@app.command()
def serve(
    config: Path | None = typer.Option(None, "--config", help="Path to settings YAML."),
    host: str | None = typer.Option(None, "--host", help="Override API host."),
    port: int | None = typer.Option(None, "--port", help="Override API port."),
) -> None:
    """Run the FastAPI service."""
    import uvicorn
    from gar.api import create_app

    container = build_container(config)
    api = create_app(container)
    uvicorn.run(
        api,
        host=host or container.settings.app.host,
        port=port or container.settings.app.port,
    )
