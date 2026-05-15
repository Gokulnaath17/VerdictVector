from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path

from gar.settings import Settings, get_settings
from gar.utils import safe_collection_name


@dataclass(frozen=True)
class ResetResult:
    removed: list[Path]
    skipped: list[Path]
    memory_cleared: bool


def reset_pipeline_data(settings: Settings, corpus: str | None = None) -> ResetResult:
    removed: list[Path] = []
    skipped: list[Path] = []

    if corpus:
        collection_name = safe_collection_name(corpus)
        _delete_chroma_collection(settings, collection_name)
        sparse_path = settings.paths.sparse_dir / f"{collection_name}.jsonl"
        upload_path = settings.paths.upload_dir / collection_name
        _remove_path(sparse_path, removed, skipped)
        _remove_path(upload_path, removed, skipped)
    else:
        for path in (
            settings.paths.chroma_dir,
            settings.paths.sparse_dir,
            settings.paths.upload_dir,
        ):
            _remove_path(path, removed, skipped)

    settings.ensure_directories()
    return ResetResult(
        removed=removed,
        skipped=skipped,
        memory_cleared=clear_process_memory(),
    )


def clear_process_memory() -> bool:
    get_settings.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def _delete_chroma_collection(settings: Settings, collection_name: str) -> None:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.paths.chroma_dir))
        client.delete_collection(collection_name)
    except Exception:
        pass


def _remove_path(path: Path, removed: list[Path], skipped: list[Path]) -> None:
    if not path.exists():
        skipped.append(path)
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(path)
