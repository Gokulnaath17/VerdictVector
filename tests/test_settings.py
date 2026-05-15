from pathlib import Path

import pytest

from gar.settings import load_settings


def test_load_settings_from_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "settings.yaml"
    config.write_text(
        """
llm:
  api_key_env: GAR_TEST_KEY
  model: test-model
chunking:
  chunk_tokens: 100
  chunk_overlap_tokens: 20
paths:
  chroma_dir: chroma
  sparse_dir: sparse
  upload_dir: uploads
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("GAR_TEST_KEY", "secret")

    settings = load_settings(config)

    assert settings.llm.model == "test-model"
    assert settings.llm.api_key == "secret"
    assert settings.chunking.chunk_tokens == 100
    assert settings.ingestion.dense_upsert_batch_size == 64


def test_invalid_chunk_overlap_raises(tmp_path: Path):
    config = tmp_path / "settings.yaml"
    config.write_text(
        """
chunking:
  chunk_tokens: 100
  chunk_overlap_tokens: 100
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        load_settings(config)


def test_invalid_dense_upsert_batch_size_raises(tmp_path: Path):
    config = tmp_path / "settings.yaml"
    config.write_text(
        """
ingestion:
  dense_upsert_batch_size: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dense_upsert_batch_size"):
        load_settings(config)
