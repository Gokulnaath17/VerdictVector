from pathlib import Path

from gar.maintenance import reset_pipeline_data
from gar.settings import Settings


def test_reset_pipeline_data_removes_storage_and_recreates_dirs(tmp_path: Path):
    settings = Settings.model_validate(
        {
            "paths": {
                "chroma_dir": str(tmp_path / "chroma"),
                "sparse_dir": str(tmp_path / "sparse"),
                "upload_dir": str(tmp_path / "uploads"),
            }
        }
    )
    settings.ensure_directories()
    (settings.paths.chroma_dir / "index.bin").write_text("dense", encoding="utf-8")
    (settings.paths.sparse_dir / "case-a.jsonl").write_text("sparse", encoding="utf-8")
    (settings.paths.upload_dir / "case-a").mkdir()

    result = reset_pipeline_data(settings)

    assert result.memory_cleared is True
    assert {path.name for path in result.removed} == {"chroma", "sparse", "uploads"}
    assert settings.paths.chroma_dir.exists()
    assert settings.paths.sparse_dir.exists()
    assert settings.paths.upload_dir.exists()
    assert not (settings.paths.chroma_dir / "index.bin").exists()
