"""Tests for configuration paths and isolated directory creation."""

from pathlib import Path

import pytest

from f1pace import config


@pytest.mark.parametrize(
    "name",
    ["ROOT", "DATA_DIR", "CACHE_DIR", "PROCESSED_DIR", "MODELS_DIR", "FIG_DIR", "TABLE_DIR", "APP_DATA_DIR", "ARCHIVE_CATALOG_PATH"],
)
def test_config_paths_are_path_objects(name: str) -> None:
    """Catch missing paths or paths exposed as strings."""
    assert isinstance(getattr(config, name, None), Path)


@pytest.mark.parametrize("preexisting", [False, True])
def test_ensure_dirs_creates_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool
) -> None:
    """Create missing parents and tolerate existing directories without data loss."""
    root = tmp_path / "project"
    paths = {
        "ROOT": root,
        "DATA_DIR": root / "data",
        "CACHE_DIR": root / "data" / "cache",
        "PROCESSED_DIR": root / "data" / "processed",
        "MODELS_DIR": root / "models",
        "FIG_DIR": root / "reports" / "figures",
        "TABLE_DIR": root / "reports" / "tables",
    }
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path, raising=False)

    sentinel = paths["CACHE_DIR"] / "existing.txt"
    if preexisting:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("keep", encoding="utf-8")
    else:
        assert not root.exists()

    config.ensure_dirs()
    config.ensure_dirs()

    for path in paths.values():
        assert path.is_dir()
    if preexisting:
        assert sentinel.read_text(encoding="utf-8") == "keep"
