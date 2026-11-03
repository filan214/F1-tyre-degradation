"""Phase 2 command orchestration and complete-output safeguards."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from f1pace import config, manifest, plots

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(autouse=True)
def label_synthetic_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even isolated orchestration-test charts must declare synthetic inputs."""
    original = plots.plot_race_manifest

    def labelled(frame: pd.DataFrame, **kwargs) -> Path:
        return original(frame, **kwargs, synthetic=True)

    monkeypatch.setattr(plots, "plot_race_manifest", labelled)


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def log_template(isolated_config: Path) -> Path:
    """Keep synthetic run entries away from the real experiment history."""
    path = isolated_config / "EXPERIMENT_LOG.md"
    path.write_text("# Log\n\n## Entries\n\n---\n\n## Key results summary\n", encoding="utf-8")
    return path


def test_phase2_scripts_export_matching_manifest_audit_and_parquet(phase2_sessions, log_template, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete small build writes both copies, ordered audits, split totals and fig04."""
    monkeypatch.setattr(config, "TRAIN_YEARS", [2024])
    _script("02_build_manifest").run_manifest(source="archive")
    _script("03_build_dataset").run_dataset(source="archive")
    first = config.PROCESSED_DIR / "race_manifest.csv"
    assert first.read_bytes() == (config.TABLE_DIR / first.name).read_bytes()
    laps = pd.read_parquet(config.PROCESSED_DIR / "laps_clean.parquet")
    assert len(laps) == 12
    assert laps.groupby("split").size().to_dict() == {"test": 4, "train": 8}
    audit = pd.read_csv(config.TABLE_DIR / "audit_all_races.csv")
    assert len(audit) == 33 and audit["race_id"].nunique() == 3
    assert (config.FIG_DIR / "fig04_race_manifest_summary.png").is_file()
    assert not (config.PROCESSED_DIR / "features.parquet").exists()
    log = log_template.read_text(encoding="utf-8")
    assert "Phase 2" in log and "train" in log and "test" in log
    assert "No manual overrides" in log


def test_dataset_script_refuses_to_publish_partial_build(phase2_sessions, log_template) -> None:
    """Failed included races are exported and logged without a misleading parquet."""
    config.ensure_dirs()
    manifest.build_manifest([2024, 2025]).to_csv(config.PROCESSED_DIR / "race_manifest.csv", index=False)
    phase2_sessions[(2024, 1)] = RuntimeError("synthetic late load failure")
    with pytest.raises(RuntimeError, match="incomplete"):
        _script("03_build_dataset").run_dataset()
    assert not (config.PROCESSED_DIR / "laps_clean.parquet").exists()
    errors = pd.read_csv(config.TABLE_DIR / "dataset_load_errors.csv")
    assert errors["race_id"].tolist() == ["2024_01"]
    assert "synthetic late load failure" in log_template.read_text(encoding="utf-8")


def test_manifest_rebuild_preserves_manual_reason(phase2_sessions, log_template, monkeypatch: pytest.MonkeyPatch) -> None:
    """A written human directive survives fresh automatic evidence and reaches the log."""
    monkeypatch.setattr(config, "TRAIN_YEARS", [2024])
    config.ensure_dirs()
    frame = manifest.build_manifest([2024, 2025])
    frame.loc[0, ["manual_override", "manual_note"]] = ["exclude", "Human review: synthetic concern"]
    frame.to_csv(config.PROCESSED_DIR / "race_manifest.csv", index=False)
    _script("02_build_manifest").run_manifest()
    saved = manifest.read_manifest(config.PROCESSED_DIR / "race_manifest.csv")
    assert not saved.iloc[0]["include"] and saved.iloc[0]["auto_include"]
    assert "Human review: synthetic concern" in log_template.read_text(encoding="utf-8")
