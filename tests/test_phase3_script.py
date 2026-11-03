"""Cached part B exports measured eligibility counts without loading sessions."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from f1pace import config, plots

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_build_dataset.py"


def _script():
    spec = importlib.util.spec_from_file_location("phase3_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cached_feature_inputs(isolated_config: Path, clean_stint: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Cache synthetic train/test laps and label the isolated generated figure."""
    config.ensure_dirs()
    test = clean_stint.assign(race_id="2025_01", year=2025, split="test")
    pd.concat([clean_stint, test], ignore_index=True).to_parquet(config.PROCESSED_DIR / "laps_clean.parquet", index=False)
    path = isolated_config / "EXPERIMENT_LOG.md"
    path.write_text("# Log\n\n## Entries\n\n---\n\n## Key results summary\n", encoding="utf-8")
    original = plots.plot_degradation_by_compound

    def labelled(frame: pd.DataFrame, **kwargs) -> Path:
        return original(frame, **kwargs, synthetic=True)

    monkeypatch.setattr(plots, "plot_degradation_by_compound", labelled)
    return path


def test_part_b_uses_cached_clean_laps_and_logs_drops(cached_feature_inputs: Path) -> None:
    """Run the actual feature stage with no source loader or manifest available."""
    _script().main(["--part", "features"])
    frame = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    assert frame.groupby("split").size().to_dict() == {"test": 7, "train": 7}
    audit = pd.read_csv(config.TABLE_DIR / "feature_eligibility_audit.csv")
    assert audit.groupby("split")["rows_dropped"].sum().to_dict() == {"test": 3, "train": 3}
    assert audit.loc[audit["rule_id"].eq("F4"), "rows_after"].tolist() == [7, 7]
    assert (config.FIG_DIR / "fig05_degradation_by_compound.png").is_file()
    assert not (config.MODELS_DIR / "TEST_OPENED.txt").exists()
    log = cached_feature_inputs.read_text(encoding="utf-8")
    assert "Phase 3" in log and "rows_dropped" in log and "test" in log and "train" in log
    assert "No model" in log


def test_no_training_history_reports_empty_eligibility_without_publishing(cached_feature_inputs: Path) -> None:
    """An empty eligible training set must not masquerade as a completed feature build."""
    path = config.PROCESSED_DIR / "laps_clean.parquet"
    pd.read_parquet(path).groupby("race_id").head(3).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="training"):
        _script().run_features()
    assert not (config.PROCESSED_DIR / "features.parquet").exists()
    assert (config.TABLE_DIR / "feature_eligibility_audit.csv").is_file()
    assert "incomplete" in cached_feature_inputs.read_text(encoding="utf-8")
