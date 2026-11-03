"""Fixed wet-race decisions, per-race failures, overrides and CSV integrity."""

from pathlib import Path

import pandas as pd
import pytest

from f1pace import manifest


def test_manifest_keeps_dry_races_and_fixed_splits(phase2_sessions: dict) -> None:
    """Classify complete dry sessions and keep the requested test ingestion separate."""
    result = manifest.build_manifest([2024, 2025])
    assert result["race_id"].tolist() == ["2024_01", "2024_02", "2025_01"]
    assert result["split"].tolist() == ["train", "train", "test"]
    assert result["include"].all()
    assert result["exclude_reason"].eq("").all()
    assert result["manual_override"].eq("").all()
    assert result["n_laps_raw"].tolist() == [4, 4, 4]
    assert result["n_laps_clean"].tolist() == [4, 4, 4]


@pytest.mark.parametrize("wet_count,expected", [(1, True), (2, False)])
def test_wet_compound_threshold_is_strict(phase2_sessions: dict, wet_count: int, expected: bool) -> None:
    """Exactly 5% wet tyres is allowed; a share above 5% is excluded."""
    session = phase2_sessions[(2024, 1)]
    session.laps = pd.concat([session.laps] * 5, ignore_index=True)
    session.laps["LapNumber"] = range(2, 22)
    session.laps.loc[:wet_count - 1, "Compound"] = "INTERMEDIATE"
    row = manifest.build_manifest([2024]).iloc[0]
    assert bool(row["include"]) is expected
    assert row["wet_compound_share"] == wet_count / 20
    assert row["exclude_reason"] == ("" if expected else "wet_compound")
    assert row["needs_manual_review"]


def test_rainfall_excludes_even_without_wet_tyres(phase2_sessions: dict) -> None:
    """Any session rainfall is sufficient and disagreement is flagged for review."""
    phase2_sessions[(2024, 1)].weather_data.loc[0, "Rainfall"] = True
    row = manifest.build_manifest([2024]).iloc[0]
    assert not row["include"]
    assert row["any_rainfall"]
    assert row["exclude_reason"] == "rainfall"
    assert "rainfall_without_wet_threshold" in row["review_reasons"]


def test_both_wet_conditions_keep_one_primary_reason(phase2_sessions: dict) -> None:
    """Do not double-count a race failing both automatic wet conditions."""
    phase2_sessions[(2024, 1)].laps["Compound"] = "WET"
    phase2_sessions[(2024, 1)].weather_data.loc[0, "Rainfall"] = True
    row = manifest.build_manifest([2024]).iloc[0]
    assert not row["include"]
    assert row["exclude_reason"] == "wet_compound"
    assert row["any_rainfall"] and row["wet_compound_share"] == 1


def test_load_failure_does_not_stop_other_races(phase2_sessions: dict) -> None:
    """Retain a failed race's identity and continue building later race rows."""
    phase2_sessions[(2024, 1)] = RuntimeError("synthetic missing source")
    result = manifest.build_manifest([2024, 2025])
    assert len(result) == 3
    assert result.iloc[0]["exclude_reason"] == "load_error"
    assert "synthetic missing source" in result.iloc[0]["error_detail"]
    assert result.iloc[1:]["include"].all()


@pytest.mark.parametrize("condition", ["empty", "unknown", "missing_channel"])
def test_unverifiable_weather_is_not_treated_as_dry(phase2_sessions: dict, condition: str) -> None:
    """Unknown weather must fail closed while the next race still loads."""
    session = phase2_sessions[(2024, 1)]
    if condition == "empty":
        session.weather_data = session.weather_data.iloc[:0]
    elif condition == "unknown":
        session.weather_data["Rainfall"] = pd.array([False, pd.NA, False, False], dtype="boolean")
    else:
        session.weather_data = session.weather_data.drop(columns="Rainfall")
    result = manifest.build_manifest([2024])
    assert result.iloc[0]["exclude_reason"] == "load_error"
    assert not result.iloc[0]["include"]
    assert result.iloc[1]["include"]


def test_manual_include_preserves_automatic_evidence(phase2_sessions: dict) -> None:
    """A manual directive is explicit, auditable, and does not rewrite wet observations."""
    phase2_sessions[(2024, 1)].weather_data.loc[0, "Rainfall"] = True
    previous = pd.DataFrame({"race_id": ["2024_01"], "manual_override": ["include"], "manual_note": ["Synthetic human review"]})
    row = manifest.build_manifest([2024], previous=previous).iloc[0]
    assert row["include"] and row["any_rainfall"]
    assert not row["auto_include"]
    assert row["auto_exclude_reason"] == "rainfall"
    assert row["manual_override"] == "include"
    assert row["manual_note"] == "Synthetic human review"


def test_manual_include_cannot_recover_unloaded_data(phase2_sessions: dict) -> None:
    """No override can manufacture a successfully loaded race."""
    phase2_sessions[(2024, 1)] = RuntimeError("unavailable")
    previous = pd.DataFrame({"race_id": ["2024_01"], "manual_override": ["include"]})
    row = manifest.build_manifest([2024], previous=previous).iloc[0]
    assert not row["include"]
    assert row["exclude_reason"] == "load_error"


def test_manual_exclude_has_explicit_reason(phase2_sessions: dict) -> None:
    """Do not invent rainfall to explain a human exclusion of a dry race."""
    previous = pd.DataFrame({"race_id": ["2024_01"], "manual_override": ["exclude"]})
    row = manifest.build_manifest([2024], previous=previous).iloc[0]
    assert not row["include"] and row["auto_include"]
    assert row["exclude_reason"] == "manual_override"


def test_invalid_override_is_rejected(phase2_sessions: dict) -> None:
    """An unrecognized human directive must not silently become an inclusion."""
    previous = pd.DataFrame({"race_id": ["2024_01"], "manual_override": ["maybe"]})
    with pytest.raises(ValueError, match="override"):
        manifest.build_manifest([2024], previous=previous)


def test_csv_false_is_not_truthy(phase2_sessions: dict, tmp_path: Path) -> None:
    """CSV string booleans must be parsed explicitly before dataset selection."""
    phase2_sessions[(2024, 1)].weather_data.loc[0, "Rainfall"] = True
    frame = manifest.build_manifest([2024])
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    loaded = manifest.read_manifest(path)
    assert not loaded.iloc[0]["include"]
    assert loaded.iloc[1]["include"]


def test_read_manifest_applies_new_manual_directive(phase2_sessions: dict, tmp_path: Path) -> None:
    """A directive added after a build is effective when preparing the dataset."""
    frame = manifest.build_manifest([2024])
    frame.loc[0, "manual_override"] = "exclude"
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    assert not manifest.read_manifest(path).iloc[0]["include"]


def test_red_flag_is_reviewed_without_changing_dry_inclusion(phase2_sessions: dict) -> None:
    """Phase 1's no-red preference must not become an extra Phase 2 exclusion rule."""
    phase2_sessions[(2024, 1)].track_status.loc[0, "Status"] = "5"
    row = manifest.build_manifest([2024]).iloc[0]
    assert row["include"]
    assert "red_flag_session" in row["review_reasons"]


def test_manifest_summary_counts_primary_reasons(phase2_sessions: dict) -> None:
    """Summary counts must reconcile to all scheduled races exactly once."""
    phase2_sessions[(2024, 1)].weather_data.loc[0, "Rainfall"] = True
    result = manifest.build_manifest([2024, 2025])
    summary = manifest.summarise_manifest(result)
    assert summary["races"].sum() == 3
    assert summary.loc[summary["reason"].eq("rainfall"), "races"].sum() == 1
