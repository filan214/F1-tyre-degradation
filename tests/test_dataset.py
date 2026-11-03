"""Dataset assembly preserves source laps and keeps fixed season splits separate."""

import pandas as pd
import pytest

from f1pace import data, dataset, manifest


def test_dataset_loads_included_races_only(phase2_sessions, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual exclusions and False CSV decisions must not reach the loader."""
    frame = manifest.build_manifest([2024, 2025])
    frame.loc[frame["race_id"].eq("2024_02"), "manual_override"] = "exclude"
    calls = []
    original = manifest.load_phase2_race

    def load(year: int, round_number: int, source: str = "auto"):
        calls.append((year, round_number))
        return original(year, round_number, source)

    monkeypatch.setattr(manifest, "load_phase2_race", load)
    result = dataset.assemble_clean_laps(frame)
    assert calls == [(2024, 1), (2025, 1)]
    assert len(result.laps) == 8
    assert result.failures.empty
    assert result.laps.groupby("split").size().to_dict() == {"test": 4, "train": 4}


def test_dataset_cleans_before_weather_and_preserves_lap_identity(phase2_sessions, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not renumber or interpolate gaps while preparing combined clean laps."""
    session = phase2_sessions[(2024, 1)]
    junk = session.laps.iloc[[0]].copy()
    junk["LapTime"] = pd.NaT
    junk["LapNumber"] = 999
    session.laps = pd.concat([session.laps, junk], ignore_index=True)
    session.laps["LapTime"] = pd.to_timedelta(session.laps["LapTime"])
    frame = manifest.build_manifest([2024])
    original = data.attach_weather

    def attach(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        assert laps["LapTime"].notna().all()
        return original(laps, weather)

    monkeypatch.setattr(data, "attach_weather", attach)
    result = dataset.assemble_clean_laps(frame)
    first = result.laps.loc[result.laps["race_id"].eq("2024_01")]
    assert first["LapNumber"].tolist() == [2, 3, 4, 5]
    assert first["Stint"].tolist() == [2, 2, 2, 2]
    assert len(result.audit) == 22
    assert result.audit.loc[result.audit["rule_id"].eq("C1"), "laps_removed"].sum() == 1
    assert not {"lag1", "mean_last3", "target_delta", "tyre_age"} & set(result.laps)


def test_dataset_failure_is_recorded_and_other_races_continue(phase2_sessions) -> None:
    """An included race that becomes unavailable must not silently disappear."""
    frame = manifest.build_manifest([2024, 2025])
    phase2_sessions[(2024, 1)] = RuntimeError("synthetic cache lost")
    result = dataset.assemble_clean_laps(frame)
    assert len(result.laps) == 8
    assert result.failures["race_id"].tolist() == ["2024_01"]
    assert "synthetic cache lost" in result.failures.iloc[0]["error_detail"]
    assert set(result.audit["race_id"]) == {"2024_02", "2025_01"}


def test_missing_weather_is_preserved_and_flagged(phase2_sessions) -> None:
    """Do not fill unmatched weather from future samples or discard those laps silently."""
    phase2_sessions[(2024, 1)].weather_data["Time"] += pd.Timedelta(days=1)
    frame = manifest.build_manifest([2024])
    result = dataset.assemble_clean_laps(frame)
    first = result.laps.loc[result.laps["race_id"].eq("2024_01")]
    assert len(first) == 4 and first["track_temp_c"].isna().all()
    assert result.review.iloc[0]["issue"] == "missing_matched_weather"
    assert result.review.iloc[0]["n_laps"] == 4


def test_stale_manifest_count_is_a_build_failure(phase2_sessions) -> None:
    """A changed cleaning result requires a rebuilt manifest, not hidden count drift."""
    frame = manifest.build_manifest([2024])
    frame.loc[0, "n_laps_clean"] = 999
    result = dataset.assemble_clean_laps(frame)
    assert result.failures["race_id"].tolist() == ["2024_01"]
    assert "count" in result.failures.iloc[0]["error_detail"].lower()


def test_mislabeled_test_season_is_rejected_before_build(phase2_sessions) -> None:
    """A manually altered split must not leak 2025 laps into training preparation."""
    frame = manifest.build_manifest([2024, 2025])
    frame.loc[frame["year"].eq(2025), "split"] = "train"
    with pytest.raises(ValueError, match="split"):
        dataset.assemble_clean_laps(frame)
