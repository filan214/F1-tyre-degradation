"""Tests of real frame transformations with only network boundaries replaced."""

from types import SimpleNamespace

import fastf1
from fastf1.exceptions import DataNotLoadedError
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f1pace import config, data


def test_required_columns_exist_in_installed_fastf1() -> None:
    """Catch an installed FastF1 schema that cannot meet the loading contract."""
    assert set(data.REQUIRED_LAP_COLUMNS) <= set(fastf1.core.Laps._COLUMNS)


def test_enable_cache_uses_config(isolated_config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable caching under the configured data directory."""
    calls = []
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda path: calls.append(path))
    data.enable_cache()
    assert config.CACHE_DIR.is_dir()
    assert calls == [str(config.CACHE_DIR)]


def test_get_race_schedule_projects_source_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exclude testing and return the guide's plain schedule columns."""
    calls = []
    source = pd.DataFrame({"RoundNumber": [1], "EventName": ["Synthetic GP"], "Location": ["Synthetic"], "EventFormat": ["conventional"]})

    def fetch(year: int, *, include_testing: bool) -> pd.DataFrame:
        calls.append((year, include_testing))
        return source

    monkeypatch.setattr(fastf1, "get_event_schedule", fetch)
    result = data.get_race_schedule(2024)
    assert calls == [(2024, False)]
    assert result.to_dict("records") == [{"year": 2024, "round": 1, "event_name": "Synthetic GP", "location": "Synthetic", "event_format": "conventional"}]


def test_load_race_uses_required_load_flags(synthetic_session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch only race timing, weather, and messages, with telemetry disabled."""
    calls = []
    synthetic_session.load = lambda **kwargs: calls.append(kwargs)

    def fetch(year: int, round_number: int, session: str):
        assert (year, round_number, session) == (2024, 1, "R")
        return synthetic_session

    monkeypatch.setattr(fastf1, "get_session", fetch)
    assert data.load_race(2024, 1) is synthetic_session
    assert calls == [{"laps": True, "telemetry": False, "weather": True, "messages": True}]


def test_load_race_rejects_missing_deleted(synthetic_session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop at load time when a required source column is absent."""
    synthetic_session.laps = synthetic_session.laps.drop(columns="Deleted")
    synthetic_session.load = lambda **kwargs: None
    monkeypatch.setattr(fastf1, "get_session", lambda *args: synthetic_session)
    with pytest.raises(data.MissingLapColumnsError, match="Deleted"):
        data.load_race(2024, 1)


def test_load_race_explains_unavailable_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial FastF1 load must explain archive access instead of leaking a getter error."""
    class UnavailableSession:
        def load(self, **kwargs) -> None:
            pass

        @property
        def laps(self) -> pd.DataFrame:
            raise DataNotLoadedError()

    monkeypatch.setattr(fastf1, "get_session", lambda *args: UnavailableSession())
    with pytest.raises(RuntimeError, match="No lap timing.*archive"):
        data.load_race(2024, 1)


def test_laps_to_frame_copies_and_adds_metadata(synthetic_session) -> None:
    """Convert lap durations to seconds without mutating the session."""
    original = synthetic_session.laps.copy(deep=True)
    result = data.laps_to_frame(synthetic_session)
    assert type(result) is pd.DataFrame
    assert result.loc["kept_0", "lap_time_s"] == 100.0
    assert result["race_id"].eq("2024_01").all()
    assert result["year"].eq(2024).all()
    assert result["round"].eq(1).all()
    assert result["circuit"].eq("Synthetic circuit").all()
    result.loc["kept_0", "LapNumber"] = 999
    assert_frame_equal(synthetic_session.laps, original)


def test_laps_to_frame_names_missing_columns(synthetic_session) -> None:
    """Report every missing required column instead of silently renaming it."""
    synthetic_session.laps = synthetic_session.laps.drop(columns=["LapTime", "Deleted"])
    with pytest.raises(data.MissingLapColumnsError, match="Deleted.*LapTime|LapTime.*Deleted"):
        data.laps_to_frame(synthetic_session)


def test_race_id_uses_two_digit_round() -> None:
    """Keep race identifiers consistent with the documented example."""
    assert data.race_id(2024, 1) == "2024_01"


def test_attach_weather_is_backward_and_preserves_rows() -> None:
    """Never attach future weather, including unsorted/duplicate indexes and NaT."""
    laps = pd.DataFrame({"LapStartTime": pd.to_timedelta([25, 5, 20, None], unit="s"), "Time": pd.to_timedelta([30, 10, 25, 40], unit="s"), "tag": ["late", "early", "exact", "unknown"]}, index=[7, 7, 2, 1])
    weather = pd.DataFrame({"Time": pd.to_timedelta([30, 10, 20], unit="s"), "TrackTemp": [300.0, 100.0, 200.0], "AirTemp": [30.0, 10.0, 20.0], "Rainfall": [True, False, False]})
    original = laps.copy(deep=True)
    result = data.attach_weather(laps, weather)
    assert result.index.tolist() == [7, 7, 2, 1]
    assert result["tag"].tolist() == ["late", "early", "exact", "unknown"]
    assert result.iloc[0]["track_temp_c"] == 200.0
    assert result.iloc[2]["air_temp_c"] == 20.0
    assert pd.isna(result.iloc[1]["track_temp_c"])
    assert pd.isna(result.iloc[3]["track_temp_c"])
    assert result["rainfall"].dtype == pd.BooleanDtype()
    assert_frame_equal(laps, original)
    assert_frame_equal(result[laps.columns], laps)


def test_attach_weather_rejects_missing_channels(synthetic_session) -> None:
    """Missing source weather columns must not turn into invented measurements."""
    with pytest.raises(ValueError, match="Rainfall"):
        data.attach_weather(synthetic_session.laps, synthetic_session.weather_data.drop(columns="Rainfall"))


def test_validate_exploration_accepts_verified_dry_session(synthetic_session) -> None:
    """A dry session with known track status is eligible for exploration."""
    assert data.validate_exploration_session(synthetic_session) is None


@pytest.mark.parametrize("condition", ["rain", "wet_tyre", "red_flag", "empty_weather", "unknown_rain", "empty_track", "unknown_track"])
def test_validate_exploration_rejects_unfit_session(synthetic_session, condition: str) -> None:
    """Reject wet, red-flag, or unverifiable race conditions."""
    if condition == "rain":
        synthetic_session.weather_data.loc[0, "Rainfall"] = True
    elif condition == "wet_tyre":
        synthetic_session.laps["Compound"] = "WET"
    elif condition == "red_flag":
        synthetic_session.track_status.loc[1, "Status"] = "5"
    elif condition == "empty_weather":
        synthetic_session.weather_data = synthetic_session.weather_data.iloc[:0]
    elif condition == "unknown_rain":
        synthetic_session.weather_data["Rainfall"] = pd.array([False, pd.NA, False, False], dtype="boolean")
    elif condition == "empty_track":
        synthetic_session.track_status = synthetic_session.track_status.iloc[:0]
    else:
        synthetic_session.track_status["Status"] = ["1", None, "1"]
    with pytest.raises(ValueError):
        data.validate_exploration_session(synthetic_session)
