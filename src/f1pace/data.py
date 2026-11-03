"""FastF1 race loading, explicit schema checks, and past-only weather joins."""

import logging
from typing import Protocol

import fastf1
import pandas as pd
from fastf1.core import Session
from fastf1.events import Event
from fastf1.exceptions import DataNotLoadedError

from f1pace import config

_logger = logging.getLogger(__name__)

REQUIRED_LAP_COLUMNS = (
    "Time", "Driver", "DriverNumber", "LapTime", "LapNumber", "Stint",
    "PitOutTime", "PitInTime", "Compound", "TyreLife", "FreshTyre", "Team",
    "LapStartTime", "LapStartDate", "TrackStatus", "Position", "Deleted",
    "DeletedReason", "FastF1Generated", "IsAccurate",
)
_WEATHER_COLUMNS = ("Time", "TrackTemp", "AirTemp", "Rainfall")


class RaceSession(Protocol):
    """The timing/weather interface shared by live and archived race data."""

    event: Event
    laps: pd.DataFrame
    weather_data: pd.DataFrame
    track_status: pd.DataFrame


class MissingLapColumnsError(ValueError):
    """The installed or loaded FastF1 lap schema cannot satisfy the guide."""


def _require_lap_columns(laps: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_LAP_COLUMNS) - set(laps.columns))
    if missing:
        raise MissingLapColumnsError("Missing required FastF1 lap columns: " + ", ".join(missing))


def enable_cache() -> None:
    """Create project directories and enable FastF1 caching under DATA_DIR."""
    config.ensure_dirs()
    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    _logger.info("FastF1 cache: %s", config.CACHE_DIR)


def get_race_schedule(year: int) -> pd.DataFrame:
    """Return the year's Grand Prix weekends, excluding testing events."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    frame = pd.DataFrame(schedule)[["RoundNumber", "EventName", "Location", "EventFormat"]].copy()
    frame = frame.rename(columns={
        "RoundNumber": "round", "EventName": "event_name",
        "Location": "location", "EventFormat": "event_format",
    })
    frame.insert(0, "year", year)
    return frame


def load_race(year: int, round_number: int) -> Session:
    """Load a Grand Prix race with weather and messages, without telemetry."""
    session = fastf1.get_session(year, round_number, config.SESSION)
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        laps = session.laps
    except DataNotLoadedError as exc:
        raise RuntimeError(
            f"No lap timing was loaded for {race_id(year, round_number)}. "
            "Check access to F1's archive and network connectivity. New archive downloads "
            "now require F1 TV authentication, supported by FastF1 3.9+. "
            "See the archive-access note in README.md; existing cached timing can still be used."
        ) from exc
    _logger.info("FastF1 %s lap columns: %s", fastf1.__version__, list(laps.columns))
    _require_lap_columns(laps)
    return session


def race_id(year: int, round_number: int) -> str:
    """Format a stable identifier such as 2024_01."""
    return f"{year}_{round_number:02d}"


def laps_to_frame(session: RaceSession) -> pd.DataFrame:
    """Copy session laps into a plain frame with identifiers and lap seconds."""
    _require_lap_columns(session.laps)
    frame = pd.DataFrame(session.laps).copy(deep=True)
    year = int(session.event.year)
    round_number = int(session.event["RoundNumber"])
    frame["race_id"] = race_id(year, round_number)
    frame["year"] = year
    frame["round"] = round_number
    frame["circuit"] = session.event["Location"]
    frame["lap_time_s"] = frame["LapTime"].dt.total_seconds()
    return frame


def attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach weather at or before lap start, preserving row order and index."""
    missing = sorted(set(_WEATHER_COLUMNS) - set(weather.columns))
    if missing:
        raise ValueError("Missing weather columns: " + ", ".join(missing))
    if "LapStartTime" not in laps:
        raise ValueError("Missing lap column: LapStartTime")
    if weather["Time"].isna().any():
        raise ValueError("Weather Time contains missing timestamps")
    frame = laps.copy(deep=True)
    lookup = pd.DataFrame({
        "LapStartTime": laps["LapStartTime"].to_numpy(),
        "_row_order": range(len(laps)),
    })
    past = weather[list(_WEATHER_COLUMNS)].rename(columns={
        "Time": "_weather_time", "TrackTemp": "track_temp_c",
        "AirTemp": "air_temp_c", "Rainfall": "rainfall",
    })
    matched = pd.merge_asof(
        lookup.dropna(subset=["LapStartTime"]).sort_values("LapStartTime", kind="stable"),
        past.sort_values("_weather_time", kind="stable"),
        left_on="LapStartTime", right_on="_weather_time", direction="backward",
    ).set_index("_row_order").reindex(range(len(laps)))
    for column in ("track_temp_c", "air_temp_c"):
        frame[column] = matched[column].to_numpy()
    frame["rainfall"] = pd.array(matched["rainfall"], dtype="boolean")
    return frame


def validate_exploration_session(session: RaceSession) -> None:
    """Require verifiably dry conditions and no red flag for Phase 1."""
    weather = session.weather_data
    track = session.track_status
    if weather.empty or "Rainfall" not in weather:
        raise ValueError("Cannot verify dry conditions: weather/Rainfall data is missing")
    rainfall = weather["Rainfall"]
    if rainfall.isna().any() or not rainfall.isin([True, False]).all():
        raise ValueError("Cannot verify dry conditions: Rainfall contains unknown values")
    if rainfall.eq(True).any():
        raise ValueError("Exploration requires a dry race: rainfall was recorded")
    if session.laps.empty:
        raise ValueError("No race laps were loaded")
    wet_share = session.laps["Compound"].isin(["INTERMEDIATE", "WET"]).mean()
    if wet_share > config.WET_COMPOUND_SHARE_MAX:
        raise ValueError(f"Exploration requires a dry race: wet compound share {wet_share:.3%}")
    if track.empty or "Status" not in track or track["Status"].isna().any():
        raise ValueError("Cannot verify no red flag: track status data is missing")
    if not track["Status"].isin(["1", "2", "4", "5", "6", "7"]).all():
        raise ValueError("Cannot verify no red flag: unknown track status code")
    if track["Status"].eq("5").any() or session.laps["TrackStatus"].astype("string").str.contains("5", na=False).any():
        raise ValueError("Exploration requires a race with no red flag")
