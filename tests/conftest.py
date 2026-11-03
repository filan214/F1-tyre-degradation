"""Small, explicitly synthetic lap and session fixtures."""

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import fastf1
import pandas as pd
import pytest

from f1pace import archive, config


@pytest.fixture
def clean_stint() -> pd.DataFrame:
    """Ten already-clean synthetic laps with a hand-checkable 2 s/lap trend."""
    return pd.DataFrame({
        "race_id": ["2024_01"] * 10, "year": [2024] * 10, "round": [1] * 10,
        "circuit": ["Synthetic circuit"] * 10, "Driver": ["AAA"] * 10,
        "Team": ["Synthetic Team"] * 10, "LapNumber": list(range(2, 12)),
        "Stint": [1] * 10, "lap_time_s": [100., 102., 104., 106., 108., 110., 112., 114., 116., 118.],
        "TyreLife": list(range(1, 11)), "FreshTyre": [True] * 10,
        "Compound": ["SOFT"] * 10, "track_temp_c": [30.] * 10,
        "air_temp_c": [20.] * 10, "split": ["train"] * 10,
    })


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every project path into the test's temporary directory."""
    paths = {
        "ROOT": tmp_path,
        "DATA_DIR": tmp_path / "data",
        "CACHE_DIR": tmp_path / "data" / "cache",
        "PROCESSED_DIR": tmp_path / "data" / "processed",
        "MODELS_DIR": tmp_path / "models",
        "FIG_DIR": tmp_path / "reports" / "figures",
        "TABLE_DIR": tmp_path / "reports" / "tables",
    }
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path)
    return tmp_path


@pytest.fixture
def synthetic_laps() -> pd.DataFrame:
    """Provide one first-match failure per rule and four retained laps."""
    rows = []
    changes = [
        {"LapTime": pd.NaT, "LapNumber": 1},
        {"LapNumber": 1, "PitInTime": pd.Timedelta(seconds=20)},
        {"PitInTime": pd.Timedelta(seconds=20), "PitOutTime": pd.Timedelta(seconds=10)},
        {"PitOutTime": pd.Timedelta(seconds=10)},
        {"TrackStatus": "12", "Deleted": True},
        {"Deleted": True},
        {"IsAccurate": False},
        {"FastF1Generated": True},
        {"Compound": "UNKNOWN"},
        {"LapTime": pd.Timedelta(seconds=150)},
    ]
    for i in range(14):
        row = {
            "Time": pd.Timedelta(seconds=1100 + i * 100),
            "Driver": "AAA" if i % 2 == 0 else "BBB",
            "DriverNumber": "1" if i % 2 == 0 else "2",
            "LapTime": pd.Timedelta(seconds=100 if i < 10 else 100 + i - 10),
            "LapNumber": float(i + 10),
            "Stint": 1.0 if i < 10 else 2.0,
            "PitOutTime": pd.NaT,
            "PitInTime": pd.NaT,
            "Compound": "SOFT" if i % 2 == 0 else "MEDIUM",
            "TyreLife": float(i + 1),
            "FreshTyre": True,
            "Team": "Synthetic Team",
            "LapStartTime": pd.Timedelta(seconds=1000 + i * 100),
            "LapStartDate": pd.Timestamp("2024-03-02 15:00:00") + pd.Timedelta(seconds=i * 100),
            "TrackStatus": "1",
            "Position": float(i % 2 + 1),
            "Deleted": False,
            "DeletedReason": "",
            "FastF1Generated": False,
            "IsAccurate": True,
            "race_id": "2024_01",
            "year": 2024,
            "round": 1,
            "circuit": "Synthetic circuit",
        }
        if i < 10:
            row.update(changes[i])
        rows.append(row)
    frame = pd.DataFrame(rows, index=[f"C{i}" for i in range(1, 11)] + [f"kept_{i}" for i in range(4)])
    frame.index.name = "source_row"
    for column in ["LapTime", "PitInTime", "PitOutTime"]:
        frame[column] = pd.to_timedelta(frame[column])
    frame["lap_time_s"] = frame["LapTime"].dt.total_seconds()
    return frame


@pytest.fixture
def synthetic_session(synthetic_laps: pd.DataFrame) -> SimpleNamespace:
    """Provide a synthetic dry training session without a red flag."""
    event = fastf1.events.Event(
        {"RoundNumber": 1, "EventName": "Synthetic Grand Prix", "Location": "Synthetic circuit"},
        year=2024,
    )
    weather = pd.DataFrame({
        "Time": pd.to_timedelta([0, 500, 1000, 2000], unit="s"),
        "TrackTemp": [30.0, 31.0, 32.0, 33.0],
        "AirTemp": [20.0, 21.0, 22.0, 23.0],
        "Rainfall": [False] * 4,
    })
    track = pd.DataFrame({"Status": ["1", "2", "1"]})
    return SimpleNamespace(event=event, laps=synthetic_laps.copy(), weather_data=weather, track_status=track)


@pytest.fixture
def phase2_sessions(synthetic_session, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace source boundaries with small independent train/test preparation races."""
    sessions = {}
    for year, round_number in [(2024, 1), (2024, 2), (2025, 1)]:
        session = deepcopy(synthetic_session)
        session.event.year = year
        session.event["RoundNumber"] = round_number
        session.laps = session.laps.loc[[f"kept_{i}" for i in range(4)]].copy()
        session.laps["LapNumber"] = [2, 3, 4, 5]
        session.laps["Driver"] = "AAA"
        sessions[(year, round_number)] = session

    def schedule(year: int) -> pd.DataFrame:
        return pd.DataFrame([{"year": y, "round": r, "event_name": "Synthetic Grand Prix", "location": "Synthetic circuit", "event_format": "conventional"} for y, r in sessions if y == year])

    def load(year: int, round_number: int):
        value = sessions[(year, round_number)]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(archive, "get_catalog_schedule", schedule)
    monkeypatch.setattr(archive, "load_catalogued_race", load)
    return sessions
