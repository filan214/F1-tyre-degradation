"""Tests for free archive normalization, provenance, and integrity checks."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import pandas as pd
import pytest

from f1pace import archive, config, data


@pytest.fixture
def archive_payloads() -> dict:
    """Provide a tiny synthetic export using the publisher's actual JSON keys."""
    return {
        "session_laptimes.json": {
            "time": [100.0, 101.5], "lap": [2, 3], "compound": ["SOFT", "SOFT"],
            "stint": [1, 1], "life": [2, 3], "pos": [1, 1], "status": ["1", "1"],
            "sesT": [120.0, 131.5], "drv": ["AAA", "AAA"], "dNum": ["1", "1"],
            "pout": ["None", "None"], "pin": ["None", "None"], "fresh": [True, True],
            "team": ["Synthetic", "Synthetic"], "lST": [20.0, 30.0],
            "lSD": ["2024-03-02T15:00:20.000000000", "2024-03-02T15:00:30.000000000"],
            "del": [False, False], "delR": ["None", "None"], "ff1G": [False, False],
            "iacc": [True, True],
            "wT": [999.0, 999.0], "wTT": [999.0, 999.0], "ms1": ["unused", "unused"],
        },
        "weather.json": {"wT": [10.0, 25.0], "wAT": [20.0, 21.0], "wTT": [30.0, 35.0], "wR": [False, False]},
        "rcm.json": {"time": ["2024-03-02T15:00:00", "2024-03-02T16:30:00"], "cat": ["Flag", "Flag"], "msg": ["GREEN LIGHT", "CHEQUERED FLAG"], "flag": ["GREEN", "CHEQUERED"]},
        "drivers.json": {"drivers": [{"driver": "AAA", "dn": "1", "team": "Synthetic"}]},
    }


@pytest.fixture
def archive_bytes(archive_payloads: dict, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Set expected fixture hashes while retaining the production integrity logic."""
    contents = {name: json.dumps(payload).encode("utf-8") for name, payload in archive_payloads.items()}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    monkeypatch.setattr(archive, "FILE_SHA256", hashes, raising=False)
    return contents


def test_archive_maps_all_required_fields_without_attached_weather(archive_payloads: dict) -> None:
    """Restore the required schema and discard unsafe per-lap weather and extras."""
    original = deepcopy(archive_payloads)
    session = archive.parse_archived_race(archive_payloads, 2024, 1)
    assert set(data.REQUIRED_LAP_COLUMNS) <= set(session.laps)
    assert "wTT" not in session.laps and "ms1" not in session.laps
    frame = data.laps_to_frame(session)
    assert frame["lap_time_s"].tolist() == [100.0, 101.5]
    assert frame["LapNumber"].tolist() == [2, 3]
    assert frame["race_id"].eq("2024_01").all()
    assert frame["circuit"].eq("Sakhir").all()
    assert frame["LapStartTime"].tolist() == [pd.Timedelta(seconds=20), pd.Timedelta(seconds=30)]
    joined = data.attach_weather(frame, session.weather_data)
    assert joined["track_temp_c"].tolist() == [30.0, 35.0]
    assert archive_payloads == original


def test_archive_nulls_are_preserved(archive_payloads: dict) -> None:
    """Missing durations and booleans remain missing, rather than fabricated values."""
    payload = archive_payloads["session_laptimes.json"]
    payload["time"][0] = "None"
    payload["pin"][1] = 4500.25
    payload["del"][0] = "None"
    session = archive.parse_archived_race(archive_payloads, 2024, 1)
    assert pd.isna(session.laps.loc[0, "LapTime"])
    assert pd.isna(session.laps.loc[0, "Deleted"])
    assert pd.isna(session.laps.loc[0, "DeletedReason"])
    assert session.laps.loc[1, "PitInTime"] == pd.Timedelta(seconds=4500.25)


def test_archive_missing_lap_column_stops_with_original_name(archive_payloads: dict) -> None:
    """Missing quality fields must stop loading and identify the FastF1 field."""
    del archive_payloads["session_laptimes.json"]["del"]
    with pytest.raises(data.MissingLapColumnsError, match="Deleted"):
        archive.parse_archived_race(archive_payloads, 2024, 1)


def test_archive_rejects_unequal_column_lengths(archive_payloads: dict) -> None:
    """Reject a truncated archive instead of padding missing lap values."""
    archive_payloads["session_laptimes.json"]["iacc"].pop()
    with pytest.raises(ValueError, match="length"):
        archive.parse_archived_race(archive_payloads, 2024, 1)


@pytest.mark.parametrize("filename,key", [("session_laptimes.json", "iacc"), ("weather.json", "wR")])
def test_archive_rejects_strings_as_boolean_flags(archive_payloads: dict, filename: str, key: str) -> None:
    """The string False must never become truthy by a generic bool conversion."""
    archive_payloads[filename][key][0] = "False"
    with pytest.raises(ValueError, match="boolean"):
        archive.parse_archived_race(archive_payloads, 2024, 1)


def test_red_flag_in_race_control_cannot_be_hidden_by_green_laps(archive_payloads: dict) -> None:
    """Cross-check full race-control messages, including red flags between laps."""
    archive_payloads["rcm.json"]["flag"][0] = "RED"
    archive_payloads["rcm.json"]["msg"][0] = "RED FLAG"
    session = archive.parse_archived_race(archive_payloads, 2024, 1)
    assert session.laps["TrackStatus"].eq("1").all()
    with pytest.raises(ValueError, match="red flag"):
        data.validate_exploration_session(session)


def test_missing_race_control_channel_stops_archive(archive_payloads: dict) -> None:
    """No race-control source means no verifiable red-flag eligibility."""
    del archive_payloads["rcm.json"]["flag"]
    with pytest.raises(ValueError, match="Flag"):
        archive.parse_archived_race(archive_payloads, 2024, 1)


def test_archive_downloads_only_missing_verified_files(
    isolated_config: Path, archive_bytes: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin URLs, verify downloaded bytes, then reuse cache without network calls."""
    calls = []

    def download(request: urllib.request.Request, *, timeout: int) -> io.BytesIO:
        calls.append(request.full_url)
        assert archive.SOURCE_COMMIT in request.full_url
        return io.BytesIO(archive_bytes[request.full_url.rsplit("/", 1)[1]])

    monkeypatch.setattr(urllib.request, "urlopen", download)
    session = archive.load_archived_race(2024, 1)
    assert len(calls) == 4
    assert len(session.laps) == 2
    assert session.provenance["commit"] == archive.SOURCE_COMMIT
    assert {f["name"]: f["sha256"] for f in session.provenance["files"]} == archive.FILE_SHA256
    assert (config.DATA_DIR / "raw" / "tracinginsights" / "2024_01" / "provenance.json").is_file()
    archive.load_archived_race(2024, 1)
    assert len(calls) == 4


def test_archive_rejects_checksum_corruption(
    isolated_config: Path, archive_bytes: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never silently accept or replace altered raw files."""
    folder = config.DATA_DIR / "raw" / "tracinginsights" / "2024_01"
    folder.mkdir(parents=True)
    for name, content in archive_bytes.items():
        (folder / name).write_bytes(content)
    (folder / "session_laptimes.json").write_bytes(archive_bytes["session_laptimes.json"] + b" ")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("Corrupted cache must not trigger an automatic replacement"))
    with pytest.raises(ValueError, match="[Cc]hecksum"):
        archive.load_archived_race(2024, 1)


def test_archive_rejects_locked_and_unverified_races(isolated_config: Path) -> None:
    """Only the specifically verified training-race archive is enabled in Phase 1."""
    for year, round_number in [(2025, 1), (2024, 2)]:
        with pytest.raises(ValueError, match="verified archive"):
            archive.load_archived_race(year, round_number)


def test_archive_rejects_mislabeled_provenance(
    isolated_config: Path, archive_bytes: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest must not relabel the verified 2024 files as the locked season."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, **kwargs: io.BytesIO(archive_bytes[request.full_url.rsplit("/", 1)[1]]))
    archive.load_archived_race(2024, 1)
    path = config.DATA_DIR / "raw" / "tracinginsights" / "2024_01" / "provenance.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["year"] = 2025
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="[Pp]rovenance"):
        archive.load_archived_race(2024, 1)


@pytest.fixture
def catalog_spec(archive_bytes: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> dict:
    """Provide an isolated pinned catalog entry for another training race."""
    base = f"https://raw.githubusercontent.com/TracingInsights/2023/{archive.SOURCE_COMMIT}/Synthetic%20Grand%20Prix/Race/"
    spec = {
        "year": 2023, "round": 2, "event_name": "Synthetic Grand Prix", "location": "Synthetic city",
        "event_format": "conventional", "source": "TracingInsights/2023", "commit": archive.SOURCE_COMMIT,
        "path": "Synthetic%20Grand%20Prix/Race/", "source_errors": [],
        "files": [{"name": name, "url": base + name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in archive_bytes.items()],
    }
    monkeypatch.setattr(archive, "_read_catalog", lambda: {"schema_version": 1, "races": [spec]}, raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, **kwargs: io.BytesIO(archive_bytes[request.full_url.rsplit("/", 1)[1]]))
    return spec


def test_catalogued_loader_preserves_event_metadata(isolated_config, catalog_spec: dict) -> None:
    """Generalized archives must not label every circuit as Bahrain/Sakhir."""
    session = archive.load_catalogued_race(2023, 2)
    frame = data.laps_to_frame(session)
    assert session.event["EventName"] == "Synthetic Grand Prix"
    assert frame["race_id"].eq("2023_02").all()
    assert frame["circuit"].eq("Synthetic city").all()
    assert session.provenance["source"] == "TracingInsights/2023"


def test_catalog_schedule_projects_metadata(catalog_spec: dict) -> None:
    """Return scheduled races without loading timing or making network calls."""
    frame = archive.get_catalog_schedule(2023)
    assert frame.to_dict("records") == [{"year": 2023, "round": 2, "event_name": "Synthetic Grand Prix", "location": "Synthetic city", "event_format": "conventional"}]


def test_catalogued_loader_rejects_unsupported_year(isolated_config, catalog_spec: dict) -> None:
    """Do not silently ingest seasons outside the configured Phase 2 years."""
    with pytest.raises(ValueError, match="configured"):
        archive.load_catalogued_race(2026, 1)


def test_catalogued_loader_rejects_unpinned_hash(isolated_config, catalog_spec: dict) -> None:
    """A malformed catalog hash must not become an unchecked download."""
    catalog_spec["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="hash"):
        archive.load_catalogued_race(2023, 2)


def test_roster_can_include_a_nonstarter(archive_payloads: dict) -> None:
    """Metadata may name a DNS driver; actual lap rows must never be invented."""
    archive_payloads["drivers.json"]["drivers"].append({"driver": "BBB", "dn": "2", "team": "Synthetic"})
    session = archive.parse_archived_race(archive_payloads, 2024, 1)
    assert session.drivers_without_laps == ("BBB",)
    assert len(session.laps) == 2
