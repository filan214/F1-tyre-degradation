"""Verified free FastF1-derived archive support for Phase 1."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
from typing import Any
import urllib.request

from fastf1.events import Event
import pandas as pd

from f1pace import config, data

_logger = logging.getLogger(__name__)
SOURCE_COMMIT = "d1543a71370d9b617b627319f8ad80b2f85dfd5e"
SOURCE_URL = "https://github.com/TracingInsights/2024"
SUPPORTED_RACES = frozenset({(2024, 1)})
FILE_SHA256 = {
    "session_laptimes.json": "508f22e47936ac8de631e21524ed7b9808c1d7e0e52729d22dc5162bf8e04aae",
    "weather.json": "50d14901d8b3982dd55190ba2467167c3593d63ed0e1167a3b379baba6483975",
    "rcm.json": "9c5b3cf273d2a2f25235f0e137e1505deb3fc5c50bd8424e98db2cc29ad64994",
    "drivers.json": "eb28f95809c6cd356f5d6b79147825dcb1d4d64255702c6befd3f314d2e03789",
}
_BASE_URL = f"https://raw.githubusercontent.com/TracingInsights/2024/{SOURCE_COMMIT}/Bahrain%20Grand%20Prix/Race/"
_LAP_COLUMNS = {
    "sesT": "Time", "drv": "Driver", "dNum": "DriverNumber", "time": "LapTime",
    "lap": "LapNumber", "stint": "Stint", "pout": "PitOutTime", "pin": "PitInTime",
    "compound": "Compound", "life": "TyreLife", "fresh": "FreshTyre", "team": "Team",
    "lST": "LapStartTime", "lSD": "LapStartDate", "status": "TrackStatus", "pos": "Position",
    "del": "Deleted", "delR": "DeletedReason", "ff1G": "FastF1Generated", "iacc": "IsAccurate",
}
_WEATHER_COLUMNS = {"wT": "Time", "wAT": "AirTemp", "wTT": "TrackTemp", "wR": "Rainfall"}
_CONTROL_COLUMNS = {"time": "Time", "cat": "Category", "msg": "Message", "flag": "Flag"}


@dataclass
class ArchivedRace:
    """An archive-backed race; track_status is a derived code inventory, not a timeline."""

    event: Event
    laps: pd.DataFrame
    weather_data: pd.DataFrame
    race_control_messages: pd.DataFrame
    track_status: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)
    drivers_without_laps: tuple[str, ...] = ()


def _check_race(year: int, round_number: int) -> None:
    if (year, round_number) not in SUPPORTED_RACES:
        raise ValueError("No verified archive is enabled for this race; Phase 1 supports only 2024 round 1")


def _frame(payload: Any, columns: dict[str, str], label: str) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError(f"Missing or invalid archive file: {label}")
    missing = [f"{name} ({key})" for key, name in columns.items() if key not in payload]
    if missing:
        error = data.MissingLapColumnsError if label == "laps" else ValueError
        raise error(f"Missing {label} archive fields: " + ", ".join(missing))
    if any(not isinstance(payload[key], list) for key in columns):
        raise ValueError(f"{label} archive columns must be arrays")
    lengths = {len(payload[key]) for key in columns}
    if len(lengths) != 1 or lengths == {0}:
        raise ValueError(f"{label} archive column length mismatch or empty data")
    return pd.DataFrame({name: [None if value == "None" else value for value in payload[key]] for key, name in columns.items()})


def _booleans(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for name in columns:
        if frame[name].dropna().map(lambda value: not isinstance(value, bool)).any():
            raise ValueError(f"Archive {name} must contain boolean values or missing values")
        frame[name] = pd.array(frame[name], dtype="boolean")


def parse_archived_race(payloads: dict[str, Any], year: int, round_number: int) -> ArchivedRace:
    """Normalize verified export fields without imputing laps or using attached weather."""
    _check_race(year, round_number)
    return _parse_payloads(payloads, year, round_number, "Bahrain Grand Prix", "Sakhir")


def _parse_payloads(payloads: dict[str, Any], year: int, round_number: int, event_name: str, location: str) -> ArchivedRace:
    laps = _frame(payloads.get("session_laptimes.json"), _LAP_COLUMNS, "laps")
    for name in ("Time", "LapTime", "PitOutTime", "PitInTime", "LapStartTime"):
        laps[name] = pd.to_timedelta(pd.to_numeric(laps[name], errors="raise"), unit="s")
    for name in ("LapNumber", "Stint", "TyreLife", "Position"):
        laps[name] = pd.to_numeric(laps[name], errors="raise")
    laps["LapStartDate"] = pd.to_datetime(laps["LapStartDate"], format="ISO8601", errors="raise")
    for name in ("Driver", "DriverNumber", "Compound", "Team", "TrackStatus", "DeletedReason"):
        laps[name] = laps[name].astype("string")
    _booleans(laps, ("FreshTyre", "Deleted", "FastF1Generated", "IsAccurate"))
    data._require_lap_columns(laps)

    weather = _frame(payloads.get("weather.json"), _WEATHER_COLUMNS, "weather")
    weather["Time"] = pd.to_timedelta(pd.to_numeric(weather["Time"], errors="raise"), unit="s")
    for name in ("AirTemp", "TrackTemp"):
        weather[name] = pd.to_numeric(weather[name], errors="raise")
    _booleans(weather, ("Rainfall",))

    control = _frame(payloads.get("rcm.json"), _CONTROL_COLUMNS, "race-control")
    control["Time"] = pd.to_datetime(control["Time"], format="ISO8601", errors="raise")
    if control["Time"].isna().any() or not control["Flag"].notna().any():
        raise ValueError("Race-control timestamps and observed flags are required to verify race conditions")
    drivers = payloads.get("drivers.json", {}).get("drivers")
    if not isinstance(drivers, list) or not drivers:
        raise ValueError("Archive driver roster is missing")
    roster = {str(driver["driver"]) for driver in drivers}
    observed = set(laps["Driver"].dropna())
    if not observed <= roster:
        raise ValueError("Archive laps and driver roster do not agree")
    nonstarters = tuple(sorted(roster - observed))

    # This inventory is only an eligibility diagnostic. It is not the original
    # track-status timeline and is never attached as a per-lap replacement.
    codes = set("".join(laps["TrackStatus"].dropna()))
    red_message = control["Flag"].eq("RED").any() or control["Message"].astype("string").str.contains(r"\bRED FLAG\b", case=False, na=False).any()
    if red_message:
        codes.add("5")
    event = Event({"RoundNumber": round_number, "EventName": event_name, "Location": location}, year=year)
    return ArchivedRace(event=event, laps=laps, weather_data=weather, race_control_messages=control, track_status=pd.DataFrame({"Status": sorted(codes)}), drivers_without_laps=nonstarters)


def load_archived_race(year: int, round_number: int) -> ArchivedRace:
    """Download/reuse the pinned free Bahrain archive and verify every source hash."""
    _check_race(year, round_number)
    spec = {
        "year": year, "round": round_number, "event_name": "Bahrain Grand Prix", "location": "Sakhir",
        "source": "TracingInsights/2024", "commit": SOURCE_COMMIT,
        "files": [{"name": name, "url": _BASE_URL + name, "sha256": digest} for name, digest in FILE_SHA256.items()],
    }
    return _load_spec(spec)


def _read_catalog() -> dict[str, Any]:
    catalog = json.loads(config.ARCHIVE_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1 or not isinstance(catalog.get("races"), list):
        raise ValueError("Invalid packaged archive catalog")
    return catalog


def get_catalog_schedule(year: int) -> pd.DataFrame:
    """Return pinned race metadata for a configured Phase 2 season."""
    if year not in config.TRAIN_YEARS + config.TEST_YEARS:
        raise ValueError("Year is outside the configured training/test seasons")
    races = [race for race in _read_catalog()["races"] if race["year"] == year]
    if not races:
        raise ValueError(f"No pinned race schedule for {year}")
    frame = pd.DataFrame(races)[["year", "round", "event_name", "location", "event_format"]].copy()
    if frame["round"].duplicated().any():
        raise ValueError(f"Duplicate rounds in archive catalog for {year}")
    return frame.sort_values("round").reset_index(drop=True)


def load_catalogued_race(year: int, round_number: int) -> ArchivedRace:
    """Load a pinned configured-season archive for fixed Phase 2 ingestion only."""
    if year not in config.TRAIN_YEARS + config.TEST_YEARS:
        raise ValueError("Year is outside the configured training/test seasons")
    races = [race for race in _read_catalog()["races"] if (race["year"], race["round"]) == (year, round_number)]
    if len(races) != 1:
        raise ValueError(f"No unique pinned archive for {data.race_id(year, round_number)}")
    return _load_spec(races[0])


def _load_spec(spec: dict[str, Any]) -> ArchivedRace:
    year, round_number = int(spec["year"]), int(spec["round"])
    if spec.get("source_errors"):
        raise ValueError("Unusable archive source: " + "; ".join(spec["source_errors"]))
    if spec["source"] != f"TracingInsights/{year}" or not re.fullmatch(r"[a-f0-9]{40}", spec["commit"]):
        raise ValueError("Invalid archive source identity or commit")
    expected_files = spec["files"]
    if len(expected_files) != 4 or {entry["name"] for entry in expected_files} != set(FILE_SHA256):
        raise ValueError("Archive must pin all four expected files")
    prefix = f"https://raw.githubusercontent.com/{spec['source']}/{spec['commit']}/"
    for entry in expected_files:
        if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError("Invalid pinned archive hash")
        if not entry["url"].startswith(prefix) or not entry["url"].endswith("/Race/" + entry["name"]):
            raise ValueError("Archive URL does not match the pinned source")
    folder = config.DATA_DIR / "raw" / "tracinginsights" / data.race_id(year, round_number)
    folder.mkdir(parents=True, exist_ok=True)
    payloads = {}
    files = []
    for entry in expected_files:
        name, expected = entry["name"], entry["sha256"]
        path = folder / name
        url = entry["url"]
        if path.is_file():
            content = path.read_bytes()
        else:
            _logger.info("Downloading free archive: %s", url)
            request = urllib.request.Request(url, headers={"User-Agent": "f1pace/0.1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read(5_000_001)
            if len(content) > 5_000_000:
                raise ValueError(f"Archive file exceeds expected size: {name}")
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch for archive {name}; raw data was not replaced")
        payloads[name] = json.loads(content)
        if not path.exists():
            path.write_bytes(content)
        files.append({"name": name, "url": url, "sha256": digest, "bytes": len(content)})

    manifest_path = folder / "provenance.json"
    if manifest_path.exists():
        provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = {"schema_version": 1, **{key: spec[key] for key in ("source", "commit", "year", "round", "event_name", "location")}}
        if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in identity.items()):
            raise ValueError("Archive provenance identity does not match the verified race")
        recorded = provenance.get("files")
        if not isinstance(recorded, list) or len(recorded) != len(files) or any(not isinstance(entry, dict) for entry in recorded):
            raise ValueError("Archive provenance file records are incomplete")
        if {entry.get("name"): entry for entry in recorded} != {entry["name"]: entry for entry in files}:
            raise ValueError("Archive provenance does not match the pinned source and hashes")
    else:
        provenance = {
            "schema_version": 1, **{key: spec[key] for key in ("source", "commit", "year", "round", "event_name", "location")},
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "files": files,
        }
        manifest_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    session = _parse_payloads(payloads, year, round_number, spec["event_name"], spec["location"])
    session.provenance = provenance
    _logger.info("Verified free archive %s at commit %s", data.race_id(year, round_number), spec["commit"])
    return session
