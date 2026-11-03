"""Fixed wet-race inclusion rules, auditable overrides, and review diagnostics."""

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from f1pace import archive, cleaning, config, data

_logger = logging.getLogger(__name__)
MANIFEST_COLUMNS = [
    "race_id", "year", "round", "event_name", "circuit", "split", "n_laps_raw",
    "n_laps_clean", "wet_compound_share", "any_rainfall", "include", "exclude_reason", "manual_override",
]


def split_for_year(year: int) -> str:
    """Assign a season to its fixed configured split without consulting lap outcomes."""
    if set(config.TRAIN_YEARS) & set(config.TEST_YEARS):
        raise ValueError("Training and test seasons overlap")
    if year in config.TRAIN_YEARS:
        return "train"
    if year in config.TEST_YEARS:
        return "test"
    raise ValueError(f"Year {year} is outside the configured seasons")


def load_phase2_race(year: int, round_number: int, source: str = "auto") -> data.RaceSession:
    """Load fixed-preprocessing inputs through the free catalog or explicit FastF1."""
    split_for_year(year)
    if source in {"auto", "archive"}:
        return archive.load_catalogued_race(year, round_number)
    if source == "fastf1":
        return data.load_race(year, round_number)
    raise ValueError(f"Unknown Phase 2 source: {source}")


def validate_race_identity(laps: pd.DataFrame, year: int, round_number: int) -> None:
    """Reject mislabeled or ambiguous driver-lap records before split assignment."""
    expected = data.race_id(year, round_number)
    if not laps["race_id"].eq(expected).all() or not laps["year"].eq(year).all() or not laps["round"].eq(round_number).all():
        raise ValueError(f"Loaded session identity does not match {expected}")
    if laps.duplicated(["race_id", "Driver", "LapNumber"]).any():
        raise ValueError(f"Duplicate driver-lap records in {expected}")


def _populate(row: dict[str, Any], session: data.RaceSession) -> None:
    raw = data.laps_to_frame(session)
    row["n_laps_raw"] = len(raw)
    if raw.empty:
        raise ValueError("No race laps loaded")
    validate_race_identity(raw, row["year"], row["round"])
    clean, _ = cleaning.clean_laps(raw)
    row["n_laps_clean"] = len(clean)
    weather = session.weather_data
    missing = sorted({"Time", "TrackTemp", "AirTemp", "Rainfall"} - set(weather.columns))
    if missing or weather.empty:
        raise ValueError("Missing weather data/channels: " + ", ".join(missing))
    rainfall = weather["Rainfall"]
    if weather["Time"].isna().any() or rainfall.isna().any() or not rainfall.isin([True, False]).all():
        raise ValueError("Cannot verify dry conditions: unknown weather timestamps or Rainfall")
    row["wet_compound_share"] = float(raw["Compound"].isin(["INTERMEDIATE", "WET"]).mean())
    row["any_rainfall"] = bool(rainfall.eq(True).any())
    row["unknown_compound_share"] = float((~raw["Compound"].isin(config.DRY_COMPOUNDS | {"INTERMEDIATE", "WET"})).mean())
    row["clean_fraction"] = len(clean) / len(raw)
    track = session.track_status
    row["red_flag_session"] = bool(not track.empty and "Status" in track and track["Status"].eq("5").any())
    row["drivers_without_laps"] = ",".join(getattr(session, "drivers_without_laps", ()))
    if row["wet_compound_share"] > config.WET_COMPOUND_SHARE_MAX:
        row["auto_exclude_reason"] = "wet_compound"
    elif row["any_rainfall"]:
        row["auto_exclude_reason"] = "rainfall"
    else:
        row["auto_exclude_reason"] = ""
    row["auto_include"] = not row["auto_exclude_reason"]
    provenance = getattr(session, "provenance", {})
    row["data_source"] = provenance.get("source", "FastF1")
    row["source_commit"] = provenance.get("commit", "")


def _review_reasons(row: dict[str, Any]) -> list[str]:
    reasons = []
    if row["auto_exclude_reason"] == "load_error":
        reasons.append("load_error")
    else:
        share = row["wet_compound_share"]
        if row["any_rainfall"] and share <= config.WET_COMPOUND_SHARE_MAX:
            reasons.append("rainfall_without_wet_threshold")
        if share > config.WET_COMPOUND_SHARE_MAX and not row["any_rainfall"]:
            reasons.append("wet_tyres_without_recorded_rain")
        if 0 < share <= config.WET_COMPOUND_SHARE_MAX:
            reasons.append("sparse_wet_tyre_use")
        if row["unknown_compound_share"] > 0:
            reasons.append("unknown_compound")
        if row["auto_include"] and row["clean_fraction"] < 0.5:
            reasons.append("low_clean_retention")
        if row["n_laps_clean"] == 0:
            reasons.append("no_clean_laps")
        if row["red_flag_session"]:
            reasons.append("red_flag_session")
        if row["drivers_without_laps"]:
            reasons.append("roster_drivers_without_laps")
    return reasons


def apply_manual_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit human directives while retaining the automatic evidence."""
    result = frame.copy(deep=True)
    result["manual_override"] = result["manual_override"].fillna("").astype(str).str.strip().str.lower()
    if not result["manual_override"].isin(["", "include", "exclude"]).all():
        raise ValueError("Invalid manual_override; use blank, include, or exclude")
    if "auto_include" not in result:
        result["auto_include"] = result["include"]
        result["auto_exclude_reason"] = result["exclude_reason"].fillna("")
    result["include"] = result["auto_include"].astype(bool)
    result["exclude_reason"] = result["auto_exclude_reason"].fillna("")
    forced_in = result["manual_override"].eq("include") & result["auto_exclude_reason"].ne("load_error")
    forced_out = result["manual_override"].eq("exclude")
    result.loc[forced_in, "include"] = True
    result.loc[forced_in, "exclude_reason"] = ""
    result.loc[forced_out, "include"] = False
    result.loc[forced_out & result["exclude_reason"].eq(""), "exclude_reason"] = "manual_override"
    return result


def build_manifest(
    years: Sequence[int] | None = None, *, source: str = "auto", previous: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Inspect every scheduled race; catch individual load/schema failures and continue."""
    if source not in {"auto", "archive", "fastf1"}:
        raise ValueError(f"Unknown Phase 2 source: {source}")
    selected_years = list(config.TRAIN_YEARS + config.TEST_YEARS if years is None else years)
    for year in selected_years:
        split_for_year(year)
    overrides = {}
    if previous is not None and not previous.empty:
        if previous["race_id"].duplicated().any():
            raise ValueError("Duplicate race IDs in previous manifest")
        for item in previous.fillna("").to_dict("records"):
            directive = str(item.get("manual_override", "")).strip().lower()
            if directive not in {"", "include", "exclude"}:
                raise ValueError(f"Invalid manual_override for {item['race_id']}")
            overrides[item["race_id"]] = (directive, str(item.get("manual_note", "")))
    get_schedule = data.get_race_schedule if source == "fastf1" else archive.get_catalog_schedule
    schedule = pd.concat([get_schedule(year) for year in selected_years], ignore_index=True)
    if schedule.empty or schedule.duplicated(["year", "round"]).any():
        raise ValueError("Race schedule is empty or contains duplicate race identities")
    rows = []
    for position, event in enumerate(schedule.to_dict("records"), 1):
        year, round_number = int(event["year"]), int(event["round"])
        identifier = data.race_id(year, round_number)
        directive, note = overrides.get(identifier, ("", ""))
        row = {
            "race_id": identifier, "year": year, "round": round_number,
            "event_name": event["event_name"], "circuit": event["location"], "split": split_for_year(year),
            "n_laps_raw": None, "n_laps_clean": None, "wet_compound_share": None, "any_rainfall": None,
            "auto_include": False, "auto_exclude_reason": "load_error", "include": False,
            "exclude_reason": "load_error", "manual_override": directive, "manual_note": note,
            "error_detail": "", "unknown_compound_share": None, "clean_fraction": None,
            "red_flag_session": False, "drivers_without_laps": "", "data_source": "", "source_commit": "",
        }
        _logger.info("Manifest %d/%d: %s (%s)", position, len(schedule), identifier, row["event_name"])
        try:
            _populate(row, load_phase2_race(year, round_number, source))
        except Exception as exc:
            row["auto_include"] = False
            row["auto_exclude_reason"] = "load_error"
            row["error_detail"] = f"{type(exc).__name__}: {exc}"
            _logger.warning("Excluded %s: %s", identifier, row["error_detail"])
        reasons = _review_reasons(row)
        if directive:
            reasons.append("manual_override")
        row["review_reasons"] = ";".join(reasons)
        row["needs_manual_review"] = bool(reasons)
        rows.append(row)
    result = apply_manual_overrides(pd.DataFrame(rows))
    for name in ("n_laps_raw", "n_laps_clean"):
        result[name] = result[name].astype("Int64")
    result["any_rainfall"] = pd.array(result["any_rainfall"], dtype="boolean")
    return validate_manifest(result)


def _boolean(series: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    normalized = series.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False})
    missing = series.isna() | series.eq("")
    if (normalized.isna() & ~missing).any() or (not nullable and normalized.isna().any()):
        raise ValueError(f"Invalid boolean values in manifest {name}")
    return normalized.astype("boolean" if nullable else bool)


def validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate manifest identity, split labels and explicit booleans before use."""
    missing = sorted(set(MANIFEST_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Missing manifest columns: " + ", ".join(missing))
    result = frame.copy(deep=True)
    if result["race_id"].duplicated().any():
        raise ValueError("Duplicate race IDs in manifest")
    for name in ("year", "round", "n_laps_raw", "n_laps_clean"):
        values = result[name].mask(result[name].eq(""), pd.NA)
        result[name] = pd.to_numeric(values, errors="raise").astype("Int64")
    for item in result[["race_id", "year", "round", "split"]].to_dict("records"):
        if item["race_id"] != data.race_id(item["year"], item["round"]) or item["split"] != split_for_year(item["year"]):
            raise ValueError("Manifest race identity or split does not match fixed season configuration")
    for name in ("include", "auto_include", "needs_manual_review", "red_flag_session"):
        if name in result:
            result[name] = _boolean(result[name], name)
    result["any_rainfall"] = _boolean(result["any_rainfall"], "any_rainfall", nullable=True)
    for name in ("exclude_reason", "auto_exclude_reason", "manual_override", "manual_note", "review_reasons", "error_detail"):
        if name in result:
            result[name] = result[name].fillna("").astype(str)
    if not result["exclude_reason"].isin(["", "load_error", "wet_compound", "rainfall", "manual_override"]).all():
        raise ValueError("Invalid manifest exclusion reason")
    return apply_manual_overrides(result)


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a generated manifest and apply any newly supplied human directives."""
    return validate_manifest(pd.read_csv(path, keep_default_na=False))


def summarise_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Count each race exactly once by season, split and primary decision reason."""
    result = frame.copy()
    result["decision"] = result["include"].map({True: "included", False: "excluded"})
    result["reason"] = result["exclude_reason"].where(~result["include"], "included")
    return result.groupby(["year", "split", "decision", "reason"], dropna=False).size().rename("races").reset_index()


def manual_review_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Select flagged race evidence for a human review without changing decisions."""
    columns = ["race_id", "year", "event_name", "include", "exclude_reason", "review_reasons", "wet_compound_share", "any_rainfall", "n_laps_raw", "n_laps_clean", "error_detail", "manual_override", "manual_note"]
    return frame.loc[frame["needs_manual_review"], columns].copy()
