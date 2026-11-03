"""Phase 2 clean-lap dataset assembly; no feature construction."""

from dataclasses import dataclass
import logging

import pandas as pd

from f1pace import cleaning, data, manifest

_logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """Successful clean laps, additive audits, failed races and weather diagnostics."""

    laps: pd.DataFrame
    audit: pd.DataFrame
    failures: pd.DataFrame
    review: pd.DataFrame


def assemble_clean_laps(frame: pd.DataFrame, *, source: str = "auto") -> DatasetResult:
    """Load included races, clean before weather, and continue after race failures.

    The caller must inspect failures before publishing a complete dataset. Missing
    matched weather is retained as missing and flagged, never filled from the future.
    """
    selected = manifest.validate_manifest(frame)
    selected = selected.loc[selected["include"]]
    laps, audits, failures, review = [], [], [], []
    for position, row in enumerate(selected.to_dict("records"), 1):
        identifier = row["race_id"]
        metadata = {name: row[name] for name in ("race_id", "year", "round", "split")}
        _logger.info("Dataset %d/%d: %s (%s)", position, len(selected), identifier, row["event_name"])
        try:
            session = manifest.load_phase2_race(int(row["year"]), int(row["round"]), source)
            raw = data.laps_to_frame(session)
            manifest.validate_race_identity(raw, int(row["year"]), int(row["round"]))
            clean, audit = cleaning.clean_laps(raw)
            if pd.isna(row["n_laps_raw"]) or pd.isna(row["n_laps_clean"]) or len(raw) != row["n_laps_raw"] or len(clean) != row["n_laps_clean"]:
                raise ValueError("Lap count differs from manifest; rebuild the manifest first")
            joined = data.attach_weather(clean, session.weather_data)
            joined["split"] = row["split"]
            provenance = getattr(session, "provenance", {})
            joined["data_source"] = provenance.get("source", "FastF1")
            joined["source_commit"] = provenance.get("commit", "")
            missing = joined[["track_temp_c", "air_temp_c", "rainfall"]].isna().any(axis=1)
            if missing.any():
                review.append({**metadata, "issue": "missing_matched_weather", "n_laps": int(missing.sum())})
            for name, value in metadata.items():
                audit[name] = value
            laps.append(joined)
            audits.append(audit)
            _logger.info("%s: %d raw -> %d clean; %d missing weather", identifier, len(raw), len(clean), missing.sum())
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({**metadata, "error_detail": message})
            _logger.error("Dataset race failed %s: %s", identifier, message)
    return DatasetResult(
        laps=pd.concat(laps, ignore_index=True) if laps else pd.DataFrame(),
        audit=pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(columns=["rule_id", "rule_description", "laps_removed", "laps_remaining", "race_id", "year", "round", "split"]),
        failures=pd.DataFrame(failures, columns=["race_id", "year", "round", "split", "error_detail"]),
        review=pd.DataFrame(review, columns=["race_id", "year", "round", "split", "issue", "n_laps"]),
    )
