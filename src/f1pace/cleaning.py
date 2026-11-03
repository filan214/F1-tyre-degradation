"""Ordered lap cleaning with additive first-removal audits."""

import logging

import pandas as pd

from f1pace import config

_logger = logging.getLogger(__name__)

_RULES = (
    ("C1", "Missing lap timing"),
    ("C2", "Standing-start lap"),
    ("C3", "In-lap"),
    ("C4", "Out-lap"),
    ("C5", "Track status not exactly green"),
    ("C6", "Stewards-deleted lap"),
    ("C7", "Unsynced lap timing"),
    ("C8", "FastF1-generated lap"),
    ("C9", "Non-dry or unknown compound"),
    ("C10", "Slower than 107% of surviving race median"),
)
_REQUIRED = {
    "LapTime", "LapNumber", "Stint", "PitInTime", "PitOutTime", "TrackStatus",
    "Deleted", "IsAccurate", "FastF1Generated", "Compound", "lap_time_s",
}


def removal_reason(laps: pd.DataFrame) -> pd.Series:
    """Label each lap with its first matching cleaning rule, or kept."""
    missing = sorted(_REQUIRED - set(laps.columns))
    if missing:
        raise ValueError("Missing cleaning columns: " + ", ".join(missing))
    reasons = pd.Series("kept", index=laps.index, name="removal_reason", dtype="object")
    masks = (
        laps["LapTime"].isna(),
        laps["LapNumber"].eq(1),
        laps["PitInTime"].notna(),
        laps["PitOutTime"].notna(),
        laps["TrackStatus"].ne(config.GREEN_TRACK_STATUS).fillna(True),
        laps["Deleted"].eq(True).fillna(False),
        laps["IsAccurate"].eq(False).fillna(False),
        laps["FastF1Generated"].eq(True).fillna(False),
        ~laps["Compound"].isin(config.DRY_COMPOUNDS),
    )
    for (rule_id, _), mask in zip(_RULES, masks):
        reasons.loc[reasons.eq("kept") & mask] = rule_id

    survivors = laps.loc[reasons.eq("kept")]
    if not survivors.empty:
        if "race_id" in laps:
            medians = survivors.groupby("race_id", dropna=False)["lap_time_s"].median()
            threshold = laps["race_id"].map(medians) * config.OUTLIER_FACTOR
        else:
            threshold = survivors["lap_time_s"].median() * config.OUTLIER_FACTOR
        reasons.loc[reasons.eq("kept") & laps["lap_time_s"].gt(threshold)] = "C10"
    return reasons


def clean_laps(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return retained laps and an additive raw/C1–C10 audit."""
    reasons = removal_reason(laps)
    remaining = len(laps)
    rows = [{"rule_id": "raw", "rule_description": "All input laps", "laps_removed": 0, "laps_remaining": remaining}]
    for rule_id, description in _RULES:
        removed = int(reasons.eq(rule_id).sum())
        remaining -= removed
        rows.append({"rule_id": rule_id, "rule_description": description, "laps_removed": removed, "laps_remaining": remaining})
        _logger.info("%s: removed %d, remaining %d", rule_id, removed, remaining)
    return laps.loc[reasons.eq("kept")].copy(deep=True), pd.DataFrame(rows)


def validate_clean_laps(laps: pd.DataFrame) -> None:
    """Enforce the Phase 1 post-cleaning sanity checks before export."""
    if laps.empty:
        raise ValueError("No clean laps survived")
    if laps[["PitInTime", "PitOutTime"]].notna().any().any():
        raise ValueError("Clean laps contain pit entry/exit timestamps")
    if not laps["TrackStatus"].eq(config.GREEN_TRACK_STATUS).fillna(False).all():
        raise ValueError("Clean laps contain non-green track status")
    if not laps["Compound"].isin(config.DRY_COMPOUNDS).all():
        raise ValueError("Clean laps contain non-dry compounds")
    if not laps["lap_time_s"].between(60, 200).all():
        raise ValueError("Clean lap times must be between 60 and 200 seconds")
