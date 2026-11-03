"""Fixed past-only lap history, stint context and next-lap targets."""

import logging

import numpy as np
import pandas as pd

from f1pace import data, manifest

_logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "lag1", "lag2", "lag3", "mean_last3", "lag1_minus_lag3", "stint_slope",
    "n_clean_in_stint", "tyre_age", "lap_number", "stint", "track_temp_c", "air_temp_c",
]
CATEGORICAL_FEATURES = ["compound", "circuit", "driver", "team", "fresh_tyre"]
TARGET = "target_delta"

_SOURCE_NAMES = {
    "Driver": "driver", "Team": "team", "LapNumber": "lap_number", "Stint": "stint",
    "TyreLife": "tyre_age", "FreshTyre": "fresh_tyre", "Compound": "compound",
}
_REQUIRED = {
    "race_id", "year", "round", "circuit", "lap_time_s", "track_temp_c", "air_temp_c",
    *_SOURCE_NAMES,
}
_OUTPUT = [
    "race_id", "year", "round", "circuit", "driver", "team", "lap_number", "stint",
    "lag1", "lag2", "lag3", "lag4", "lag5", "mean_last3", "lag1_minus_lag3",
    "stint_slope", "n_clean_in_stint", "tyre_age", "fresh_tyre", "compound",
    "track_temp_c", "air_temp_c", "lap_time_s", TARGET, "split",
]


def _prepare(laps_clean: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED - set(laps_clean))
    if missing:
        raise ValueError("Missing clean-lap feature columns: " + ", ".join(missing))
    frame = laps_clean.rename(columns=_SOURCE_NAMES).copy(deep=True)
    for column in ("race_id", "driver"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing feature identity: {column}")
    for column in ("year", "round", "lap_number", "stint"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or not (values.gt(0) & values.mod(1).eq(0)).all():
            raise ValueError(f"Invalid positive integer feature identity: {column}")
        frame[column] = values.astype("int64")
    expected_split = frame["year"].map(manifest.split_for_year)
    if "split" in frame and not frame["split"].eq(expected_split).all():
        raise ValueError("Feature split does not match configured seasons")
    frame["split"] = expected_split
    expected_races = pd.Series([data.race_id(y, r) for y, r in zip(frame["year"], frame["round"])], index=frame.index, dtype="string")
    if not frame["race_id"].eq(expected_races).all():
        raise ValueError("Feature race identity does not match year/round")
    if frame.duplicated(["race_id", "driver", "lap_number"]).any():
        raise ValueError("Duplicate clean driver-lap identities")
    frame["lap_time_s"] = pd.to_numeric(frame["lap_time_s"], errors="raise").astype(float)
    if not np.isfinite(frame["lap_time_s"]).all() or not frame["lap_time_s"].gt(0).all():
        raise ValueError("Clean lap times must be finite and positive; do not impute lap timing")
    return frame.sort_values(["race_id", "driver", "lap_number"], kind="stable").reset_index(drop=True)


def build_features_with_audit(laps_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build §6 columns and first-removal eligibility counts per fixed split.

    Inputs must already have passed Phase 2 cleaning. No target-dependent cleaning
    is repeated here. Missing optional history stays NaN. All trend statistics
    are shifted before use, so neither lap t nor future targets enter its inputs.
    """
    frame = _prepare(laps_clean)
    history = frame.groupby(["race_id", "driver"], sort=False)
    present = pd.Series(True, index=frame.index)
    consecutive = present.copy()
    same_stint = present.copy()
    optional_valid = present.copy()
    for k in range(1, 6):
        previous_lap = history["lap_number"].shift(k)
        previous_stint = history["stint"].shift(k)
        previous_time = history["lap_time_s"].shift(k)
        optional_valid &= previous_lap.eq(frame["lap_number"] - k) & previous_stint.eq(frame["stint"])
        frame[f"lag{k}"] = previous_time if k <= 3 else previous_time.where(optional_valid)
        if k <= 3:
            present &= previous_lap.notna()
            consecutive &= previous_lap.eq(frame["lap_number"] - k)
            same_stint &= previous_stint.eq(frame["stint"])

    stint_keys = [frame[key] for key in ("race_id", "driver", "stint")]
    stint_groups = frame.groupby(["race_id", "driver", "stint"], sort=False)
    n = stint_groups.cumcount()
    x, y = frame["lap_number"].astype(float), frame["lap_time_s"]

    def previous_sum(values: pd.Series) -> pd.Series:
        return values.groupby(stint_keys, sort=False).cumsum().groupby(stint_keys, sort=False).shift(1)

    sx, sy, sxx, sxy = (previous_sum(values) for values in (x, y, x * x, x * y))
    denominator = n * sxx - sx * sx
    frame["stint_slope"] = ((n * sxy - sx * sy) / denominator.where(denominator.gt(0))).where(n.ge(3))
    frame["n_clean_in_stint"] = n
    frame["mean_last3"] = (frame["lag1"] + frame["lag2"] + frame["lag3"]) / 3
    frame["lag1_minus_lag3"] = frame["lag1"] - frame["lag3"]
    frame[TARGET] = frame["lap_time_s"] - frame["mean_last3"]

    keep = pd.Series(True, index=frame.index)
    audit_rows = []
    rules = [
        ("F1", "Current lap already clean (Phase 2 input)", pd.Series(True, index=frame.index)),
        ("F2", "At least three prior clean laps in race/driver", present),
        ("F3", "Prior three lap numbers exactly consecutive", consecutive),
        ("F4", "Prior three laps in the current stint", same_stint),
    ]
    for rule_id, description, eligible in rules:
        for split in sorted(frame["split"].unique()):
            scope = frame["split"].eq(split)
            before = int((keep & scope).sum())
            after = int((keep & eligible & scope).sum())
            audit_rows.append({"split": split, "rule_id": rule_id, "rule_description": description, "rows_before": before, "rows_dropped": before - after, "rows_after": after})
            _logger.info("%s %s: %d -> %d rows (%d dropped)", split, rule_id, before, after, before - after)
        keep &= eligible
    optional_metadata = [name for name in ("data_source", "source_commit") if name in frame]
    result = frame.loc[keep, _OUTPUT + optional_metadata].copy().reset_index(drop=True)
    audit = pd.DataFrame(audit_rows, columns=["split", "rule_id", "rule_description", "rows_before", "rows_dropped", "rows_after"])
    return result, audit


def build_features(laps_clean: pd.DataFrame) -> pd.DataFrame:
    """Return eligible past-only features and target columns from already-clean laps."""
    return build_features_with_audit(laps_clean)[0]


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return copies of the fixed model inputs, target delta and race groups."""
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = sorted(set(columns + [TARGET, "race_id"]) - set(df))
    if missing:
        raise ValueError("Missing feature matrix columns: " + ", ".join(missing))
    return df[columns].copy(), df[TARGET].copy(), df["race_id"].copy()
