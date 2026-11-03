"""Tests for ordered, additive cleaning without changing source lap identity."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f1pace import cleaning


def test_each_lap_gets_its_first_removal_reason(synthetic_laps: pd.DataFrame) -> None:
    """Overlapping failures must count against the first matching rule only."""
    result = cleaning.removal_reason(synthetic_laps)
    assert result.tolist() == [f"C{i}" for i in range(1, 11)] + ["kept"] * 4
    assert result.index.equals(synthetic_laps.index)


def test_cleaning_audit_adds_up_and_preserves_identity(synthetic_laps: pd.DataFrame) -> None:
    """Keep original lap/stint/index values and make removals additive."""
    original = synthetic_laps.copy(deep=True)
    clean, audit = cleaning.clean_laps(synthetic_laps)
    assert_frame_equal(clean, original.loc[[f"kept_{i}" for i in range(4)]])
    assert_frame_equal(synthetic_laps, original)
    assert audit.columns.tolist() == ["rule_id", "rule_description", "laps_removed", "laps_remaining"]
    assert audit["rule_id"].tolist() == ["raw"] + [f"C{i}" for i in range(1, 11)]
    assert audit["laps_removed"].tolist() == [0] + [1] * 10
    assert audit["laps_remaining"].tolist() == list(range(14, 3, -1))
    assert audit["laps_removed"].sum() + len(clean) == len(original)


def test_outlier_threshold_uses_survivors_and_is_strict(synthetic_laps: pd.DataFrame) -> None:
    """Exclude earlier-rule failures from the median and retain exactly 107%."""
    base = synthetic_laps.loc[["kept_0"]].copy()
    rows = pd.concat([base] * 5, ignore_index=True)
    rows["LapTime"] = pd.to_timedelta([100, 100, 100, 107, 107.001], unit="s")
    rows["lap_time_s"] = [100, 100, 100, 107, 107.001]
    bad = base.copy()
    bad["lap_time_s"] = 10000
    bad["LapTime"] = pd.to_timedelta(10000, unit="s")
    bad["PitInTime"] = pd.to_timedelta(1, unit="s")
    rows = pd.concat([rows, pd.concat([bad] * 10)], ignore_index=True)
    reasons = cleaning.removal_reason(rows)
    assert reasons.iloc[:5].tolist() == ["kept", "kept", "kept", "kept", "C10"]
    assert reasons.iloc[5:].eq("C3").all()


def test_outlier_threshold_is_per_race(synthetic_laps: pd.DataFrame) -> None:
    """A slower circuit must not be filtered using a different race's median."""
    rows = pd.concat([synthetic_laps.loc[["kept_0"]]] * 6, ignore_index=True)
    rows["race_id"] = ["2024_01"] * 3 + ["2024_02"] * 3
    rows["lap_time_s"] = [70, 70, 76, 150, 150, 155]
    rows["LapTime"] = pd.to_timedelta(rows["lap_time_s"], unit="s")
    assert cleaning.removal_reason(rows).tolist() == ["kept", "kept", "C10", "kept", "kept", "kept"]


def test_empty_cleaning_input_has_complete_zero_audit(synthetic_laps: pd.DataFrame) -> None:
    """Return an empty frame and eleven zero-count audit rows."""
    clean, audit = cleaning.clean_laps(synthetic_laps.iloc[:0])
    assert clean.empty
    assert len(audit) == 11
    assert audit["laps_removed"].eq(0).all()
    assert audit["laps_remaining"].eq(0).all()


def test_all_laps_removed_before_c10(synthetic_laps: pd.DataFrame) -> None:
    """An empty median pool must not manufacture outlier removals."""
    clean, audit = cleaning.clean_laps(synthetic_laps.iloc[:3])
    assert clean.empty
    assert audit.iloc[-1]["laps_removed"] == 0


def test_null_track_status_is_removed(synthetic_laps: pd.DataFrame) -> None:
    """Unknown track status cannot be treated as all-green."""
    rows = synthetic_laps.loc[["kept_0"]].copy()
    rows["TrackStatus"] = pd.NA
    assert cleaning.removal_reason(rows).tolist() == ["C5"]


def test_nullable_boolean_flags_follow_explicit_rules(synthetic_laps: pd.DataFrame) -> None:
    """Null booleans must not crash or silently become explicit True/False."""
    rows = synthetic_laps.loc[["kept_0"]].copy()
    for name in ["Deleted", "IsAccurate", "FastF1Generated"]:
        rows[name] = pd.array([pd.NA], dtype="boolean")
    assert cleaning.removal_reason(rows).tolist() == ["kept"]


def test_validate_clean_laps_accepts_clean_frame(synthetic_laps: pd.DataFrame) -> None:
    """All Phase 1 sanity checks should accept a valid cleaned frame."""
    clean, _ = cleaning.clean_laps(synthetic_laps)
    assert cleaning.validate_clean_laps(clean) is None


@pytest.mark.parametrize("column,value", [("PitInTime", pd.Timedelta(seconds=1)), ("PitOutTime", pd.Timedelta(seconds=1)), ("TrackStatus", "12"), ("Compound", "WET"), ("lap_time_s", 59.0), ("lap_time_s", 201.0)])
def test_validate_clean_laps_rejects_invalid_frame(synthetic_laps: pd.DataFrame, column: str, value: object) -> None:
    """Failed sanity checks must stop exploration before exporting results."""
    clean = synthetic_laps.loc[["kept_0"]].copy()
    clean[column] = value
    with pytest.raises(ValueError):
        cleaning.validate_clean_laps(clean)


def test_validate_clean_laps_rejects_empty(synthetic_laps: pd.DataFrame) -> None:
    """No surviving laps is a failure, rather than a vacuous successful audit."""
    with pytest.raises(ValueError, match="No clean laps"):
        cleaning.validate_clean_laps(synthetic_laps.iloc[:0])
