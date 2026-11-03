"""Hand-checked feature values, history eligibility and model-column isolation."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from f1pace import features


def test_hand_built_stint_has_correct_lags_context_and_target(clean_stint: pd.DataFrame) -> None:
    """Catch shifts in the wrong direction and accidental use of current lap time."""
    before = clean_stint.copy(deep=True)
    result = features.build_features(clean_stint)
    assert result["lap_number"].tolist() == [5, 6, 7, 8, 9, 10, 11]
    row = result.iloc[0]
    assert row[["lag1", "lag2", "lag3", "mean_last3", "lag1_minus_lag3"]].tolist() == [104, 102, 100, 102, 4]
    assert row["target_delta"] == 4 and row["lap_time_s"] == 106
    assert row["stint_slope"] == pytest.approx(2)
    assert row["n_clean_in_stint"] == 3
    assert row[["tyre_age", "fresh_tyre", "compound", "driver", "team"]].tolist() == [4, True, "SOFT", "AAA", "Synthetic Team"]
    assert np.isnan(row["lag4"]) and np.isnan(row["lag5"])
    assert result.loc[result["lap_number"].eq(6), "lag4"].item() == 100
    assert result.loc[result["lap_number"].eq(7), "lag5"].item() == 100
    assert_frame_equal(clean_stint, before)


def test_missing_t_minus_two_drops_row_without_renumbering(clean_stint: pd.DataFrame) -> None:
    """A removed lap cannot be bridged by taking the previous three surviving rows."""
    result = features.build_features(clean_stint.loc[clean_stint["LapNumber"].ne(4)])
    assert result["lap_number"].tolist() == [8, 9, 10, 11]
    assert 6 not in result["lap_number"].to_list()


def test_history_never_crosses_stint_change(clean_stint: pd.DataFrame) -> None:
    """A new stint must accumulate three eligible prior laps of its own."""
    clean_stint.loc[clean_stint["LapNumber"].ge(7), "Stint"] = 2
    result = features.build_features(clean_stint)
    assert result["lap_number"].tolist() == [5, 6, 10, 11]
    row = result.loc[result["lap_number"].eq(10)].iloc[0]
    assert row["n_clean_in_stint"] == 3
    assert np.isnan(row["lag4"]) and np.isnan(row["lag5"])


def test_slope_uses_all_earlier_clean_stint_laps_even_before_gap(clean_stint: pd.DataFrame) -> None:
    """The expanding OLS trend must not silently become a last-three regression."""
    source = clean_stint.iloc[:7].copy()
    source["LapNumber"] = [2, 3, 4, 8, 9, 10, 11]
    source["lap_time_s"] = [100., 104., 103., 115., 114., 120., 127.]
    result, audit = features.build_features_with_audit(source)
    assert result["lap_number"].tolist() == [11]
    assert result.iloc[0]["stint_slope"] == pytest.approx(67 / 29)
    assert result.iloc[0]["n_clean_in_stint"] == 6
    assert result.iloc[0][["lag4", "lag5"]].isna().all()
    assert audit["rows_before"].tolist() == [7, 7, 4, 1]
    assert audit["rows_after"].tolist() == [7, 4, 1, 1]
    assert audit["rows_dropped"].tolist() == [0, 3, 3, 0]


def test_histories_are_isolated_by_race_and_driver_with_shuffled_input(clean_stint: pd.DataFrame) -> None:
    """Changing another driver or race must not enter this driver's lag history."""
    other_driver = clean_stint.assign(Driver="BBB", lap_time_s=200.)
    other_race = clean_stint.assign(race_id="2024_02", round=2, lap_time_s=300.)
    mixed = pd.concat([clean_stint, other_driver, other_race], ignore_index=True).sample(frac=1, random_state=42)
    result = features.build_features(mixed)
    first = result.loc[result["race_id"].eq("2024_01") & result["driver"].eq("AAA")].reset_index(drop=True)
    assert_frame_equal(first, features.build_features(clean_stint).reset_index(drop=True))
    assert result.groupby(["race_id", "driver"]).size().tolist() == [7, 7, 7]


@pytest.mark.parametrize("length", [0, 1, 3])
def test_insufficient_history_returns_empty_feature_schema(clean_stint: pd.DataFrame, length: int) -> None:
    """Short sessions should not invent history or fail on empty expanding windows."""
    result = features.build_features(clean_stint.iloc[:length])
    assert result.empty
    assert set(features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES + [features.TARGET, "race_id", "lag4", "lag5"]) <= set(result)


def test_duplicate_driver_lap_is_rejected(clean_stint: pd.DataFrame) -> None:
    """Ambiguous source laps must never turn into consecutive history."""
    with pytest.raises(ValueError, match="Duplicate"):
        features.build_features(pd.concat([clean_stint, clean_stint.iloc[[0]]]))


def test_missing_source_column_is_reported(clean_stint: pd.DataFrame) -> None:
    """Fail explicitly instead of substituting a missing weather or tyre channel."""
    with pytest.raises(ValueError, match="track_temp_c"):
        features.build_features(clean_stint.drop(columns="track_temp_c"))


def test_feature_matrix_excludes_target_identifiers_and_optional_lags(clean_stint: pd.DataFrame) -> None:
    """The consumer must get only the fixed model inputs, aligned target and race groups."""
    frame = features.build_features(clean_stint)
    x, y, groups = features.feature_matrix(frame)
    assert x.columns.tolist() == ["lag1", "lag2", "lag3", "mean_last3", "lag1_minus_lag3", "stint_slope", "n_clean_in_stint", "tyre_age", "lap_number", "stint", "track_temp_c", "air_temp_c", "compound", "circuit", "driver", "team", "fresh_tyre"]
    assert not {"lap_time_s", "target_delta", "lag4", "lag5", "race_id", "split"} & set(x)
    assert y.name == "target_delta" and y.eq(4).all()
    assert groups.eq("2024_01").all() and x.index.equals(y.index) and x.index.equals(groups.index)
    x.iloc[0, 0] = -999
    assert frame.iloc[0]["lag1"] == 104
