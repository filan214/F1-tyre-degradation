"""Generated race groups stay intact across fixed seasons and grouped folds."""

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from f1pace import config, features


def test_train_test_races_and_groupkfold_have_no_overlap(clean_stint: pd.DataFrame) -> None:
    """Validate the feature builder's race labels consumed by GroupKFold, without fitting."""
    races = []
    for year, round_number in [(2023, 1), (2023, 2), (2023, 3), (2024, 1), (2024, 2), (2025, 1), (2025, 2)]:
        races.append(clean_stint.assign(race_id=f"{year}_{round_number:02d}", year=year, round=round_number, split="test" if year == 2025 else "train"))
    frame = features.build_features(pd.concat(races, ignore_index=True))
    train = frame.loc[frame["split"].eq("train")]
    test = frame.loc[frame["split"].eq("test")]
    assert set(train["race_id"]) == {"2023_01", "2023_02", "2023_03", "2024_01", "2024_02"}
    assert set(test["race_id"]) == {"2025_01", "2025_02"}
    assert set(train["race_id"]).isdisjoint(test["race_id"])
    x, y, groups = features.feature_matrix(train)
    for fitting, validation in GroupKFold(n_splits=config.CV_FOLDS).split(x, y, groups):
        assert set(groups.iloc[fitting]).isdisjoint(groups.iloc[validation])
        assert set(groups.iloc[fitting]) | set(groups.iloc[validation]) == set(train["race_id"])


def test_incorrect_split_is_rejected(clean_stint: pd.DataFrame) -> None:
    """Never accept a test-season race mislabeled as training data."""
    wrong = clean_stint.assign(race_id="2025_01", year=2025, split="train")
    with pytest.raises(ValueError, match="split"):
        features.build_features(wrong)
