"""Current and future lap targets must never change model inputs for lap t."""

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from f1pace import features


@pytest.mark.parametrize("lap_t", [5, 8, 11])
def test_changing_lap_t_and_every_later_lap_leaves_model_features_unchanged(clean_stint: pd.DataFrame, lap_t: int) -> None:
    """Catch target leakage through lags, expanding slope, counts or aggregate context."""
    before = features.build_features(clean_stint)
    altered = clean_stint.copy(deep=True)
    future = altered["LapNumber"].ge(lap_t)
    altered.loc[future, "lap_time_s"] += [1000. + 137 * i for i in range(int(future.sum()))]
    after = features.build_features(altered)
    columns = features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES
    original_row = before.loc[before["lap_number"].eq(lap_t), columns]
    changed_row = after.loc[after["lap_number"].eq(lap_t), columns]
    assert len(original_row) == 1
    assert_frame_equal(original_row, changed_row, check_exact=True)
    assert after.loc[after["lap_number"].eq(lap_t), features.TARGET].item() != before.loc[before["lap_number"].eq(lap_t), features.TARGET].item()
