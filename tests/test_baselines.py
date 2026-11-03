"""Baseline predictions are seconds and use strictly past clean history."""
import numpy as np
import pandas as pd
import pytest
from f1pace import baselines, features


def test_baselines_linear_stint_and_fallback(clean_stint):
    """Known 2 s/lap slope predicts lap 5 at 106 seconds, not a delta."""
    frame = features.build_features(clean_stint)
    prepared = baselines.attach_stint_history(frame, clean_stint)
    assert baselines.b1_last_lap(prepared).iloc[0] == 104
    assert baselines.b2_mean3(prepared).iloc[0] == 102
    prediction, fallback = baselines.b3_mean5(prepared)
    assert fallback.tolist() == [True, True, False, False, False, False, False]
    assert prediction.iloc[2] == 104
    np.testing.assert_allclose(baselines.b4_stint_trend(prepared), frame['lap_time_s'])
    assert list(baselines.all_baselines(prepared)) == ['pred_b1', 'pred_b2', 'pred_b3', 'pred_b4']


def test_b4_all_prior_history_across_gap_and_no_future_leakage(clean_stint):
    """OLS uses earlier noneligible clean laps, including before a gap."""
    clean = clean_stint.iloc[:7].copy()
    clean['LapNumber'] = [2, 3, 4, 8, 9, 10, 11]
    clean['lap_time_s'] = [100., 104., 103., 115., 114., 120., 127.]
    frame = features.build_features(clean)
    prepared = baselines.attach_stint_history(frame, clean)
    expected = 656 / 6 + (67 / 29) * (11 - 6)
    assert baselines.b4_stint_trend(prepared).item() == pytest.approx(expected)
    clean.loc[clean['LapNumber'].ge(11), 'lap_time_s'] = 9999
    altered = baselines.attach_stint_history(frame, clean)
    assert baselines.b4_stint_trend(altered).item() == expected


def test_b4_resets_at_stint_and_isolates_driver(clean_stint):
    clean_stint.loc[clean_stint['LapNumber'].ge(7), 'Stint'] = 2
    other = clean_stint.assign(Driver='BBB', lap_time_s=500.)
    clean = pd.concat([clean_stint, other], ignore_index=True).sample(frac=1, random_state=42)
    frame = features.build_features(clean)
    predicted = baselines.b4_stint_trend(baselines.attach_stint_history(frame, clean))
    np.testing.assert_allclose(predicted, frame['lap_time_s'])


def test_b3_requires_both_optional_lags(clean_stint):
    frame = features.build_features(clean_stint).iloc[[3]].copy()
    frame['lag4'] = np.nan
    prediction, flag = baselines.b3_mean5(frame)
    assert flag.item() and prediction.item() == frame['mean_last3'].item()


def test_history_missing_target_identity_rejected(clean_stint):
    frame = features.build_features(clean_stint)
    with pytest.raises(ValueError, match='history'):
        baselines.attach_stint_history(frame, clean_stint.iloc[:3])
