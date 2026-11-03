"""MAE aggregates identical finite rows and weights races equally."""
import numpy as np
import pandas as pd
import pytest
from f1pace import evaluate


def test_mae_and_equal_race_summary():
    frame = pd.DataFrame({'race_id': ['a','a','a','b'], 'lap_time_s': [100.,100.,100.,100.], 'pred_b1': [101.,101.,101.,103.], 'pred_b2': [102.,102.,102.,102.]})
    assert evaluate.mae(frame.lap_time_s, frame.pred_b1) == 1.5
    by_race = evaluate.mae_by_race(frame, ['pred_b1', 'pred_b2'])
    summary = evaluate.summarise(by_race).set_index('predictor')
    assert summary.loc['pred_b1', 'mean_mae_s'] == 2
    assert summary.loc['pred_b1', 'median_mae_s'] == 2
    assert summary.loc['pred_b1', 'std_mae_s'] == pytest.approx(np.sqrt(2))
    assert summary.loc['pred_b2', 'std_mae_s'] == 0


@pytest.mark.parametrize('prediction', [[1.,np.nan], [1.,np.inf], [1.]])
def test_mae_rejects_missing_or_mismatched_predictions(prediction):
    with pytest.raises(ValueError):
        evaluate.mae([1.,2.], prediction)


def test_mae_rejects_series_index_mismatch():
    with pytest.raises(ValueError, match='align'):
        evaluate.mae(pd.Series([1.,2.], index=[0,1]), pd.Series([1.,2.], index=[1,0]))
