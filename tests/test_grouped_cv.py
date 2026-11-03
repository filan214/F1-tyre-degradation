"""Grouped OOF predictions and fold-local importance on synthetic training data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from f1pace import config, evaluate, features, models


@pytest.fixture
def training_frame(clean_stint):
    races = []
    for r in range(1, 6):
        clean = clean_stint.assign(race_id=f'2024_{r:02d}', round=r, circuit=f'Circuit {r}')
        clean['lap_time_s'] += r * np.arange(len(clean)) ** 2 / 20
        races.append(features.build_features(clean))
    return pd.concat(races, ignore_index=True)


def test_grouped_cv_matches_manual_oof_and_original_offset(training_frame, monkeypatch):
    monkeypatch.setattr(models, 'param_grid', lambda name: {'model__alpha': [1.]})
    x, y, groups = features.feature_matrix(training_frame)
    params, result = evaluate.run_grouped_cv('ridge', x, y, groups, training_frame)
    expected = np.empty(len(x))
    for train, valid in GroupKFold(config.CV_FOLDS).split(x, y, groups):
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[valid])
        pipe = models.make_pipeline('ridge').set_params(**params).fit(x.iloc[train], y.iloc[train])
        expected[valid] = pipe.predict(x.iloc[valid]) + training_frame.iloc[valid]['mean_last3']
    np.testing.assert_allclose(result, expected)
    assert result.index.equals(training_frame.index)
    importance = evaluate.grouped_permutation_importance('ridge', params, x, y, groups, training_frame, n_repeats=2)
    assert len(importance) == 5 * 2 * len(x.columns)
    assert set(importance['feature']) == set(x.columns)


def test_cv_rejects_nontraining_and_misalignment(training_frame):
    x, y, groups = features.feature_matrix(training_frame)
    wrong = training_frame.assign(split='test')
    with pytest.raises(ValueError, match='training'):
        evaluate.run_grouped_cv('ridge', x, y, groups, wrong)
    with pytest.raises(ValueError, match='align'):
        evaluate.run_grouped_cv('ridge', x, y, groups.iloc[::-1], training_frame)
