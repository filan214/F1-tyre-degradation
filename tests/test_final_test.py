"""Synthetic-only checks for the one-time test opening and saved predictions."""

import json

import numpy as np
import pandas as pd
import pytest

from f1pace import config, features, plots


@pytest.fixture
def final_inputs(isolated_config, clean_stint, monkeypatch):
    """Keep all test artifacts and model spies inside tmp_path."""
    from f1pace import final_test

    config.ensure_dirs()
    clean = clean_stint.assign(year=2025, race_id='2025_01', split='test')
    pd.concat([clean_stint, clean]).to_parquet(config.PROCESSED_DIR / 'laps_clean.parquet', index=False)
    pd.concat([features.build_features(clean_stint), features.build_features(clean)]).to_parquet(config.PROCESSED_DIR / 'features.parquet', index=False)
    selection = {'chosen_model': 'rf', 'baseline': 'pred_b1', 'training_years': [2023, 2024], 'best_params': {name: {} for name in ['ridge', 'rf', 'xgb']}}
    (config.MODELS_DIR / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    (config.ROOT / 'EXPERIMENT_LOG.md').write_text('# Entries\n\n---\n\n## Key results summary\n\n## Deviations from IMPLEMENTATION_GUIDE.md\n\n---\n\n## Error analysis findings (Phase 7)\n', encoding='utf-8')
    calls = []

    class SavedModel:
        def __init__(self, name):
            self.name = name

        def get_params(self):
            return {}

        def predict(self, X):
            assert (config.MODELS_DIR / 'TEST_OPENED.txt').exists()
            assert list(X) == features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES
            calls.append(self.name)
            return np.full(len(X), {'ridge': 3., 'rf': 10., 'xgb': 4.}[self.name])

        def fit(self, *args, **kwargs):
            pytest.fail('Final evaluation must never fit a model')

    for name in ['ridge', 'rf', 'xgb']:
        (config.MODELS_DIR / f'{name}.joblib').write_bytes(b'synthetic placeholder')
    monkeypatch.setattr(final_test.joblib, 'load', lambda path: SavedModel(path.stem))
    original = plots.plot_test_comparison
    monkeypatch.setattr(plots, 'plot_test_comparison', lambda frame, chosen, baseline: original(frame, chosen, baseline, synthetic=True))
    return calls


def test_one_pass_outputs_offsets_negative_results_and_guard(final_inputs, monkeypatch):
    from f1pace import final_test

    original = pd.read_parquet
    reads = []

    def checked(path, **kwargs):
        assert (config.MODELS_DIR / 'TEST_OPENED.txt').exists()
        assert kwargs['filters'] == [('year', 'in', config.TEST_YEARS), ('split', '==', 'test')]
        reads.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', checked)
    final_test.run_final_test()
    assert reads == ['features.parquet', 'laps_clean.parquet']
    assert final_inputs == ['ridge', 'rf', 'xgb']
    summary = pd.read_csv(config.TABLE_DIR / 'test_summary.csv').set_index('predictor')
    assert len(summary) == 7
    assert summary.loc['pred_ridge', 'mean_mae_s'] == 1
    assert summary.loc['pred_rf', 'mean_mae_s'] == 6
    assert summary.loc['pred_xgb', 'mean_mae_s'] == 0
    assert summary.loc['pred_b4', 'mean_mae_s'] == 0
    assert summary.loc['pred_rf', 'races_lost_vs_baseline'] == 1
    assert summary.loc['pred_xgb', 'races_won_vs_baseline'] == 1
    assert summary.loc['pred_b1', 'races_tied_vs_baseline'] == 1
    by_race = pd.read_csv(config.TABLE_DIR / 'test_results_by_race.csv')
    assert by_race['n_laps'].tolist() == [7]
    saved = original(config.PROCESSED_DIR / 'test_predictions.parquet')
    assert saved['year'].eq(2025).all() and len(saved) == 7
    np.testing.assert_allclose(saved['pred_rf'], saved['mean_last3'] + 10)
    assert (config.FIG_DIR / 'fig09_test_model_vs_baseline_by_race.png').exists()
    log = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    assert 'test set is now opened' in log and '**rf**' in log
    assert 'does not beat' in log and '2025_01' in log
    with pytest.raises(FileExistsError, match='i-understand-rerun'):
        final_test.run_final_test()
    assert len(reads) == 2 and len(final_inputs) == 3


def test_explicit_rerun_preserves_opening_and_logs_deviation(final_inputs):
    from f1pace import final_test

    marker = config.MODELS_DIR / 'TEST_OPENED.txt'
    marker.write_text('Original opening timestamp\n', encoding='utf-8')
    final_test.run_final_test(understand_rerun=True)
    assert marker.read_text(encoding='utf-8').startswith('Original opening timestamp\n')
    log = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    deviation = log.split('## Deviations from IMPLEMENTATION_GUIDE.md')[1]
    assert '--i-understand-rerun' in deviation


def test_failure_after_opening_keeps_marker_and_records_failure(final_inputs, monkeypatch):
    from f1pace import final_test

    def fail(*args, **kwargs):
        raise ValueError('synthetic read failure')

    monkeypatch.setattr(pd, 'read_parquet', fail)
    with pytest.raises(ValueError, match='synthetic read failure'):
        final_test.run_final_test()
    assert (config.MODELS_DIR / 'TEST_OPENED.txt').exists()
    log = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    assert 'synthetic read failure' in log


def test_missing_model_fails_before_opening(final_inputs):
    from f1pace import final_test

    (config.MODELS_DIR / 'rf.joblib').unlink()
    with pytest.raises(FileNotFoundError):
        final_test.run_final_test()
    assert not (config.MODELS_DIR / 'TEST_OPENED.txt').exists()
