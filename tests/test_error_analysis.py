"""Hand-calculated breakdowns and analysis-only orchestration on synthetic data."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from f1pace import config, evaluate, plots


def test_breakdown_distinguishes_equal_race_and_pooled_mae():
    frame = pd.DataFrame({'race_id': ['a', 'b', 'b', 'b'], 'compound': ['SOFT'] * 4,
                          'lap_time_s': [100.] * 4, 'pred_rf': [104., 100., 100., 100.]})
    row = evaluate.breakdown(frame, 'pred_rf', 'compound').iloc[0]
    assert row['n_laps'] == 4 and row['n_races'] == 2
    assert row['mae_s'] == 1 and row['mean_race_mae_s'] == 2


@pytest.mark.parametrize('column,value', [('compound', None), ('race_id', None), ('pred_rf', np.nan)])
def test_breakdown_rejects_missing_rows(column, value):
    frame = pd.DataFrame({'race_id': ['a'], 'compound': ['SOFT'], 'lap_time_s': [100.], 'pred_rf': [101.]})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluate.breakdown(frame, 'pred_rf', 'compound')


def test_fixed_group_boundaries_and_input_unchanged():
    from f1pace import error_analysis

    frame = pd.DataFrame({'tyre_age': [1, 5, 6, 10, 11, 20, 21, 30, 31, 100],
                          'n_clean_in_stint': [3, 9, 10, 11, 3, 9, 10, 11, 3, 9],
                          'stint_slope': [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0]})
    original = frame.copy(deep=True)
    grouped = error_analysis.add_analysis_groups(frame)
    assert grouped['tyre_age_bucket'].tolist() == ['1-5', '1-5', '6-10', '6-10', '11-20', '11-20', '21-30', '21-30', '31+', '31+']
    assert grouped['stint_phase'].iloc[:4].tolist() == ['Early (<10 prior clean laps)'] * 2 + ['Late (>=10 prior clean laps)'] * 2
    assert grouped['slope_group'].iloc[:3].tolist() == ['Low (<=0 s/lap)'] * 2 + ['High (>0 s/lap)']
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match='tyre'):
        error_analysis.add_analysis_groups(frame.assign(tyre_age=101))


@pytest.fixture
def analysis_inputs(isolated_config, monkeypatch):
    from f1pace import error_analysis

    config.ensure_dirs()
    frame = pd.DataFrame({'race_id': ['2025_01'] * 3 + ['2025_02'] * 3, 'year': [2025] * 6,
                          'round': [1] * 3 + [2] * 3, 'split': ['test'] * 6,
                          'driver': ['AAA'] * 6, 'lap_number': [5, 15, 25] * 2,
                          'circuit': ['Synthetic A'] * 3 + ['Synthetic B'] * 3,
                          'compound': ['SOFT', 'MEDIUM', 'HARD'] * 2,
                          'tyre_age': [5, 15, 25] * 2, 'n_clean_in_stint': [3, 10, 20] * 2,
                          'stint_slope': [-.1, 0., .1] * 2, 'lap_time_s': [100.] * 6,
                          'pred_rf': [101., 100., 102., 102., 100., 102.], 'pred_b1': [102.] * 6})
    frame.to_parquet(config.PROCESSED_DIR / 'test_predictions.parquet', index=False)
    selection = {'chosen_model': 'rf', 'baseline': 'pred_b1'}
    path = config.MODELS_DIR / 'selection.json'
    path.write_text(json.dumps(selection), encoding='utf-8')
    marker = {**selection, 'artifact_sha256': {'selection.json': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (config.MODELS_DIR / 'TEST_OPENED.txt').write_text(json.dumps(marker) + '\nCOMPLETED synthetic\n', encoding='utf-8')
    evaluate.mae_by_race(frame, ['pred_rf', 'pred_b1']).to_csv(config.TABLE_DIR / 'test_results_by_race.csv', index=False)
    (config.ROOT / 'EXPERIMENT_LOG.md').write_text('# Entries\n\n---\n\n## Key results summary\n\n## Error analysis findings (Phase 7)\n\n---\n\n## Portfolio (Phase 8)\n', encoding='utf-8')
    original = plots.plot_error_breakdown
    monkeypatch.setattr(plots, 'plot_error_breakdown', lambda table, by, model, baseline: original(table, by, model, baseline, synthetic=True))
    return frame


def test_analysis_reads_only_saved_predictions_and_preserves_marker(analysis_inputs, monkeypatch):
    from f1pace import error_analysis
    import joblib

    def forbidden(*args, **kwargs):
        pytest.fail('Analysis must not load a model')

    monkeypatch.setattr(joblib, 'load', forbidden)
    original = pd.read_parquet
    calls = []

    def checked(path, **kwargs):
        assert path.name == 'test_predictions.parquet'
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', checked)
    marker = (config.MODELS_DIR / 'TEST_OPENED.txt').read_bytes()
    tables, findings = error_analysis.run_error_analysis()
    assert len(calls) == 1 and len(findings) == 7
    assert (config.MODELS_DIR / 'TEST_OPENED.txt').read_bytes() == marker
    for by, table in tables.items():
        assert table['n_laps'].sum() == 6
        assert (config.TABLE_DIR / f'error_by_{by}.csv').exists()
    assert (config.FIG_DIR / 'fig10_error_by_tyre_age.png').exists()
    assert (config.FIG_DIR / 'fig11_error_by_compound.png').exists()
    table = tables['compound'].set_index('compound')
    assert table.loc['SOFT', 'improvement_s'] == .5
    assert table.loc['HARD', 'improvement_s'] == 0
    log = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    assert 'Phase 7' in log and 'not causal' in log and 'clean-history proxy' in log


def test_analysis_refuses_unopened_test_before_reading(isolated_config, monkeypatch):
    from f1pace import error_analysis

    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: pytest.fail('No read allowed'))
    with pytest.raises(FileNotFoundError):
        error_analysis.run_error_analysis()


def test_analysis_rejects_changed_phase6_scores(analysis_inputs):
    from f1pace import error_analysis

    path = config.TABLE_DIR / 'test_results_by_race.csv'
    table = pd.read_csv(path)
    table['pred_rf'] += 1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Phase 6'):
        error_analysis.run_error_analysis()
    assert not (config.TABLE_DIR / 'error_by_circuit.csv').exists()


@pytest.mark.parametrize('baseline_error,expected_gain,expected_pct', [(1., -1., -100.), (0., -2., np.nan)])
def test_paired_negative_gain_and_zero_baseline(baseline_error, expected_gain, expected_pct):
    """Retain a model loss and leave percentage undefined for a perfect baseline."""
    from f1pace.error_analysis import compare_breakdown

    frame = pd.DataFrame({'race_id': ['a', 'b'], 'compound': ['SOFT', 'SOFT'],
                          'lap_time_s': [100., 100.], 'pred_rf': [102., 102.],
                          'pred_b1': [100. + baseline_error] * 2})
    result = compare_breakdown(frame, 'pred_rf', 'pred_b1', 'compound').iloc[0]
    assert result['improvement_s'] == expected_gain
    if np.isnan(expected_pct):
        assert np.isnan(result['improvement_pct'])
    else:
        assert result['improvement_pct'] == expected_pct
    assert result['races_lost'] == 2 and result['races_won'] == 0
