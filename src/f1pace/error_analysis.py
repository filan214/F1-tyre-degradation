"""Descriptive error breakdowns from the single saved final-test prediction pass."""

import hashlib
import json
import logging

import numpy as np
import pandas as pd

from f1pace import config, evaluate, plots, reporting

_logger = logging.getLogger(__name__)
_DIMENSIONS = ('tyre_age_bucket', 'compound', 'circuit', 'stint_phase', 'slope_group')
_AGE_LABELS = ('1-5', '6-10', '11-20', '21-30', '31+')


def add_analysis_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Label fixed reporting strata without changing model inputs or predictions."""
    result = frame.copy()
    if not np.isfinite(result[['tyre_age', 'n_clean_in_stint', 'stint_slope']]).all().all():
        raise ValueError('Nonfinite tyre age or stint context')
    if not result['tyre_age'].between(1, config.TYRE_AGE_BUCKETS[-1]).all():
        raise ValueError('Invalid tyre age outside configured buckets')
    if (result['n_clean_in_stint'] < 3).any():
        raise ValueError('Eligible rows require at least three prior clean laps')
    result['tyre_age_bucket'] = pd.cut(result['tyre_age'], bins=config.TYRE_AGE_BUCKETS, labels=_AGE_LABELS, right=True)
    result['stint_phase'] = pd.Categorical(np.where(result['n_clean_in_stint'] < 10, 'Early (<10 prior clean laps)', 'Late (>=10 prior clean laps)'), categories=['Early (<10 prior clean laps)', 'Late (>=10 prior clean laps)'], ordered=True)
    result['slope_group'] = pd.Categorical(np.where(result['stint_slope'] <= 0, 'Low (<=0 s/lap)', 'High (>0 s/lap)'), categories=['Low (<=0 s/lap)', 'High (>0 s/lap)'], ordered=True)
    return result


def compare_breakdown(frame: pd.DataFrame, model: str, baseline: str, by: str) -> pd.DataFrame:
    """Compare frozen predictors on the same rows, with support and weighting clear."""
    model_table = evaluate.breakdown(frame, model, by).rename(columns={'mae_s': 'model_pooled_mae_s', 'mean_race_mae_s': 'model_mean_race_mae_s'})
    base_table = evaluate.breakdown(frame, baseline, by).rename(columns={'mae_s': 'baseline_pooled_mae_s', 'mean_race_mae_s': 'baseline_mean_race_mae_s'})
    table = model_table.merge(base_table, on=[by, 'n_laps', 'n_races'], validate='one_to_one')
    table['model'] = model
    table['baseline'] = baseline
    table['improvement_s'] = table['baseline_mean_race_mae_s'] - table['model_mean_race_mae_s']
    table['improvement_pct'] = 100 * table['improvement_s'] / table['baseline_mean_race_mae_s'].replace(0, np.nan)
    counts = []
    for label, group in frame.groupby(by, observed=True, sort=True):
        errors = evaluate.mae_by_race(group, [model, baseline])
        counts.append({by: label, 'races_won': int((errors[model] < errors[baseline]).sum()),
                       'races_lost': int((errors[model] > errors[baseline]).sum()),
                       'races_tied': int((errors[model] == errors[baseline]).sum())})
    return table.merge(pd.DataFrame(counts), on=by, validate='one_to_one')


def _read_saved_results() -> tuple[pd.DataFrame, str, str]:
    marker = (config.MODELS_DIR / 'TEST_OPENED.txt').read_text(encoding='utf-8').splitlines()
    if not marker or not any(line.startswith('COMPLETED ') for line in marker[1:]):
        raise ValueError('Phase 6 must be completed before analysis')
    opened = json.loads(marker[0])
    selection_path = config.MODELS_DIR / 'selection.json'
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != opened['artifact_sha256']['selection.json']:
        raise ValueError('Training selection changed after test opening')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection['chosen_model'], selection['baseline']) != (opened['chosen_model'], opened['baseline']):
        raise ValueError('Selection disagrees with test marker')
    model, baseline = 'pred_' + selection['chosen_model'], selection['baseline']
    frame = pd.read_parquet(config.PROCESSED_DIR / 'test_predictions.parquet', engine='pyarrow')
    if frame.empty or not frame['year'].isin(config.TEST_YEARS).all() or not frame['split'].eq('test').all():
        raise ValueError('Analysis requires saved test-only predictions')
    expected = frame['year'].astype(str) + '_' + frame['round'].astype(int).astype(str).str.zfill(2)
    if not frame['race_id'].eq(expected).all() or frame.duplicated(['race_id', 'driver', 'lap_number']).any():
        raise ValueError('Invalid saved test row identities')
    current = evaluate.mae_by_race(frame, [model, baseline]).set_index('race_id').sort_index()
    phase6 = pd.read_csv(config.TABLE_DIR / 'test_results_by_race.csv').set_index('race_id').sort_index()
    if not current.index.equals(phase6.index) or not np.allclose(current[['n_laps', model, baseline]], phase6[['n_laps', model, baseline]], rtol=0, atol=1e-12):
        raise ValueError('Saved predictions disagree with Phase 6 scores')
    return frame, model, baseline


def _describe(table: pd.DataFrame, by: str) -> str:
    return '; '.join(f'{row[by]}: model {row["model_mean_race_mae_s"]:.4f} s vs baseline {row["baseline_mean_race_mae_s"]:.4f} s (gain {row["improvement_s"]:+.4f} s; {row["n_laps"]:,} laps, {row["n_races"]} races)' for row in table.to_dict('records'))


def _findings(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], model: str, baseline: str) -> list[str]:
    race = evaluate.mae_by_race(frame, [model, baseline])
    overall_model, overall_base = race[model].mean(), race[baseline].mean()
    wins, losses = int((race[model] < race[baseline]).sum()), int((race[model] > race[baseline]).sum())
    circuits = tables['circuit'].sort_values('improvement_s')
    worst, best = circuits.iloc[0], circuits.iloc[-1]
    return [
        f'The training-selected {model} has mean race MAE {overall_model:.6f} s versus {overall_base:.6f} s for {baseline}, winning {wins} and losing {losses} of {len(race)} races. The comparison retains all {len(frame):,} saved eligible laps; no model was selected using these results.',
        'Tyre-age results: ' + _describe(tables['tyre_age_bucket'], 'tyre_age_bucket') + '. These describe performance by age, not the incremental effect of the tyre-age feature.',
        'Compound results: ' + _describe(tables['compound'], 'compound') + '. Compound groups can contain different races and drivers, so their error differences do not establish tyre performance differences.',
        'Early/late clean-history proxy: ' + _describe(tables['stint_phase'], 'stint_phase') + '. Early means fewer than 10 strictly earlier clean laps in the stint; late means at least 10. Removed laps are not counted, so this is not physical stint age or proximity to a pit stop.',
        'Signed prior stint trend: ' + _describe(tables['slope_group'], 'slope_group') + '. Low means <=0 s/lap and high means >0 s/lap; high describes rising historical lap times, not greater slope magnitude or proven degradation.',
        f'Across circuits, the largest model gain is {best["circuit"]}: {best["improvement_s"]:+.6f} s ({best["n_laps"]:,} laps, {best["n_races"]} races); the smallest gain is {worst["circuit"]}: {worst["improvement_s"]:+.6f} s ({worst["n_laps"]:,} laps, {worst["n_races"]} races). Negative gains are losses; all circuits remain in error_by_circuit.csv.',
        'These are descriptive, not causal results on the already opened test set. Each reported mean gives equal weight to races represented in that group; tables also give pooled lap MAE and support counts. Groups may differ in race composition and contain few laps. No confidence intervals or independent tyre-age contribution are established; fuel, traffic, pace management and track evolution remain mixed in. No feature or hyperparameter changes follow from this analysis.',
    ]


def run_error_analysis() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Write Phase 7 tables, figures and seven findings without new predictions."""
    config.ensure_dirs()
    document = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    if '\n---\n\n## Key results summary' not in document:
        raise ValueError('Experiment log is missing its entry boundary')
    frame, model, baseline = _read_saved_results()
    grouped = add_analysis_groups(frame)
    tables = {by: compare_breakdown(grouped, model, baseline, by) for by in _DIMENSIONS}
    for by, table in tables.items():
        if table['n_laps'].sum() != len(frame):
            raise ValueError(f'Analysis grouping {by} lost rows')
        table.to_csv(config.TABLE_DIR / f'error_by_{by}.csv', index=False)
    findings = _findings(frame, tables, model, baseline)
    body = 'Analysis of saved Phase 6 predictions only. The test set remains opened; no model was loaded, fitted or predicted again. Frozen training selection and original per-race scores were verified.\n\n'
    body += 'Fixed grouping: configured tyre-age edges ' + str(config.TYRE_AGE_BUCKETS) + '; early <10 versus late >=10 prior clean stint laps; low <=0 versus high >0 signed stint_slope (s/lap). Only observed groups appear; empty groups are not assigned zero error. Primary comparison and figures use equal-race mean MAE within each group. Positive improvement_s means the model wins.\n\n'
    body += '\n\n'.join(f'{index}. {finding}' for index, finding in enumerate(findings, 1))
    for by, table in tables.items():
        body += f'\n\nFull {by} comparison (`reports/tables/error_by_{by}.csv`):\n\n' + reporting.markdown_table(table)
    reporting.append_phase7_log('saved-prediction error analysis', body)
    # Numeric results are logged before rendering so a figure failure cannot hide them.
    for by in ('tyre_age_bucket', 'compound'):
        plots.plot_error_breakdown(tables[by], by, model, baseline)
    _logger.info('Phase 7 saved five comparison tables, two figures and seven findings for %d laps', len(frame))
    return tables, findings
