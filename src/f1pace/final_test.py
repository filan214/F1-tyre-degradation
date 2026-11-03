"""One-time held-out evaluation of frozen pipelines and training-selected choices."""

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from f1pace import baselines, config, evaluate, features, plots, reporting

_MODELS = ('ridge', 'rf', 'xgb')
_BASELINES = ('pred_b1', 'pred_b2', 'pred_b3', 'pred_b4')
_LOG_BOUNDARY = '\n---\n\n## Key results summary'
_DEVIATION_BOUNDARY = '\n---\n\n## Error analysis findings'
_logger = logging.getLogger(__name__)


def _preflight() -> tuple[dict, dict, dict]:
    """Check saved artifacts and log boundaries before opening test data."""
    paths = [config.MODELS_DIR / 'selection.json',
             *(config.MODELS_DIR / f'{name}.joblib' for name in _MODELS),
             config.PROCESSED_DIR / 'features.parquet',
             config.PROCESSED_DIR / 'laps_clean.parquet']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    document = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    if _LOG_BOUNDARY not in document or _DEVIATION_BOUNDARY not in document:
        raise ValueError('Experiment log is missing an entry or deviation boundary')
    selection = json.loads(paths[0].read_text(encoding='utf-8'))
    if selection['chosen_model'] not in _MODELS or selection['baseline'] not in _BASELINES:
        raise ValueError('Invalid training-selected model or baseline')
    if selection['training_years'] != config.TRAIN_YEARS:
        raise ValueError('Selection training seasons differ from config')
    pipelines = {name: joblib.load(config.MODELS_DIR / f'{name}.joblib') for name in _MODELS}
    for name, pipeline in pipelines.items():
        for parameter, value in selection['best_params'][name].items():
            if pipeline.get_params()[parameter] != value:
                raise ValueError(f'Saved {name} parameters differ from training selection')
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths[:4]}
    return selection, pipelines, hashes


def _open_test(marker: Path, *, understand_rerun: bool, selection: dict, hashes: dict) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
    entry = json.dumps({'timestamp': timestamp, 'chosen_model': selection['chosen_model'],
                        'baseline': selection['baseline'], 'artifact_sha256': hashes}) + '\n'
    try:
        with marker.open('x', encoding='utf-8') as handle:
            handle.write(entry)
    except FileExistsError:
        if not understand_rerun:
            raise FileExistsError('Test set already opened; explicit --i-understand-rerun is required') from None
        path = config.ROOT / 'EXPERIMENT_LOG.md'
        document = path.read_text(encoding='utf-8')
        deviation = f'\n\n### {timestamp} — Phase 6 — explicit test rerun\n\n--i-understand-rerun was supplied. This is a repeat evaluation of the opened test set, not a fresh holdout. Original opening marker retained.\n'
        path.write_text(document.replace(_DEVIATION_BOUNDARY, deviation + _DEVIATION_BOUNDARY, 1), encoding='utf-8')
        with marker.open('a', encoding='utf-8') as handle:
            handle.write('EXPLICIT RERUN ' + entry)


def run_final_test(*, understand_rerun: bool = False) -> pd.DataFrame:
    """Open once, predict without fitting, and record every model and baseline result."""
    marker = config.MODELS_DIR / 'TEST_OPENED.txt'
    if marker.exists() and not understand_rerun:
        raise FileExistsError('Test set already opened; explicit --i-understand-rerun is required')
    selection, pipelines, hashes = _preflight()
    config.ensure_dirs()
    _open_test(marker, understand_rerun=understand_rerun, selection=selection, hashes=hashes)
    try:
        result = _evaluate_saved(pipelines, selection)
    except Exception as exc:
        reporting.append_phase6_log('evaluation failed after opening',
                                    f'The test set is now opened. Marker retained. Failure: `{type(exc).__name__}: {exc}`. No automatic rerun.\n')
        raise
    with marker.open('a', encoding='utf-8') as handle:
        handle.write('COMPLETED ' + datetime.now().astimezone().isoformat(timespec='seconds') + '\n')
    return result


def _evaluate_saved(pipelines: dict, selection: dict) -> pd.DataFrame:
    filters = [('year', 'in', config.TEST_YEARS), ('split', '==', 'test')]
    frame = pd.read_parquet(config.PROCESSED_DIR / 'features.parquet', filters=filters, engine='pyarrow')
    clean = pd.read_parquet(config.PROCESSED_DIR / 'laps_clean.parquet', filters=filters, engine='pyarrow')
    for table in (frame, clean):
        if table.empty or not table['year'].isin(config.TEST_YEARS).all() or not table['split'].eq('test').all():
            raise ValueError('Final evaluation requires nonempty test-only inputs')
        if not table['race_id'].str[:4].eq(table['year'].astype(str)).all():
            raise ValueError('Race IDs disagree with test seasons')
    if frame.duplicated(['race_id', 'driver', 'lap_number']).any():
        raise ValueError('Duplicate eligible test rows')
    X, _, _ = features.feature_matrix(frame)
    scored = frame.copy()
    for name, pipeline in pipelines.items():
        _logger.info('Predicting %s on %d test rows / %d races', name, len(frame), frame['race_id'].nunique())
        delta = np.asarray(pipeline.predict(X), dtype=float)
        if delta.shape != (len(frame),) or not np.isfinite(delta).all():
            raise ValueError(f'Invalid {name} predictions')
        scored[f'pred_{name}'] = frame['mean_last3'].to_numpy() + delta
    prepared = baselines.attach_stint_history(frame, clean)
    baseline_predictions = baselines.all_baselines(prepared)
    scored[list(_BASELINES)] = baseline_predictions
    _, fallback = baselines.b3_mean5(prepared)
    scored['b3_fallback'] = fallback
    predictors = [*_BASELINES, *(f'pred_{name}' for name in _MODELS)]
    by_race = evaluate.mae_by_race(scored, predictors)
    summary = evaluate.summarise(by_race)
    baseline = selection['baseline']
    chosen = selection['chosen_model']
    base_mean = float(summary.loc[summary['predictor'].eq(baseline), 'mean_mae_s'].item())
    summary['comparison_baseline'] = baseline
    summary['improvement_s'] = base_mean - summary['mean_mae_s']
    summary['improvement_pct'] = 100 * summary['improvement_s'] / base_mean if base_mean else np.nan
    for label, operation in [('won', 'lt'), ('lost', 'gt'), ('tied', 'eq')]:
        summary[f'races_{label}_vs_baseline'] = [int(getattr(by_race[name], operation)(by_race[baseline]).sum()) for name in summary['predictor']]
    by_race.to_csv(config.TABLE_DIR / 'test_results_by_race.csv', index=False)
    summary.to_csv(config.TABLE_DIR / 'test_summary.csv', index=False)
    scored.to_parquet(config.PROCESSED_DIR / 'test_predictions.parquet', index=False)
    plots.plot_test_comparison(by_race, chosen, baseline)
    selected = summary.loc[summary['predictor'].eq(f'pred_{chosen}')].iloc[0]
    verdict = 'beats' if selected['mean_mae_s'] < base_mean else 'does not beat'
    losers = summary.loc[summary['mean_mae_s'].ge(base_mean) & summary['predictor'].ne(baseline), 'predictor'].tolist()
    losses = by_race.loc[by_race[f'pred_{chosen}'].gt(by_race[baseline]), 'race_id'].tolist()
    body = '**The test set is now opened.** All results below are final held-out 2025 results; no feature, model, hyperparameter or selection changes were made.\n\n'
    body += f'{len(frame):,} eligible laps across {len(by_race)} races; identical rows for all seven predictors. Saved delta predictions were converted with the original mean_last3. B4 uses all strictly earlier clean laps in each stint.\n\n'
    body += f'Training-selected final model remains **{chosen}**; the comparison baseline remains **{baseline}**, selected in Phase 4. No test-based reselection. The chosen model **{verdict}** the baseline on mean per-race MAE.\n\n'
    body += 'Full summary (equal weight per race; sample SD, ddof=1; strict wins and explicit ties):\n\n' + reporting.markdown_table(summary)
    body += '\n\nFull per-race MAE in seconds:\n\n' + reporting.markdown_table(by_race)
    body += f'\n\nNegative results: predictors not beating {baseline} on mean race MAE: {", ".join(losers) or "none"}. Chosen-model race losses: {", ".join(losses) or "none"}.\n\n'
    body += f'B3 fallback: {int(fallback.sum()):,}/{len(fallback):,} ({100 * fallback.mean():.6f}%).\n\n'
    body += 'Saved: reports/tables/test_results_by_race.csv, reports/tables/test_summary.csv, reports/figures/fig09_test_model_vs_baseline_by_race.png, and data/processed/test_predictions.parquet. models/TEST_OPENED.txt records opening time and frozen model hashes. No training, refitting or new data download. Phase 7 has not started.\n'
    reporting.append_phase6_log('final test results', body)
    _logger.info('Test evaluation complete. Selected model %s %s %s.', chosen, verdict, baseline)
    return summary
