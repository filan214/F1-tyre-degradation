"""Phase 5: training-only grouped CV, saved pipelines and fold importance."""

import json
import argparse
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from f1pace import baselines, config, evaluate, features, plots, reporting


def run_training(*, estimator_jobs: int = -1) -> list[Path]:
    """Train fixed grids on training races and publish auditable development results."""
    config.ensure_dirs()
    filters = [('year', 'in', config.TRAIN_YEARS), ('split', '==', 'train')]
    frame = pd.read_parquet(config.PROCESSED_DIR / 'features.parquet', filters=filters, engine='pyarrow')
    clean = pd.read_parquet(config.PROCESSED_DIR / 'laps_clean.parquet', filters=filters, engine='pyarrow')
    for table in (frame, clean):
        if table.empty or not table['year'].isin(config.TRAIN_YEARS).all() or not table['split'].eq('train').all():
            raise ValueError('Phase 5 requires training-only inputs')
    X, y, groups = features.feature_matrix(frame)
    # Validate before fitting or computing any performance metrics.
    evaluate._training_splits(X, y, groups, frame)
    phase4 = pd.read_csv(config.TABLE_DIR / 'baselines_train_by_race.csv').set_index('race_id').sort_index()
    phase4_summary = evaluate.summarise(phase4.reset_index())
    baseline = str(phase4_summary.iloc[0]['predictor'])
    prepared = baselines.attach_stint_history(frame, clean)
    predictions = baselines.all_baselines(prepared)
    current = evaluate.mae_by_race(pd.concat([frame, predictions], axis=1), list(predictions)).set_index('race_id').sort_index()
    if not current.index.equals(phase4.index) or not np.allclose(current[['n_laps', *predictions]], phase4[['n_laps', *predictions]], rtol=0, atol=1e-10):
        raise ValueError('Phase 4 scores do not match current training rows; rerun Phase 4')
    scored = frame.copy()
    scored[baseline] = predictions[baseline]
    params = {}
    outputs = []
    folds = None
    for name in ['ridge', 'rf', 'xgb']:
        result = evaluate.fit_grouped_cv(name, X, y, groups, frame, estimator_jobs=estimator_jobs)
        scored[f'pred_{name}'] = result.oof_lap_time
        params[name] = result.best_params
        search_path = config.TABLE_DIR / f'cv_search_{name}.csv'
        result.search_results.to_csv(search_path, index=False)
        model_path = config.MODELS_DIR / f'{name}.joblib'
        temporary = model_path.with_suffix('.joblib.tmp')
        joblib.dump(result.fitted_pipeline, temporary, compress=3)
        temporary.replace(model_path)
        outputs.extend([search_path, model_path])
        if folds is not None and not folds.equals(result.fold_membership):
            raise ValueError('Model CV folds differ')
        folds = result.fold_membership
        partial = evaluate.summarise(evaluate.mae_by_race(scored, [baseline, f'pred_{name}']))
        reporting.append_phase5_log(f'{name} search and OOF complete', f'Training only: {len(frame)} laps / {groups.nunique()} races. Best params: `{json.dumps(result.best_params, sort_keys=True)}`.\n\n' + reporting.markdown_table(partial) + f'\n\nRefitted full-training pipeline saved to `models/{name}.joblib`. Development CV uses parameters selected on the same folds (not nested CV).\n')
    predictors = [baseline, 'pred_ridge', 'pred_rf', 'pred_xgb']
    by_race = evaluate.mae_by_race(scored, predictors)
    summary = evaluate.summarise(by_race)
    base_mean = float(summary.loc[summary['predictor'].eq(baseline), 'mean_mae_s'].item())
    summary['improvement_s'] = base_mean - summary['mean_mae_s']
    summary['improvement_pct'] = 100 * summary['improvement_s'] / base_mean if base_mean else np.nan
    summary['races_won_vs_baseline'] = [int(by_race[name].lt(by_race[baseline]).sum()) for name in summary['predictor']]
    selected = summary.loc[summary['predictor'].ne(baseline)].iloc[0]
    chosen = str(selected['predictor']).removeprefix('pred_')
    by_path, summary_path = config.TABLE_DIR / 'cv_results_by_race.csv', config.TABLE_DIR / 'cv_summary.csv'
    by_race.to_csv(by_path, index=False)
    summary.to_csv(summary_path, index=False)
    folds.to_csv(config.TABLE_DIR / 'cv_fold_membership.csv', index=False)
    scored[['race_id', 'year', 'driver', 'lap_number', 'lap_time_s', *predictors]].to_parquet(config.PROCESSED_DIR / 'cv_oof_train.parquet', index=False)
    outputs.extend([by_path, summary_path, plots.plot_cv_comparison(by_race)])
    importance = evaluate.grouped_permutation_importance(chosen, params[chosen], X, y, groups, frame, estimator_jobs=estimator_jobs)
    importance.to_csv(config.TABLE_DIR / 'cv_importance_by_fold.csv', index=False)
    aggregate = importance.groupby('feature')['importance_s'].agg(mean_importance_s='mean', std_importance_s='std').reset_index()
    aggregate.to_csv(config.TABLE_DIR / 'cv_feature_importance.csv', index=False)
    outputs.append(plots.plot_cv_importance(aggregate))
    selection = {'chosen_model': chosen, 'reason': 'lowest mean per-race training OOF MAE among ML models', 'baseline': baseline, 'baseline_mean_mae_s': base_mean, 'model_mean_mae_s': float(selected['mean_mae_s']), 'beats_baseline': bool(selected['mean_mae_s'] < base_mean), 'best_params': params, 'training_years': config.TRAIN_YEARS, 'seed': config.SEED, 'cv_folds': config.CV_FOLDS}
    (config.MODELS_DIR / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    verdict = 'The chosen model beats the baseline in development CV.' if selection['beats_baseline'] else 'No model beats the best baseline; retain the baseline as the stronger development result.'
    body = f'Training only: {len(frame)} laps / {groups.nunique()} races. Both parquet reads used training-year/split filters. No 2025 rows were loaded, predicted or evaluated.\n\nBest parameters:\n\n```json\n{json.dumps(params, indent=2, sort_keys=True)}\n```\n\n'
    body += reporting.markdown_table(summary)
    body += f"\n\nChosen final ML model: **{chosen}**, with mean race OOF MAE **{selected['mean_mae_s']:.6f} s**, the lowest among ML models (ties break by predictor name). Phase 4 baseline: **{baseline}**, **{base_mean:.6f} s**. {verdict}\n\n"
    body += f'Hyperparameters were selected by GridSearchCV negative MAE on five GroupKFold folds. OOF predictions use the selected parameters and same folds, so these are non-nested development estimates. Each saved pipeline is refitted on all training rows. Preprocessing is fitted inside each pipeline/fold; no global scaler or encoder. Outer jobs are serial, estimator jobs={estimator_jobs}.\n\n'
    body += 'Permutation importance uses freshly fitted pipelines on each training fold and permutes only its held-out validation inputs, five repeats, seed 42. Delta MAE equals lap-time MAE with the original mean_last3 offset held fixed. Error bars are SD across fold/repeat values, not confidence intervals.\n\nSaved:\n\n'
    body += '\n'.join(f'- `{path.relative_to(config.ROOT).as_posix()}`' for path in outputs)
    body += '\n\nAdditional evidence: cv_fold_membership.csv, cv_importance_by_fold.csv, cv_feature_importance.csv, data/processed/cv_oof_train.parquet and models/selection.json. Phase 6 has not started; no test-opened marker created.\n'
    reporting.append_phase5_log('training model selection complete', body)
    print(summary.to_string(index=False))
    print(f'Chosen model: {chosen}. {verdict}')
    return outputs


def main() -> None:
    """Execute Phase 5 only, reporting errors with a nonzero exit status."""
    reporting.configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimator-jobs', type=int, default=-1, help='Use 1 where Windows sandbox worker pools are unavailable')
    args = parser.parse_args()
    try:
        run_training(estimator_jobs=args.estimator_jobs)
    except Exception as exc:
        logging.getLogger(__name__).exception('Phase 5 failed: %s', exc)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
