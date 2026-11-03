"""Training-only end-to-end orchestration with small synthetic grids."""
import importlib.util
from pathlib import Path
import joblib
import pandas as pd
from f1pace import baselines, config, evaluate, features, models, plots


def test_training_script_filters_inputs_saves_pipelines_and_reports(isolated_config, clean_stint, monkeypatch):
    config.ensure_dirs()
    clean = pd.concat([clean_stint.assign(race_id=f'2024_{r:02d}', round=r, circuit=f'Circuit {r}') for r in range(1, 6)], ignore_index=True)
    frame = features.build_features(clean)
    clean.to_parquet(config.PROCESSED_DIR / 'laps_clean.parquet', index=False)
    frame.to_parquet(config.PROCESSED_DIR / 'features.parquet', index=False)
    preds = baselines.all_baselines(baselines.attach_stint_history(frame, clean))
    by_race = evaluate.mae_by_race(pd.concat([frame, preds], axis=1), list(preds))
    by_race.to_csv(config.TABLE_DIR / 'baselines_train_by_race.csv', index=False)
    evaluate.summarise(by_race).to_csv(config.TABLE_DIR / 'baselines_train_summary.csv', index=False)
    (config.ROOT / 'EXPERIMENT_LOG.md').write_text('# Log\n\n---\n\n## Key results summary\n', encoding='utf-8')
    original_read = pd.read_parquet
    reads = []
    def read(path, **kwargs):
        assert kwargs['filters'] == [('year', 'in', config.TRAIN_YEARS), ('split', '==', 'train')]
        reads.append(Path(path).name)
        return original_read(path, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', read)
    monkeypatch.setattr(models, 'param_grid', lambda name: {'model__alpha': [1]} if name == 'ridge' else {'model__n_estimators': [3], 'model__max_depth': [2]})
    for function in ['plot_cv_comparison', 'plot_cv_importance']:
        original = getattr(plots, function)
        monkeypatch.setattr(plots, function, lambda table, original=original: original(table, synthetic=True))
    path = Path(__file__).resolve().parents[1] / 'scripts' / '05_train_models.py'
    spec = importlib.util.spec_from_file_location('phase5', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_training(estimator_jobs=1)
    assert reads == ['features.parquet', 'laps_clean.parquet']
    for name in ['ridge','rf','xgb']:
        pipeline = joblib.load(config.MODELS_DIR / f'{name}.joblib')
        if name != 'ridge':
            assert pipeline.get_params()['model__n_jobs'] == 1
        assert len(pipeline.predict(features.feature_matrix(frame)[0])) == len(frame)
    summary = pd.read_csv(config.TABLE_DIR / 'cv_summary.csv')
    assert len(summary) == 4 and summary['n_races'].eq(5).all()
    assert (config.FIG_DIR / 'fig07_cv_model_vs_baseline.png').is_file()
    assert (config.FIG_DIR / 'fig08_feature_importance_cv.png').is_file()
    assert not (config.MODELS_DIR / 'TEST_OPENED.txt').exists()
    assert 'Phase 5' in (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
