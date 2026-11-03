"""Phase 4 reads only training rows and exports real baseline summaries."""
import importlib.util
from pathlib import Path
import pandas as pd
from f1pace import config, features, plots


def test_training_only_script_exports(isolated_config, clean_stint, monkeypatch):
    """Enforce parquet predicates before loading and verify the full orchestration."""
    config.ensure_dirs()
    clean_stint.to_parquet(config.PROCESSED_DIR / 'laps_clean.parquet', index=False)
    features.build_features(clean_stint).to_parquet(config.PROCESSED_DIR / 'features.parquet', index=False)
    (config.ROOT / 'EXPERIMENT_LOG.md').write_text('# Entries\n\n---\n\n## Key results summary\n', encoding='utf-8')
    read = pd.read_parquet
    calls = []
    def checked(path, **kwargs):
        assert kwargs['filters'] == [('year', 'in', config.TRAIN_YEARS), ('split', '==', 'train')]
        calls.append(Path(path).name)
        return read(path, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', checked)
    original = plots.plot_baseline_mae
    monkeypatch.setattr(plots, 'plot_baseline_mae', lambda frame: original(frame, synthetic=True))
    path = Path(__file__).resolve().parents[1] / 'scripts' / '04_run_baselines.py'
    spec = importlib.util.spec_from_file_location('phase4', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_baselines()
    assert calls == ['features.parquet', 'laps_clean.parquet']
    result = pd.read_csv(config.TABLE_DIR / 'baselines_train_by_race.csv')
    assert result['race_id'].tolist() == ['2024_01']
    assert result['pred_b4'].item() == 0
    assert (config.FIG_DIR / 'fig06_baseline_mae_train.png').is_file()
    log = (config.ROOT / 'EXPERIMENT_LOG.md').read_text(encoding='utf-8')
    assert 'Phase 4' in log and 'pred_b4' in log and 'fallback' in log
