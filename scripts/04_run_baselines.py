"""Phase 4: score four baselines using training rows only."""

import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from f1pace import baselines, config, evaluate, plots, reporting


def run_baselines() -> list[Path]:
    """Read training parquet rows, score identical laps, export and log results."""
    config.ensure_dirs()
    if set(config.TRAIN_YEARS) & set(config.TEST_YEARS):
        raise ValueError('Training and test seasons overlap')
    filters = [('year', 'in', config.TRAIN_YEARS), ('split', '==', 'train')]
    frame = pd.read_parquet(config.PROCESSED_DIR / 'features.parquet', filters=filters, engine='pyarrow')
    clean = pd.read_parquet(config.PROCESSED_DIR / 'laps_clean.parquet', filters=filters, engine='pyarrow')
    for loaded in (frame, clean):
        if loaded.empty or not loaded['year'].isin(config.TRAIN_YEARS).all() or not loaded['split'].eq('train').all():
            raise ValueError('Phase 4 requires nonempty training-only input')
        expected = loaded['year'].astype(str) + '_' + loaded['round'].astype(int).astype(str).str.zfill(2)
        if not loaded['race_id'].eq(expected).all():
            raise ValueError('Training race identity mismatch')
    prepared = baselines.attach_stint_history(frame, clean)
    predictions = baselines.all_baselines(prepared)
    scored = pd.concat([frame, predictions], axis=1)
    _, fallback = baselines.b3_mean5(prepared)
    by_race = evaluate.mae_by_race(scored, list(predictions))
    summary = evaluate.summarise(by_race)
    fallback_by_race = frame[['race_id']].assign(b3_fallback=fallback).groupby('race_id')['b3_fallback'].agg(fallback_rows='sum', eligible_rows='size').reset_index()
    fallback_by_race['fallback_rate'] = fallback_by_race['fallback_rows'] / fallback_by_race['eligible_rows']
    outputs = [config.TABLE_DIR / 'baselines_train_by_race.csv', config.TABLE_DIR / 'baselines_train_summary.csv', config.TABLE_DIR / 'baselines_train_b3_fallback.csv']
    for table, path in zip((by_race, summary, fallback_by_race), outputs):
        table.to_csv(path, index=False)
    outputs.append(plots.plot_baseline_mae(by_race))
    best = summary.iloc[0]
    body = (
        f'Training only: {len(frame)} eligible laps across {len(by_race)} races, years {config.TRAIN_YEARS}. Both parquet reads apply year/split predicates before returning rows. No 2025 rows were loaded for analysis or changed.\n\n'
        'All baselines predict lap-time seconds on identical rows. Summary weights races equally; standard deviation is the sample standard deviation (ddof=1).\n\n'
        + reporting.markdown_table(summary)
        + f"\n\nBest baseline (the bar to beat): **{best['predictor']}**, mean race MAE **{best['mean_mae_s']:.6f} s**. Selected by lowest mean race MAE; ties break by predictor name.\n\n"
        + f'B3 fallback: **{int(fallback.sum())}/{len(fallback)} rows ({fallback.mean():.6%})**. Fallback uses B2 whenever either optional lag is missing.\n\n'
        + 'B4 extrapolates the OLS line over all strictly earlier clean laps of the same race/driver/stint. Its intercept is reconstructed from past mean lap/time; early clean laps omitted by feature eligibility still contribute. These baseline-only statistics are not added to the feature parquet or model feature list.\n\n'
        + 'Saved:\n\n' + '\n'.join(f'- `{p.relative_to(config.ROOT).as_posix()}`' for p in outputs)
        + '\n\nPhase 4 complete. No ML model, tuning, cross-validation or final test evaluation was run.\n'
    )
    reporting.append_phase4_log('training baselines', body)
    print(summary.to_string(index=False))
    print(f"Best: {best['predictor']}; B3 fallback {int(fallback.sum())}/{len(fallback)} ({fallback.mean():.4%})")
    return outputs


def main() -> None:
    """Run only Phase 4 and report failures with a nonzero exit status."""
    reporting.configure_logging()
    try:
        run_baselines()
    except Exception as exc:
        logging.getLogger(__name__).error('Baseline run failed: %s', exc)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
