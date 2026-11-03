"""Race-level errors and training-only grouped model development."""

from collections.abc import Sequence
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from f1pace import config, features, models

_logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Training search evidence, fully refitted pipeline and OOF lap predictions."""
    best_params: dict
    fitted_pipeline: Pipeline
    oof_lap_time: pd.Series
    search_results: pd.DataFrame
    fold_membership: pd.DataFrame


def _training_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series, df: pd.DataFrame) -> list:
    if df.empty or not df['year'].isin(config.TRAIN_YEARS).all() or not df['split'].eq('train').all() or set(config.TRAIN_YEARS) & set(config.TEST_YEARS):
        raise ValueError('CV requires training-only rows and disjoint configured seasons')
    if not X.index.equals(df.index) or not y.index.equals(df.index) or not groups.index.equals(df.index):
        raise ValueError('CV inputs must align by index')
    if not groups.equals(df['race_id']) or not y.equals(df[features.TARGET]) or not X.equals(df[features.NUMERIC_FEATURES + features.CATEGORICAL_FEATURES]):
        raise ValueError('CV inputs must align with the feature frame')
    expected = df['year'].astype(str) + '_' + df['round'].astype(int).astype(str).str.zfill(2)
    if not df['race_id'].eq(expected).all() or df.duplicated(['race_id', 'driver', 'lap_number']).any():
        raise ValueError('Invalid training race/lap identity')
    if groups.nunique() < config.CV_FOLDS or not np.isfinite(y).all():
        raise ValueError('Insufficient training races or nonfinite targets')
    if not np.allclose(df['mean_last3'] + y, df['lap_time_s']):
        raise ValueError('Target delta and lap-time offset disagree')
    return list(GroupKFold(n_splits=config.CV_FOLDS).split(X, y, groups))


def fit_grouped_cv(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, df: pd.DataFrame, *, estimator_jobs: int = -1) -> CVResult:
    """Search training race folds, refit all training rows, then produce OOF deltas."""
    splits = _training_splits(X, y, groups, df)
    _logger.info('%s: grid search on %d training laps / %d races', name, len(df), groups.nunique())
    pipeline = models.make_pipeline(name)
    if 'model__n_jobs' in pipeline.get_params():
        pipeline.set_params(model__n_jobs=estimator_jobs)
    search = GridSearchCV(pipeline, models.param_grid(name), cv=splits, scoring='neg_mean_absolute_error', refit=True, n_jobs=1, error_score='raise', verbose=2)
    search.fit(X, y, groups=groups)
    _logger.info('%s best params: %s; generating OOF predictions', name, search.best_params_)
    delta = cross_val_predict(search.best_estimator_, X, y, groups=groups, cv=splits, n_jobs=1, method='predict')
    lap_time = pd.Series(delta + df['mean_last3'].to_numpy(), index=df.index, name=f'pred_{name}')
    if not np.isfinite(lap_time).all():
        raise ValueError('Nonfinite OOF predictions')
    membership = []
    for fold, (train, valid) in enumerate(splits, 1):
        if not set(groups.iloc[train]).isdisjoint(groups.iloc[valid]):
            raise ValueError('Race overlap in CV fold')
        membership.extend({'race_id': race, 'validation_fold': fold} for race in sorted(groups.iloc[valid].unique()))
    return CVResult(search.best_params_, search.best_estimator_, lap_time, pd.DataFrame(search.cv_results_), pd.DataFrame(membership))


def run_grouped_cv(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, df: pd.DataFrame, *, estimator_jobs: int = -1) -> tuple[dict, pd.Series]:
    """Return best parameters and grouped OOF lap-time predictions in seconds."""
    result = fit_grouped_cv(name, X, y, groups, df, estimator_jobs=estimator_jobs)
    return result.best_params, result.oof_lap_time


def grouped_permutation_importance(name: str, params: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series, df: pd.DataFrame, *, n_repeats: int = 5, estimator_jobs: int = -1) -> pd.DataFrame:
    """Permute raw inputs only on validation folds of freshly fitted pipelines."""
    rows = []
    for fold, (train, valid) in enumerate(_training_splits(X, y, groups, df), 1):
        _logger.info('%s permutation importance: validation fold %d/%d', name, fold, config.CV_FOLDS)
        pipeline = models.make_pipeline(name).set_params(**params)
        if 'model__n_jobs' in pipeline.get_params():
            pipeline.set_params(model__n_jobs=estimator_jobs)
        pipeline.fit(X.iloc[train], y.iloc[train])
        result = permutation_importance(pipeline, X.iloc[valid], y.iloc[valid], scoring='neg_mean_absolute_error', n_repeats=n_repeats, random_state=config.SEED, n_jobs=1)
        rows.extend({'feature': feature, 'fold': fold, 'repeat': repeat + 1, 'importance_s': float(value)} for i, feature in enumerate(X.columns) for repeat, value in enumerate(result.importances[i]))
    return pd.DataFrame(rows)


def mae(y_true: Sequence[float] | pd.Series, y_pred: Sequence[float] | pd.Series) -> float:
    """Return mean absolute lap-time error in seconds, rejecting missing pairs."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series) and not y_true.index.equals(y_pred.index):
        raise ValueError('Truth and predictions must align by index')
    truth, prediction = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or truth.shape != prediction.shape or not truth.size:
        raise ValueError('MAE requires equal nonempty one-dimensional arrays')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('MAE requires finite truth and predictions on all rows')
    return float(np.abs(truth-prediction).mean())


def mae_by_race(df: pd.DataFrame, pred_cols: list[str], truth_col: str = 'lap_time_s') -> pd.DataFrame:
    """Return one wide MAE row per race, with identical rows for each predictor."""
    if df.empty or not pred_cols or df['race_id'].isna().any():
        raise ValueError('Race evaluation requires rows, predictors and race identities')
    return pd.DataFrame([{'race_id': race, 'n_laps': len(group), **{name: mae(group[truth_col], group[name]) for name in pred_cols}} for race, group in df.groupby('race_id', sort=True)])


def summarise(by_race: pd.DataFrame) -> pd.DataFrame:
    """Summarise equally weighted race MAEs; std uses sample ddof=1."""
    columns = [c for c in by_race if c.startswith('pred_')]
    if by_race.empty or not columns or by_race['race_id'].duplicated().any():
        raise ValueError('Summary requires unique races and prediction MAEs')
    if not np.isfinite(by_race[columns].to_numpy(dtype=float)).all():
        raise ValueError('Race MAEs must all be finite')
    return pd.DataFrame([{'predictor': name, 'n_races': len(by_race), 'mean_mae_s': by_race[name].mean(), 'median_mae_s': by_race[name].median(), 'std_mae_s': by_race[name].std(ddof=1)} for name in columns]).sort_values(['mean_mae_s', 'predictor']).reset_index(drop=True)


def breakdown(df: pd.DataFrame, pred_col: str, by: str) -> pd.DataFrame:
    """Return pooled and equal-race MAE per observed group, without dropping rows."""
    required = {by, 'race_id', 'lap_time_s', pred_col}
    if missing := required - set(df):
        raise ValueError(f'Missing breakdown columns: {sorted(missing)}')
    if df.empty or df[[by, 'race_id']].isna().any().any():
        raise ValueError('Breakdown requires nonempty rows and nonmissing groups/races')
    mae(df['lap_time_s'], df[pred_col])
    rows = []
    for label, group in df.groupby(by, observed=True, sort=True):
        per_race = mae_by_race(group, [pred_col])
        rows.append({by: label, 'n_laps': len(group), 'n_races': len(per_race),
                     'mae_s': mae(group['lap_time_s'], group[pred_col]),
                     'mean_race_mae_s': per_race[pred_col].mean()})
    return pd.DataFrame(rows)
