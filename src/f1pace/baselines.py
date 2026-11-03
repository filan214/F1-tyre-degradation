"""Four deterministic next-lap baselines, expressed in seconds."""

import numpy as np
import pandas as pd


def attach_stint_history(df: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly earlier clean-stint OLS statistics to eligible feature rows.

    The full clean history is necessary: eligible feature rows omit the opening
    laps and rows after gaps, which must still contribute to the stint trend.
    """
    source = clean.rename(columns={'Driver': 'driver', 'LapNumber': 'lap_number', 'Stint': 'stint'}).copy()
    keys = ['race_id', 'driver', 'lap_number', 'stint']
    if source[keys].isna().any().any() or source.duplicated(keys[:3]).any():
        raise ValueError('Invalid or duplicate clean history identities')
    source = source.sort_values(keys[:3]).reset_index(drop=True)
    if not np.isfinite(source[['lap_number', 'lap_time_s']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite clean history')
    groups = [source[k] for k in ['race_id', 'driver', 'stint']]
    n = source.groupby(['race_id', 'driver', 'stint']).cumcount()
    x, y = source['lap_number'].astype(float), source['lap_time_s'].astype(float)

    def past_sum(values: pd.Series) -> pd.Series:
        return values.groupby(groups).cumsum().groupby(groups).shift(1)

    sx, sy, sxx, sxy = (past_sum(v) for v in (x, y, x*x, x*y))
    denominator = n*sxx-sx*sx
    source['_b4_mean_lap'] = sx / n.where(n.ge(3))
    source['_b4_mean_time'] = sy / n.where(n.ge(3))
    source['_b4_slope'] = (n*sxy-sx*sy) / denominator.where(n.ge(3) & denominator.gt(0))
    extra = ['_b4_mean_lap', '_b4_mean_time', '_b4_slope']
    result = df.drop(columns=extra, errors='ignore').merge(source[keys+extra], on=keys, how='left', validate='one_to_one', sort=False)
    result.index = df.index
    if not np.isfinite(result[extra].to_numpy(dtype=float)).all():
        raise ValueError('Missing or insufficient clean stint history for B4')
    return result


def b1_last_lap(df: pd.DataFrame) -> pd.Series:
    """Predict the previous clean lap time in seconds."""
    return df['lag1'].copy().rename('pred_b1')


def b2_mean3(df: pd.DataFrame) -> pd.Series:
    """Predict the prior-three-clean-lap mean in seconds."""
    return df['mean_last3'].copy().rename('pred_b2')


def b3_mean5(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Predict the five-lap mean, or B2; return the per-row fallback flag."""
    fallback = df[['lag4', 'lag5']].isna().any(axis=1).rename('b3_fallback')
    prediction = df[[f'lag{k}' for k in range(1, 6)]].mean(axis=1, skipna=False)
    return prediction.where(~fallback, df['mean_last3']).rename('pred_b3'), fallback


def b4_stint_trend(df: pd.DataFrame) -> pd.Series:
    """Extrapolate all strictly past clean stint laps' OLS line to the current lap."""
    return (df['_b4_mean_time'] + df['_b4_slope'] * (df['lap_number'] - df['_b4_mean_lap'])).rename('pred_b4')


def all_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Return four aligned lap-time prediction columns on identical rows."""
    result = pd.concat([b1_last_lap(df), b2_mean3(df), b3_mean5(df)[0], b4_stint_trend(df)], axis=1)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError('Baseline predictions must all be finite on the same rows')
    return result
