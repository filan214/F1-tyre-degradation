"""Seeded delta-regression pipelines with fold-local preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from f1pace import config, features


def make_pipeline(name: str) -> Pipeline:
    """Build an unfitted pipeline predicting target_delta in seconds."""
    if name == 'ridge':
        model = Ridge(random_state=config.SEED)
    elif name == 'rf':
        model = RandomForestRegressor(n_estimators=300, random_state=config.SEED, n_jobs=-1)
    elif name == 'xgb':
        model = XGBRegressor(objective='reg:squarederror', subsample=0.8, colsample_bytree=0.8, random_state=config.SEED, n_jobs=-1)
    else:
        raise ValueError(f'Unknown model: {name}')
    preprocess = ColumnTransformer([
        ('numeric', StandardScaler() if name == 'ridge' else 'passthrough', features.NUMERIC_FEATURES),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features.CATEGORICAL_FEATURES),
    ])
    return Pipeline([('preprocess', preprocess), ('model', model)])


def param_grid(name: str) -> dict[str, list]:
    """Return the fixed guide grid, using pipeline-qualified parameter names."""
    grids = {
        'ridge': {'model__alpha': [0.1, 1, 10, 100]},
        'rf': {'model__max_depth': [None, 12], 'model__min_samples_leaf': [5, 20]},
        'xgb': {'model__n_estimators': [300, 600], 'model__learning_rate': [0.03, 0.1], 'model__max_depth': [3, 5]},
    }
    if name not in grids:
        raise ValueError(f'Unknown model: {name}')
    return grids[name]
