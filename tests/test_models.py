"""Pipeline preprocessing and seeded model grids."""
import numpy as np
import pytest
from sklearn.base import clone
from f1pace import models, features, config


@pytest.mark.parametrize('name', ['ridge','rf','xgb'])
def test_pipeline_fits_and_handles_unknown_categories(name, clean_stint):
    frame = features.build_features(clean_stint)
    x, y, _ = features.feature_matrix(frame)
    pipeline = models.make_pipeline(name)
    if name != 'ridge':
        pipeline.set_params(model__n_estimators=3, model__n_jobs=1)
    assert pipeline.get_params()['model__random_state'] == config.SEED
    pipeline.fit(x, y)
    changed = x.copy()
    changed['driver'] = 'UNSEEN'
    changed['circuit'] = 'NEW CIRCUIT'
    assert np.isfinite(pipeline.predict(changed)).all()
    assert not hasattr(clone(pipeline).named_steps['preprocess'], 'transformers_')
    for parameter in models.param_grid(name):
        assert parameter in pipeline.get_params()


def test_grid_sizes_and_unknown_model():
    from sklearn.model_selection import ParameterGrid
    assert [len(ParameterGrid(models.param_grid(n))) for n in ['ridge','rf','xgb']] == [4,4,8]
    with pytest.raises(ValueError):
        models.make_pipeline('unknown')
