import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ('src', 'models', 'allocation', 'analysis'):
    sys.path.insert(0, os.path.join(ROOT, sub))


@pytest.fixture(scope='session')
def cfg():
    with open(os.path.join(ROOT, 'configs', 'experiment_config.yaml')) as f:
        return yaml.safe_load(f)


@pytest.fixture
def fast_cfg(cfg):
    """A copy of the real config with tiny model grids, for tests that need
    to actually run train_all_models end-to-end without paying for the real
    (48-combo XGBoost / 9-combo RF) hyperparameter search."""
    import copy
    c = copy.deepcopy(cfg)
    c['models']['xgboost'] = {
        'max_depth': [3], 'learning_rate': [0.1], 'n_estimators': [20],
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'eval_metric': 'logloss', 'early_stopping_rounds': 5,
    }
    c['models']['random_forest'] = {
        'n_estimators': 20, 'max_depth': [5], 'min_samples_leaf': [1],
    }
    c['models']['logistic_regression'] = {
        'C': [1.0], 'max_iter': 200, 'solver': 'lbfgs',
    }
    return c


def all_config_keys(cfg):
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]
