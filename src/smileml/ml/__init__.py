from .hyperband import Hyperband
from .randombits import RandomBitFeatures


__all__ = [
    'Hyperband',
    'RandomBitFeatures'
]


# Import xgboost tools

xgbdeps = [
    'XGBClassifierWithEarlyStopping', 'XGBRegressorWithEarlyStopping',
    'xgboost_hyperband_classifier', 'xgboost_hyperband_regressor',
    'ContinuableXGBClassifier', 'ContinuableXGBRegressor'
]

for dep in xgbdeps:
    template = '''
try:
    from .xgboost import %s
except:
    from ..utils import DummyImport
    %s = DummyImport('Cannot import XGBoost')
    '''
    exec(template % (dep, dep))

__all__ += xgbdeps

# Import lightgbm tools

lgbmdeps = [
    'lgbm_hyperband_classifier', 'lgbm_hyperband_regressor',
    'ContinuableLGBMClassifier', 'ContinuableLGBMRegressor'
]

for dep in lgbmdeps:
    template = '''
try:
    from .lightgbm import %s
except:
    from ..utils import DummyImport
    %s = "DummyImport('Cannot import lightgbm')"
    '''
    exec(template % (dep, dep))

__all__ += lgbmdeps
