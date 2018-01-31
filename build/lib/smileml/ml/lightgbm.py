from sklearn.pipeline import make_pipeline
from scipy.stats.distributions import uniform, randint
from .hyperband import Hyperband
from .preprocessing import simple_proc_for_tree_algoritms
from lightgbm import sklearn as lgbmsk
train_ = lgbmsk.train


def newtrain(params, *args, **kwargs):
    if '_Booster' in params:
        booster = params.pop('_Booster')
        # print('heyhey %s / %s' % (params['n_estimators'], booster.current_iteration()))
        params = dict(params)
        params['n_estimators'] -= booster.current_iteration()
        return train_(params, *args, **kwargs, init_model=booster)
    return train_(params, *args, **kwargs)


lgbmsk.train = newtrain


def lgbm_hyperband_classifier(numeric_features, categoric_features, learning_rate=0.08):
    """
    Simple classification pipeline using hyperband to optimize lightgbm hyper-parameters

    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _lgbm_hyperband_model('classification', numeric_features, categoric_features, learning_rate)


def lgbm_hyperband_regressor(numeric_features, categoric_features, learning_rate=0.08):
    """
    Simple classification pipeline using hyperband to optimize lightgbm hyper-parameters

    Parameters
    ----------
    `numeric_features` : The list of numeric features
    `categoric_features` : The list of categoric features
    `learning_rate` : The learning rate
    """

    return _lgbm_hyperband_model('regression', numeric_features, categoric_features, learning_rate)


def _lgbm_hyperband_model(task, numeric_features, categoric_features, learning_rate=0.08):
    param_space = {
        'num_leaves': randint(3, 99),
        'max_depth': randint(2, 11),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 10),
        'max_bin': randint(100, 400),
        'min_child_weight': randint(1, 10),
        'min_child_samples': randint(1, 11)
    }

    model = ContinuableLGBMClassifier(learning_rate=learning_rate) \
        if task == 'classification' else ContinuableLGBMRegressor(learning_rate=learning_rate)

    return make_pipeline(
        simple_proc_for_tree_algoritms(numeric_features, categoric_features),
        Hyperband(
            model,
            feat_space=param_space,
            task=task
        )
    )


class ContinuableLGBMClassifier(lgbmsk.LGBMClassifier):
    """
    .. code-block:: python

       clf = ContinuableLGBMClassifier(n_estimators=100)
       clf.fit(X, Y)
       clf.set_params(n_estimators=110)
       clf.fit(X, Y)  # train 10 more estimators, not from scratch
    """

    def get_params(self, deep=True):
        ret = super(ContinuableLGBMClassifier, self).get_params()
        if self._Booster is not None:
            ret['_Booster'] = self._Booster
        return ret


class ContinuableLGBMRegressor(lgbmsk.LGBMRegressor):
    """
    .. code-block:: python

       clf = ContinuableLGBMRegressor(n_estimators=100)
       clf.fit(X, Y)
       clf.set_params(n_estimators=110)
       clf.fit(X, Y)  # train 10 more estimators, not from scratch
    """

    def get_params(self, deep=True):
        ret = super(ContinuableLGBMRegressor, self).get_params()
        if self._Booster is not None:
            ret['_Booster'] = self._Booster
        return ret
