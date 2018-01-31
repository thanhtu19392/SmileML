from sklearn import metrics
from sklearn.externals.joblib import Parallel, delayed
from scipy.stats.mstats import mquantiles
import numpy as np
import pandas as pd


def _is_classication(model, Xval):
    try:
        model.predict_proba(Xval[:3])
        return True
    except:
        return False


def compute_features_impact(model, Xval, Yval, row_sample=50000, features=None, n_jobs=-1):
    """Compute the features impact for each features.

    Feature Impact for a given column measures how much worse a
    model’s error score would be if we made predictions after randomly
    shuffling that column (while leaving other columns unchanged).
    This technique is sometimes called Permutation Importance.

    The calculation may take time. To speed up you can either:
    - Use sampling with the `row_sample` parameter (in number of rows, 0 for all rows)
    - Specify the columns of interest with `features` paramters (list of columns, None for all)
    """

    if not features:
        features = Xval.columns

    Xval, Yval = _row_sample(row_sample, Xval, Yval)

    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    scores = parallel(
        delayed(_shuffle_and_score)(
            model, Xval, Yval, col)
        for col in [None] + features.tolist())
    perf0, perfs = scores[0], scores[1:]
    perfs = pd.Series(perfs, features)
    return (1 - perfs / perf0).sort_values(ascending=False)


def _shuffle_and_score(model, Xval, Yval, col):
    if _is_classication(model, Xval):
        metric_func = metrics.roc_auc_score

        def predict_fn(X):
            return model.predict_proba(X)[:, 1]
    else:
        metric_func = metrics.r2_score
        predict_fn = model.predict

    if col is not None:
        Xval = Xval.copy()
        Xval[col] = np.random.permutation(Xval[col])
    return metric_func(Yval, predict_fn(Xval))


def compute_partial_dependence(model, Xval, features=None,
                               row_sample=10000,
                               percentiles=(0.05, 0.95), grid_resolution=20,
                               n_jobs=-1):
    """
    Compute the partial dependence (of each feature on the prediction function)

    For a linear model, we can look at the regression coefficients to tell whether
    a feature impacts positively or negatively the predictions

    For a more complex model, we use `partial dependence` to visualize this relationship

    The calculation may take time. To speed up you can either:
    - Use sampling with the `row_sample` parameter (in number of rows, 0 for all rows)
    - Specify the columns of interest with `features` paramters (list of columns, None for all)
    """

    if features is None:
        features = Xval.columns

    Xval, = _row_sample(row_sample, Xval)

    for feat in features:
        grid, pdp = _partial_dependence(model, Xval, feat, n_jobs=n_jobs)
        yield feat, pd.Series(pdp, grid)


def _partial_dependence(model, Xval, feature,
                        percentiles=(0.05, 0.95), grid_resolution=20, n_jobs=-1):
    X = Xval.copy()
    if X[feature].dtype == 'O':  # TODO: deal with NA
        grid = most_freq(X[feature], grid_resolution)
    else:
        grid = grid_from_X(X[feature], percentiles, grid_resolution)

    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    pdp = parallel(
        delayed(_replace_and_predict)(
            model, Xval, feature, value)
        for value in grid)
    return list(grid), pdp


def _replace_and_predict(model, Xval, feature, value):
    X = Xval.copy()
    X[feature] = value
    if _is_classication(model, X):
        return model.predict_proba(X)[:, 1].mean()
    else:
        return model.predict(X).mean()


def grid_from_X(x, percentiles=(0.05, 0.95), grid_resolution=100):
    """Generate a grid of points based on the ``percentiles of ``x``.
    """
    x = x[~x.isnull()]
    if len(percentiles) != 2:
        raise ValueError('percentile must be tuple of len 2')
    if not all(0. <= x <= 1. for x in percentiles):
        raise ValueError('percentile values must be in [0, 1]')

    uniques = np.unique(x)
    if uniques.shape[0] < grid_resolution:
        # feature has low resolution use unique vals
        return uniques
    else:
        emp_percentiles = mquantiles(x, prob=percentiles)
        # create axis based on percentiles and grid resolution
        return np.linspace(emp_percentiles[0],
                           emp_percentiles[1],
                           num=grid_resolution, endpoint=True)


def most_freq(x, k=10, min_freq=10):
    freq = x.value_counts()
    return freq[freq > min_freq].sort_values(ascending=False)[:k].index


def _row_sample(row_sample, *dfs):
    nrows = dfs[0].shape[0]
    assert all(d.shape[0] == nrows for d in dfs)
    if row_sample > 0 and nrows > row_sample:
        idx = np.random.randint(0, nrows, row_sample)
        return [d.iloc[idx] for d in dfs]
    else:
        return dfs
