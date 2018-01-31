import numpy as np


class RandomBitFeatures:
    """
    Implemtation of Random Bits Features Generation
    https://arxiv.org/pdf/1501.02990.pdf

    This method generates a large number (10^4-10^6) of random binary
    intermediate/derived features based on the original input matrix.
    According to the authors, it works well with `LogisticRegression` and
    `LinearRegression` using `L-BFGS` solver.

    All the features are generated independently. In order to economize
    transient memory, we do it by chunk. So each time, we generate only
    `per_chunk` features, then concatenate them.

    Parameters
    ----------
    `n_meta` : Number of binary features to generate
    `per_chunk` : Number of features to generate per chunk

    Examples
    --------
    .. code-block:: python

          pipe = make_pipeline(
              one_hot_categoric_and_imputer_numeric,
              StandardScaler(),
              RandomBitFeatures(),
              LogisticRegression(solver='lbfgs')
          )
          pipe.fit(Xtrain, Ytrain)

    """

    def __init__(self, n_meta=10000, per_chunk=5000):
        self.n_meta = n_meta
        self._chunk = per_chunk

    def fit(self, X, y=None):
        params = []
        n = self.n_meta
        while n > 0:
            w, fidx, z = self._partial_fit(X, min(n, self._chunk))
            params.append((w, fidx, z))
            n -= self._chunk
        self.params = params
        return self

    def _partial_fit(self, X, n_meta):
        n_sample, n_cols = X.shape
        weights = np.random.normal(size=(3, n_meta))
        featidx = np.random.randint(0, n_cols, (3, n_meta))
        weights3d = np.repeat(weights[np.newaxis, :, :], n_sample, axis=0)
        weighted_features = np.sum((X[:, featidx] * weights3d), axis=1)
        zidx = np.random.randint(0, n_sample, n_meta)
        z = weighted_features[zidx, np.arange(len(zidx))]
        return weights, featidx, z

    def transform(self, X):
        ret = []
        for W, idx, z in self.params:
            ret.append(self._partial_transform(X, W, idx, z))
        return np.concatenate(ret, axis=1)

    def _partial_transform(self, X, W, idx, z):
        weights3d = np.repeat(W[np.newaxis, :, :], X.shape[0], axis=0)
        weighted_features = np.sum((X[:, idx] * weights3d), axis=1)
        return (weighted_features > np.repeat(z[np.newaxis, :], X.shape[0], axis=0))
