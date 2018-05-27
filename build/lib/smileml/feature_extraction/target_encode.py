import pandas as pd
import collections
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn import model_selection

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
        Target Encoder for mean
    """
    def __init__(self, columns = None, suffix = '_mean'):
        self.columns = columns
        self.suffix = suffix
    
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        # Default to using all the categorical columns
        columns = [col for col in X.columns if X.dtypes[col] in ('object', 'category')]\
            if self.columns is None\
            else self.columns
        
        X = pd.concat((X[columns], y.rename('y')), axis='columns')
        self.posteriors_ = {}
        for col in self.columns:
            gp = X.groupby(col)['y']
            self.posteriors_[col] = gp.mean()            
        return self
    
    def transform(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for col in self.columns:
            posteriors = self.posteriors_[col]
            X[col + self.suffix] = X[col].map(posteriors)
        return X

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, n_splits=5, shuffle=True, random_state=None, suffix='_mean'):
        self.columns = columns
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.suffix = suffix

    def fit(self, X, y):
        self.k_fold_ = model_selection.KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        return self

    def transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        data = pd.concat((X, y), axis='columns')
        y_col = data.columns[-1]
        means = {col: pd.Series() for col in self.columns}

        for fit_idx, val_idx in self.k_fold_.split(data):

            fit, val = data.iloc[fit_idx], data.iloc[val_idx]

            for col in self.columns:

                col_means = fit.groupby(col)[y_col].mean()

                means[col] = pd.concat(
                    (
                        means[col],
                        val[[col]].join(col_means, on=col)[y_col].fillna(col_means.mean())
                    ),
                    axis='rows'
                )

        for col in means:
            X[col + self.suffix] = means[col]

        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
    
class SmoothTargetEncoder(BaseEstimator, TransformerMixin):

    """
    Reference: https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators
    Args:
        columns (list of strs): Columns to encode.
        weighting (int or dict): Value(s) used to weight each prior.
        suffix (str): Suffix used for naming the newly created variables.
    """

    def __init__(self, columns=None, prior_weight=100, suffix='_mean'):
        self.columns = columns
        self.prior_weight = prior_weight
        self.suffix = suffix

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        # Default to using all the categorical columns
        columns = [col for col in X.columns if X.dtypes[col] in ('object', 'category')]\
            if self.columns is None\
            else self.columns

        # Compute prior and posterior probabilities for each feature
        X = pd.concat((X[columns], y.rename('y')), axis='columns')
        self.prior_ = y.mean()
        self.posteriors_ = {}
        for col in columns:
            agg = X.groupby(col)['y'].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            pw = self.prior_weight
            self.posteriors_[col] = collections.defaultdict(
                lambda: self.prior_,
                ((pw * self.prior_ + counts * means) / (pw + counts)).to_dict()
            )

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for col in self.columns:
            posteriors = self.posteriors_[col]
            X[col + self.suffix] = X[col].map(posteriors)

        return X