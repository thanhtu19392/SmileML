import readline  # noqa: F401,E731 # needed, otherwise rpy2 won't work.

# See https://github.com/ContinuumIO/anaconda-issues/issues/152
from .utils import optional_import
from .pipeline import TolerantLabelEncoder, ColumnsSelector, ColumnApplier, FillNaN
from sklearn.pipeline import make_union, make_pipeline
from sklearn,preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
importr = optional_import('importr', 'rpy2.robjects.packages')
pandas2ri = optional_import('rpy2.robjects.pandas2ri')
robj = optional_import('robjects', 'rpy2')


def compute_ace(df, numeric_features, categoric_features, target, target_as_cat):
    """
    Compute the ACE correlation (http://www.stat.cmu.edu/~ryantibs/datamining/lectures/11-cor2-marked.pdf)
    between each of the numeric/categoric features vs the target.
    The `target` variable can be treated either as numeric or as categoric using `target_as_cat` parameter.
    """

    pandas2ri.activate()
    acepack = importr('acepack')
    rstats = importr('stats')

    df = _prepare(df, numeric_features, categoric_features)

    # Ordinal encoded categorics + numerics
    transformer = _ordinal_transformer(numeric_features, categoric_features)
    orddf = transformer.fit_transform(df)

    cat_indexes = list(range(1, 1 + len(categoric_features)))
    if target_as_cat:
        cat_indexes += [0]
    a = acepack.ace(orddf, df[target], cat=robj.IntVector(cat_indexes))
    acescores = rstats.cor(a.rx('tx')[0], a.rx('ty')[0])
    ace = pd.DataFrame({
        "Feature": categoric_features + numeric_features,
        "Ace": acescores
    })
    ace.Ace = ace.Ace.abs()
    ace = ace.sort_values(by="Ace", ascending=False)
    return ace


def compute_pairwise_ace(df, numeric_features, categoric_features):
    pandas2ri.activate()

    path = os.path.dirname(os.path.abspath(__file__))
    robj.r('source')(os.path.join(path, 'ace.r'))

    df = _prepare(df, numeric_features, categoric_features)
    df, r_num_names, r_cat_names = \
        _rename(df, numeric_features, categoric_features)

    corr = robj.globalenv['compute_pairwise_ace'](
        df, robj.StrVector(r_num_names), robj.StrVector(r_cat_names))

    corr = pd.DataFrame(np.array(corr))
    corr.columns = (categoric_features + numeric_features)

    pandas2ri.deactivate()

    return corr


def _ordinal_transformer(numeric_features, categoric_features):
    cat_pipe = make_pipeline(
        ColumnsSelector(categoric_features),
        FillNaN('missing'),
        ColumnApplier(TolerantLabelEncoder())
    )
    num_pipe = make_pipeline(
        ColumnsSelector(numeric_features),
        FillNaN(-999),
    )
    if categoric_features and numeric_features:
        return make_union(cat_pipe, num_pipe)
    elif categoric_features:
        return cat_pipe
    return num_pipe


def _prepare(df, numeric_features, categoric_features):
    '''
    Transform all boolean columns to float (R does not like boolean)
    and all categoric columns to string (LabelEncoder does not like boolean in mixed type)
    '''

    df = df.copy()
    df[numeric_features] = df[numeric_features].astype(float)
    df = df.fillna(-777)
    for c in categoric_features:
        df[c] = df[c].astype(str)
    return df


def _rename(df, numeric_features, categoric_features):
    """
    Standalize the columns' names, as R does not like space, quotes, etc in the names either
    """

    df = df[categoric_features + numeric_features]
    r_num_names = ['cont%s' % i for i, _ in enumerate(numeric_features)]
    r_cat_names = ['cat%s' % i for i, _ in enumerate(categoric_features)]
    df.columns = r_cat_names + r_num_names
    return df, r_num_names, r_cat_names
