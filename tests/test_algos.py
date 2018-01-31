from smileml.ml import RandomBitFeatures, ContinuableXGBClassifier
from smileml.ml.preprocessing import simple_proc_for_linear_algoritms
from smileml.utils import DummyImport
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pytest
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

path = os.path.join(dir_path, 'data/titanic.csv')
df = pd.read_csv(path)
CAT = ['Sex', 'Embarked']
NUM = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']


def test_randombits():
    pipe = make_pipeline(
        simple_proc_for_linear_algoritms(NUM, CAT),
        RandomBitFeatures(),
        LogisticRegression()
    )
    scores = cross_val_score(pipe, df, df.Survived, cv=2, scoring='roc_auc')
    assert scores.mean() > 0.8


@pytest.mark.skipif(not isinstance(ContinuableXGBClassifier, DummyImport),
                    reason="when xgboost is not installed")
def test_xgb_not_installed():
    try:
        ContinuableXGBClassifier()
        assert False
    except ImportError as e:
        assert str(e) == 'Cannot import XGBoost'


@pytest.mark.skipif(isinstance(ContinuableXGBClassifier, DummyImport),
                    reason="requires xgboost")
def test_xgb():
    # assert isinstance(ContinuableXGBClassifier, DummyImport)
    pipe = make_pipeline(
        simple_proc_for_linear_algoritms(NUM, CAT),
        ContinuableXGBClassifier()
    )
    scores = cross_val_score(pipe, df, df.Survived, cv=2, scoring='roc_auc')
    assert scores.mean() > 0.8
