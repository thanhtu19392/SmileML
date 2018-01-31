
from smileml.pipeline import ColumnsSelector


def test_main():
    a = ColumnsSelector(['A'])
    b = a.fit(None, None)
    assert a == b
