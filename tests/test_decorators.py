import pytest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal

from blocks.decorators import validate_select, register_feature_names, output_pandas_dataframe
from blocks.base import BaseTransformer

length = 50
n_paths = 10


end_date = datetime.now().date()
start_date = end_date - timedelta(days=length - 1)
index = pd.date_range(start=start_date, end=end_date, freq="D")


assets = [f'asset_{n}' for n in range(1, n_paths + 1)]
factors = [
    ['group1', 'group2', 'group3', 'group4', 'group5'] * 2,
    [f'factor_{n}' for n in range(1, n_paths + 1)]
]
multi_index = pd.MultiIndex.from_arrays(factors, names=('Group', 'Factor'))

df1 = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=multi_index,
    index=index,
)
df2 = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=assets,
    index=index,
)


class MyClass(BaseTransformer):
    CHECK_SELECT = {'a': 'foo', 'b': 'bar'}

    @validate_select(CHECK_SELECT)
    def __init__(self, select: str):
        self.select = select

    @register_feature_names
    def fit(self, X, y=None):
        return self

    @output_pandas_dataframe
    def __call__(cls, X, y=None):
        return X


def test_validate_select():
    # Test valid selections
    MyClass('a')
    MyClass('b')
    with pytest.raises(TypeError):
        MyClass('hello')
        MyClass(123)
        MyClass(3.14)
        MyClass(True)
        MyClass(['a'])
        MyClass({'a': 1})
        MyClass(None)


def test_additional_valid_options():
    # Adding more valid options dynamically
    MyClass.CHECK_SELECT['d'] = lambda x: x
    MyClass.CHECK_SELECT['e'] = lambda x: x
    # Ensure they are valid
    MyClass('d')
    MyClass('e')


def test_register_feature_names():
    transformer = MyClass('a').fit(df1)
    assert_index_equal(transformer.columns_, df1.columns)

    transformer = MyClass('a').fit(df2)
    assert_index_equal(transformer.columns_, df2.columns)
