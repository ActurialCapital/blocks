import pytest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal

from sklearn.linear_model import LinearRegression

from blocks.decorators import ( 
    validate_select, 
    register_feature_names, 
    output_pandas_dataframe
)

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
df3 = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=assets,
    index=index,
)

class MyClass:
    CHECK_SELECT = {'a': 'foo', 'b': 'bar'}

    @validate_select(CHECK_SELECT)
    def __init__(self, select: str):
        self.select = select

    @register_feature_names
    def fit(self, X, y):
        self.model = LinearRegression().fit(X, y)
        return self

    @output_pandas_dataframe
    def predict(self, X, y=None):
        return self.model.predict(X)

   


def test_validate_select():
    # Test valid selections
    MyClass('a')
    MyClass('b')
    with pytest.raises(TypeError):
        MyClass()
    with pytest.raises(TypeError):
        MyClass('hello')
    with pytest.raises(TypeError):
        MyClass(123)
    with pytest.raises(TypeError):
        MyClass(3.14)
    with pytest.raises(TypeError):
        MyClass(True)
    with pytest.raises(TypeError):
        MyClass(['a'])
    with pytest.raises(TypeError):
        MyClass({'a': 1})
    with pytest.raises(TypeError):
        MyClass(None)


def test_additional_valid_options():
    # Adding more valid options dynamically
    MyClass.CHECK_SELECT['d'] = lambda x: x
    MyClass.CHECK_SELECT['e'] = lambda x: x
    # Ensure they are valid
    MyClass('d')
    MyClass('e')


def test_register_feature_names():
    transformer = MyClass('a').fit(df1, df2)
    assert_index_equal(transformer.columns_, df1.columns)

    transformer = MyClass('a').fit(df2, df1)
    assert_index_equal(transformer.columns_, df2.columns)

def test_output_pandas_dataframe():
    arr = df3.to_numpy()
    assert isinstance(arr, np.ndarray)
    
    output = LinearRegression().fit(df1, df2).predict(df3)
    assert isinstance(output, np.ndarray)
    
    myclass = MyClass('a').fit(df1, df2)
    pred = myclass.predict(df3)
    assert isinstance(pred, pd.DataFrame)
    assert_index_equal(myclass.columns_, df1.columns)
    
    
    