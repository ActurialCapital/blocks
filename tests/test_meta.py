import pytest 

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

from pandas.testing import assert_frame_equal

import blocks as bk


length = 50
n_paths = 10


end_date = datetime.now().date()
start_date = end_date - timedelta(days=length - 1)
index = pd.date_range(start=start_date, end=end_date, freq="D")


assets = [f'asset_{n}' for n in range(1, n_paths + 1)]
factors = [f'factor_{n}' for n in range(1, n_paths + 1)]


X_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=factors,
    index=index
)

y_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=assets,
    index=index
)
y_train_ = y_train.copy()
y_train_.iloc[2:10, 0:3] = pd.NA

y_test = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=factors,
    index=index
)


def test_vector_regression():
    # Model based
    model = bk.VectorRegressor(LinearRegression)
    model.fit(X_train, y_train_)
    pred = model.transform(y_test)
    
    # Iterating through assets (vector by vector)
    predictions = []
    for asset in assets:
        Xi = y_train_[asset].dropna()
        yi = X_train.dropna()
        Xi, yi = Xi.align(yi, join='inner', axis=0)
        arr = LinearRegression().fit(yi, Xi).predict(y_test)
        predictions.append(pd.DataFrame(arr, columns=[asset], index=y_test.index))
    output = pd.concat(predictions, axis=1)
    
    # Assert
    assert_frame_equal(pred, output)
    
    # Test All NaNs
    X_train_nans = pd.DataFrame(index=X_train.index, columns=X_train.columns)
    with pytest.raises(ValueError):
        model.fit(X_train_nans, y_train_)
    
        
def test_estimator_transformer():
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(y_test)
    pred = pd.DataFrame(pred, index=y_test.index, columns=y_test.columns)

    model = bk.EstimatorTransformer(LinearRegression())
    model.fit(X_train, y_train)
    output = model.transform(y_test)
    
    # Assert
    assert_frame_equal(pred, output)
    
    # Test check_X_y
    model = bk.EstimatorTransformer(LinearRegression(), check_input=True)
    new_y_train = np.select([y_train > 0, y_train <= 0], [True, False], default=True)
    model.fit(X_train, new_y_train)
    output = model.transform(y_test)
    assert isinstance(output, pd.DataFrame)
    
    new_y_train = np.select([y_train > 0, y_train <= 0], ['foo', 'bar'], default='foo')
    with pytest.raises(ValueError):
        model.fit(X_train, new_y_train)
    
    
def test_factor():
    pass
    
    