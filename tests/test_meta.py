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
    pred = bk.VectorRegressor(LinearRegression).fit(X_train, y_train_).transform(y_test)
    
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
    