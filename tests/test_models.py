import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

from pandas.testing import assert_frame_equal

from blocks import VectorRegressor


length = 50
n_paths = 10


end_date = datetime.now().date()
start_date = end_date - timedelta(days=length - 1)
index = pd.date_range(start=start_date, end=end_date, freq="D")


assets = [f'asset_{n}' for n in range(1, n_paths + 1)]
factors = [f'factor_{n}' for n in range(1, n_paths + 1)]


X_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=assets,
    index=index
)
X_train.iloc[2:10, 0:3] = pd.NA

y_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=factors,
    index=index
)

y_test = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=factors,
    index=index
)


def test_vector_regression():
    # Model based
    pred = VectorRegressor(LinearRegression).fit(X_train, y_train).transform(y_test)
    
    # Iterating through assets (vector by vector)
    predictions = []
    for asset in assets:
        Xi = X_train[asset].dropna()
        yi = y_train.dropna()
        Xi, yi = Xi.align(yi, join='inner', axis=0)
        arr = LinearRegression().fit(yi, Xi).predict(y_test)
        predictions.append(pd.DataFrame(arr, columns=[asset], index=y_test.index))
    output = pd.concat(predictions, axis=1)
    
    # Assert
    assert_frame_equal(pred, output)
        