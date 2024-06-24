import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

from blocks import VectorRegressor


if __name__ == "__main__":

    length = 5000
    n_paths = 10
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=length - 1)
    
    X_train = pd.DataFrame(
        np.random.normal(size=(length, n_paths)),
        columns=[f'asset_{n}' for n in range(1, n_paths + 1)],
        index=pd.date_range(start=start_date, end=end_date, freq="D")
    )

    y_train = pd.DataFrame(
        np.random.normal(size=(length, n_paths)),
        columns=[f'factor_{n}' for n in range(1, n_paths + 1)],
        index=pd.date_range(start=start_date, end=end_date, freq="D")
    )
    
    y_test = pd.DataFrame(
        np.random.normal(size=(length, n_paths)),
        columns=[f'factor_{n}' for n in range(1, n_paths + 1)],
        index=pd.date_range(start=start_date, end=end_date, freq="D")
    )
    

    self = VectorRegressor(LinearRegression)
    self.fit(X_train, y_train)
    pred = self.transform(y_test)
    
