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
    

    model = VectorRegressor(LinearRegression)
    model.fit(y_train, X_train)
    pred = model.transform(y_test)
    #              asset_1   asset_2   asset_3  ...   asset_8   asset_9  asset_10
    # 2010-10-24  0.073575  0.058114 -0.033778  ... -0.033542  0.050167 -0.029552
    # 2010-10-25  0.071293  0.040653 -0.072565  ...  0.032462  0.056727 -0.037706
    # 2010-10-26 -0.010592  0.054215  0.016596  ...  0.031619  0.048295 -0.030476
    # 2010-10-27  0.055741  0.046580 -0.036209  ... -0.007525  0.034053 -0.034472
    # 2010-10-28  0.001309  0.092880  0.010326  ... -0.010217  0.029000 -0.047707
    #              ...       ...       ...  ...       ...       ...       ...
    # 2024-06-27 -0.075750  0.081202  0.049279  ...  0.066508 -0.000665  0.012373
    # 2024-06-28  0.024678  0.064660 -0.039270  ...  0.030297  0.080765 -0.051954
    # 2024-06-29  0.059261 -0.042952 -0.049490  ...  0.067667  0.007781 -0.008127
    # 2024-06-30  0.023214  0.051821 -0.013794  ...  0.056274  0.043157  0.001860
    # 2024-07-01 -0.008298  0.064166  0.066089  ... -0.021113  0.067569  0.035714
    
    # [5000 rows x 10 columns]
    
