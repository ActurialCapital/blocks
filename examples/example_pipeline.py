import pandas as pd

from sklearn.datasets import make_regression

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

try:
    from sklego.meta import EstimatorTransformer
except:
    raise FileNotFoundError(
        "Please install scikit-lego for run this example: "
        "`pip install scikit-lego`"
    )

from blocks import BlockPipeline, custom_log_callback


if __name__ == "__main__":

    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    df = pd.DataFrame(X)
    
    pipe = BlockPipeline([
        ("scaler", StandardScaler()),
        ("regression", EstimatorTransformer(LinearRegression()))
    ],
        record='scaler',
        log_callback=custom_log_callback
    )

    pipe.fit(df, y)
    # [custom_log_callback:84] - [scaler][StandardScaler()] shape=(1000, 10) time=0s
    
    predicted = pipe.transform(df)
    
    pipe.name_record
    'scaler'
   
    pipe.record.dtype
    # dtype('float64')