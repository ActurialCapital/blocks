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
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(666, 10) time=0s
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(667, 10) time=0s
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(667, 10) time=0s
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(1000, 10) time=0s
    
    predicted = pipe.transform(df)
    
    pipe.name_record
    'scaler'
   
    pipe.record
    # array([[ 1.29991924,  0.53833915, -0.96691497, ..., -2.13589292,
    #         -0.1470757 , -0.49519353],
    #        [-0.10391147,  0.54363887, -1.47146186, ..., -1.28665738,
    #          1.27469839, -2.75380778],
    #        [-0.03563881, -0.23822331, -1.03076527, ...,  1.43736601,
    #          0.77444089, -0.30153937],
    #        ...,
    #        [ 0.22356613,  0.3766578 ,  0.16055413, ..., -0.79878902,
    #         -3.66274703,  1.44552405],
    #        [ 1.73026657, -0.77120787,  0.28806519, ..., -1.29184535,
    #          1.32950426, -0.23281487],
    #        [ 0.1930198 ,  0.3759506 ,  0.68242895, ..., -1.62110848,
    #          0.72334675,  2.72615764]])
