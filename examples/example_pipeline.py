import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA 
from sklearn.pipeline import FeatureUnion

try:
    from sklego.preprocessing import ColumnSelector
except:
    raise FileNotFoundError(
        "Please install scikit-lego for run this example: "
        "`pip install scikit-lego`"
    )

from blocks import BlockPipeline, custom_log_callback


if __name__ == "__main__":
    
    # Seed
    np.random.seed(123)

    # Params
    length, paths = 20, 10

    # Data
    n_samples, n_features = 3, 5
    X = np.zeros((n_samples, n_features))
    y = np.arange(n_samples)
  
    # Model
    pipe = BlockPipeline([
        ("my_models", FeatureUnion([
            ("path1", BlockPipeline([
                ("select1", ColumnSelector([0, 1, 2, 3, 4])),
                ("pca", PCA(n_components=3)),
            ])),
            ("path2", BlockPipeline([
                ("select2", ColumnSelector([5, 6, 7, 8, 9])),
                ("pca", PCA(n_components=2)),
            ]))
        ])),
        ("linreg", LinearRegression())
    ], 
        record_from='my_models',
        log_callback=custom_log_callback
    )
    
    pipe.fit(X, y=y)
    
    