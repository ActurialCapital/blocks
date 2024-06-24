import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

from blocks.transformers import (
    BaseTransformer,
    register_feature_names,
    output_pandas_dataframe
)


class VectorRegressor(BaseTransformer):
    """
    Vector regression estimator.

    Unlike the general implementations provided by `sklearn`, the 
    `VectorRegression` estimator is univariate, operating on a vector-by-vector 
    basis. This feature is particularly beneficial when performing 
    `LinearRegression`. Additionally, unlike sklearn's `LinearRegression`, 
    `VectorRegression` can handle missing values encoded as NaN natively.

    Notes
    -----
    For supervised learning, you might want to consider 
    `HistGradientBoostingRegressor` which accept missing values encoded as 
    `NaNs` natively. 
    Alternatively, it is possible to preprocess the data, for instance by using 
    an imputer transformer in a pipeline or drop samples with missing values. 
    See [Imputation](https://scikit-learn.org/stable/modules/impute.html) 
    Finally, You can find a list of all estimators that handle `NaN` values at 
    the following [page](https://scikit-learn.org/stable/modules/impute.html).

    Parameters
    ----------
    model_cls : BaseEstimator, optional
        `sklearn` Regression model. If None, defaults to `LinearRegression`.
    kwargs
        Model key-words arguments

    """

    def __init__(self, model_cls: BaseEstimator = None, **kwargs):
        self.model_cls = model_cls or LinearRegression
        self.kwargs = kwargs
        super().__init__()

    @register_feature_names
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> "BaseTransformer":
        """
        Fit the underlying estimator on training data `X` and `y`.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.DataFrame
            Target values.
        **kwargs : dict
            Additional keyword arguments passed to the `fit` method of the 
            underlying estimator.

        Returns
        -------
        self : BaseTransformer
            The fitted transformer.
        """
        self.models = {}
        for label in X.columns:
            Xi = X[label].dropna()
            yi = y.dropna()
            Xi, yi = Xi.align(yi, join='inner', axis=0)
            fitted_model = self.model_cls(**self.kwargs).fit(yi, Xi)
            self.models[label] = fitted_model

        return self

    @output_pandas_dataframe
    def __call__(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        predictions = []
        for label, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pd.DataFrame(pred, columns=[label], index=X.index))

        return pd.concat(predictions, axis=1)
