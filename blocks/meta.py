import pandas as pd

from sklearn.base import TransformerMixin, MetaEstimatorMixin, BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_X_y

from blocks.base import (
    BaseTransformer,
    BaseSampler,
    BaseFactor,
    BaseDataLoader
)
from blocks.decorators import register_feature_names, output_pandas_dataframe
from blocks.transformers import RateOfChange
from blocks.pipeline import BlockPipeline, make_block_pipeline


class EstimatorTransformer(BaseTransformer):
    """
    Allow using an estimator as a transformer in an earlier step of a pipeline.
    This wrapper is the `EstimatorTransformer` from 
    [sklearn-lego](https://koaning.github.io/scikit-lego/), in which we added 
    a preprocessing functionnality.

    !!! warning

        By default all the checks on the inputs `X` and `y` are delegated to 
        the wrapped estimator. To change such behaviour, set `check_input` 
        to `True`.

    Parameters
    ----------
    estimator : scikit-learn compatible estimator
        The estimator to be applied to the data, used as transformer.
    predict_func : str, optional
        The method called on the estimator when transforming e.g. 
        (`"predict"`, `"predict_proba"`). Default to "predict".
    check_input : bool, 
        Whether or not to check the input data. If False, the checks are 
        delegated to the wrapped estimator. Default to False.
    preprocessors : BasePreprocessor | List[BasePreprocessor]. optional
        Data preprocessing, which involves both `X` and `y` and could not be
        a transformer. Defaults to None.

    Attributes
    ----------
    estimator_ : scikit-learn compatible estimator
        The fitted underlying estimator.
    multi_output_ : bool
        Whether or not the estimator is multi output.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        predict_func="predict",
        check_input=False
    ):
        self.estimator = estimator
        self.predict_func = predict_func
        self.check_input = check_input
        super().__init__()

    @register_feature_names
    def fit(self, X, y, **kwargs) -> "EstimatorTransformer":
        """
        Fit the underlying estimator on training data `X` and `y`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional keyword arguments passed to the `fit` method of the 
            underlying estimator.

        Returns
        -------
        self : WrapEstimator
            The fitted transformer.
        """
        if self.check_input:
            X, y = check_X_y(
                X, y, estimator=self, dtype=FLOAT_DTYPES, multi_output=True
            )
        self.multi_output_ = len(y.shape) > 1
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **kwargs)
        return self

    @output_pandas_dataframe
    def __call__(cls, X: pd.DataFrame, y=None):
        X = X.loc[:, cls.columns_]  # Added to match preprocessed data
        check_is_fitted(cls, "estimator_")
        output = getattr(cls.estimator_, cls.predict_func)(X)
        return output if cls.multi_output_ else output.reshape(-1, 1)


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
        X = X.dropna()
        if X.empty:
            raise ValueError(
                'Variable `X` should not be empty after dropping NaNs.'
            )
        self.models = {}
        for label in y.columns:
            yi = y[label].dropna()
            if not yi.empty:
                yi, Xi = yi.align(X, join='inner', axis=0)
                fitted_model = self.model_cls(**self.kwargs).fit(Xi, yi)
                self.models[label] = fitted_model

        return self

    def __call__(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        predictions = []
        for label, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pd.DataFrame(
                pred, columns=[label], index=X.index))

        return pd.concat(predictions, axis=1)

