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


class Factor(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, template: BaseFactor):
        self.template = template

    def fit(self, market_data: BaseDataLoader, y=None) -> "Factor":
        # is market based
        if self.template.market_feature:
            # Get market data through `market_feature` param
            df = market_data.get(self.template.market_feature)
            # Preprocess data with inputs param
            processed_data = self.process_single(df, **self.template.inputs)
            # Calculate market returns percentage change
            self.market_returns_ = RateOfChange().transform(processed_data)
        else:
            self.market_returns_ = None
        return self

    def transform(self, factor_data: BaseDataLoader) -> pd.Series:
        # Extract raw factor data
        raws = tuple(
            factor_data.get(label).dropna(how='all', axis=1)
            for label in (self.template.X, self.template.y)
            if label is not None and factor_data.get(label) is not None
        )

        # Preprocess raw factor data
        preprocessed = tuple(
            self.process_single(raw, **self.template.inputs)
            for raw in raws
        )
        if self.template.preprocess:
            preprocessed = self._build_pipeline(
                self.template.preprocess, *preprocessed)
            if isinstance(preprocessed, tuple):
                preprocessed = preprocessed[0],

        # Main transformation
        final = self._build_pipeline(
            self.template.pipeline, *preprocessed, self.market_returns_)

        # Postprocess
        if isinstance(self.template.outputs, dict):
            final = self.process_single(final, **self.template.outputs)

        return (
            final.rename(self.template.name).to_frame()
            if isinstance(final, pd.Series)
            else final.rename(
                columns={col: self.template.name for col in final.columns}
            )
            if isinstance(final, pd.DataFrame)
            else final
        )

    def _build_pipeline(self, steps, *data):
        pipe = make_block_pipeline(*steps)
        func = self._final_estimator_func(pipe)
        return getattr(pipe, func)(*data)

    @staticmethod
    def _final_estimator_func(
        transformer: BaseSampler | BaseTransformer | BlockPipeline
    ) -> str:
        if hasattr(transformer, "fit_resample"):
            return "fit_resample"

        elif hasattr(transformer, "fit_transform"):
            return "fit_transform"

        else:
            raise AttributeError(
                f"'{transformer}' object has no attribute 'fit_resample' "
                "nor 'fit_transform'."
            )

    @classmethod
    def process_single(cls, df: pd.DataFrame, **inputs_outputs) -> pd.DataFrame:
        """Process input or output data."""

        resample = inputs_outputs.pop("resample", False)
        interpolate = inputs_outputs.pop("interpolate", False)
        transformers = inputs_outputs.pop("transform", False)

        if inputs_outputs:
            raise TypeError(
                'Unknown keyword arguments: "{}"'.format(
                    list(inputs_outputs.keys())[0])
            )

        if resample:
            if resample.get('freq') and resample.get('agg'):
                df = df.resample(resample.get('freq')).agg(resample.get('agg'))
            else:
                raise ValueError(
                    'Resample parameter(s) "freq" or/and "agg" is(are) missing.'
                )

        if interpolate:
            if interpolate.get('method') and interpolate.get('limit_area'):
                df = df.interpolate(**interpolate)
            else:
                raise ValueError(
                    'Interpolate parameter(s) "method" or/and "limit_area" '
                    'is(are) missing.'
                )

        if transformers:
            df = cls._build_pipeline(transformers, *df)

        return df
