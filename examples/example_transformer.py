import pandas as pd
import numpy as np

from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_X_y
from sklearn.linear_model import LinearRegression

from blocks import BaseTransformer, register_feature_names, output_pandas_dataframe


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


if __name__ == "__main__":
    
    # Seed
    np.random.seed(123)

    # Params
    length, paths = 20, 10

    # Data
    X_train = pd.DataFrame(np.random.normal(size=(length, paths)))
    y_train = pd.DataFrame(np.random.normal(size=(length, paths)))
    
    # Model
    model = EstimatorTransformer(LinearRegression())
    model.fit_transform(X_train, y_train)
    #            0         1         2  ...         7         8         9
    # 0   0.818850 -0.285340  1.148225  ... -0.048604  0.267721 -0.062422
    # 1   0.608831  0.180718 -0.547601  ... -0.082058 -1.248803  0.830978
    # 2  -1.453790 -0.486099  0.126714  ... -0.834990 -1.376769 -0.683481
    # 3  -1.455287  1.064817  0.181793  ...  0.093992  0.668834 -0.797081
    # 4  -0.195593  0.599702 -0.299884  ... -0.557779 -0.796464  0.301506
    # 5   0.970664  0.143538  0.204189  ... -0.014414 -0.414175  1.239389
    # 6  -0.817856 -0.657872  0.390037  ...  0.134894  0.777642  0.772993
    # 7  -0.829962 -1.017737  1.319455  ... -0.261809 -0.599947 -0.324221
    # 8   0.142467 -0.447537 -1.405364  ...  1.289830 -0.984811  0.895566
    # 9  -1.969365 -0.499164  0.417222  ...  0.332517 -0.340200  0.462234
    # 10  0.022387 -0.686029 -0.134274  ... -0.522257  0.306323  0.015881
    # 11  0.582035  0.348426 -0.680475  ...  0.201938  0.269188 -0.029778
    # 12 -0.201209  0.249758 -0.883989  ...  0.175911  0.333873  0.675057
    # 13  0.117551  1.388019  0.067231  ... -0.154581  0.231246  0.058665
    # 14 -0.156191  0.134812  0.440362  ... -0.369014  0.192424  1.065127
    # 15 -0.499416  0.101037 -0.737347  ...  0.483561 -1.610633 -0.974305
    # 16  0.114131 -0.449019 -0.936939  ... -0.173636 -0.166308  0.059910
    # 17 -0.658070 -0.211834  1.136115  ... -0.992536  0.399813 -0.052303
    # 18  0.598966 -0.586364 -0.315949  ...  0.206467 -0.309194  0.120099
    # 19  0.465532  0.388126 -0.971358  ...  0.366386  0.076330 -0.525496
    
    # [20 rows x 10 columns]
    