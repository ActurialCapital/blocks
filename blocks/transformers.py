from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, asdict
from functools import wraps

import pandas as pd
import numpy as np

from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing import FunctionTransformer

from imblearn import FunctionSampler

AnyArray = pd.Series | pd.DataFrame | np.ndarray

def validate_select(transformer_dict: dict):
    """
    Decorator to validate the selected transformation against available 
    options.

    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if kwargs.get("select", False):
                select = kwargs.get("select")
            else:
                try:
                    select = args[0]
                except:
                    raise ValueError(
                        "Parameter `select` has not been provided. Choose from "
                        f"{list(transformer_dict.keys())}."
                    )
            # Check if select is in transformer_dict keys
            if select not in transformer_dict:
                raise ValueError(
                    "Invalid selection for operation. Choose from "
                    f"{list(transformer_dict.keys())}."
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def register_feature_names(func):
    """
    Decorator to register pandas feature names.
    """
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        return func(self, X, *args, **kwargs)

    return wrapper


def output_pandas_dataframe(func):
    """
    Decorator to register pandas feature names.
    """
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        output = func(self, X, *args, **kwargs)
        return pd.DataFrame(output, index=X.index, columns=self.columns_)

    return wrapper


@dataclass
class ParamSampler:
    accept_sparse: bool = False
    kw_args: Dict[str, Any] = None
    validate: bool = False


class BaseSampler(ABC, FunctionSampler):
    """
    Abstract base class for data transformation via `imblearn.FunctionSampler`.
    This class provides an interface for transforming data. Subclasses
    should implement the `transform` method to apply specific transformation
    steps to the data.

    """

    def __init__(self, params: Dict[str, Any] | ParamSampler = None):
        params = params or asdict(ParamSampler())
        super().__init__(func=self, **params)

    @abstractmethod
    def __call__(self, X: AnyArray, y: AnyArray = None) -> AnyArray:
        pass


@dataclass
class ParamTransformer:
    inverse_func: object = None
    validate: bool = False
    accept_sparse: bool = False
    check_inverse: bool = True
    feature_names_out = None
    kw_args: Dict[str, Any] = None
    inv_kw_args: Dict[str, Any] = None


class BaseTransformer(ABC, OneToOneFeatureMixin, FunctionTransformer):
    """
    Abstract base class for data transformation.
    This class provides an interface for transforming data. Subclasses
    should implement the `transform` method to apply specific transformation
    steps to the data.

    """

    def __init__(self, params: Dict[str, Any] | ParamTransformer = None):
        params = params or asdict(ParamTransformer())
        super().__init__(func=self, **params)

    def check_kwargs(self, selected: str, kw_args: str):
        if self.select == selected:
            key = self.kwargs.get(kw_args)
            if key is None:
                raise ValueError(f"Missing {kw_args} to compute {selected}.")

    @register_feature_names
    def fit(self, X, y, **kwargs) -> "BaseTransformer":
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
        self : BaseTransformer
            The fitted transformer.
        """
        return self

    @abstractmethod
    def __call__(self, X: AnyArray, y: AnyArray = None) -> AnyArray:
        pass
