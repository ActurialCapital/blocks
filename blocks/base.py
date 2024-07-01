from abc import ABC, abstractmethod
import typing as tp
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from imblearn import FunctionSampler

from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing import FunctionTransformer

from blocks.decorators import register_feature_names


AnyArray = pd.Series | pd.DataFrame | np.ndarray


@dataclass
class ParamSampler:
    accept_sparse: bool = False
    kw_args: tp.Dict[str, tp.Any] = None
    validate: bool = False


class BaseSampler(ABC, FunctionSampler):
    """
    Abstract base class for data transformation via `imblearn.FunctionSampler`.
    This class provides an interface for transforming data. Subclasses
    should implement the `transform` method to apply specific transformation
    steps to the data.

    """

    def __init__(self, params: tp.Dict[str, tp.Any] | ParamSampler = None):
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
    kw_args: tp.Dict[str, tp.Any] = None
    inv_kw_args: tp.Dict[str, tp.Any] = None


class BaseTransformer(ABC, OneToOneFeatureMixin, FunctionTransformer):
    """
    Abstract base class for data transformation.
    This class provides an interface for transforming data. Subclasses
    should implement the `transform` method to apply specific transformation
    steps to the data.

    """

    def __init__(self, params: tp.Dict[str, tp.Any] | ParamTransformer = None):
        params = params or asdict(ParamTransformer())
        super().__init__(func=self, **params)

    def check_kwargs(self, selected: str, kw_args: str):
        if self.select == selected:
            key = self.kwargs.get(kw_args)
            if key is None:
                raise ValueError(f"Missing {kw_args} to compute {selected}.")

    @register_feature_names
    def fit(self, X, y=None, **kwargs) -> "BaseTransformer":
        """
        Fit the underlying estimator on training data `X` and `y`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values. Defaults to None.
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


@dataclass
class BaseFactor:
    """Configuration for factor building."""
    tags: list
    name: str
    X: str
    y: str
    market_feature: str
    inputs: dict
    outputs: dict
    pipeline: tuple
    preprocess: tuple = None

    def __post_init__(self):
        # Overriding the default __repr__ method to show ClassName()
        self.__class__.__repr__ = lambda self: f"{self.__class__.__name__}"


class BaseDataLoader:
    pass
