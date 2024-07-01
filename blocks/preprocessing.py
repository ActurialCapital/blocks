import typing as tp

import pandas as pd
import numpy as np

from blocks.base import BaseSampler
from blocks.decorators import validate_select


class ShiftFactor(BaseSampler):
    """
    Class for applying shifting to `X` while `y` remains the same.

    Parameters
    ----------
    shift_by : int, optional
        Shift index by desired number of periods. The number of periods by 
        which to shift `y`. Default is 1.

    """

    def __init__(self, shift_by: int = 1):
        self.shift_by = shift_by
        super().__init__()

    def __call__(
        cls,
        X: pd.Series | pd.DataFrame,
        y: pd.Series | pd.DataFrame
    ) -> tp.Tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
        """
        Preprocesses the data by shifting `X` while `y` remains constant.
        This method shifts the `X` data by a specified number of periods
        and aligns it with the `X` data. This is commonly used in 
        time-series forecasting where the goal is to predict future values 
        based on past observations.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            The data to shift values.
        y : pd.Series | pd.DataFrame
            The data to align with shifted values.

        Returns
        -------
        tp.Tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]
            X: The shifted factor data.
            y: The aligned training data.
        """
        # Apply transformations and return the transformed datasets
        X = X.shift(cls.shift_by).iloc[cls.shift_by:]
        y = y.iloc[cls.shift_by:]
        return X, y


class BasisOperation(BaseSampler):
    """
    Transformer class for executing basic elementary arithmetic operations 
    between two DataFrames.

    Performs element-wise addition, subtraction, multiplication, or division 
    between two DataFrames. Additionally, this class supports computing lagged 
    versions of the resulting DataFrame by shifting the data by a specified 
    number of periods before performing the operation.

    Available options:

    * 'add': Performs element-wise addition (X + y).
    * 'diff': Calculates the element-wise difference (X * y).
    * 'product': Multiplies the elements of the DataFrames (X * y).
    * 'ratio': Divides the elements of the primary DataFrame by the 
    secondary DataFrame (X / y).

    Attributes
    ----------
    select : str
        Specifies the type of arithmetic operation to perform on the input 
        DataFrames. Each operation is applied on a per-cell basis between the 
        two DataFrames.
    shift_by : int, optional
        Specifies the number of periods to shift the primary DataFrame (`X`) 
        by before performing the selected operation. This parameter allows for
        the computation of lagged operations, which can be useful for time 
        series analysis. A shift of 0 applies the operation directly without 
        lag. Defaults to 1.

    Methods
    -------
    transform
        Calculates the selected operation between X and y.
    """

    TRANSFORMERS = {
        'sum': lambda x, y: x + y,
        'diff': lambda x, y: x - y,
        'product': lambda x, y: x * y,
        'ratio': lambda x, y: x / y
    }

    @validate_select(TRANSFORMERS)
    def __init__(
        self,
        select: str,
        shift_by: int = 0,
    ):
        self.select = select
        self.shift_by = shift_by
        super().__init__()

    def __call__(
        cls,
        X: pd.Series | pd.DataFrame,
        y: pd.Series | pd.DataFrame
    ) -> tp.Tuple[pd.Series | pd.DataFrame, None]:
        """
        Transforms the input DataFrames by calculating the selected basic
        operation.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Primary DataFrame.
        y : pd.Series | pd.DataFrame
            Secondary DataFrame

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing the result of the selected operation between 
            X and y.
        """
        X, y = X.align(y, join='inner')
        X = X.shift(cls.shift_by).iloc[cls.shift_by:]
        y = y.iloc[cls.shift_by:]
        operation = cls.TRANSFORMERS[cls.select]
        return operation(X, y), None


class Filter(BaseSampler):
    """
    A class for applying various filters to raw data based on the specified 
    threshold, frequency, and aggregation method. This preprocessing step 
    allows for data cleaning and conditioning before further analysis or 
    modeling.

    Parameters
    ----------
    source: pd.Series | pd.DataFrame
        Data source to filter with.
    sign : str
        The comparison method to use for filtering. Supported methods include 
        '>=', '<=', and '=='. This parameter dictates how data points are 
        compared to the threshold value to determine if they should be 
        filtered.
    thresh : float
        The threshold value for the filter. Data points will be compared to 
        this value using the specified method.
    freq : str
        The frequency at which to resample the data. This parameter is passed 
        directly to pandas' `resample` method and
        should be a string representing a valid offset alias 
        (e.g., 'D' for daily).
    agg : str
        The aggregation method to use when resampling. This parameter is passed 
        directly to pandas' `agg` method and determines how data points are 
        aggregated within each resampling period.

    Raises
    ------
    ValueError
        If an unsupported filtering method is specified.
    """

    FILTER_TRANSFORMERS = {
        '>=': lambda x, thresh: x >= thresh,
        '<=': lambda x, thresh: x <= thresh,
        '==': lambda x, thresh: x == thresh
    }

    def __init__(
        self,
        source: pd.Series | pd.DataFrame,
        sign: str = '>=',
        thresh: float = 0,
        freq: str = 'D',
        agg: str = 'last'
    ):
        self.source = source
        self.sign = sign
        self.thresh = thresh
        self.freq = freq
        self.agg = agg
        super().__init__()

    def __call__(
        cls,
        X: pd.Series | pd.DataFrame,
        y: pd.Series | pd.DataFrame
    ) -> tp.Tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
        """
        Transform raw data using filters.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            The data to apply filter.
        y : pd.Series | pd.DataFrame
            The data to filter by.

        Returns
        -------
        pd.Series | pd.DataFrame
            The filter Data.

        """
        # Resample source data
        source = cls.source.resample(cls.freq).agg(cls.agg).ffill()
        aligned = source.loc[y.index, y.columns]
        # Mask data
        operation = cls.FILTER_TRANSFORMERS[cls.sign]
        mask_data = operation(aligned, cls.thresh)
        # Apply mask
        return X, y.where(mask_data, np.nan)
