import pandas as pd
import numpy as np

from blocks.base import BaseTransformer
from blocks.decorators import validate_select


class ColumnAverage(BaseTransformer):
    """
    Transformer class for calculating various types of means within a 
    DataFrame.

    This class allows for the calculation of different mean values across the
    DataFrame's axis, depending on the selected method. It supports simple, 
    weighted, expanding, rolling, exponential moving average (EMA), and grouped 
    mean calculations.

    Available options:

    * `simple`: Calculates the arithmetic mean across the DataFrame's axis.
    * `weighted`: Calculates a weighted mean, using weights specified in 
    `kwargs`.
    * `expanding`: Calculates the mean on an expanding window, including 
    all previous data points.
    * `rolling`: Calculates the mean using a fixed*size rolling window 
    specified in `kwargs`.
    * `ema`: Calculates the exponential moving average over the DataFrame, 
    with decay specified in `kwargs`.
    * `grouped`: Calculates the mean for each group specified by a key in 
    `kwargs`.

    Parameters
    ----------
    select : str, optional
        Specifies the type of mean calculation to perform. Defaults to 'simple'.
    name : str, optional
        Specifies the name of the output pandas Series. This can be used to 
        label the output for clarity. Defaults to 'Feature'.
    kwargs : dict
        Provides specific calculation type keyword arguments. This can include 
        weights for the 'weighted' mean, window size for the 'rolling' mean, 
        span for the 'ema', and group keys for the 'grouped' mean, among 
        others.

    """

    TRANSFORMERS = {
        'simple': lambda x: x.mean(axis=1),
        'weighted': lambda x, weights: (x * weights).mean(axis=1),
        'expanding': lambda x: x.expanding().mean(),
        'rolling': lambda x, window: x.rolling(window=window).mean(),
        'ema': lambda x, span: x.ewm(span=span).mean(),
        'grouped': lambda x, group_by: x.groupby(level=group_by, axis=1).mean()
    }

    @validate_select(TRANSFORMERS)
    def __init__(self, select: str, name: str = 'feature', **kwargs):
        self.select = select
        self.name = name
        self.kwargs = kwargs
        super().__init__()

    def __call__(cls, X: pd.Series | pd.DataFrame, y=None) -> pd.Series | pd.DataFrame:
        """
        Transforms the input DataFrame by calculating the specified type of 
        mean.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Data for which to calculate the mean.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing the calculated mean values.
        """
        operation = cls.TRANSFORMERS[cls.select]
        cls.check_kwargs("weighted", "weights")
        cls.check_kwargs("ema", "span")
        cls.check_kwargs("rolling", "window")
        cls.check_kwargs("grouped", "group_by")
        result = operation(X, **cls.kwargs)
        return result.rename(cls.name) if isinstance(result, pd.Series) else result


class RateOfChange(BaseTransformer):
    """
    Rate of Change (ROC) Transformer for financial time series data.
    This transformer calculates the rate of change of a DataFrame over a
    specified window period.

    Parameters
    ----------
    window : int, optional
        Periods to shift for forming percent change. Defaults to 1.

    """

    def __init__(self, window: int = 1):
        self.window = window
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.Series | pd.DataFrame:
        """
        Transforms the input DataFrame by calculating the rate of change.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Data containing the financial time series data.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing the rate of change values.
        """
        return X.pct_change(cls.window).where(X.notna(), np.nan)


class Rolling(BaseTransformer):
    """
    Transformer class for calculating rolling statistics of a DataFrame.
    This transformer computes various rolling statistics for each column in the 
    DataFrame over a specified window period. The operation to perform is 
    chosen through the `select` parameter.

    Available options:

    * `sum`: Rolling sum of values.
    * `mean`: Rolling mean of values.
    * `median`: Rolling median of values.
    * `var`: Rolling variance of values.
    * `std`: Rolling standard deviation of values.
    * `min`: Rolling minimum value.
    * `max`: Rolling maximum value.
    * `quantile`: Rolling quantile of values.
    * `custom`: Custom rolling function - It requires passing a function 
    through `kwargs`.
    * `corr`: Rolling correlation of values.
    * `cov`: Rolling covariance of values.
    * `skew`: Rolling skewness of values.
    * `kurt`: Rolling kurtosis of values.
    * `count`: Rolling count of non*NA values.
    * `rank`: Rolling rank of values.

    Parameters
    ----------
    select : str
        The type of rolling operation to perform. Defauts to "mean".
    window : int
        The size of the moving window. This is the number of observations used 
        for calculating the statistic. The window will be centered on each 
        observation unless specified otherwise in `kwargs`.
    rate_of_change : bool, optional
        If True, the transformer calculates the rate of change of the rolling 
        calculation, comparing the current value to the value at the start of 
        the window. Defaults to False.
    division_rolling : bool, optional
        If True, divides the DataFrame by its rolling calculation for the 
        selected operation over the specified window period, effectively 
        normalizing the data. Defaults to False.
    kwargs : dict
        Additional keyword arguments to pass to the rolling calculation. This 
        could include arguments like `min_periods`, `center`, `win_type` for 
        window type, and any custom parameter required by a `custom` function 
        passed in `select`.

    """

    TRANSFORMERS = {
        'sum': lambda x, window: x.rolling(window).sum(),
        'mean': lambda x, window: x.rolling(window).mean(),
        'median': lambda x, window: x.rolling(window).median(),
        'var': lambda x, window: x.rolling(window).var(),
        'std': lambda x, window, ann: x.rolling(window).std(ddof=1) * np.sqrt(ann),
        'min': lambda x, window: x.rolling(window).min(),
        'max': lambda x, window: x.rolling(window).max(),
        'quantile': lambda x, window, q: x.rolling(window).quantile(q),
        'custom': lambda x, window, func: x.rolling(window).apply(func, raw=False),
        'corr': lambda x, window, y: x.rolling(window).corr(y),
        'cov': lambda x, window, y: x.rolling(window).cov(y),
        'skew': lambda x, window: x.rolling(window).skew(),
        'kurt': lambda x, window: x.rolling(window).kurt(),
        'count': lambda x, window: x.rolling(window).count(),
        'rank': lambda x, window: x.rolling(window).rank(),
    }

    @validate_select(TRANSFORMERS)
    def __init__(
        self,
        select: str,
        window: int = 252,
        division_rolling: bool = False,
        **kwargs
    ):
        self.select = select
        self.window = window
        self.division_rolling = division_rolling
        self.kwargs = kwargs
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.Series | pd.DataFrame:
        """
        Transforms the input DataFrame by calculating the rolling standard 
        deviation.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Data containing the financial time series data.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing the rolling standard deviation values.
        """
        operation = cls.TRANSFORMERS[cls.select]
        cls.check_kwargs("std", "ann")
        cls.check_kwargs("custom", "func")
        cls.check_kwargs("quantile", "q")
        result = operation(X, cls.window, **cls.kwargs)
        if cls.division_rolling:
            return X / result

        return result


class Zscore(BaseTransformer):
    """
    Z-score Transformer for normalizing financial time series data.
    This transformer calculates the Z-score for each column in the DataFrame 
    over a specified window period.

    Parameters
    ----------
    window : int, optional
        The window size for calculating the rolling mean and standard 
        deviation. Defaults to 252.

    """

    def __init__(self, window: int = 252):
        self.window = window
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.Series | pd.DataFrame:
        """
        Transforms the input DataFrame by calculating the Z-score.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Data containing the financial time series data.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing the Z-score values.
        """
        return X.transform(
            lambda x: (x - x.rolling(cls.window).mean()) /
            x.rolling(cls.window).std(ddof=1)
        )


class QuantileRanks(BaseTransformer):
    """
    Transformer class to convert predictions into quantile-based signals.

    This class extends `BaseTransformer` to transform a DataFrame of 
    predictions into quantile-based signals, based on the specified number of 
    quantiles.

    Parameters
    ----------
    number_q : int, optional
        The number of quantiles to transform the data into. Defaults to 4.
    group_by : str or list, optional
        The column(s) to group by when calculating quantiles. If provided, 
        quantiles are computed within each group.

    """

    def __init__(self, number_q: int = 4, group_by: str | list = None):
        self.number_q = number_q
        self.group_by = group_by
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.Series | pd.DataFrame:
        """
        Transforms the predictions into quantile-based signals.
        This method applies a quantile transformation to the provided 
        DataFrame, dividing it into the specified number of quantiles. If 
        `group_by` is provided, the transformation is applied within each 
        group.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Data containing the financial time series data.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing the transformed quantile-based signals.
        """
        # Transpose
        transposed_data = X.T
        # Drop NaN
        clean_data = transposed_data.dropna(how='all', axis=1)
        # Group By functionality, if applicable
        if isinstance(cls.group_by, (list, str)):
            clean_data = clean_data.groupby(level=cls.group_by)
        # Transform to ranks
        ranks = clean_data.transform(
            lambda df: pd.qcut(
                df, cls.number_q, labels=False, duplicates='drop'
            )
        ).T
        return ranks


class Signal(BaseTransformer):
    """
    Transformer class to convert ranks into investment signals.

    Parameters
    ----------
    select : str, optional
        Select type of data to be computed. If `rank`, it generates a mapping 
        dictionary based on unique values in the DataFrame. if 'float', it 
        splits the into positive (1) and negative (-1) scores. Defaults to 
        `rank`.
    higher_is_better : bool, optional
        Determines the direction of the signal. If True, higher ranks
        lead to positive signals and vice versa. Defaults to True.
    fees : float, optional
        Adjusts float by considering the transaction cost. It might skip 
        generating a positive (negative) signal if the expected return is lower
        (greater) than the transaction cost. It is only applied if `select` is
        `float`. Defaults to 0.
    apply_thresholder : bool, optional
        Applies signal thresholds to determine when to enter or exit trades.
        This functionality will only change signals if the change is greater 
        than a specified threshold. Defaults to False.
    threshold : float
        The minimum change in absolute signal required to trigger a new 
        investment position.
    apply_smoother : bool, optional
        Apply smoothing signals with a rolling window transformation. Defaults
        to False.
    rolling : str, optional
        The type of rolling operation to perform. For more information. 
        please see `Rolling().ROLLLING_TRANSFORMERS`.
    **rolling_kwargs
        Specific rolling calculation type keyword argument(s).

    """

    TRANSFORMERS = {'rank': None, 'number': None}

    @validate_select(TRANSFORMERS)
    def __init__(
        self,
        select: str,
        higher_is_better: bool = True,
        fees: float = 0,
        apply_thresholder: bool = False,
        threshold: float = 0.1,
        apply_smoother: bool = False,
        rolling_kwargs: dict = None
    ):
        self.select = select
        self.higher_is_better = higher_is_better
        self.fees = fees
        self.apply_thresholder = apply_thresholder
        self.threshold = threshold
        self.apply_smoother = apply_smoother
        self.rolling_kwargs = rolling_kwargs or {}
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transforms the input DataFrame into investment signals.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Data containing the financial time series data.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data with transformed investment signals.

        """
        signals = (
            cls.ranking_model(X, cls.higher_is_better)
            if cls.select == 'rank'
            else cls.threshold_model(X, cls.fees)
        )

        if cls.apply_smoother:
            signals = cls.smoother(signals, cls.rolling_kwargs)

        if cls.apply_thresholder:
            signals = cls.thresholder(signals, cls.threshold)

        return signals

    @staticmethod
    def ranking_model(
        X: pd.Series | pd.DataFrame,
        higher_is_better: bool = True,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate signals from mapping dictionary based on unique values in the 
        DataFrame.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            The input pandas DataFrame containing data to be mapped.
        higher_is_better : bool, optional
            Score mapping. If True, values are -1 for the minimum value, 1 for 
            the maximum value. If False, values are 1 for the minimum valie, 
            -1 for the maximun. In both cases, 0 for all other values. Defaults
            to True.

        Returns
        -------
        pd.Series | pd.DataFrame
            A DataFrame with transformed investment signals.

        """
        columns = X.columns
        if isinstance(columns, pd.MultiIndex):
            # WARNING: If multiindex column, we focus on the last column name level
            level_name = columns.names[-1]
            X.columns = X.columns.get_level_values(level_name)
        # Get Unique Keys
        keys = sorted(set(X.stack()))
        lower, upper = (-1, 1) if higher_is_better else (1, -1)
        scores = {
            key: lower if key == min(keys)
            else upper if key == max(keys)
            else np.nan
            for key in keys
        }
        results = X.apply(lambda x: x.map(scores))
        results.columns = columns
        return results

    @staticmethod
    def threshold_model(
        X: pd.Series | pd.DataFrame,
        fees: float = 0,
    ) -> pd.Series | pd.DataFrame:
        """
        Transforms the input DataFrame into investment signals.

        Parameters
        ----------
        df : pd.Series | pd.DataFrame
            Data containing the financial time series data.
        fees : float, optional
            Adjusts float by considering the transaction cost. It might skip 
            generating a positive (negative) signal if the expected return is 
            lower (greater) than the transaction cost. It is only applied if 
            `select` is `float`. Defaults to 0.

        Returns
        -------
        pd.DataFrame
            A DataFrame with transformed investment signals.

        """
        X = X - fees
        condition = [X > 0, X <= 0]
        choices = [1, -1]
        reshaped = np.select(condition, choices, default=np.nan)
        return pd.DataFrame(reshaped, index=X.index, columns=X.columns)

    @staticmethod
    def smoother(
        signals: pd.Series | pd.DataFrame,
        rolling_kwargs: dict = None
    ) -> pd.Series | pd.DataFrame:
        """
        Apply smoothing rules to signals.

        Parameters
        ----------
        signals : pd.Series | pd.DataFrame
            Data containing the financial time series signals.

        Returns
        -------
       pd.Series | pd.DataFrame
            Data with transformed smoothing signals.

        """
        # Remove NaNs
        signals = signals.fillna(0)
        # Smoothed signals
        model = Rolling(**rolling_kwargs)
        smoothed = model.transform(signals)
        # Create signals
        condition = [smoothed > 0, smoothed <= 0]
        choices = [1, -1]
        reshaped = np.select(condition, choices, default=np.nan)
        return pd.DataFrame(
            reshaped,
            index=smoothed.index,
            columns=smoothed.columns
        )

    @staticmethod
    def thresholder(
        signals: pd.Series | pd.DataFrame,
        threshold: float = 0.1,
    ) -> pd.Series | pd.DataFrame:
        """
        Apply thresholding rules to signals.

        Parameters
        ----------
        signals : pd.Series | pd.DataFrame
            Data containing the financial time series signals.

        Returns
        -------
        pd.Series | pd.DataFrame
            Data with transformed thresholding signals.

        """
        return signals.diff().abs().ge(threshold).astype(int) * np.sign(signals)
