
import warnings

from itertools import combinations
from scipy.stats import mode

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

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


class MovingAverageCrossover(BaseTransformer):
    def __init__(self, windows=list, ma='sma'):
        self.windows = windows
        self.ma = ma.upper()
        super().__init__()

        # Validate window list
        if len(windows) != len(set(windows)):
            raise ValueError("All values in windows must be unique.")

        if any(not isinstance(i, int) or i <= 0 for i in windows):
            raise ValueError(
                "All values in windows must be positive integers.")

    @staticmethod
    def _compute_moving_average(X: pd.DataFrame, window: int, avg_type: str) -> pd.DataFrame:
        if avg_type == 'SMA':
            return X.rolling(window=window, min_periods=1).mean()

        elif avg_type == 'EMA':
            return X.ewm(span=window, adjust=False).mean()

        else:
            raise ValueError(
                "Unsupported moving average type. Use 'SMA' or 'EMA'."
            )

    def __call__(cls, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure windows are sorted to maintain short < long
        sorted_windows = sorted(cls.windows)

        results = []
        for short, long in combinations(sorted_windows, 2):
            sw = cls._compute_moving_average(X, short, cls.ma)
            lw = cls._compute_moving_average(X, long, cls.ma)
            labels = [f'{cls.ma}({col}, {short}, {long})' for col in X.columns]

            # Signals: Faster MA > Slower MA
            sig = np.where(sw > lw, 1.0, 0.0)
            result = pd.DataFrame(sig, index=X.index, columns=labels).diff()
            results.append(result.fillna(0))

        return pd.concat(results, axis=1)


class RSTradeEntry(BaseTransformer):
    def __init__(self, window=14, thresh: tuple = None):
        self.window = window
        self.thresh = thresh or (30, 70)
        super().__init__()

    def __call__(cls, X: pd.Series) -> pd.DataFrame:
        # Calculate RSI using pandas_ta
        rsi = ta.rsi(X, length=cls.window)

        # Initialize signals array
        signals = np.zeros_like(rsi)
        # Generate signals based on RSI thresholds
        signals[(rsi.shift(1) < cls.thresh[1]) & (rsi >= cls.thresh[1])] = -1
        signals[(rsi.shift(1) > cls.thresh[0]) & (rsi <= cls.thresh[0])] = 1

        # Adjust signals to ensure no consecutive same non-zero signals
        last_signal = 0
        for i in range(len(signals)):
            if signals[i] != 0:
                if signals[i] == last_signal:
                    signals[i] = 0
                else:
                    last_signal = signals[i]

        return pd.Series(signals, index=rsi.index, name=f'RSI({cls.window})')


class RSInterval(BaseTransformer):
    def __init__(self, window=14, thresh: tuple = None):
        self.window = window
        self.thresh = thresh or (30, 70)
        super().__init__()

    def __call__(cls, X: pd.Series) -> pd.DataFrame:
        # Calculate RSI using pandas_ta
        rsi = ta.rsi(X, length=cls.window)

        # Initialize signals array
        signals = np.zeros_like(rsi)

        # Generate signals based on RSI thresholds
        overbought = False
        oversold = False

        for i in range(1, len(rsi)):
            if oversold:
                if rsi[i] >= 50:
                    oversold = False
                signals[i] = 1
            elif overbought:
                if rsi[i] <= 50:
                    overbought = False
                signals[i] = -1
            else:
                if (
                    rsi[i-1] < cls.thresh[0] and
                    rsi[i] >= cls.thresh[0] and
                    rsi[i] <= 50
                ):
                    oversold = True
                    signals[i] = 1
                elif (
                    rsi[i-1] > cls.thresh[1] and
                    rsi[i] <= cls.thresh[1] and
                    rsi[i] >= 50
                ):
                    overbought = True
                    signals[i] = -1

        return pd.Series(signals, index=rsi.index, name=f'RSI({cls.window})')


class FilterCollinear(BaseTransformer):
    def __init__(
        self,
        target: str = None,
        subset: str | list = None,
        threshold: float = 5.0
    ):
        self.target = target
        self.subset = subset
        self.threshold = threshold
        super().__init__()

    def _compute_variance_inflation_factor(self, X: pd.DataFrame) -> pd.Series:
        # initialize dictionaries
        results = {}  # , tolerance_dict = {}, {}
        exogs = (
            X.columns
            if self.target is None
            else [col for col in X.columns if col != self.target]
        )
        # form input data for each exogenous variable
        for exog in exogs:
            endog = [i for i in exogs if i != exog]
            Xi, y = X[endog], X[exog]

            # extract r-squared from the fit
            r_squared = LinearRegression().fit(Xi, y).score(Xi, y)
            # calculate VIF and tolerance
            tol = 1 - r_squared
            results[exog] = 1 / tol
            # tolerance_dict[exog] = tol

        return pd.Series(results).sort_values(ascending=False)

    def _iter_filter(self, X, vif):
        while (vif > self.threshold).any() == True:
            removed = vif.drop(vif.index[0])  # Sorted
            X = X[removed.index]
            vif = self._compute_variance_inflation_factor(X)

        return X

    def __call__(cls, X: pd.DataFrame) -> pd.DataFrame:
        if cls.subset is not None:
            # Ensure subset is a list
            if isinstance(cls.subset, str):
                subset = [cls.subset]
            else:
                subset = cls.subset
            # Interpolate only the subset of columns
            Xi = X.copy()
            vif = cls._compute_variance_inflation_factor(X[subset])
            Xi = cls._iter_filter(X[subset], vif)
        else:
            # Interpolate all columns
            vif = cls._compute_variance_inflation_factor(X)
            Xi = cls._iter_filter(X, vif)

        return Xi


class LinearImputer(BaseTransformer):
    def __init__(self, subset: str | list = None, **kwargs):
        self.subset = subset
        self.kwargs = kwargs
        super().__init__()

    def __call__(cls, X: pd.DataFrame) -> pd.DataFrame:
        # Check if subset is provided
        if cls.subset is not None:
            # Ensure subset is a list
            if isinstance(cls.subset, str):
                subset = [cls.subset]
            else:
                subset = cls.subset

            # Interpolate only the subset of columns
            Xi = X.copy()
            Xi[subset] = X[subset].interpolate(**cls.kwargs)
        else:
            # Interpolate all columns
            Xi = X.interpolate(**cls.kwargs)

        # Return the DataFrame with interpolated values
        return Xi


class ForestImputer(BaseTransformer):
    def __init__(
        self,
        subset: str | list = None,
        max_iter=10,
        decreasing=False,
        missing_values=np.nan,
        copy=True,
        n_estimators=100,
        criterion=('squared_error', 'gini'),
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None
    ):
        self.subset = subset
        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        super().__init__()

    @staticmethod
    def _get_mask(X, value_to_mask):
        """Compute the boolean mask X == missing_values."""
        if value_to_mask == "NaN" or np.isnan(value_to_mask):
            return np.isnan(X)
        else:
            return X == value_to_mask

    def _miss_forest(self, Ximp, mask):
        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)

            # Reg criterion
            reg_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[0]

            # Instantiate regression model
            rf_regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=reg_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start)

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(
                col_modes, missing_cat_cols)

            # Classfication criterion
            clf_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[1]

            # Instantiate classification model
            rf_classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=clf_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                class_weight=self.class_weight)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])

        while (
                gamma_new < gamma_old or gamma_newcat < gamma_oldcat) and \
                self.iter_count_ < self.max_iter:

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    rf_classifier.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_classifier.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis
                else:
                    rf_regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                gamma_newcat = np.sum(
                    (Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_])
                ) / n_catmissing
            if self.num_vars_ is not None:
                gamma_new = np.sum(
                    (Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2
                ) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            print("Iteration:", self.iter_count_)
            self.iter_count_ += 1

        return Ximp_old

    def _apply_fit(self, X, y=None, cat_vars=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Check data integrity and calling arguments
        force_all_finite = (
            False
            if self.missing_values in ["NaN", np.nan]
            else True
        )
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite=force_all_finite,
            copy=self.copy
        )

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = self._get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Check cat_vars type and convert if necessary
        if cat_vars is not None:
            if type(cat_vars) == int:
                cat_vars = [cat_vars]
            elif type(cat_vars) == list or type(cat_vars) == np.ndarray:
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array of ints."
                    )
            else:
                raise ValueError(
                    "cat_vars needs to be either an int or an array of ints."
                )

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else None

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = (
            np.nanmean(X[:, num_vars], axis=0)
            if num_vars is not None
            else None
        )
        col_modes = (
            mode(X[:, cat_vars], axis=0, nan_policy='omit')[0]
            if cat_vars is not None
            else None
        )

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}

        return self

    def fit(self, X, y=None, cat_vars=None):
        """
        Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        cat_vars : int or array of ints, optional (default = None)
            An int or an array containing column indices of categorical
            variable(s)/feature(s) present in the dataset X.
            ``None`` if there are no categorical variables in the dataset.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check if subset is provided
        if self.subset is not None:
            # Ensure subset is a list
            if isinstance(self.subset, str):
                subset = [self.subset]
            else:
                subset = self.subset

            self._apply_fit(X[subset], y, cat_vars)

        else:
            self._apply_fit(X, y, cat_vars)

        return self

    def _apply_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Check data integrity
        force_all_finite = (
            False
            if self.missing_values in ["NaN", np.nan]
            else True
        )
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            force_all_finite=force_all_finite,
            copy=self.copy
        )

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = self._get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (
            (0 if self.num_vars_ is None else len(self.num_vars_)) +
            (0 if self.cat_vars_ is None else len(self.cat_vars_))
        )
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError(
                "Incompatible dimension between the fitted dataset and the one "
                "to be transformed.")

        # Check if anything is actually missing and if not return original X
        mask = self._get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn(
                "No missing value located; returning original dataset."
            )
            return X

        # Call cls function to impute missing
        X = self._miss_forest(X, mask)

        # Return imputed dataset
        return X

    def __call__(cls, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """
        # Confirm whether fit() has been called
        check_is_fitted(cls, ["cat_vars_", "num_vars_", "statistics_"])

        # Check if subset is provided
        if cls.subset is not None:
            # Ensure subset is a list
            if isinstance(cls.subset, str):
                subset = [cls.subset]
            else:
                subset = cls.subset

            Xi = X.copy()
            Xi[subset] = cls._apply_transform(X[subset])

        else:
            Xi = cls._apply_transform(X)

        return Xi

    def fit_transform(self, X, y=None, **fit_params):
        """Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X)
