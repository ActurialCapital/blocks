import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pandas.testing import assert_frame_equal, assert_series_equal

import blocks as bk


length = 50
n_paths = 10


end_date = datetime.now().date()
start_date = end_date - timedelta(days=length - 1)
index = pd.date_range(start=start_date, end=end_date, freq="D")


assets = [f'asset_{n}' for n in range(1, n_paths + 1)]
factors = [
    ['group1', 'group2', 'group3', 'group4', 'group5'] * 2,
    [f'factor_{n}' for n in range(1, n_paths + 1)]
]
multi_index = pd.MultiIndex.from_arrays(factors, names=('Group', 'Factor'))

X_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=multi_index,
    index=index,
)

y_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=assets,
    index=index
)
y_train_ = y_train.copy()
y_train_.iloc[2:10, 0:3] = pd.NA
X_test = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=multi_index,
    index=index
)
y_test = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=assets,
    index=index
)


@pytest.mark.parametrize("select", [select for select in bk.ColumnAverage.TRANSFORMERS.keys()])
def test_column_average(select):
    if select == 'weighted':
        kwargs = dict(weights=np.random.normal(size=n_paths))
    elif select == 'ema':
        kwargs = dict(span=2)
    elif select == 'rolling':
        kwargs = dict(window=3)
    elif select == 'grouped':
        kwargs = dict(group_by='Group')
    else:
        kwargs = {}

    pred = bk.ColumnAverage(select, **kwargs).fit_transform(X_train)

    operation = bk.ColumnAverage.TRANSFORMERS[select]
    result = operation(X_train, **kwargs)
    output = result.rename('feature') if isinstance(
        result, pd.Series) else result

    if isinstance(pred, pd.DataFrame):
        assert_frame_equal(pred, output)

    elif isinstance(pred, pd.Series):
        assert_series_equal(pred, output)

    else:
        raise ValueError(
            'pred is neither a pandas Series nor DataFrame object')


@pytest.mark.parametrize("window", [window for window in range(1, 5)])
def test_roc(window):
    pred = bk.RateOfChange(window).fit_transform(y_train_)

    output = y_train_.pct_change(window).where(y_train_.notna(), np.nan)

    assert_frame_equal(pred, output)


@pytest.mark.parametrize("select", [select for select in bk.Rolling.TRANSFORMERS.keys()])
@pytest.mark.parametrize("window", [window for window in range(2, 5)])
@pytest.mark.parametrize("division_rolling", [True, False])
def test_rolling(select, window, division_rolling):
    if select == 'quantile':
        kwargs = dict(q=0.75)
    elif select == 'custom':
        kwargs = dict(func=np.mean)
    elif select == 'std':
        kwargs = dict(ann=252)
    elif select in ['cov', 'corr']:
        kwargs = dict(y=y_test)
    else:
        kwargs = {}

    pred = bk.Rolling(
        select,
        window,
        division_rolling,
        **kwargs
    ).fit_transform(y_train)

    operation = bk.Rolling.TRANSFORMERS[select]
    output = operation(y_train, window, **kwargs)
    if division_rolling:
        output = y_train / output

    assert_frame_equal(pred, output)


@pytest.mark.parametrize("window", [window for window in range(2, 5)])
def test_zscore(window):
    pred = bk.Zscore(window).fit_transform(X_train)

    output = X_train.transform(
        lambda x: (x - x.rolling(window).mean()) /
        x.rolling(window).std(ddof=1)
    )

    assert_frame_equal(pred, output)


@pytest.mark.parametrize("number_q", [1, 2, 3, 4])
@pytest.mark.parametrize("group_by", [None, 'Group'])
def test_quantile_ranks(number_q, group_by):
    pred = bk.QuantileRanks(number_q, group_by).fit_transform(X_train)

    transposed_data = X_train.T
    # Drop NaN
    clean_data = transposed_data.dropna(how='all', axis=1)
    # Group By functionality, if applicable
    if isinstance(group_by, (list, str)):
        clean_data = clean_data.groupby(level=group_by)
    # Transform to ranks
    ranks = clean_data.transform(
        lambda df: pd.qcut(
            df, number_q, labels=False, duplicates='drop'
        )
    ).T
    output = ranks

    assert_frame_equal(pred, output)

def apply_signal_from_rank(X, select, higher_is_better, fees):
    columns = X.columns
    if isinstance(columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values('Factor')
    # Get Unique Keys
    keys = sorted(set(X.stack()))
    lower, upper = (-1, 1) if higher_is_better else (1, -1)
    scores = {
        key: lower if key == min(keys)
        else upper if key == max(keys)
        else np.nan
        for key in keys
    }
    output = X.apply(lambda x: x.map(scores))
    output.columns = columns
    return output

def apply_signal_from_number(X, select, higher_is_better, fees):
    X = X - fees
    condition = [X > 0, X <= 0]
    choices = [1, -1]
    reshaped = np.select(condition, choices, default=np.nan)
    output = pd.DataFrame(reshaped, index=X.index, columns=X.columns)

    return output


@pytest.mark.parametrize("select", [select for select in bk.Signal.TRANSFORMERS.keys()])
@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize("fees", [0, 0.05])
def test_signal(select, higher_is_better, fees):
    X = X_train.copy()
    if select == 'rank':
        X = X.rank(axis=1)
        output = apply_signal_from_rank(X.copy(), select, higher_is_better, fees)
    else:
        output = apply_signal_from_number(X.copy(), select, higher_is_better, fees)
        
    # Transformer Model
    pred = bk.Signal(select, higher_is_better, fees).fit_transform(X)

    assert_frame_equal(pred, output)


@pytest.mark.parametrize("select", [select for select in bk.Signal.TRANSFORMERS.keys()])
@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize("fees", [0, 0.05])
@pytest.mark.parametrize("threshold", [0.01, -.02])
def test_signal_thresholder(select, higher_is_better, fees, threshold):
    X = X_train.copy()
    if select == 'rank':
        X = X.rank(axis=1)
        signals = apply_signal_from_rank(X.copy(), select, higher_is_better, fees)
    else:
        signals = apply_signal_from_number(X.copy(), select, higher_is_better, fees)
        
    output = signals.diff().abs().ge(threshold).astype(int) * np.sign(signals)
    
    # Transformer Model
    pred = bk.Signal(select, higher_is_better, fees, apply_thresholder=True, threshold=threshold).fit_transform(X)

    assert_frame_equal(pred, output)


@pytest.mark.parametrize("select", [select for select in bk.Signal.TRANSFORMERS.keys()])
@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize("fees", [0, 0.05])
def test_signal_smoother(select, higher_is_better, fees):
    X = X_train.copy()
    if select == 'rank':
        X = X.rank(axis=1)
        signals = apply_signal_from_rank(X.copy(), select, higher_is_better, fees)
    else:
        signals = apply_signal_from_number(X.copy(), select, higher_is_better, fees)
        
    rolling_kwargs = dict(select='mean', window=3, division_rolling=False)
    
    # Remove NaNs
    signals = signals.fillna(0)
    # Smoothed signals
    model = bk.Rolling(**rolling_kwargs)
    smoothed = model.transform(signals)
    # Create signals
    condition = [smoothed > 0, smoothed <= 0]
    choices = [1, -1]
    reshaped = np.select(condition, choices, default=np.nan)
    output = pd.DataFrame(
        reshaped,
        index=smoothed.index,
        columns=smoothed.columns
    )
    
    # Transformer Model
    pred = bk.Signal(select, higher_is_better, fees, apply_smoother=True, rolling_kwargs=rolling_kwargs).fit_transform(X)

    assert_frame_equal(pred, output)
