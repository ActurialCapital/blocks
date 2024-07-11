import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from datetime import datetime, timedelta
from scipy.stats import mode

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_raise_message
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

df1 = pd.DataFrame({
    'Asset1': [100, 102, 101, 99, 98, np.nan, 90, 107, 109, 98],
    'Asset2': [100, 99, 88, 109, 122, 100, 90, 107, np.nan, np.nan]
},
    index=pd.date_range(start='2023-01-01', periods=10, freq='D'),
)

df2 = pd.DataFrame(
    {'a': [1, 1, 2, 3, 4, 1],
     'b': [2, 2, 3, 2, 1, 3],
     'c': [4, 6, 7, 8, 9, 5],
     'd': [4, 3, 4, 5, 4, 6],
     'e': [8, 8, 14, 15, 17, 20]}
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
        output = apply_signal_from_rank(
            X.copy(), select, higher_is_better, fees)
    else:
        output = apply_signal_from_number(
            X.copy(), select, higher_is_better, fees)

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
        signals = apply_signal_from_rank(
            X.copy(), select, higher_is_better, fees)
    else:
        signals = apply_signal_from_number(
            X.copy(), select, higher_is_better, fees)

    output = signals.diff().abs().ge(threshold).astype(int) * np.sign(signals)

    # Transformer Model
    pred = bk.Signal(select, higher_is_better, fees,
                     apply_thresholder=True, threshold=threshold).fit_transform(X)

    assert_frame_equal(pred, output)


@pytest.mark.parametrize("select", [select for select in bk.Signal.TRANSFORMERS.keys()])
@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize("fees", [0, 0.05])
def test_signal_smoother(select, higher_is_better, fees):
    X = X_train.copy()
    if select == 'rank':
        X = X.rank(axis=1)
        signals = apply_signal_from_rank(
            X.copy(), select, higher_is_better, fees)
    else:
        signals = apply_signal_from_number(
            X.copy(), select, higher_is_better, fees)

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
    pred = bk.Signal(select, higher_is_better, fees, apply_smoother=True,
                     rolling_kwargs=rolling_kwargs).fit_transform(X)

    assert_frame_equal(pred, output)


def get_data(
    n_rows=20,
    n_cols=5,
    missingness=0.2,
    min_val=0,
    max_val=10,
    missing_values=np.nan,
    rand_seed=1337
):
    rand_gen = np.random.RandomState(seed=rand_seed)
    X = rand_gen.randint(
        min_val,
        max_val,
        n_rows * n_cols
    ).reshape(
        n_rows,
        n_cols
    ).astype(float)

    # Introduce NaNs if missingness > 0
    if missingness > 0:
        # If missingness >= 1 then use it as approximate (see below) count
        if missingness >= 1:
            n_missing = missingness
        else:
            # If missingness is between (0, 1] then use it as approximate %
            # of total cells that are NaNs
            n_missing = int(np.ceil(missingness * n_rows * n_cols))

        # Generate row, col index pairs and introduce NaNs
        # NOTE: Below does not account for repeated index pairs so NaN
        # count/percentage might be less than specified in function call
        nan_row_idx = rand_gen.randint(0, n_rows, n_missing)
        nan_col_idx = rand_gen.randint(0, n_cols, n_missing)
        X[nan_row_idx, nan_col_idx] = missing_values

    return X


def test_linear_imputer():
    model = bk.LinearImputer()
    interpolated = model.transform(df1)

    output = df1.interpolate()

    pd.testing.assert_frame_equal(interpolated, output)


@pytest.mark.parametrize("subset", ['Asset1', 'Asset2'])
def test_linear_imputer_with_kwargs(subset):
    model = bk.LinearImputer(subset=subset, method="pad", limit_area="inside")
    interpolated = model.transform(df1)

    output = df1.copy()
    output[subset] = df1[subset].interpolate(method="pad", limit_area="inside")

    pd.testing.assert_frame_equal(interpolated, output)


def test_filter_multicollinear():
    model = bk.FilterCollinear(threshold=5.0)
    filtered = model.transform(df2)
    output = df2.loc[:, ['d', 'c', 'b']]
    pd.testing.assert_frame_equal(filtered, output)


def test_filter_multicollinear_with_subset():
    model = bk.FilterCollinear(subset=['a', 'b', 'c'], threshold=5.0)
    filtered = model.transform(df2)
    output = df2.loc[:, ['c', 'b']]
    pd.testing.assert_frame_equal(filtered, output)


def test_filter_multicollinear_with_target():
    model = bk.FilterCollinear(target='a', threshold=5.0)
    filtered = model.transform(df2)
    output = df2.loc[:, ['d', 'c', 'b']]
    pd.testing.assert_frame_equal(filtered, output)

    # Previously not removed
    model = bk.FilterCollinear(target='d', threshold=5.0)
    filtered = model.transform(df2)
    output = df2.loc[:, ['c', 'b', 'e']]
    pd.testing.assert_frame_equal(filtered, output)


def test_forest_imputer_imputation_shape():
    # Verify the shapes of the imputed matrix
    n_rows = 10
    n_cols = 2
    X = get_data(n_rows, n_cols)
    imputer = bk.ForestImputer()
    X_imputed = imputer.fit_transform(X)
    assert_array_equal(X_imputed.shape, (n_rows, n_cols))


def test_forest_imputer_zero():
    # Test imputation when missing_values == 0
    missing_values = 0
    imputer = bk.ForestImputer(missing_values=missing_values, random_state=0)

    # Test with missing_values=0 when NaN present
    X = get_data(min_val=0)
    msg = "Input contains NaN."
    assert_raise_message(ValueError, msg, imputer.fit, X)

    # Test with all zeroes in a column
    X = np.array([
        [1, 0, 0, 0, 5],
        [2, 1, 0, 2, 3],
        [3, 2, 0, 0, 0],
        [4, 6, 0, 5, 13],
    ])
    msg = "One or more columns have all rows missing."
    assert_raise_message(ValueError, msg, imputer.fit, X)


def test_forest_imputer_zero_part2():
    # Test with an imputable matrix and compare with missing_values="NaN"
    X_zero = get_data(min_val=1, missing_values=0)
    X_nan = get_data(min_val=1, missing_values=np.nan)
    statistics_mean = np.nanmean(X_nan, axis=0)

    imputer_zero = bk.ForestImputer(missing_values=0, random_state=1337)
    imputer_nan = bk.ForestImputer(missing_values=np.nan, random_state=1337)

    assert_array_equal(
        imputer_zero.fit_transform(X_zero),
        imputer_nan.fit_transform(X_nan)
    )
    assert_array_equal(
        imputer_zero.statistics_.get("col_means"),
        statistics_mean
    )


def test_forest_imputer_numerical_single():
    # Test imputation with default parameter values

    # Test with a single missing value
    df = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      2],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(df, axis=0)

    y = df[:, 0]
    X = df[:, 1:]
    good_rows = np.where(~np.isnan(y))[0]
    bad_rows = np.where(np.isnan(y))[0]

    rf = RandomForestRegressor(n_estimators=10, random_state=1337)
    rf.fit(X=X[good_rows], y=y[good_rows])
    pred_val = rf.predict(X[bad_rows])[0]

    df_imputed = np.array([
        [1,         0,      0,      1],
        [2,         1,      2,      2],
        [3,         2,      3,      2],
        [pred_val,  4,      5,      5],
        [6,         7,      6,      7],
        [8,         8,      8,      8],
        [16,        15,     18,    19],
    ])

    imputer = bk.ForestImputer(n_estimators=10, random_state=1337)

    assert np.allclose(imputer.fit_transform(df), df_imputed, atol=0.3)
    assert_array_equal(imputer.statistics_.get('col_means'), statistics_mean)


def test_forest_imputer_numerical_multiple():
    # Test with two missing values for multiple iterations
    df = np.array([
        [1,      0,      np.nan, 1],
        [2,      1,      2,      2],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(df, axis=0)
    n_rows, n_cols = df.shape

    # Fit missforest and transform
    imputer = bk.ForestImputer(random_state=1337)
    df_imp1 = imputer.fit_transform(df)

    # Get iterations used by missforest above
    max_iter = imputer.iter_count_

    # Get NaN mask
    nan_mask = np.isnan(df)
    nan_rows, nan_cols = np.where(nan_mask)

    # Make initial guess for missing values
    df_imp2 = df.copy()
    df_imp2[nan_rows, nan_cols] = np.take(statistics_mean, nan_cols)

    # Loop for max_iter count over the columns with NaNs
    for _ in range(max_iter):
        for c in nan_cols:
            # Identify all other columns (i.e. predictors)
            not_c = np.setdiff1d(np.arange(n_cols), c)
            # Identify rows with NaN and those without in 'c'
            y = df_imp2[:, c]
            X = df_imp2[:, not_c]
            good_rows = np.where(~nan_mask[:, c])[0]
            bad_rows = np.where(nan_mask[:, c])[0]

            # Fit model and predict
            rf = RandomForestRegressor(n_estimators=100, random_state=1337)
            rf.fit(X=X[good_rows], y=y[good_rows])
            pred_val = rf.predict(X[bad_rows])

            # Fill in values
            df_imp2[bad_rows, c] = pred_val

    assert np.allclose(df_imp1, df_imp2, atol=0.12)
    assert_array_equal(imputer.statistics_.get('col_means'), statistics_mean)


def test_forest_imputer_categorical_single():
    # Test imputation with default parameter values

    # Test with a single missing value
    df = np.array([
        [0,      0,      0,      1],
        [0,      1,      2,      2],
        [0,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [1,      7,      6,      7],
        [1,      8,      8,      8],
        [1,     15,     18,     19],
    ])

    y = df[:, 0]
    X = df[:, 1:]
    good_rows = np.where(~np.isnan(y))[0]
    bad_rows = np.where(np.isnan(y))[0]

    rf = RandomForestClassifier(n_estimators=10, random_state=1337)
    rf.fit(X=X[good_rows], y=y[good_rows])
    pred_val = rf.predict(X[bad_rows])[0]

    df_imputed = np.array([
        [0,         0,      0,      1],
        [0,         1,      2,      2],
        [0,         2,      3,      2],
        [pred_val,  4,      5,      5],
        [1,         7,      6,      7],
        [1,         8,      8,      8],
        [1,         15,     18,     19],
    ])

    imputer = bk.ForestImputer(n_estimators=10, random_state=1337)
    assert_array_equal(imputer.fit_transform(df, cat_vars=0), df_imputed)
    assert_array_equal(imputer.fit_transform(df, cat_vars=[0]), df_imputed)


def test_forest_imputer_categorical_multiple():
    # Test with two missing values for multiple iterations
    df = np.array([
        [0,      0,      np.nan, 1],
        [0,      1,      1,      2],
        [0,      2,      1,      2],
        [np.nan, 4,      1,      5],
        [1,      7,      0,      7],
        [1,      8,      0,      8],
        [1,     15,      0,     19],
        [1,     18,      0,     17],
    ])
    cat_vars = [0, 2]
    statistics_mode = mode(df, axis=0, nan_policy='omit').mode
    n_rows, n_cols = df.shape

    # Fit missforest and transform
    imputer = bk.ForestImputer(random_state=1337)
    df_imp1 = imputer.fit_transform(df, cat_vars=cat_vars)

    # Get iterations used by missforest above
    max_iter = imputer.iter_count_

    # Get NaN mask
    nan_mask = np.isnan(df)
    nan_rows, nan_cols = np.where(nan_mask)

    # Make initial guess for missing values
    df_imp2 = df.copy()
    df_imp2[nan_rows, nan_cols] = np.take(statistics_mode, nan_cols)

    # Loop for max_iter count over the columns with NaNs
    for _ in range(max_iter):
        for c in nan_cols:
            # Identify all other columns (i.e. predictors)
            not_c = np.setdiff1d(np.arange(n_cols), c)
            # Identify rows with NaN and those without in 'c'
            y = df_imp2[:, c]
            X = df_imp2[:, not_c]
            good_rows = np.where(~nan_mask[:, c])[0]
            bad_rows = np.where(nan_mask[:, c])[0]

            # Fit model and predict
            rf = RandomForestClassifier(n_estimators=100, random_state=1337)
            rf.fit(X=X[good_rows], y=y[good_rows])
            pred_val = rf.predict(X[bad_rows])

            # Fill in values
            df_imp2[bad_rows, c] = pred_val

    assert_array_equal(df_imp1, df_imp2)
    assert_array_equal(imputer.statistics_.get('col_modes')[0],
                       statistics_mode[cat_vars][0])


def test_forest_imputer_mixed_multiple():
    # Test with mixed data type
    df = np.array([
        [np.nan, 0,      0,      1],
        [0,      1,      2,      2],
        [0,      2,      3,      2],
        [1,      4,      5,      5],
        [1,      7,      6,      7],
        [1,      8,      8,      8],
        [1,     15,     18,      np.nan],
    ])

    n_rows, n_cols = df.shape
    cat_vars = [0]
    num_vars = np.setdiff1d(range(n_cols), cat_vars)
    statistics_mode = mode(df, axis=0, nan_policy='omit').mode
    statistics_mean = np.nanmean(df, axis=0)

    # Fit missforest and transform
    imputer = bk.ForestImputer(random_state=1337)
    df_imp1 = imputer.fit_transform(df, cat_vars=cat_vars)

    # Get iterations used by missforest above
    max_iter = imputer.iter_count_

    # Get NaN mask
    nan_mask = np.isnan(df)
    nan_rows, nan_cols = np.where(nan_mask)

    # Make initial guess for missing values
    df_imp2 = df.copy()
    df_imp2[0, 0] = statistics_mode[0]
    df_imp2[6, 3] = statistics_mean[3]

    # Loop for max_iter count over the columns with NaNs
    for _ in range(max_iter):
        for c in nan_cols:
            # Identify all other columns (i.e. predictors)
            not_c = np.setdiff1d(np.arange(n_cols), c)
            # Identify rows with NaN and those without in 'c'
            y = df_imp2[:, c]
            X = df_imp2[:, not_c]
            good_rows = np.where(~nan_mask[:, c])[0]
            bad_rows = np.where(nan_mask[:, c])[0]

            # Fit model and predict
            if c in cat_vars:
                rf = RandomForestClassifier(n_estimators=100,
                                            random_state=1337)
            else:
                rf = RandomForestRegressor(n_estimators=100,
                                           random_state=1337)
            rf.fit(X=X[good_rows], y=y[good_rows])
            pred_val = rf.predict(X[bad_rows])

            # Fill in values
            df_imp2[bad_rows, c] = pred_val

    assert_array_equal(df_imp1, df_imp2)
    assert_array_equal(imputer.statistics_.get('col_means'),
                       statistics_mean[num_vars])
    assert_array_equal(imputer.statistics_.get('col_modes')[0],
                       statistics_mode[cat_vars])


def test_statstics_fit_transform():
    # Test statistics_ when data in fit() and transform() are different
    X = np.array([
        [1,      0,      0,      1],
        [2,      1,      2,      2],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [8,      8,      8,      8],
        [16,     15,     18,    19],
    ])
    statistics_mean = np.nanmean(X, axis=0)

    Y = np.array([
        [0,      0,      0,      0],
        [2,      2,      2,      1],
        [3,      2,      3,      2],
        [np.nan, 4,      5,      5],
        [6,      7,      6,      7],
        [9,      9,      8,      8],
        [16,     15,     18,    19],
    ])

    imputer = bk.ForestImputer()
    imputer.fit(X).transform(Y)
    assert_array_equal(imputer.statistics_.get('col_means'), statistics_mean)


def test_default_with_invalid_input():
    # Test imputation with default values and invalid input

    # Test with all rows missing in a column
    X = np.array([
        [np.nan,    0,      0,      1],
        [np.nan,    1,      2,      np.nan],
        [np.nan,    2,      3,      np.nan],
        [np.nan,    4,      5,      5],
    ])
    imputer = bk.ForestImputer(random_state=1337)
    msg = "One or more columns have all rows missing."
    assert_raise_message(ValueError, msg, imputer.fit, X)

    # Test with inf present
    X = np.array([
        [np.inf, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [6, 6, 2, 5, 7],
    ])
    msg = "+/- inf values are not supported."
    assert_raise_message(ValueError, msg, bk.ForestImputer().fit, X)

    # Test with inf present in matrix passed in transform()
    X = np.array([
        [np.inf, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [6, 6, 2, 5, 7],
    ])

    X_fit = np.array([
        [0, 1, 1, 2, np.nan],
        [2, 1, 2, 2, 3],
        [3, 2, 3, 3, 8],
        [np.nan, 6, 0, 5, 13],
        [np.nan, 7, 0, 7, 8],
        [6, 6, 2, 5, 7],
    ])
    msg = "+/- inf values are not supported."
    assert_raise_message(
        ValueError, msg, bk.ForestImputer().fit(X_fit).transform, X)
