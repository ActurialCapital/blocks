import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pandas.testing import assert_frame_equal

import blocks as bk


length = 50
n_paths = 10


end_date = datetime.now().date()
start_date = end_date - timedelta(days=length - 1)
index = pd.date_range(start=start_date, end=end_date, freq="D")


assets = [f'asset_{n}' for n in range(1, n_paths + 1)]
factors = [f'factor_{n}' for n in range(1, n_paths + 1)]


X_train = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=factors,
    index=index
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
    columns=factors,
    index=index
)
y_test = pd.DataFrame(
    np.random.normal(size=(length, n_paths)),
    columns=factors,
    index=index
)


@pytest.mark.parametrize("shift_by", [shift_by for shift_by in range(0, 5)])
def test_shift_factor(shift_by):
    Xi, yi = bk.ShiftFactor(shift_by).fit_resample(X_train, y_train)

    Xt = X_train.shift(shift_by).iloc[shift_by:]
    yt = y_train.iloc[shift_by:]

    assert_frame_equal(Xi, Xt)
    assert_frame_equal(yi, yt)


@pytest.mark.parametrize("shift_by", [shift_by for shift_by in range(0, 5)])
@pytest.mark.parametrize("select", [select for select in bk.BasisOperation.TRANSFORMERS.keys()])
def test_basis_operation(shift_by, select):
    pred, _ = bk.BasisOperation(select, shift_by).fit_resample(X_train, y_train)

    X, y = X_train.align(y_train, join='inner')
    X = X.shift(shift_by).iloc[shift_by:]
    y = y.iloc[shift_by:]
    operation = bk.BasisOperation.TRANSFORMERS[select]
    output = operation(X, y)

    assert_frame_equal(pred, output)

    with pytest.raises(TypeError):
        bk.BasisOperation('hello').fit_resample(X_train, y_train)


@pytest.mark.parametrize("sign, thresh", [(sign, thresh) for sign, thresh in zip(('<=', '==', '>='), (-.2, 0, .2))])
@pytest.mark.parametrize("freq", [freq for freq in ['D', 'ME']])
@pytest.mark.parametrize("agg", [agg for agg in ['last', 'mean']])
def test_filter(sign, thresh, freq, agg):
    # Resample accordingly
    y = y_train.copy().resample(freq).agg(agg).ffill()

    _, pred = bk.Filter(y_train, sign, thresh, freq, agg).fit_resample(X_train, y)

    # Resample source data
    source = y_train.resample(freq).agg(agg).ffill()
    aligned = source.loc[y.index, y.columns]
    # Mask data
    operation = bk.Filter.FILTER_TRANSFORMERS[sign]
    mask_data = operation(aligned, thresh)
    # Apply mask
    output = y.where(mask_data, np.nan)

    assert_frame_equal(pred, output)
