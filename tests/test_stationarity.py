import numpy as np
import pandas as pd

from prediction_markets_forecasting.stationarity import (
    decide,
    exog_features,
    exog_next,
    ndiffs,
    nsdiffs,
)

rng = np.random.default_rng(0)
IDX = pd.date_range("2015-01-05", periods=500, freq="W-MON")


def test_ndiffs_white_noise_is_stationary():
    assert ndiffs(pd.Series(rng.normal(size=500), index=IDX)) == 0


def test_ndiffs_random_walk_needs_one_difference():
    assert ndiffs(pd.Series(rng.normal(size=500).cumsum(), index=IDX)) == 1


def test_nsdiffs_detects_strong_seasonality():
    t = np.arange(500)
    seasonal = pd.Series(10 * np.sin(2 * np.pi * t / 52) + rng.normal(size=500), index=IDX)
    assert nsdiffs(seasonal, 52) == 1
    assert nsdiffs(pd.Series(rng.normal(size=500), index=IDX), 52) == 0


def _cointegrated_pair():
    x = pd.Series(100 + rng.normal(size=500).cumsum(), index=IDX)
    y = 2 + 0.5 * x + rng.normal(scale=0.5, size=500)
    return y, x


def test_decide_picks_ecm_for_cointegrated_pair():
    y, x = _cointegrated_pair()
    unrelated = pd.Series(rng.normal(size=500).cumsum(), index=IDX)
    design = decide(y, pd.DataFrame({"x": x, "noise": unrelated}), season=52)
    assert design.d == 1
    assert design.coint_driver == "x"
    assert design.exog_mode == "ecm"


def test_exog_features_use_only_past_information():
    y, x = _cointegrated_pair()
    X = pd.DataFrame({"x": x})
    design = decide(y, X, season=52)
    feats, coef = exog_features(y, X, design)
    a, b = coef
    t = IDX[100]
    prev = IDX[99]
    assert np.isclose(feats.loc[t, "ect_lag1"], y[prev] - a - b * x[prev])
    assert np.isclose(feats.loc[t, "d_x_lag1"], x[prev] - x[IDX[98]])
    nxt = exog_next(y, X, design, coef)
    assert np.isclose(nxt.loc[0, "ect_lag1"], y.iloc[-1] - a - b * x.iloc[-1])


def test_ahead_driver_enters_without_lag():
    # a forecast known before t: y_t = forecast_t + noise
    f = pd.Series(80 + rng.normal(size=500).cumsum() * 0.1, index=IDX)
    y = f + rng.normal(scale=0.3, size=500)
    X = pd.DataFrame({"fcst": f})
    design = decide(y, X, season=52, ahead=["fcst"])
    feats, coef = exog_features(y, X, design)
    t = IDX[100]
    col = [c for c in feats.columns if c.endswith("_ahead")][0]
    if design.exog_mode == "level":
        assert np.isclose(feats.loc[t, col], f[t])
    else:
        assert np.isclose(feats.loc[t, col], f[t] - f[IDX[99]])
    nxt_date = IDX[-1] + pd.Timedelta(weeks=1)
    X_ext = pd.concat([X, pd.DataFrame({"fcst": [123.0]}, index=[nxt_date])])
    row = exog_next(y, X_ext, design, coef, nxt_date)
    expected = 123.0 if design.exog_mode == "level" else 123.0 - f.iloc[-1]
    assert np.isclose(row.loc[0, col], expected)
