"""Automatic stationarity decisions shared by every forecasting notebook.

Rules follow Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.):

- ``d`` (regular differences): difference until ADF rejects a unit root AND KPSS does not reject
  stationarity. If the tests disagree, the series is differenced (conservative for prices).
- ``D`` (seasonal differences): one seasonal difference when the STL seasonal strength F_s > 0.64.
- Exogenous drivers (Engle-Granger): when the target and a driver are both I(1) and cointegrated,
  the model gets an error-correction term (ECM) plus driver changes; otherwise only driver changes.
  Regressing one non-stationary level on another without cointegration gives spurious results.

Every regressor row for period t only uses information known at t-1, so the features are safe for
one-step-ahead cross-validation and live forecasts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, coint, kpss


def kpss_pvalue(x: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # p-values outside the lookup table are clipped to [0.01, 0.1]
        return float(kpss(x, regression="c", nlags="auto")[1])


def adf_pvalue(x: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(x, autolag="AIC")[1])


def is_stationary(x: pd.Series, alpha: float = 0.05) -> bool:
    """Confirmatory rule: ADF rejects a unit root AND KPSS does not reject stationarity.

    When the two tests disagree (common in short or very persistent series, where KPSS has little
    power) the series is treated as non-stationary — the safe choice for prices.
    """
    return adf_pvalue(x) < alpha and kpss_pvalue(x) >= alpha


def ndiffs(y: pd.Series, alpha: float = 0.05, max_d: int = 2) -> int:
    """Number of first differences needed until ADF and KPSS agree the series is stationary."""
    x = pd.Series(y).dropna()
    for d in range(max_d + 1):
        if is_stationary(x, alpha):
            return d
        x = x.diff().dropna()
    return max_d


def seasonal_strength(y: pd.Series, m: int) -> float:
    """STL seasonal strength F_s = max(0, 1 - Var(R) / Var(S + R)); 0 with too little data."""
    x = pd.Series(y).dropna()
    if m < 2 or len(x) < 2 * m + 1:
        return 0.0
    res = STL(x.to_numpy(), period=m, robust=True).fit()
    return float(max(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid)))


def nsdiffs(y: pd.Series, m: int, threshold: float = 0.64) -> int:
    """One seasonal difference when the seasonal strength exceeds the threshold."""
    return int(seasonal_strength(y, m) > threshold)


def coint_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Engle-Granger cointegration p-value (H0: no cointegration)."""
    df = pd.concat([y, x], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(coint(df.iloc[:, 0], df.iloc[:, 1])[1])


@dataclass
class Design:
    """Automatic modeling decisions for one target series."""

    season: int
    d: int
    d_log: int
    D: int
    seasonal_strength: float
    adf_p: float
    kpss_p: float
    drivers: list[str] = field(default_factory=list)
    driver_d: dict[str, int] = field(default_factory=dict)
    coint_p: dict[str, float] = field(default_factory=dict)
    coint_driver: str | None = None

    @property
    def exog_mode(self) -> str:
        if not self.drivers:
            return "none"
        if self.d == 0:
            return "level"
        return "ecm" if self.coint_driver else "diff"

    def summary(self) -> dict:
        out = {
            "d": self.d,
            "d_log": self.d_log,
            "D": self.D,
            "seasonal_strength": round(self.seasonal_strength, 3),
            "ADF_p": round(self.adf_p, 4),
            "KPSS_p": round(self.kpss_p, 4),
            "exog_mode": self.exog_mode,
            "ecm_driver": self.coint_driver,
        }
        out.update({f"coint_p[{c}]": round(p, 4) for c, p in self.coint_p.items()})
        return out


def decide(y: pd.Series, X: pd.DataFrame | None, season: int, alpha: float = 0.05) -> Design:
    """Run every test and return the decisions the models must follow."""
    y = y.dropna()
    adf_p = adf_pvalue(y)
    design = Design(
        season=season,
        d=ndiffs(y, alpha),
        d_log=ndiffs(np.log(y[y > 0]), alpha),
        D=nsdiffs(y, season),
        seasonal_strength=seasonal_strength(y, season),
        adf_p=adf_p,
        kpss_p=kpss_pvalue(y),
    )
    if X is None or X.empty:
        return design
    design.drivers = list(X.columns)
    for c in X.columns:
        xc = X[c].dropna()
        design.driver_d[c] = ndiffs(xc, alpha)
        if design.d == 1 and design.driver_d[c] == 1:  # Engle-Granger needs both series I(1)
            design.coint_p[c] = coint_pvalue(y, xc)
    significant = {c: p for c, p in design.coint_p.items() if p < alpha}
    if significant:
        design.coint_driver = min(significant, key=significant.get)
    return design


def _coint_vector(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS of y on x: returns (intercept, slope) of the long-run relation."""
    df = pd.concat([y, x], axis=1).dropna()
    slope, intercept = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)
    return float(intercept), float(slope)


def exog_features(
    y: pd.Series, X: pd.DataFrame, design: Design, coef: tuple[float, float] | None = None
) -> tuple[pd.DataFrame, tuple[float, float] | None]:
    """Regressors for period t built only from information known at t-1.

    ``X`` holds each driver's value known at each period of ``y``. In "ecm" mode the long-run
    vector is estimated on ``y`` itself (pass the training window only), and returned so the same
    vector is reused for the next-step row.
    """
    Xy = X.loc[y.index, design.drivers]
    feats = pd.DataFrame(index=y.index)
    if design.exog_mode == "level":
        for c in design.drivers:
            feats[f"{c}_lag1"] = Xy[c].shift(1)
        return feats, None
    if design.exog_mode == "ecm":
        if coef is None:
            coef = _coint_vector(y, Xy[design.coint_driver])
        a, b = coef
        feats["ect_lag1"] = (y - a - b * Xy[design.coint_driver]).shift(1)
    for c in design.drivers:
        feats[f"d_{c}_lag1"] = Xy[c].diff().shift(1)
    return feats, coef


def exog_next(
    y: pd.Series, X: pd.DataFrame, design: Design, coef: tuple[float, float] | None
) -> pd.DataFrame:
    """Regressor row for the period right after ``y`` ends (values known at its last date)."""
    last, prev = y.index[-1], y.index[-2]
    row = {}
    if design.exog_mode == "level":
        for c in design.drivers:
            row[f"{c}_lag1"] = X.loc[last, c]
        return pd.DataFrame([row])
    if design.exog_mode == "ecm":
        a, b = coef
        row["ect_lag1"] = y.loc[last] - a - b * X.loc[last, design.coint_driver]
    for c in design.drivers:
        row[f"d_{c}_lag1"] = X.loc[last, c] - X.loc[prev, c]
    return pd.DataFrame([row])
