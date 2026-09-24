"""Exogenous driver discovery: lead-lag correlation of changes, Granger causality, selection rule.

Correlating *levels* of trending series is misleading, so everything here works on changes.
A driver is useful only if it moves before the target:
- observed drivers (known after the fact) must lead by k >= 1 periods;
- "ahead" drivers (forecasts for period t, known before t) may count at k = 0.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def lead_lag(y: pd.Series, X: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    """corr(Δy_t, Δx_{t−k}) for every driver (rows) and lag k = 0..max_lag (columns)."""
    dy, dX = y.diff(), X.diff()
    return pd.DataFrame(
        {k: [dy.corr(dX[c].shift(k)) for c in dX.columns] for k in range(max_lag + 1)},
        index=dX.columns,
    )


def granger_p(y: pd.Series, x: pd.Series, ahead: bool, maxlag: int = 4) -> float:
    """Min p-value (lags 1..maxlag) that driver changes improve the forecast of target changes.

    For ahead drivers the value for t is known before t, so it is shifted one period earlier and
    "lag 1" becomes the contemporaneous forecast.
    """
    dx = x.diff().shift(-1) if ahead else x.diff()
    data = pd.concat([y.diff(), dx], axis=1).dropna()
    if len(data) < 5 * maxlag + 10:
        return np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        res = grangercausalitytests(data, maxlag=maxlag)
    return float(min(res[lag][0]["ssr_ftest"][1] for lag in res))


def discover(
    y: pd.Series, X: pd.DataFrame, ahead: list[str], max_lag: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Driver table (level corr, predictive lead-lag corr, Granger p) and the lead-lag matrix."""
    X = X.reindex(y.index)
    cc = lead_lag(y, X, max_lag)
    rows = []
    for c in X.columns:
        first_k = 0 if c in ahead else 1
        pred = cc.loc[c, first_k:]
        best_k = int(pred.abs().idxmax()) if pred.notna().any() else first_k
        rows.append(
            {
                "driver": c,
                "ahead": c in ahead,
                "level_corr": y.corr(X[c]),
                "corr_k0": cc.loc[c, 0],
                "best_pred_lag": best_k,
                "corr_best_pred": pred.get(best_k, np.nan),
                "granger_p": granger_p(y, X[c], c in ahead),
            }
        )
    table = pd.DataFrame(rows).set_index("driver")
    table["abs_pred_corr"] = table["corr_best_pred"].abs()
    return table.sort_values("abs_pred_corr", ascending=False), cc


def select(
    table: pd.DataFrame,
    X: pd.DataFrame,
    exclude: list[str] | None = None,
    max_drivers: int = 3,
    p_max: float = 0.01,
    collinear: float = 0.9,
) -> list[str]:
    """Keep Granger-significant drivers, strongest first, skipping near-duplicates."""
    dX = X.diff()
    chosen: list[str] = []
    candidates = table[(table["granger_p"] < p_max) & ~table.index.isin(exclude or [])]
    for c in candidates.sort_values("abs_pred_corr", ascending=False).index:
        if all(abs(dX[c].corr(dX[s])) < collinear for s in chosen):
            chosen.append(c)
        if len(chosen) == max_drivers:
            break
    return chosen
