"""The multi-model forecasting tournament from the MAS 640 Logitech notebook, one step ahead.

Families: SARIMAX / Log-SARIMAX (AIC picks the order among the automatic d/D, then CV scores it),
SARIMAX-X (ECM / driver regressors), ETS (10 configs), 7 Nixtla StatsForecast models, Gradient
Boosting and Random Forest (with and without drivers), Naive / Seasonal Naive / Drift baselines,
and a best-per-family ensemble. Every model is scored with expanding-window one-step CV.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsforecast import StatsForecast
from statsforecast.models import (
    MSTL,
    AutoARIMA,
    AutoCES,
    AutoTheta,
    DynamicOptimizedTheta,
    OptimizedTheta,
)
from statsforecast.models import AutoETS as NixtlaAutoETS
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .stationarity import Design, exog_features, exog_next

# Same 24 orders as the original notebook; d and D are replaced by the automatic decisions
SARIMAX_CONFIGS = [
    (0, 1, 1, 0, 1, 1), (0, 1, 1, 1, 1, 1), (0, 1, 2, 0, 1, 1), (0, 1, 2, 1, 1, 1),
    (1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1), (1, 1, 2, 0, 1, 1), (1, 1, 2, 1, 1, 1),
    (2, 1, 1, 0, 1, 1), (2, 1, 1, 1, 1, 1), (2, 1, 2, 0, 1, 1), (2, 1, 2, 1, 1, 1),
    (0, 0, 1, 1, 0, 1), (0, 0, 2, 1, 0, 1), (1, 0, 0, 1, 0, 0), (1, 0, 1, 1, 0, 0),
    (1, 0, 1, 1, 0, 1), (2, 0, 0, 1, 0, 0), (2, 0, 1, 1, 0, 0), (2, 0, 1, 1, 0, 1),
    (0, 1, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 1), (1, 1, 0, 0, 1, 1),
]  # fmt: skip
ETS_CONFIGS = [
    ("add", False, "add", "ETS(A,A,A)"), ("add", True, "add", "ETS(A,Ad,A)"),
    ("mul", False, "add", "ETS(A,M,A)"), ("mul", True, "add", "ETS(A,Md,A)"),
    ("add", False, "mul", "ETS(M,A,M)"), ("add", True, "mul", "ETS(M,Ad,M)"),
    ("mul", False, "mul", "ETS(M,M,M)"), ("mul", True, "mul", "ETS(M,Md,M)"),
    (None, False, "add", "ETS(A,N,A)"), (None, False, "mul", "ETS(M,N,M)"),
]  # fmt: skip
NIXTLA = {
    "AutoARIMA": AutoARIMA,
    "AutoETS": NixtlaAutoETS,
    "AutoTheta": AutoTheta,
    "AutoCES": AutoCES,
    "DynOptTheta": DynamicOptimizedTheta,
    "OptTheta": OptimizedTheta,
    "MSTL": MSTL,
}
ML_TEMPLATES = [
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                                   random_state=42)),
    ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
]  # fmt: skip
ENSEMBLE_NIXTLA = ["Nixtla_AutoARIMA", "Nixtla_AutoCES", "Nixtla_DynOptTheta", "Nixtla_MSTL"]


@dataclass
class FreqConfig:
    series: str  # Kalshi series ticker
    freq: str  # pandas frequency of the target grid
    season: int
    n_folds: int  # one-step CV folds
    min_train: int
    n_lags: int
    calendar: str  # dow | woy | month | doy — seasonal position for ML sin/cos features


def mape_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def ts_cv_rmse(full_series, model_fn, n_folds=24, val_size=1, min_train=60):
    """Expanding-window CV. Returns (pooled RMSE, MAPE, errors, predictions)."""
    errs, preds = [], []
    n = len(full_series)
    for fold in range(n_folds):
        val_end = n - fold * val_size
        val_start = val_end - val_size
        if val_start < min_train:
            break
        train_part = full_series.iloc[:val_start]
        val_part = full_series.iloc[val_start:val_end]
        try:
            fc = np.asarray(model_fn(train_part), dtype=float)[:val_size]
            if len(fc) < val_size or not np.all(np.isfinite(fc)):
                continue
            errs.append(val_part.values - fc)
            preds.append(pd.Series(fc, index=val_part.index))
        except Exception:
            pass
    if not errs:
        return float("inf"), float("inf"), np.array([]), pd.Series(dtype=float)
    e = np.concatenate(errs)
    p = pd.concat(preds).sort_index()
    return float(np.sqrt(np.mean(e**2))), mape_score(full_series.loc[p.index], p), e, p


def cal_pos(idx: pd.DatetimeIndex, calendar: str) -> tuple[np.ndarray, int]:
    if calendar == "dow":
        return np.asarray(idx.dayofweek), 7
    if calendar == "woy":
        return np.asarray(idx.isocalendar().week, dtype=int), 52
    if calendar == "doy":
        return np.asarray(idx.dayofyear), 365
    return np.asarray(idx.month), 12


def build_ml_features(series: pd.Series, n_lags: int, season: int, calendar: str) -> pd.DataFrame:
    """Lag, rolling, calendar and year-over-year features — the original notebook's recipe."""
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["roll_mean_12"] = df["y"].shift(1).rolling(12).mean()
    df["roll_std_6"] = df["y"].shift(1).rolling(6).std()
    pos, per = cal_pos(df.index, calendar)
    df["cal_sin"] = np.sin(2 * np.pi * pos / per)
    df["cal_cos"] = np.cos(2 * np.pi * pos / per)
    df["yoy_change"] = df["y"].shift(season)
    return df.dropna()


def ml_next_row(history: pd.Series, next_date, n_lags: int, season: int, calendar: str) -> dict:
    h = list(history.values)
    row = {f"lag_{lag}": h[-lag] for lag in range(1, n_lags + 1)}
    row["roll_mean_3"] = np.mean(h[-3:])
    row["roll_mean_6"] = np.mean(h[-6:])
    row["roll_mean_12"] = np.mean(h[-12:])
    row["roll_std_6"] = np.std(h[-6:], ddof=1)
    pos, per = cal_pos(pd.DatetimeIndex([next_date]), calendar)
    row["cal_sin"] = np.sin(2 * np.pi * pos[0] / per)
    row["cal_cos"] = np.cos(2 * np.pi * pos[0] / per)
    row["yoy_change"] = h[-season]
    return row


@dataclass
class TournamentResult:
    ranking: pd.DataFrame
    cv_err: dict
    cv_pred: dict
    forecasts: dict
    members: list
    next_date: pd.Timestamp
    ts: pd.Series
    fns: dict  # model name -> one-step forecast function (train series -> next value)
    step: pd.DateOffset

    @property
    def best(self) -> str:
        return self.ranking.iloc[0]["Model"]

    def steps_to(self, date: pd.Timestamp) -> int:
        """Number of periods from the last observation to ``date``."""
        return len(pd.date_range(self.ts.index[-1], date, freq=self.step)) - 1

    def forecast_path(self, model: str, steps: int) -> np.ndarray:
        """Recursive multi-step forecast: each step feeds the previous forecast back as history."""
        if model == "Ensemble":
            parts = [self.forecast_path(m, steps) for m in self.members if m in self.fns]
            return np.mean(parts, axis=0)
        fn, y, out = self.fns[model], self.ts.copy(), []
        for _ in range(steps):
            f = float(np.asarray(fn(y))[0])
            out.append(f)
            y = pd.concat([y, pd.Series([f], index=[y.index[-1] + self.step])])
        return np.array(out)


def run_tournament(
    y_full: pd.Series, X_full: pd.DataFrame, cfg: FreqConfig, design: Design, log=print
) -> TournamentResult:
    """Run every model family on one target series and rank them by one-step CV RMSE.

    ``X_full`` holds each driver's value known at each date; it may extend past the last target date
    (ahead drivers need their value at the next date).
    """
    warnings.filterwarnings("ignore")
    freq, s, n_folds, min_train, n_lags, calendar = (
        cfg.freq, cfg.season, cfg.n_folds, cfg.min_train, cfg.n_lags, cfg.calendar,
    )  # fmt: skip
    step = pd.tseries.frequencies.to_offset(freq)
    next_date = y_full.index[-1] + step
    Xk = X_full[design.drivers] if design.drivers else pd.DataFrame(index=y_full.index)
    ts = y_full
    if design.drivers:
        ok = Xk.reindex(y_full.index).notna().all(axis=1)
        ts = y_full[ok.idxmax() :]
    log(
        f"Length: {len(ts)} | {ts.index.min():%Y-%m-%d} → {ts.index.max():%Y-%m-%d} "
        f"| next target date: {next_date:%Y-%m-%d}"
    )
    log(
        f"  Automatic design: d={design.d}, d_log={design.d_log}, D={design.D}, "
        f"exog mode={design.exog_mode}"
        + (f" (ECM on {design.coint_driver})" if design.coint_driver else "")
        + (f" | ahead drivers: {design.ahead}" if design.ahead else "")
    )

    results, cv_err, cv_pred, forecasts, fns = [], {}, {}, {}, {}
    scale_factor = max(ts.mean() / 1000, 1)

    def record(model, mtype, fn, exog=False, aic=np.nan):
        rmse, mape, e, p = ts_cv_rmse(ts, fn, n_folds=n_folds, val_size=1, min_train=min_train)
        if rmse < float("inf"):
            results.append({"Model": model, "Type": mtype, "CV_RMSE": rmse, "MAPE": mape,
                            "n_cv": len(e), "AIC": aic, "Exog": exog})  # fmt: skip
            cv_err[model], cv_pred[model], fns[model] = e, p, fn
            try:
                forecasts[model] = float(np.asarray(fn(ts))[0])
            except Exception:
                pass

    # 1-2. SARIMAX & Log-SARIMAX: automatic d/D, AIC over (p, q, P, Q), then CV of the winner
    for log_t in (False, True):
        d_use = design.d_log if log_t else design.d
        grid = sorted({(p, d_use, q, P, design.D, Q) for (p, _, q, P, _, Q) in SARIMAX_CONFIGS})
        y_fit = np.log(ts / scale_factor) if log_t else ts / scale_factor
        res = []
        for p, d, q, P, D, Q in grid:
            try:
                fit = SARIMAX(y_fit.values, order=(p, d, q), seasonal_order=(P, D, Q, s),
                              enforce_stationarity=False, enforce_invertibility=False,
                              ).fit(disp=False, maxiter=100)  # fmt: skip
                if np.isfinite(fit.aic):
                    res.append(((p, d, q), (P, D, Q, s), fit.aic))
            except Exception:
                pass
        if not res:
            continue
        order, seas, aic = min(res, key=lambda r: r[2])

        def make_sarimax(log_t=log_t, order=order, seas=seas):
            def fn(tr):
                yy = np.log(tr / scale_factor) if log_t else tr / scale_factor
                fit = SARIMAX(yy.values, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False,
                              ).fit(disp=False, maxiter=100)  # fmt: skip
                fc = np.asarray(fit.forecast(1), dtype=float)
                return (np.exp(fc) if log_t else fc) * scale_factor

            return fn

        label = "LogSARIMAX" if log_t else "SARIMAX"
        record(f"{label}{order}x{seas}", label, make_sarimax(), aic=aic)
        log(f"  {label}: {len(grid)} orders with d={d_use}, D={design.D} → best AIC "
            f"{order}x{seas} (AIC={aic:.1f})")  # fmt: skip

        if not log_t and design.drivers:
            # SARIMAX-X: differenced target with ECM / driver-change regressors (no spurious levels)
            def make_sarimax_x(order=order, seas=seas):
                def fn(tr):
                    feats, coef = exog_features(tr, Xk, design)
                    if design.d >= 1:
                        w, o = tr.diff(), (order[0], order[1] - 1, order[2])
                    else:
                        w, o = tr, order
                    df = pd.concat([w.rename("w"), feats], axis=1).dropna()
                    fit = SARIMAX(df["w"].values, order=o, seasonal_order=seas,
                                  exog=df.drop(columns="w").values, enforce_stationarity=False,
                                  enforce_invertibility=False,
                                  ).fit(disp=False, maxiter=100)  # fmt: skip
                    nxt = exog_next(tr, Xk, design, coef, tr.index[-1] + step)
                    fc = float(
                        np.asarray(fit.forecast(1, exog=nxt[df.columns.drop("w")].values))[0]
                    )
                    return np.array([tr.iloc[-1] + fc if design.d >= 1 else fc])

                return fn

            record(f"SARIMAX-X[{design.exog_mode}]{order}x{seas}", "SARIMAX", make_sarimax_x(),
                   exog=True)  # fmt: skip

    # 3. ETS
    for trend, damped, seasonal_type, ets_name in ETS_CONFIGS:

        def make_ets(t=trend, dd=damped, ss=seasonal_type):
            def fn(tr):
                m = ExponentialSmoothing(tr.values, trend=t, damped_trend=dd if t else False,
                                         seasonal=ss, seasonal_periods=s)  # fmt: skip
                return m.fit(optimized=True).forecast(1)

            return fn

        record(ets_name, "ETS", make_ets())

    # 4. Nixtla
    for nx_name, nx_cls in NIXTLA.items():

        def make_nx(cls=nx_cls):
            def fn(tr):
                sf = StatsForecast(models=[cls(season_length=s)], freq=freq, n_jobs=1)
                sf.fit(
                    pd.DataFrame({"unique_id": 0.0, "ds": tr.index, "y": tr.values.astype(float)})
                )
                fc = sf.predict(h=1)
                return fc[[c for c in fc.columns if c not in ("ds", "unique_id")][0]].values

            return fn

        record(f"Nixtla_{nx_name}", "Nixtla", make_nx())
    log(f"  Nixtla: {sum(r['Type'] == 'Nixtla' for r in results)}/7 models evaluated")

    # 5. ML — predicts the change when d >= 1 (trees cannot extrapolate trending levels)
    for ml_name, tmpl in ML_TEMPLATES:
        for use_x in (False, True) if design.drivers else (False,):

            def make_ml(tmpl=tmpl, use_x=use_x):
                def fn(tr):
                    target = (tr.diff() if design.d >= 1 else tr).dropna()
                    feat = build_ml_features(target, n_lags, s, calendar)
                    if use_x:
                        xf, coef = exog_features(tr, Xk, design)
                        feat = feat.join(xf, how="inner").dropna()
                    if len(feat) < 30:
                        return None
                    X_ = feat.drop(columns=["y"])
                    model = clone(tmpl).fit(X_, feat["y"])
                    nxt_date = tr.index[-1] + step
                    row = ml_next_row(target, nxt_date, n_lags, s, calendar)
                    if use_x:
                        row.update(exog_next(tr, Xk, design, coef, nxt_date).iloc[0].to_dict())
                    pred = float(model.predict(pd.DataFrame([row])[X_.columns])[0])
                    return np.array([tr.iloc[-1] + pred if design.d >= 1 else pred])

                return fn

            record(ml_name + ("-X" if use_x else ""), "ML", make_ml(), exog=use_x)

    # 6. Baselines
    record("Naive", "Baseline", lambda tr: tr.values[-1:])
    record("SeasonalNaive", "Baseline", lambda tr: np.array([tr.values[-s]]))
    for frac in [0.25, 0.50, 0.75]:
        record(
            f"Drift_{int(frac * 100)}pct",
            "Baseline",
            lambda tr, f=frac: np.array([tr.values[-s] + (tr.values[-s] - tr.values[-2 * s]) * f]),
        )

    # Ensemble: best model per family, scored on the dates every member forecast
    rk = pd.DataFrame(results)
    fam_best = [
        rk[rk["Type"] == t].sort_values("CV_RMSE")["Model"].iloc[0]
        for t in ["SARIMAX", "LogSARIMAX", "ETS", "ML"]
        if (rk["Type"] == t).any()
    ]
    members = fam_best + [m for m in ENSEMBLE_NIXTLA + ["Naive"] if m in cv_pred]
    P = pd.concat({m: cv_pred[m] for m in members}, axis=1).dropna()
    ens = P.mean(axis=1)
    e = ts.loc[ens.index].values - ens.values
    ens_rmse = float(np.sqrt(np.mean(e**2)))
    results.append({"Model": "Ensemble", "Type": "Ensemble", "CV_RMSE": ens_rmse,
                    "MAPE": mape_score(ts.loc[ens.index], ens), "n_cv": len(e), "AIC": np.nan,
                    "Exog": any("-X" in m for m in members)})  # fmt: skip
    cv_err["Ensemble"], cv_pred["Ensemble"] = e, ens
    forecasts["Ensemble"] = float(np.mean([forecasts[m] for m in members if m in forecasts]))

    ranking = pd.DataFrame(results).sort_values("CV_RMSE").reset_index(drop=True)
    ranking.index += 1
    naive_rmse = ranking.loc[ranking["Model"] == "Naive", "CV_RMSE"].iloc[0]
    ranking["Skill_vs_Naive"] = 1 - ranking["CV_RMSE"] / naive_rmse
    ranking["Forecast"] = ranking["Model"].map(forecasts)
    out = TournamentResult(ranking, cv_err, cv_pred, forecasts, members, next_date, ts, fns, step)
    b = ranking.iloc[0]
    log(f"  >> Best CV: {b['Model']} (RMSE={b['CV_RMSE']:.4g}, MAPE={b['MAPE']:.2f}%, "
        f"skill vs Naive={b['Skill_vs_Naive']:+.1%}) → next = {b['Forecast']:.4f}")  # fmt: skip
    return out


def timed(fn, *args, log=print, **kwargs):
    t0 = time.time()
    out = fn(*args, log=log, **kwargs)
    log(f"  ({time.time() - t0:.0f}s)")
    return out
