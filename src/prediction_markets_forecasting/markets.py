"""Market data builders: the target X (from the resolution source) and its driver candidates.

Every builder returns a ``MarketData`` with the same shape, so the template notebook runs Parts 1–6
identically for any market. Adding a market = a spec in ``specs.py`` (+ a builder if it is a new
kind of market).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import kalshi, sources


@dataclass
class MarketData:
    Y: dict[str, pd.Series]  # target per frequency
    source: dict[str, pd.Series]  # per-observation origin label (resolution source vs proxy)
    X: dict[
        str, pd.DataFrame
    ]  # drivers known at each date (ahead drivers may extend 1 period past Y)
    driver_info: pd.DataFrame
    ahead: list[str] = field(default_factory=list)
    placebo: str | None = None
    reports: dict[str, pd.DataFrame] = field(default_factory=dict)  # tables for Part 1
    bridges: dict[str, pd.DataFrame] = field(default_factory=dict)  # resolution vs proxy, to plot
    notes: list[str] = field(default_factory=list)


def _driver_info(raw: dict[str, pd.Series], names: dict[str, str], as_of, ahead=()) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver": k,
                "description": names.get(k, k),
                "ahead": k in ahead,
                "start": v.dropna().index.min().date(),
                "end": v.dropna().index.max().date(),
                "n_obs": int(v.notna().sum()),
                "days_stale": (as_of - v.dropna().index.max()).days,
            }
            for k, v in raw.items()
        ]
    ).set_index("driver")


def _observed_known(raw: dict[str, pd.Series], as_of) -> pd.DataFrame:
    """Calendar-daily grid, forward-filled, shifted one day: the value used for date t was public
    before t (no look-ahead)."""
    X = pd.DataFrame(raw).sort_index()
    X = X.reindex(pd.date_range(X.index.min(), as_of, freq="D")).ffill()
    return X.shift(1)


# ---------------------------------------------------------------------------------------------
# Family B — AAA retail gasoline (national): daily / weekly / monthly
# ---------------------------------------------------------------------------------------------
GAS_FRED = {
    "DGASUSGULF": "Gulf Coast gasoline spot (FRED)",
    "DGASNYH": "NY Harbor gasoline spot (FRED)",
    "DCOILWTICO": "WTI crude spot (FRED)",
    "DCOILBRENTEU": "Brent crude spot (FRED)",
    "DHOILNYH": "NY Harbor heating oil spot (FRED)",
    "DTWEXBGS": "Trade-weighted US dollar (FRED)",
    "DGS10": "10Y Treasury yield — placebo (FRED)",
}
GAS_YAHOO = {"RB=F": "RBOB gasoline futures (Yahoo)", "CL=F": "WTI crude futures (Yahoo)"}


def build_aaa_gas(spec, as_of: pd.Timestamp) -> MarketData:
    p = spec.params
    aaa, reps = {}, []
    for k, cfg in spec.freqs.items():
        aaa[k], rep = kalshi.settlements(cfg.series)
        reps.append({"frequency": k, **rep})
    reports = {"Kalshi settlement data quality": pd.DataFrame(reps)}

    # Weekly: EIA retail (Mondays, since 1990) bridged to AAA's Monday settlements
    eia_w = sources.fred("GASREGW")
    aaa_w = aaa["weekly"][aaa["weekly"].index.dayofweek == 0]
    bw = pd.concat(
        [aaa_w.rename("resolution (AAA)"), eia_w.rename("proxy (EIA)")], axis=1, sort=True
    ).dropna()
    bias_w = (bw.iloc[:, 0] - bw.iloc[:, 1]).mean()
    y_w = (eia_w + bias_w).combine_first(aaa_w).asfreq("W-MON")
    src_w = pd.Series("EIA + bias", index=y_w.index)
    common = aaa_w.index.intersection(y_w.index)
    y_w.loc[common], src_w.loc[common] = aaa_w.loc[common], "AAA (Kalshi)"
    n_ffill_w = int(y_w.isna().sum())
    y_w, src_w = y_w.ffill()[p["weekly_start"] :], src_w[p["weekly_start"] :]

    # Monthly: AAA on the last day of the month; EIA interpolated to month-ends as the proxy
    eia_d = eia_w.resample("D").interpolate("time")
    eia_me = eia_d[eia_d.index.is_month_end]
    bm = pd.concat([aaa["monthly"].rename("resolution (AAA)"), eia_me.rename("proxy (EIA interp)")],
                   axis=1, sort=True).dropna()  # fmt: skip
    bias_m = (bm.iloc[:, 0] - bm.iloc[:, 1]).mean()
    y_m = (eia_me + bias_m).combine_first(aaa["monthly"])
    y_m.loc[aaa["monthly"].index] = aaa["monthly"]
    y_m = y_m.asfreq("ME").ffill()[p["monthly_start"] :]
    is_aaa = y_m.index.isin(aaa["monthly"].index)
    src_m = pd.Series(np.where(is_aaa, "AAA (Kalshi)", "EIA interp + bias"),
                      index=y_m.index)  # fmt: skip

    # Daily: AAA only — keep the last contiguous block (gaps <= 3 days)
    obs = aaa["daily"]
    gap = obs.index.to_series().diff().dt.days
    start_d = gap[gap > 3].index.max() if (gap > 3).any() else obs.index.min()
    y_d = obs[start_d:].asfreq("D")
    n_ffill_d = int(y_d.isna().sum())
    y_d = y_d.ffill()
    src_d = pd.Series("AAA (Kalshi)", index=y_d.index)

    reports["Bridge: resolution source vs long-history proxy"] = pd.DataFrame(
        [
            {"frequency": name, "overlap": len(b),
             "mean_diff": (b.iloc[:, 0] - b.iloc[:, 1]).mean(),
             "std_diff": (b.iloc[:, 0] - b.iloc[:, 1]).std(),
             "corr_levels": b.iloc[:, 0].corr(b.iloc[:, 1]),
             "corr_changes": b.iloc[:, 0].diff().corr(b.iloc[:, 1].diff())}
            for name, b in (("weekly", bw), ("monthly", bm))
        ]
    )  # fmt: skip

    raw = {k: sources.fred(k) for k in GAS_FRED}
    raw |= sources.yahoo_close(list(GAS_YAHOO))
    Xk = _observed_known(raw, as_of)
    Y = {
        "daily": y_d.rename("daily"),
        "weekly": y_w.rename("weekly"),
        "monthly": y_m.rename("monthly"),
    }
    return MarketData(
        Y=Y,
        source={"daily": src_d, "weekly": src_w, "monthly": src_m},
        X={k: Xk.reindex(y.index) for k, y in Y.items()},
        driver_info=_driver_info(raw, GAS_FRED | GAS_YAHOO, as_of),
        placebo="DGS10",
        reports=reports,
        bridges={"weekly": bw, "monthly": bm},
        notes=[
            f"Weekly: AAA − EIA bias = {bias_w * 100:+.2f}¢ applied to EIA before AAA history "
            f"({len(common)} AAA weeks, {n_ffill_w} forward-filled gaps).",
            f"Monthly: AAA − EIA(interp) bias = {bias_m * 100:+.2f}¢ ({len(bm)} AAA month-ends).",
            f"Daily: last contiguous AAA block from {start_d:%Y-%m-%d} "
            f"({n_ffill_d} forward-filled days).",
        ],
    )


# ---------------------------------------------------------------------------------------------
# Family A — daily high temperature at a Kalshi station
# ---------------------------------------------------------------------------------------------
def build_weather_high(spec, as_of: pd.Timestamp) -> MarketData:
    p = spec.params
    cfg = spec.freqs["daily"]
    kal, rep = kalshi.settlements(cfg.series)
    reports = {"Kalshi settlement data quality": pd.DataFrame([{"frequency": "daily", **rep}])}

    noaa = sources.noaa_daily(p["station"], "TMAX", p["normals"][0], f"{as_of:%Y-%m-%d}")
    b = pd.concat([kal.rename("resolution (Kalshi)"), noaa.rename(f"proxy (NOAA {p['station']})")],
                  axis=1, sort=True).dropna()  # fmt: skip
    diff = b.iloc[:, 0] - b.iloc[:, 1]
    reports["Bridge: Kalshi settlement vs NOAA station record"] = pd.DataFrame(
        [{"overlap_days": len(b), "exact_match_%": (diff == 0).mean() * 100,
          "within_1F_%": (diff.abs() <= 1).mean() * 100, "mean_diff_F": diff.mean(),
          "max_abs_diff_F": diff.abs().max()}]
    )  # fmt: skip

    # Target: NOAA station TMAX from the first date the weather-model archive exists; Kalshi's own
    # settlement overrides wherever it exists (it *is* the resolution value)
    y = noaa[p["model_start"] :].combine_first(kal)
    y.loc[kal.index.intersection(y.index)] = kal
    y = y[p["model_start"] :].asfreq("D")
    n_ffill = int(y.isna().sum())
    y = y.ffill().rename("daily")
    src = pd.Series(np.where(y.index.isin(kal.index), "Kalshi settlement", "NOAA"), index=y.index)

    # Drivers known before the day: each weather model's forecast issued the day before, and the
    # 1991–2020 climatological normal for the calendar day. Placebo: 10Y Treasury yield.
    fc = sources.open_meteo_daily_max_previous_day(
        p["lat"],
        p["lon"],
        p["nwp_models"],
        p["model_start"],
        f"{as_of + pd.Timedelta(days=1):%Y-%m-%d}",
        timezone=p["tz"],
    )
    base = noaa[p["normals"][0] : p["normals"][1]]
    doy_mean = base.groupby(base.index.dayofyear).mean()
    doy_mean = pd.concat([doy_mean.iloc[-15:], doy_mean, doy_mean.iloc[:15]])  # wrap the year edge
    smooth = doy_mean.rolling(31, center=True).mean().iloc[15:-15]
    idx = pd.date_range(p["model_start"], as_of + pd.Timedelta(days=1), freq="D")
    climo = pd.Series(smooth.reindex(idx.dayofyear).values, index=idx, name="climo_normal")
    placebo = _observed_known({"DGS10": sources.fred("DGS10")}, as_of + pd.Timedelta(days=1))[
        "DGS10"
    ]

    X = fc.reindex(idx).join(climo).join(placebo.reindex(idx))
    X = X.loc[:, X.notna().sum() > 365]  # drop models without at least a year of archive
    names = {c: f"{c.removeprefix('fcst_')} daily-max forecast issued the day before (Open-Meteo)"
             for c in X.columns if c.startswith("fcst_")}  # fmt: skip
    names |= {"climo_normal": "1991–2020 normal daily max for the calendar day (NOAA)",
              "DGS10": "10Y Treasury yield — placebo (FRED)"}  # fmt: skip
    ahead = [c for c in X.columns if c != "DGS10"]
    return MarketData(
        Y={"daily": y},
        source={"daily": src},
        X={"daily": X},
        driver_info=_driver_info({c: X[c] for c in X.columns}, names, as_of, ahead),
        ahead=ahead,
        placebo="DGS10",
        reports=reports,
        bridges={"daily": b},
        notes=[
            f"Target from {p['model_start']} (weather-model archive start); {n_ffill} missing days "
            "forward-filled.",
            "Weather-model daily max uses 24 hours of US Eastern *standard* time, matching "
            "the NWS climate day (midnight-to-midnight LST, also during daylight saving time).",
        ],
    )


BUILDERS = {"aaa_gas": build_aaa_gas, "weather_high": build_weather_high}


def build(spec, as_of: pd.Timestamp) -> MarketData:
    return BUILDERS[spec.builder](spec, as_of)
