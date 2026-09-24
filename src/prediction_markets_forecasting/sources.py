"""Free public data sources (no API keys): FRED, Yahoo Finance, NOAA GHCN-Daily, Open-Meteo."""

from __future__ import annotations

import time

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
NOAA = "https://www.ncei.noaa.gov/access/services/data/v1"
OPEN_METEO_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fred(series_id: str) -> pd.Series:
    df = pd.read_csv(FRED_CSV.format(series_id), parse_dates=["observation_date"])
    s = pd.to_numeric(df[series_id], errors="coerce")
    return pd.Series(s.values, index=df["observation_date"], name=series_id).dropna()


def yahoo_close(tickers: list[str], start: str = "2004-01-01") -> dict[str, pd.Series]:
    import yfinance as yf

    close = yf.download(tickers, start=start, progress=False, auto_adjust=False)["Close"]
    return {t: close[t].dropna().rename(t) for t in tickers}


def noaa_daily(station: str, element: str, start: str, end: str) -> pd.Series:
    """GHCN-Daily values in standard units (°F for temperatures), requested in 5-year chunks."""
    parts = []
    s, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    while s <= end_ts:
        e = min(s + pd.DateOffset(years=5) - pd.Timedelta(days=1), end_ts)
        r = requests.get(
            NOAA,
            params={
                "dataset": "daily-summaries",
                "stations": station,
                "dataTypes": element,
                "startDate": f"{s:%Y-%m-%d}",
                "endDate": f"{e:%Y-%m-%d}",
                "units": "standard",
                "format": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        parts += r.json()
        s = e + pd.Timedelta(days=1)
        time.sleep(0.5)
    df = pd.DataFrame(parts)
    if df.empty:
        return pd.Series(dtype=float, name=element)
    out = pd.Series(
        pd.to_numeric(df[element], errors="coerce").values, index=pd.to_datetime(df["DATE"])
    )
    return out[~out.index.duplicated()].sort_index().dropna().rename(element)


def open_meteo_daily_max_previous_day(
    lat: float,
    lon: float,
    models: list[str],
    start: str,
    end: str,
    lead_days: int = 1,
    timezone: str = "Etc/GMT+5",
) -> pd.DataFrame:
    """Daily max 2 m temperature (°F) that each weather model forecast ``lead_days`` days earlier.

    Uses Open-Meteo's Previous Runs API (archived forecasts, not observations), so the value for day
    D is what a bettor could have seen before D. ``Etc/GMT+5`` = US Eastern *standard* time: the NWS
    climate day runs midnight-to-midnight local standard time all year, including during DST.
    """
    var = f"temperature_2m_previous_day{lead_days}"
    frames = []
    s, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    while s <= end_ts:
        e = min(pd.Timestamp(f"{s.year}-12-31"), end_ts)
        r = requests.get(
            OPEN_METEO_PREVIOUS_RUNS,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": var,
                "models": ",".join(models),
                "temperature_unit": "fahrenheit",
                "timezone": timezone,
                "start_date": f"{s:%Y-%m-%d}",
                "end_date": f"{e:%Y-%m-%d}",
            },
            timeout=120,
        )
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(
            {k: v for k, v in h.items() if k != "time"}, index=pd.to_datetime(h["time"])
        )
        frames.append(df)
        s = e + pd.Timedelta(days=1)
        time.sleep(0.3)
    hourly = pd.concat(frames).sort_index()
    hourly = hourly[~hourly.index.duplicated()]
    # a day's max needs all 24 hours; incomplete days (e.g. still in the future) become NaN
    daily = hourly.resample("D").max().where(hourly.resample("D").count() == 24)
    daily.columns = [c.replace(f"{var}_", "fcst_") for c in daily.columns]
    return daily
