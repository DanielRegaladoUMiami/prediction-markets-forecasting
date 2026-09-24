"""Kalshi public API (v2): settlements, open strike ladders, fees and bracket probabilities."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def get(path: str, key: str = "markets", **params) -> list[dict]:
    """Page through an endpoint with cursor pagination, backing off on HTTP 429."""
    out, cursor = [], None
    while True:
        q = {"limit": 1000, **params}
        if cursor:
            q["cursor"] = cursor
        for attempt in range(6):
            r = requests.get(BASE + path, params=q, timeout=30)
            if r.status_code != 429:
                break
            time.sleep(2**attempt)
        r.raise_for_status()
        d = r.json()
        out += d.get(key, [])
        cursor = d.get("cursor")
        if not cursor or not d.get(key):
            return out
        time.sleep(0.25)


def settlements(series_ticker: str) -> tuple[pd.Series, dict]:
    """Exact resolution values (``expiration_value``) of every settled event of a series.

    The historical endpoint also returns legacy tickers (``AAAGASD-23OCT03-US``) and sometimes
    events of other series (monthly ``AAAGASM-*`` inside the weekly series), so events are kept only
    when their prefix matches the series (with or without the ``KX`` prefix).
    """
    raw = get("/historical/markets", series_ticker=series_ticker) + get(
        "/markets", series_ticker=series_ticker, status="settled"
    )
    report = {"series": series_ticker, "markets": len(raw)}
    if not raw:
        return pd.Series(dtype=float, name=series_ticker), report | {"clean_obs": 0}
    df = pd.DataFrame(
        [{"event": m["event_ticker"], "value": m.get("expiration_value")} for m in raw]
    )
    report["events"] = df["event"].nunique()
    parts = df["event"].str.extract(r"^(?P<prefix>[A-Z0-9]+)-(?P<date>\d{2}[A-Z]{3}\d{2})")
    own = parts["prefix"].isin([series_ticker, series_ticker.removeprefix("KX")])
    report["foreign_events_dropped"] = int(df.loc[~own, "event"].nunique())
    df = df[own].assign(date_str=parts.loc[own, "date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    report["non_numeric_events"] = int(
        df.groupby("event")["value"].apply(lambda v: v.isna().all()).sum()
    )
    df = df.dropna(subset=["value"]).drop_duplicates("event")
    df["date"] = pd.to_datetime(df["date_str"], format="%y%b%d")
    s = df.sort_values("date").set_index("date")["value"]
    s = s[~s.index.duplicated()].rename(series_ticker)
    report |= {
        "clean_obs": len(s),
        "start": s.index.min().date() if len(s) else None,
        "end": s.index.max().date() if len(s) else None,
    }
    return s, report


def ladder(series_ticker: str) -> pd.DataFrame:
    """Open markets of a series with their bracket definition and quotes.

    Each market becomes an interval [low, high) of the underlying:
    ``greater`` (X > floor), ``less`` (X < cap) and ``between`` (floor <= X <= cap).
    """
    rows = []
    for m in get("/markets", series_ticker=series_ticker, status="open"):

        def num(f, m=m):
            v = m.get(f)
            return float(v) if v not in (None, "") else np.nan

        stype = m.get("strike_type") or "greater"
        floor, cap = num("floor_strike"), num("cap_strike")
        if stype == "greater" and np.isnan(floor):
            floor = float(m["ticker"].split("-")[-1].lstrip("T"))
        rows.append(
            {
                "ticker": m["ticker"],
                "event_date": pd.to_datetime(m["event_ticker"].split("-")[1][:7], format="%y%b%d"),
                "close_time": pd.to_datetime(m.get("close_time")),
                "strike_type": stype,
                "floor": floor,
                "cap": cap,
                "yes_bid": num("yes_bid_dollars"),
                "yes_ask": num("yes_ask_dollars"),
                "volume": num("volume_fp"),
                "open_interest": num("open_interest_fp"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.loc[df["yes_ask"] <= 0, "yes_ask"] = np.nan  # 0 = no offer
    df["no_ask"] = 1 - df["yes_bid"]  # one order book: buying NO = selling YES at the bid
    df["mid"] = (df["yes_bid"] + df["yes_ask"].fillna(1)) / 2
    return df.sort_values(["event_date", "strike_type", "floor", "cap"]).reset_index(drop=True)


def bracket_bounds(row: pd.Series, integer: bool) -> tuple[float, float]:
    """Interval of the underlying that makes the market resolve YES.

    For integer-valued underlyings (temperatures in °F) we use a continuity correction:
    ``X > 91`` means X >= 92, i.e. the continuous interval (91.5, inf).
    """
    half = 0.5 if integer else 0.0
    if row["strike_type"] == "greater":
        return row["floor"] + half, np.inf
    if row["strike_type"] == "less":
        return -np.inf, row["cap"] - half
    return row["floor"] - half, row["cap"] + half


def prob_yes(row: pd.Series, mu: float, sigma: float, integer: bool) -> float:
    lo, hi = bracket_bounds(row, integer)
    return float(norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma))


def fee(p: float | np.ndarray) -> float | np.ndarray:
    """Kalshi taker fee per contract ≈ 0.07 · P · (1 − P) (rounded up per order)."""
    return 0.07 * p * (1 - p)
