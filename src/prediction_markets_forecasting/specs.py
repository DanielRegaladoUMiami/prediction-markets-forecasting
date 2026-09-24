"""One spec per market. Adding a market to the system = adding an entry to REGISTRY.

Render its notebook with:  uv run python -m prediction_markets_forecasting.render <key>
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tournament import FreqConfig


@dataclass
class MarketSpec:
    key: str
    number: str  # notebook number, e.g. "01"
    title: str
    family: str  # A weather | B lagging statistic | C macro release | D futures price
    underlying: str  # what X is, in words
    units: str
    integer: bool  # underlying resolves on whole numbers (temperatures) → continuity correction
    builder: str  # key of markets.BUILDERS
    freqs: dict[str, FreqConfig]
    discovery_freq: str  # frequency used to discover drivers (the longest reliable history)
    nowcast_freq: str | None = None  # finer series used as a random-walk cross-check in Part 6
    params: dict = field(default_factory=dict)
    limitations: list[tuple[str, str, str]] = field(default_factory=list)


def weather_high(
    key, number, city, series, station, lat, lon, tz, nwp_models=None, model_start="2021-04-01"
) -> MarketSpec:
    """Daily high temperature at the station a Kalshi KXHIGH* series settles on."""
    return MarketSpec(
        key=key,
        number=number,
        title=f"Daily High Temperature — {city}",
        family="A weather",
        underlying=f"Maximum temperature (°F) at {station} for the calendar day, as reported "
        "by the NWS climate report / The Weather Company",
        units="°F",
        integer=True,
        builder="weather_high",
        freqs={"daily": FreqConfig(series, "D", 7, 45, 365, 7, "doy")},
        discovery_freq="daily",
        params={
            "station": station,
            "lat": lat,
            "lon": lon,
            "tz": tz,
            "model_start": model_start,
            "normals": ("1991-01-01", "2020-12-31"),
            "nwp_models": nwp_models
            or ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"],
        },
        limitations=[
            (
                "Only day-before model runs",
                "Morning-of runs (fresher) are not used",
                "Add Open-Meteo previous_day0 / HRRR for same-day bets",
            ),
            (
                "Normal errors",
                "Convective days (sea breeze, storms) have fatter tails",
                "Empirical / Student-t errors by season",
            ),
            (
                "Station vs grid point",
                "Model grid cells average land and sea near the coast",
                "The tournament's bias correction (ECM / regression) learns the station offset",
            ),
            (
                "Market already sees the day",
                "Once the event day starts, observations beat forecasts",
                "Bet the evening before or early morning; compare with live METAR",
            ),
        ],  # fmt: skip
    )


REGISTRY: dict[str, MarketSpec] = {
    "aaa_gas_us": MarketSpec(
        key="aaa_gas_us",
        number="01",
        title="AAA US Regular Gas Price",
        family="B lagging statistic",
        underlying="AAA national average price of regular gasoline ($/gal) on the contract date",
        units="$/gal",
        integer=False,
        builder="aaa_gas",
        freqs={
            "daily": FreqConfig("KXAAAGASD", "D", 7, 28, 60, 7, "dow"),
            "weekly": FreqConfig("KXAAAGASW", "W-MON", 52, 26, 156, 12, "woy"),
            "monthly": FreqConfig("KXAAAGASM", "ME", 12, 24, 60, 12, "month"),
        },
        discovery_freq="weekly",
        nowcast_freq="daily",
        params={"weekly_start": "2015-01-01", "monthly_start": "2007-01-01"},
        limitations=[
            (
                "Daily history is short (~6 months)",
                "Daily models and CV are noisy",
                "Record AAA daily forward; paid AAA/OPIS history",
            ),
            (
                "Monthly model is stale",
                "Forecasts month-end from last month-end",
                "Use the daily nowcast in Part 6",
            ),
            (
                "Market sees live station prices",
                "Market σ is 2x tighter on daily/weekly",
                "Bet weekly/monthly early in the period",
            ),
            (
                "Missing fundamentals",
                "Inventories and refinery utilization not included",
                "EIA API (free key)",
            ),
            (
                "No backtest vs historical Kalshi prices",
                "Edges are live signals, not proven P&L",
                "Replay Kalshi candlesticks",
            ),
        ],  # fmt: skip
    ),
    "high_temp_miami": weather_high(
        "high_temp_miami", "02", "Miami", "KXHIGHMIA", "USW00012839", 25.7906, -80.3164, "Etc/GMT+5"
    ),
}


def get_spec(key: str) -> MarketSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown market '{key}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]
