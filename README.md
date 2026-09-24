# Prediction Markets Forecasting

Forecasting prediction markets (Kalshi — also tradable on Robinhood — and Polymarket) where public data
leads, or is sharper than, the crowd. **One system, one notebook template, one spec per market.**

## How it works

Every market is rendered from the same notebook (`notebooks/template.ipynb`), which follows the pipeline of
[product-sales-forecasting](https://github.com/DanielRegaladoUMiami/product-sales-forecasting) (MAS 640):

| Part | What happens |
|------|--------------|
| 1 | Target straight from Kalshi's settlement values, validated against a long-history public record; drivers as *the value known at each date* |
| 2 | EDA + driver discovery: lead-lag correlation of changes, Granger causality, placebo, collinearity filter |
| 3 | Diagnostics + **automatic stationarity**: `d` (ADF + KPSS), `D` (STL strength), cointegration → ECM |
| 4 | Tournament: SARIMAX, Log-SARIMAX, SARIMAX-X, ETS ×10, 7 Nixtla models, GB/RF (± drivers), baselines, ensemble — one-step expanding CV, skill vs random walk |
| 5 | Critique & next steps |
| 6 | Forecast distribution → probability of every Kalshi bracket → edge after fees, with a market-sharpness sanity check |

Adding a market = one entry in `src/prediction_markets_forecasting/specs.py`
(plus a builder in `markets.py` if it is a new kind of market).

## Market families

| Family | Examples | How Kalshi resolves | Approach |
|---|---|---|---|
| A — Weather | Daily high temperature (~20 cities), rain | NWS climate report at one station | Weather-model forecasts as drivers (MOS-style bias correction) |
| B — Lagging statistic | AAA gas (national, states), mortgage rate, ERCOT peak load | A published number that trails observable prices | Forecast the level with leading drivers |
| C — Macro release | CPI, unemployment, payrolls | BLS / BEA release | Nowcast by components |
| D — Futures price | WTI, natural gas, metals, FX | Futures / oracle price | Forecast volatility, compare with options |

## Markets

| # | Market | Kalshi series | Notebook | Status |
|---|--------|---------------|----------|--------|
| 00 | Kalshi commodity market scan | all | `00_kalshi_market_scan.ipynb` | done |
| 01 | AAA US regular gas — daily / weekly / monthly | `KXAAAGASD` / `KXAAAGASW` / `KXAAAGASM` | `01_aaa_gas_us.ipynb` | beats random walk; market sharper on daily |
| 02 | Daily high temperature — Miami (MIA) | `KXHIGHMIA` | `02_high_temp_miami.ipynb` | v1 |

## How to run

```bash
export UV_PROJECT_ENVIRONMENT=.venv.nosync   # repo lives under iCloud; see CLAUDE.md
uv sync
uv run python scripts/build_template.py                          # after editing the template
uv run python -m prediction_markets_forecasting.render high_temp_miami
uv run python -m prediction_markets_forecasting.render --all
uv run python -m prediction_markets_forecasting.render high_temp_miami --fast   # 3-fold pipeline check
uv run pytest
```

Data sources are free and keyless: Kalshi public API, FRED, Yahoo Finance, NOAA GHCN-Daily, Open-Meteo.

## License

Apache 2.0
