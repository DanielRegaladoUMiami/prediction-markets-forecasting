# Roadmap — Prediction Markets Forecasting

## Current milestone: v0.1
- [x] Initial scaffold
- [x] Data loading inside notebooks: Kalshi settlements (historical + live), FRED CSV, yfinance
- [ ] Forward recorder for Kalshi market prices (Part 6 already snapshots each run to data/kalshi_snapshots/)
- [x] Data quality: AAA vs EIA bridge (weekly change corr 0.997, bias +1.5¢); legacy/foreign tickers filtered
- [x] Notebook 01 v1 — AAA US gas at three horizons: daily (KXAAAGASD), weekly (KXAAAGASW), monthly (KXAAAGASM); Logitech pipeline + exogenous + Part 6

## Next up
- Notebook 01 v2: asymmetric error-correction model (retail margin vs lagged wholesale), RVP season dummy, forward selection by CV
- EIA API (inventories, refinery utilization) — needs free EIA_API_KEY
- Backtest model probabilities vs historical Kalshi prices (candlesticks)
- Notebook 02 — futures-settled markets (WTI, gold, metals): forecast volatility (GARCH, OVX/GVZ), compare Kalshi ladder vs options-implied distribution
- Notebook 02 — AAA Florida gas daily (KXAAAGASDFL) and other liquid states
- Metals (gold, silver, copper monthly)
- Polymarket equivalents

## Done
- Repo created
- Kalshi commodity market scan (see CLAUDE.md)
