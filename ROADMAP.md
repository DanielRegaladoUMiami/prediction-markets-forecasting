# Roadmap — Prediction Markets Forecasting

## Current milestone: v0.1
- [x] Initial scaffold
- [x] Kalshi market scan (00)
- [x] AAA gas daily/weekly/monthly (01): AAA↔EIA bridge, drivers, tournament, Kalshi translation
- [x] Automatic stationarity module (ADF+KPSS d, STL D, Engle-Granger → ECM), with tests
- [x] One system: specs + market builders + tournament + pricing in `src/`, single template, papermill render
- [x] Daily high temperature — Miami (02): NOAA↔Kalshi bridge (99.6% exact), day-before weather-model forecasts
- [ ] Forward recorder for Kalshi prices (Part 6 snapshots each run to data/kalshi_snapshots/; needs a schedule)

## Next up
- More temperature cities as specs (NYC, LA, Chicago, Atlanta, SF…) — verify each series' station first
- Backtest model probabilities vs historical Kalshi prices (candlesticks) — the only proof of edge
- Weather: same-day model runs (previous_day0 / HRRR) and live METAR for morning bets
- Gas v2: asymmetric ECM (rockets & feathers), RVP season dummy, EIA inventories (free EIA_API_KEY)
- AAA gas by state (KXAAAGASD<ST>) — history from EIA state series where available
- Family C: CPI nowcast; Family D: volatility pricing (GARCH, OVX/GVZ) for WTI / gold
- Polymarket equivalents

## Done
- Repo created
