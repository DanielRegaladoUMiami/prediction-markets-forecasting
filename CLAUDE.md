# Prediction Markets Forecasting

## Goal
Forecast prediction markets (Kalshi, Polymarket, Robinhood — Robinhood lists Kalshi contracts) where public
data *leads* or is sharper than the crowd. **One system**: every market is a spec in `specs.py`, rendered from
the single `notebooks/template.ipynb` (built by `scripts/build_template.py`) via
`python -m prediction_markets_forecasting.render <key>`. Never hand-write a per-market notebook.

Market families: A weather (NWS station), B lagging statistic (AAA gas, mortgage, ERCOT), C macro release
(CPI, jobs), D futures price (WTI, gold, FX — volatility, not level).

Every notebook replicates the pipeline of Daniel's Logitech notebook
(github.com/DanielRegaladoUMiami/product-sales-forecasting, `Individual_Project_DanielRegalado.ipynb`):
Part 1 cleaning → Part 2 EDA → Part 3 diagnostics (ADF/KPSS, decomposition, ACF/PACF) → Part 4 14-model
tournament (SARIMAX + Log-SARIMAX by AIC; ETS 10 configs, 7 Nixtla models, GB/RF, Seasonal Naive/Drift by
expanding-window CV; ranking by CV RMSE; family ensemble; MAPE tiers) → Part 5 critique.

Every market is forecast at daily, weekly and monthly horizons when Kalshi lists all three.
Data quality comes first: pull the exact resolution-source series before modeling.

Agreed adaptations ("skeleton 100% + adjustments"):
1. Frequency / season_length / horizon are parameters that match the contract (W/52, D/7, MS/12; horizon = steps to resolution).
2. Exogenous drivers (e.g. RBOB, WTI) — the original notebook's own "Limitations" next step.
3. More CV folds (long histories).
4. Part 6: prediction intervals → P(bracket) → edge vs Kalshi price net of fees.
5. **Stationarity is automatic, always** — every notebook calls `prediction_markets_forecasting.stationarity.decide()`
   (Part 3.4) and every model follows its output; never hand-pick `d`/`D` or feed raw non-stationary exog levels:
   - `d`: difference until ADF rejects a unit root AND KPSS does not reject stationarity (disagree → difference)
   - `D`: one seasonal difference if STL seasonal strength > 0.64
   - Exog: Engle-Granger cointegration → ECM term + driver changes; else driver changes; stationary target → lagged levels
   - AIC only compares SARIMAX orders with the same `d`/`D`; ML predicts Δy when `d ≥ 1`

## Market findings (2026-09-23 scan of Kalshi public API)
- Retail gas (AAA) has real edge potential: retail lags wholesale (RBOB) by ~1–2 weeks.
  Volume: KXAAAGASW (weekly, ~266K), KXAAAGASM (monthly, ~237K), KXAAAGASD (daily, ~175K), KXAAAGASDFL (Florida).
- Futures-settled markets (KXWTI, KXNATGASW, KXBRENTMON, KXCOPPERMON, KXGOLDMON; Pyth/ICE) are near random walks —
  expect Naive to win; useful as a control, not as an edge.
- Kalshi API v2 volume fields are strings: `volume_fp`, `volume_24h_fp`, `open_interest_fp`, `last_price_dollars`.
- Kalshi retains limited price history → record market prices forward from day one.
- `/historical/markets` returns legacy tickers (`AAAGASD-23OCT03-US`) and even foreign events (monthly `AAAGASM-*`
  inside the weekly series) — always filter events by series prefix. Settled markets carry `expiration_value`
  (exact resolution value); some are non-numeric ("No").
- AAA vs EIA weekly (FRED `GASREGW`): change corr 0.997, AAA ≈ EIA + 1.5¢ — valid proxy for long history.
- Weather: Kalshi `KXHIGHMIA` settlement = NOAA GHCN TMAX at USW00012839 on 99.6% of 1,152 days. Day-before
  model forecasts from Open-Meteo Previous Runs API (GFS from 2021, ECMWF/ICON/GEM from 2024); daily max taken
  in `Etc/GMT+5` (NWS climate day = local *standard* time). Raw model maxima run several °F cold at MIA.
- Kalshi ladders mix `greater` / `less` / `between` strikes (`floor_strike`, `cap_strike`); integer
  underlyings need a continuity correction (`kalshi.bracket_bounds`).
- The market is much sharper than public-data models on the daily AAA contract (σ ~0.7¢ vs ~1.6¢): traders see
  live station prices. Realistic edge is weekly/monthly early in the period, not daily.

## Stack
- Python 3.11+
- uv (build/dep management)
- ruff (lint + format, pre-commit)

## Current milestone
v0.1 — one system (specs + template); 01 AAA gas (D/W/M), 02 Miami daily high temperature

## Local rules
- Conventional Commits (feat:, fix:, docs:, refactor:, chore:, test:)
- No `Co-Authored-By` in commits — sole author is Daniel
- Use `uv` not pip; `uv add <pkg>` to add deps
- Pre-commit hooks (ruff) run on every commit
- README/docs in English; conversation can be Spanish
- API keys (FRED, EIA) live in `.env` / shell profile — never pasted in chat, never committed

## Environment gotcha (iCloud)
The repo lives under ~/Desktop (iCloud-synced). iCloud hides `.pth` files (breaks the editable install) and keeps
re-materializing a cloud copy of `.venv` (a "dataless" ghost that blocks `rm`). The working venv is `.venv.nosync/`
(iCloud skips `*.nosync`). Always point uv at it:
```bash
export UV_PROJECT_ENVIRONMENT=.venv.nosync   # or prefix each command
```
In Cursor, select `.venv.nosync/bin/python` as the interpreter. Ignore any `.venv` folder iCloud creates.

## How to run
```bash
export UV_PROJECT_ENVIRONMENT=.venv.nosync
uv sync
uv run jupyter lab notebooks/
```

## Where things live
- `src/prediction_markets_forecasting/`: `specs.py` (one spec per market), `markets.py` (data builders →
  `MarketData`), `kalshi.py`, `sources.py` (FRED, Yahoo, NOAA, Open-Meteo), `selection.py` (driver discovery),
  `stationarity.py`, `tournament.py`, `pricing.py` (Part 6), `render.py` (papermill)
- Template: `scripts/build_template.py` → `notebooks/template.ipynb`; rendered: `notebooks/NN_<key>.ipynb`
- Tests: `tests/`
- Experiments log: `docs/experiments/`
- Roadmap: `ROADMAP.md`
