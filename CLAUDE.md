# Prediction Markets Forecasting

## Goal
Forecast prediction markets (Kalshi, Polymarket, Robinhood — Robinhood lists Kalshi contracts) where public
daily/weekly data *leads* the official resolution source. Start with commodities (AAA retail gas, energy,
metals). One notebook per market.

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

## Market findings (2026-09-23 scan of Kalshi public API)
- Retail gas (AAA) has real edge potential: retail lags wholesale (RBOB) by ~1–2 weeks.
  Volume: KXAAAGASW (weekly, ~266K), KXAAAGASM (monthly, ~237K), KXAAAGASD (daily, ~175K), KXAAAGASDFL (Florida).
- Futures-settled markets (KXWTI, KXNATGASW, KXBRENTMON, KXCOPPERMON, KXGOLDMON; Pyth/ICE) are near random walks —
  expect Naive to win; useful as a control, not as an edge.
- Kalshi API v2 volume fields are strings: `volume_fp`, `volume_24h_fp`, `open_interest_fp`, `last_price_dollars`.
- Kalshi retains limited price history → record market prices forward from day one.

## Stack
- Python 3.11+
- uv (build/dep management)
- ruff (lint + format, pre-commit)

## Current milestone
v0.1 — scaffold + notebook 01 (AAA US gas: daily KXAAAGASD, weekly KXAAAGASW, monthly KXAAAGASM)

## Local rules
- Conventional Commits (feat:, fix:, docs:, refactor:, chore:, test:)
- No `Co-Authored-By` in commits — sole author is Daniel
- Use `uv` not pip; `uv add <pkg>` to add deps
- Pre-commit hooks (ruff) run on every commit
- README/docs in English; conversation can be Spanish
- API keys (FRED, EIA) live in `.env` / shell profile — never pasted in chat, never committed

## How to run
```bash
uv sync
uv run jupyter lab notebooks/
```

## Where things live
- Source (shared data loaders, model tournament): `src/prediction_markets_forecasting/`
- Notebooks (one per market): `notebooks/NN_<market>.ipynb`
- Tests: `tests/`
- Experiments log: `docs/experiments/`
- Roadmap: `ROADMAP.md`
