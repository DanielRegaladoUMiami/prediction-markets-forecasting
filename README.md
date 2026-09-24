# Prediction Markets Forecasting

Forecasting prediction markets (Kalshi, Polymarket, Robinhood) where public daily/weekly data leads the
official resolution source — starting with commodities (AAA retail gas, energy, metals). One notebook per market.

## Approach

Each notebook follows the pipeline from
[product-sales-forecasting](https://github.com/DanielRegaladoUMiami/product-sales-forecasting):
cleaning → EDA → diagnostics → a 14-model tournament (SARIMAX, Log-SARIMAX, ETS, 7 Nixtla models,
Gradient Boosting, Random Forest, Seasonal Naive, Drift) with expanding-window CV → ensemble → critique.

Adaptations for betting markets: contract-matched frequency and horizon, exogenous drivers
(e.g. RBOB futures for retail gas), and a final step that turns the forecast distribution into bracket
probabilities and compares them with the market price net of fees.

## Markets

| # | Market | Kalshi series | Status |
|---|--------|---------------|--------|
| 00 | Kalshi commodity market scan | all | done |
| 01 | AAA US regular gas — daily / weekly / monthly | `KXAAAGASD` / `KXAAAGASW` / `KXAAAGASM` | v1 — beats random walk; not yet sharper than the market |

## How to run

```bash
uv sync
```

## License

Apache 2.0
