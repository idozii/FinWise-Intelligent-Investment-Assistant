# FinWise — Evidence-Centered Financial Forecasting Research

**Research portfolio and educational market-data dashboard.** FinWise asks whether a more complex forecaster beats a persistent last-close baseline, how reliable empirical prediction intervals are, and what happens when a prediction policy abstains.

**Scientific boundary:** the public market dashboard's ARIMA, Holt-Winters, linear-trend and moving-average forecasts are **illustrative, not empirically calibrated**. Offline Ridge/conformal experiments do **not** validate live forecasting or any trading strategy. No investment advice, profit or theoretical coverage guarantees are claimed.

## Interfaces and technology

| Component | Implemented |
|---|---|
| API | FastAPI, Python |
| Market UI | HTML/CSS/JS, Chart.js, Plotly, stock and crypto history |
| Legacy live forecasts | ARIMA, Holt-Winters, Linear Trend, Moving Average |
| Research panels | Historical external-asset benchmark, Ridge model-internal contributions, development risk–availability policies |
| Research portfolio | Standalone `/research.html` with methods, results and R10 gate |
| Local scientific evidence | R6 integrity audit, R8 delayed-feedback calibration, R9 abstention sensitivity, R10 research gate, R11 manuscript |
| Provenance | SHA-256 manifests, frozen source protocols, deterministic CI tests |

**Not currently in the production API:** a validated live Ridge model, online conformal guarantee, trading execution, portfolio advice or real-time market-risk protection.

## Start locally

From repository root:

```bash
python -m pip install -r backend/requirements.txt -r research/requirements-ml.txt \
  'httpx>=0.27,<1' 'pytest>=8,<10'
uvicorn backend.app:app --reload --port 8000
```

Open:
- [Research portfolio](http://127.0.0.1:8000/research.html)
- [Market dashboard](http://127.0.0.1:8000/)
- [API capabilities](http://127.0.0.1:8000/api/capabilities)
- [Operational readiness](http://127.0.0.1:8000/ready)

**Optional research exports:** the portfolio page renders historical metrics only if the related audited offline JSON files exist. If absent, it displays a clear unavailable state—no hard-coded optimistic results.

```bash
# Requires the original, unmodified local Tiingo M5-v2 sources and first-run outputs.
python -m research.w2_evidence_export \
  --result-dir reports/backtest/m5_tiingo_v2 \
  --output reports/backtest/w2_research_evidence.json

# Requires local R8/R9 development artifacts and matching original source hashes.
python -m research.w4_policy_export \
  --r8-dir reports/backtest/r8_development \
  --r9-dir reports/backtest/r9_risk_coverage_dev \
  --dev-predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --output reports/backtest/w4_reliability_policies.json

# Optional M4.1 exploratory Ridge explanations, with SHA checks.
python -m research.w3_attribution_export \
  --attributions-dir reports/backtest/m41_explanations \
  --output reports/backtest/w3_ridge_explanations.json
```

If you generated these files on another Git branch, they remain on your local disk unless moved or overwritten. Do not commit vendor-licensed raw prices, API tokens or source data snapshots to this public repository.

## What the research actually found

The M5-v2 first retrospective external-asset evaluation studied GLD, IWM, XLE and XLV with 64 eligible origins per asset/horizon, H1 and H7. Ridge's observed MAE was higher than persistence in **all 8/8** comparisons; **all 8/8** descriptive paired bootstrap intervals included zero. Nominal 90% Ridge ready-origin coverage met/exceeded 90% in **2/8** cells. These are historical/descriptive findings, not prospective confirmation.

The R6 audit also documented an official XLE 2-for-1 share split effective 5 December 2025, which is crossed by a large-error forecast; raw Close is not a split-neutral economic return. The original results were not modified.

On *development-only* AAPL/MSFT/SPY/TSLA origins, R8's delayed-feedback ACI-style calibration showed higher ready-origin coverage in 14/16 groups and wider mean relative intervals in 16/16. R9 demonstrated that a 10% width cap can sharply reduce acceptance; on MSFT Ridge H7, adaptive accepted 17/143 vs 40/143 for fixed rolling and selected relative MAE was 3.16% vs 2.90% on different origin subsets. This is **not** evidence of an independently superior method.

**R10 remains blocked:** a genuinely unused dataset, point-in-time split/dividend-safe target, data rights, frozen policy, and precommitted statistical analysis are required before new independent testing.

## Research docs

- [First M5-v2 evidence audit](research/M51_FIRST_RUN_EVIDENCE.md)
- [XLE corporate-action case study](research/R6_CORPORATE_ACTIONS.md)
- [R7 economic estimand gate](research/R7_PRICE_ESTIMAND_PROTOCOL.md)
- [R8 exploratory adaptive calibration](research/R8_ADAPTIVE_DEVELOPMENT.md)
- [R9 selective risk / availability](research/R9_SELECTIVE_RISK_DEV.md)
- [R10 independent evaluation readiness](research/R10_INDEPENDENT_EVALUATION_GATE.md)
- [R11 manuscript draft](research/paper/R11_MANUSCRIPT_DRAFT.md)

The paper draft is not submitted, accepted or a proven novel algorithm.

## Deployment, data rights and security

Use `FINWISE_DATA_DIR` to point to local market data outside the Git working tree. Do not publish provider tokens or redistribute raw market bars without confirming the applicable license. Market provider APIs may rate-limit requests.

`/health` returns path-free liveness; `/ready` checks local frontend files with no network dependency. HTTP responses include minimal no-sniff/referrer/frame headers and `/api/` endpoints use no-store.

**Public deployment is not yet a production security certification**: rate limiting, CPU/resource controls, rate-limited vendor caching, dependency scanning, HTTPS/domain/CORS review, source licensing and operator monitoring still require deployment-specific approval. Run locally before any public release.

## Development checks

```bash
python -m pytest -q tests/test_w6_portfolio.py tests/test_w5_operational_safety.py tests/test_w51_arima_index.py
```

The repository currently uses stacked review branches for research and product milestones. Work should be reviewed and integrated in dependency order; do not assume `main` already contains every experimental branch.

**Maintainer:** [idozii](https://github.com/idozii).
