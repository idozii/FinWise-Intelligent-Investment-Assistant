# FinWise M4 — transparency dashboard (offline)

The M4 renderer produces one standalone HTML file from **real M2.1 and M3
evaluation outputs**. It intentionally does not fabricate equity charts,
risk scores, portfolio allocations or model feature attributions.

## Run locally

After running M2.1 and M3 with matching input data and historical horizons:

```bash
python -m research.dashboard \
  --robustness-dir reports/backtest/m21_multiasset \
  --uncertainty-dir reports/backtest/m3_intervals \
  --output reports/backtest/finwise_research_dashboard.html
```

Open `reports/backtest/finwise_research_dashboard.html` in your browser.
No server or API credentials are needed.

## Visible research information

- Asset and horizon selectors, matched-origin price MAE and RMSE data
- Naive baseline vs challenger error levels
- Descriptive paired-bootstrap confidence intervals for MAE differences
- Historical volatility-regime error tables
- Past-error-calibrated empirical interval coverage, interval readiness and
  accepted proportion
- SHA-256 fingerprints of inputs and clear exploratory status
- Documented feature families (not SHAP or causal importance)

**Safeguards:** No remote assets/scripts, no market data upload, no arbitrary
HTML injection from asset labels, no invented stock performance. This dashboard
has *no* online brokerage actions, personalized finance recommendations or
validated calibration guarantees. It is a decision-support **research report**.

For any public portfolio/demo, share the screenshot and written protocol;
only publish datasets and artifact contents when redistribution rights permit.
Please avoid marketing statements like "90% accurate prices" or "validated
stock recommendations." SHAP, scenario stress tests and production API
integration are separate future work, not shipped M4 features.
