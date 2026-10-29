# FinWise W4 — Historical reliability–availability policy explorer

**Scientific status: development-only; NOT validated live forecasting.**

## What W4 implements

The W4 panel is a reproducible, interactive companion to R9. A user
can explore how the fraction of forecasts emitted and conditional
absolute relative error change when the **maximum interval width**
is set to one of 5%, 10%, 15%, 20% or 30%. Controls include four
**development** assets (AAPL/MSFT/SPY/TSLA), naive or Ridge, and H1/H7.
For each combination, show:
- Fixed rolling and delayed-feedback ACI-style methods side by side.
- Accepted forecast count, divided by **all** forecast origins and
  separately by calibrated/ready origins.
- Empirical interval coverage among all calibrated origins and separately
  among accepted origins; these are descriptive, not guarantees.
- Mean selected relative absolute error, in percent of last observed
  Close, and a risk–availability scatter curve across thresholds.
- "NO FORECAST ACCEPTED" with **N/A** risk, never "0 error".
- A small-sample warning when fewer than 30 origins are selected.

The visualization uses research CSV artifacts **only after** a source
fingerprint check and full R9 arithmetic recomputation. No market bars
or original target/predicted rows are published in the API summary.
It does not retrain the forecaster and cannot turn these development
comparisons into a prospective independent evaluation.

**Selection changes the sample**. A smaller conditional MAE in one
threshold/method may simply reflect different accepted origins. It
is NOT proof that the method improved the underlying predictions.
Coverage on selected origins is not guaranteed to be 90% by this
procedure.

## Local steps

The R8 development CSV must already exist, and the source M2.1 CSV
must be identical to the one recorded in R8's SHA-256 manifest.

```bash
git fetch origin
git switch feat/w4-risk-availability-ux-stacked
git pull --ff-only origin feat/w4-risk-availability-ux-stacked

# Step 1: R9 exploratory diagnostics (once)
python -m research.r9_selective_risk \
  --r8-dir reports/backtest/r8_development \
  --dev-predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --output-dir reports/backtest/r9_risk_coverage_dev

# Step 2: verify every R9 aggregate against the original R8 inputs
python -m research.w4_policy_export \
  --r8-dir reports/backtest/r8_development \
  --r9-dir reports/backtest/r9_risk_coverage_dev \
  --dev-predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --output reports/backtest/w4_reliability_policies.json

# Step 3: web tests and local server
python -m pip install -r backend/requirements.txt -r research/requirements-ml.txt \
  'httpx>=0.27,<1' pytest
python -m pytest -q tests/test_r9_selective_risk.py tests/test_w4_policies.py tests/test_w4_http_smoke.py
uvicorn backend.app:app --reload --port 8000
```

Open `http://127.0.0.1:8000`. The W4 panel is titled
**Selective Forecasting · Development Policy Explorer**.
Endpoint: `GET /api/research/reliability-policies`. If the
local JSON is missing it returns 404; invalid/unsupported exports
receive 422 rather than invented data.

To refresh static assets after switching branches, force reload
(`Ctrl+Shift+R`). W4 version-tags both frontend JavaScript and CSS.

## Interpretation and public demo restrictions

- W1's **live** ARIMA/Holt-Winters heuristic bounds remain explicitly
  unvalidated, with no 95% coverage or safe-investment claims.
- M5-v2/GLD/IWM/XLE/XLV remain untouched, excluded from all R9/W4
  development policy selection. A newly chosen width cutoff requires
  its own frozen, genuinely untouched test before evidence-based claims.
- R8's ACI-style code is not a licensed or theorem-equivalent
  implementation of a published multi-step adaptive coverage guarantee.
- Current R7 independent-evaluation gate is BLOCKED, so W4 ships
  a **research exploration UI**, not a production decision engine.
- Investigating XLE split errors in the original M5 first run
  cannot justify silently replacing its metrics.
