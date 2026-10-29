# W3 — Explainable Ridge (historical development only)

W3 provides a **read-only** summary of M4.1's exact additive
`StandardScaler + Ridge(alpha=10)` model decomposition. Feature
contributions sum with the intercept to the **predicted log return**.
They are not SHAP values, causal effects, investment returns, or
explanations of the live ARIMA model.

The original M4.1 run explains **32 origins per asset/horizon** on
AAPL/MSFT/SPY/TSLA at horizons 1 and 7 (256 origin/model forecasts).
W3 refuses files that contain M5 holdout asset data or inconsistent
model reconstruction checks. Only top six mean-absolute-contribution
features per asset/horizon are exported, without original market bars.

## Generate the optional local export

If your M4.1 artifacts already exist:

```bash
git switch feat/w3-ridge-explainability-ui
python -m research.w3_attribution_export \
  --attributions-dir reports/backtest/m41_explanations \
  --output reports/backtest/w3_ridge_explanations.json
```

The exporter verifies hashes of the original local input CSVs as recorded
by M4.1, all 256 reconstructed forecasts and every group of feature
summaries. If one source has changed, do not bypass the verification:
restore the original source snapshot or describe a newly versioned
exploratory attribution run.

```bash
python -m pip install -r backend/requirements.txt -r research/requirements-ml.txt \
  'httpx>=0.27,<1' pytest
python -m pytest -q tests/test_w3_explanations.py tests/test_w3_http_smoke.py
uvicorn backend.app:app --reload --port 8000
```

Open `http://127.0.0.1:8000`. The optional explanation panel
supports development asset/horizon selection and displays the six
largest mean absolute log-return contributions. The endpoint
`GET /api/research/explanations` returns 404 when no export is
available, 422 on invalid data, and **never** fits the model online.

W3 also version-tags frontend JS/CSS files to avoid the stale-style
problem seen in the user's W2 screenshot. If visual styles still look
stale, force reload with Ctrl+Shift+R and inspect the loaded CSS.

W3 is not a confirmation of Ridge accuracy. The first-run M5-v2
retrospective evaluation found no Ridge MAE advantage in 8/8 cells;
the model explanation only describes its own calculations.
