# FinWise W2 — historical Research Transparency Dashboard

**Status:** implemented optional, read-only historical evidence.
Does **not** add Ridge to live ARIMA forecast API and does not validate
the current live heuristic 95% bounds.

## Architecture

- \`research/w2_evidence_export.py\` validates original Tiingo M5-v2
  price snapshot SHA-256 and the first-run prediction/interval/score
  artifacts through the M5.1 evidence audit.
- Export contains only 8 aggregate comparison rows: GLD/IWM/XLE/XLV,
  horizons 1/7. It includes MAE for Ridge/Naive, descriptive paired
  block-bootstrap CIs, ready-origin empirical coverage, acceptance,
  and matched selected MAE. No raw vendor price bars, labels, forecast
  rows, provider token or original local CSV paths are exported.
- \`backend/research_evidence.py\` parses a small local JSON file with a
  fixed allowlisted schema and rejects unrecognized/invalid results.
- \`GET /api/research/evidence\` returns only the historical aggregate
  evidence, **404 if not installed**, 422 if invalid, never fetching live
  market data or invoking the offline model test.
- \`frontend/app.js\` uses \`textContent\` to render all dynamic table
  cells, keeping the live forecast panel separate.

## Local setup

From the repository root, keep the four original Tiingo M5-v2 price
files and the first-run evidence in their original absolute paths.

\`\`\`bash
git switch feat/w2-historical-research-dashboard
python -m research.w2_evidence_export \
  --result-dir reports/backtest/m5_tiingo_v2 \
  --output reports/backtest/w2_research_evidence.json

python -m pip install -r backend/requirements.txt 'httpx>=0.27,<1' pytest
python -m pytest -q tests/test_w2_evidence.py tests/test_w2_http_smoke.py

uvicorn backend.app:app --reload --port 8000
\`\`\`

The default endpoint looks for
\`reports/backtest/w2_research_evidence.json\` under the repository root.
You can override it with \`FINWISE_RESEARCH_SUMMARY_PATH\` (local absolute
JSON file path). The package is immutable unless byte-identical;
version any corrections explicitly after their scientific review.

Open \`http://127.0.0.1:8000\` to see the M5-v2 historical panel or
\`http://127.0.0.1:8000/api/research/evidence\` for the JSON.
The page intentionally shows a no-data state when the exported
artifact is absent rather than inventing benchmark numbers.

## Interpretation constraints

- The eight original Ridge−Naive MAE differences are **positive**,
  meaning no observed Ridge improvement over Naive in those cells;
  all eight descriptive bootstrap 95% CIs include zero.
- Ridge empirical nominal-90% coverage is ≥90% in 2/8 cells;
  these are 64-origin time-dependent estimates, not guarantees.
- XLE H7 has **0/64** Ridge accepted forecasts, and the raw price
  target crosses a split with known announcement timing.
- Selected MAE on jointly accepted origins is conditional and can
  look different from all-origin MAE; no unconditional superiority claim.
- Data was from a *retrospective external asset* study; not live
  model performance and not an independent prospective market test.
- Vendor redistribution and hosting permissions must be checked
  before making any public research dataset or derived data available;
  this feature defaults to locally generated **aggregates only**.

## Future W3/W4 gates

W3 can display only validated model contribution decompositions with
exact reconstruction checks and no causal claims. W4 cannot reuse
these original M5 outcomes to claim coverage of a modified live
selective-forecast policy; requires an entirely separate protocol and
authorized fresh evaluation.
