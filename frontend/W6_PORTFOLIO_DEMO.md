# W6 — Public-facing portfolio page (local demo first)

The W6 research narrative lives at `frontend/research.html` and
does not alter the market forecast API. It links to the historical
research evidence cards and scientific methodology and distinguishes
M5 historical first-run evidence from R8/R9 development sensitivity.
It shows a no-data state when no audited local evidence package exists
instead of fabricating data. All dynamic text goes through safe DOM
`textContent` elements.

W6 is **stacked on W4** to preserve the latest interactive
risk–availability explorer. It also incorporates the operational
privacy/readiness headers from W5 and the ARIMA RangeIndex fix from
W5.1; the W6 PR documents this integration, since W4 and W5 are
currently on separate feature branches.

## Run locally

```bash
git fetch origin
git switch feat/w6-portfolio-research-demo
git pull --ff-only origin feat/w6-portfolio-research-demo

python -m pip install -r backend/requirements.txt -r research/requirements-ml.txt \
  'httpx>=0.27,<1' 'pytest>=8,<10'
python -m pytest -q tests/test_w6_portfolio.py \
  tests/test_w5_operational_safety.py tests/test_w51_arima_index.py

uvicorn backend.app:app --reload --port 8000
```

Open `http://127.0.0.1:8000/research.html` for the
research portfolio and `http://127.0.0.1:8000/` for
the original market dashboard. Research results appear only if local
offline-verified JSON exports exist:

- `reports/backtest/w2_research_evidence.json`
- `reports/backtest/w3_ridge_explanations.json`
- `reports/backtest/w4_reliability_policies.json`

Check endpoints `/health`, `/ready`,
`/api/capabilities`, and the three
`/api/research/...` routes.

## Public deployment still blocked by operational decisions

Do not automatically deploy/release without:
- data provider license review (including redistribution of derived data),
- API vendor quota, rate-limiting and CPU budget enforcement,
- HTTPS, CORS/allowed hosts and monitoring review,
- dependency audits, secrets management and security assessment,
- user-facing limitations and honest research evidence provenance,
- review of the PR merge chain and deployment authority.

This phase creates a **local research portfolio demo** only, not a
live hosted URL or validated financial advice app. M5 first-run and
R10 readiness constraints remain unchanged.
