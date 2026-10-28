# W1 — FinWise web architecture and API contract v1.1

## As-is architecture

Backend: `backend/app.py` FastAPI; stock and crypto
history + legacy ARIMA/Holt-Winters/linear/moving-average forecasts.
Frontend: static `frontend/index.html`, `app.js`,
`styles.css`, Chart.js and Plotly. Production mounts the
frontend at the same FastAPI origin. The offline M1–M5.1 research pipeline
**is not integrated into the live inference API**.

Do not describe existing live ARIMA forecast bounds as "calibrated 95%
coverage". The legacy API labels `lower95` and
`upper95` have been retained for backwards compatibility,
but this milestone makes their missing calibration evidence explicit.

## Implemented contracts

- `GET /api/capabilities` is a no-network static
  capability handshake with immutable `contractVersion` value
  `finwise-web-api-v1.1`.
- `forecastReliability.legacyIntervalsEmpiricallyValidated=false`.
  W1 refuses model-registry drift and does not claim Ridge/conformal
  integration, live validated uncertainty or investment advice.
- Existing `/api/stocks/{symbol}/forecast` and
  `/api/crypto/{coin}/forecast` responses retain every prior field
  and add `metadata` with unvalidated interval semantics,
  adjustment provenance limitation, and horizon date assumptions.
- Equities use weekday future dates, **not** an exchange-holiday-aware
  trading calendar. Crypto uses daily calendar dates.
- CORS is restricted to explicit local-development origins or
  `FINWISE_CORS_ORIGINS`, and no credentialed wildcard is used.
- UI shows capability availability and an explicit reliability warning
  without attempting to claim actual empirical interval coverage.

## Manual smoke commands

```bash
python -m pip install -r backend/requirements.txt 'httpx>=0.27,<1' pytest
python -m pytest -q tests/test_w1_contracts.py tests/test_w1_api_smoke.py

# In a separate shell from repo root:
uvicorn backend.app:app --reload --port 8000

# Then:
curl -fsS http://127.0.0.1:8000/api/capabilities
curl -fsS http://127.0.0.1:8000/api/forecast-models
```

Run the web at `http://127.0.0.1:8000` for same-origin
production-like development, or run the separate frontend static server
on port 5500. No market download is made by the capabilities route.
Stock history/forecast routes may fetch data if local snapshots are
missing; do not use them as implicit holdout-data tests.

## Contract evolution gates for W2 and W4

- W2 may expose **explicitly historical** M2–M5 research summaries
  through a separate read-only endpoint only when outputs/provenance
  and appropriate display rights are present. Never blend this with
  the live forecast endpoint as if it were a live performance guarantee.
- W4 may add real calibration methods and an abstention status only
  after independent testing and research approval; build a **new**
  versioned response contract rather than relabeling legacy heuristics.
- Add typed schemas, caching/source metadata and API threat-model
  review before public production deployment. W1 is an integration
  contract milestone, **not a production security certification**.

## W1 acceptance criteria

Stable v1.1 capabilities handshake, backward-compatible additive
metadata on both live forecast routes, UI uncertainty disclosure,
explicit CORS allowlist, deterministic contract tests and HTTP
smoke tests on both Python 3.11 and 3.12. No changes to M5 outcomes.
