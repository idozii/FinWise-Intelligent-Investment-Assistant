# W5 — FinWise operational reliability and deployment safety (phase 1)

## Implemented (CI-smoke tested, not audited for public release)

- `GET /health`: minimal liveness response `{"status":"ok"}`
  without the old absolute private `dataDir` path in the HTTP body.
- `GET /ready`: checks that the static frontend entrypoint and required
  JS/CSS files exist **without** contacting Yahoo, Tiingo, CoinGecko
  or accessing the R10 data. Research JSON exports are optional and
  their absence must not be conflated with a broken web deployment.
- Common HTTP protections: `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`.
- All `/api/` endpoints use `Cache-Control: no-store` to prevent
  browser/proxy caching potentially stale data or local aggregate research
  results. Static assets retain browser caching and versioned URLs.
- No changes to forecasts, source vendors, experiment samples or frozen
  scientific artifacts.
- Deterministic FastAPI TestClient integration and regression tests.

## Run local verification

```bash
git fetch origin
git switch feat/w5-production-operational-safety
git pull --ff-only origin feat/w5-production-operational-safety

python -m pip install -r backend/requirements.txt \
  'httpx>=0.27,<1' 'pytest>=8,<10'
python -m pytest -q tests/test_w5_operational_safety.py
uvicorn backend.app:app --reload --port 8000

curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -I http://127.0.0.1:8000/api/capabilities
```

## Public-deployment gate (STILL OPEN)

This is **not** a production security certification. Prior to a
public rollout:
- Configure CORS allowlists, domain/HTTPS, resource budgets, timeout
  limits, vendor quota handling and rate-limiting. Live forecast
  fitting is CPU-costly; public anonymous access needs abuse controls.
- Separate publisher-licensed Tiingo market data and non-public
  research snapshots from the public static assets and any repository
  artifacts.
- Add structured redacted logs, cache strategy with TTL and request
  coalescing for permitted market-data endpoints (and keep Yahoo
  rate-limiting behavior ethical).
- Assess input sanitization, dependency vulnerabilities, user-supplied
  market symbol constraints, security headers/CSP, startup/deployment
  health checks and policy/version stability.
- Obtain deployment-specific authorization and test public behavior.
  No automatic external deployment or GitHub `main` merge.

**Compatibility note:** the old `GET /health` `dataDir` field is
deliberately omitted to avoid exposing absolute filesystem paths.
Any operator relying on that field should instead obtain configuration
from authorized local deployment logs, not an anonymous health route.
