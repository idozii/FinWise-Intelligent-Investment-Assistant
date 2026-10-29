# W8 — FinWise public UI and Render deployment runbook

**State:** Release candidate only. Not a published website, production security assessment, or a fully independent R10 research test.

## Architecture and public safety

- A single FastAPI Render Web Service serves the research landing page, static Market Explorer, JavaScript, CSS and optional read-only offline evidence APIs. **No separate frontend deployment/CORS** is required.
- Public / redirects to /research.html when FINWISE_PUBLIC_LANDING=true.
- The Market Explorer remains at /index.html and now places the market workspace ahead of the detailed offline evidence library.
- FINWISE_ENABLE_LIVE_MARKET=false blocks all four stock/crypto price-history and forecast endpoints with HTTP 503 **before calling Yahoo or CoinGecko**. UI describes the public research preview instead of showing a broken chart. Existing local mode remains functional.
- Research evidence JSON endpoints are optional. With no vetted aggregate files they show an unavailable status rather than fabricated verified metrics.
- Public research landing includes three clearly qualified historical M5-v2 highlights (0/8 Ridge point-estimate wins; 8/8 descriptive confidence intervals containing zero; 2/8 Ridge cells with >=90% empirical ready-origin coverage). They are **NOT live market performance** or proof of algorithmic novelty.

## Preferred path: upgrade the EXISTING FinWise Render Web Service

The connected Render account **already has** a FinWise Docker Web Service:

- URL: https://finwise-intelligent-investment-assistant.onrender.com
- Dashboard: https://dashboard.render.com/web/srv-d793sgudqaus73dbv0ig
- Repository: idozii/FinWise-Intelligent-Investment-Assistant
- Deployed branch: main
- Runtime: Docker; region Singapore; Free plan; auto-deploy on commits.

**Do not create a duplicate service** unless you specifically want a temporary
staging domain. The render.yaml Blueprint below is an **optional new staging
service**, NOT the default route for the existing site.

The W8 Dockerfile now copies ONLY backend and frontend, without the old
COPY data step, uses one Uvicorn worker, and defaults to safe public mode:
FINWISE_PUBLIC_LANDING=true and FINWISE_ENABLE_LIVE_MARKET=false.
The .dockerignore prevents market CSVs/research artifacts/secrets from
entering the build context.

**Recommended release procedure:**

1. Review W8 Draft PR and wait for green CI; inspect the screenshot locally
   using the public-preview flags (instructions below).
2. Once you approve it, merge the W8/W6 feature branch changes into main
   **in proper dependency order** or create a reviewed release PR from main.
   GitHub main is protected from unreviewed automatic merges.
3. The existing Render service has auto-deploy on main. Render will build the
   new Dockerfile and redeploy this SAME service/URL. Alternatively use
   Deploy > Manual Deploy after the merge, if the source branch is already main.
4. In [its Render Environment settings](https://dashboard.render.com/web/srv-d793sgudqaus73dbv0ig),
   confirm FINWISE_PUBLIC_LANDING=true,
   FINWISE_ENABLE_LIVE_MARKET=false and do not insert Tiingo secrets.
   Existing environment variables can override Docker defaults.
5. Before sharing the URL widely, check /research.html, /index.html, /ready,
   /api/public-config, and that the market forecast API returns 503 with the
   preview notice rather than making Yahoo requests.
6. Do **not** assume CI alone verifies a completed Render deployment.
   Verify the deployed commit SHA, HTTP responses and browser UI after
   production redeploy.

If you need an entirely separate staging site without touching production,
you *can* use the optional Render Blueprint instructions that follow. It
will create an additional Free service and consume workspace free-instance
hours; avoid this unless staging isolation is worthwhile.

## Optional: NEW Render staging preview (not preferred)

This branch includes a root render.yaml that defines ONE service:

| Field | Value |
|---|---|
| Service | finwise-research-preview |
| Type | Web Service, Python runtime |
| Region | Singapore |
| Branch | feat/w8-public-ui-render-release |
| Plan | Free (preview only; cold starts after idle) |
| Python | 3.11.11 |
| Build | pip install --no-cache-dir -r backend/requirements.txt |
| Start | uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1 |
| Health | /ready |
| FINWISE_PUBLIC_LANDING | true |
| FINWISE_ENABLE_LIVE_MARKET | false |
| FINWISE_DATA_DIR | /tmp/finwise-unlicensed-data |

1. First review the GitHub PR, its CI, and license/privacy checklist; do not merge directly to main.
2. Open https://dashboard.render.com and connect your GitHub account if needed.
3. Select **New → Blueprint**, choose the repository idozii/FinWise-Intelligent-Investment-Assistant and branch feat/w8-public-ui-render-release.
4. Review the one Web Service and all environment variables; do **not** add or publish Tiingo/Yahoo tokens or any unlicensed raw price CSV.
5. Approve creating the Render service. Once deployed, Render supplies your actual onrender.com URL. Use that exact URL; do not guess a hostname.
6. Check root, /research.html, /index.html, /health, /ready, /api/public-config. Forecast endpoints must return HTTP **503** in this preview; no upstream market vendor calls should happen.
7. To remove Free cold starts for a recruiter-focused demo, upgrade the service's compute plan if you agree to the cost, or use a separate static-only landing with no dynamic research API.

Alternatively, select **New → Web Service** and manually copy the build, start, branch and environment fields above. The Render service must bind to 0.0.0.0 and $PORT, not localhost.

Render documentation:
- https://render.com/docs/deploy-fastapi
- https://render.com/docs/blueprint-spec
- https://render.com/docs/free
- https://render.com/docs/python-version

Render's Free Python services spin down after 15 minutes idle, have temporary filesystem state, and are meant for prototypes rather than traffic-sensitive production. The runtime is pinned because Render's default Python 3.14.8 is newer than this project's verified Python 3.11 runtime.

## Local public-preview smoke test

From repository root:

~~~bash
git fetch origin
git switch feat/w8-public-ui-render-release
git pull --ff-only origin feat/w8-public-ui-render-release
python -m pip install -r backend/requirements.txt 'httpx>=0.27,<1' 'pytest>=8,<10'
python -m pytest -q tests/test_w8_public_release.py tests/test_w6_portfolio.py
FINWISE_PUBLIC_LANDING=true FINWISE_ENABLE_LIVE_MARKET=false \
  uvicorn backend.app:app --host 127.0.0.1 --port 8000
~~~

In a new terminal (with Uvicorn still running):

~~~bash
curl -i http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/api/public-config
curl -s http://127.0.0.1:8000/ready
curl -i 'http://127.0.0.1:8000/api/stocks/AAPL/forecast?days=7&model=arima'
~~~

Expected: root 307 redirect to /research.html; config marketEnabled false; readiness frontendAssetsPresent true; market forecast 503 and no Yahoo/CoinGecko request.

For local development with live data as before, unset the two FINWISE flags and use the original Uvicorn launch command. **That local mode is not safe to expose publicly without further controls.**

## Evidence licensing and research trust

The repository ignores reports/backtest/ and raw market vendor CSVs. A Render Free instance will not have your machine's local R11 figures, source CSVs or aggregate evidence automatically. That is intentional. The website shows research methodology and first-run conclusions without claiming to reproduce the live API.

Only after reviewing the vendor's permission to redistribute *derived results* should a public evidence bundle be deliberately versioned and deployed with:
- w2_research_evidence.json (first M5-v2 eight-group retrospective audit);
- w3_ridge_explanations.json (offline M4.1 development model decomposition);
- w4_reliability_policies.json (R9 160 development policy cells).

**Never commit API tokens, publisher-restricted vendor bars, original Tiingo snapshots, participant financial records or other unlicensed datasets.** Preserve the original audit manifests and use a new version for any corrected result.

## Production upgrades AFTER the research preview

Before enabling FINWISE_ENABLE_LIVE_MARKET=true:
- Verify current vendor access, redistribution and automated-fetch Terms of Service.
- Add per-IP and per-service rate limits, CPU and concurrency budgets, request coalescing, explicit timeouts and stale-result TTL cache.
- Prevent users from causing expensive anonymous ARIMA runs on each request; consider a scheduled, permitted precompute of a small fixed stock universe instead.
- Configure trusted proxy/HTTPS, CORS and allowed hosts, tighten symbol/model filters, monitor failures, pin dependencies and scan vulnerabilities.
- Check keyboard navigation, mobile tables, resilience to CDN blockers, loading and error states.
- Review deployment-region privacy and only store user information if necessary.

A deployment is **not** created merely by this branch or the blueprint; the repository owner must connect GitHub and approve the new public service.
