# M5-v2 — Tiingo provider amendment to blocked Yahoo-v1

## Why this exists

On 2026-10-28, the Yahoo-v1 experiment's first reserved asset (GLD)
failed with `YFRateLimitError: Too Many Requests` even after using a
no-retry, resumable downloader. No GLD/IWM/XLE/XLV forecast outcomes
were inspected by that route, according to the user-provided logs.
**Do not alter or delete the M5-v1 protocol or rewrite its history.**

M5-v2 changes **only the source vendor** to the authenticated
Tiingo EOD API, **using its raw close field, not adjClose**.
The reserved assets, date cutoff (all forecast targets at or before
2026-10-27), models, features, horizons, calibration and acceptance
thresholds remain identical. This is a distinctly labeled, separately
hashed research run, not a clandestine replacement or a prospective
temporal holdout.

Spec: `research/protocols/m5_tiingo_external_assets_v2.json`.
Pinned SHA-256: `7c844332597962da3fa1fb68eba776279d695a19fe991cd545934fd3347f274a`.

**Important distinctions:**
- Tiingo raw Close is not necessarily byte-for-byte equivalent to Yahoo
  unadjusted Close. Results from M5-v2 should not be pooled with Yahoo-v1
  as if they shared identical data lineage.
- The four assets are held out from previous model comparisons. Their
  historical dates overlap explored assets. External-asset replication
  is not an untouched future market-event evaluation.
- The token belongs to the user. Do not paste it into GitHub, chat or logs.
  No payment is required by FinWise; check current provider entitlements
  and licensing before using the account. Data is for local research;
  don't publish Tiingo raw bars without appropriate redistribution rights.
- Do not run model-evaluation commands until preflight and the user's
  explicit unused-asset attestation have been checked.

## Step A — obtain Tiingo token (local only)

Register/log in at https://www.tiingo.com/ and view the API token from the
developer documentation or dashboard. In a terminal on your own machine:

```bash
read -rs -p 'Tiingo API token: ' TIINGO_API_KEY; echo
export TIINGO_API_KEY
```

This prompts for the token without echoing it or putting its value into
shell history. Keep this terminal session open for the downloader.

## Step B — download the *separate* Tiingo-v2 dataset snapshot

```bash
python -m scripts.m5_download_tiingo
```

The downloader requests **2021-10-27 to 2026-10-27**, one ticker at a
time, and saves in `data/processed/m5_tiingo_v2/`.
It uses `Authorization: Token ...` in an HTTP request header,
not a URL query parameter. Each saved CSV has a companion
`.csv.source.json` with provider ID, date window, raw price
definition, row count and SHA-256. The token is not recorded.
Existing files are never overwritten, and rate limits halt the batch.

If Tiingo returns 401/403, verify the token and subscription plan.
If it returns 429, stop requests and respect the provider's quota.

## Step C — preflight ONLY (no Close values read by evaluator)

```bash
python -m research.frozen_eval_tiingo preflight \
  --assets GLD=data/processed/m5_tiingo_v2/GLD_daily_processed.csv \
           IWM=data/processed/m5_tiingo_v2/IWM_daily_processed.csv \
           XLE=data/processed/m5_tiingo_v2/XLE_daily_processed.csv \
           XLV=data/processed/m5_tiingo_v2/XLV_daily_processed.csv \
  --output-dir reports/backtest/m5_tiingo_v2
```

Send the preflight counts/hashes to the researcher. Preflight validates
sidecar provenance, hash lock, presence/sorted dates and enough evaluation
origins without computing predictive performance.

## Step D — gated model evaluation (DO NOT run yet)

After explicit authorization and confirming that GLD/IWM/XLE/XLV have
never been used for FinWise model selection, the researcher can run the
same CLI with `evaluate` and both
`--attest-unused-assets --authorize-outcome-review` flags.

No success claim is justified until preflight and actual evaluation run.
A failed source download, unverified source license, malformed data or
changed snapshot should remain blocked and documented as such.

Original Yahoo-v1 files/protocol can continue independently if Yahoo
later resumes service. Do not use Tiingo files with the Yahoo-v1
`research.frozen_eval` command.
