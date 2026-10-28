# FinWise M5 — Frozen external-asset validation protocol

## Stage and scientific status

M1–M4 examined AAPL, MSFT, TSLA, SPY. M3.1 showed the Ridge model has
higher MAE than naive at **all seven jointly accepted, comparable
asset/horizon combinations**; at TSLA H7, no forecast is jointly accepted.
M3 nominal 90% intervals showed empirical ready-origin coverage below 90%
in every sampled asset/horizon. None of these is a prospective result.

M5 reserves **GLD, IWM, XLE and XLV**, without inspecting their outcomes or
choosing hyperparameters based on their performance. The design is committed
to research/protocols/m5_external_assets_v1.json. Its exact SHA-256 is
`0d64713935f95c46b30317d705e3fc495bc6f40eda3bf0d81115ff536330c0ce`.
Changing this protocol requires an openly documented amendment and new
version, not silently replacing the existing one.

**Design:** retrospectively evaluate new assets in the **predeclared
forecast-origin range 2025-01-01 through 2026-10-27** with targets also
inside that cutoff. Models: naive and Ridge(alpha=10); horizons 1 and 7;
min_train 252; step 7; fixed M2 features and prequential matured-label
refits. Empirical interval diagnostics use alpha .1, 30-error warmup,
60-error rolling window and a 10% width-based accept rule. Absolute price
MAE, paired bootstrap differences, coverage and jointly accepted MAE are
reported *for every* asset/horizon; all comparisons are descriptive.
Unadjusted Close convention is fixed as in M1–M4, with documented split
and dividend limitations. No transaction-cost or profit claims.

This is **NOT a prospective temporal holdout**: historical market events
overlap explored stocks. Stronger generalization claims require a future,
untouched chronological holdout with a new prospective protocol.
User must attest in writing that none of the four reserved assets has
been previously evaluated/selected using its performance in this project.
The tool cannot verify that factual condition.

## Step 1 — download reserved datasets once, WITHOUT running performance tests

This snippet does not compute backtest metrics:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import yfinance as yf

out = Path('data/processed')
out.mkdir(parents=True, exist_ok=True)
for ticker in ['GLD', 'IWM', 'XLE', 'XLV']:
    dest = out / f'{ticker}_daily_processed.csv'
    if dest.exists():
        print('Exists (will not overwrite)', dest)
        continue
    df = yf.Ticker(ticker).history(period='5y', auto_adjust=False)
    if df.empty:
        raise RuntimeError(f'No data for {ticker}')
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.to_csv(dest, index=False)
    print('Saved', ticker, len(df), 'rows')
PY
```

Retain the downloaded files unmodified. Record vendor/version/date metadata
separately if available. Do not inspect performance, plots or future labels
for these assets before recording the preflight fingerprint.

## Step 2 — safe preflight (no Close values are read)

```bash
python -m research.frozen_eval preflight \
  --assets GLD=data/processed/GLD_daily_processed.csv \
           IWM=data/processed/IWM_daily_processed.csv \
           XLE=data/processed/XLE_daily_processed.csv \
           XLV=data/processed/XLV_daily_processed.csv \
  --output-dir reports/backtest/m5_external_validation
```

This checks the exact reserved asset set, date schema, sorted/unique dates,
at least 30 eligible frozen test origins, and market file SHA-256 hashes.
It writes `preflight_manifest.json` with **no performance outcomes**.
Check and retain this manifest before the first actual evaluation. If the
data snapshot changes, the locked evaluation refuses to run: use a new
output directory and disclose the change, not quietly replacing the old
manifest.

## Step 3 — only after checking and approving the preflight

```bash
python -m research.frozen_eval evaluate \
  --assets GLD=data/processed/GLD_daily_processed.csv \
           IWM=data/processed/IWM_daily_processed.csv \
           XLE=data/processed/XLE_daily_processed.csv \
           XLV=data/processed/XLV_daily_processed.csv \
  --output-dir reports/backtest/m5_external_validation \
  --attest-unused-assets --authorize-outcome-review
```

Do **not** run this until you've reviewed the locked design and can honestly
attest that the reserved assets have not been used for M1–M4 model selection.
The command is fail-closed without both flags, matching local file hashes
and protocol fingerprint. It intentionally refuses to overwrite the first
completed output: maintain a versioned archive rather than "improving"
results on the same holdout.

Results: `holdout_predictions.csv`, `holdout_interval_predictions.csv`,
`holdout_metrics.csv`, `holdout_paired_comparisons.csv`,
`holdout_calibration.csv`, `holdout_regime_coverage.csv`,
`holdout_matched_accepted.csv`, `holdout_result_manifest.json`.

## Reporting and research safeguards

- Do not change models/features/period/acceptance threshold after viewing results.
- Do not cherry-pick favourable assets/horizons. Provide all outcomes and
  state the exploratory descriptive uncertainty limits.
- Do not treat nominal conformal coverage as a guarantee in nonstationary
  financial time series. Document dependent observations and small cells.
- Do not claim causal explanations, production deployability, investment
  returns or scientific novelty without independent evidence.
- External-asset validation and genuinely prospective validation have
  different evidentiary strength and must be described distinctly.
- Reproducibility: keep source hashes, full Git SHA and the protocol hash.
- A paper submission still requires external literature review, data
  redistribution/licensing confirmation, independent baselines, ablations,
  uncertainty diagnostics and a stronger frozen test design.
