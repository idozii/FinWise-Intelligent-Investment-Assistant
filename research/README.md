# FinWise research evaluation — protocol v1

This evaluation is **offline, out-of-sample and informational**, not financial advice. Synthetic-test results are not market performance. The evaluator does not download data, tune hyperparameters on test windows or claim investment profitability.

## Protocol

- Expanding-window walk-forward: fit only observations before each forecast origin.
- Same fixed historical origins for each model, with a persistence/last-close (`naive`) baseline.
- Default: 252 training observations, seven subsequent observations per origin, step seven (non-overlapping forecast windows). Use fixed settings before interpreting results.
- Per-horizon MAE, RMSE, improvement vs persistence, direction accuracy for non-flat forecasts only, and the eligible-direction sample count.
- Strictly one asset and one observation per UTC day. Intraday crypto requires explicit aggregation to the last observation per UTC day.
- Records every prediction, summary, and source SHA-256/settings manifest in the output directory.
- ARIMA and Holt-Winters fall back to linear when unstable, matching the API's existing behavior; `effective_model` and `fallback_rate` make this visible.

**Limitations:** point-prediction parameters mirror the existing FastAPI models but remain separately implemented; this is not yet a shared production engine. The current API's 95% bounds have not been empirically calibrated. Historical data may be revised by its provider. No evidence of live predictive advantage or trading profits is claimed.

## Stocks (processed CSV needed)

```bash
python -m research.backtest \
  --input data/processed/AAPL_daily_processed.csv \
  --date-col Date --price-col Close \
  --min-train 252 --horizon 7 --step 7 \
  --models naive,moving_average,linear \
  --output-dir reports/backtest/aapl
```

Optional: add `holt_winters,arima` to `--models` for slower statistical forecasting benchmarks.

## Crypto (explicit UTC-day aggregation)

```bash
python -m research.backtest \
  --input data/processed/crypto_prices_processed.csv \
  --date-col timestamp --price-col price \
  --asset-col coin --asset bitcoin --aggregate-daily-last \
  --min-train 180 --horizon 7 --step 7 \
  --output-dir reports/backtest/bitcoin
```

Each horizon step is a **next observation**, not necessarily a calendar day for equities (weekends and market holidays are absent).

Run deterministic unit tests:

```bash
python -m pip install -r backend/requirements.txt pytest
python -m pytest -q tests/test_backtest.py
```

Outputs: `predictions.csv`, `metrics.csv`, and `manifest.json`. Generated outputs and market CSVs must not be committed.

## Next phase

Freeze date/asset splits and authorized snapshots; add log-return baselines and ablations, calibrated interval coverage and width, statistical confidence intervals, cross-regime robustness checks, and a transaction-cost-aware *paper* simulation. Before presenting any research results, document data licensing and limitations.

## M2 exploratory ML benchmark (Ridge, XGBoost, LightGBM)

M2 evaluates direct future **log-return targets** with features computed only
from the last 30 observed daily prices. Models share evaluation origins and
horizons with an exact naive last-price reference. Each origin refits models
using only labels whose target timestamps are already observed (prequential
online evaluation). No data provider is called by this module.

Features: rolling log returns (1/3/5/10/20/30), volatility (5/20/30),
price versus simple averages (5/20), and fraction of positive returns (10).
Model hyperparameters are fixed inside research/ml_backtest.py.

Install optional models:

```bash
python -m pip install -r backend/requirements.txt -r research/requirements-ml.txt pytest
```

Quick exploratory AAPL comparison (use the same 7-origin step as M1 for
comparability; XGBoost and LightGBM may take longer):

```bash
python -m research.ml_backtest \
  --input data/processed/AAPL_daily_processed.csv \
  --min-train 252 --horizons 1,7 --step 7 \
  --models naive,ridge,xgb,lightgbm \
  --output-dir reports/backtest/aapl_ml
```

Output: ml_predictions.csv, ml_metrics.csv, ml_manifest.json. The manifest
labels results **EXPLORATORY_ONLY_NOT_INDEPENDENT_TEST**.

**Critical methodological limits:** The same historical AAPL data have already
been examined in M1; these results cannot be called an independent final test.
At each walk-forward origin, earlier test labels may enter later training after
they mature; this is valid online refitting, not a fixed untouched holdout.
Do not tune models on these outputs and then claim test accuracy. Reserve an
independently frozen time/asset holdout, validate stability across different
market regimes, evaluate prediction intervals and consider adjusted-price
definitions before publication. Horizon 7 means seven trading observations
for stocks, not seven calendar days.
