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
