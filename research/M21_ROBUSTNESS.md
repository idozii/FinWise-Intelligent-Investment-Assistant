# FinWise M2.1 — multi-asset robustness (exploratory)

Run **offline** on locally generated Yahoo-style processed CSVs. This is
neither an untouched independent test nor an estimate of investable returns.

## Inputs and execution

Provide separate price series for each named asset. The tool deliberately
does not fetch data or silently combine disparate instruments.

```bash
python -m research.robustness \
  --assets AAPL=data/processed/AAPL_daily_processed.csv \
           MSFT=data/processed/MSFT_daily_processed.csv \
           TSLA=data/processed/TSLA_daily_processed.csv \
           SPY=data/processed/SPY_daily_processed.csv \
  --min-train 252 --horizons 1,7 --step 7 \
  --models naive,ridge \
  --block-length 5 --bootstrap-iterations 1000 \
  --output-dir reports/backtest/m21_multiasset
```

For slower comparison, run separately with
`--models naive,ridge,xgb,lightgbm`. Do not compare results from different
origin sets or historical data snapshots as if they were paired.
For crypto input, use the M1/M2 asset-specific daily UTC aggregation command
first; this M2.1 runner expects the stock-style Date/Close schema.

## What is actually measured

- Exactly matched horizon, asset and origin between challenger and naive.
- Absolute-error *difference* `MAE(challenger) - MAE(naive)`, positive means
  challenger performs worse.
- Exploratory moving-block-bootstrap 95% percentile CI for that paired
  difference, fixed random seed 42. The block unit is the forecast origin.
- Lagged 20-return realized-volatility labels: low/medium/high, thresholds
  estimated entirely from observations prior to the first evaluated origin
  (per asset); labels at origin only depend on prices known then.
- Model-specific per-horizon MAE/RMSE by volatility regime.

Files: `asset_predictions.csv`, `paired_comparisons.csv`,
`regime_metrics.csv`, `robustness_manifest.json`.

## Scientific limitations and gates

- Hyperparameters are fixed but the same AAPL period has already been
  inspected. All output is labeled **exploratory**.
- A nonzero bootstrap CI is not formal confirmatory evidence: different
  assets, horizons, adaptive re-fitting, overlapping periods and selection
  imply multiple testing / nonstationarity.
- Block length five origins is an analysis choice, not a universal correct
  independence scale. Sensitivity to block length should be documented.
- Data providers revise prices and corporate action adjustments; source
  licensing, split adjustment and date semantics need precommitment.
- Final evaluation must freeze a future or otherwise untouched test sample
  *before* looking at its labels, avoid hyperparameter selection on it,
  and specify how many outcomes are compared.
- The multiasset study is not a trading backtest or investment recommendation.
