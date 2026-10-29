# FinWise M3 — time-respecting forecast uncertainty

**Status: exploratory, offline research prototype.** This module consumes
previously generated point forecasts; it does not alter the public API,
provide investment advice or make model-specific reliability guarantees.

## Procedure

At each forecast origin, for every asset, model and horizon separately:

1. Find error outcomes from *earlier origins* whose target date has already
   occurred by the current train_end. Do not use later labels.
2. Take the most recent N matured absolute errors (default 60).
3. After a minimum warmup (default 30), compute order statistic
   `ceil((n+1)*(1-alpha))` to set a symmetric absolute-error interval.
4. Mark an interval 'ready' only if finite and adequately calibrated.
5. Accept a forecast only if the full interval width divided by the last
   observed price is <= a fixed threshold (default 10%).
6. Report empirical interval coverage, width, and selective MAE alongside
   abstention rates. Never confuse abstentions with successes.

The calibrated radius uses **previously observed** prediction errors.
The current forecast's target is used only afterward to score results.
No additional calibration data is required to run the M3 script.

## Run against M2.1 output

```bash
python -m research.uncertainty \
  --predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --alpha 0.10 --min-calibration 30 \
  --calibration-window 60 --max-relative-width 0.10 \
  --output-dir reports/backtest/m3_intervals
```

Or use M2's AAPL-only file and set `--asset AAPL`:

```bash
python -m research.uncertainty \
  --predictions reports/backtest/aapl_ml/ml_predictions.csv \
  --asset AAPL --output-dir reports/backtest/m3_aapl
```

Outputs: `interval_predictions.csv`, `interval_metrics.csv`,
`interval_manifest.json`.

## Interpretation boundaries

The procedure is conformal-*style*, not a guarantee of nominal coverage
under non-exchangeable and nonstationary financial data. Rolling windows
and abstention can change selected-population coverage; coverage on selected
points is explicitly shown. Calibration and risk thresholds are fixed
exploratory defaults, not parameters optimized for actual returns.
The script makes no risk-adjusted portfolio allocation decisions.
A genuinely independent, pre-registered, data-frozen evaluation must
precede any claim about generalization or financial utility.
