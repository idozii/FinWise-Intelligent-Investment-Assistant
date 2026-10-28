# FinWise M3.1 — calibration & fair selection audit

**Research-only.** Runs offline using *the exact local* M2.1 origin predictions
and M3 calibrated interval CSVs. It refuses unmatched origins, source/interval
price inconsistencies, duplicate keys or impossible reported coverage.

## Reproduce

```bash
python -m research.calibration_audit \
  --robustness-dir reports/backtest/m21_multiasset \
  --uncertainty-dir reports/backtest/m3_intervals \
  --thresholds 0.05,0.10,0.20,0.40 \
  --output-dir reports/backtest/m31_audit

python -m pytest -q tests/test_calibration_audit.py
```

Artifacts (all excluded from git by `reports/backtest/`):
- `model_audit.csv`: accepted, ready and all-origin denominators; empirical
  coverage vs nominal target, selected MAE and small-sample warning.
- `regime_coverage.csv`: descriptive low/medium/high volatility coverage
  and acceptance counts (not conditional coverage guarantees).
- `width_sensitivity.csv`: accepted count and retrospective MAE/coverage
  at **fixed exploratory** width thresholds 5%, 10%, 20%, 40%.
- `matched_accepted_comparisons.csv`: naive vs Ridge/other model
  **on the same origins that BOTH models accepted**. Unmatched accepted MAEs
  must not be compared across models. NaN means no comparable selection.
- `audit_manifest.json`: source hashes, protocol and caveats.

Interpretation: all results are exploratory; the alpha=0.10 design target
does **not** assure 90% coverage under financial market regime shifts.
Small accepted counts mean selected coverage estimates are unstable.
The width sensitivity is a post-hoc stress test, **not** hyperparameter
tuning eligible for an independent final evaluation. Do not optimize
the acceptance threshold on these outcomes and label the same dates as
unseen test data.

## Decision gate before M4.1/M5

Document pre-declared minimum useful acceptance, a frozen price definition
(split adjustment vs raw close), calendar conventions, sample sizes,
multi-comparison handling and a truly unseen holdout. SHAP should explain
the trained estimator on origin-available features, not claim causal effects.
