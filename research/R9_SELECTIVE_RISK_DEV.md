# R9 — Development-only risk–availability policy sweep

**Stage:** exploratory post-hoc development analysis, NOT a new external test.
The R7 independent evaluation gate remains blocked until a new, legitimately
untouched asset/time period, economic estimand, license and protocol are frozen.

## Motivation from real R8 results

Adaptive ACI-style coverage increased in 14/16 development
asset/model/horizon cells, stayed equal in 2/16, and widths increased in
16/16. Ridge AAPL H7 accepted forecasts dropped from 15 to 0; MSFT H7
from 40 to 17; TSLA H7 remained at zero. These are development outcomes
and cannot establish that adaptive calibration is preferable.

## R9 evaluation definitions

Use only R8 calibrated interval rows for AAPL/MSFT/SPY/TSLA (both
Ridge and Naive, H1 and H7, both fixed rolling and delayed-feedback
ACI-style methods). The original M2.1 prediction file and R8 manifest
are required to validate source SHA-256.

For each **predeclared exploratory** interval-width cutoff of 5%, 10%,
15%, 20%, 30%, accept an origin only when its interval was ready at
issue-time and its full interval width divided by last observed Close
is no greater than that cutoff. We do **not** retrain, adjust the
intervals, or select origins based on their realized error.

Report: n_origin, n_ready, n_accepted, acceptance on *all* origins,
acceptance among ready, interval coverage among all ready and selected,
MAE and relative MAE among selected only, selected interval width,
and a status for zero/small samples. When selected n=0, its MAE and
coverage are **undefined (NaN)**, not zero.

Ridge-versus-Naive selected MAE comparisons are reported only for
*jointly accepted identical forecast origins* at the same method and
threshold; n<30 cells are explicitly exploratory. Comparing two
calibration methods' selected MAE does NOT estimate a causal improvement,
since their accepted origins differ while point predictions are
identical. A threshold chosen after reviewing these results is
development-driven and requires a NEW frozen evaluation.

## Local commands

```bash
python -m research.r9_selective_risk \
  --r8-dir reports/backtest/r8_development \
  --dev-predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --output-dir reports/backtest/r9_risk_coverage_dev
```

Files:
- `r9_risk_coverage_dev.csv` (160 cells: 4×2×2×2×5).
- `r9_matched_model_comparisons_dev.csv` (80 paired groups).
- `r9_manifest.json` (originating SHA-256, no-refit/holdout guards).

Research references:
- Geifman and El-Yaniv, *SelectiveNet: A Deep Neural Network with an
  Integrated Reject Option*, ICML 2019:
  https://proceedings.mlr.press/v97/geifman19a.html
- Bao et al., *CAP: A General Algorithm for Online Selective Conformal
  Prediction with FCR Control*, JMLR 2025:
  https://www.jmlr.org/papers/v26/24-0452.html
- Sale and Ramdas, *Online Selective Conformal Prediction: Errors and
  Solutions*, TMLR 2025: https://mcml.ai/publications/sr25a/

These references establish a substantial existing literature: R9
does not claim novelty or their conditional-coverage guarantees.
Production thresholds should not be chosen from the M5 first test.
