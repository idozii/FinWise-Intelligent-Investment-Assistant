# R8 — Adaptive uncertainty (exploratory development only)

## Boundary

R7's new-independent-holdout gate remains blocked. This is **NOT** a
new test. R8 accepts ONLY the pre-explored AAPL/MSFT/SPY/TSLA M2.1
prediction CSV and rejects any dataset containing GLD/IWM/XLE/XLV, the
M5-v2 external evaluation assets.

The module **does not retrain models**; it reads already generated
origin-level Ridge/Naive predictions and compares two methods:

1. `fixed_rolling`: unchanged M3 rolling absolute-residual conformal-style
   interval (nominal alpha .10, 30 matured residual minimum, 60 window).
2. `delayed_feedback_aci_style`: the same rolling residual reference
   but the current miscoverage target alpha changes ONLY when a prior
   forecast's true target date has matured. Update:
   `alpha_new = clip(alpha_old + 0.02 * (0.1 - 1{miss}), .01, .50)`.
   The score for the current forecast is **never** used to set its
   own interval. Each model, asset and horizon has independent state.

Both use the same 10% full-interval/last-close width threshold for
abstention. Sample sizes, ready-origin coverage, selected coverage,
mean relative width and selection rates are all reported; the ACI
finite-sample radius may be infinite at extreme alpha and is then
reported as unready.

**Theoretical boundary:** This is an *ACI-style experimental baseline*,
not an exact reproduction of a published multi-step ACI theorem. Clipping,
delayed/overlapping labels, quantile discreteness, asset drift and
forecast error dependence limit direct transfer of formal guarantees.
No causal, profit or live performance claims. Results are development
hypotheses only, not M5 or R10 external validation.

Relevant literature:
- Gibbs and Candès, Conformal Inference for Online Prediction with
  Arbitrary Distribution Shifts, JMLR 2024:
  https://www.jmlr.org/papers/v25/22-1218.html
- Hallberg Szabadváry, Adaptive Conformal Inference for Multi-Step
  Ahead Time-Series Forecasting Online, PMLR 2024:
  https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html

## Running the first development comparison

```bash
python -m research.r8_adaptive_development \
  --predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --output-dir reports/backtest/r8_development

python -m pytest -q tests/test_r8_adaptive_development.py
```

Outputs: `r8_intervals_development.csv`,
`r8_comparison_development.csv`, `r8_manifest.json`.
**Do not** pass M5-v2 holdout_predictions.csv; input gates reject it.
Any choice of ACI gamma or new calibration methodology driven by this
development data must be preregistered before truly independent R10
evaluation. Preserve all negative results and compare coverage jointly
with width/abstention, not in isolation.
