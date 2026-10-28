# FinWise M4.1 — exact Ridge prediction explanation

**Goal:** explain how the already implemented M2 Ridge regressor constructed
each forecast, rather than claiming that ML found economically causal drivers.

At an origin, M4.1 refits the *same fixed Ridge(alpha=10.0)* and
StandardScaler as M2 using only labels matured by that origin. For its
standardized feature vector z and coefficient vector w, the Ridge-predicted
log-return is `intercept + sum(z_i * w_i)`. Each logged feature
contribution is exactly `z_i * w_i`. The pipeline cross-checks
the reconstructed next-price against M2's `fit_predict` and fails if they
diverge.

## Local run

```bash
python -m research.ridge_attribution \
  --assets AAPL=data/processed/AAPL_daily_processed.csv \
           MSFT=data/processed/MSFT_daily_processed.csv \
           TSLA=data/processed/TSLA_daily_processed.csv \
           SPY=data/processed/SPY_daily_processed.csv \
  --min-train 252 --horizons 1,7 --step 7 \
  --max-origins 32 \
  --output-dir reports/backtest/m41_explanations

python -m research.dashboard \
  --robustness-dir reports/backtest/m21_multiasset \
  --uncertainty-dir reports/backtest/m3_intervals \
  --attributions reports/backtest/m41_explanations/ridge_feature_summary.csv \
  --output reports/backtest/finwise_research_m41.html
```

Files:
- `ridge_contributions.csv`: per-origin feature values, standardized
  coefficients and exact additive terms in predicted log-return space.
- `ridge_feature_summary.csv`: retrospective mean absolute
  contributions and positive/negative sign frequencies across an evenly
  spaced, limited set of historical origins.
- `ridge_reconstruction_checks.csv`: verification that decomposed
  predictions match M2 Ridge point predictions.
- `attribution_manifest.json`: model protocol, source SHA256 and
  origin-sampling rule.

**Not SHAP. Not causal. Not a proof of feature relevance out of sample.**
Collinear features can redistribute Ridge coefficients, and changing a
feature value without recalculating correlated values or retraining does
not model a real market intervention. Accuracy and reliability must still
be evaluated independently. Never call signed contribution on log-return
a USD price change, investment return, or causal economic explanation.

Offline output only; no secrets or raw market prices are uploaded.
