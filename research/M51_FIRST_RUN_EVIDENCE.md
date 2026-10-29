# M5.1 — first-run evidence audit (no retesting, no model tuning)

The first actual M5-v2 evaluation has now been run by the dataset owner,
with the four Tiingo snapshot hashes set at preflight. This is a **post hoc
descriptive audit** of those *existing* outputs, not another independent
evaluation and not a new opportunity to choose Ridge hyperparameters.

## Primary outcome interpretation from the user's supplied first run

- All 8 asset × horizon Ridge−Naive MAE point differences were positive
  (Ridge higher MAE than naive).
- All 8 descriptive moving-block bootstrap 95% intervals included zero:
  do **not** infer statistically proven universal inferiority.
- Ridge ready-origin 90%-nominal empirical coverage met/exceeded .9 in
  2 of the 8 cells (IWM H7 and XLE H7). GLD H1=.78125, GLD H7=.765625
  were especially low on 64 origins each.
- XLE H7 Ridge accepted zero out of 64, compared with Naive 14/64.
  It is invalid to report Ridge selected MAE for that cell.
- Ridge on GLD H7 and IWM H7 had lower MAE **on jointly accepted origins**
  (−.247728 across n=52 and −.251417 across n=13 respectively). These
  are conditioned, smaller, potentially selection-biased subgroups and do
  not reverse the first-run overall MAE findings.
- XLE H7 had RMSE 6.269 for Naive but MAE 2.777, suggesting a long
  right-tail of absolute errors. Check the top errors and raw Close
  corporate actions without deleting data or changing scores.

**All figures above were pasted by the user, not independently
computed from the local files by the author of this PR.** The script
below cross-checks all arithmetic against those existing first-run
outputs and their hashes when run on the user's machine.

## Run (does NOT refit or download anything)

```bash
python -m research.m5_evidence_audit \
  --result-dir reports/backtest/m5_tiingo_v2 \
  --output-dir reports/backtest/m51_evidence_audit
```

This requires original Tiingo snapshot CSVs and their sidecars to
still be in the exact paths stored in preflight. It reads existing
`holdout_predictions.csv` and
`holdout_interval_predictions.csv` to verify the following:

- Frozen M5-v2 SHA-256 and exact asset/model/horizon identities.
- Preflight source hashes against original Tiingo CSV + sidecars.
- 1-to-1 paired origin/model/horizon keys, dates, interval flags,
  interval membership and empirical coverage.
- Recomputed MAE/RMSE and original fixed-seed paired bootstrap results.
- Original calibration metrics, acceptance denominators and jointly
  accepted origin metrics.
- All outcomes kept in the original sample, no exclusions.
- Descriptive top-3 largest absolute prediction errors for each
  asset × model × horizon for *investigation only*.

Separate artifacts:
- `reports/backtest/m51_evidence_audit/m51_audit_manifest.json` contains
  SHA-256 hashes of all input files and a disclosure that no refitting
  or selection occurred.
- `m51_evidence_summary.md` is a compact results summary.
- `m51_top_errors_descriptive.csv` helps inspect extreme
  predictions and events but is *not* an exclusion rule.

## Scientific publication gate

Do not reuse GLD/IWM/XLE/XLV on these dates for more model selection
and call the same observations an independent test. To assess a new
model/interval rule, preregister a **different** future time period or
untouched asset set; use the first-run results transparently as
development evidence. Distinguish nominal interval target vs
empirical coverage and acknowledge time dependence, vendor raw-close
definition, multiple comparisons and cross-asset historical overlap.
