# R6 — Corporate action integrity audit

## Why this milestone exists

M5-v2 evaluated Tiingo raw Close (not adjusted Close) on GLD, IWM,
XLE and XLV. The first evaluation and M5.1 evidence audit found a
large XLE H7 error: 44.18 on the 2025-12-02 → 2025-12-11 origin/target
pair, corresponding to a 49.0126% raw-price movement.

**Independently verified fact:** State Street announced a 2-for-1
split for XLE on 2025-11-20; XLE traded at its new post-split price
starting 2025-12-05. This corporate action was **known prior to the
2025-12-02 forecast origin**, so it is not an unexpected black-swan
price collapse. The forecast/raw-price task spanning the split is
economically tricky. Issuer source:
https://investors.statestreet.com/investor-news-events/press-releases/news-details/2025/State-Street-Investment-Management-Announces-Share-Splits-for-Five-Select-Sector-SPDR-ETFs/default.aspx

The event is recorded in
`research/protocols/r6_verified_corporate_actions_v1.json`.
Tiingo documents its `close`, `adjClose`, `splitFactor`
and `divCash` fields separately:
https://www.tiingo.com/documentation/end-of-day

## Execute R6 read-only audit

Keep the original local Tiingo snapshot CSV files and first-run outputs
in place. Run from the repository root:

```bash
python -m research.r6_corporate_action_audit \
  --result-dir reports/backtest/m5_tiingo_v2 \
  --output-dir reports/backtest/r6_corporate_actions
cat reports/backtest/r6_corporate_actions/r6_integrity_report.md
```

The run verifies M5-v2 source hashes and previously reported scores,
then identifies:
- Forecasts whose origin is before the split-effective date and whose
  target is on or after that date.
- Forecast origins within the first 30 observed trading dates
  following the split (possible retraining/feature consequences).
- Whether the event was announced at the relevant forecast origin.
- Split-day and previous-session **raw** closes, price ratio and
  hypothetical split-neutral *one-session* return, **not model returns**.

## Files

`r6_event_linked_origins.csv`: only event-flagged origins, descriptive.

`r6_event_impact_descriptive.csv`: all 16 asset/model/horizon
groups, original MAE, crossing error contribution, and counts.

`r6_split_date_raw_close_check.csv`: official event and
local Tiingo raw close ratio.

`r6_audit_manifest.json`: SHA-256 of first-run input
artifacts and event registry, plus preservation assertions.

`r6_integrity_report.md`: concise evidence-based summary.

**Hard boundary:** Never alter or overwrite the first M5-v2 files,
remove or rescore split-crossing forecasts, or claim this post-hoc
audit is another independent test. A split-aware target definition
must be fixed in a distinct **future** evaluation protocol, with
appropriate as-of features and independent data. Raw and adjusted
prices have different scientific interpretations; adjusting the
reference returns alone is not equivalent to refitting a corrected
prediction target.

## R6 acceptance criteria

- Exact source/evaluation hash check, independent recomputation of
  published first-run metrics, no model retraining.
- Official split date and announcement provenance recorded.
- Crossing and post-split forecasts identified reproducibly.
- Unchanged original MAE and all original origins retained.
- Separate audit artifacts (never under the M5 frozen result folder).
- Deterministic tests, including a synthetic known-date split.

## Next gate (R7)

Freeze a coherent, split-aware estimand and explicitly state what can
be known at each forecast origin, with source methodology/licensing,
then preregister fresh tests. Do **not** cherry-pick or repair the
existing M5 result to present stronger performance.
