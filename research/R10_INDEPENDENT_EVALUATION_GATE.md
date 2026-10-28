# R10 — Independent evaluation preregistration readiness

**Current status: BLOCKED; no new testing permission, no unseen asset/date range
chosen, no independent evaluation has been run.**

The first M5-v2 retrospective external-asset evaluation has already been
inspected. The experimental R8/R9 methods and W4 width policies were
analyzed on the previously explored development universe. Those results
are useful for method development but must never be rebranded as a
new independent test.

## Registration and boundaries

The template at
`research/protocols/r10_independent_evaluation_draft_v1.json`
records a research question and all mandatory choices **as null**:
new independent assets and dates, data entitlements and licensing,
input source fingerprints, strict point-in-time price/return estimand
with split/dividend handling, model and code commit SHA,
final acceptance policy, primary endpoint, sample-size/power and
statistical inference including multiple comparisons, missing-data/
survivorship policy, and human approval.

R7's readiness report must be verified first. A checked R7/R6 lineage
alone **does not** confer the right to execute R10.

```bash
python -m research.r10_readiness \
  --r7-report reports/backtest/r7_readiness_blocked.json \
  --output reports/backtest/r10_preregistration_blocked.json

python -m pytest -q tests/test_r10_readiness.py
```

Expected outcome: `BLOCKED_REQUIRES_PRECOMMITTED_NEW_DATASET_AND_AUTHORIZATION`,
with `test_execution_performed=false` and
`data_download_performed=false`.
This is a safeguard, not an implementation failure.

## What must be decided BEFORE first test

1. **Economic estimand:** raw Close vs split-neutral returns vs dividends/
   total returns. Define observable-as-of disclosures, announcement and
   effective dates, calendar and vendor revision policy. Using
   retrospective `adjClose` as if future splits were known is not allowed.
2. **Independence:** genuinely unseen and separately permissioned assets
   and/or new future time period; do not silently reuse any examined
   GLD/IWM/XLE/XLV M5 outcomes or R8/R9 development outcomes for model
   selection in the new test.
3. **Method:** preregister baseline models, adaptive algorithm, gamma,
   calibration window, width/abstention threshold and all comparators.
   The R8/R9 development grid may inform this choice ONLY before
   seeing new outcomes.
4. **Primary analysis:** choose explicit risk–availability utility,
   eligible origin count, target interval coverage, confidence procedures,
   alpha allocation across hypotheses, and how zero-acceptance is handled.
5. **Governance:** vendor data rights for analysis and publication,
   privacy/security of tokens, source SHA-256 manifests, blind access
   and written authorization.
6. **Reproducibility:** freeze a hash-locked final protocol and exact code
   SHA *before* outcomes, implement independent first-run execution as a
   **separate** future reviewed PR. Do not simply turn this draft flag
   to `true` and run tests.

No guarantee of acceptance at a research venue or statistical coverage
follows from this template. Negative results must be preserved.
