# R11 — FinWise research manuscript draft and evidence registry

The manuscript is at `research/paper/R11_MANUSCRIPT_DRAFT.md`.

**Its figures are not final publication-ready data.** The table values
are grounded in the dataset owner's pasted M5 first run and verified
local M5.1/R6/R8/R9 audit outputs from 28 October 2026, but the
source input snapshots have **not** been copied into this Git branch.
They must be validated by the local evidence generator before any
manuscript submission.

## Run evidence verification locally

```bash
python -m research.r11_paper_evidence \
  --m5-dir reports/backtest/m5_tiingo_v2 \
  --r6-manifest reports/backtest/r6_corporate_actions/r6_audit_manifest.json \
  --r8-dir reports/backtest/r8_development \
  --r9-dir reports/backtest/r9_risk_coverage_dev \
  --dev-predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --output-dir reports/backtest/r11_manuscript_evidence

cat reports/backtest/r11_manuscript_evidence/r11_generated_tables.md
python -m pytest -q tests/test_r11_paper_evidence.py
```

The generator recomputes all M5 scores and fixed-seed paired bootstrap
intervals from the **frozen first-run** predictions, checks that the
source hashes match the preflight, checks that the R6 manifest kept
original artifacts intact, enforces the original R8 source fingerprint,
recomputes each R9 policy cell, verifies the recorded R9 summary, then
writes SHA-recorded evidence tables and a machine-readable registry.

No price download, model refit, new holdout scoring or source
modification occurs. Generated tables are local paper artifacts. Do
not commit raw vendor market data without reviewing provider rights.

## Scientific and publishing gate

- R10 future independent evaluation remains BLOCKED and has not been run.
- R8/R9 adaptive and selection thresholds were explored on historical
  development outcomes, not precommitted on an untouched future test.
- Formal coverage theorems of Gibbs & Candès, CAP, or other published
  methods have not been established for this custom delayed-feedback
  clipped ACI-style implementation.
- Literature review is preliminary. "Novel algorithm" is **not** claimed.
- External source licensing, multiple comparisons, sensitivity to XLE
  corporate action, broader baselines, table regeneration, and
  supervisor/venue review remain open.
- This is a serious research draft, **not a submitted, reviewed or
  accepted conference publication**.

The paper can be used as an honest research portfolio and advisor
discussion artifact now; publication claims require evidence.
