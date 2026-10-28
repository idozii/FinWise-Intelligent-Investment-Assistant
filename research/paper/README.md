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


## R11.1 — Publication figures (optional verified local paper assets)

After a successful `R11 EVIDENCE VERIFIED` local run, generate seven
**descriptive** plots, each in raster PNG and vector SVG formats:

```bash
python -m pip install 'matplotlib>=3.8,<4'
python -m research.r111_scientific_figures \
  --m5-dir reports/backtest/m5_tiingo_v2 \
  --r6-manifest reports/backtest/r6_corporate_actions/r6_audit_manifest.json \
  --r8-dir reports/backtest/r8_development \
  --r9-dir reports/backtest/r9_risk_coverage_dev \
  --dev-predictions reports/backtest/m21_multiasset/asset_predictions.csv \
  --r11-registry reports/backtest/r11_manuscript_evidence/r11_evidence_registry.json \
  --output-dir reports/backtest/r111_publication_figures
python -m pytest -q tests/test_r111_scientific_figures.py
```

Expected 7 sets of plots (14 image assets):

| Figure(s) | Input scope | Interpretation |
|---|---|---|
| `fig01_m5_paired_mae` | M5 original retrospective external first run | Paired Ridge−Naive raw-Close MAE, descriptive 95% moving-block bootstrap, n=64 per asset/horizon; no multiple comparison adjustment; split caveat |
| `fig02_r8_naive_coverage_width`, `fig02_r8_ridge_coverage_width` | R8 exploratory development | Matched ready-origin empirical coverage versus mean relative interval width; fixed and delayed-feedback ACI-style on the *same* point forecasts |
| `fig03_r9_{aapl,msft,spy,tsla}_risk_availability` | R9 exploratory development | Ridge H7 selected relative MAE (%) against fraction of **all** origins accepted, across 5 width thresholds; no plotted point for 0 accepted |

The figure generator **regenerates** the hash/audit-checked R11 registry
from local source inputs and compares it with the prior saved copy,
re-aggregates R8 from origin-level interval records, recomputes all
160 R9 policy cells and verifies the saved CSV before drawing.

It does not download prices, refit models, access R10 outcomes or
alter M5/R6/R8/R9/R11 sources. Outputs are written to a **new**
directory; if a previous image or manifest would change, rerun into a
distinct versioned output folder rather than overwriting an earlier
evidence artifact.

The manifest `r111_figures_manifest.json` includes
source and image SHA-256 hashes and explicitly states that the R10
independent evaluation has **not** occurred.

**Figure captions and submission guardrails:**
- State that M5 bootstrap intervals are descriptive, not adjusted for
  multiplicity, and do not alone show universal Ridge inferiority.
- Separate M5 retrospective external-asset results from R8/R9
  development results. Never combine these samples as if iid.
- The 90% nominal interval target is **not** a guarantee.
- A policy with no accepted forecasts has **unavailable** selected
  error, not zero error.
- Never use a post-hoc R9 width cutoff selection as evidence of
  independent improvement, nor crop figures to hide unfavorable assets.
- Vendor-derived plotted aggregates require a final data licensing
  review before redistribution or publication.


## R11.2 figure readability revision (separate output directory)

The first R11.1 plots were locally verified. Visual review showed
overlapping labels on R8 coverage–width panels and R9 risk–availability
plots, especially at policies with repeated acceptance sets. R11.2
revises **only presentation**, not any forecast errors, coverage,
bootstrap confidence intervals, development origins or test eligibility.

Changes:
- R8 places H1 and H7 in separate axes, identifies assets by a single
  color legend and fixed/ACI by marker shape. Its coverage y-axis
  clearly says it is **zoomed** to 75–97%.
- R9 shows each distinct (acceptance, selected error) state as a discrete
  scatter point, without misleading interpolating connecting segments.
- A companion table lists **all five** width thresholds (5/10/15/20/30%)
  for both models, including thresholds sharing the same accepted
  origin set and those with **zero accepted forecasts**.
- Selected risk at n=0 displays `N/A`, never 0;
  risks based on n<30 are marked with `*`.
- R9's selected relative MAE axes begin at zero, to avoid visually
  exaggerating tiny differences between methods.

Use a NEW output directory so the first R11.1 artifacts are
preserved and the version SHA manifest is separate:

```bash
git fetch origin
git switch feat/r112-paper-figure-readability
python -m research.r111_scientific_figures \
  --r11-registry reports/backtest/r11_manuscript_evidence/r11_evidence_registry.json \
  --output-dir reports/backtest/r112_publication_figures
python -m pytest -q tests/test_r111_scientific_figures.py tests/test_r112_figure_readability.py
```

The default M5/R6/R8/R9/dev data paths are the same as R11.1.
Expected: `R11.2 FIGURES VERIFIED`, seven plots in PNG and
SVG and a SHA-256 manifest. The underlying original M5-v2 scores
and R10 no-independent-test gate must remain untouched.
