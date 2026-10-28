# FinWise — Research + Web Roadmap (2026–2027)

**Version:** 2026-10-28 / milestone-managed plan. Targets are estimates, not scheduled background execution.

## Non-negotiable scientific guardrails

- M5-v2 Tiingo first-run remains immutable. All follow-up diagnoses are descriptive.
- Do not evaluate new models on already-explored GLD/IWM/XLE/XLV (same dates) and call it a new independent test.
- Don't call heuristic API intervals empirically calibrated; research performance is historical and offline.
- Keep source license, hashes, announcement timing, split adjustment, and no-future-label policies explicit.
- PRs remain drafts until human review, tests green and proper dependency merge order; no automatic main merge.

## Current implementation wave — R6 + W1

| ID | Scope | Deliverables | Acceptance gate |
|---|---|---|---|
| R6 | Official corporate actions and data-integrity audit | Verified XLE 2:1 split registry; read-only event-origin annotation; split-day raw price check; SHA-256 audit manifest | First-run scores/snapshots unchanged; deterministic synthetic tests; original M5 hashes verified |
| W1 | Web/API architecture and contracts | API capabilities v1.1; legacy forecast metadata; honest uncertainty labels; CORS allowlist; HTTP smoke tests | Stock/crypto legacy endpoints retain compatibility; no implied live Ridge/conformal; CI 3.11/3.12 |

## Next research/product waves

| Planned wave | Research | Product | Publication and release gate |
|---|---|---|---|
| Weeks 2–3 | R7 — define price/return estimand, corporate-action timing, data contracts, hypotheses, baselines, and new prospective or untouched holdout | W2 — historical research transparency dashboard (offline validated artifacts only) | No new holdout results viewed during model choice; dashboard separates historical vs live |
| Weeks 4–7 | R8 — adaptive vs rolling conformal calibration, coverage/width/shift diagnostics, ablations | W3 — explainability from exact Ridge decomposition with reconstruction checks | Never label explanations causal; full baselines and adjusted multiple comparisons |
| Weeks 7–10 | R9 — selective forecasting, risk–coverage and conditional selection diagnostics | W4 — uncertainty/abstention UI backed by research-grade validation | No automatic trading decision claims or coverage guarantees |
| Weeks 11–13 | R10 — freeze and execute truly new evaluation under authorized, unused time/asset scope | W5 — API tests, caching, logging, security audit, versioned data provenance and deployment | One-shot first-run artifacts, hash locked, documented negative results |
| Weeks 13–16 | R11 — scientific manuscript with citations, related work, rigorous limitations and reproducibility | W6 — public demo/README/case studies | Publish results that survive audit and respect market vendor licensing |

### Dependency order

`R6` informs `R7`; `W1` unlocks `W2`. R8/R9 develop using *development data only*. R10 cannot commence until R7 preregistration and authorization. R11 should not promise publishability without novelty evaluation or sufficiently strong evidence.

### Existing branch integration

Existing M1–M5 work uses stacked feature PRs. R6 and W1 are parallel branches based on `feat/m51-first-evaluation-evidence-audit`; review each PR separately. Future implementation tasks should start from a reviewed integration branch, avoiding unreviewed direct changes to `main`.
