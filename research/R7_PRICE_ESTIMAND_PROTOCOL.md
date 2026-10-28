# FinWise R7 — price estimand and independent evaluation readiness

**State:** DRAFT — no new dataset named, no future independent evaluation approved.

## Motivation

The frozen Tiingo M5-v2 first-run evaluated historical **unadjusted raw Close**.
R6 linked a 2025-12-02 → 2025-12-11 XLE H7 forecast to a known
(announced 2025-11-20) 2-for-1 share split effective 2025-12-05.
One XLE H7 origin accounted for 24.9% of Naive total absolute error
and 22.5% of Ridge total absolute error. The first-run numbers stay unchanged.

Issuer: https://investors.statestreet.com/investor-news-events/press-releases/news-details/2025/State-Street-Investment-Management-Announces-Share-Splits-for-Five-Select-Sector-SPDR-ETFs/default.aspx

Tiingo EOD fields \`close\`, \`adjClose\`, \`splitFactor\`, \`divCash\` have
different statistical meanings:
https://www.tiingo.com/documentation/end-of-day

## Scientific question / proposed target

**Can adaptive calibration and abstention improve observed reliability
and risk–coverage trade-offs under market changes?**

The *proposed*, not yet authorized, successor target is a split-neutral
log-return estimand with the decision-time information set recorded:
what splits/dividends were known at each origin, when announced, when
effective, whether prices were adjusted as of origin, and how training
features/labels were constructed without post-origin corporate actions.

**Do not** use retrospective \`adjClose\` as though its whole history had
been known at earlier origins: later corporate actions can revise
earlier adjustment factors. Future dividends require a separately
specified convention; a split-neutral return is not automatically a
dividend-adjusted total return. Avoid mixing adjusted returns with raw
Close price MAE as if their units were identical.

An as-of feature set for origin t may use only facts with an
availability timestamp no later than t. A split announced before t
but effective later may be known for operational handling; a future
unannounced split cannot become a feature at t. Historical target
construction and as-of training labels must be defined and tested
before using this in a new model benchmark.

## Draft protocol and hard gate

Files:
- \`research/protocols/r7_estimand_readiness_draft_v1.json\`: declared
  explored assets, unresolved target conventions, missing future scope.
- \`research/r7_protocol_readiness.py\`: verifies R6 preservation evidence,
  timing of verified corporate actions, and reports missing gates.

Run **without running another model test**:

\`\`\`bash
python -m research.r7_protocol_readiness \
  --r6-manifest reports/backtest/r6_corporate_actions/r6_audit_manifest.json \
  --output reports/backtest/r7_readiness_blocked.json
\`\`\`

Expected status: \`BLOCKED_REQUIRES_NEW_HOLDOUT_AND_ESTIMAND_PREREGISTRATION\`.
That is **correct**: a truly new evaluation is not yet authorized.
No GLD/IWM/XLE/XLV M5 first-run period can be reused to claim a second
independent test after observing performance.

## Next R7 scientific acceptance criteria

1. Choose one explicit economic estimand (and evaluate baseline on the
   **same target/units**), adjusted for splits/dividends only under a
   documented rule with origin-date availability.
2. Decide source entitlement, licensed historical point-in-time data,
   instruments/dates and treatment of delistings, missing bars,
   dividends, splits and market holidays.
3. Lock all assets, data timeframe, model candidates, calibration
   baselines, widths/acceptance thresholds and multiple-testing policy.
4. Implement positive/negative as-of/leakage unit tests and source hashes.
5. Obtain research authorization for the *new* dataset before test execution.
6. Keep prior M5, M5.1 and R6 results intact and disclose caveats in paper.

R7 first creates a **protocol and readiness gate**, not a reinterpreted
post-hoc profit or corrected-model accuracy claim.
