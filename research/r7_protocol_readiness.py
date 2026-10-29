"""R7 research-estimand readiness gate; intentionally cannot run model tests.

Validates documented corporate-action timing, original evidence preservation,
and required decisions before a NEW evaluation may be preregistered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from research.r6_corporate_action_audit import REGISTRY, load_events

SPEC = Path(__file__).parent / "protocols" / "r7_estimand_readiness_draft_v1.json"
PROTOCOL = "finwise-r7-estimand-and-evaluation-readiness-v1"
MISSING_HOLDOUT = (
    "asset_set", "start_date", "end_date", "data_vendor",
    "licensing_authorization", "provenance_sha256",
    "independence_attestation",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_known_as_of(event: dict, origin_date: str) -> str:
    """Distinguish announced split from previously unannounced split event."""
    announced = pd.Timestamp(event["announcement_date"], tz="UTC")
    effective = pd.Timestamp(event["effective_date"], tz="UTC")
    origin = pd.Timestamp(origin_date, tz="UTC")
    if origin < announced:
        return "NOT_ANNOUNCED_AT_ORIGIN"
    if origin < effective:
        return "ANNOUNCED_FUTURE_SPLIT"
    return "SPLIT_EFFECTIVE_OR_PAST"


def check_readiness(spec_path: Path = SPEC,
                    r6_manifest: Path | None = None) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected R7 preregistration draft protocol ID")
    if spec.get("status") != "DRAFT_BLOCKED_NO_TEST_EXECUTION":
        raise ValueError("R7 draft cannot authorize model evaluation")
    if not set(["GLD", "IWM", "XLE", "XLV"]).issubset(
        set(spec["prior_explored_assets"])
    ):
        raise ValueError("M5-v2 already evaluated assets must be disclosed")
    if spec.get("evaluation_gate") != "BLOCKED_UNTIL_PRECOMMITTED_NEW_SCOPE_AND_PERMISSIONS":
        raise ValueError("Invalid experiment gate")
    original = {}
    if r6_manifest is not None:
        original = json.loads(r6_manifest.read_text(encoding="utf-8"))
        if (original.get("protocol") != "finwise-r6-corporate-action-audit-v1"
                or original.get("refit") is not False
                or original.get("excluded_origins") != 0
                or original.get("modified_original_scores") is not False):
            raise ValueError("R6 evidence preservation requirements not satisfied")
    events = load_events()
    evidence = [
        {
            "asset": e["asset"],
            "ratio_new_per_old": e["ratio_new_per_old"],
            "announced_at_2025_12_02": event_known_as_of(e, "2025-12-02"),
            "status_before_announcement": event_known_as_of(e, "2025-11-19"),
            "official_source_url": e["source_url"],
        }
        for e in events
    ]
    if not evidence or not any(
        e["announced_at_2025_12_02"] == "ANNOUNCED_FUTURE_SPLIT"
        for e in evidence
    ):
        raise ValueError("Expected issuer-announced XLE event timing not found")
    scope = spec["required_new_holdout"]
    missing = [x for x in MISSING_HOLDOUT if not scope.get(x)]
    if scope.get("execution_authorization") is not False:
        raise ValueError("No new test is authorized by this draft")
    if set(spec["price_estimand"].get("requires_decision", [])) == set():
        raise ValueError("Missing estimand decision inventory")
    return {
        "protocol": PROTOCOL,
        "status": "BLOCKED_REQUIRES_NEW_HOLDOUT_AND_ESTIMAND_PREREGISTRATION",
        "test_execution_authorized": False,
        "model_evaluation_performed": False,
        "missing_new_holdout_fields": missing,
        "old_m5_snapshots_reused_for_new_test": False,
        "r6_manifest_checked": r6_manifest is not None,
        "r6_evidence_sha256": digest(r6_manifest) if r6_manifest else None,
        "events_registry_sha256": digest(REGISTRY),
        "event_asof_checks": evidence,
        "required_price_estimand_decisions": spec["price_estimand"]["requires_decision"],
        "next_step": (
            "Freeze an as-of corporate-action-safe target definition and an"
            " independently authorized untouched dataset before any R8/R10 evaluation"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6-manifest", type=Path,
                        default=Path("reports/backtest/r6_corporate_actions/r6_audit_manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/backtest/r7_readiness_blocked.json"))
    args = parser.parse_args(argv)
    if not args.r6_manifest.exists():
        raise FileNotFoundError("Run the read-only R6 audit before R7 readiness")
    report = check_readiness(r6_manifest=args.r6_manifest)
    if args.output.resolve() == args.r6_manifest.resolve():
        raise ValueError("Do not overwrite source evidence with R7 report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
