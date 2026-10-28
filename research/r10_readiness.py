"""R10 *pre-registration readiness only*; no external test execution code.

Existing M5 first-run outcomes have been inspected. The R10 dataset, estimand,
source rights, time/asset independence and error-control plan must be set in
a different, genuinely new protocol before any evaluation can be authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DRAFT_PATH = (
    Path(__file__).parent / "protocols" / "r10_independent_evaluation_draft_v1.json"
)
PROTOCOL = "finwise-r10-prospective-evaluation-registration-draft-v1"
NECESSARY = (
    "independent_asset_ids", "independent_start_date", "independent_end_date",
    "evidence_unused_attestation", "time_split_policy",
    "point_in_time_data_vendor", "provider_data_rights",
    "source_manifest_sha256", "price_target_estimand",
    "splits_and_dividends_asof_policy", "model_and_calibration_code_commit",
    "model_candidates_and_hyperparameters",
    "selected_abstention_threshold", "precommitted_primary_endpoint",
    "primary_acceptance_and_coverage_tradeoff_rule",
    "statistical_inference_plan", "multiple_comparisons_policy",
    "missing_data_and_delisting_policy",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readiness(
    protocol_path: Path = DRAFT_PATH,
    r7_report: Path | None = None,
) -> dict:
    spec = json.loads(protocol_path.read_text(encoding="utf-8"))
    if spec.get("protocol") != PROTOCOL:
        raise ValueError("Wrong R10 registration protocol")
    if spec.get("status") != "BLOCKED_MISSING_NEW_SCOPE_ESTIMAND_AND_AUTHORIZATION":
        raise ValueError("Draft R10 protocol cannot authorize evaluation")
    if spec.get("evaluation_tooling_status") != "NOT_IMPLEMENTED_AND_NOT_AUTHORIZED":
        raise ValueError("No R10 test tooling is authorized")
    rules = spec.get("guardrails", {})
    if not rules or not all(value is True for value in rules.values()):
        raise ValueError("R10 scientific guardrail was disabled")
    prior = spec.get("prior_used_evidence", [])
    if not isinstance(prior, list) or not any("M5-v2" in x for x in prior):
        raise ValueError("Previously inspected M5-v2 outcomes must be disclosed")
    if not any("R9" in x for x in prior):
        raise ValueError("Exploratory R9 threshold selection must be disclosed")
    experiment = spec.get("experiment")
    if not isinstance(experiment, dict) or set(experiment) != (
        set(NECESSARY) | {"human_data_access_and_test_execution_approval"}
    ):
        raise ValueError("R10 has incomplete research preregistration fields")
    if experiment["human_data_access_and_test_execution_approval"] is not False:
        raise ValueError("Do not authorize independent test through this template")
    missing = [name for name in NECESSARY if not experiment.get(name)]
    if not missing:
        raise ValueError("This is a draft gate, not a runner for completed protocols")
    r7 = None
    if r7_report:
        r7 = json.loads(r7_report.read_text(encoding="utf-8"))
        if (
            r7.get("status") != "BLOCKED_REQUIRES_NEW_HOLDOUT_AND_ESTIMAND_PREREGISTRATION"
            or r7.get("test_execution_authorized") is not False
            or r7.get("model_evaluation_performed") is not False
            or r7.get("old_m5_snapshots_reused_for_new_test") is not False
            or r7.get("r6_manifest_checked") is not True
        ):
            raise ValueError("R7/R6 evidence gate is not trusted")
    return {
        "protocol": PROTOCOL,
        "status": "BLOCKED_REQUIRES_PRECOMMITTED_NEW_DATASET_AND_AUTHORIZATION",
        "evaluation_authorized": False,
        "test_outcomes_accessed": False,
        "test_execution_performed": False,
        "data_download_performed": False,
        "missing_research_registration_fields": missing,
        "human_approval_missing": True,
        "template_sha256": sha256(protocol_path),
        "r7_evidence_verified": r7 is not None,
        "r7_evidence_sha256": sha256(r7_report) if r7_report else None,
        "prior_evidence_excluded_from_new_test": True,
        "next_step": spec["next_step"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r7-report", type=Path,
        default=Path("reports/backtest/r7_readiness_blocked.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/backtest/r10_preregistration_blocked.json"),
    )
    args = parser.parse_args(argv)
    if not args.r7_report.is_file():
        raise FileNotFoundError(
            "Run the R7 preregistration check and preserve its report first"
        )
    if args.output.resolve() in (
        args.r7_report.resolve(), DRAFT_PATH.resolve()
    ):
        raise ValueError("Readiness output cannot overwrite original evidence")
    result = readiness(r7_report=args.r7_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
