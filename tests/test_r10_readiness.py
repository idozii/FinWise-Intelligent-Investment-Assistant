"""R10 preregistration must block model tests even if a draft is edited."""
import json

import pytest

from research.r10_readiness import (
    DRAFT_PATH, PROTOCOL, readiness,
)


def test_r10_requires_new_data_and_does_not_run_tests():
    report = readiness()
    assert report["protocol"] == PROTOCOL
    assert report["status"] == "BLOCKED_REQUIRES_PRECOMMITTED_NEW_DATASET_AND_AUTHORIZATION"
    assert report["evaluation_authorized"] is False
    assert report["test_outcomes_accessed"] is False
    assert report["test_execution_performed"] is False
    assert report["data_download_performed"] is False
    assert "source_manifest_sha256" in report["missing_research_registration_fields"]
    assert "multiple_comparisons_policy" in report["missing_research_registration_fields"]


def test_valid_prior_r7_manifest_must_be_blocked(tmp_path):
    report = {
        "status": "BLOCKED_REQUIRES_NEW_HOLDOUT_AND_ESTIMAND_PREREGISTRATION",
        "test_execution_authorized": False,
        "model_evaluation_performed": False,
        "old_m5_snapshots_reused_for_new_test": False,
        "r6_manifest_checked": True,
    }
    path = tmp_path / "r7.json"
    path.write_text(json.dumps(report))
    assert readiness(r7_report=path)["r7_evidence_verified"] is True
    report["test_execution_authorized"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not trusted"):
        readiness(r7_report=path)


def test_draft_status_must_not_be_forged_as_approved(tmp_path):
    draft = json.loads(DRAFT_PATH.read_text())
    draft["status"] = "APPROVED"
    file = tmp_path / "draft.json"
    file.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="cannot authorize"):
        readiness(protocol_path=file)


def test_explored_test_evidence_must_be_disclosed(tmp_path):
    draft = json.loads(DRAFT_PATH.read_text())
    draft["prior_used_evidence"] = ["M2.1 dev"]
    file = tmp_path / "draft.json"
    file.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="M5-v2"):
        readiness(protocol_path=file)


def test_forbid_activating_execution_flag(tmp_path):
    draft = json.loads(DRAFT_PATH.read_text())
    draft["experiment"]["human_data_access_and_test_execution_approval"] = True
    file = tmp_path / "draft.json"
    file.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="Do not authorize"):
        readiness(protocol_path=file)


def test_guardrails_cannot_be_disabled(tmp_path):
    draft = json.loads(DRAFT_PATH.read_text())
    draft["guardrails"]["no_test_labels_or_outcomes_accessed_during_preregistration"] = False
    file = tmp_path / "draft.json"
    file.write_text(json.dumps(draft))
    with pytest.raises(ValueError, match="guardrail"):
        readiness(protocol_path=file)
