"""R7 gate is a preregistration/readiness artifact, NOT an evaluation runner."""
import json
from pathlib import Path

import pytest

from research.r7_protocol_readiness import (
    PROTOCOL, SPEC, check_readiness, event_known_as_of,
)
from research.r6_corporate_action_audit import load_events


def test_existing_event_announcement_timing_is_point_in_time():
    event = next(e for e in load_events() if e["asset"] == "XLE")
    assert event_known_as_of(event, "2025-11-19") == "NOT_ANNOUNCED_AT_ORIGIN"
    assert event_known_as_of(event, "2025-12-02") == "ANNOUNCED_FUTURE_SPLIT"
    assert event_known_as_of(event, "2025-12-05") == "SPLIT_EFFECTIVE_OR_PAST"


def test_unapproved_new_holdout_remains_blocked():
    report = check_readiness()
    assert report["protocol"] == PROTOCOL
    assert report["test_execution_authorized"] is False
    assert report["model_evaluation_performed"] is False
    assert "asset_set" in report["missing_new_holdout_fields"]
    assert "licensing_authorization" in report["missing_new_holdout_fields"]
    assert not report["old_m5_snapshots_reused_for_new_test"]


def test_r6_preservation_manifest_is_checked(tmp_path):
    file = tmp_path / "manifest.json"
    manifest = {
        "protocol": "finwise-r6-corporate-action-audit-v1",
        "refit": False, "excluded_origins": 0,
        "modified_original_scores": False,
    }
    file.write_text(json.dumps(manifest))
    report = check_readiness(r6_manifest=file)
    assert report["r6_manifest_checked"] is True
    assert len(report["r6_evidence_sha256"]) == 64
    manifest["excluded_origins"] = 1
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="preservation"):
        check_readiness(r6_manifest=file)


def test_never_activate_gate_by_editing_draft(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["status"] = "READY"
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="cannot authorize"):
        check_readiness(spec_path=file)
    spec["status"] = "DRAFT_BLOCKED_NO_TEST_EXECUTION"
    spec["required_new_holdout"]["execution_authorization"] = True
    file.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="No new test"):
        check_readiness(spec_path=file)


def test_no_prior_holdout_assets_allowed_to_disappear(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["prior_explored_assets"] = ["AAPL", "MSFT"]
    file = tmp_path / "spec.json"
    file.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="disclosed"):
        check_readiness(spec_path=file)
