"""W2 evidence contract: offline aggregates only; never leak raw price bars."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.research_evidence import load_export, validate_export
from research.w2_evidence_export import SCHEMA, make_summary


def input_tables():
    metrics, paired, calibrated, matched = [], [], [], []
    for asset in ["GLD", "IWM", "XLE", "XLV"]:
        for h in (1, 7):
            for model, mae in [("naive", 2.0), ("ridge", 2.1)]:
                metrics.append({
                    "asset": asset, "model": model,
                    "horizon_step": h, "n_origins": 64,
                    "mae": mae,
                })
            paired.append({
                "asset": asset, "model": "ridge", "horizon_step": h,
                "n_origins": 64,
                "mae_delta_model_minus_naive": .1,
                "bootstrap_ci_low": -.2, "bootstrap_ci_high": .4,
            })
            calibrated.append({
                "asset": asset, "model": "ridge", "horizon_step": h,
                "n_total": 64, "n_ready": 64, "n_accepted": 32,
                "coverage_ready": .85,
            })
            matched.append({
                "asset": asset, "model": "ridge", "horizon_step": h,
                "n_origins": 64, "n_both_accepted": 25,
                "delta_mae_model_minus_naive_same_accepted": -.01,
            })
    return [pd.DataFrame(x) for x in (metrics, paired, calibrated, matched)]


def package():
    return make_summary(*input_tables(), {
        asset: "a" * 64 for asset in ("GLD", "IWM", "XLE", "XLV")
    }, "b" * 64)


def test_complete_8_cell_export_strictly_historical_and_no_prices():
    data = package()
    assert data["schemaVersion"] == SCHEMA
    assert data["executionMode"] == "OFFLINE_AUDITED_EXPORT_NOT_LIVE_PREDICTION"
    assert len(data["results"]) == 8
    response = validate_export(data)
    assert len(response["results"]) == 8
    assert "sourceSha256" not in response  # no unnecessary provider fingerprints in UI
    assert "predictedClose" not in json.dumps(response)
    assert "last_observed_close" not in json.dumps(response)
    assert all(row["bootstrapContainsZero"] for row in response["results"])


def test_missing_or_duplicate_metrics_fail_closed():
    rows = input_tables()
    rows[0] = rows[0].iloc[1:].copy()
    with pytest.raises(ValueError, match="Missing or duplicated"):
        make_summary(*rows, {asset:"a"*64 for asset in ("GLD","IWM","XLE","XLV")}, "b"*64)
    data = package()
    data["results"][0]["asset"] = "TSLA"
    with pytest.raises(ValueError, match="Unexpected historical"):
        validate_export(data)


def test_reject_unverified_claims_and_invalid_interval_data():
    data = package()
    data["executionMode"] = "LIVE_REALTIME_VALIDATED"
    with pytest.raises(ValueError, match="explicitly offline"):
        validate_export(data)
    data = package()
    data["results"][0]["ridgeCoverageReady"] = 1.5
    with pytest.raises(ValueError, match="coverage"):
        validate_export(data)
    data = package()
    data["results"][0]["bootstrap95Low"] = float("nan")
    with pytest.raises(ValueError, match="numeric"):
        validate_export(data)


def test_local_summary_load_missing_and_oversize(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_export(tmp_path / "missing.json")
    path = tmp_path / "oversized.json"
    path.write_text("z" * 256001)
    with pytest.raises(ValueError, match="oversized"):
        load_export(path)
    path.write_text(json.dumps(package()))
    assert len(load_export(path)["results"]) == 8


def test_frontend_renders_historical_metadata_without_innerhtml():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend" / "index.html").read_text()
    js = (root / "frontend" / "app.js").read_text()
    assert 'id="researchEvidenceRows"' in html
    assert "HISTORICAL · NOT LIVE" in html
    assert "/api/research/evidence" in js
    assert "td.textContent = value" in js
    assert "researchDetailsEl.hidden = true" in js
    assert "researchRowsEl.innerHTML" not in js
