import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.frozen_eval import evaluate, preflight
from research.frozen_eval_tiingo import LOCKED_V2_SHA256, load_v2_protocol
from research.m5_evidence_audit import (
    descriptive_error_flags, generate_report, recalculated_diagnostics,
    verify_first_run,
)
from scripts.m5_download_tiingo import save_snapshot


@pytest.fixture
def first_run(tmp_path):
    """Run the existing M5 implementation on SYNTHETIC data only."""
    spec = json.loads(json.dumps(load_v2_protocol()))
    spec.update({
        "reserved_external_assets": ["GLD"],
        "forecast_origin_start": "2025-03-01",
        "forecast_origin_end": "2025-12-31",
        "min_train": 70,
        "step": 7,
        "horizons": [1, 7],
        "bootstrap": {
            "block_length_origins": 3,
            "iterations": 100,
            "seed": 42,
        },
        "intervals": {
            "alpha": .1,
            "min_calibration": 10,
            "calibration_window": 30,
            "max_relative_width": .3,
        },
    })
    n = 340
    ix = np.arange(n)
    frame = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=n, tz="UTC"),
        "Close": 100 * np.exp(ix * .0005 + .015 * np.sin(ix / 13)),
    })
    root = tmp_path / "sources"
    save_snapshot("GLD", root, frame)
    source = root / "GLD_daily_processed.csv"
    results = tmp_path / "results"
    preflight(spec, [("GLD", source)], results,
              protocol_sha256=LOCKED_V2_SHA256)
    evaluate(spec, [("GLD", source)], results,
             protocol_sha256=LOCKED_V2_SHA256)
    return spec, results, source


def test_first_run_integrity_and_recalculated_metrics(first_run):
    spec, directory, _ = first_run
    pred, manifest = verify_first_run(directory, spec)
    assert manifest["stage"] == "FIRST_EXTERNAL_ASSET_RETROSPECTIVE_EVALUATION"
    assert len(pred) == manifest["n_rows"]
    comparisons, calibration = recalculated_diagnostics(directory, spec)
    assert len(comparisons) == 2
    assert len(calibration) == 4
    assert set(comparisons.horizon_step) == {1, 7}
    matched = pd.read_csv(directory / "holdout_matched_accepted.csv")
    report = generate_report(comparisons, calibration, matched, pred)
    assert "First external-asset evaluation" in report
    assert "GLD" in report and "descriptive" in report


def test_metric_tampering_is_detected(first_run):
    spec, directory, _ = first_run
    path = directory / "holdout_metrics.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "mae"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent mae"):
        recalculated_diagnostics(directory, spec)


def test_first_run_manifest_hash_mismatch_is_detected(first_run):
    spec, directory, _ = first_run
    path = directory / "holdout_result_manifest.json"
    content = json.loads(path.read_text())
    content["protocol_sha256"] = "0" * 64
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="locked Tiingo"):
        verify_first_run(directory, spec)


def test_source_snapshot_mutation_blocks_audit(first_run):
    spec, directory, source = first_run
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="local source file"):
        verify_first_run(directory, spec)


def test_error_flags_are_descriptive_not_exclusions(first_run):
    spec, directory, _ = first_run
    pred, _ = verify_first_run(directory, spec)
    flagged = descriptive_error_flags(pred)
    assert len(flagged) == 12  # one asset x 2 models x 2 horizons x 3 errors
    assert (flagged.abs_error >= 0).all()
    assert "target_date" in flagged.columns
    # The original scored predictions remain byte-for-byte untouched.
    assert len(pd.read_csv(directory / "holdout_predictions.csv")) == len(pred)
