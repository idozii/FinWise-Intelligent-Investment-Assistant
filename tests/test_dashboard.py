import json
from pathlib import Path

import pandas as pd
import pytest

from research.dashboard import build_payload, generate


def write_inputs(tmp_path, asset="AAPL"):
    robust = tmp_path / "robust"
    interval = tmp_path / "interval"
    robust.mkdir()
    interval.mkdir()
    observations = []
    for origin in (70, 77, 84):
        for model, predicted in (("naive", 100.0), ("ridge", 100.5)):
            observations.append({
                "asset": asset, "model": model, "horizon_step": 1,
                "origin_index": origin, "train_end": "2025-01-01",
                "actual": 101.0, "predicted": predicted
            })
    pd.DataFrame(observations).to_csv(robust / "asset_predictions.csv", index=False)
    pd.DataFrame([{
        "asset": asset, "model": "ridge", "horizon_step": 1,
        "mae_delta_model_minus_naive": -0.5,
        "bootstrap_ci_low": -0.5, "bootstrap_ci_high": -0.5,
        "n_origins": 3
    }]).to_csv(robust / "paired_comparisons.csv", index=False)
    pd.DataFrame([{
        "asset": asset, "model": "ridge", "regime": "medium",
        "horizon_step": 1, "mae": 0.5, "n_origins": 3
    }]).to_csv(robust / "regime_metrics.csv", index=False)
    pd.DataFrame([{
        "asset": asset, "model": "ridge", "horizon_step": 1,
        "empirical_coverage_ready": .9, "n_calibrated": 2,
        "acceptance_rate_all_origins": .5, "coverage_accepted": 1.,
        "mean_relative_width_ready": .04, "mae_accepted": .5
    }]).to_csv(interval / "interval_metrics.csv", index=False)
    return robust, interval


def test_dashboard_reports_actual_csv_values_not_mock_claims(tmp_path):
    robust, interval = write_inputs(tmp_path)
    data = build_payload(robust, interval)
    baseline = next(x for x in data["model_metrics"] if x["model"] == "naive")
    ridge = next(x for x in data["model_metrics"] if x["model"] == "ridge")
    assert baseline["mae"] == pytest.approx(1.)
    assert ridge["mae"] == pytest.approx(.5)
    assert data["stage"] == "EXPLORATORY_NOT_INDEPENDENT_HOLDOUT"
    assert all(len(digest) == 64 for digest in data["source_sha256"].values())
    assert len(data["assets"]) == 1


def test_dashboard_escape_embedded_html_asset_string(tmp_path):
    robust, interval = write_inputs(tmp_path, asset="AAPL</script><script>alert(1)</script>")
    dest = tmp_path / "report.html"
    generate(robust, interval, dest)
    content = dest.read_text()
    assert "AAPL</script><script>alert(1)</script>" not in content
    assert "AAPL\\u003c/script\\u003e" in content
    assert "EXPLORATORY" in content
    assert "SHAP" in content
    assert "not investment advice" in content.lower()


def test_dashboard_missing_file_or_column_fails_closed(tmp_path):
    robust, interval = write_inputs(tmp_path)
    (interval / "interval_metrics.csv").unlink()
    with pytest.raises(FileNotFoundError):
        build_payload(robust, interval)
