import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.frozen_eval import (
    LOCKED_PROTOCOL_SHA256, PROTOCOL_FILE, _assets_exact, _dates_only,
    check_preflight, eligible_origin_indices, evaluate, load_locked_protocol,
    preflight,
)


def fixture_spec():
    # Small synthetic test settings only; real lock has a different test window.
    locked = load_locked_protocol()
    locked = json.loads(json.dumps(locked))
    locked["reserved_external_assets"] = ["TEST"]
    locked["prior_explored_assets"] = ["AAPL"]
    locked["forecast_origin_start"] = "2025-03-01"
    locked["forecast_origin_end"] = "2025-12-31"
    locked["min_train"] = 70
    locked["horizons"] = [1, 7]
    locked["step"] = 7
    locked["models"] = ["naive", "ridge"]
    locked["intervals"] = {
        "alpha": 0.1, "min_calibration": 10,
        "calibration_window": 30, "max_relative_width": 0.3
    }
    locked["bootstrap"] = {
        "block_length_origins": 3, "iterations": 100, "seed": 42
    }
    return locked


def make_csv(tmp_path, *, n=340):
    index = np.arange(n)
    dates = pd.date_range("2025-01-01", periods=n, tz="UTC")
    prices = 100. * np.exp(.0007 * index + .007 * np.sin(index / 8))
    target = tmp_path / "TEST.csv"
    pd.DataFrame({"Date": dates, "Close": prices}).to_csv(target, index=False)
    return target


def test_locked_protocol_bytes_and_reserved_assets():
    assert PROTOCOL_FILE.is_file()
    spec = load_locked_protocol()
    assert len(LOCKED_PROTOCOL_SHA256) == 64
    assert spec["models"] == ["naive", "ridge"]
    assert set(spec["reserved_external_assets"]) == {"IWM", "GLD", "XLE", "XLV"}
    assert not (set(spec["reserved_external_assets"])
                & set(spec["prior_explored_assets"]))


def test_protocol_modification_requires_new_explicit_lock(tmp_path):
    altered = tmp_path / "tampered.json"
    altered.write_text(PROTOCOL_FILE.read_text().replace(
        "GLD", "AAPL"
    ))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_locked_protocol(altered)


def test_frozen_asset_set_rejects_explored_or_missing_assets():
    spec = load_locked_protocol()
    with pytest.raises(ValueError, match="Exact frozen external asset set"):
        _assets_exact(spec, ["AAPL=x.csv"])
    valid = _assets_exact(spec, [
        "XLV=xlv.csv", "XLE=xle.csv", "IWM=iwm.csv", "GLD=gld.csv",
    ])
    assert [asset for asset, _ in valid] == ["GLD", "IWM", "XLE", "XLV"]


def test_preflight_does_not_read_close_values(tmp_path):
    spec = fixture_spec()
    file = make_csv(tmp_path)
    content = pd.read_csv(file)
    content["Close"] = "unreadable_price_for_preflight"
    content.to_csv(file, index=False)
    output = tmp_path / "preflight"
    report = preflight(spec, [("TEST", file)], output)
    assert report["stage"] == "PREFLIGHT_ONLY_NO_OUTCOME_REVIEW"
    assert not report["sources"][0]["preflight_inspected_close_values"]
    assert report["sources"][0]["n_eligible_origins"] >= 30
    assert check_preflight(spec, [("TEST", file)], output) == report


def test_preflight_hash_blocks_modified_snapshot(tmp_path):
    spec = fixture_spec()
    file = make_csv(tmp_path)
    output = tmp_path / "preflight"
    preflight(spec, [("TEST", file)], output)
    frame = pd.read_csv(file)
    frame.loc[5, "Close"] += 1
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="digest changed"):
        check_preflight(spec, [("TEST", file)], output)
    with pytest.raises(FileExistsError, match="Preflight changed"):
        preflight(spec, [("TEST", file)], output)


def test_preflight_origin_target_dates_obey_frozen_window(tmp_path):
    spec = fixture_spec()
    file = make_csv(tmp_path)
    dates = _dates_only(file, spec)
    eligible = eligible_origin_indices(dates, spec)
    assert len(eligible) >= 30
    for i in eligible:
        assert dates.iloc[i-1] >= pd.Timestamp("2025-03-01", tz="UTC")
        assert dates.iloc[i-1+7] <= pd.Timestamp("2025-12-31", tz="UTC")


def test_frozen_evaluation_roundtrip_is_gated_by_hashes(tmp_path):
    """Synthetic evaluation exercises the actual prequential M2/M3/M3.1 stack."""
    spec = fixture_spec()
    file = make_csv(tmp_path)
    output = tmp_path / "test"
    preflight(spec, [("TEST", file)], output)
    report = evaluate(spec, [("TEST", file)], output)
    assert report["stage"] == "FIRST_EXTERNAL_ASSET_RETROSPECTIVE_EVALUATION"
    assert report["n_rows"] > 50
    files = [
        "holdout_metrics.csv", "holdout_paired_comparisons.csv",
        "holdout_calibration.csv", "holdout_regime_coverage.csv",
        "holdout_matched_accepted.csv",
    ]
    for name in files:
        assert (output / name).is_file()
    pred = pd.read_csv(output / "holdout_predictions.csv")
    assert set(pred.model) == {"naive", "ridge"}
    assert set(pred.asset) == {"TEST"}
    assert pred.target_date.le(spec["forecast_origin_end"]).all()
    assert pred.train_end.ge(spec["forecast_origin_start"]).all()
    fair = pd.read_csv(output / "holdout_matched_accepted.csv")
    assert len(fair) == 2
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate(spec, [("TEST", file)], output)


def test_missing_file_preflight_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        preflight(fixture_spec(), [("TEST", tmp_path / "missing.csv")],
                  tmp_path / "out")
