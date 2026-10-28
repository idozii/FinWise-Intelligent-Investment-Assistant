import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.r6_corporate_action_audit import (
    REGISTRY, annotate, audit, load_events, split_day_prices,
)
from research.frozen_eval import evaluate, preflight
from research.frozen_eval_tiingo import LOCKED_V2_SHA256, load_v2_protocol
from scripts.m5_download_tiingo import save_snapshot


def local_prices(tmp_path):
    p = tmp_path / "XLE.csv"
    dates = pd.bdate_range("2025-11-28", "2025-12-17", tz="UTC")
    closes = np.array([90., 91., 92., 45.8, 46., 46.5, 47., 47.2, 47.3, 47.4, 47.6, 47.8, 48., 48.2])
    assert len(dates) == len(closes)
    pd.DataFrame({"Date": dates, "Close": closes}).to_csv(p, index=False)
    return p


def test_official_event_record_is_not_a_heuristic():
    events = load_events()
    e = events[0]
    assert e["asset"] == "XLE"
    assert e["announcement_date"] == "2025-11-20"
    assert e["effective_date"] == "2025-12-05"
    assert e["ratio_new_per_old"] == 2
    assert "statestreet.com" in e["source_url"]


def test_split_scale_uses_adjacent_trading_sessions(tmp_path):
    p = local_prices(tmp_path)
    event = load_events()[0]
    d = split_day_prices(p, event)
    assert d["split_bar_available"]
    assert d["previous_trading_date"] == "2025-12-04"
    assert d["split_trading_date"] == "2025-12-05"
    assert abs(d["raw_price_ratio"] - .5) < .03
    assert abs(d["hypothetical_split_neutral_return"]) < .1


def test_crossing_and_post_split_origin_flags_do_not_rewrite_predictions(tmp_path):
    p = local_prices(tmp_path)
    e = load_events()
    forecasts = pd.DataFrame([
        {"asset": "XLE", "model": "naive", "horizon_step": 7,
         "origin_index": 1, "train_end": "2025-12-02",
         "target_date": "2025-12-11", "actual": 47., "predicted": 92.},
        {"asset": "XLE", "model": "ridge", "horizon_step": 7,
         "origin_index": 2, "train_end": "2025-12-11",
         "target_date": "2025-12-22", "actual": 48., "predicted": 48.1},
        {"asset": "GLD", "model": "naive", "horizon_step": 7,
         "origin_index": 2, "train_end": "2025-12-02",
         "target_date": "2025-12-11", "actual": 100., "predicted": 99.},
    ])
    snapshot = forecasts.copy(deep=True)
    annotated, check = annotate(forecasts, e, {"XLE": p})
    assert annotated.crosses_verified_split.tolist() == [True, False, False]
    assert annotated.post_split_origin_first_30_trading_days.tolist() == [False, True, False]
    assert annotated.event_announced_before_origin.tolist() == [True, True, False]
    assert check.loc[0, "asset"] == "XLE"
    pd.testing.assert_frame_equal(forecasts, snapshot)


def test_audit_rejects_same_output_location(tmp_path):
    with pytest.raises(ValueError, match="cannot overwrite"):
        audit(tmp_path, tmp_path)


def test_full_synthetic_first_run_audit_leaves_metrics_unchanged(tmp_path):
    spec = json.loads(json.dumps(load_v2_protocol()))
    spec.update({
        "reserved_external_assets": ["XLE"], "min_train": 70,
        "step": 7, "horizons": [1, 7],
        "forecast_origin_start": "2025-03-01",
        "forecast_origin_end": "2025-12-31",
        "bootstrap": {"block_length_origins": 3, "iterations": 100, "seed": 42},
        "intervals": {"alpha": .1, "min_calibration": 10,
                      "calibration_window": 30, "max_relative_width": .3},
    })
    n = 340
    idx = np.arange(n)
    dates = pd.date_range("2025-01-01", periods=n, tz="UTC")
    # Synthetic sequence with artificial stock split on the official split date.
    close = 100 * np.exp(.0004 * idx + .01 * np.sin(idx / 8))
    close[dates >= pd.Timestamp("2025-12-05", tz="UTC")] /= 2
    raw = pd.DataFrame({"Date": dates, "Close": close})
    folder = tmp_path / "prices"
    save_snapshot("XLE", folder, raw)
    file = folder / "XLE_daily_processed.csv"
    original = tmp_path / "original"
    preflight(spec, [("XLE", file)], original, protocol_sha256=LOCKED_V2_SHA256)
    evaluate(spec, [("XLE", file)], original, protocol_sha256=LOCKED_V2_SHA256)
    baseline = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
    result = audit(original, tmp_path / "r6")
    assert result["excluded_origins"] == 0
    assert result["refit"] is False
    for name, content in baseline.items():
        assert (original/name).read_bytes() == content
    df = pd.read_csv(tmp_path / "r6" / "r6_event_impact_descriptive.csv")
    assert len(df) == 4
    assert df.loc[(df.asset == "XLE") & (df.horizon_step == 7), "n_crossing_split"].max() >= 1
