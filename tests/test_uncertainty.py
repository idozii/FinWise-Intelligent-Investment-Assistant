import numpy as np
import pandas as pd
import pytest

from research.uncertainty import finite_sample_radius, intervals, report


def toy_predictions(n=55, *, horizon=1, step=1, model="ridge"):
    dates = pd.date_range("2025-01-01", periods=n * step + horizon + 3, tz="UTC")
    rows = []
    for k in range(n):
        i = 40 + k * step
        rows.append({
            "asset": "AAPL", "model": model, "horizon_step": horizon,
            "origin_index": i,
            "train_end": dates[k * step].date().isoformat(),
            "target_date": dates[k * step + horizon].date().isoformat(),
            "last_observed_close": 100.0, "actual": 101.0 + (k % 2),
            "predicted": 100.0,
        })
    return pd.DataFrame(rows)


def test_finite_sample_quantile_is_conservative_rank():
    assert finite_sample_radius([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.2) == 8
    assert np.isinf(finite_sample_radius([1], 0.1))
    with pytest.raises(ValueError):
        finite_sample_radius([1, 2], 0)


def test_first_forecast_never_calibrates_on_own_label():
    x = toy_predictions(n=10)
    annotated = intervals(x, alpha=0.5, min_calibration=2, calibration_window=5)
    assert not annotated.iloc[0].interval_ready
    assert not annotated.iloc[1].interval_ready
    assert annotated.iloc[2].interval_ready


def test_far_future_label_does_not_mature_early():
    x = toy_predictions(n=15, horizon=7)
    annotated = intervals(x, alpha=0.5, min_calibration=2, calibration_window=10)
    assert annotated.loc[6, "calibration_n"] == 0
    assert annotated.loc[7, "calibration_n"] == 1
    assert annotated.loc[8, "calibration_n"] == 2
    assert annotated.loc[8, "interval_ready"]


def test_future_actual_change_does_not_change_past_intervals():
    x = toy_predictions(n=40, horizon=2)
    changed = x.copy()
    changed.loc[25:, "actual"] += 1000
    a = intervals(x, min_calibration=5, calibration_window=10)
    b = intervals(changed, min_calibration=5, calibration_window=10)
    for key in ("interval_lower", "interval_upper", "relative_width",
                "calibration_n", "accept_forecast"):
        pd.testing.assert_series_equal(a.loc[:25, key], b.loc[:25, key])


def test_non_leaking_abstention_from_width_only():
    x = toy_predictions(n=40)
    narrow = intervals(x, min_calibration=3, calibration_window=10,
                       max_relative_width=0.09)
    strict = intervals(x, min_calibration=3, calibration_window=10,
                       max_relative_width=0.001)
    assert narrow.accept_forecast.sum() > strict.accept_forecast.sum()
    assert not strict.accept_forecast.any()
    metrics = report(narrow)
    assert metrics.n_origins.iloc[0] == 40
    assert metrics.n_calibrated.iloc[0] > 0


def test_models_and_assets_calibrated_separately():
    a = toy_predictions(n=15, model="ridge")
    b = toy_predictions(n=15, model="xgb")
    b["asset"] = "MSFT"
    b["actual"] += 1000
    result = intervals(pd.concat([a, b], ignore_index=True),
                       alpha=0.5, min_calibration=2, calibration_window=5)
    assert set(result.asset) == {"MSFT", "AAPL"}
    assert result.query("asset == 'AAPL'").radius.dropna().max() < 10
    assert result.query("asset == 'MSFT'").radius.dropna().min() > 900


def test_reject_duplicate_origin_and_future_targets():
    x = toy_predictions(n=10)
    with pytest.raises(ValueError, match="Duplicate"):
        intervals(pd.concat([x, x.iloc[:1]]))
    x.loc[0, "target_date"] = x.loc[0, "train_end"]
    with pytest.raises(ValueError, match="strictly later"):
        intervals(x)
