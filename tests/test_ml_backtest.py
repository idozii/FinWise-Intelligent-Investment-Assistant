import numpy as np
import pandas as pd
import pytest

from research.ml_backtest import (
    FEATURE_NAMES, LOOKBACK, features_at, historical_training_examples,
    fit_predict, ml_walk_forward,
)
from research.backtest import summarize


def sample(n=105):
    prices = 150 * np.exp(0.0007 * np.arange(n) + 0.01 * np.sin(np.arange(n) / 7))
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        "close": prices,
    })


def test_features_use_only_history():
    raw = np.log(sample().close.to_numpy())
    changed = raw.copy()
    changed[80:] += 1
    np.testing.assert_array_equal(features_at(raw, 79), features_at(changed, 79))
    assert len(features_at(raw, 79)) == len(FEATURE_NAMES)


def test_training_labels_must_mature_before_forecast():
    raw = np.log(sample().close.to_numpy())
    origin, horizon = 80, 7
    x, y = historical_training_examples(raw, origin, horizon)
    assert len(x) == origin - horizon - LOOKBACK
    assert len(y) == len(x)
    assert y[-1] == pytest.approx(raw[origin - 1] - raw[origin - 1 - horizon])
    changed = raw.copy()
    changed[origin:] += 50
    x2, y2 = historical_training_examples(changed, origin, horizon)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)


def test_ml_origin_target_alignment_and_naive_baseline():
    frame = sample()
    results = ml_walk_forward(frame, min_train=70, horizons=(1, 7), step=7,
                              models=("naive", "ridge"))
    assert results.origin_index.nunique() == 5
    assert len(results) == 5 * 2 * 2
    for row in results.itertuples():
        assert row.target_date > row.train_end
        assert row.actual == pytest.approx(frame.iloc[row.origin_index - 1 + row.horizon_step].close)
        assert row.last_observed_close == pytest.approx(frame.iloc[row.origin_index - 1].close)
        if row.model == "naive":
            assert row.predicted == pytest.approx(row.last_observed_close)


def test_modified_future_cannot_change_previous_origin_predictions():
    raw = np.log(sample().close.to_numpy())
    changed = raw.copy()
    changed[80:] += 0.9
    for model in ("naive", "ridge"):
        for horizon in (1, 7):
            assert fit_predict(raw, 80, horizon, model) == pytest.approx(
                fit_predict(changed, 80, horizon, model)
            )


def test_fixed_ridge_is_reproducible_and_metrics_pair_naive():
    frame = sample()
    a = ml_walk_forward(frame, min_train=70, horizons=(1, 7), step=14,
                        models=("naive", "ridge"))
    b = ml_walk_forward(frame, min_train=70, horizons=(1, 7), step=14,
                        models=("naive", "ridge"))
    pd.testing.assert_frame_equal(a, b)
    metric = summarize(a)
    assert set(metric.model) == {"naive", "ridge"}
    assert len(metric) == 4
    assert metric.mae.notna().all()
    assert metric.loc[metric.model == "naive", "n_directional_evaluable"].eq(0).all()


def test_eval_dates_select_forecast_origins_only():
    frame = sample()
    dates = frame.date.dt.date.astype(str)
    start = dates.iloc[77]
    end = dates.iloc[91]
    result = ml_walk_forward(frame, min_train=70, horizons=(1, 7), step=7,
                             models=("naive",), eval_start=start, eval_end=end)
    assert result.train_end.min() >= start
    assert result.train_end.max() <= end


def test_reject_invalid_experiment():
    frame = sample()
    with pytest.raises(ValueError, match="naive"):
        ml_walk_forward(frame, min_train=70, models=("ridge",))
    with pytest.raises(ValueError, match="horizons"):
        ml_walk_forward(frame, min_train=70, horizons=(0,), models=("naive",))
    with pytest.raises(ValueError, match="Not enough"):
        ml_walk_forward(frame.iloc[:35], min_train=70, models=("naive",))


def test_xgb_and_lightgbm_optional_single_origin():
    frame = sample()
    raw = np.log(frame.close.to_numpy())
    for model in ("xgb", "lightgbm"):
        pytest.importorskip("xgboost" if model == "xgb" else "lightgbm")
        prediction = fit_predict(raw, 80, 7, model)
        assert np.isfinite(prediction) and prediction > 0
