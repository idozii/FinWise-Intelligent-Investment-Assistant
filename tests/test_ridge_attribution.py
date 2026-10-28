import numpy as np
import pandas as pd
import pytest

from research.ml_backtest import FEATURE_NAMES
from research.ridge_attribution import explain_at_origin, build_attributions


def example_prices(n=115):
    idx = np.arange(n)
    close = 120.0 * np.exp(.0009 * idx + .012 * np.sin(idx / 9))
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, tz="UTC"),
        "close": close,
    })


def test_exact_additive_reconstruction_matches_production_m2_formula():
    logs = np.log(example_prices().close.to_numpy())
    for horizon in (1, 7):
        table, audit = explain_at_origin(logs, origin=75, horizon=horizon)
        assert list(table.feature) == list(FEATURE_NAMES)
        assert audit["max_abs_reconstruction_error"] < 1e-8
        assert audit["predicted_log_return"] == pytest.approx(
            audit["intercept_log_return"] +
            table.contribution_log_return.sum(), abs=1e-12
        )
        assert audit["predicted_close"] > 0


def test_explanations_never_observe_prices_after_origin():
    logs = np.log(example_prices().close.to_numpy())
    changed = logs.copy()
    changed[80:] += 4.0
    for horizon in (1, 7):
        table_a, report_a = explain_at_origin(logs, origin=80, horizon=horizon)
        table_b, report_b = explain_at_origin(changed, origin=80, horizon=horizon)
        pd.testing.assert_frame_equal(table_a, table_b)
        assert report_a["predicted_close"] == pytest.approx(report_b["predicted_close"])


def test_historical_origin_subsampling_has_valid_dates_and_features():
    all_data, summary, verified = build_attributions(
        example_prices(), "AAPL", min_train=70, horizons=(1, 7),
        step=7, max_origins=3,
    )
    assert len(verified) == 6
    assert len(all_data) == len(verified) * len(FEATURE_NAMES)
    assert set(summary.feature) == set(FEATURE_NAMES)
    assert all_data.target_date.gt(all_data.train_end).all()
    assert verified.max_abs_reconstruction_error.max() < 1e-8
    assert set(summary.attribution_type) == {
        "exact_linear_model_decomposition_not_SHAP_not_causal"
    }


def test_reject_short_training_history_or_bad_max_origins():
    with pytest.raises(ValueError):
        explain_at_origin(np.log(example_prices().close.to_numpy()), origin=32, horizon=7)
    with pytest.raises(ValueError, match="positive"):
        build_attributions(example_prices(), "AAPL", min_train=70, max_origins=0)


def test_future_actual_change_does_not_change_earlier_feature_attribution():
    frame = example_prices()
    mutated = frame.copy()
    mutated.loc[80:, "close"] *= 3
    a, _, _ = build_attributions(
        frame, "AAPL", min_train=70, horizons=(1,), step=7, max_origins=10
    )
    b, _, _ = build_attributions(
        mutated, "AAPL", min_train=70, horizons=(1,), step=7, max_origins=10
    )
    a = a[a.origin_index <= 80].reset_index(drop=True)
    b = b[b.origin_index <= 80].reset_index(drop=True)
    for col in ("raw_feature_value", "standardized_feature_value",
                "ridge_standardized_coefficient", "contribution_log_return",
                "predicted_close"):
        np.testing.assert_allclose(a[col], b[col], rtol=0, atol=1e-10)
