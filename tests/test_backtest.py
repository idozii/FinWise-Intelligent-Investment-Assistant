import numpy as np
import pandas as pd
import pytest

from research.backtest import load_price_csv, predict_prices, summarize, walk_forward


def daily_frame(n=90, *, slope=1.0):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        "close": 100.0 + slope * np.arange(n),
    })


def test_walk_forward_origin_count_and_strict_temporal_boundary():
    frame = daily_frame()
    outcome = walk_forward(frame, min_train=50, horizon=5, step=5,
                           models=("naive", "linear"))
    assert outcome.origin_index.nunique() == 8
    assert len(outcome) == 8 * 5 * 2
    for row in outcome.itertuples():
        assert row.target_date > row.train_end
        assert row.actual == frame.iloc[row.origin_index + row.horizon_step - 1].close


def test_future_perturbation_does_not_affect_earlier_origin():
    original = daily_frame()
    modified = original.copy()
    modified.loc[65:, "close"] += 10000
    baseline = walk_forward(original, min_train=50, horizon=5, step=5,
                            models=("naive", "linear", "moving_average"))
    changed = walk_forward(modified, min_train=50, horizon=5, step=5,
                           models=("naive", "linear", "moving_average"))
    before = baseline.loc[baseline.origin_index <= 65, "predicted"].to_numpy()
    after = changed.loc[changed.origin_index <= 65, "predicted"].to_numpy()
    np.testing.assert_allclose(before, after)


def test_linear_trend_beats_naive_on_exact_line():
    results = walk_forward(daily_frame(), min_train=50, horizon=5, step=5,
                           models=("naive", "linear"))
    summary = summarize(results)
    linear = summary[summary.model == "linear"]
    assert linear.mae.max() < 1e-8
    assert (linear.mae_improvement_vs_naive_pct > 99.99).all()
    assert (linear.directional_accuracy_nonflat == 1.0).all()
    naive = summary[summary.model == "naive"]
    assert naive.n_directional_evaluable.eq(0).all()


def test_flat_series_without_naive_error_has_undefined_improvement():
    prices = daily_frame(slope=0.0)
    result = summarize(walk_forward(prices, min_train=50, horizon=3, step=3,
                                    models=("naive", "moving_average")))
    assert result.mae_improvement_vs_naive_pct.isna().all()
    assert result.mae.max() == pytest.approx(0)


def test_csv_refuses_mixed_assets_and_duplicate_days(tmp_path):
    path = tmp_path / "coins.csv"
    pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z",
                                "2025-01-02T00:00:00Z"],
                  "price": [100.0, 101.0, 102.0],
                  "coin": ["bitcoin", "bitcoin", "ethereum"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Multiple assets"):
        load_price_csv(path, date_col="timestamp", price_col="price", asset_col="coin")
    with pytest.raises(ValueError, match="Multiple prices"):
        load_price_csv(path, date_col="timestamp", price_col="price",
                       asset_col="coin", asset="bitcoin", aggregate_daily_last=False)


def test_csv_explicit_daily_aggregation_uses_last_observation(tmp_path):
    path = tmp_path / "coins.csv"
    pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z",
                                "2025-01-02T00:00:00Z"],
                  "price": [100.0, 101.0, 103.0], "coin": ["bitcoin"] * 3}).to_csv(path, index=False)
    frame = load_price_csv(path, date_col="timestamp", price_col="price",
                           asset_col="coin", asset="bitcoin", aggregate_daily_last=True)
    assert frame.close.tolist() == [101.0, 103.0]


def test_reject_bad_price_or_short_data(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2025-01-01", "2025-01-02"], "Close": [100, -1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly positive"):
        load_price_csv(path)
    with pytest.raises(ValueError, match="Not enough"):
        walk_forward(daily_frame(42), min_train=40, horizon=7)


def test_fallback_is_labelled_for_short_holt_history():
    predicted, effective_model = predict_prices(np.arange(10, dtype=float) + 100, 2, "holt_winters")
    assert len(predicted) == 2
    assert effective_model == "linear_fallback"
