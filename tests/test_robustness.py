import numpy as np
import pandas as pd
import pytest

from research.robustness import (
    origin_volatility, paired_block_bootstrap, analyze_asset, parse_assets
)


def mock_prices(n=110):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, tz="UTC"),
        "close": 100 * np.exp(0.001 * np.arange(n) +
                              0.008 * np.sin(np.arange(n) / 5)),
    })


def test_regime_is_past_only():
    base = mock_prices()
    changed = base.copy()
    changed.loc[90:, "close"] *= 3
    first = origin_volatility(base, 70)
    second = origin_volatility(changed, 70)
    assert {i: first[i] for i in range(70, 91)} == {
        i: second[i] for i in range(70, 91)
    }


def test_bootstrap_identical_model_has_zero_loss_difference():
    rows = []
    for origin in range(30):
        for model in ("naive", "ridge"):
            rows.append({"origin_index": origin, "model": model,
                         "actual": float(origin + 10),
                         "predicted": float(origin + 11)})
    a = pd.DataFrame(rows)
    mean, lo, hi = paired_block_bootstrap(a, iterations=100)
    assert (mean, lo, hi) == pytest.approx((0, 0, 0))


def test_bootstrap_detects_consistent_underperformance():
    rows = []
    for origin in range(30):
        rows.extend([
            {"origin_index": origin, "model": "naive", "actual": 100., "predicted": 101.},
            {"origin_index": origin, "model": "ridge", "actual": 100., "predicted": 110.},
        ])
    assert paired_block_bootstrap(pd.DataFrame(rows), iterations=100) == pytest.approx((9, 9, 9))


def test_multia_asset_outputs_are_paired_and_labeled(tmp_path):
    frame = mock_prices()
    path = tmp_path / "MSFT.csv"
    pd.DataFrame({
        "Date": frame.date, "Close": frame.close
    }).to_csv(path, index=False)
    predictions, comparisons, regimes, meta = analyze_asset(
        "MSFT", path, min_train=70, horizons=(1, 7), step=7,
        models=("naive", "ridge"), block_length=3, bootstrap_iterations=100,
    )
    assert set(predictions.asset) == {"MSFT"}
    assert set(predictions.regime).issubset({"low", "medium", "high"})
    assert comparisons.shape[0] == 2
    assert comparisons.n_origins.min() > 0
    assert set(regimes.model) == {"naive", "ridge"}
    assert len(meta["csv_sha256"]) == 64
    assert predictions.target_date.gt(predictions.train_end).all()


def test_asset_parser_rejects_duplicate_assets():
    with pytest.raises(ValueError, match="repeated"):
        parse_assets(["AAPL=one.csv", "AAPL=two.csv"])
    assert parse_assets(["MSFT=/tmp/msft.csv"]) == [
        ("MSFT", __import__("pathlib").Path("/tmp/msft.csv"))]
