import numpy as np
import pandas as pd
import pytest

from research.calibration_audit import (
    matched_selection_audit, model_audit, regime_audit,
    validated_rows, width_sensitivity,
)


def dataframes():
    source, interval = [], []
    naive_width = [np.nan, np.nan, 0.04, 0.08, 0.12, 0.06]
    ridge_width = [np.nan, np.nan, 0.06, 0.12, 0.08, 0.06]
    for model, widths, predicted in [
        ("naive", naive_width, 100.0),
        ("ridge", ridge_width, 104.0),
    ]:
        for i, width in enumerate(widths):
            date = (pd.Timestamp("2025-01-01") +
                    pd.Timedelta(days=i)).date().isoformat()
            target = (pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat()
            row = {
                "asset": "AAPL", "model": model,
                "horizon_step": 1, "origin_index": i + 40,
                "regime": "high" if i % 2 else "low",
                "train_end": date, "target_date": target,
                "actual": 101.0, "predicted": predicted, "last_observed_close": 100.0,
            }
            source.append(row)
            ready = i >= 2
            lo = (max(0, predicted - width * 50) if ready else np.nan)
            hi = (predicted + width * 50 if ready else np.nan)
            extra = {
                "interval_ready": ready,
                "accept_forecast": bool(ready and width <= 0.1),
                "interval_lower": lo, "interval_upper": hi,
                "relative_width": width, "calibration_n": i if ready else 0,
                "covered": bool(lo <= row["actual"] <= hi) if ready else None,
            }
            interval.append({k: v for k, v in row.items() if k != "regime"} | extra)
    return pd.DataFrame(source), pd.DataFrame(interval)


def store(tmp_path):
    source, interval = dataframes()
    s = tmp_path / "asset_predictions.csv"
    i = tmp_path / "interval_predictions.csv"
    source.to_csv(s, index=False)
    interval.to_csv(i, index=False)
    return s, i


def test_matched_comparisons_share_accepted_origins(tmp_path):
    s, i = store(tmp_path)
    rows = validated_rows(s, i)
    fair = matched_selection_audit(rows).iloc[0]
    assert fair.n_origins == 6
    assert fair.n_both_calibrated == 4
    assert fair.n_both_accepted == 2  # only origins 42 and 45
    assert fair.naive_mae_same_accepted == pytest.approx(1)
    assert fair.model_mae_same_accepted == pytest.approx(3)
    assert fair.delta_mae_model_minus_naive_same_accepted == pytest.approx(2)
    assert fair.comparison_status == "SMALL_N_EXPLORATORY"


def test_model_and_regime_reports_include_denominators(tmp_path):
    s, i = store(tmp_path)
    rows = validated_rows(s, i)
    report = model_audit(rows, alpha=.1)
    assert (report.n_ready == 4).all()
    assert (report.n_accepted == 3).all()
    assert (report.acceptance_rate_ready == .75).all()
    assert (report.acceptance_rate_all == .5).all()
    assert set(regime_audit(rows).regime) == {"low", "high"}


def test_sensitivity_is_monotone_in_selection_not_in_outcome(tmp_path):
    s, i = store(tmp_path)
    rows = validated_rows(s, i)
    tab = width_sensitivity(rows, (.05, .10, .20))
    for model in ("naive", "ridge"):
        data = tab.query("model == @model").sort_values("relative_width_threshold")
        assert data.n_accepted.tolist() == [1 if model == "naive" else 0, 3, 4]
    with pytest.raises(ValueError):
        width_sensitivity(rows, (0,))


def test_source_price_mismatch_is_detected(tmp_path):
    s, i = store(tmp_path)
    source = pd.read_csv(s)
    source.loc[0, "actual"] += 200
    source.to_csv(s, index=False)
    with pytest.raises(ValueError, match="differ"):
        validated_rows(s, i)


def test_reported_coverage_must_match_interval_membership(tmp_path):
    s, i = store(tmp_path)
    interval = pd.read_csv(i)
    interval.loc[interval.interval_ready, "covered"] = False
    interval.to_csv(i, index=False)
    with pytest.raises(ValueError, match="coverage differs"):
        validated_rows(s, i)


def test_unready_cannot_be_accepted(tmp_path):
    s, i = store(tmp_path)
    interval = pd.read_csv(i)
    interval.loc[0, "accept_forecast"] = True
    interval.to_csv(i, index=False)
    with pytest.raises(ValueError, match="Uncalibrated"):
        validated_rows(s, i)


def test_no_overlap_returns_explicit_status_not_fake_delta(tmp_path):
    s, i = store(tmp_path)
    interval = pd.read_csv(i)
    interval.loc[interval.model == "ridge", "accept_forecast"] = False
    interval.to_csv(i, index=False)
    fair = matched_selection_audit(validated_rows(s, i)).iloc[0]
    assert fair.comparison_status == "NO_MATCHED_ACCEPTED"
    assert np.isnan(fair.delta_mae_model_minus_naive_same_accepted)


def test_exact_origin_set_is_required(tmp_path):
    s, i = store(tmp_path)
    interval = pd.read_csv(i)
    interval = interval.iloc[1:].reset_index(drop=True)
    interval.to_csv(i, index=False)
    with pytest.raises(ValueError, match="count mismatch"):
        validated_rows(s, i)
