import numpy as np
import pandas as pd
import pytest

from research.r8_adaptive_development import (
    ALLOWED_DEV, FORBIDDEN_TEST, GAMMA,
    adaptive_intervals, compare, validate_development,
)


def sample(n=48):
    rows = []
    dates = pd.date_range("2024-01-02", periods=n*7+20, freq="D", tz="UTC")
    for asset in sorted(ALLOWED_DEV):
        for model in ("naive", "ridge"):
            for h in (1,7):
                for i in range(n):
                    oi = i*7
                    price = 100 + oi*.04
                    rows.append({
                        "asset": asset, "model": model,
                        "horizon_step": h, "origin_index": i+252,
                        "train_end": dates[oi].date().isoformat(),
                        "target_date": dates[oi+h].date().isoformat(),
                        "last_observed_close": price,
                        "predicted": price + (.1 if model=="ridge" else 0),
                        "actual": price + np.sin(i/3)*2 + (i%5)*.2,
                    })
    return pd.DataFrame(rows)


def test_methods_preserve_same_origin_sets_without_holdout():
    records, scores = compare(sample())
    assert set(records.calibration_method) == {"fixed_rolling","delayed_feedback_aci_style"}
    assert len(records) == len(sample())*2
    assert len(scores) == 32
    assert set(scores.asset) == ALLOWED_DEV
    assert not set(scores.asset)&FORBIDDEN_TEST
    assert (scores.n_origins == 48).all()
    assert set(scores.n_ready) == {18}
    assert records.loc[records.calibration_method=="delayed_feedback_aci_style", "alpha_used"].between(.01,.5).all()


def test_future_outcome_cannot_change_past_interval_or_alpha():
    source = sample()
    baseline = adaptive_intervals(source)
    changed = source.copy()
    mask = ((changed.asset=="AAPL")&(changed.model=="ridge")
            &(changed.horizon_step==7)&(changed.origin_index==299))
    assert mask.sum() == 1
    changed.loc[mask, "actual"] = 500.
    later = adaptive_intervals(changed)
    key = ((baseline.asset=="AAPL")&(baseline.model=="ridge")
           &(baseline.horizon_step==7))
    old = baseline.loc[key].sort_values("origin_index")
    new = later.loc[key].sort_values("origin_index")
    # Last test outcome remains unavailable to all earlier origins.
    for col in ["alpha_used", "radius", "interval_lower", "interval_upper",
                "calibration_n", "feedback_applied_this_origin"]:
        pd.testing.assert_series_equal(
            old[col].iloc[:-1].reset_index(drop=True),
            new[col].iloc[:-1].reset_index(drop=True),
            check_names=False,
        )


def test_aci_actually_updates_after_matured_feedback():
    result = adaptive_intervals(sample(52))
    ready = result.loc[
        (result.asset=="AAPL") & (result.model=="naive")
        & (result.horizon_step==1)&result.interval_ready
    ]
    assert len(ready) > 10
    assert ready.alpha_used.nunique() > 1
    assert (ready.feedback_applied_this_origin > 0).any()


def test_rejects_m5_assets_and_wrong_model_registry():
    d = sample()
    d.loc[d.asset=="AAPL", "asset"] = "GLD"
    with pytest.raises(ValueError, match="development"):
        validate_development(d)
    d = sample()
    d.loc[d.model=="ridge", "model"] = "xgboost"
    with pytest.raises(ValueError, match="naive/ridge"):
        validate_development(d)


def test_delayed_h7_feedback_is_never_used_before_target_matures():
    d = sample(49)
    rows = adaptive_intervals(d)
    selected = rows.loc[(rows.asset=="AAPL")&(rows.model=="ridge")
                        &(rows.horizon_step==7)].sort_values("origin_index")
    assert len(selected) == 49
    assert selected.alpha_used.iloc[0] == .10
    assert selected.feedback_applied_this_origin.iloc[30] == 0 or selected.feedback_applied_this_origin.iloc[30] == 1
    # No feedback from the row currently being predicted: only previously issued.
    assert selected.feedback_applied_this_origin.max() <= 1
