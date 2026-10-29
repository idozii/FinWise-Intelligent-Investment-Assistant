import json
import numpy as np
import pandas as pd
import pytest

from research.r8_adaptive_development import compare
from research.r9_selective_risk import (
    METHODS, THRESHOLDS, load_dev_input, policy_rows, validate_r8_intervals,
)


def development_predictions(n=42):
    rows=[]
    dates=pd.date_range("2025-01-01",periods=n*8+20,tz="UTC")
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for model in ("naive","ridge"):
            for h in (1,7):
                for i in range(n):
                    start=i*7
                    last=100+i*.1
                    rows.append({
                        "asset":asset,"model":model,"horizon_step":h,
                        "origin_index":252+i,
                        "train_end":dates[start].date().isoformat(),
                        "target_date":dates[start+h].date().isoformat(),
                        "last_observed_close":last,
                        "actual":last+1.8*np.sin(i/5)+.2,
                        "predicted":last+(0 if model=="naive" else .1),
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def records():
    return compare(development_predictions())[0]


def test_5_thresholds_same_origin_risk_and_correct_nans(records):
    curve, matched=policy_rows(records)
    assert len(curve)==4*2*2*2*5
    assert len(matched)==4*2*2*5
    assert set(curve.max_relative_width)==set(THRESHOLDS)
    assert set(curve.calibration_method)==METHODS
    assert (curve.n_accepted<=curve.n_ready).all()
    assert (curve.acceptance_rate_all<=1).all()
    assert (curve.acceptance_rate_ready<=1).all()
    original=records.loc[(records.asset=="AAPL")&
                         (records.model=="ridge")&
                         (records.horizon_step==1)&
                         (records.calibration_method=="fixed_rolling")]
    selected=curve.loc[(curve.asset=="AAPL")&
                      (curve.model=="ridge")&
                      (curve.horizon_step==1)&
                      (curve.calibration_method=="fixed_rolling")&
                      (curve.max_relative_width==.10)].iloc[0]
    assert selected.n_accepted==original.accept_forecast.sum()
    assert selected.n_ready==original.interval_ready.sum()


def test_empty_acceptance_has_missing_not_zero_risk(records):
    # Construct all positive widths above all configured thresholds.
    data=records.copy()
    ready=data.interval_ready
    data.loc[ready,"relative_width"]=.40
    data.loc[ready,"interval_upper"] = (
        data.loc[ready,"predicted"]+.2*data.loc[ready,"last_observed_close"]
    )
    data.loc[ready,"interval_lower"] = (
        data.loc[ready,"predicted"]-.2*data.loc[ready,"last_observed_close"]
    )
    data.loc[ready,"covered"] = (
        (data.loc[ready,"actual"]>=data.loc[ready,"interval_lower"])&
        (data.loc[ready,"actual"]<=data.loc[ready,"interval_upper"])
    )
    data.loc[ready,"accept_forecast"]=False
    curve, matched=policy_rows(data)
    assert (curve.n_accepted==0).all()
    assert curve.mae_selected.isna().all()
    assert curve.interval_coverage_selected.isna().all()
    assert (curve.selection_warning=="NO_ACCEPTED").all()
    assert matched.ridge_minus_naive_mae_same_accepted.isna().all()
    assert (matched.status=="NO_JOINT_ACCEPTANCE").all()


def test_invalid_accepted_flag_and_covered_flag_rejected(records):
    bad=records.copy()
    first=bad.loc[bad.interval_ready].index[0]
    bad.loc[first,"accept_forecast"]=not bool(bad.loc[first,"accept_forecast"])
    with pytest.raises(ValueError,match="acceptance"):
        validate_r8_intervals(bad)
    bad=records.copy()
    bad.loc[first,"covered"]=not bool(bad.loc[first,"covered"])
    with pytest.raises(ValueError,match="Covered flags"):
        validate_r8_intervals(bad)


def test_no_holdout_asset_and_no_cross_method_forecast_tampering(records):
    bad=records.copy()
    bad.loc[bad.asset=="AAPL","asset"]="GLD"
    with pytest.raises(ValueError,match="exploratory"):
        validate_r8_intervals(bad)
    bad=records.copy()
    mask=(bad.asset=="AAPL")&(bad.model=="ridge")&(
        bad.calibration_method=="delayed_feedback_aci_style"
    )&(bad.origin_index==252)&(bad.horizon_step==1)
    bad.loc[mask,"predicted"]+=1
    with pytest.raises(ValueError,match="different predicted"):
        validate_r8_intervals(bad)


def test_manifest_fingerprint_prevents_replaced_development_sources(records,tmp_path):
    root=tmp_path/"r8";root.mkdir()
    source=root/"r8_intervals_development.csv"
    records.to_csv(source,index=False)
    dev=tmp_path/"asset_predictions.csv"
    dev.write_text("valid synthetic source")
    import hashlib
    digest=hashlib.sha256(dev.read_bytes()).hexdigest()
    manifest=root/"r8_manifest.json"
    manifest.write_text(json.dumps({
        "protocol":"finwise-r8-development-aci-style-v1",
        "stage":"EXPLORATORY_DEVELOPMENT_ONLY_NOT_INDEPENDENT_TEST",
        "m5_test_assets_used":False,
        "assets":["AAPL","MSFT","SPY","TSLA"],
        "methods":["fixed_rolling","delayed_feedback_aci_style"],
        "source_sha256":digest,
    }))
    assert len(load_dev_input(source,manifest,dev))==len(records)
    dev.write_text("replaced source after looking at test outcomes")
    with pytest.raises(ValueError,match="changed since R8"):
        load_dev_input(source,manifest,dev)
