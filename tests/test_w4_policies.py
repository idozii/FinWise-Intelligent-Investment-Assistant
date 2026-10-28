import json

import numpy as np
import pandas as pd
import pytest

from backend.reliability_policies import load_policy_package, validate_policy_package
from research.r8_adaptive_development import compare
from research.r9_selective_risk import digest, policy_rows
from research.w4_policy_export import SCHEMA, build_export


@pytest.fixture
def develop(tmp_path):
    n=42
    dates=pd.date_range("2025-01-01",periods=n*8+20,tz="UTC")
    rows=[]
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for model in ("naive","ridge"):
            for h in (1,7):
                for i in range(n):
                    k=i*7
                    last=100+.1*i
                    rows.append({
                        "asset":asset,"model":model,"horizon_step":h,
                        "origin_index":i+252,
                        "train_end":dates[k].date().isoformat(),
                        "target_date":dates[k+h].date().isoformat(),
                        "last_observed_close":last,
                        "actual":last + 1.8*np.sin(i/5)+.2,
                        "predicted":last+(0 if model=="naive" else .1),
                    })
    dev=tmp_path/"asset_predictions.csv"
    source=pd.DataFrame(rows);source.to_csv(dev,index=False)
    r8=tmp_path/"r8";r8.mkdir()
    annotated,_=compare(source)
    intervals=r8/"r8_intervals_development.csv"
    annotated.to_csv(intervals,index=False)
    m8=r8/"r8_manifest.json"
    m8.write_text(json.dumps({
        "protocol":"finwise-r8-development-aci-style-v1",
        "stage":"EXPLORATORY_DEVELOPMENT_ONLY_NOT_INDEPENDENT_TEST",
        "m5_test_assets_used":False,
        "assets":["AAPL","MSFT","SPY","TSLA"],
        "methods":["fixed_rolling","delayed_feedback_aci_style"],
        "source_sha256":digest(dev),
    }))
    r9=tmp_path/"r9";r9.mkdir()
    curve,paired=policy_rows(annotated)
    curve.to_csv(r9/"r9_risk_coverage_dev.csv",index=False)
    paired.to_csv(r9/"r9_matched_model_comparisons_dev.csv",index=False)
    (r9/"r9_manifest.json").write_text(json.dumps({
        "protocol":"finwise-r9-development-risk-availability-v1",
        "stage":"EXPLORATORY_DEV_POST_HOC_POLICY_SWEEP_NOT_INDEPENDENT_TEST",
        "m5_assets_used":False,
        "model_refit":False,
        "thresholds":[.05,.1,.15,.2,.3],
        "source_r8_sha256":digest(intervals),
        "r8_manifest_sha256":digest(m8),
        "original_dev_predictions_sha256":digest(dev),
    }))
    return r8,r9,dev


def test_export_full_dev_only_policy_grid(develop,tmp_path):
    r8,r9,dev=develop
    payload=build_export(r8,r9,dev)
    assert payload["schemaVersion"]==SCHEMA
    assert len(payload["results"])==160
    assert not any(x["asset"] in ("GLD","IWM","XLE","XLV") for x in payload["results"])
    path=tmp_path/"w4.json"
    path.write_text(json.dumps(payload))
    published=load_policy_package(path)
    assert len(published["results"])==160
    assert "actual" not in json.dumps(published)
    assert "last_observed_close" not in json.dumps(published)


def test_tampered_r9_csv_cannot_be_exported(develop):
    r8,r9,dev=develop
    p=r9/"r9_risk_coverage_dev.csv"
    df=pd.read_csv(p)
    df.loc[0,"n_accepted"]+=1
    df.to_csv(p,index=False)
    with pytest.raises(ValueError,match="n_accepted"):
        build_export(r8,r9,dev)


def test_changed_underlying_r8_input_detected(develop):
    r8,r9,dev=develop
    p=r8/"r8_intervals_development.csv"
    p.write_text(p.read_text()+"\n")
    with pytest.raises(ValueError,match="fingerprints changed"):
        build_export(r8,r9,dev)


def test_invalid_or_live_policy_claims_fail_closed(develop):
    r8,r9,dev=develop
    package=build_export(r8,r9,dev)
    package["executionMode"]="LIVE_CALIBRATED"
    with pytest.raises(ValueError,match="audited exploratory"):
        validate_policy_package(package)
    package=build_export(r8,r9,dev)
    package["results"][0]["asset"]="GLD"
    with pytest.raises(ValueError,match="no M5 test"):
        validate_policy_package(package)


def test_zero_acceptance_is_not_zero_error(develop):
    r8,r9,dev=develop
    payload=build_export(r8,r9,dev)
    row=payload["results"][0]
    row.update({
        "nAccepted":0,"acceptanceAll":0.0,"acceptanceReady":0.0,
        "intervalCoverageSelected":None,"maeSelected":None,
        "relativeMaeSelected":None,"selectedMeanWidth":None,
        "status":"NO_ACCEPTED",
    })
    verified=validate_policy_package(payload)
    selected=verified["results"][0]
    assert selected["nAccepted"]==0
    assert selected["maeSelected"] is None
    assert selected["relativeMaeSelected"] is None
    payload["results"][0]["maeSelected"]=0.0
    with pytest.raises(ValueError,match="undefined risk"):
        validate_policy_package(payload)


def test_frontend_separates_dev_reliability_from_live_prediction():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    html=(root/"frontend/index.html").read_text()
    js=(root/"frontend/app.js").read_text()
    assert 'id="policyRiskChart"' in html
    assert 'id="policyWidth"' in html
    assert "NOT LIVE" in html
    assert "/api/research/reliability-policies" in js
    assert "No forecast accepted" not in js or "N/A" in js
    assert "dt.textContent = key" in js
    assert "dd.textContent = value" in js
    assert 'href="./styles.css?v=w4-20261028"' in html
