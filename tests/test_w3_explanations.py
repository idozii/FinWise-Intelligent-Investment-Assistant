import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.model_explanations import load_explanations
from research.ml_backtest import FEATURE_NAMES
from research.w3_attribution_export import build_export, sha, SCHEMA, ATTRIBUTION_TYPE


@pytest.fixture
def synthetic_m41(tmp_path):
    # Synthetic input is NOT any external M5 source.
    directory = tmp_path/"m41"
    directory.mkdir()
    sources=[]
    checks=[]
    summary=[]
    for a in ("AAPL","MSFT","SPY","TSLA"):
        file=tmp_path/f"{a}.csv"
        file.write_text("Date,Close\n2025-01-01,100\n")
        sources.append({"asset":a,"path":str(file),
                        "sha256":sha(file),"n_source_records":1})
        for h in (1,7):
            for i in range(32):
                checks.append({
                    "asset":a,"origin_index":i+252,"horizon_step":h,
                    "max_abs_reconstruction_error":1e-13,
                    "predicted_close":100.,"reference_model_close":100.,
                })
            for i, feature in enumerate(FEATURE_NAMES):
                summary.append({
                    "asset":a,"horizon_step":h,"feature":feature,"n_origins":32,
                    "mean_abs_contribution_log_return":.001*(len(FEATURE_NAMES)-i),
                    "mean_signed_contribution_log_return":.0001,
                    "fraction_positive_contribution":.5,
                    "attribution_type":ATTRIBUTION_TYPE,
                })
    (directory/"attribution_manifest.json").write_text(json.dumps({
        "protocol":"finwise-m41-ridge-decomposition-v1",
        "stage":"EXPLORATORY_MODEL_INTERNAL_NOT_CAUSAL",
        "external_requests":0,"max_origins":32,
        "horizons":[1,7],"feature_names":FEATURE_NAMES,
        "sources":sources,
    }))
    pd.DataFrame(checks).to_csv(directory/"ridge_reconstruction_checks.csv",index=False)
    pd.DataFrame(summary).to_csv(directory/"ridge_feature_summary.csv",index=False)
    return directory


def test_export_only_original_dev_assets_and_reconstructed_predictions(synthetic_m41,tmp_path):
    package=build_export(synthetic_m41)
    assert package["schemaVersion"]==SCHEMA
    assert len(package["results"])==8
    assert all(len(g["features"])==6 for g in package["results"])
    assert all(g["nOrigins"]==32 for g in package["results"])
    assert "GLD" not in json.dumps(package)
    path=tmp_path/"summary.json"
    path.write_text(json.dumps(package))
    public=load_explanations(path)
    assert len(public["results"])==8
    assert "causal" in public["warning"]
    assert "price" not in json.dumps(public["results"]).lower()


def test_price_reconstruction_mismatch_rejected(synthetic_m41):
    path=synthetic_m41/"ridge_reconstruction_checks.csv"
    df=pd.read_csv(path)
    df.loc[0,"reference_model_close"]=110.
    df.to_csv(path,index=False)
    with pytest.raises(ValueError,match="reconstruction"):
        build_export(synthetic_m41)


def test_source_hash_integrity_rejected(synthetic_m41):
    manifest=json.loads((synthetic_m41/"attribution_manifest.json").read_text())
    p=Path(manifest["sources"][0]["path"])
    p.write_text("tampered price source")
    with pytest.raises(ValueError,match="Source CSV changed"):
        build_export(synthetic_m41)


def test_holdout_asset_cannot_be_added(synthetic_m41):
    manifest_path=synthetic_m41/"attribution_manifest.json"
    m=json.loads(manifest_path.read_text())
    m["sources"][0]["asset"]="GLD"
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="development"):
        build_export(synthetic_m41)


def test_corrupt_export_schema_rejected(synthetic_m41,tmp_path):
    payload=build_export(synthetic_m41)
    payload["mode"]="LIVE_FORECAST_EXPLAINED"
    p=tmp_path/"bad.json";p.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="Unapproved"):
        load_explanations(p)


def test_frontend_explanation_controls_are_text_safe():
    root=Path(__file__).resolve().parents[1]
    html=(root/"frontend/index.html").read_text()
    js=(root/"frontend/app.js").read_text()
    assert 'id="explanationBars"' in html
    assert 'id="explanationHorizon"' in html
    assert "td.textContent = value" in js
    assert "name.textContent = feature.name" in js
    assert 'href="./styles.css?v=w3-20261028"' in html
    assert "not SHAP" in html
