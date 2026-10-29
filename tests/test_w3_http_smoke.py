import json

from fastapi.testclient import TestClient

import backend.app as application


def package():
    rows=[]
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for h in (1,7):
            rows.append({
                "asset":asset,"horizon":h,"nOrigins":32,
                "maxReconstructionError":1e-13,
                "features":[{
                    "name":f"feature_{i}",
                    "meanAbsLogReturnContribution":.001*(6-i),
                    "meanSignedLogReturnContribution":.0001,
                    "fractionPositive":.5,
                } for i in range(6)],
            })
    return {
        "schemaVersion":"finwise-ridge-explanation-development-v1",
        "mode":"HISTORICAL_EXPLORATORY_MODEL_INTERNAL_NOT_LIVE",
        "attributionType":"exact_linear_model_decomposition_not_SHAP_not_causal",
        "contributionUnit":"predicted_log_return","results":rows,
    }


def test_without_source_returns_safe_404(monkeypatch,tmp_path):
    monkeypatch.setenv("FINWISE_EXPLANATIONS_PATH",str(tmp_path/"missing.json"))
    response=TestClient(application.app).get("/api/research/explanations")
    assert response.status_code==404


def test_read_only_aggregate_contract(monkeypatch,tmp_path):
    p=tmp_path/"summary.json";p.write_text(json.dumps(package()))
    monkeypatch.setenv("FINWISE_EXPLANATIONS_PATH",str(p))
    client=TestClient(application.app)
    res=client.get("/api/research/explanations")
    assert res.status_code==200
    body=res.json()
    assert body["mode"]=="HISTORICAL_EXPLORATORY_MODEL_INTERNAL_NOT_LIVE"
    assert len(body["results"])==8
    assert "actual_close" not in res.text
    assert "predicted_close" not in res.text
    assert client.post("/api/research/explanations").status_code==405


def test_malformed_internal_explanation_rejected(monkeypatch,tmp_path):
    item=package()
    item["results"][0]["maxReconstructionError"]=10.
    p=tmp_path/"invalid.json";p.write_text(json.dumps(item))
    monkeypatch.setenv("FINWISE_EXPLANATIONS_PATH",str(p))
    assert TestClient(application.app).get("/api/research/explanations").status_code==422
