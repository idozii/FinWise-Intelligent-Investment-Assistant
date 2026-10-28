import json

from fastapi.testclient import TestClient
import backend.app as application


def package():
    rows=[]
    for asset in ("AAPL","MSFT","SPY","TSLA"):
        for model in ("naive","ridge"):
            for h in (1,7):
                for method in ("fixed_rolling","delayed_feedback_aci_style"):
                    for t in (.05,.1,.15,.2,.3):
                        rows.append({
                            "asset":asset,"model":model,"horizon":h,"method":method,
                            "maxRelativeWidth":t,
                            "nOrigins":143,"nReady":113,"nAccepted":20,
                            "acceptanceAll":20/143,
                            "acceptanceReady":20/113,
                            "intervalCoverageReady":.87,
                            "intervalCoverageSelected":.8,
                            "maeSelected":2.5,
                            "relativeMaeSelected":.025,
                            "selectedMeanWidth":.04,
                            "status":"SMALL_N_DESCRIPTIVE",
                        })
    return {
        "schemaVersion":"finwise-dev-reliability-policy-explorer-v1",
        "evidenceType":"EXPLORATORY_DEVELOPMENT_POLICY_SENSITIVITY",
        "executionMode":"OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING",
        "widthThresholds":[.05,.1,.15,.2,.3],
        "assets":["AAPL","MSFT","SPY","TSLA"],
        "methods":["fixed_rolling","delayed_feedback_aci_style"],
        "results":rows,
    }


def test_absent_research_export_returns_404(monkeypatch,tmp_path):
    monkeypatch.setenv("FINWISE_RELIABILITY_POLICIES_PATH",str(tmp_path/"missing.json"))
    assert TestClient(application.app).get("/api/research/reliability-policies").status_code==404


def test_readonly_historical_policy_does_not_expose_source_bars(monkeypatch,tmp_path):
    p=tmp_path/"w4.json";p.write_text(json.dumps(package()))
    monkeypatch.setenv("FINWISE_RELIABILITY_POLICIES_PATH",str(p))
    client=TestClient(application.app)
    response=client.get("/api/research/reliability-policies")
    assert response.status_code==200
    payload=response.json()
    assert payload["executionMode"]=="OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING"
    assert len(payload["results"])==160
    assert "predictedClose" not in response.text
    assert "last_observed_close" not in response.text
    assert client.post("/api/research/reliability-policies").status_code==405


def test_fake_live_validated_claim_is_rejected(monkeypatch,tmp_path):
    payload=package()
    payload["executionMode"]="LIVE_TRADING_CONFIRMED"
    p=tmp_path/"bad.json";p.write_text(json.dumps(payload))
    monkeypatch.setenv("FINWISE_RELIABILITY_POLICIES_PATH",str(p))
    response=TestClient(application.app).get("/api/research/reliability-policies")
    assert response.status_code==422
