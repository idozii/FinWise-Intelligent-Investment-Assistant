"""W5 operational safety checks are deterministic and do not call market vendors."""
import json
from fastapi.testclient import TestClient

import backend.app as application


def test_health_does_not_leak_host_paths_or_secrets(monkeypatch):
    monkeypatch.setattr(application, "DATA_DIR", "/home/user/private/financial/data")
    client = TestClient(application.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"ok"}
    assert "/home/user/private" not in response.text
    assert "dataDir" not in response.text


def test_readiness_network_free_with_optional_research_artifacts(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("Readiness must not contact Yahoo, Tiingo or CoinGecko")
    monkeypatch.setattr(application.yf, "download", forbid)
    monkeypatch.setattr(application.yf.Ticker, "history", forbid)
    response = TestClient(application.app).get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["frontendAssetsPresent"] is True
    assert payload["externalDataProviderProbed"] is False
    assert payload["livePredictionValidated"] is False


def test_readiness_reports_missing_frontend_assets_without_path_leak(monkeypatch,tmp_path):
    monkeypatch.setattr(application, "FRONTEND_DIR", tmp_path)
    response = TestClient(application.app).get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["frontendAssetsPresent"] is False
    assert str(tmp_path) not in response.text


def test_security_headers_and_no_caching_on_api():
    client = TestClient(application.app)
    for endpoint in ("/health", "/ready", "/api/capabilities"):
        response = client.get(endpoint)
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-frame-options"] == "DENY"
    api = client.get("/api/capabilities")
    assert api.headers["cache-control"] == "no-store"
    assert "no-store" not in client.get("/health").headers.get("cache-control", "")


def test_get_only_research_endpoints_and_failure_does_not_leak_paths(monkeypatch,tmp_path):
    monkeypatch.setenv("FINWISE_RELIABILITY_POLICIES_PATH",str(tmp_path/"missing.json"))
    client=TestClient(application.app)
    resp=client.get("/api/research/reliability-policies")
    assert resp.status_code==404
    assert resp.headers["cache-control"]=="no-store"
    assert str(tmp_path) not in resp.text
    assert client.post("/api/research/reliability-policies").status_code==405
