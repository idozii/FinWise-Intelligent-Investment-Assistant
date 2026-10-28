"""HTTP smoke: W2 historical endpoint returns only explicit local exports."""
import json

from fastapi.testclient import TestClient

import backend.app as application
from tests.test_w2_evidence import package


def test_evidence_unconfigured_returns_404_and_no_implicit_backtest(monkeypatch, tmp_path):
    monkeypatch.setenv("FINWISE_RESEARCH_SUMMARY_PATH", str(tmp_path / "missing.json"))
    response = TestClient(application.app).get("/api/research/evidence")
    assert response.status_code == 404
    assert "No audited historical" in response.json()["detail"]


def test_evidence_endpoint_strictly_read_only_and_historical(monkeypatch, tmp_path):
    path = tmp_path / "audited.json"
    path.write_text(json.dumps(package()))
    monkeypatch.setenv("FINWISE_RESEARCH_SUMMARY_PATH", str(path))
    client = TestClient(application.app)
    response = client.get("/api/research/evidence")
    assert response.status_code == 200
    body = response.json()
    assert body["executionMode"] == "OFFLINE_AUDITED_EXPORT_NOT_LIVE_PREDICTION"
    assert len(body["results"]) == 8
    assert "predictedClose" not in response.text
    assert "sourceSha256" not in body
    assert client.post("/api/research/evidence").status_code == 405


def test_malformed_research_export_returns_422(monkeypatch, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"schemaVersion":"forged_live_data"}')
    monkeypatch.setenv("FINWISE_RESEARCH_SUMMARY_PATH", str(path))
    response = TestClient(application.app).get("/api/research/evidence")
    assert response.status_code == 422
    assert "failed validation" in response.json()["detail"]
