"""HTTP-level API compatibility smoke tests; no vendor network traffic."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import backend.app as application


def test_capabilities_endpoint_is_network_free_and_truthful(monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError("Capabilities endpoint attempted to download market data")
    monkeypatch.setattr(application.yf, "download", blocked)
    client = TestClient(application.app)
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    payload = res.json()
    assert payload["contractVersion"] == "finwise-web-api-v1.1"
    assert payload["researchModelIntegration"] == (
        "historical_offline_only_not_in_live_forecast_api"
    )
    assert payload["forecastReliability"]["legacyIntervalsEmpiricallyValidated"] is False


def test_stock_forecast_keeps_existing_response_fields_and_adds_disclosure(monkeypatch):
    n = 40
    frame = pd.DataFrame({
        "Date": pd.bdate_range("2026-01-01", periods=n, tz="UTC"),
        "Close": np.linspace(100., 110., n),
    })
    monkeypatch.setattr(application, "_load_symbol_data", lambda symbol: frame)
    monkeypatch.setattr(application, "_run_forecast",
                        lambda closes, days, model: (
                            np.full(days, 111.), np.full(days, 105.),
                            np.full(days, 117.), "Mock forecast"
                        ))
    response = TestClient(application.app).get("/api/stocks/AAPL/forecast?days=7&model=linear")
    assert response.status_code == 200
    body = response.json()
    assert set(["symbol", "horizonDays", "model", "modelId", "forecast", "metadata"]) <= set(body)
    assert len(body["forecast"]) == 7
    assert {"date", "predictedClose", "lower95", "upper95"} <= set(body["forecast"][0])
    assert body["metadata"]["intervalEmpiricallyCalibrated"] is False
    assert body["metadata"]["researchEvidenceAttached"] is False


def test_crypto_forecast_metadata_uses_calendar_days(monkeypatch):
    n = 40
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, tz="UTC"),
        "price": np.linspace(100., 110., n),
    })
    monkeypatch.setattr(application, "_load_coin_data", lambda coin: frame)
    monkeypatch.setattr(application, "_run_forecast",
                        lambda closes, days, model: (
                            np.full(days, 111.), np.full(days, 105.),
                            np.full(days, 117.), "Mock forecast"
                        ))
    response = TestClient(application.app).get("/api/crypto/bitcoin/forecast?days=7&model=linear")
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 7
    assert body["metadata"]["horizonUnit"] == "calendar_days"
    assert body["metadata"]["intervalCoverageGuarantee"] is False
