"""W8 public release: protect market providers and serve research-first UI."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import backend.app as application


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(application.app)


def test_public_config_is_explicit_and_no_live_claim(monkeypatch):
    monkeypatch.setenv("FINWISE_ENABLE_LIVE_MARKET", "false")
    monkeypatch.setenv("FINWISE_PUBLIC_LANDING", "true")
    data = client.get("/api/public-config").json()
    assert data == {
        "marketEnabled": False,
        "publicResearchLanding": True,
        "marketForecastsValidated": False,
        "historicalResearchIsNotLive": True,
    }


@pytest.mark.parametrize("route", (
    "/api/stocks/AAPL/history?days=365",
    "/api/stocks/AAPL/forecast?days=7&model=arima",
    "/api/crypto/bitcoin/history?days=365",
    "/api/crypto/bitcoin/forecast?days=7&model=arima",
))
def test_public_market_endpoints_fail_closed_before_vendor_calls(
    route, monkeypatch
):
    monkeypatch.setenv("FINWISE_ENABLE_LIVE_MARKET", "false")
    def forbidden(*args, **kwargs):
        raise AssertionError("No vendor may be contacted during public preview")
    monkeypatch.setattr(application, "_load_symbol_data", forbidden)
    monkeypatch.setattr(application, "_load_coin_data", forbidden)
    response = client.get(route)
    assert response.status_code == 503
    assert "disabled on this public research demo" in response.json()["detail"]
    assert response.headers["cache-control"] == "no-store"


def test_explicit_local_market_mode_still_enabled(monkeypatch):
    monkeypatch.setenv("FINWISE_ENABLE_LIVE_MARKET", "true")
    assert client.get("/api/public-config").json()["marketEnabled"] is True
    application._require_live_market() is None


def test_market_workspace_is_before_optional_research_cards():
    market = (ROOT/"frontend"/"index.html").read_text()
    assert market.index('id="market-workspace"') < market.index('id="researchEvidenceTitle"')
    assert 'id="marketPreviewNotice"' in market
    assert "Explore markets." in market
    assert "not a guarantee" in market.lower() or "not advice" in market.lower()
    assert "Ridge Feature Contributions" in market
    assert "Selective Forecasting" in market
    assert 'href="./styles.css?v=w8-public-1"' in market
    js = (ROOT/"frontend"/"app.js").read_text()
    assert '"/api/public-config"' in js
    assert "if (!marketEnabled) return" in js
    assert "market-disabled" in js
    assert "await loadPublicMode();" in js


def test_research_landing_has_qualified_non_live_result_cards():
    response = client.get("/research.html")
    assert response.status_code == 200
    content = response.text
    assert "0 of 8" in content
    assert "8 of 8" in content
    assert "2 of 8" in content
    assert "first-run" in content
    assert 'href="./research.css?v=w8-public-1"' in content
    assert "not a live validated trading signal" in content


def test_render_blueprint_is_safe_free_preview_configuration():
    blueprint = (ROOT/"render.yaml").read_text()
    assert "plan: free" in blueprint
    assert "runtime: python" in blueprint
    assert "region: singapore" in blueprint
    assert "PYTHON_VERSION" in blueprint and "3.11.11" in blueprint
    assert "FINWISE_ENABLE_LIVE_MARKET" in blueprint
    assert 'value: "false"' in blueprint
    assert "FINWISE_PUBLIC_LANDING" in blueprint
    assert "uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1" in blueprint
    assert "TIINGO_API_KEY" not in blueprint
    assert (ROOT/"docs/W8_PUBLIC_RELEASE.md").is_file()
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["frontendAssetsPresent"] is True
