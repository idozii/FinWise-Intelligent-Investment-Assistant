"""W6 public research portfolio smoke tests: research claims and safe endpoints."""
from pathlib import Path

from fastapi.testclient import TestClient

import backend.app as application


ROOT = Path(__file__).resolve().parents[1]


def test_research_portfolio_page_served_from_same_origin_without_market_fetch(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Static research portfolio must not hit vendors")
    monkeypatch.setattr(application.yf, "download", blocked)
    client = TestClient(application.app)
    page = client.get("/research.html")
    assert page.status_code == 200
    assert "A forecast is useful only when you understand" in page.text
    assert "R10" in page.text and "not yet authorized" in page.text
    assert "HISTORICAL" in page.text
    assert "not investment advice" in page.text
    assert 'src="./research.js?v=w6-20261028"' in page.text
    assert 'href="./research.css?v=w6-20261028"' in page.text


def test_market_and_research_pages_link_to_each_other():
    client = TestClient(application.app)
    market = client.get("/")
    research = client.get("/research.html")
    assert market.status_code == research.status_code == 200
    assert 'href="./research.html"' in market.text
    assert 'href="./index.html"' in research.text
    for asset in ("research.js", "research.css", "app.js", "styles.css"):
        assert client.get("/" + asset).status_code == 200


def test_portfolio_script_does_not_fabricate_market_performance_or_inject_html():
    code = (ROOT / "frontend" / "research.js").read_text(encoding="utf-8")
    assert '"/api/research/evidence"' in code
    assert '"/api/research/reliability-policies"' in code
    assert "/api/stocks/" not in code
    assert "/api/crypto/" not in code
    assert ".textContent =" in code
    assert ".innerHTML" not in code
    assert "No performance numbers are fabricated here" not in code or "not configured" in code


def test_readiness_requires_portfolio_files_but_not_private_research_data(monkeypatch,tmp_path):
    client = TestClient(application.app)
    assert client.get("/ready").json()["status"] == "ready"
    monkeypatch.setattr(application, "FRONTEND_DIR", tmp_path)
    readiness = client.get("/ready").json()
    assert readiness["status"] == "degraded"
    assert readiness["frontendAssetsPresent"] is False
    assert readiness["researchExportsRequiredForLiveness"] is False
    assert str(tmp_path) not in str(readiness)


def test_no_live_reliability_claim_or_source_vendor_token_leak():
    page = (ROOT/"frontend"/"research.html").read_text()
    assert "not a live validated trading signal" in page
    assert "No results here" not in page or "No" in page
    for secret in ("TIINGO_API_KEY", "TIINGO_TOKEN", "Authorization: Token"):
        assert secret not in page
    assert "Ridge Feature Contributions" in (
        ROOT/"frontend"/"index.html"
    ).read_text()
    assert "Selective Forecasting" in (ROOT/"frontend"/"index.html").read_text()
