"""W1 API metadata contract must never imply M5 results run in live API."""
from pathlib import Path

import pytest

from backend.contracts import (
    API_CONTRACT_VERSION, EXPECTED_LEGACY_MODELS,
    LEGACY_INTERVAL_METHOD, RESEARCH_STATUS,
    capabilities_document, legacy_forecast_metadata,
)


def test_capabilities_reflect_exact_legacy_models_and_uncertainty_limitations():
    doc = capabilities_document(sorted(EXPECTED_LEGACY_MODELS))
    assert doc["contractVersion"] == API_CONTRACT_VERSION
    assert set(doc["supportedModelIds"]) == EXPECTED_LEGACY_MODELS
    assert doc["researchModelIntegration"] == RESEARCH_STATUS
    reliability = doc["forecastReliability"]
    assert not reliability["legacyIntervalsEmpiricallyValidated"]
    assert reliability["intervalMethod"] == LEGACY_INTERVAL_METHOD
    assert reliability["nominalLabelOnly"]
    assert not reliability["autoAbstention"]
    assert doc["dataSemantics"]["noResearchSnapshotOrTestSetConnected"]
    assert not reliability["investmentAdvice"]


def test_public_model_drift_fails_closed():
    with pytest.raises(ValueError, match="model registry changed"):
        capabilities_document(["ridge"])
    with pytest.raises(ValueError, match="model registry changed"):
        capabilities_document(sorted(EXPECTED_LEGACY_MODELS) + ["ridge"])


@pytest.mark.parametrize("asset_class,horizon", [
    ("stocks", "weekday_estimates_not_exchange_holiday_aware"),
    ("crypto", "calendar_days"),
])
def test_legacy_forecasts_disclose_uncalibrated_interval(asset_class, horizon):
    meta = legacy_forecast_metadata(asset_class)
    assert meta["status"] == "UNVALIDATED_ESTIMATE"
    assert meta["horizonUnit"] == horizon
    assert not meta["intervalEmpiricallyCalibrated"]
    assert not meta["intervalCoverageGuarantee"]
    assert not meta["researchEvidenceAttached"]
    assert "not calibrated" in meta["disclaimer"]
    assert meta["priceAdjustmentProvenance"] == "not_verified_by_legacy_endpoint"


def test_unknown_asset_class_is_rejected():
    with pytest.raises(ValueError, match="Unknown asset class"):
        legacy_forecast_metadata("forex")


def test_frontend_has_explicit_reliability_disclosure():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend" / "index.html").read_text()
    js = (root / "frontend" / "app.js").read_text()
    assert 'id="capabilityStatus"' in html
    assert 'id="forecastEvidence"' in html
    assert "Lower bound (heuristic)" in html
    assert "Upper bound (heuristic)" in html
    assert "/api/capabilities" in js
    assert "intervalEmpiricallyCalibrated === false" in js
    assert "Ridge/conformal research remains offline" in js


def test_fastapi_routes_wired_to_versioned_metadata():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "backend" / "app.py").read_text()
    assert '@app.get("/api/capabilities")' in app_source
    assert 'legacy_forecast_metadata("stocks")' in app_source
    assert 'legacy_forecast_metadata("crypto")' in app_source
    assert 'allow_origins=_CORS_ORIGINS' in app_source
    assert 'allow_credentials=False' in app_source
