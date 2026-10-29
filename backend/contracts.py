"""FinWise W1 versioned, truthful API contracts shared across legacy endpoints.

No M5 research results are imported into live predictions. This module is
pure (no market I/O) so API consumers can safely inspect capabilities.
"""
from __future__ import annotations

API_CONTRACT_VERSION = "finwise-web-api-v1.1"
LEGACY_INTERVAL_METHOD = "heuristic_residual_bounds_not_empirically_calibrated"
RESEARCH_STATUS = "historical_offline_only_not_in_live_forecast_api"
EXPECTED_LEGACY_MODELS = frozenset(
    ("arima", "holt_winters", "linear", "moving_average")
)


def capabilities_document(model_ids: list[str]) -> dict:
    """Fail closed on unexpected public model registry drift."""
    models = set(model_ids)
    if models != EXPECTED_LEGACY_MODELS:
        raise ValueError(
            "Public model registry changed; update and review W1 contract explicitly"
        )
    return {
        "contractVersion": API_CONTRACT_VERSION,
        "forecastApi": "legacy_statistical_models",
        "supportedModelIds": sorted(models),
        "researchModelIntegration": RESEARCH_STATUS,
        "forecastReliability": {
            "legacyIntervalsEmpiricallyValidated": False,
            "intervalMethod": LEGACY_INTERVAL_METHOD,
            "intervalNamesLegacy": ["lower95", "upper95"],
            "nominalLabelOnly": True,
            "investmentAdvice": False,
            "autoAbstention": False,
        },
        "dateSemantics": {
            "equities": "weekday_estimates_not_exchange_holiday_aware",
            "crypto": "calendar_days",
        },
        "dataSemantics": {
            "legacyLiveSource": "processed_files_or_vendor_fetch",
            "priceAdjustmentProvenance": "not_verified_by_legacy_endpoint",
            "noResearchSnapshotOrTestSetConnected": True,
        },
    }


def legacy_forecast_metadata(asset_class: str) -> dict:
    """Stable additive metadata; preserve existing response fields and names."""
    if asset_class not in ("stocks", "crypto"):
        raise ValueError("Unknown asset class")
    return {
        "contractVersion": API_CONTRACT_VERSION,
        "status": "UNVALIDATED_ESTIMATE",
        "intervalMethod": LEGACY_INTERVAL_METHOD,
        "intervalEmpiricallyCalibrated": False,
        "intervalCoverageGuarantee": False,
        "researchEvidenceAttached": False,
        "priceAdjustmentProvenance": "not_verified_by_legacy_endpoint",
        "horizonUnit": (
            "weekday_estimates_not_exchange_holiday_aware"
            if asset_class == "stocks" else "calendar_days"
        ),
        "disclaimer": (
            "Illustrative forecasts and heuristic residual intervals; "
            "not calibrated probabilities, trading signals or investment advice."
        ),
    }
