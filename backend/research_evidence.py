"""W2 read-only contract for an OPTIONAL local audited historical JSON export.

This never reads M5 CSV bars or connects to research prediction logic; only
whitelisted aggregate fields from an explicitly configured JSON file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "finwise-historical-research-evidence-v1"
FIELDS = (
    "asset", "horizon", "nOrigins", "naiveMae", "ridgeMae",
    "ridgeMinusNaiveMae", "bootstrap95Low", "bootstrap95High",
    "bootstrapContainsZero", "ridgeReady", "ridgeAccepted",
    "ridgeCoverageReady", "jointlyAccepted", "selectedRidgeMinusNaiveMae",
)
ASSETS = frozenset(("GLD", "IWM", "XLE", "XLV"))


def validate_export(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("schemaVersion") != SCHEMA:
        raise ValueError("Incompatible historical evidence export schema")
    if raw.get("evidenceType") != "HISTORICAL_RETROSPECTIVE_EXTERNAL_ASSET_FIRST_RUN":
        raise ValueError("Not an authorized historical first-run evidence type")
    if raw.get("executionMode") != "OFFLINE_AUDITED_EXPORT_NOT_LIVE_PREDICTION":
        raise ValueError("Research export must be explicitly offline")
    if raw.get("sourceProvider") != "Tiingo EOD raw Close":
        raise ValueError("Unknown source/provider for this frozen study")
    if not isinstance(raw.get("originalEvaluationProtocolSha256"), str) or len(
        raw["originalEvaluationProtocolSha256"]
    ) != 64:
        raise ValueError("Missing frozen protocol provenance")
    series = raw.get("results")
    if not isinstance(series, list) or len(series) != 8:
        raise ValueError("Expected exactly eight historical comparison cells")
    keys = set()
    for row in series:
        if not isinstance(row, dict) or not set(FIELDS) <= row.keys():
            raise ValueError("Missing historical comparison fields")
        asset, horizon = row["asset"], row["horizon"]
        if asset not in ASSETS or type(horizon) is not int or horizon not in (1, 7):
            raise ValueError("Unexpected historical result group")
        key = (asset, horizon)
        if key in keys:
            raise ValueError("Duplicate historical result group")
        keys.add(key)
        if (type(row["nOrigins"]) is not int or row["nOrigins"] < 1 or
                type(row["ridgeReady"]) is not int or
                type(row["ridgeAccepted"]) is not int or
                type(row["jointlyAccepted"]) is not int):
            raise ValueError("Invalid historical count field")
        if not (0 <= row["jointlyAccepted"] <= row["ridgeAccepted"] <=
                row["ridgeReady"] <= row["nOrigins"]):
            raise ValueError("Invalid historical acceptance counts")
        numerics = [row[x] for x in (
            "naiveMae", "ridgeMae", "ridgeMinusNaiveMae",
            "bootstrap95Low", "bootstrap95High", "ridgeCoverageReady"
        )]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) for v in numerics):
            raise ValueError("Invalid historical numeric value")
        if not 0 <= row["ridgeCoverageReady"] <= 1:
            raise ValueError("Invalid historical coverage fraction")
        if row["bootstrap95Low"] > row["bootstrap95High"]:
            raise ValueError("Inverted descriptive bootstrap interval")
        if abs((row["ridgeMae"] - row["naiveMae"]) -
               row["ridgeMinusNaiveMae"]) > 1e-7:
            raise ValueError("Unmatched Ridge-Naive MAE difference")
        if (row["selectedRidgeMinusNaiveMae"] is not None and
                (not isinstance(row["selectedRidgeMinusNaiveMae"], (int, float))
                 or not math.isfinite(row["selectedRidgeMinusNaiveMae"]))):
            raise ValueError("Invalid selected origin difference")
    if keys != {(asset, h) for asset in ASSETS for h in (1, 7)}:
        raise ValueError("Missing registered asset/horizon groups")
    # Strict allowlist prevents exposure of raw market bars or snapshot paths.
    return {
        "schemaVersion": SCHEMA,
        "evidenceType": raw["evidenceType"],
        "executionMode": raw["executionMode"],
        "sourceProvider": raw["sourceProvider"],
        "predictionTarget": "unadjusted_raw_close_subject_to_corporate_actions",
        "nominalIntervalCoverage": 0.9,
        "originalEvaluationProtocolSha256": raw["originalEvaluationProtocolSha256"],
        "results": [{k: row[k] for k in FIELDS} for row in series],
        "disclaimer": (
            "Historical first-run research only, not live prediction reliability. "
            "Intervals are descriptive; raw-close XLE includes a 2:1 split "
            "effective 2025-12-05. Not financial advice."
        ),
    }


def load_export(file: Path) -> dict:
    if not file.is_file():
        raise FileNotFoundError("Historical research evidence export not configured")
    if file.stat().st_size > 256_000:
        raise ValueError("Refusing oversized historical evidence package")
    document = json.loads(file.read_text(encoding="utf-8"))
    return validate_export(document)
