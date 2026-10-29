"""Strict client-safe W4 DEVELOPMENT policy summary (optional local JSON only)."""
from __future__ import annotations
import json
import math
from pathlib import Path

SCHEMA = "finwise-dev-reliability-policy-explorer-v1"
METHODS = frozenset(("fixed_rolling", "delayed_feedback_aci_style"))
ASSETS = frozenset(("AAPL", "MSFT", "SPY", "TSLA"))
MODELS = frozenset(("naive", "ridge"))
HORIZONS = frozenset((1, 7))
WIDTHS = (0.05, 0.1, 0.15, 0.2, 0.3)
STATUS = frozenset(("NO_ACCEPTED", "SMALL_N_DESCRIPTIVE",
                    "DESCRIPTIVE_NOT_GUARANTEED"))


def _number(value, *, required: bool = True):
    if value is None and not required:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Invalid policy numeric field")
    return float(value)


def validate_policy_package(data: dict) -> dict:
    if (not isinstance(data, dict)
        or data.get("schemaVersion") != SCHEMA
        or data.get("evidenceType") != "EXPLORATORY_DEVELOPMENT_POLICY_SENSITIVITY"
        or data.get("executionMode") != "OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING"
        or data.get("widthThresholds") != list(WIDTHS)
        or set(data.get("assets", [])) != ASSETS
        or set(data.get("methods", [])) != METHODS):
        raise ValueError("W4 requires an audited exploratory development package")
    raw = data.get("results")
    if not isinstance(raw, list) or len(raw) != 160:
        raise ValueError("Expected 160 complete development policy groups")
    seen = set()
    rows = []
    for r in raw:
        if not isinstance(r, dict):
            raise ValueError("Malformed policy row")
        a, m, h, method, width = (
            r.get("asset"), r.get("model"), r.get("horizon"),
            r.get("method"), r.get("maxRelativeWidth"),
        )
        if (a not in ASSETS or m not in MODELS or type(h) is not int
            or h not in HORIZONS or method not in METHODS
            or type(width) not in (float, int) or width not in WIDTHS):
            raise ValueError("Unexpected policy group (no M5 test assets allowed)")
        key = (a, m, h, method, width)
        if key in seen:
            raise ValueError("Duplicate policy group")
        seen.add(key)
        n, ready, accepted = (r.get("nOrigins"), r.get("nReady"), r.get("nAccepted"))
        if (any(type(v) is not int for v in (n,ready,accepted)) or
            not (30 <= n and 0 <= accepted <= ready <= n)):
            raise ValueError("Invalid acceptance denominator")
        acceptance = _number(r.get("acceptanceAll"))
        ready_accept = _number(r.get("acceptanceReady"))
        if (not 0 <= acceptance <= 1 or not 0 <= ready_accept <= 1 or
            abs(acceptance-accepted/n)>1e-8 or
            abs(ready_accept-(accepted/ready if ready else 0))>1e-8):
            raise ValueError("Inconsistent acceptance fraction")
        coverage_ready = _number(r.get("intervalCoverageReady"),required=False)
        coverage_selected = _number(r.get("intervalCoverageSelected"),required=False)
        mae = _number(r.get("maeSelected"),required=False)
        relative_mae = _number(r.get("relativeMaeSelected"),required=False)
        width_selected = _number(r.get("selectedMeanWidth"),required=False)
        if coverage_ready is not None and not 0 <= coverage_ready <= 1:
            raise ValueError("Invalid ready coverage fraction")
        if coverage_selected is not None and not 0 <= coverage_selected <= 1:
            raise ValueError("Invalid selected coverage fraction")
        if (mae is not None and mae<0 or relative_mae is not None and relative_mae<0
            or width_selected is not None and width_selected<0):
            raise ValueError("Risk and selected width must be nonnegative")
        if accepted==0:
            if any(v is not None for v in (coverage_selected,mae,relative_mae,width_selected)):
                raise ValueError("No-accepted forecasts must have undefined risk")
            if r.get("status")!="NO_ACCEPTED":
                raise ValueError("Invalid no-forecast warning")
        elif (any(v is None for v in (coverage_selected,mae,relative_mae,width_selected))
              or r.get("status") not in STATUS - {"NO_ACCEPTED"}):
            raise ValueError("Invalid selected risk fields/status")
        rows.append({
            "asset":a,"model":m,"horizon":h,"method":method,
            "maxRelativeWidth":float(width),
            "nOrigins":n,"nReady":ready,"nAccepted":accepted,
            "acceptanceAll":acceptance,
            "acceptanceReady":ready_accept,
            "intervalCoverageReady":coverage_ready,
            "intervalCoverageSelected":coverage_selected,
            "maeSelected":mae,
            "relativeMaeSelected":relative_mae,
            "selectedMeanWidth":width_selected,
            "status":r["status"],
        })
    expected = {(a,m,h,method,width) for a in ASSETS for m in MODELS
                for h in HORIZONS for method in METHODS for width in WIDTHS}
    if seen != expected:
        raise ValueError("Incomplete width risk/availability design")
    return {
        "schemaVersion":SCHEMA,
        "executionMode":"OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING",
        "evidenceType":"EXPLORATORY_DEVELOPMENT_POLICY_SENSITIVITY",
        "nominalIntervalCoverage":.90,
        "widthThresholds":list(WIDTHS),
        "methods":sorted(METHODS),
        "results":rows,
        "warning":"Exploratory development-only selective forecasting. "
                  "Selected coverage is not a conditional guarantee. "
                  "Zero selected means no estimate, not zero error; not live forecasts.",
    }


def load_policy_package(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("No local audited R9 policy export configured")
    if path.stat().st_size > 512_000:
        raise ValueError("Oversized research policy package")
    return validate_policy_package(json.loads(path.read_text(encoding="utf-8")))
