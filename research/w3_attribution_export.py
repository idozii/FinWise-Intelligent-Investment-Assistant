"""W3 offline export of exact exploratory Ridge feature contributions (M4.1).

No market bars or origin-level forecasts are served. Requires original
source hashes and exact reconstruction checks, no M5 test asset permitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.ml_backtest import FEATURE_NAMES

SCHEMA = "finwise-ridge-explanation-development-v1"
DEV_ASSETS = frozenset(("AAPL", "MSFT", "SPY", "TSLA"))
ATTRIBUTION_TYPE = "exact_linear_model_decomposition_not_SHAP_not_causal"
MAX_ERROR = 1e-8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_export(source_dir: Path) -> dict:
    meta = json.loads((source_dir/"attribution_manifest.json").read_text())
    if (meta.get("protocol") != "finwise-m41-ridge-decomposition-v1"
            or meta.get("stage") != "EXPLORATORY_MODEL_INTERNAL_NOT_CAUSAL"
            or meta.get("external_requests") != 0):
        raise ValueError("Not a validated M4.1 exploratory attribution run")
    if set(meta.get("horizons", [])) != {1, 7} or meta.get("max_origins") != 32:
        raise ValueError("M4.1 parameters differ from original development protocol")
    if set(meta.get("feature_names", [])) != set(FEATURE_NAMES):
        raise ValueError("Ridge attribution features are inconsistent")
    sources = meta.get("sources", [])
    if set(s.get("asset") for s in sources) != DEV_ASSETS or len(sources) != 4:
        raise ValueError("Only original four exploratory development assets allowed")
    for source in sources:
        p = Path(source["path"])
        if not p.is_file() or sha(p) != source["sha256"]:
            raise ValueError(f"Source CSV changed or absent for {source['asset']}")
    checks = pd.read_csv(source_dir/"ridge_reconstruction_checks.csv")
    summary = pd.read_csv(source_dir/"ridge_feature_summary.csv")
    for col in ("max_abs_reconstruction_error", "reference_model_close", "predicted_close"):
        if col not in checks or not np.isfinite(checks[col].to_numpy(float)).all():
            raise ValueError("Missing or nonfinite M4.1 reconstruction checks")
    if (checks.max_abs_reconstruction_error > MAX_ERROR).any() or (
        np.abs(checks.reference_model_close-checks.predicted_close) > MAX_ERROR
    ).any():
        raise ValueError("Model prediction reconstruction mismatch")
    if checks.duplicated(["asset","origin_index","horizon_step"]).any():
        raise ValueError("Duplicate explained forecast")
    if (set(checks.asset) != DEV_ASSETS or set(checks.horizon_step) != {1,7}
            or len(checks) != 256):
        raise ValueError("Expected exact 256 audited Ridge explanations")
    if summary.duplicated(["asset","horizon_step","feature"]).any():
        raise ValueError("Duplicate feature contribution summary")
    if set(summary.feature) != set(FEATURE_NAMES) or set(summary.asset) != DEV_ASSETS:
        raise ValueError("Unexpected feature names or asset")
    if (summary.attribution_type != ATTRIBUTION_TYPE).any():
        raise ValueError("SHAP/causal attribution type is not supported")
    for col in ("n_origins","mean_abs_contribution_log_return",
                "mean_signed_contribution_log_return","fraction_positive_contribution"):
        if col not in summary or not np.isfinite(summary[col].to_numpy(float)).all():
            raise ValueError("Nonfinite M4.1 contribution summary")
    rows = []
    for asset in sorted(DEV_ASSETS):
        for h in (1, 7):
            view = summary[(summary.asset == asset) & (summary.horizon_step == h)]
            matched = checks[(checks.asset == asset) & (checks.horizon_step == h)]
            if (set(view.feature) != set(FEATURE_NAMES) or len(view)!=len(FEATURE_NAMES)
                    or len(matched)!=32 or not (view.n_origins == len(matched)).all()):
                raise ValueError("Incomplete M4.1 group or inconsistent origin counts")
            if (view.mean_abs_contribution_log_return < 0).any() or (
                ~view.fraction_positive_contribution.between(0,1)
            ).any():
                raise ValueError("Invalid feature contribution summaries")
            ranked = view.sort_values(
                "mean_abs_contribution_log_return", ascending=False
            ).head(6)
            rows.append({
                "asset":asset,"horizon":h,"nOrigins":len(matched),
                "features":[{
                    "name":r.feature,
                    "meanAbsLogReturnContribution":float(r.mean_abs_contribution_log_return),
                    "meanSignedLogReturnContribution":float(r.mean_signed_contribution_log_return),
                    "fractionPositive":float(r.fraction_positive_contribution),
                } for r in ranked.itertuples(index=False)],
                "maxReconstructionError":float(matched.max_abs_reconstruction_error.max()),
            })
    return {
        "schemaVersion":SCHEMA,
        "mode":"HISTORICAL_EXPLORATORY_MODEL_INTERNAL_NOT_LIVE",
        "model":"StandardScaler + Ridge(alpha=10)",
        "contributionUnit":"predicted_log_return",
        "attributionType":ATTRIBUTION_TYPE,
        "sampleOriginsPerAssetHorizon":32,
        "results":rows,
        "warning":"Signed standardized Ridge contributions explain the model's "
                  "own predicted log-return, not raw price, causality, SHAP or investment value.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attributions-dir", type=Path, default=Path("reports/backtest/m41_explanations"))
    p.add_argument("--output", type=Path, default=Path("reports/backtest/w3_ridge_explanations.json"))
    args=p.parse_args(argv)
    if args.output.resolve().is_relative_to(args.attributions_dir.resolve()):
        raise ValueError("Do not alter original M4.1 attribution output directory")
    package = build_export(args.attributions_dir)
    payload = json.dumps(package,indent=2,allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and args.output.read_text() != payload:
        raise FileExistsError("Existing W3 export differs; version the revision instead")
    args.output.write_text(payload)
    print("W3 PASS: exported 8 exploratory development groups; no price bars")
    print(f"Saved {args.output}, sha256={sha(args.output)}")


if __name__=="__main__":
    main()
