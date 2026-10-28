"""W4 OFFLINE research-only risk/availability export, never live calibration.

Recomputes R9 dev-only policies from the original R8 dev intervals and
checks CSV equality before allowing aggregate JSON to be shown on web.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.r9_selective_risk import (
    THRESHOLDS, METHODS, ALLOWED_DEV, MODELS, HORIZONS,
    digest, load_dev_input, policy_rows,
)

SCHEMA = "finwise-dev-reliability-policy-explorer-v1"


def _confirm_table(actual: pd.DataFrame, expected: pd.DataFrame,
                   keys: list[str], numeric: list[str], name: str) -> None:
    if len(actual)!=len(expected) or actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"{name}: missing or duplicate policy rows")
    paired = actual.merge(expected, on=keys, validate="one_to_one", indicator=True,
                          suffixes=("_actual", "_expected"), how="outer")
    if not (paired._merge=="both").all():
        raise ValueError(f"{name}: policy keys changed")
    for col in numeric:
        if not np.allclose(
            pd.to_numeric(paired[f"{col}_actual"],errors="raise"),
            pd.to_numeric(paired[f"{col}_expected"],errors="raise"),
            equal_nan=True, rtol=1e-8, atol=1e-9,
        ):
            raise ValueError(f"{name}: {col} does not match recomputed development results")


def build_export(r8_dir: Path, r9_dir: Path, dev_source: Path) -> dict:
    meta=json.loads((r9_dir/"r9_manifest.json").read_text())
    if (meta.get("protocol")!="finwise-r9-development-risk-availability-v1"
        or meta.get("stage")!="EXPLORATORY_DEV_POST_HOC_POLICY_SWEEP_NOT_INDEPENDENT_TEST"
        or meta.get("m5_assets_used") is not False
        or meta.get("model_refit") is not False
        or meta.get("thresholds")!=list(THRESHOLDS)):
        raise ValueError("W4 requires the original development-only R9 protocol")
    original_r8 = r8_dir/"r8_intervals_development.csv"
    original_manifest = r8_dir/"r8_manifest.json"
    if (meta.get("source_r8_sha256")!=digest(original_r8)
        or meta.get("r8_manifest_sha256")!=digest(original_manifest)
        or meta.get("original_dev_predictions_sha256")!=digest(dev_source)):
        raise ValueError("R9 development source fingerprints changed")
    rows=load_dev_input(original_r8, original_manifest, dev_source)
    expected_curve, expected_matched=policy_rows(rows)
    curve=pd.read_csv(r9_dir/"r9_risk_coverage_dev.csv")
    matched=pd.read_csv(r9_dir/"r9_matched_model_comparisons_dev.csv")
    keys=["asset","model","horizon_step","calibration_method","max_relative_width"]
    _confirm_table(curve,expected_curve,keys,
                   ["n_origins","n_ready","n_accepted","acceptance_rate_all",
                    "acceptance_rate_ready","interval_coverage_ready",
                    "interval_coverage_selected","mae_selected","relative_mae_selected",
                    "mean_relative_interval_width_selected"],"R9 curve")
    matched_keys=["asset","horizon_step","calibration_method","max_relative_width"]
    _confirm_table(matched,expected_matched,matched_keys,[
        "n_both_accepted","naive_mae_same_accepted",
        "ridge_mae_same_accepted","ridge_minus_naive_mae_same_accepted",
    ],"R9 matched comparisons")
    if (len(curve)!=160 or set(curve.asset)!=ALLOWED_DEV
        or set(curve.model)!=MODELS or set(curve.horizon_step)!=HORIZONS
        or set(curve.calibration_method)!=METHODS):
        raise ValueError("W4 requires complete 160 dev-only R9 policy cells")

    def optional_number(value):
        return None if pd.isna(value) else float(value)

    public=[]
    for row in curve.sort_values(keys).itertuples(index=False):
        public.append({
            "asset":row.asset, "model":row.model, "horizon":int(row.horizon_step),
            "method":row.calibration_method,
            "maxRelativeWidth":float(row.max_relative_width),
            "nOrigins":int(row.n_origins),"nReady":int(row.n_ready),
            "nAccepted":int(row.n_accepted),
            "acceptanceAll":float(row.acceptance_rate_all),
            "acceptanceReady":float(row.acceptance_rate_ready),
            "intervalCoverageReady":optional_number(row.interval_coverage_ready),
            "intervalCoverageSelected":optional_number(row.interval_coverage_selected),
            "maeSelected":optional_number(row.mae_selected),
            "relativeMaeSelected":optional_number(row.relative_mae_selected),
            "selectedMeanWidth":optional_number(row.mean_relative_interval_width_selected),
            "status":row.selection_warning,
        })
    return {
        "schemaVersion":SCHEMA,
        "evidenceType":"EXPLORATORY_DEVELOPMENT_POLICY_SENSITIVITY",
        "executionMode":"OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING",
        "nominalIntervalCoverage":.90,
        "widthThresholds":list(THRESHOLDS),
        "methods":sorted(METHODS),
        "assets":sorted(ALLOWED_DEV),
        "results":public,
        "limitations":[
            "No M5 holdout asset was used: R9 is DEVELOPMENT-ONLY",
            "Changing width thresholds after viewing dev results cannot establish independent evidence",
            "Selected errors compare different subsets across policies; not causal effect",
            "Zero accepted forecasts have null risk, not zero risk",
            "Coverage on selected forecasts has no demonstrated conditional guarantee",
            "All metrics describe offline already-scored forecasts, not live API predictions",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r8-dir",type=Path,default=Path("reports/backtest/r8_development"))
    p.add_argument("--r9-dir",type=Path,default=Path("reports/backtest/r9_risk_coverage_dev"))
    p.add_argument("--dev-predictions",type=Path,
                   default=Path("reports/backtest/m21_multiasset/asset_predictions.csv"))
    p.add_argument("--output",type=Path,default=Path("reports/backtest/w4_reliability_policies.json"))
    args=p.parse_args(argv)
    if args.output.resolve().is_relative_to(args.r8_dir.resolve()) or (
        args.output.resolve().is_relative_to(args.r9_dir.resolve())
    ):
        raise ValueError("Do not overwrite R8/R9 original research artifacts")
    package=build_export(args.r8_dir,args.r9_dir,args.dev_predictions)
    payload=json.dumps(package,indent=2,allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and args.output.read_text()!=payload:
        raise FileExistsError("Existing W4 policy export differs; preserve older version")
    args.output.write_text(payload)
    print(f"W4 PASS: {len(package['results'])} dev-only policy cells; no raw prices or live claims")
    print(f"SHA-256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")
    print(f"Saved to {args.output}")


if __name__=="__main__":
    main()
