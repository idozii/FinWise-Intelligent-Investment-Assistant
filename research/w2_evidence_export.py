"""W2 public-safe aggregate export from the IMMUTABLE first M5-v2 evaluation.

This exporter uses independently rechecked origin-level results, not
backend prediction code. It publishes no vendor bars, API tokens, or raw
per-origin target/prediction rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.frozen_eval_tiingo import load_v2_protocol
from research.m5_evidence_audit import (
    recalculated_diagnostics, verify_first_run,
)

SCHEMA = "finwise-historical-research-evidence-v1"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_summary(metrics: pd.DataFrame, pairs: pd.DataFrame,
                 cal: pd.DataFrame, matched: pd.DataFrame,
                 source_hashes: dict[str, str], protocol_sha: str) -> dict:
    """Strictly predeclared metrics from all 4 assets and 2 horizons."""
    assets = ("GLD", "IWM", "XLE", "XLV")
    horizons = (1, 7)
    rows = []
    for asset in assets:
        if len(source_hashes.get(asset, "")) != 64:
            raise ValueError(f"Missing dataset SHA-256 for {asset}")
        for horizon in horizons:
            def take(df, *, model="ridge"):
                query = (df.asset == asset) & (df.horizon_step == horizon)
                if "model" in df.columns:
                    query = query & (df.model == model)
                matches = df.loc[query]
                if len(matches) != 1:
                    raise ValueError(f"Missing or duplicated {asset} {horizon} {model}")
                return matches.iloc[0]
            baseline = take(metrics, model="naive")
            ridge = take(metrics)
            comparison = take(pairs)
            calibration = take(cal)
            selection = take(matched)
            n = int(ridge.n_origins)
            if n <= 0 or int(baseline.n_origins) != n:
                raise ValueError("Baseline/ML forecast origin counts do not agree")
            if int(calibration.n_total) != n or int(selection.n_origins) != n:
                raise ValueError("Calibrated and paired denominators disagree")
            if not np.isclose(
                float(ridge.mae) - float(baseline.mae),
                float(comparison.mae_delta_model_minus_naive), rtol=1e-8, atol=1e-8
            ):
                raise ValueError("Paired MAE delta does not match model scores")
            if not (0 <= int(calibration.n_accepted) <= int(calibration.n_ready) <= n):
                raise ValueError("Invalid acceptance denominator")
            if not (0 <= float(calibration.coverage_ready) <= 1):
                raise ValueError("Invalid coverage fraction")
            if int(selection.n_both_accepted) > int(calibration.n_accepted):
                raise ValueError("Jointly accepted exceeds Ridge accepted origins")
            ci_low = float(comparison.bootstrap_ci_low)
            ci_high = float(comparison.bootstrap_ci_high)
            if not np.isfinite([ci_low, ci_high]).all() or ci_low > ci_high:
                raise ValueError("Invalid bootstrap bounds")
            rows.append({
                "asset": asset,
                "horizon": int(horizon),
                "nOrigins": n,
                "naiveMae": float(baseline.mae),
                "ridgeMae": float(ridge.mae),
                "ridgeMinusNaiveMae": float(comparison.mae_delta_model_minus_naive),
                "bootstrap95Low": ci_low,
                "bootstrap95High": ci_high,
                "bootstrapContainsZero": bool(ci_low <= 0 <= ci_high),
                "ridgeReady": int(calibration.n_ready),
                "ridgeAccepted": int(calibration.n_accepted),
                "ridgeCoverageReady": float(calibration.coverage_ready),
                "jointlyAccepted": int(selection.n_both_accepted),
                "selectedRidgeMinusNaiveMae": (
                    None if pd.isna(selection.delta_mae_model_minus_naive_same_accepted)
                    else float(selection.delta_mae_model_minus_naive_same_accepted)
                ),
            })
    if len(rows) != 8:
        raise ValueError("Expected all eight frozen asset-horizon comparisons")
    return {
        "schemaVersion": SCHEMA,
        "evidenceType": "HISTORICAL_RETROSPECTIVE_EXTERNAL_ASSET_FIRST_RUN",
        "executionMode": "OFFLINE_AUDITED_EXPORT_NOT_LIVE_PREDICTION",
        "sourceProvider": "Tiingo EOD raw Close",
        "predictionTarget": "unadjusted_raw_close_subject_to_corporate_actions",
        "originalEvaluationProtocolSha256": protocol_sha,
        "sourceSha256": {asset: source_hashes[asset] for asset in assets},
        "researchModel": "ridge",
        "baseline": "naive_last_close",
        "nominalIntervalCoverage": 0.9,
        "bootstrapInterpretation": "descriptive_only_not_multiple_testing_adjusted",
        "methodLimitations": [
            "Historical asset-generalization, NOT an unseen future market-event test",
            "All 64-origin cells can be time-dependent; bootstrap intervals are descriptive",
            "Corporate actions can structurally alter raw Close (XLE split in December 2025)",
            "Selection-conditioned comparisons cannot demonstrate abstention gains",
            "No live model reliability or profitable investment claims",
        ],
        "results": rows,
    }


def export(directory: Path, output: Path) -> dict:
    spec = load_v2_protocol()
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Research export must not overwrite frozen original directory")
    _, original = verify_first_run(directory, spec)
    recalculated_diagnostics(directory, spec)  # verify all summaries and bootstrap
    metrics = pd.read_csv(directory / "holdout_metrics.csv")
    paired = pd.read_csv(directory / "holdout_paired_comparisons.csv")
    cal = pd.read_csv(directory / "holdout_calibration.csv")
    matched = pd.read_csv(directory / "holdout_matched_accepted.csv")
    package = make_summary(metrics, paired, cal, matched,
                           original["preflight_sources_sha256"],
                           original["protocol_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(package, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise FileExistsError("Existing evidence export changed; preserve earlier version")
    output.write_text(payload, encoding="utf-8")
    return package


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path,
                        default=Path("reports/backtest/m5_tiingo_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/backtest/w2_research_evidence.json"))
    args = parser.parse_args(argv)
    package = export(args.result_dir, args.output)
    print(f"W2 exported {len(package['results'])} historical results, no raw price bars")
    print(f"SHA-256 package: {_digest(args.output)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
