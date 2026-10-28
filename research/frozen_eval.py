"""M5 frozen-design external-asset replication.

Run 'preflight' to verify local CSV fingerprints and time coverage WITHOUT
reading their Close values. 'evaluate' requires explicit authorization,
untouched-asset attestation, and matching preflight SHA-256 hashes.

This is retrospective external-ASSET validation, not a prospective temporal
holdout, nor a proof of finite-sample coverage or trading utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.backtest import load_price_csv, summarize
from research.calibration_audit import (
    matched_selection_audit, model_audit, regime_audit, validated_rows,
)
from research.ml_backtest import ml_walk_forward
from research.robustness import (
    origin_volatility, paired_block_bootstrap, parse_assets,
)
from research.uncertainty import intervals

PROTOCOL_FILE = Path(__file__).parent / "protocols" / "m5_external_assets_v1.json"
LOCKED_PROTOCOL_SHA256 = "0d64713935f95c46b30317d705e3fc495bc6f40eda3bf0d81115ff536330c0ce"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_protocol(path: Path = PROTOCOL_FILE) -> dict:
    if _digest(path) != LOCKED_PROTOCOL_SHA256:
        raise ValueError("M5 protocol fingerprint mismatch: do not edit frozen protocol")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("protocol") != "finwise-m5-external-asset-validation-v1":
        raise ValueError("Invalid locked protocol ID")
    if sorted(spec["reserved_external_assets"]) != ["GLD", "IWM", "XLE", "XLV"]:
        raise ValueError("Unexpected test asset set")
    if set(spec["reserved_external_assets"]) & set(spec["prior_explored_assets"]):
        raise ValueError("Holdout asset overlaps exploratory assets")
    return spec


def _assets_exact(spec: dict, values: list[str]) -> list[tuple[str, Path]]:
    pairs = parse_assets(values)
    if set(asset for asset, _ in pairs) != set(spec["reserved_external_assets"]):
        raise ValueError("Exact frozen external asset set is required")
    return sorted(pairs, key=lambda pair: pair[0])


def eligible_origin_indices(dates: pd.Series, spec: dict) -> list[int]:
    """Only test origins with ALL horizons' future targets within end cutoff."""
    horizons = tuple(spec["horizons"])
    min_train, step = spec["min_train"], spec["step"]
    start = pd.Timestamp(spec["forecast_origin_start"], tz="UTC")
    end = pd.Timestamp(spec["forecast_origin_end"], tz="UTC")
    origins = []
    for i in range(min_train, len(dates) - max(horizons) + 1, step):
        forecast = dates.iloc[i-1]
        final_target = dates.iloc[i - 1 + max(horizons)]
        if start <= forecast <= end and final_target <= end:
            origins.append(i)
    return origins


def _dates_only(path: Path, spec: dict) -> pd.Series:
    date_col = spec["dataset"]["date_column"]
    price_col = spec["dataset"]["price_column"]
    headers = pd.read_csv(path, nrows=0).columns
    if date_col not in headers or price_col not in headers:
        raise ValueError(f"{path}: expected {date_col}/{price_col} headers")
    # Do not inspect close/label outcomes in preregistration preflight.
    raw = pd.read_csv(path, usecols=[date_col])
    dates = pd.to_datetime(raw[date_col], errors="raise", utc=True)
    if dates.isna().any():
        raise ValueError(f"{path}: missing dates")
    dates = dates.dt.normalize()
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"{path}: unsorted or duplicate daily dates")
    return dates


def preflight(spec: dict, paths: list[tuple[str, Path]], output_dir: Path) -> dict:
    entries = []
    for asset, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing local CSV for {asset}: {path}")
        dates = _dates_only(path, spec)
        origins = eligible_origin_indices(dates, spec)
        if len(origins) < 30:
            raise ValueError(f"{asset}: only {len(origins)} frozen evaluation origins; require >=30")
        entries.append({
            "asset": asset, "file": str(path.resolve()),
            "source_sha256": _digest(path),
            "n_daily_records": len(dates),
            "earliest_date": dates.iloc[0].date().isoformat(),
            "latest_date": dates.iloc[-1].date().isoformat(),
            "n_eligible_origins": len(origins),
            "first_eligible_origin": dates.iloc[origins[0]-1].date().isoformat(),
            "last_eligible_origin": dates.iloc[origins[-1]-1].date().isoformat(),
            "preflight_inspected_close_values": False,
        })
    result = {
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "stage": "PREFLIGHT_ONLY_NO_OUTCOME_REVIEW",
        "external_asset_validation_not_prospective": True,
        "manual_asset_noninspection_attestation_required": True,
        "sources": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "preflight_manifest.json"
    if dest.exists():
        # Stop silently replacing evidence after an evaluation was already run.
        before = json.loads(dest.read_text(encoding="utf-8"))
        if before != result:
            raise FileExistsError(
                "Preflight changed; preserve prior manifest and use a new output directory"
            )
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def check_preflight(spec: dict, paths: list[tuple[str, Path]], directory: Path) -> dict:
    path = directory / "preflight_manifest.json"
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked.get("stage") != "PREFLIGHT_ONLY_NO_OUTCOME_REVIEW":
        raise ValueError("Invalid preflight stage")
    if locked["protocol_sha256"] != LOCKED_PROTOCOL_SHA256:
        raise ValueError("Preflight is from a different frozen protocol")
    recorded = {entry["asset"]: entry for entry in locked["sources"]}
    if set(recorded) != set(spec["reserved_external_assets"]):
        raise ValueError("Preflight missing frozen assets")
    for asset, file in paths:
        if str(file.resolve()) != recorded[asset]["file"]:
            raise ValueError(f"{asset}: preflight path mismatch")
        if _digest(file) != recorded[asset]["source_sha256"]:
            raise ValueError(f"{asset}: market snapshot digest changed after preflight")
        if len(eligible_origin_indices(_dates_only(file, spec), spec)) != recorded[asset]["n_eligible_origins"]:
            raise ValueError(f"{asset}: evaluation origin count changed")
    return locked


def evaluate(
    spec: dict, paths: list[tuple[str, Path]], output_dir: Path
) -> dict:
    locked = check_preflight(spec, paths, output_dir)
    for filename in ("holdout_predictions.csv", "holdout_interval_predictions.csv",
                     "holdout_metrics.csv", "holdout_result_manifest.json"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite existing frozen test output: {filename}")
    all_predictions, all_intervals = [], []
    for asset, file in paths:
        frame = load_price_csv(file)
        pred = ml_walk_forward(
            frame, min_train=spec["min_train"],
            horizons=tuple(spec["horizons"]),
            step=spec["step"], models=tuple(spec["models"]),
        )
        pred.insert(0, "asset", asset)
        regimes = origin_volatility(frame, spec["min_train"])
        pred["regime"] = pred["origin_index"].map(regimes)
        if pred.regime.isna().any():
            raise ValueError(f"{asset}: missing lagged volatility regime")
        settings = spec["intervals"]
        annotated = intervals(
            pred, alpha=settings["alpha"],
            min_calibration=settings["min_calibration"],
            calibration_window=settings["calibration_window"],
            max_relative_width=settings["max_relative_width"],
        )
        indices = set(eligible_origin_indices(frame.date, spec))
        pred = pred.loc[pred.origin_index.isin(indices)]
        annotated = annotated.loc[annotated.origin_index.isin(indices)]
        if pred.empty or annotated.empty:
            raise ValueError(f"{asset}: no complete test-origin observations")
        all_predictions.append(pred)
        all_intervals.append(annotated)

    predictions = pd.concat(all_predictions, ignore_index=True)
    annotated = pd.concat(all_intervals, ignore_index=True)
    # Write both origin-level outputs before paired validation, using identical
    # strict schemas to the already tested M3.1 audit.
    pred_path = output_dir / "holdout_predictions.csv"
    interval_path = output_dir / "holdout_interval_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    annotated.to_csv(interval_path, index=False)
    validated = validated_rows(pred_path, interval_path)
    results = []
    for asset in sorted(predictions.asset.unique()):
        subset = predictions.loc[predictions.asset == asset]
        base_metrics = summarize(subset.drop(columns=["asset", "regime"]))
        base_metrics.insert(0, "asset", asset)
        results.append(base_metrics)
    metrics = pd.concat(results, ignore_index=True)
    metrics.to_csv(output_dir / "holdout_metrics.csv", index=False)
    b = spec["bootstrap"]
    comparisons = []
    for asset in sorted(predictions.asset.unique()):
        for horizon in spec["horizons"]:
            subset = predictions[
                (predictions.asset == asset) & (predictions.horizon_step == horizon)
            ]
            diff, lo, hi = paired_block_bootstrap(
                subset, block_length=b["block_length_origins"],
                iterations=b["iterations"], seed=b["seed"],
            )
            comparisons.append({
                "asset": asset, "horizon_step": horizon, "model": "ridge",
                "n_origins": subset.origin_index.nunique(),
                "mae_delta_model_minus_naive": diff,
                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                "model_worse_if_delta_positive": diff > 0,
            })
    pd.DataFrame(comparisons).to_csv(output_dir / "holdout_paired_comparisons.csv", index=False)
    model_audit(validated, spec["intervals"]["alpha"]).to_csv(
        output_dir / "holdout_calibration.csv", index=False)
    regime_audit(validated).to_csv(output_dir / "holdout_regime_coverage.csv", index=False)
    matched_selection_audit(validated).to_csv(
        output_dir / "holdout_matched_accepted.csv", index=False)
    report = {
        "protocol": spec["protocol"],
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "stage": "FIRST_EXTERNAL_ASSET_RETROSPECTIVE_EVALUATION",
        "not_prospective_temporal_holdout": True,
        "preflight_sources_sha256": {
            s["asset"]: s["source_sha256"] for s in locked["sources"]
        },
        "n_rows": len(predictions),
        "origin_start": spec["forecast_origin_start"],
        "origin_end": spec["forecast_origin_end"],
        "evaluation_models": spec["models"],
        "prequential_update": "previous test labels may enter later fits only once matured",
        "caveats": (
            "Assumes external assets were not inspected for model selection. "
            "Historical dates overlap previously explored assets. "
            "Bootstrap intervals are descriptive, no multiple-testing adjustment. "
            "90% conformal-style coverage is not guaranteed under market regime shifts."
        ),
    }
    (output_dir / "holdout_result_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "evaluate"))
    parser.add_argument("--assets", nargs="+", required=True,
                        metavar="ASSET=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorize-outcome-review", action="store_true")
    parser.add_argument("--attest-unused-assets", action="store_true")
    args = parser.parse_args(argv)
    spec = load_locked_protocol()
    pairs = _assets_exact(spec, args.assets)
    if args.stage == "preflight":
        result = preflight(spec, pairs, args.output_dir)
        print("Preflight passed; no Close values or outcome metrics inspected")
        for item in result["sources"]:
            print(item["asset"], item["n_daily_records"],
                  "rows,", item["n_eligible_origins"], "candidate test origins")
        print("Do NOT use evaluate until evidence and noninspection attestation are reviewed.")
    else:
        if not args.authorize_outcome_review or not args.attest_unused_assets:
            raise PermissionError(
                "Locked evaluation requires both --authorize-outcome-review "
                "and --attest-unused-assets. Do not reuse exploratory assets."
            )
        report = evaluate(spec, pairs, args.output_dir)
        print(f"Frozen external-asset evaluation finished: {report['n_rows']} "
              f"origin/model/horizon rows; see {args.output_dir}")


if __name__ == "__main__":
    main()
