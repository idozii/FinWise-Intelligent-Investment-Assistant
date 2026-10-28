"""R9 development-only selective forecasting risk/availability study.

Read a single existing R8 dev interval artifact; NEVER score M5 test assets
or refit forecasters. Sweep only predeclared width policies using widths
available at each origin. Scores are descriptive and selection-conditioned.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.r8_adaptive_development import ALLOWED_DEV, FORBIDDEN_TEST, PROTOCOL as R8_PROTOCOL

PROTOCOL = "finwise-r9-development-risk-availability-v1"
METHODS = frozenset(("fixed_rolling", "delayed_feedback_aci_style"))
MODELS = frozenset(("naive", "ridge"))
HORIZONS = frozenset((1, 7))
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
KEYS = ["asset", "model", "horizon_step", "origin_index"]
ROW_ID = KEYS + ["calibration_method"]
BASE_FIELDS = ["train_end", "target_date", "actual", "predicted", "last_observed_close"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dev_input(source: Path, manifest_path: Path,
                   source_predictions: Path) -> pd.DataFrame:
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (meta.get("protocol") != R8_PROTOCOL
        or meta.get("stage") != "EXPLORATORY_DEVELOPMENT_ONLY_NOT_INDEPENDENT_TEST"
        or meta.get("m5_test_assets_used") is not False
        or set(meta.get("assets", [])) != ALLOWED_DEV
        or set(meta.get("methods", [])) != METHODS):
        raise ValueError("Not an original R8 exploratory development interval artifact")
    if digest(source_predictions) != meta.get("source_sha256"):
        raise ValueError("Original M2.1 development predictions changed since R8")
    # Explicitly forbid holdout artifact names irrespective of whether masked rows were removed.
    if any("holdout" in str(p).lower() or "m5_tiingo" in str(p).lower()
           for p in (source, manifest_path, source_predictions)):
        raise ValueError("M5 holdout artifacts must not be used in R9")
    frame = pd.read_csv(source)
    validate_r8_intervals(frame)
    return frame


def validate_r8_intervals(frame: pd.DataFrame) -> None:
    expected = set(ROW_ID + BASE_FIELDS + [
        "interval_ready", "accept_forecast", "relative_width", "covered",
        "interval_lower", "interval_upper",
    ])
    missing = expected - set(frame)
    if missing:
        raise ValueError(f"R8 interval columns missing: {sorted(missing)}")
    if frame.empty or set(frame.asset) != ALLOWED_DEV or (
        set(frame.asset) & FORBIDDEN_TEST
    ):
        raise ValueError("R9 ONLY accepts 4 exploratory AAPL/MSFT/SPY/TSLA assets")
    if (set(frame.model) != MODELS or set(frame.horizon_step) != HORIZONS
        or set(frame.calibration_method) != METHODS):
        raise ValueError("R9 requires both original calibration methods and models/horizons")
    if frame.duplicated(ROW_ID).any():
        raise ValueError("Duplicated R8 calibrated origin keys")
    counts = frame.groupby(["asset", "model", "horizon_step", "calibration_method"]).size()
    if len(counts) != 32 or counts.min() != counts.max() or counts.min() < 30:
        raise ValueError("Incomplete or unbalanced R8 development origin groups")
    for x in ["actual", "predicted", "last_observed_close"]:
        if not np.isfinite(pd.to_numeric(frame[x], errors="raise").to_numpy(float)).all():
            raise ValueError(f"Invalid {x} values")
    if (frame.actual <= 0).any() or (frame.last_observed_close <= 0).any():
        raise ValueError("Prices must be positive")
    for x in ["train_end", "target_date"]:
        dates = pd.to_datetime(frame[x], errors="raise", utc=True)
        if dates.isna().any():
            raise ValueError(f"Missing {x}")
    origin = pd.to_datetime(frame.train_end, utc=True)
    target = pd.to_datetime(frame.target_date, utc=True)
    if (target <= origin).any():
        raise ValueError("Target dates must follow origins")
    for column in ["interval_ready", "accept_forecast"]:
        if not frame[column].isin([True, False]).all():
            raise ValueError(f"{column} must be boolean")
    if (frame.accept_forecast & ~frame.interval_ready).any():
        raise ValueError("Cannot accept an unready interval")
    ready = frame.loc[frame.interval_ready]
    if (not np.isfinite(ready.relative_width.to_numpy(float)).all()
        or (ready.relative_width < 0).any()
        or not np.isfinite(ready[["interval_lower", "interval_upper"]].to_numpy(float)).all()):
        raise ValueError("Ready intervals require nonnegative finite width/bounds")
    if (ready.interval_upper < ready.interval_lower).any():
        raise ValueError("Inverted interval")
    actual_covered = (ready.interval_lower <= ready.actual) & (
        ready.actual <= ready.interval_upper
    )
    if not (ready.covered.astype(bool).to_numpy() == actual_covered.to_numpy()).all():
        raise ValueError("Covered flags are inconsistent with interval outcomes")
    if (ready.accept_forecast.to_numpy() !=
        (ready.relative_width.to_numpy(float) <= .10 + 1e-12)).any():
        raise ValueError("R8 original acceptance at 10% does not match width")
    # Both calibration methods must describe EXACT SAME forecast, actual, and dates.
    wide = frame.set_index(ROW_ID)
    left = wide.xs("fixed_rolling", level="calibration_method")
    right = wide.xs("delayed_feedback_aci_style", level="calibration_method")
    if not left.index.equals(right.index):
        raise ValueError("R8 methods do not share the same forecast origins")
    for col in BASE_FIELDS:
        if col in ("train_end", "target_date"):
            if not left[col].equals(right[col]):
                raise ValueError(f"R8 methods use different {col} at same origin")
        elif not np.allclose(left[col].to_numpy(float), right[col].to_numpy(float),
                             atol=1e-9, rtol=1e-12):
            raise ValueError(f"R8 methods have different {col} at same origin")


def policy_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_r8_intervals(frame)
    # Decisions use ONLY readiness and interval width available at issue time.
    scored = frame.copy()
    scored["absolute_error"] = (scored.actual - scored.predicted).abs()
    scored["relative_absolute_error"] = scored.absolute_error / scored.last_observed_close
    rows, matched = [], []
    for threshold in THRESHOLDS:
        scored["policy_accepted"] = (
            scored.interval_ready & (scored.relative_width <= threshold)
        )
        for (asset, model, horizon, method), group in scored.groupby(
            ["asset", "model", "horizon_step", "calibration_method"], sort=True
        ):
            ready = group.loc[group.interval_ready]
            accepted = group.loc[group.policy_accepted]
            n = len(group)
            n_selected = len(accepted)
            rows.append({
                "asset": asset, "model": model, "horizon_step": int(horizon),
                "calibration_method": method, "max_relative_width": threshold,
                "n_origins": n, "n_ready": len(ready), "n_accepted": n_selected,
                "acceptance_rate_all": n_selected/n,
                "acceptance_rate_ready": n_selected/len(ready) if len(ready) else 0.,
                "interval_coverage_ready": float(ready.covered.astype(bool).mean())
                    if len(ready) else np.nan,
                "interval_coverage_selected": float(accepted.covered.astype(bool).mean())
                    if n_selected else np.nan,
                "mae_selected": float(accepted.absolute_error.mean())
                    if n_selected else np.nan,
                "relative_mae_selected": float(accepted.relative_absolute_error.mean())
                    if n_selected else np.nan,
                "mean_relative_interval_width_selected": float(accepted.relative_width.mean())
                    if n_selected else np.nan,
                "selection_warning": ("NO_ACCEPTED" if not n_selected
                    else "SMALL_N_DESCRIPTIVE" if n_selected < 30
                    else "DESCRIPTIVE_NOT_GUARANTEED"),
            })
        for (asset, horizon, method), group in scored.groupby(
            ["asset", "horizon_step", "calibration_method"], sort=True
        ):
            key = ["origin_index", "train_end", "target_date"]
            naive = group.loc[(group.model == "naive") & group.policy_accepted]
            ridge = group.loc[(group.model == "ridge") & group.policy_accepted]
            both = naive[key + ["absolute_error"]].merge(
                ridge[key + ["absolute_error"]], on=key, how="inner",
                validate="one_to_one", suffixes=("_naive", "_ridge")
            )
            n_both = len(both)
            delta = (
                float((both.absolute_error_ridge - both.absolute_error_naive).mean())
                if n_both else np.nan
            )
            matched.append({
                "asset": asset, "horizon_step": int(horizon),
                "calibration_method": method,
                "max_relative_width": threshold,
                "n_both_accepted": n_both,
                "naive_mae_same_accepted": float(both.absolute_error_naive.mean())
                    if n_both else np.nan,
                "ridge_mae_same_accepted": float(both.absolute_error_ridge.mean())
                    if n_both else np.nan,
                "ridge_minus_naive_mae_same_accepted": delta,
                "status": ("NO_JOINT_ACCEPTANCE" if not n_both
                           else "SMALL_N_DESCRIPTIVE" if n_both < 30
                           else "CONDITIONAL_EXPLORATORY"),
            })
    return pd.DataFrame(rows), pd.DataFrame(matched)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r8-dir", type=Path,
                   default=Path("reports/backtest/r8_development"))
    p.add_argument("--dev-predictions", type=Path,
                   default=Path("reports/backtest/m21_multiasset/asset_predictions.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("reports/backtest/r9_risk_coverage_dev"))
    args = p.parse_args(argv)
    source = args.r8_dir / "r8_intervals_development.csv"
    manifest = args.r8_dir / "r8_manifest.json"
    if args.output_dir.resolve() in (
        args.r8_dir.resolve(), args.dev_predictions.parent.resolve()
    ):
        raise ValueError("R9 output must not overwrite prior development data")
    df = load_dev_input(source, manifest, args.dev_predictions)
    curve, matched = policy_rows(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_dir/"r9_risk_coverage_dev.csv", index=False)
    matched.to_csv(args.output_dir/"r9_matched_model_comparisons_dev.csv", index=False)
    report = {
        "protocol": PROTOCOL,
        "stage": "EXPLORATORY_DEV_POST_HOC_POLICY_SWEEP_NOT_INDEPENDENT_TEST",
        "source_r8_sha256": digest(source),
        "r8_manifest_sha256": digest(manifest),
        "original_dev_predictions_sha256": digest(args.dev_predictions),
        "thresholds": list(THRESHOLDS),
        "not_optimized_for_test_assets": True,
        "m5_assets_used": False,
        "model_refit": False,
        "selection_uses_only_origin_available_interval_width": True,
        "selected_risk_is_conditional_and_not_comparable_without_same_origin_matching": True,
        "methods_same_point_predictions_differ_only_by_interval_selection": True,
        "caveat": "Do not choose a threshold on M5 results or imply conditional coverage "
                  "guarantees. Sweeps are exploratory; policy needs a new frozen evaluation.",
    }
    (args.output_dir/"r9_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"R9 DEV ONLY: {len(curve)} policy cells, {len(matched)} paired origin comparisons")
    print(curve.loc[(curve.max_relative_width==.10)&(curve.model=="ridge"),
                    ["asset","horizon_step","calibration_method","n_accepted",
                     "interval_coverage_selected","mae_selected","selection_warning"]].to_string(index=False))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
