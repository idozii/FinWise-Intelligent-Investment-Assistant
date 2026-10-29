"""FinWise M3.1: audit prequential coverage and compare abstaining models fairly.

Offline descriptive diagnostics; no statistical coverage or trading guarantees.
The acceptance rule depends ONLY on forecast-origin interval width.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "finwise-m31-calibration-audit-v1"
KEY = ["asset", "model", "horizon_step", "origin_index"]
REQUIRED_SOURCE = set(KEY + [
    "regime", "train_end", "target_date", "actual", "predicted",
    "last_observed_close"
])
REQUIRED_INTERVAL = set(KEY + [
    "train_end", "target_date", "actual", "predicted", "last_observed_close",
    "interval_ready", "accept_forecast", "interval_lower", "interval_upper",
    "relative_width", "covered", "calibration_n",
])


def _bool_column(series: pd.Series, name: str) -> pd.Series:
    """Accept strict booleans or literal CSV True/False; reject nulls."""
    mapped = series.map({True: True, False: False, "True": True, "False": False})
    if mapped.isna().any():
        raise ValueError(f"Invalid or missing {name} flags")
    return mapped.astype(bool)


def validated_rows(source_path: Path, interval_path: Path) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    intervals = pd.read_csv(interval_path)
    for name, frame, required in (
        ("source", source, REQUIRED_SOURCE),
        ("interval", intervals, REQUIRED_INTERVAL),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame.empty or frame.duplicated(KEY).any():
            raise ValueError(f"Empty or duplicate {name} origin/model/horizon keys")
    if len(source) != len(intervals):
        raise ValueError("Source/interval row count mismatch")

    for col in ("interval_ready", "accept_forecast"):
        intervals[col] = _bool_column(intervals[col], col)
    paired = source.merge(
        intervals, on=KEY, validate="one_to_one",
        how="outer", indicator=True, suffixes=("_source", "_interval"),
    )
    if (paired["_merge"] != "both").any():
        raise ValueError("Source and interval outputs have unmatched rows")

    for col in ("train_end", "target_date"):
        if not paired[f"{col}_source"].astype(str).equals(
            paired[f"{col}_interval"].astype(str)
        ):
            raise ValueError(f"Source/interval {col} mismatch")
    for col in ("actual", "predicted", "last_observed_close"):
        a = paired[f"{col}_source"].to_numpy(dtype=float)
        b = paired[f"{col}_interval"].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{col}: nonfinite values")
        if not np.allclose(a, b, rtol=1e-11, atol=1e-8):
            raise ValueError(f"{col}: source/interval values differ")
        paired[col] = a
    if (paired["accept_forecast"] & ~paired["interval_ready"]).any():
        raise ValueError("Uncalibrated forecasts cannot be accepted")

    paired["abs_error"] = (paired.actual - paired.predicted).abs()
    # M3 copies the source's regime column into interval_predictions.csv.
    # A Pandas merge therefore suffixes both copies; older M3 exports may
    # omit regime on the interval side, so accept that schema as well.
    if "regime_source" in paired.columns:
        if "regime_interval" not in paired.columns:
            raise ValueError("Source/interval regime columns are incomplete")
        source_regime = paired["regime_source"]
        interval_regime = paired["regime_interval"]
        if (source_regime.isna().any() or interval_regime.isna().any()
                or not source_regime.astype(str).equals(interval_regime.astype(str))):
            raise ValueError("Source/interval regime values differ")
    else:
        source_regime = paired["regime"]
        if source_regime.isna().any():
            raise ValueError("Missing source regime values")
    paired["regime"] = source_regime.astype(str)
    ready = paired.loc[paired.interval_ready]
    for col in ("interval_lower", "interval_upper", "relative_width"):
        if not np.isfinite(ready[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Ready intervals contain missing {col}")
    if (ready.interval_lower > ready.interval_upper).any():
        raise ValueError("Inverted prediction interval")
    recomputed = ((ready.actual >= ready.interval_lower)
                  & (ready.actual <= ready.interval_upper))
    reported = _bool_column(ready.covered, "covered")
    if not np.array_equal(recomputed.to_numpy(), reported.to_numpy()):
        raise ValueError("Reported interval coverage differs from source values")
    paired["covered_bool"] = False
    paired.loc[ready.index, "covered_bool"] = recomputed
    return paired.sort_values(KEY).reset_index(drop=True)


def model_audit(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    result = []
    for (asset, model, horizon), group in rows.groupby(
        ["asset", "model", "horizon_step"], sort=True
    ):
        ready = group[group.interval_ready]
        selected = ready[ready.accept_forecast]
        coverage = float(ready.covered_bool.mean()) if len(ready) else float("nan")
        result.append({
            "asset": asset, "model": model, "horizon_step": horizon,
            "n_total": len(group), "n_ready": len(ready),
            "n_accepted": len(selected),
            "acceptance_rate_all": len(selected) / len(group),
            "acceptance_rate_ready": len(selected) / len(ready) if len(ready) else float("nan"),
            "coverage_ready": coverage,
            "coverage_gap_vs_nominal_pp": 100 * (coverage - (1-alpha)),
            "coverage_selected": (float(selected.covered_bool.mean())
                                  if len(selected) else float("nan")),
            "mae_selected": (float(selected.abs_error.mean())
                             if len(selected) else float("nan")),
            "mean_relative_width_ready": (float(ready.relative_width.mean())
                                          if len(ready) else float("nan")),
            "selection_sample_warning": "SMALL_N" if len(selected) < 30 else "OK",
        })
    return pd.DataFrame(result)


def regime_audit(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (asset, model, horizon, regime), group in rows.groupby(
        ["asset", "model", "horizon_step", "regime"], sort=True
    ):
        ready = group[group.interval_ready]
        selected = ready[ready.accept_forecast]
        result.append({
            "asset": asset, "model": model, "horizon_step": horizon, "regime": regime,
            "n_total": len(group), "n_ready": len(ready), "n_accepted": len(selected),
            "empirical_coverage_ready": (float(ready.covered_bool.mean())
                                         if len(ready) else float("nan")),
            "acceptance_rate_ready": (len(selected)/len(ready)
                                      if len(ready) else float("nan")),
            "empirical_coverage_selected": (float(selected.covered_bool.mean())
                                            if len(selected) else float("nan")),
        })
    return pd.DataFrame(result)


def width_sensitivity(rows: pd.DataFrame, thresholds: tuple[float, ...]) -> pd.DataFrame:
    if not thresholds or any(not np.isfinite(t) or t <= 0 for t in thresholds):
        raise ValueError("Sensitivity thresholds must be finite and positive")
    result = []
    for (asset, model, horizon), group in rows.groupby(
        ["asset", "model", "horizon_step"], sort=True
    ):
        ready = group[group.interval_ready]
        for threshold in thresholds:
            # No future actuals or observed coverage determine selection.
            accepted = ready.loc[ready.relative_width <= threshold]
            result.append({
                "asset": asset, "model": model, "horizon_step": horizon,
                "relative_width_threshold": threshold,
                "n_ready": len(ready), "n_accepted": len(accepted),
                "acceptance_rate_ready": (len(accepted)/len(ready)
                                          if len(ready) else float("nan")),
                "coverage_accepted": (float(accepted.covered_bool.mean())
                                      if len(accepted) else float("nan")),
                "mae_accepted": (float(accepted.abs_error.mean())
                                 if len(accepted) else float("nan")),
            })
    return pd.DataFrame(result)


def matched_selection_audit(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare naive vs each challenger on precisely the SAME accepted origins."""
    output = []
    for (asset, horizon), group in rows.groupby(["asset", "horizon_step"], sort=True):
        baseline = group[group.model == "naive"][
            ["origin_index", "actual", "abs_error", "interval_ready",
             "accept_forecast"]
        ]
        if baseline.empty or baseline.origin_index.duplicated().any():
            raise ValueError(f"Expected a unique naive row per origin: {asset} H{horizon}")
        for model in sorted(set(group.model) - {"naive"}):
            contender = group[group.model == model][
                ["origin_index", "actual", "abs_error", "interval_ready",
                 "accept_forecast"]
            ]
            merged = baseline.merge(
                contender, on="origin_index", how="outer", indicator=True,
                validate="one_to_one", suffixes=("_naive", "_model"),
            )
            if (merged._merge != "both").any() or len(merged) != len(baseline):
                raise ValueError("Mismatched origins for model/naive comparison")
            if not np.allclose(merged.actual_naive, merged.actual_model, rtol=0, atol=1e-8):
                raise ValueError("Actual values differ for paired forecasts")
            both_ready = merged.interval_ready_naive & merged.interval_ready_model
            both_accepted = (merged.accept_forecast_naive
                             & merged.accept_forecast_model & both_ready)
            selected = merged.loc[both_accepted]
            n = len(selected)
            output.append({
                "asset": asset, "horizon_step": horizon, "model": model,
                "n_origins": len(merged),
                "n_both_calibrated": int(both_ready.sum()),
                "n_both_accepted": n,
                "naive_mae_same_accepted": (float(selected.abs_error_naive.mean())
                                            if n else float("nan")),
                "model_mae_same_accepted": (float(selected.abs_error_model.mean())
                                            if n else float("nan")),
                "delta_mae_model_minus_naive_same_accepted": (
                    float((selected.abs_error_model-selected.abs_error_naive).mean())
                    if n else float("nan")
                ),
                "comparison_status": ("NO_MATCHED_ACCEPTED" if n == 0
                                      else "SMALL_N_EXPLORATORY" if n < 30
                                      else "EXPLORATORY"),
            })
    return pd.DataFrame(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05,0.10,0.20,0.40")
    args = parser.parse_args(argv)
    source = args.robustness_dir / "asset_predictions.csv"
    intervals = args.uncertainty_dir / "interval_predictions.csv"
    conf = args.uncertainty_dir / "interval_manifest.json"
    manifest = json.loads(conf.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "finwise-prequential-conformal-v1":
        raise ValueError("Unsupported uncertainty protocol")
    alpha = float(manifest["alpha"])
    if not 0 < alpha < 1:
        raise ValueError("Invalid source alpha")
    values = tuple(float(v.strip()) for v in args.thresholds.split(","))
    rows = validated_rows(source, intervals)
    reports = {
        "model_audit.csv": model_audit(rows, alpha),
        "regime_coverage.csv": regime_audit(rows),
        "width_sensitivity.csv": width_sensitivity(rows, values),
        "matched_accepted_comparisons.csv": matched_selection_audit(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in reports.items():
        frame.to_csv(args.output_dir / filename, index=False)
    audit_manifest = {
        "protocol": PROTOCOL,
        "stage": "EXPLORATORY_DIAGNOSTICS_NOT_INDEPENDENT_HOLDOUT",
        "sources_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, intervals, conf)
        },
        "nominal_alpha": alpha,
        "original_max_relative_width": manifest["max_relative_width"],
        "sensitivity_thresholds": values,
        "selection_rule": "threshold uses interval relative_width at each historical origin; outcome not used for selection",
        "fair_comparison": "naive and challenger compared only on origins both actually accepted",
        "limitations": "multiple threshold analyses are exploratory; do not tune using these and claim independent test superiority",
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(audit_manifest, indent=2) + "\n", encoding="utf-8")
    print("=== Empirical calibration and abstention ===")
    print(reports["model_audit.csv"].to_string(index=False))
    print("\n=== Fair comparison on jointly accepted forecast origins ===")
    print(reports["matched_accepted_comparisons.csv"].to_string(index=False))
    print(f"\nSaved M3.1 audit to {args.output_dir}")


if __name__ == "__main__":
    main()
