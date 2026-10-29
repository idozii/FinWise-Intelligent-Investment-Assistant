"""R8 development-only adaptive calibration benchmark (NOT a holdout test).

Compares fixed rolling M3 calibration with a delayed-feedback, per-horizon
ACI-style alpha update. No future outcome may update a current interval.
No empirical coverage guarantee is claimed; no M5 external assets allowed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.uncertainty import finite_sample_radius, intervals as m3_intervals

PROTOCOL = "finwise-r8-development-aci-style-v1"
ALLOWED_DEV = frozenset(("AAPL", "MSFT", "SPY", "TSLA"))
FORBIDDEN_TEST = frozenset(("GLD", "IWM", "XLE", "XLV"))
NOMINAL_ALPHA = .10
GAMMA = .02
ALPHA_MIN, ALPHA_MAX = .01, .50
MIN_CALIBRATION = 30
WINDOW = 60
WIDTH_LIMIT = .10
COLS = ("asset", "model", "horizon_step", "origin_index", "train_end",
        "target_date", "last_observed_close", "actual", "predicted")


def validate_development(source: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLS) - set(source)
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    if source.empty or set(source.asset) != ALLOWED_DEV:
        raise ValueError("R8 development requires exactly AAPL/MSFT/SPY/TSLA, never M5 test assets")
    if set(source.asset) & FORBIDDEN_TEST:
        raise ValueError("M5 frozen test assets are not R8 development data")
    if set(source.model) != {"naive", "ridge"}:
        raise ValueError("R8 compares frozen naive/ridge models only")
    if set(source.horizon_step) != {1, 7}:
        raise ValueError("R8 compares H1 and H7 only")
    data = source.copy()
    data["train_end"] = pd.to_datetime(data.train_end, utc=True, errors="raise")
    data["target_date"] = pd.to_datetime(data.target_date, utc=True, errors="raise")
    if data[["train_end", "target_date"]].isna().any().any():
        raise ValueError("Forecast timestamp missing")
    if (data.target_date <= data.train_end).any():
        raise ValueError("Targets must mature strictly after their forecast origins")
    if data.duplicated(["asset", "model", "horizon_step", "origin_index"]).any():
        raise ValueError("Duplicate forecast origins")
    for col in ("last_observed_close", "actual", "predicted"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col].to_numpy(float)).all():
            raise ValueError("Forecast prices must be finite")
    if (data.actual <= 0).any() or (data.last_observed_close <= 0).any():
        raise ValueError("Price inputs must be positive")
    return data


def adaptive_intervals(source: pd.DataFrame) -> pd.DataFrame:
    data = validate_development(source)
    out = []
    for _, group in data.groupby(["asset", "model", "horizon_step"], sort=True):
        group = group.sort_values(["train_end", "origin_index"])
        issued_errors = []
        feedback_queue = []
        current_alpha = NOMINAL_ALPHA
        for row in group.itertuples(index=False):
            # Only intervals issued at past origins and labels matured by NOW.
            remaining = []
            updates = 0
            for target_date, covered in feedback_queue:
                if target_date <= row.train_end:
                    current_alpha = float(np.clip(
                        current_alpha + GAMMA * (NOMINAL_ALPHA - int(not covered)),
                        ALPHA_MIN, ALPHA_MAX,
                    ))
                    updates += 1
                else:
                    remaining.append((target_date, covered))
            feedback_queue = remaining
            matured = [error for date, error in issued_errors if date <= row.train_end]
            recent = matured[-WINDOW:]
            radius = (finite_sample_radius(recent, current_alpha)
                      if len(recent) >= MIN_CALIBRATION else float("inf"))
            ready = bool(np.isfinite(radius))
            predicted = float(row.predicted)
            last_close = float(row.last_observed_close)
            lower = max(0., predicted-radius) if ready else float("nan")
            upper = predicted+radius if ready else float("nan")
            relative_width = (upper-lower)/last_close if ready else float("nan")
            covered = bool(lower <= float(row.actual) <= upper) if ready else None
            out.append({
                **row._asdict(), "calibration_method": "delayed_feedback_aci_style",
                "alpha_used": current_alpha, "feedback_applied_this_origin": updates,
                "calibration_n": len(recent), "radius": radius if ready else float("nan"),
                "interval_lower": lower, "interval_upper": upper,
                "relative_width": relative_width,
                "interval_ready": ready,
                "accept_forecast": bool(ready and relative_width <= WIDTH_LIMIT),
                "covered": covered,
            })
            # Error and coverage are queued AFTER issuing the interval.
            issued_errors.append((row.target_date, abs(float(row.actual)-predicted)))
            if ready:
                feedback_queue.append((row.target_date, covered))
    result = pd.DataFrame(out)
    for col in ("train_end", "target_date"):
        result[col] = result[col].dt.strftime("%Y-%m-%d")
    return result.sort_values(
        ["asset", "model", "horizon_step", "origin_index"]
    ).reset_index(drop=True)


def compare(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = validate_development(source)
    standard = m3_intervals(
        data, alpha=NOMINAL_ALPHA,
        min_calibration=MIN_CALIBRATION, calibration_window=WINDOW,
        max_relative_width=WIDTH_LIMIT,
    )
    standard.insert(standard.columns.get_loc("calibration_n"),
                    "calibration_method", "fixed_rolling")
    standard["alpha_used"] = NOMINAL_ALPHA
    standard["feedback_applied_this_origin"] = 0
    adjusted = adaptive_intervals(data)
    annotated = pd.concat([standard, adjusted], ignore_index=True)
    results = []
    for (asset, model, h, method), g in annotated.groupby(
        ["asset", "model", "horizon_step", "calibration_method"], sort=True
    ):
        ready = g.loc[g.interval_ready]
        selected = ready.loc[ready.accept_forecast]
        results.append({
            "asset": asset, "model": model, "horizon_step": int(h),
            "calibration_method": method,
            "n_origins": len(g), "n_ready": len(ready), "n_accepted": len(selected),
            "coverage_ready": float(ready.covered.astype(bool).mean()) if len(ready) else float("nan"),
            "coverage_selected": float(selected.covered.astype(bool).mean()) if len(selected) else float("nan"),
            "mean_relative_width_ready": float(ready.relative_width.mean()) if len(ready) else float("nan"),
            "acceptance_rate_all": len(selected)/len(g),
            "empirical_absolute_coverage_gap_from_90pct": (
                abs(float(ready.covered.astype(bool).mean())-.9)
                if len(ready) else float("nan")
            ),
        })
    return annotated, pd.DataFrame(results)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True,
                   help="M2.1 DEVELOPMENT asset_predictions.csv; never M5 files")
    p.add_argument("--output-dir", type=Path, default=Path("reports/backtest/r8_development"))
    args = p.parse_args(argv)
    if "m5_" in str(args.predictions).lower() or "holdout" in str(args.predictions).lower():
        raise ValueError("Never supply M5 holdout predictions to R8")
    if args.output_dir.resolve() == args.predictions.parent.resolve():
        raise ValueError("Do not write R8 outputs into original input folder")
    source = pd.read_csv(args.predictions)
    annotated, stats = compare(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_dir/"r8_intervals_development.csv", index=False)
    stats.to_csv(args.output_dir/"r8_comparison_development.csv", index=False)
    (args.output_dir/"r8_manifest.json").write_text(json.dumps({
        "protocol": PROTOCOL,
        "stage": "EXPLORATORY_DEVELOPMENT_ONLY_NOT_INDEPENDENT_TEST",
        "source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "assets": sorted(ALLOWED_DEV),
        "methods": ["fixed_rolling", "delayed_feedback_aci_style"],
        "gamma": GAMMA, "nominal_alpha": NOMINAL_ALPHA,
        "alpha_bounds": [ALPHA_MIN, ALPHA_MAX],
        "min_calibration": MIN_CALIBRATION, "window": WINDOW,
        "width_limit": WIDTH_LIMIT,
        "future_label_update_prohibited": True,
        "m5_test_assets_used": False,
        "model_fitted": False,
        "caveat": "ACI-style baseline only, clipped alpha and delayed overlapping labels; "
                  "NO published theoretical guarantees transferred to this implementation.",
    }, indent=2)+"\n")
    print(f"R8 DEV ONLY: {len(annotated)} interval rows, {len(stats)} descriptive groups")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
