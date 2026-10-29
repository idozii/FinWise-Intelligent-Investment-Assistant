"""M3: prequential interval calibration and safe abstention for offline forecasts.

Uses only errors from previously matured target dates at each origin.
No exchangeability, conditional-coverage, or trading-profit guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "finwise-prequential-conformal-v1"


def finite_sample_radius(errors: list[float], alpha: float) -> float:
    """Rank ceil((n+1)*(1-alpha)); +inf if there is no finite order statistic."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not errors:
        return float("inf")
    data = np.asarray(errors, dtype=float)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Calibration errors must be nonnegative and finite")
    rank = int(np.ceil((len(data) + 1) * (1 - alpha)))
    if rank > len(data):
        return float("inf")
    return float(np.partition(data, rank - 1)[rank - 1])


def intervals(
    predictions: pd.DataFrame,
    *,
    alpha: float = 0.1,
    min_calibration: int = 30,
    calibration_window: int = 60,
    max_relative_width: float = 0.10,
    default_asset: str = "AAPL",
) -> pd.DataFrame:
    """Prequential absolute-residual intervals, per asset/model/horizon.

    Prediction timestamp is train_end. Calibration can use an earlier
    forecast's error only if its target_date <= current train_end.
    No future target labels are read when constructing current bounds.
    """
    if not 0 < alpha < 1 or min_calibration < 1 or calibration_window < min_calibration:
        raise ValueError("Require 0<alpha<1, 1<=min_calibration<=calibration_window")
    if not 0 < max_relative_width <= 10:
        raise ValueError("max_relative_width must be positive and <= 10")
    required = {"model", "horizon_step", "origin_index", "train_end", "target_date",
                "last_observed_close", "actual", "predicted"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    source = predictions.copy()
    if "asset" not in source:
        source.insert(0, "asset", default_asset)
    for col in ("train_end", "target_date"):
        source[col] = pd.to_datetime(source[col], utc=True, errors="raise")
        if source[col].isna().any():
            raise ValueError(f"Missing {col}")
    if (source.target_date <= source.train_end).any():
        raise ValueError("Target dates must be strictly later than forecast origins")
    for col in ("last_observed_close", "actual", "predicted"):
        source[col] = pd.to_numeric(source[col], errors="raise")
        if not np.isfinite(source[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must be finite")
    if (source.last_observed_close <= 0).any() or (source.actual <= 0).any():
        raise ValueError("Observed prices must be positive")
    if source.duplicated(["asset", "model", "horizon_step", "origin_index"]).any():
        raise ValueError("Duplicate asset/model/horizon/origin rows")

    records = []
    for _, data in source.groupby(["asset", "model", "horizon_step"], sort=True):
        data = data.sort_values(["train_end", "origin_index"])
        errors: list[tuple[pd.Timestamp, float]] = []
        for row in data.itertuples(index=False):
            matured = [err for date, err in errors if date <= row.train_end]
            recent = matured[-calibration_window:]
            radius = (finite_sample_radius(recent, alpha)
                      if len(recent) >= min_calibration else float("inf"))
            ready = bool(np.isfinite(radius))
            pred = float(row.predicted)
            last = float(row.last_observed_close)
            lower = max(0.0, pred - radius) if ready else float("nan")
            upper = pred + radius if ready else float("nan")
            width_ratio = (upper - lower) / last if ready else float("nan")
            accept = bool(ready and width_ratio <= max_relative_width)
            records.append({
                **row._asdict(),
                "calibration_n": len(recent),
                "radius": radius if ready else float("nan"),
                "interval_lower": lower,
                "interval_upper": upper,
                "relative_width": width_ratio,
                "interval_ready": ready,
                "accept_forecast": accept,
                "covered": bool(lower <= row.actual <= upper) if ready else None,
            })
            # Queue the outcome for future origins. Never use it in this origin.
            errors.append((row.target_date, float(abs(row.actual - pred))))
    result = pd.DataFrame(records)
    for col in ("train_end", "target_date"):
        result[col] = result[col].dt.strftime("%Y-%m-%d")
    return result.sort_values(
        ["asset", "model", "horizon_step", "origin_index"]
    ).reset_index(drop=True)


def report(annotated: pd.DataFrame) -> pd.DataFrame:
    """Describe empirical coverage and selected-risk MAE (not formal guarantees)."""
    rows = []
    for (asset, model, horizon), group in annotated.groupby(
        ["asset", "model", "horizon_step"], sort=True
    ):
        ready = group.loc[group.interval_ready]
        selected = ready.loc[ready.accept_forecast]
        rows.append({
            "asset": asset, "model": model, "horizon_step": int(horizon),
            "n_origins": len(group), "n_calibrated": len(ready),
            "n_accepted": len(selected),
            "acceptance_rate_all_origins": len(selected) / len(group),
            "empirical_coverage_ready": (float(ready.covered.astype(bool).mean())
                                         if len(ready) else float("nan")),
            "mean_relative_width_ready": (float(ready.relative_width.mean())
                                          if len(ready) else float("nan")),
            "coverage_accepted": (float(selected.covered.astype(bool).mean())
                                  if len(selected) else float("nan")),
            "mae_accepted": (float(np.mean(np.abs(selected.actual - selected.predicted)))
                             if len(selected) else float("nan")),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset", default="AAPL",
                        help="Used only when input CSV has no asset column")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-calibration", type=int, default=30)
    parser.add_argument("--calibration-window", type=int, default=60)
    parser.add_argument("--max-relative-width", type=float, default=0.10)
    args = parser.parse_args(argv)
    source = pd.read_csv(args.predictions)
    annotated = intervals(
        source, alpha=args.alpha,
        min_calibration=args.min_calibration,
        calibration_window=args.calibration_window,
        max_relative_width=args.max_relative_width,
        default_asset=args.asset,
    )
    summary = report(annotated)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_dir / "interval_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "interval_metrics.csv", index=False)
    manifest = {
        "protocol": PROTOCOL,
        "stage": "EXPLORATORY_PREQUENTIAL_NOT_FINITE_SAMPLE_COVERAGE_GUARANTEE",
        "prediction_source": str(args.predictions),
        "prediction_source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "alpha": args.alpha, "min_calibration": args.min_calibration,
        "calibration_window": args.calibration_window,
        "max_relative_width": args.max_relative_width,
        "selection": "accept if finite past-only residual interval and width/last observed close <= fixed cutoff",
        "calibration": "per asset/model/horizon; strictly matured targets only",
        "limitations": [
            "Time-series distributions may shift; conformal exchangeability does not hold",
            "Warmup abstentions and selective coverage are reported separately",
            "Thresholds were not statistically validated on independent data",
            "This does not establish investment utility, trading returns or safe financial decisions",
        ],
        "external_requests": 0,
    }
    (args.output_dir / "interval_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved exploratory prequential interval analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
