"""M4.1: exact additive Ridge prediction decomposition for FinWise.

Standardized Ridge log-return contributions reconstruct the model's *own*
predicted log-return, not causal influence or SHAP values. No network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.backtest import load_price_csv
from research.ml_backtest import (
    FEATURE_NAMES, LOOKBACK, features_at, historical_training_examples, fit_predict,
)
from research.robustness import parse_assets

PROTOCOL = "finwise-m41-ridge-decomposition-v1"


def explain_at_origin(
    log_prices: np.ndarray, origin: int, horizon: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Refit only with matured labels and exactly reconstruct Ridge forecast."""
    if origin >= len(log_prices) or origin < LOOKBACK + horizon + 2:
        raise ValueError("Invalid origin or insufficient matured labels")
    x_train, y_train = historical_training_examples(log_prices, origin, horizon)
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    estimator.fit(x_train, y_train)
    current = features_at(log_prices, origin - 1).reshape(1, -1)
    scaler: StandardScaler = estimator.named_steps["standardscaler"]
    ridge: Ridge = estimator.named_steps["ridge"]
    transformed = scaler.transform(current)[0]
    coefficients = np.asarray(ridge.coef_, dtype=float)
    contributions = transformed * coefficients
    intercept = float(ridge.intercept_)
    predicted_log_return = intercept + float(contributions.sum())
    model_log_return = float(estimator.predict(current)[0])
    if not np.isclose(predicted_log_return, model_log_return, rtol=0, atol=1e-11):
        raise ValueError("Feature contributions do not reproduce Ridge log-return")
    start_price = float(np.exp(log_prices[origin - 1]))
    explained_price = start_price * float(np.exp(predicted_log_return))
    api_equivalent_price = fit_predict(log_prices, origin, horizon, "ridge")
    if not np.isclose(explained_price, api_equivalent_price, rtol=1e-11, atol=1e-8):
        raise ValueError("Decomposition does not match M2 model forecast")
    details = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "raw_feature_value": current[0],
        "standardized_feature_value": transformed,
        "ridge_standardized_coefficient": coefficients,
        "contribution_log_return": contributions,
    })
    residual = {
        "intercept_log_return": intercept,
        "contribution_sum_log_return": float(contributions.sum()),
        "predicted_log_return": predicted_log_return,
        "predicted_close": explained_price,
        "reference_model_close": api_equivalent_price,
        "max_abs_reconstruction_error": float(abs(explained_price-api_equivalent_price)),
    }
    return details, residual


def build_attributions(
    prices: pd.DataFrame, asset: str, *,
    min_train: int = 252, horizons: tuple[int, ...] = (1, 7),
    step: int = 7, max_origins: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not horizons or any(h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("Invalid horizons")
    if step < 1 or max_origins < 1:
        raise ValueError("step and max_origins must be positive")
    if min_train < LOOKBACK + max(horizons) + 2:
        raise ValueError("Insufficient min_train for matured features/labels")
    if len(prices) < min_train + max(horizons):
        raise ValueError("Not enough rows to evaluate one complete horizon")
    close = prices["close"].to_numpy(dtype=float)
    if (close <= 0).any() or not np.isfinite(close).all():
        raise ValueError("Prices must be strictly positive and finite")
    if not prices.date.is_monotonic_increasing or prices.date.duplicated().any():
        raise ValueError("Expected chronologically sorted unique dates")
    candidates = list(range(min_train, len(prices)-max(horizons)+1, step))
    positions = np.linspace(0, len(candidates)-1,
                            num=min(len(candidates), max_origins), dtype=int)
    origins = [candidates[i] for i in sorted(set(positions))]
    outputs, verification = [], []
    log_prices = np.log(close)
    for origin in origins:
        for h in horizons:
            details, audit = explain_at_origin(log_prices, origin, h)
            target_idx = origin - 1 + h
            date = prices.iloc[origin-1]["date"].date().isoformat()
            outcome = prices.iloc[target_idx]["date"].date().isoformat()
            details.insert(0, "asset", asset)
            details.insert(1, "origin_index", origin)
            details.insert(2, "train_end", date)
            details.insert(3, "target_date", outcome)
            details.insert(4, "horizon_step", h)
            details["actual_close"] = float(close[target_idx])
            details["last_observed_close"] = float(close[origin-1])
            details["predicted_close"] = audit["predicted_close"]
            outputs.append(details)
            verification.append({
                "asset": asset, "origin_index": origin,
                "train_end": date, "target_date": outcome, "horizon_step": h,
                **audit,
            })
    results = pd.concat(outputs, ignore_index=True)
    verified = pd.DataFrame(verification)
    rows = []
    for (asset_name, horizon, feature), group in results.groupby(
        ["asset", "horizon_step", "feature"], sort=True
    ):
        x = group.contribution_log_return.to_numpy(dtype=float)
        rows.append({
            "asset": asset_name, "horizon_step": int(horizon),
            "feature": feature, "n_origins": len(group),
            "mean_abs_contribution_log_return": float(np.mean(np.abs(x))),
            "mean_signed_contribution_log_return": float(np.mean(x)),
            "fraction_positive_contribution": float(np.mean(x > 0)),
            "fraction_negative_contribution": float(np.mean(x < 0)),
            "attribution_type": "exact_linear_model_decomposition_not_SHAP_not_causal",
        })
    summary = pd.DataFrame(rows).sort_values(
        ["asset", "horizon_step", "mean_abs_contribution_log_return"],
        ascending=[True, True, False],
    )
    return results, summary, verified


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", required=True, metavar="ASSET=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--horizons", default="1,7")
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--max-origins", type=int, default=32)
    args = parser.parse_args(argv)
    horizons = tuple(int(n.strip()) for n in args.horizons.split(","))
    details, summaries, checks, sources = [], [], [], []
    for asset, path in parse_assets(args.assets):
        prices = load_price_csv(path)
        d, s, c = build_attributions(
            prices, asset, min_train=args.min_train,
            horizons=horizons, step=args.step, max_origins=args.max_origins,
        )
        details.append(d); summaries.append(s); checks.append(c)
        sources.append({
            "asset": asset, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_source_records": len(prices),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.concat(details, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    checked = pd.concat(checks, ignore_index=True)
    data.to_csv(args.output_dir / "ridge_contributions.csv", index=False)
    summary.to_csv(args.output_dir / "ridge_feature_summary.csv", index=False)
    checked.to_csv(args.output_dir / "ridge_reconstruction_checks.csv", index=False)
    manifest = {
        "protocol": PROTOCOL,
        "stage": "EXPLORATORY_MODEL_INTERNAL_NOT_CAUSAL",
        "sources": sources, "min_train": args.min_train, "horizons": horizons,
        "step": args.step, "max_origins": args.max_origins,
        "origin_selection": "evenly spaced subset of valid chronological evaluation origins",
        "feature_names": FEATURE_NAMES,
        "model": "StandardScaler followed by Ridge(alpha=10.0), fixed M2 specification",
        "interpretation": "contributions add exactly in predicted LOG-RETURN space; NOT dollar, probability, causal effect or SHAP",
        "maximum_reconstruction_discrepancy": float(checked.max_abs_reconstruction_error.max()),
        "external_requests": 0,
    }
    (args.output_dir / "attribution_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Explained {len(checked)} asset/horizon/origin Ridge forecasts")
    print(f"Maximum model reconstruction discrepancy: {manifest['maximum_reconstruction_discrepancy']:.3e}")
    print(summary.groupby(["asset", "horizon_step"]).head(4).to_string(index=False))
    print(f"Saved attribution artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
