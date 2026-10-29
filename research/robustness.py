"""M2.1: offline multi-asset robustness with time-respecting regimes and paired CIs.

Uses exploratory expanding-window ML predictions; NO independent holdout claim.
Run with local historical CSV files. Never downloads prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.backtest import load_price_csv, summarize
from research.ml_backtest import ml_walk_forward

PROTOCOL = "finwise-m21-multiasset-robustness-v1"


def origin_volatility(prices: pd.DataFrame, min_train: int, window: int = 20) -> dict[int, str]:
    """Freeze volatility thresholds using ONLY data before the first forecast."""
    close = prices["close"].to_numpy(dtype=float)
    if min_train < window + 31:
        raise ValueError("Insufficient history to freeze volatility thresholds")
    returns = np.diff(np.log(close))
    # At origin i, available log returns are returns[:i-1].
    # Thresholds are computed at past hypothetical origins < min_train.
    reference = np.array([
        np.std(returns[i - window - 1:i - 1], ddof=0)
        for i in range(window + 2, min_train)
    ])
    if not np.isfinite(reference).all() or not len(reference):
        raise ValueError("Unable to freeze historical volatility thresholds")
    lo, hi = np.quantile(reference, [1 / 3, 2 / 3])
    result: dict[int, str] = {}
    for i in range(min_train, len(close)):
        vol = float(np.std(returns[i - window - 1:i - 1], ddof=0))
        result[i] = "low" if vol < lo else "high" if vol > hi else "medium"
    return result


def paired_block_bootstrap(
    group: pd.DataFrame,
    *,
    block_length: int = 5,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """CI for mean MAE(model)-MAE(naive), resampling paired forecast origins.

    Positive difference means model is WORSE. CI is descriptive; overlapping
    horizons, regime dependence and repeated exploratory comparisons limit
    inferential interpretation.
    """
    if block_length < 1 or iterations < 100:
        raise ValueError("block_length must be >=1; iterations >=100")
    baseline = group.loc[group["model"] == "naive", ["origin_index", "actual", "predicted"]]
    if baseline.origin_index.duplicated().any():
        raise ValueError("Duplicate naive origin")
    model_names = [name for name in group.model.unique() if name != "naive"]
    if len(model_names) != 1:
        raise ValueError("Expected exactly one non-naive model and its paired naive")
    challenger = group.loc[group.model == model_names[0]]
    paired = challenger.merge(baseline, on="origin_index", validate="one_to_one",
                             suffixes=("_model", "_naive")).sort_values("origin_index")
    if len(paired) != len(challenger) or len(paired) != len(baseline):
        raise ValueError("Unmatched forecast origins")
    if not np.allclose(paired.actual_model, paired.actual_naive, rtol=0, atol=0):
        raise ValueError("Actual prices differ between paired forecasts")

    diff = (np.abs(paired.predicted_model.to_numpy() - paired.actual_model.to_numpy())
            - np.abs(paired.predicted_naive.to_numpy() - paired.actual_naive.to_numpy()))
    n = len(diff)
    if n < 2:
        return float(np.mean(diff)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = min(block_length, n)
    start_choices = np.arange(n - b + 1)
    draws = np.empty(iterations)
    for j in range(iterations):
        starts = rng.choice(start_choices, size=int(np.ceil(n / b)), replace=True)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
        draws[j] = float(diff[idx].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(diff.mean()), float(low), float(high)


def analyze_asset(
    name: str,
    csv_path: Path,
    *,
    min_train: int,
    horizons: tuple[int, ...],
    step: int,
    models: tuple[str, ...],
    block_length: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    prices = load_price_csv(csv_path)
    prediction = ml_walk_forward(
        prices, min_train=min_train, horizons=horizons, step=step, models=models
    )
    regimes = origin_volatility(prices, min_train=min_train)
    prediction.insert(0, "asset", name)
    prediction["regime"] = prediction["origin_index"].map(regimes)
    if prediction.regime.isna().any():
        raise ValueError("Unable to associate every origin with a past-only regime")
    summaries = []
    for horizon in horizons:
        for model in models:
            if model == "naive":
                continue
            pair = prediction[(prediction.horizon_step == horizon)
                              & (prediction.model.isin(("naive", model)))]
            difference, low, high = paired_block_bootstrap(
                pair, block_length=block_length, iterations=bootstrap_iterations,
            )
            summaries.append({
                "asset": name, "horizon_step": horizon, "model": model,
                "n_origins": len(pair) // 2,
                "mae_delta_model_minus_naive": difference,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "model_worse_if_delta_positive": difference > 0,
            })
    overall = pd.DataFrame(summaries)
    per_regime = []
    for regime in ("low", "medium", "high"):
        scoped = prediction[prediction.regime == regime]
        if scoped.empty:
            continue
        summary = summarize(scoped.drop(columns=["asset", "regime"]))
        summary.insert(0, "asset", name)
        summary.insert(1, "regime", regime)
        per_regime.append(summary)
    regime_metrics = pd.concat(per_regime, ignore_index=True) if per_regime else pd.DataFrame()
    metadata = {
        "asset": name, "csv_path": str(csv_path),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "records": len(prices), "n_origins": int(prediction.origin_index.nunique()),
        "date_start": prices.date.iloc[0].date().isoformat(),
        "date_end": prices.date.iloc[-1].date().isoformat(),
    }
    return prediction, overall, regime_metrics, metadata


def parse_assets(values: list[str]) -> list[tuple[str, Path]]:
    pairs = []
    used = set()
    for value in values:
        if "=" not in value:
            raise ValueError("Assets must be ASSET=CSV_PATH")
        name, filename = value.split("=", 1)
        name = name.strip().upper()
        if not name or not filename.strip() or name in used:
            raise ValueError("Empty or repeated asset or path")
        used.add(name)
        pairs.append((name, Path(filename)))
    if not pairs:
        raise ValueError("At least one asset required")
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", required=True, metavar="ASSET=CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--horizons", default="1,7")
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--models", default="naive,ridge")
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args(argv)
    horizons = tuple(int(s) for s in args.horizons.split(","))
    models = tuple(s.strip() for s in args.models.split(","))
    paths = parse_assets(args.assets)
    prediction_frames, comparisons, regime_frames, source_data = [], [], [], []
    for asset, path in paths:
        prediction, compare, regime, meta = analyze_asset(
            asset, path, min_train=args.min_train, horizons=horizons,
            step=args.step, models=models, block_length=args.block_length,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        prediction_frames.append(prediction)
        comparisons.append(compare)
        regime_frames.append(regime)
        source_data.append(meta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "asset_predictions.csv", index=False)
    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    if regime_frames:
        pd.concat(regime_frames, ignore_index=True).to_csv(
            args.output_dir / "regime_metrics.csv", index=False)
    manifest = {
        "protocol": PROTOCOL, "stage": "EXPLORATORY_NOT_INDEPENDENT_TEST",
        "sources": source_data, "horizons": horizons, "models": models,
        "min_train": args.min_train, "step": args.step,
        "volatility_regime": "20-observation return volatility with thresholds frozen before first origin",
        "ci_method": "moving block bootstrap of paired absolute error differences",
        "block_length_origins": args.block_length,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": 42,
        "multiple_comparison_correction": "none; descriptive exploratory comparisons",
        "data_provider_requests": 0,
    }
    (args.output_dir / "robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(combined.to_string(index=False))
    print(f"Saved M2.1 exploratory artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
