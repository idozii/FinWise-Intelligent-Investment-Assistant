"""Offline prequential ML benchmark for FinWise: fixed horizons, past-only features.

Outputs are exploratory research artifacts, not investment advice. No network requests.
Every feature is available by the forecast origin, and training labels must have
matured before each origin. Model specifications are fixed (no test tuning).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from research.backtest import load_price_csv, summarize

PROTOCOL = "finwise-ml-walk-forward-v1"
LOOKBACK = 30
MODEL_NAMES = ("naive", "ridge", "xgb", "lightgbm")
FEATURE_NAMES = (
    "log_return_1", "log_return_3", "log_return_5", "log_return_10",
    "log_return_20", "log_return_30", "volatility_5", "volatility_20",
    "volatility_30", "price_minus_sma_5", "price_minus_sma_20",
    "positive_return_fraction_10",
)


def features_at(log_prices: np.ndarray, t: int) -> np.ndarray:
    """Build an invariant feature vector using only log prices <= index t."""
    if t < LOOKBACK or t >= len(log_prices):
        raise ValueError("At least 30 historical returns needed for features")
    past = log_prices[t - LOOKBACK : t + 1]
    returns = np.diff(past)
    vector = np.array([
        returns[-1],
        returns[-3:].sum(),
        returns[-5:].sum(),
        returns[-10:].sum(),
        returns[-20:].sum(),
        returns.sum(),
        returns[-5:].std(ddof=0),
        returns[-20:].std(ddof=0),
        returns.std(ddof=0),
        past[-1] - np.log(np.exp(past[-5:]).mean()),
        past[-1] - np.log(np.exp(past[-20:]).mean()),
        (returns[-10:] > 0).mean(),
    ], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains non-finite values")
    return vector


def historical_training_examples(
    log_prices: np.ndarray, origin: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Supervise log return t->t+h, with all labels known before origin.

    origin is first unknown row. Latest observed index is origin-1.
    Training target index must be <= origin-1, never inside the test window.
    """
    if horizon < 1 or origin > len(log_prices) or origin < LOOKBACK + horizon + 2:
        raise ValueError("Insufficient history or invalid origin/horizon")
    train_ends = range(LOOKBACK, origin - horizon)
    x = np.vstack([features_at(log_prices, t) for t in train_ends])
    y = np.array([log_prices[t + horizon] - log_prices[t] for t in train_ends])
    return x, y


def fit_predict(
    log_prices: np.ndarray, origin: int, horizon: int, model: str
) -> float:
    """Return predicted positive close for origin-1+h; no future observations."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unsupported model {model!r}: {MODEL_NAMES}")
    last_price = float(np.exp(log_prices[origin - 1]))
    if model == "naive":
        return last_price

    x_train, y_train = historical_training_examples(log_prices, origin, horizon)
    x_now = features_at(log_prices, origin - 1).reshape(1, -1)
    if model == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install research/requirements-ml.txt for xgb model") from exc
        estimator = XGBRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=10.0, objective="reg:squarederror",
            random_state=42, n_jobs=1, tree_method="hist",
        )
    else:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("Install research/requirements-ml.txt for lightgbm model") from exc
        estimator = LGBMRegressor(
            n_estimators=120, max_depth=3, num_leaves=7, learning_rate=0.035,
            min_child_samples=25, reg_lambda=10.0,
            subsample=1.0, colsample_bytree=0.85,
            random_state=42, n_jobs=1, verbosity=-1,
        )
    if model == "lightgbm":
        # Passing named features at fit and predict avoids feature-name warnings.
        fit_x = pd.DataFrame(x_train, columns=FEATURE_NAMES)
        predict_x = pd.DataFrame(x_now, columns=FEATURE_NAMES)
    else:
        fit_x, predict_x = x_train, x_now
    estimator.fit(fit_x, y_train)
    predicted_log_return = float(estimator.predict(predict_x)[0])
    prediction = last_price * np.exp(predicted_log_return)
    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError("Non-positive or non-finite predicted price")
    return float(prediction)


def ml_walk_forward(
    prices: pd.DataFrame,
    *,
    min_train: int = 252,
    horizons: tuple[int, ...] = (1, 7),
    step: int = 14,
    models: tuple[str, ...] = ("naive", "ridge", "xgb", "lightgbm"),
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> pd.DataFrame:
    """Evaluate fixed specs at identical origins/horizons, without model selection.

    A new model is fit at each origin using only historically matured labels.
    This is prequential evaluation; previous test labels can enter later fits
    after their date. It is not an isolated untouched holdout evaluation.
    """
    if min_train < LOOKBACK + max(horizons, default=0) + 2 or step < 1:
        raise ValueError("min_train must exceed lookback+horizon+1; step >= 1")
    if not horizons or len(set(horizons)) != len(horizons) or any(h < 1 for h in horizons):
        raise ValueError("horizons must be distinct, positive integers")
    if not models or len(set(models)) != len(models) or any(m not in MODEL_NAMES for m in models):
        raise ValueError("models must be unique and supported")
    if "naive" not in models:
        raise ValueError("The naive last-price baseline is required")
    if not prices["date"].is_monotonic_increasing or prices["date"].duplicated().any():
        raise ValueError("Dates must be unique and sorted")
    close = prices["close"].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Prices must be finite and positive")
    if len(close) < min_train + max(horizons):
        raise ValueError("Not enough data for any complete evaluation window")

    lo = pd.Timestamp(eval_start, tz="UTC") if eval_start else None
    hi = pd.Timestamp(eval_end, tz="UTC") if eval_end else None
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("eval_start must not exceed eval_end")
    dates = prices["date"]
    log_prices = np.log(close)
    rows = []
    for origin in range(min_train, len(prices) - max(horizons) + 1, step):
        # Evaluation range uses last observed date, not the future target.
        forecast_at = dates.iloc[origin - 1]
        if (lo is not None and forecast_at < lo) or (hi is not None and forecast_at > hi):
            continue
        for horizon in horizons:
            actual_idx = origin - 1 + horizon
            for model in models:
                # Exact persistence baseline: exp(log(price)) rounding can
                # otherwise invent a non-flat predicted direction.
                predicted = (float(close[origin - 1]) if model == "naive"
                             else fit_predict(log_prices, origin, horizon, model))
                rows.append({
                    "origin_index": origin,
                    "train_end": forecast_at.date().isoformat(),
                    "target_date": dates.iloc[actual_idx].date().isoformat(),
                    "horizon_step": horizon,
                    "model": model,
                    "effective_model": model,
                    "fallback_used": False,
                    "last_observed_close": float(close[origin - 1]),
                    "actual": float(close[actual_idx]),
                    "predicted": predicted,
                })
    if not rows:
        raise ValueError("No origins in specified evaluation dates")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--price-col", default="Close")
    parser.add_argument("--asset-col")
    parser.add_argument("--asset")
    parser.add_argument("--aggregate-daily-last", action="store_true")
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--horizons", default="1,7")
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--models", default="naive,ridge,xgb,lightgbm")
    parser.add_argument("--eval-start", help="Optional inclusive origin date YYYY-MM-DD")
    parser.add_argument("--eval-end", help="Optional inclusive origin date YYYY-MM-DD")
    args = parser.parse_args(argv)
    horizons = tuple(int(n.strip()) for n in args.horizons.split(","))
    models = tuple(n.strip() for n in args.models.split(","))
    frame = load_price_csv(
        args.input, date_col=args.date_col, price_col=args.price_col,
        asset_col=args.asset_col, asset=args.asset,
        aggregate_daily_last=args.aggregate_daily_last,
    )
    predictions = ml_walk_forward(
        frame, min_train=args.min_train, horizons=horizons, step=args.step,
        models=models, eval_start=args.eval_start, eval_end=args.eval_end,
    )
    metrics = summarize(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "ml_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "ml_metrics.csv", index=False)
    manifest = {
        "protocol": PROTOCOL,
        "research_stage": "EXPLORATORY_ONLY_NOT_INDEPENDENT_TEST",
        "method": "expanding-window online refit, only matured labels at each origin",
        "source_path": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "n_origins": int(predictions["origin_index"].nunique()),
        "first_origin": predictions["train_end"].min(),
        "last_origin": predictions["train_end"].max(),
        "horizons": horizons,
        "models": models,
        "fixed_feature_names": FEATURE_NAMES,
        "min_train": args.min_train,
        "step": args.step,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "asset": args.asset,
        "asset_col": args.asset_col,
        "date_col": args.date_col,
        "price_col": args.price_col,
        "aggregate_daily_last": args.aggregate_daily_last,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "external_market_requests": 0,
        "test_set_hyperparameter_tuning": False,
        "limitations": "No transaction costs, trading returns, untouched holdout, uncertainty calibration, or live validation",
    }
    (args.output_dir / "ml_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {manifest['n_origins']} exploratory ML origins to {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
