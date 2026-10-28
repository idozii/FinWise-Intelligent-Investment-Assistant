"""Leakage-safe expanding-window evaluation of FinWise price forecasts.

This is research evaluation on offline historical data, not investment advice.
Importing this module never fetches market data or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAMES = ("naive", "moving_average", "linear", "holt_winters", "arima")
PROTOCOL = "finwise-walk-forward-v1"


def load_price_csv(
    path: Path,
    *,
    date_col: str = "Date",
    price_col: str = "Close",
    asset_col: str | None = None,
    asset: str | None = None,
    aggregate_daily_last: bool = False,
) -> pd.DataFrame:
    """Load a single asset as sorted daily dates and strictly positive prices."""
    data = pd.read_csv(path)
    required = {date_col, price_col}
    if asset_col:
        required.add(asset_col)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if asset is not None and not asset_col:
        raise ValueError("--asset requires --asset-col to filter a multi-asset dataset")
    if asset_col:
        if asset is not None:
            data = data[data[asset_col].astype(str).str.casefold() == asset.casefold()].copy()
        elif data[asset_col].nunique(dropna=False) > 1:
            raise ValueError("Multiple assets detected; specify --asset to prevent mixing prices")
    if data.empty:
        raise ValueError("No rows for the selected asset")

    parsed_dates = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    parsed_prices = pd.to_numeric(data[price_col], errors="coerce")
    if parsed_dates.isna().any() or parsed_prices.isna().any():
        raise ValueError("Dates and prices must be present and parseable; no rows silently dropped")
    prices = parsed_prices.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and strictly positive")

    frame = pd.DataFrame({"timestamp": parsed_dates, "close": prices}).sort_values("timestamp")
    frame["date"] = frame["timestamp"].dt.normalize()
    if aggregate_daily_last:
        frame = frame.drop_duplicates(subset="date", keep="last")
    elif frame["date"].duplicated().any():
        raise ValueError("Multiple prices on a UTC date; set --aggregate-daily-last explicitly")
    frame = frame[["date", "close"]].reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two unique daily observations are required")
    return frame


def predict_prices(train: np.ndarray, steps: int, model: str) -> tuple[np.ndarray, str]:
    """Return (point predictions, effective model), exposing fallback usage."""
    values = np.asarray(train, dtype=float)
    if values.ndim != 1 or len(values) < 2 or steps < 1:
        raise ValueError("Expected a 1-D training history (>=2) and positive horizon")
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")

    if model == "naive":
        return np.full(steps, values[-1], dtype=float), model
    if model == "moving_average":
        recent = values[-365:]
        window = min(14, max(3, len(recent) // 6))
        return np.full(steps, float(np.mean(recent[-window:])), dtype=float), model
    if model == "linear":
        recent = values[-180:]
        x = np.arange(len(recent), dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        forecast_x = np.arange(len(recent), len(recent) + steps)
        return np.asarray(slope * forecast_x + intercept, dtype=float), model

    minimum = 40 if model == "arima" else 20
    if len(values) < minimum:
        predicted, _ = predict_prices(values, steps, "linear")
        return predicted, "linear_fallback"
    try:
        if model == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            fit = ARIMA(
                pd.Series(values[-600:]), order=(5, 1, 0),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit()
            pred = np.asarray(fit.get_forecast(steps=steps).predicted_mean, dtype=float)
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            series = pd.Series(values[-730:])
            fit = ExponentialSmoothing(
                series, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = np.asarray(fit.forecast(steps), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite predictions")
        return pred, model
    except Exception:
        predicted, _ = predict_prices(values, steps, "linear")
        return predicted, "linear_fallback"


def walk_forward(
    prices: pd.DataFrame,
    *,
    min_train: int = 252,
    horizon: int = 7,
    step: int = 7,
    models: tuple[str, ...] = ("naive", "moving_average", "linear"),
) -> pd.DataFrame:
    """Evaluate every model on identical, strictly future test windows."""
    if min_train < 40 or horizon < 1 or step < 1:
        raise ValueError("min_train >= 40, horizon >= 1 and step >= 1 are required")
    if len(prices) < min_train + horizon:
        raise ValueError("Not enough observations for one complete holdout horizon")
    if not models or len(set(models)) != len(models) or any(m not in MODEL_NAMES for m in models):
        raise ValueError(f"Models must be unique and drawn from {MODEL_NAMES}")
    if "naive" not in models:
        raise ValueError("Include the naive baseline to quantify improvement")

    values = prices["close"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected finite, positive close prices")
    if not prices["date"].is_monotonic_increasing or prices["date"].duplicated().any():
        raise ValueError("Dates must be sorted with one observation per UTC day")

    rows: list[dict[str, object]] = []
    for origin in range(min_train, len(prices) - horizon + 1, step):
        train = values[:origin]
        actual = values[origin:origin + horizon]
        last_observed = float(train[-1])
        for model in models:
            predicted, effective_model = predict_prices(train, horizon, model)
            for k in range(horizon):
                rows.append({
                    "origin_index": origin,
                    "train_end": prices.iloc[origin - 1]["date"].date().isoformat(),
                    "target_date": prices.iloc[origin + k]["date"].date().isoformat(),
                    "horizon_step": k + 1,
                    "model": model,
                    "effective_model": effective_model,
                    "fallback_used": effective_model != model,
                    "last_observed_close": last_observed,
                    "actual": float(actual[k]),
                    "predicted": float(predicted[k]),
                })
    return pd.DataFrame.from_records(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Horizon-specific errors; directional accuracy excludes flat predictions."""
    naive = predictions.loc[predictions["model"] == "naive"].set_index(
        ["origin_index", "horizon_step"]
    )
    rows = []
    for (model, horizon), group in predictions.groupby(["model", "horizon_step"], sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        reference = group["last_observed_close"].to_numpy(dtype=float)
        errors = actual - predicted
        paired_naive = naive.loc[list(zip(group["origin_index"], group["horizon_step"]))]
        naive_mae = float(np.mean(np.abs(
            paired_naive["actual"].to_numpy() - paired_naive["predicted"].to_numpy()
        )))
        mae = float(np.mean(np.abs(errors)))
        predicted_direction = np.sign(predicted - reference)
        actual_direction = np.sign(actual - reference)
        eligible = (predicted_direction != 0) & (actual_direction != 0)
        rows.append({
            "model": model,
            "horizon_step": int(horizon),
            "n_origins": len(group),
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae_improvement_vs_naive_pct": (100.0 * (1 - mae / naive_mae) if naive_mae > 0 else float("nan")),
            "directional_accuracy_nonflat": (
                float(np.mean(predicted_direction[eligible] == actual_direction[eligible]))
                if eligible.any() else float("nan")
            ),
            "n_directional_evaluable": int(eligible.sum()),
            "fallback_rate": float(group["fallback_used"].mean()),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Offline processed CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--price-col", default="Close")
    parser.add_argument("--asset-col", default=None)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--aggregate-daily-last", action="store_true")
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--step", type=int, default=7)
    parser.add_argument("--models", default="naive,moving_average,linear")
    args = parser.parse_args(argv)

    models = tuple(m.strip() for m in args.models.split(","))
    prices = load_price_csv(
        args.input, date_col=args.date_col, price_col=args.price_col,
        asset_col=args.asset_col, asset=args.asset,
        aggregate_daily_last=args.aggregate_daily_last,
    )
    predictions = walk_forward(
        prices, min_train=args.min_train, horizon=args.horizon,
        step=args.step, models=models,
    )
    metrics = summarize(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    manifest = {
        "protocol": PROTOCOL,
        "source_path": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_rows_after_daily_selection": len(prices),
        "first_date": prices.iloc[0]["date"].date().isoformat(),
        "last_date": prices.iloc[-1]["date"].date().isoformat(),
        "n_origins": int(predictions["origin_index"].nunique()),
        "min_train": args.min_train,
        "horizon": args.horizon,
        "step": args.step,
        "models": models,
        "asset": args.asset,
        "asset_col": args.asset_col,
        "date_col": args.date_col,
        "price_col": args.price_col,
        "aggregate_daily_last": args.aggregate_daily_last,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "external_market_requests": 0,
        "model_selection": "fixed before evaluation; no test-set tuning",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {manifest['n_origins']} evaluation origins to {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
