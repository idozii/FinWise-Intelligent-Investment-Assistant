from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pycoingecko import CoinGeckoAPI
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

app = FastAPI(title="FinWise API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("FINWISE_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
DATA_DIR = DATA_ROOT / "processed"
FRONTEND_DIR = REPO_ROOT / "frontend"
_SYMBOL_CACHE: dict[str, pd.DataFrame] = {}
_CRYPTO_CACHE: pd.DataFrame | None = None
COINGECKO_PUBLIC_MAX_DAYS = 365

SUPPORTED_STOCKS = (
    "AAPL",
    "AMZN",
    "AMD",
    "BAC",
    "GOOGL",
    "JPM",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "SPY",
    "TSLA",
)

SUPPORTED_COINS = (
    "avalanche-2",
    "binancecoin",
    "bitcoin",
    "cardano",
    "chainlink",
    "dogecoin",
    "ethereum",
    "polkadot",
    "ripple",
    "solana",
    "sui",
    "tether",
    "tron",
    "usd-coin",
)

FORECAST_MODELS = {
    "arima": {"label": "ARIMA", "description": "Autoregressive time-series forecast"},
    "holt_winters": {"label": "Holt-Winters", "description": "Exponential smoothing with trend"},
    "linear": {"label": "Linear Trend", "description": "Simple linear trend projection"},
    "moving_average": {"label": "Moving Average", "description": "Rolling average baseline"},
}


def _stock_file_map() -> dict[str, Path]:
    if not DATA_DIR.exists():
        return {}

    files: dict[str, Path] = {}
    for path in DATA_DIR.glob("*_daily_processed.csv"):
        symbol = path.name.replace("_daily_processed.csv", "").upper()
        files[symbol] = path
    return files


def _fetch_stock_data(symbol: str) -> pd.DataFrame:
    try:
        history = yf.Ticker(symbol).history(period="5y", auto_adjust=False)
    except YFRateLimitError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upstream market data provider rate-limited requests. Try again shortly.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch market data for symbol: {symbol}",
        ) from exc

    if history.empty:
        raise HTTPException(status_code=404, detail=f"No market data available for symbol: {symbol}")

    history = history.reset_index()
    if "Date" not in history.columns and "index" in history.columns:
        history = history.rename(columns={"index": "Date"})

    history["Date"] = pd.to_datetime(history["Date"], errors="coerce", utc=True)
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history = history.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return history


def _crypto_file() -> Path:
    path = DATA_DIR / "crypto_prices_processed.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Processed crypto dataset not found")
    return path


def _fetch_crypto_data(coin: str) -> pd.DataFrame:
    client = CoinGeckoAPI()
    try:
        payload = client.get_coin_market_chart_by_id(
            id=coin,
            vs_currency="usd",
            days=COINGECKO_PUBLIC_MAX_DAYS,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"CoinGecko could not return data for {coin}. "
                f"Public API access is limited to the last {COINGECKO_PUBLIC_MAX_DAYS} days."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Upstream crypto data provider is temporarily unavailable. Try again shortly.",
        ) from exc

    prices = payload.get("prices", [])
    if not prices:
        raise HTTPException(status_code=404, detail=f"No market data available for coin: {coin}")

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["coin"] = coin.lower()
    return df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)


def _load_symbol_data(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()
    if symbol in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[symbol]

    files = _stock_file_map()
    if symbol in files:
        df = pd.read_csv(files[symbol])
        if "Date" not in df.columns or "Close" not in df.columns:
            raise HTTPException(status_code=500, detail=f"Malformed dataset for symbol: {symbol}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    elif symbol in SUPPORTED_STOCKS:
        df = _fetch_stock_data(symbol)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")

    _SYMBOL_CACHE[symbol] = df
    return df


def _load_crypto_data() -> pd.DataFrame:
    global _CRYPTO_CACHE
    if _CRYPTO_CACHE is not None:
        return _CRYPTO_CACHE

    df = pd.read_csv(_crypto_file())
    if not {"timestamp", "price", "coin"}.issubset(df.columns):
        raise HTTPException(status_code=500, detail="Malformed crypto dataset")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["coin"] = df["coin"].astype(str)
    df = df.dropna(subset=["timestamp", "price", "coin"]).sort_values("timestamp").reset_index(drop=True)

    _CRYPTO_CACHE = df
    return df


def _load_coin_data(coin: str) -> pd.DataFrame:
    try:
        df = _load_crypto_data()
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        df = pd.DataFrame(columns=["timestamp", "price", "coin"])

    coin_key = coin.lower()
    coin_df = df[df["coin"].str.lower() == coin_key].copy()
    if not coin_df.empty:
        return coin_df

    if coin_key in SUPPORTED_COINS:
        return _fetch_crypto_data(coin_key)

    raise HTTPException(status_code=404, detail=f"Unknown coin: {coin}")


def _forecast_interval(predictions: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual_std = float(np.nanstd(residuals)) if residuals.size else 0.0
    margin = 1.96 * residual_std
    lower = np.maximum(predictions - margin, 0)
    upper = predictions + margin
    return lower, upper


def _forecast_linear_fallback(close_series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fallback when ARIMA is unstable: linear trend + empirical residual band.
    train = close_series.tail(min(180, len(close_series))).astype(float).to_numpy()

    x = np.arange(train.shape[0])
    slope, intercept = np.polyfit(x, train, 1)

    future_x = np.arange(train.shape[0], train.shape[0] + days)
    predictions = slope * future_x + intercept

    fitted = slope * x + intercept
    residual_std = float(np.std(train - fitted)) if train.shape[0] > 1 else 0.0
    margin = 1.96 * residual_std

    lower = np.maximum(predictions - margin, 0)
    upper = predictions + margin

    return predictions, lower, upper


def _forecast_linear(close_series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _forecast_linear_fallback(close_series, days)


def _forecast_moving_average(close_series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = close_series.dropna().astype(float).tail(min(365, len(close_series)))
    window = min(14, max(3, train.shape[0] // 6))
    rolling = train.rolling(window=window).mean()
    baseline = float(rolling.dropna().iloc[-1]) if not rolling.dropna().empty else float(train.iloc[-1])
    predictions = np.full(days, baseline, dtype=float)

    residuals = (train - rolling).dropna().to_numpy(dtype=float)
    lower, upper = _forecast_interval(predictions, residuals)
    return predictions, lower, upper


def _forecast_holt_winters(close_series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = close_series.dropna().astype(float).tail(min(730, len(close_series)))
    if train.shape[0] < 20:
        return _forecast_linear_fallback(train, days)

    try:
        model = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated")
        fitted = model.fit(optimized=True)
        predictions = np.asarray(fitted.forecast(days), dtype=float)
        residuals = np.asarray(train - fitted.fittedvalues, dtype=float)
        lower, upper = _forecast_interval(predictions, residuals)
        return predictions, lower, upper
    except Exception:
        return _forecast_linear_fallback(train, days)


def _forecast_arima(close_series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = close_series.dropna().astype(float).tail(min(600, len(close_series)))
    if train.shape[0] < 40:
        return _forecast_linear_fallback(train, days)

    try:
        model = ARIMA(train, order=(5, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit()
        forecast = fitted.get_forecast(steps=days)

        pred = np.asarray(forecast.predicted_mean, dtype=float)
        conf = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
        lower = np.maximum(conf[:, 0], 0)
        upper = conf[:, 1]
        return pred, lower, upper
    except Exception:
        return _forecast_linear_fallback(train, days)


def _run_forecast(close_series: pd.Series, days: int, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    normalized = model_name.lower()
    if normalized not in FORECAST_MODELS:
        supported = ", ".join(sorted(FORECAST_MODELS.keys()))
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'. Supported models: {supported}")

    if normalized == "arima":
        pred, lower, upper = _forecast_arima(close_series, days)
    elif normalized == "holt_winters":
        pred, lower, upper = _forecast_holt_winters(close_series, days)
    elif normalized == "moving_average":
        pred, lower, upper = _forecast_moving_average(close_series, days)
    else:
        pred, lower, upper = _forecast_linear(close_series, days)

    return pred, lower, upper, FORECAST_MODELS[normalized]["label"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "dataDir": str(DATA_DIR)}


@app.get("/api/forecast-models")
def list_forecast_models() -> dict[str, list[dict[str, str]]]:
    return {
        "models": [
            {"id": model_id, **meta}
            for model_id, meta in FORECAST_MODELS.items()
        ]
    }


@app.get("/api/stocks")
def list_stocks() -> dict[str, list[str]]:
    symbols = sorted(set(_stock_file_map().keys()) | set(SUPPORTED_STOCKS))
    return {"symbols": symbols}


@app.get("/api/crypto/coins")
def list_coins() -> dict[str, list[str]]:
    try:
        df = _load_crypto_data()
        dataset_coins = set(str(coin) for coin in df["coin"].unique().tolist())
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        dataset_coins = set()

    coins = sorted(dataset_coins | set(SUPPORTED_COINS))
    return {"coins": coins}


@app.get("/api/stocks/{symbol}/history")
def stock_history(
    symbol: str,
    days: int = Query(default=365, ge=30, le=5000),
) -> dict[str, object]:
    df = _load_symbol_data(symbol)
    recent = df.tail(days)

    rows: list[dict[str, object]] = []
    for _, row in recent.iterrows():
        rows.append(
            {
                "date": row["Date"].isoformat(),
                "close": float(row["Close"]),
                "open": float(row["Open"]) if "Open" in recent.columns else None,
                "high": float(row["High"]) if "High" in recent.columns else None,
                "low": float(row["Low"]) if "Low" in recent.columns else None,
                "volume": int(row["Volume"]) if "Volume" in recent.columns and not pd.isna(row["Volume"]) else None,
            }
        )

    latest_close = float(recent.iloc[-1]["Close"])
    previous_close = float(recent.iloc[-2]["Close"]) if len(recent) > 1 else latest_close
    pct_change = ((latest_close - previous_close) / previous_close * 100) if previous_close != 0 else 0.0

    return {
        "symbol": symbol.upper(),
        "history": rows,
        "latestClose": latest_close,
        "dailyChangePct": pct_change,
    }


@app.get("/api/stocks/{symbol}/forecast")
def stock_forecast(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30),
    model: str = Query(default="arima"),
) -> dict[str, object]:
    df = _load_symbol_data(symbol)
    close = df["Close"]

    if len(close) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough data to forecast for {symbol}")

    pred, lower, upper, model_label = _run_forecast(close, days, model)

    last_date = df.iloc[-1]["Date"]
    forecast_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)

    forecast_rows: list[dict[str, object]] = []
    for i, dt in enumerate(forecast_dates):
        forecast_rows.append(
            {
                "date": dt.isoformat(),
                "predictedClose": float(pred[i]),
                "lower95": float(lower[i]),
                "upper95": float(upper[i]),
            }
        )

    return {
        "symbol": symbol.upper(),
        "horizonDays": days,
        "model": model_label,
        "modelId": model.lower(),
        "forecast": forecast_rows,
    }


@app.get("/api/crypto/{coin}/history")
def crypto_history(
    coin: str,
    days: int = Query(default=365, ge=30, le=5000),
) -> dict[str, object]:
    coin_df = _load_coin_data(coin)

    recent = coin_df.tail(days)
    rows: list[dict[str, object]] = []
    for _, row in recent.iterrows():
        rows.append(
            {
                "date": row["timestamp"].isoformat(),
                "close": float(row["price"]),
            }
        )

    latest_close = float(recent.iloc[-1]["price"])
    previous_close = float(recent.iloc[-2]["price"]) if len(recent) > 1 else latest_close
    pct_change = ((latest_close - previous_close) / previous_close * 100) if previous_close != 0 else 0.0

    return {
        "coin": coin.lower(),
        "history": rows,
        "latestClose": latest_close,
        "dailyChangePct": pct_change,
    }


@app.get("/api/crypto/{coin}/forecast")
def crypto_forecast(
    coin: str,
    days: int = Query(default=7, ge=1, le=30),
    model: str = Query(default="arima"),
) -> dict[str, object]:
    coin_df = _load_coin_data(coin)

    price = coin_df["price"]
    if len(price) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough data to forecast for {coin}")

    pred, lower, upper, model_label = _run_forecast(price, days, model)

    last_date = coin_df.iloc[-1]["timestamp"]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="D")

    forecast_rows: list[dict[str, object]] = []
    for i, dt in enumerate(forecast_dates):
        forecast_rows.append(
            {
                "date": dt.isoformat(),
                "predictedClose": float(pred[i]),
                "lower95": float(lower[i]),
                "upper95": float(upper[i]),
            }
        )

    return {
        "coin": coin.lower(),
        "horizonDays": days,
        "model": model_label,
        "modelId": model.lower(),
        "forecast": forecast_rows,
    }


# In production, serve the frontend from the same FastAPI origin to avoid API URL confusion.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
