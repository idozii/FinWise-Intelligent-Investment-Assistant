"""Download the locked M5 Yahoo Finance price snapshots without retrying 429s.

This only collects dataset inputs; it never scores forecasts or reads
holdout outcomes for model selection. It respects Yahoo rate limits rather
than bypassing them. It never overwrites an existing asset CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import pandas as pd

RESERVED_ASSETS = ("GLD", "IWM", "XLE", "XLV")


def _validate_price_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Refuse to publish empty, incomplete or nonchronological datasets."""
    if raw.empty:
        raise ValueError(f"{ticker}: Yahoo Finance returned an empty dataset")
    data = raw.reset_index()
    if not {"Date", "Close"}.issubset(data.columns):
        raise ValueError(f"{ticker}: expected Date and Close columns")
    data["Date"] = pd.to_datetime(data["Date"], errors="raise", utc=True)
    if data["Date"].isna().any():
        raise ValueError(f"{ticker}: missing Date")
    if not data["Date"].is_monotonic_increasing or data["Date"].dt.normalize().duplicated().any():
        raise ValueError(f"{ticker}: duplicate or unsorted daily dates")
    close = pd.to_numeric(data["Close"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"{ticker}: Close must be finite and positive")
    return data


def save_one(
    ticker: str,
    output_dir: Path,
    fetch: Callable[[str], pd.DataFrame],
) -> tuple[str, str]:
    """Return status and details; never overwrite prior CSV or partial data."""
    if ticker not in RESERVED_ASSETS:
        raise ValueError("Only frozen M5 assets are permitted")
    target = output_dir / f"{ticker}_daily_processed.csv"
    if target.exists():
        return "exists", f"{ticker}: exists; preserved, no request made"
    try:
        raw = fetch(ticker)
    except Exception as exc:
        # yfinance versions differ in how they expose its exception class.
        # Stop on the public rate-limit signal, without automatic retry loops.
        if type(exc).__name__ == "YFRateLimitError":
            return "rate_limit", f"{ticker}: Yahoo Finance rate limit (429); stopped without further requests"
        raise RuntimeError(f"{ticker}: Yahoo Finance fetch failed: {type(exc).__name__}: {exc}") from exc
    data = _validate_price_frame(raw, ticker)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{ticker}_", suffix=".part.csv", dir=output_dir,
            delete=False, mode="w", encoding="utf-8", newline="",
        ) as file:
            temporary = Path(file.name)
            data.to_csv(file, index=False)
        # Atomic link never overwrites a file written by another process.
        try:
            os.link(temporary, target)
        except FileExistsError:
            return "exists", f"{ticker}: appeared during download; existing snapshot preserved"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return "saved", f"{ticker}: saved {len(data)} rows; sha256={digest}"


def download(
    output_dir: Path,
    fetch: Callable[[str], pd.DataFrame],
    tickers: tuple[str, ...] = RESERVED_ASSETS,
) -> int:
    """Run only missing assets; halt the entire batch at the first 429."""
    for ticker in tickers:
        status, message = save_one(ticker, output_dir, fetch)
        print(message)
        if status == "rate_limit":
            print("No further Yahoo requests attempted. Keep existing CSVs unchanged.")
            return 2
    print("All requested files are present; run M5 preflight separately.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--only", nargs="+", choices=RESERVED_ASSETS,
        help="Optional reserved tickers to fetch, without changing the frozen asset set",
    )
    args = parser.parse_args(argv)
    chosen = tuple(dict.fromkeys(args.only)) if args.only else RESERVED_ASSETS
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance from backend/requirements.txt") from exc
    return download(
        args.processed_dir,
        lambda symbol: yf.Ticker(symbol).history(period="5y", auto_adjust=False),
        chosen,
    )


if __name__ == "__main__":
    raise SystemExit(main())
