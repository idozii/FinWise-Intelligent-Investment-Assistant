"""M5-v2 Tiingo raw Close acquisition (no model scoring).

Provider-only protocol amendment for blocked Yahoo-v1 collection.
Uses a Tiingo API token supplied ONLY in environment variable. No API key
is written to files, URLs, logs or Git. Does not bypass rate limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from scripts.m5_download import _validate_price_frame, RESERVED_ASSETS

PROVIDER = "tiingo_eod_raw_close"
START_DATE = "2021-10-27"
END_DATE = "2026-10-27"
OUTPUT_DIR = Path("data/processed/m5_tiingo_v2")


def get_tiingo_history(ticker: str, token: str) -> pd.DataFrame:
    if ticker not in RESERVED_ASSETS:
        raise ValueError("Unexpected ticker; M5 frozen assets only")
    if not token.strip():
        raise ValueError("TIINGO_API_KEY must be set locally")
    url = (f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?"
           + urlencode({"startDate": START_DATE, "endDate": END_DATE}))
    request = Request(url, headers={
        "Authorization": f"Token {token}",
        "Accept": "application/json",
        "User-Agent": "FinWise-Research-M5-v2/1.0",
    })
    try:
        with urlopen(request, timeout=25) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        if exc.code == 429:
            raise RateLimitError("Tiingo rate limit; no automatic retries") from None
        if exc.code in (401, 403):
            raise PermissionError(
                "Tiingo authentication or plan access denied; check token and account entitlements"
            ) from None
        raise RuntimeError(f"Tiingo HTTP {exc.code}") from None
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Tiingo connection failed: {type(exc).__name__}") from None
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{ticker}: Tiingo returned no daily bar records")
    records = []
    for bar in payload:
        if not isinstance(bar, dict) or "date" not in bar or "close" not in bar:
            raise ValueError(f"{ticker}: missing raw date/close in Tiingo bar")
        # Explicitly DO NOT use 'adjClose'; avoid changing target definition.
        records.append({"Date": bar["date"], "Close": bar["close"]})
    frame = pd.DataFrame.from_records(records)
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True, errors="raise")
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    if (frame.Date < pd.Timestamp(START_DATE, tz="UTC")).any() or (
        frame.Date > pd.Timestamp(END_DATE, tz="UTC")
    ).any():
        raise ValueError(f"{ticker}: provider returned dates outside frozen request window")
    return frame


class RateLimitError(Exception):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".source.json")


def verify_snapshot(ticker: str, path: Path) -> None:
    """Enforce source sidecar and byte-exact snapshot provenance."""
    meta_path = _source_path(path)
    if not path.is_file() or not meta_path.is_file():
        raise ValueError(f"{ticker}: incomplete Tiingo snapshot/source manifest")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "asset": ticker, "provider": PROVIDER, "price_field": "close",
        "start_date": START_DATE, "end_date": END_DATE,
        "source_sha256": _sha256(path),
    }
    if any(meta.get(k) != v for k, v in required.items()):
        raise ValueError(f"{ticker}: source provenance mismatch or CSV modified")


def save_snapshot(ticker: str, output_dir: Path, frame: pd.DataFrame) -> str:
    """Validate before writing; never replace either artifact."""
    if ticker not in RESERVED_ASSETS:
        raise ValueError("Unexpected ticker")
    path = output_dir / f"{ticker}_daily_processed.csv"
    sidecar = _source_path(path)
    if path.exists() or sidecar.exists():
        verify_snapshot(ticker, path)
        return f"{ticker}: existing snapshot verified (not replaced)"
    # _validate_price_frame accepts a frame with an existing Date column.
    prepared = _validate_price_frame(frame, ticker)
    if len(prepared) < 300:
        raise ValueError(f"{ticker}: insufficient daily history for M5")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{ticker}_", dir=output_dir) as tmp:
        temp_dir = Path(tmp)
        staged = temp_dir / path.name
        staged_meta = temp_dir / sidecar.name
        prepared.to_csv(staged, index=False)
        manifest = {
            "asset": ticker, "provider": PROVIDER,
            "price_field": "close", "adjusted_close_used": False,
            "start_date": START_DATE, "end_date": END_DATE,
            "source_sha256": _sha256(staged),
            "n_rows": len(prepared),
            "collection_note": "Tiingo API token from environment variable; never persisted",
            "vendor_price_difference_note": "Not interchangeable with Yahoo raw Close without provenance caveats",
        }
        staged_meta.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # On local Linux filesystems, hard-link creation is atomic and no-overwrite.
        try:
            os.link(staged, path)
        except FileExistsError:
            verify_snapshot(ticker, path)
            return f"{ticker}: existing snapshot verified (not replaced)"
        try:
            os.link(staged_meta, sidecar)
        except FileExistsError:
            # Rare concurrent writer. Do not overwrite its manifest.
            verify_snapshot(ticker, path)
    verify_snapshot(ticker, path)
    return f"{ticker}: saved {len(prepared)} rows; sha256={_sha256(path)}"


def download(
    output_dir: Path, fetch, tickers: tuple[str, ...] = RESERVED_ASSETS
) -> int:
    for ticker in tickers:
        file = output_dir / f"{ticker}_daily_processed.csv"
        if file.exists():
            verify_snapshot(ticker, file)
            print(f"{ticker}: verified cached Tiingo CSV; skipped")
            continue
        if _source_path(file).exists():
            raise ValueError(f"{ticker}: orphan source manifest; manual review needed")
        try:
            frame = fetch(ticker)
        except RateLimitError:
            print(f"{ticker}: Tiingo 429; stop without requesting remaining assets")
            return 2
        print(save_snapshot(ticker, output_dir, frame))
    print("Tiingo M5-v2 data collected. Next step is DATE-ONLY preflight, not evaluation.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--only", nargs="+", choices=RESERVED_ASSETS)
    args = parser.parse_args(argv)
    token = os.environ.get("TIINGO_API_KEY", "")
    if not token.strip():
        raise SystemExit(
            "Set TIINGO_API_KEY in your local shell; never paste the token into chat or Git"
        )
    selected = tuple(dict.fromkeys(args.only)) if args.only else RESERVED_ASSETS
    return download(args.output_dir, lambda asset: get_tiingo_history(asset, token), selected)


if __name__ == "__main__":
    raise SystemExit(main())
