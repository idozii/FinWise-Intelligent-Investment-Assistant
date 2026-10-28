from pathlib import Path

import pandas as pd
import pytest

from scripts.m5_download import _validate_price_frame, download, save_one


def prices(n=300):
    dates = pd.date_range("2025-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"Close": [100.0 + j / 10 for j in range(n)],
                         "Volume": [123] * n}, index=pd.Index(dates, name="Date"))


def test_download_writes_expected_schema_once_without_overwrite(tmp_path):
    calls = []
    def fetch(ticker):
        calls.append(ticker)
        return prices()
    assert download(tmp_path, fetch, ("GLD",)) == 0
    target = tmp_path / "GLD_daily_processed.csv"
    assert target.exists()
    rows = pd.read_csv(target)
    assert {"Date", "Close", "Volume"}.issubset(rows.columns)
    first = target.read_bytes()
    assert download(tmp_path, fetch, ("GLD",)) == 0
    assert calls == ["GLD"]
    assert target.read_bytes() == first


def test_429_halts_without_touching_other_tickers(tmp_path):
    class YFRateLimitError(Exception):
        pass
    calls = []
    def fetch(ticker):
        calls.append(ticker)
        if ticker == "IWM":
            raise YFRateLimitError("Too Many Requests")
        return prices()
    assert download(tmp_path, fetch, ("GLD", "IWM", "XLE")) == 2
    assert calls == ["GLD", "IWM"]
    assert (tmp_path / "GLD_daily_processed.csv").exists()
    assert not (tmp_path / "IWM_daily_processed.csv").exists()
    assert not (tmp_path / "XLE_daily_processed.csv").exists()
    assert not list(tmp_path.glob("*.part.csv"))


def test_resume_after_rate_limit_skips_completed_assets(tmp_path):
    calls = []
    class YFRateLimitError(Exception):
        pass
    def first_fetch(ticker):
        calls.append(ticker)
        if ticker == "IWM":
            raise YFRateLimitError()
        return prices()
    assert download(tmp_path, first_fetch, ("GLD", "IWM")) == 2
    calls.clear()
    assert download(tmp_path, lambda ticker: calls.append(ticker) or prices(),
                    ("GLD", "IWM")) == 0
    assert calls == ["IWM"]


def test_invalid_price_data_never_creates_snapshot(tmp_path):
    bad = prices()
    bad.iloc[0, bad.columns.get_loc("Close")] = -1.0
    with pytest.raises(ValueError, match="finite and positive"):
        save_one("XLV", tmp_path, lambda _: bad)
    assert not (tmp_path / "XLV_daily_processed.csv").exists()
    assert not list(tmp_path.glob("*.part.csv"))


def test_duplicate_dates_rejected(tmp_path):
    df = prices(5)
    df.index = pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-04", "2025-01-05"
    ], utc=True)
    df.index.name = "Date"
    with pytest.raises(ValueError, match="duplicate"):
        _validate_price_frame(df, "IWM")


def test_no_unfrozen_assets_allowed(tmp_path):
    with pytest.raises(ValueError, match="Only frozen"):
        save_one("AAPL", tmp_path, lambda _: prices())


def test_empty_data_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        save_one("XLE", tmp_path, lambda _: pd.DataFrame())


def test_unexpected_network_exception_is_not_silently_ignored(tmp_path):
    def broken(_):
        raise ConnectionError("offline")
    with pytest.raises(RuntimeError, match="ConnectionError"):
        save_one("GLD", tmp_path, broken)
