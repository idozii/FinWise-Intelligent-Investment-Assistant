import json
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import pytest

from scripts.m5_download_tiingo import (
    END_DATE, START_DATE, RateLimitError, _source_path, download,
    get_tiingo_history, save_snapshot, verify_snapshot,
)
from research.frozen_eval_tiingo import (
    LOCKED_V2_SHA256, SPEC_PATH, load_v2_protocol,
)
from research.frozen_eval import LOCKED_PROTOCOL_SHA256, load_locked_protocol


def prices(n=330):
    dates = pd.date_range("2025-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"Date": dates, "Close": [100 + k / 100 for k in range(n)]})


def test_v1_and_v2_protocol_locks_are_separate(tmp_path):
    assert len(LOCKED_V2_SHA256) == 64
    assert LOCKED_V2_SHA256 != LOCKED_PROTOCOL_SHA256
    assert load_locked_protocol()["dataset"]["vendor"] == "Yahoo Finance via yfinance"
    spec = load_v2_protocol()
    assert "Tiingo" in spec["dataset"]["vendor"]
    assert set(spec["reserved_external_assets"]) == {"GLD", "IWM", "XLE", "XLV"}
    assert spec["models"] == ["naive", "ridge"]
    tampered = tmp_path / "altered.json"
    tampered.write_text(SPEC_PATH.read_text().replace("Tiingo", "Yahoo"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_v2_protocol(tampered)


def test_downloader_saves_csv_and_source_manifest(tmp_path):
    saved = save_snapshot("GLD", tmp_path, prices())
    assert "saved 330 rows" in saved
    path = tmp_path / "GLD_daily_processed.csv"
    verify_snapshot("GLD", path)
    data = pd.read_csv(path)
    assert {"Date", "Close"} <= set(data.columns)
    assert "Adj Close" not in data.columns
    source = json.loads(_source_path(path).read_text())
    assert source["price_field"] == "close"
    assert source["adjusted_close_used"] is False
    first = path.read_bytes()
    assert "existing snapshot verified" in save_snapshot("GLD", tmp_path, prices())
    assert path.read_bytes() == first


def test_modified_csv_fails_provenance(tmp_path):
    save_snapshot("IWM", tmp_path, prices())
    path = tmp_path / "IWM_daily_processed.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "Close"] = 999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="provenance mismatch"):
        verify_snapshot("IWM", path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        save_snapshot("IWM", tmp_path, prices())


def test_429_stops_and_keeps_prior_ticker(tmp_path):
    seen = []
    def fetch(asset):
        seen.append(asset)
        if asset == "IWM":
            raise RateLimitError("429")
        return prices()
    assert download(tmp_path, fetch, ("GLD", "IWM", "XLE")) == 2
    assert seen == ["GLD", "IWM"]
    assert not (tmp_path / "XLE_daily_processed.csv").exists()
    assert (tmp_path / "GLD_daily_processed.csv").exists()


def test_history_uses_raw_close_and_auth_header_not_url(monkeypatch):
    import scripts.m5_download_tiingo as downloader

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps([
                {"date": "2026-10-26T00:00:00Z", "close": 130.0, "adjClose": 999.0},
                {"date": "2026-10-27T00:00:00Z", "close": 140.0, "adjClose": 999.0},
            ]).encode()
    def fake_urlopen(request, timeout=25):
        assert "SECRET_TOKEN_TEST" not in request.full_url
        assert request.get_header("Authorization") == "Token SECRET_TOKEN_TEST"
        assert START_DATE in request.full_url
        assert END_DATE in request.full_url
        return FakeResponse()

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)
    frame = get_tiingo_history("XLE", "SECRET_TOKEN_TEST")
    assert frame.Close.tolist() == [130., 140.]
    assert "adjClose" not in frame.columns


def test_http_429_is_not_retried(monkeypatch):
    import scripts.m5_download_tiingo as downloader
    n = []
    def fail(request, timeout):
        n.append(1)
        raise HTTPError(request.full_url, 429, "Too Many Requests", None, None)
    monkeypatch.setattr(downloader, "urlopen", fail)
    with pytest.raises(RateLimitError):
        get_tiingo_history("GLD", "SECRET_TOKEN_TEST")
    assert len(n) == 1


def test_reject_insufficient_history_or_unfrozen_asset(tmp_path):
    with pytest.raises(ValueError, match="insufficient"):
        save_snapshot("XLV", tmp_path, prices(20))
    with pytest.raises(ValueError, match="Unexpected ticker"):
        save_snapshot("AAPL", tmp_path, prices())


def test_new_protocol_hash_flows_into_preflight(tmp_path):
    from research.frozen_eval import preflight
    spec = load_v2_protocol()
    tiny = dict(spec)
    tiny["reserved_external_assets"] = ["GLD"]
    tiny["forecast_origin_start"] = "2025-04-01"
    tiny["forecast_origin_end"] = "2025-11-20"
    tiny["min_train"] = 80
    path = tmp_path / "GLD_daily_processed.csv"
    prices().to_csv(path, index=False)
    result = preflight(tiny, [("GLD", path)], tmp_path / "preflight",
                       protocol_sha256=LOCKED_V2_SHA256)
    assert result["protocol_sha256"] == LOCKED_V2_SHA256
    assert result["sources"][0]["n_eligible_origins"] >= 30
