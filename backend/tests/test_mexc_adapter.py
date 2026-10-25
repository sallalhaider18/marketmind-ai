from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.adapters import mexc_futures
from app.adapters.mexc_futures import MexcApiError, MexcFuturesAdapter, parse_kline_response, to_venue_symbol


@pytest.mark.parametrize(
    "inp,out",
    [("BTCUSDT.P", "BTC_USDT"), ("ethusdt.p", "ETH_USDT"), ("BTCUSDT", "BTC_USDT"), ("BTC_USDT", "BTC_USDT")],
)
def test_symbol_mapping(inp, out):
    assert to_venue_symbol(inp) == out


def test_symbol_mapping_rejects_garbage():
    with pytest.raises(ValueError):
        to_venue_symbol("FOO")


def test_parse_kline_converts_contracts_to_base_volume():
    payload = {
        "success": True, "code": 0,
        "data": {"time": [1700000000, 1700000060], "open": [1, 2], "high": [3, 4], "low": [0.5, 1.5],
                 "close": [2, 3], "vol": [1000, 2000], "amount": [10, 20]},
    }
    df = parse_kline_response(payload, contract_size=0.0001)
    assert list(df["volume"]) == pytest.approx([0.1, 0.2])
    assert list(df["contracts"]) == [1000, 2000]
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_parse_kline_raises_on_api_error():
    with pytest.raises(MexcApiError):
        parse_kline_response({"success": False, "code": 1001, "message": "bad"}, 1.0)


def _fake_mexc(calls: list):
    """MockTransport emulating MEXC: returns every 1m candle inside [start, end]."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/contract/detail":
            return httpx.Response(200, json={"success": True, "code": 0, "data": {
                "symbol": "BTC_USDT", "contractSize": 0.0001, "priceUnit": 0.1, "minVol": 1, "volUnit": 1}})
        start = int(request.url.params["start"])
        end = int(request.url.params["end"])
        times = list(range(start, end + 1, 60))[:2000]
        n = len(times)
        return httpx.Response(200, json={"success": True, "code": 0, "data": {
            "time": times, "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
            "close": [100.5] * n, "vol": [10.0] * n, "amount": [1.0] * n}})

    return handler


def test_fetch_candles_pages_and_returns_exact_range(monkeypatch):
    monkeypatch.setattr(mexc_futures, "MIN_SECONDS_BETWEEN_REQUESTS", 0)
    calls = []
    client = httpx.Client(base_url="https://x", transport=httpx.MockTransport(_fake_mexc(calls)))
    ad = MexcFuturesAdapter(client=client)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5000)  # needs 3 kline pages
    df = ad.fetch_candles("BTCUSDT.P", "1m", start, end)

    assert len(df) == 5000
    assert df["timestamp"].iloc[0] == start
    assert df["timestamp"].iloc[-1] == end - timedelta(minutes=1)
    assert df["timestamp"].is_unique
    kline_calls = [c for c in calls if "kline" in c.url.path]
    assert len(kline_calls) == 3
    assert kline_calls[0].url.path == "/api/v1/contract/kline/BTC_USDT"
    assert kline_calls[0].url.params["interval"] == "Min1"


def test_retries_on_server_error(monkeypatch):
    monkeypatch.setattr(mexc_futures, "MIN_SECONDS_BETWEEN_REQUESTS", 0)
    monkeypatch.setattr(mexc_futures.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"success": True, "data": {
            "contractSize": 0.01, "priceUnit": 0.01, "minVol": 1, "volUnit": 1}})

    ad = MexcFuturesAdapter(client=httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler)))
    assert ad.get_contract_spec("ETHUSDT.P").contract_size == 0.01
    assert attempts["n"] == 3
