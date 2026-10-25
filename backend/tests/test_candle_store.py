import json
from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.adapters.base import ContractSpec
from app.data.candle_store import CandleStore, resample_candles, to_chart_bars
from app.data.historical import candle_path
from app.data.live_stream import LiveMarketState, MexcFuturesStream
from app.data.validation import validate_candles
from tests.test_validation_and_history import T0, FakeAdapter, make_candles


class SpecAdapter(FakeAdapter):
    venue = "mexc_futures"

    def get_contract_spec(self, s):
        return ContractSpec(s, s, 0.0001, 0.1, 1, 1)


def test_resample_15m_aggregates_correctly():
    df = make_candles(30)
    df["high"] = range(101, 131)          # rising highs
    df["volume"] = 1.0
    out = resample_candles(df, "15m")
    assert len(out) == 2
    assert out["timestamp"].iloc[1] == T0 + pd.Timedelta(minutes=15)
    assert out["high"].iloc[0] == 115 and out["volume"].iloc[0] == 15.0
    assert out["open"].iloc[0] == df["open"].iloc[0] and out["close"].iloc[1] == df["close"].iloc[-1]


def test_resample_4h_is_utc_aligned():
    df = make_candles(600, start=T0 + pd.Timedelta(minutes=90))   # starts 01:30
    out = resample_candles(df, "4h")
    assert out["timestamp"].iloc[0] == T0                          # 00:00 bucket
    assert all(t.hour % 4 == 0 and t.minute == 0 for t in out["timestamp"])


def kline(t_sec, c, q=100):
    return json.dumps({"channel": "push.kline", "symbol": "BTC_USDT",
                       "data": {"t": t_sec, "o": c, "h": c + 1, "l": c - 1, "c": c, "q": q, "a": 0}})


def make_store(tmp_path, n=10):
    p = candle_path(tmp_path, "mexc_futures", "BTCUSDT.P", "1m")
    p.parent.mkdir(parents=True)
    make_candles(n).to_parquet(p, index=False)
    live = LiveMarketState(["BTCUSDT.P"])
    live.connected = True
    store = CandleStore(SpecAdapter(), ["BTCUSDT.P"], tmp_path, live=live)
    store._contract_size["BTCUSDT.P"] = 0.0001
    return store, live


def test_live_candles_extend_history_without_gaps(tmp_path):
    store, live = make_store(tmp_path, n=10)          # stored 00:00..00:09
    stream = MexcFuturesStream(live)
    t10 = int((T0 + pd.Timedelta(minutes=10)).timestamp())
    for t, c in [(t10, 200), (t10, 201), (t10 + 60, 202), (t10 + 120, 203)]:
        stream.handle_message(kline(t, c))

    df = store.base_candles("BTCUSDT.P")
    assert len(df) == 13 and validate_candles(df, "1m").ok
    assert df["close"].iloc[10] == 201          # last update of a closed minute is kept
    assert df["close"].iloc[-1] == 203          # forming candle included
    assert df["volume"].iloc[-1] == pytest.approx(100 * 0.0001)


def test_stored_history_wins_over_live(tmp_path):
    store, live = make_store(tmp_path, n=10)
    t9 = int((T0 + pd.Timedelta(minutes=9)).timestamp())
    MexcFuturesStream(live).handle_message(kline(t9, 999))
    df = store.base_candles("BTCUSDT.P")
    assert len(df) == 10 and df["close"].iloc[-1] == 100.5


def test_old_kline_does_not_replace_newer(tmp_path):
    store, live = make_store(tmp_path)
    s = MexcFuturesStream(live)
    t = int((T0 + pd.Timedelta(minutes=20)).timestamp())
    s.handle_message(kline(t, 300))
    s.handle_message(kline(t - 60, 1))
    assert live.symbols["BTCUSDT.P"].last_candle["close"] == 300


def test_chart_bars_format(tmp_path):
    store, _ = make_store(tmp_path, n=3)
    bars = to_chart_bars(store.candles("BTCUSDT.P", "1m", 10))
    assert bars[0] == {"time": int(T0.timestamp()), "open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.5, "volume": 1.0}


def test_sync_failure_keeps_serving(tmp_path):
    class Broken(SpecAdapter):
        def fetch_candles(self, *a):
            raise ConnectionError("offline")
    store, _ = make_store(tmp_path)
    store.adapter = Broken()
    store.sync("BTCUSDT.P")
    assert "offline" in store.info("BTCUSDT.P")["sync_error"]
    assert len(store.candles("BTCUSDT.P", "1m")) == 10


def test_dashboard_endpoints(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setenv("MM_LIVE_STREAM_ENABLED", "false")
    monkeypatch.setenv("MM_HISTORY_SYNC_ENABLED", "false")
    config.get_settings.cache_clear()
    from app.main import app
    store, live = make_store(tmp_path, n=120)
    with TestClient(app) as c:
        app.state.store, app.state.live = store, live
        r = c.get("/api/dashboard/chart/BTCUSDT.P?interval=30m&limit=50").json()
        assert len(r["bars"]) == 4 and r["feed"]["history"]["stored_candles"] == 120
        assert c.get("/api/dashboard/chart/BTCUSDT.P?interval=15m").status_code == 400   # removed timeframe
        with c.websocket_connect("/ws/live/BTCUSDT.P?interval=30m") as ws:
            msg = ws.receive_json()
            assert len(msg["bars"]) == 2 and "stale" in msg["feed"]
    config.get_settings.cache_clear()


def test_long_history_merges_with_live_candles(tmp_path):
    store, live = make_store(tmp_path, n=120)               # 1m: 00:00 .. 01:59
    hist = resample_candles(make_candles(48 * 30, start=T0 - pd.Timedelta(days=30), step_min=30)
                            .assign(close=50.0), "30m")      # 30 days of 30m history before T0
    store._long[("BTCUSDT.P", "30m")] = hist
    df = store.candles("BTCUSDT.P", "30m", 5000)
    assert len(df) == 48 * 30 + 4                          # history + 4 buckets from 1m data
    assert df["close"].iloc[-1] == 100.5 and df["close"].iloc[0] == 50.0
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing
