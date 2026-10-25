from datetime import datetime, timedelta, timezone

import pandas as pd

from app.adapters.base import CANDLE_COLUMNS, MarketDataAdapter
from app.data.historical import last_closed_candle_end, load_candles, merge_candles, update_history
from app.data.validation import validate_candles

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def make_candles(n=10, start=T0, step_min=1):
    ts = [start + pd.Timedelta(minutes=i * step_min) for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1.0, "contracts": 10.0, "quote_volume": 100.0,
    })[CANDLE_COLUMNS]


def test_clean_data_passes():
    r = validate_candles(make_candles(), "1m")
    assert r.ok and r.missing_candles == 0 and r.rows == 10


def test_detects_gap():
    df = make_candles(10).drop(index=[3, 4, 5])
    r = validate_candles(df, "1m")
    assert not r.ok
    assert r.missing_candles == 3
    assert r.gaps[0]["missing"] == 3


def test_detects_duplicates_unsorted_misaligned():
    df = make_candles(5)
    df = pd.concat([df, df.iloc[[2]]])                      # duplicate
    df = df.iloc[::-1]                                      # unsorted
    df.iloc[1, 0] = df.iloc[1, 0] + pd.Timedelta(seconds=7)  # misaligned
    r = validate_candles(df, "1m")
    assert r.duplicate_timestamps == 1 and r.unsorted and r.misaligned_timestamps == 1


def test_detects_bad_ohlc_and_prices():
    df = make_candles(5)
    df.loc[1, "high"] = 98.0   # high below open/close
    df.loc[2, "low"] = -1.0    # non-positive
    df.loc[3, "volume"] = -5   # negative volume
    r = validate_candles(df, "1m")
    assert r.bad_ohlc >= 1 and r.non_positive_prices == 1 and r.negative_volume == 1


def test_merge_newest_wins():
    a = make_candles(5)
    b = make_candles(3, start=T0 + pd.Timedelta(minutes=3))
    b["close"] = 200.0
    m = merge_candles(a, b)
    assert len(m) == 6
    assert m.loc[m.timestamp == T0 + pd.Timedelta(minutes=3), "close"].item() == 200.0


def test_last_closed_candle_excludes_forming_candle():
    now = datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
    assert last_closed_candle_end("1m", now) == datetime(2026, 1, 1, 12, 34, tzinfo=timezone.utc)
    assert last_closed_candle_end("1h", now) == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeAdapter(MarketDataAdapter):
    venue = "fake"

    def __init__(self):
        self.requests = []

    def to_venue_symbol(self, s): return s
    def get_contract_spec(self, s): raise NotImplementedError

    def fetch_candles(self, symbol, interval, start, end):
        self.requests.append((start, end))
        n = int((end - start).total_seconds() // 60)
        return make_candles(n, start=pd.Timestamp(start))


def test_update_history_only_downloads_missing_ranges(tmp_path):
    ad = FakeAdapter()
    now = datetime(2026, 1, 2, 0, 0, 30, tzinfo=timezone.utc)
    path, df = update_history(ad, "BTCUSDT.P", "1m", tmp_path, days=1, now=now)
    assert len(df) == 1440 and validate_candles(df, "1m").ok
    assert len(ad.requests) == 1

    # 10 minutes later: only the new 10 candles are fetched
    path, df = update_history(ad, "BTCUSDT.P", "1m", tmp_path, days=1, now=now + timedelta(minutes=10))
    assert ad.requests[-1][1] - ad.requests[-1][0] == timedelta(minutes=10)
    assert len(load_candles(path)) == 1450
