"""In-memory candle store for the dashboard.

Combines three sources into one continuous 1m series per symbol:
  1. stored history (Parquet), kept current by a REST sync every few minutes
  2. 1m candles that closed on the live feed since the last sync
  3. the currently forming live candle
REST data always wins over live data for the same minute, because the REST
candle is the exchange's final record.

Any timeframe (5m, 15m, 1h, 4h, ...) is built from the 1m series, so all
timeframes always agree with each other.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.adapters.base import CANDLE_COLUMNS, INTERVAL_SECONDS, MarketDataAdapter
from app.data.historical import candle_path, load_candles, merge_candles, update_history
from app.data.live_stream import LiveMarketState

log = logging.getLogger(__name__)

BASE_INTERVAL = "1m"
# Chart timeframes with long history kept on disk (the 1m store only holds ~30 days)
LONG_HISTORY = {"30m": 400, "4h": 730, "1d": 1095}


def resample_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate 1m candles into a larger UTC-aligned timeframe."""
    if interval == BASE_INTERVAL or df.empty:
        return df.reset_index(drop=True)
    sec = INTERVAL_SECONDS[interval]
    out = (
        df.set_index("timestamp")
        .resample(f"{sec}s", origin="epoch", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last",
              "volume": "sum", "contracts": "sum", "quote_volume": "sum"})
        .dropna(subset=["open"])
        .reset_index()
    )
    return out[CANDLE_COLUMNS]


def live_rows(candles: list[dict], contract_size: float) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["volume"] = df["contracts"] * contract_size
    return df[CANDLE_COLUMNS]


class CandleStore:
    def __init__(
        self,
        adapter: MarketDataAdapter,
        symbols: list[str],
        raw_dir: Path,
        live: LiveMarketState | None = None,
        days: int = 30,
    ):
        self.adapter = adapter
        self.symbols = [s.upper() for s in symbols]
        self.raw_dir = raw_dir
        self.live = live
        self.days = days
        self._lock = threading.Lock()
        self._frames: dict[str, pd.DataFrame] = {}
        self._contract_size: dict[str, float] = {}
        self.last_sync: dict[str, float] = {}
        self.sync_error: dict[str, str | None] = {s: None for s in self.symbols}
        self._long: dict[tuple[str, str], pd.DataFrame] = {}
        for s in self.symbols:
            for iv in LONG_HISTORY:
                self._long[(s, iv)] = load_candles(candle_path(raw_dir, adapter.venue, s, iv))
        for s in self.symbols:  # start from whatever is already on disk
            self._frames[s] = load_candles(candle_path(raw_dir, adapter.venue, s, BASE_INTERVAL))

    # -- syncing -------------------------------------------------------------
    def sync(self, symbol: str) -> None:
        """Blocking: download missing candles via REST and save to disk."""
        try:
            if symbol not in self._contract_size:
                self._contract_size[symbol] = self.adapter.get_contract_spec(symbol).contract_size
            _, df = update_history(self.adapter, symbol, BASE_INTERVAL, self.raw_dir, days=self.days)
            with self._lock:
                self._frames[symbol] = df
            self.last_sync[symbol] = time.time()
            self.sync_error[symbol] = None
        except Exception as exc:  # keep serving what we have
            self.sync_error[symbol] = f"{type(exc).__name__}: {exc}"
            log.warning("History sync failed for %s: %s", symbol, self.sync_error[symbol])

    def sync_long(self, symbol: str) -> None:
        """Blocking: keep 30m / 4h / daily history on disk up to date (for the charts and levels)."""
        for iv, days in LONG_HISTORY.items():
            try:
                _, df = update_history(self.adapter, symbol, iv, self.raw_dir, days=days)
                with self._lock:
                    self._long[(symbol, iv)] = df
            except Exception as exc:
                log.warning("%s %s history sync failed: %s", symbol, iv, exc)

    async def run_sync_loop(self, every_seconds: float = 300) -> None:
        while True:
            for s in self.symbols:
                await asyncio.to_thread(self.sync, s)
                await asyncio.to_thread(self.sync_long, s)
            await asyncio.sleep(every_seconds)

    # -- reading -------------------------------------------------------------
    def base_candles(self, symbol: str) -> pd.DataFrame:
        s = symbol.upper()
        with self._lock:
            stored = self._frames.get(s, pd.DataFrame(columns=CANDLE_COLUMNS))
        if self.live is None or s not in self.live.symbols:
            return stored
        st = self.live.symbols[s]
        candles = list(st.closed_candles)
        if st.last_candle:
            candles.append(st.last_candle)
        size = self._contract_size.get(s)
        if not candles or size is None:
            return stored
        live = live_rows(candles, size)
        if not stored.empty:
            # only live minutes newer than history's start, history wins on overlap
            live = live[live["timestamp"] >= stored["timestamp"].iloc[0]]
        return merge_candles(live, stored)

    def candles(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        df = self.base_candles(symbol)
        if interval != BASE_INTERVAL and not df.empty:
            # only resample the tail we need (+1 bucket for alignment)
            per_bar = INTERVAL_SECONDS[interval] // 60
            df = df.tail((limit + 1) * per_bar)
            first = df["timestamp"].iloc[0]
            df = resample_candles(df, interval)
            # the first bucket may be only partly covered by 1m data: drop it unless it starts exactly
            sec = INTERVAL_SECONDS[interval]
            if (first - pd.Timestamp(0, tz="UTC")).total_seconds() % sec:
                df = df.iloc[1:]
        with self._lock:
            hist = self._long.get((symbol.upper(), interval))
        if hist is not None and not hist.empty:
            df = merge_candles(hist, df) if not df.empty else hist   # live data wins on overlap
        if df.empty:
            return df
        return df.tail(limit).reset_index(drop=True)

    def info(self, symbol: str) -> dict:
        s = symbol.upper()
        with self._lock:
            df = self._frames.get(s)
        last = None if df is None or df.empty else df["timestamp"].iloc[-1].isoformat()
        synced = self.last_sync.get(s)
        return {
            "stored_candles": 0 if df is None else len(df),
            "last_stored_candle": last,
            "last_sync": None if synced is None
            else datetime.fromtimestamp(synced, tz=timezone.utc).isoformat(),
            "sync_error": self.sync_error.get(s),
            "contract_size": self._contract_size.get(s),
        }


def to_chart_bars(df: pd.DataFrame) -> list[dict]:
    """Rows -> compact dicts for the chart: unix-second time + OHLCV (base-asset volume)."""
    if df.empty:
        return []
    t = ((df["timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()).astype("int64")
    return [
        {"time": int(ti), "open": o, "high": h, "low": l, "close": c, "volume": v}
        for ti, o, h, l, c, v in zip(t, df["open"], df["high"], df["low"], df["close"], df["volume"])
    ]
