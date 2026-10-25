"""Live MEXC futures WebSocket stream: candles + executed trades, with
reconnects, heartbeats and stale-data detection.

Protocol (MEXC contract WebSocket, wss://contract.mexc.com/edge):
  subscribe:   {"method": "sub.kline", "param": {"symbol": "BTC_USDT", "interval": "Min1"}}
               {"method": "sub.deal",  "param": {"symbol": "BTC_USDT"}}
  heartbeat:   {"method": "ping"}  -> {"channel": "pong", ...}   (server drops idle clients ~60 s)
  pushes:      channel "push.kline": data {t (open time, s), o, h, l, c, q (contracts), a (USDT), interval}
               channel "push.deal":  data {p (price), v (contracts), T (1 = taker buy, 2 = taker sell), t (ms)}
                                     (data may be a single object or a list)

The parsers below are deliberately tolerant; if MEXC changes a field the message
is logged and skipped rather than crashing the stream.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import httpx
import websockets

from app.adapters.base import Trade
from app.adapters.mexc_futures import MEXC_INTERVALS, to_venue_symbol
from app.orderbook.book import BookSync

log = logging.getLogger(__name__)

Clock = Callable[[], float]


# ---------------------------------------------------------------- parsing --
def parse_kline_push(msg: dict) -> dict | None:
    d = msg.get("data")
    if not isinstance(d, dict):
        return None
    try:
        return {
            "venue_symbol": msg.get("symbol") or d.get("symbol"),
            "interval": d.get("interval"),
            "timestamp": datetime.fromtimestamp(int(d["t"]), tz=timezone.utc),
            "open": float(d["o"]),
            "high": float(d["h"]),
            "low": float(d["l"]),
            "close": float(d["c"]),
            "contracts": float(d.get("q", 0.0)),
            "quote_volume": float(d.get("a", 0.0)),
        }
    except (KeyError, TypeError, ValueError):
        return None


def parse_deal_push(msg: dict, symbol: str) -> list[Trade]:
    d = msg.get("data")
    items = d if isinstance(d, list) else [d] if isinstance(d, dict) else []
    trades = []
    for it in items:
        try:
            side = {1: "buy", 2: "sell"}[int(it["T"])]
            trades.append(
                Trade(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(it["t"]) / 1000, tz=timezone.utc),
                    price=float(it["p"]),
                    contracts=float(it["v"]),
                    side=side,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return trades


# ------------------------------------------------------------------ state --
@dataclass
class SymbolState:
    symbol: str
    last_candle: dict | None = None
    # 1m candles that closed while streaming (filled until the next REST sync)
    closed_candles: deque = field(default_factory=lambda: deque(maxlen=1440))
    trades: deque = field(default_factory=lambda: deque(maxlen=2000))
    last_message_at: float | None = None   # monotonic clock
    messages: int = 0
    book: BookSync | None = None

    def __post_init__(self):
        if self.book is None:
            self.book = BookSync(self.symbol)


class LiveMarketState:
    """Latest live data per symbol plus feed-health information."""

    def __init__(self, symbols: list[str], stale_after: float = 10.0, clock: Clock = time.monotonic):
        self.clock = clock
        self.stale_after = stale_after
        self.symbols = {s.upper(): SymbolState(s.upper()) for s in symbols}
        self.connected = False
        self.reconnects = 0
        self.last_error: str | None = None
        self.connected_since: float | None = None

    def touch(self, symbol: str) -> SymbolState:
        st = self.symbols[symbol]
        st.last_message_at = self.clock()
        st.messages += 1
        return st

    def is_stale(self, symbol: str) -> bool:
        st = self.symbols[symbol.upper()]
        if not self.connected or st.last_message_at is None:
            return True
        return self.clock() - st.last_message_at > self.stale_after

    def trade_flow(self, symbol: str, window_seconds: float = 60.0) -> dict:
        """Aggressive buy vs sell volume (contracts) over the last window."""
        st = self.symbols[symbol.upper()]
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        buy = sell = 0.0
        n = 0
        for t in reversed(st.trades):
            if t.timestamp.timestamp() < cutoff:
                break
            n += 1
            if t.side == "buy":
                buy += t.contracts
            else:
                sell += t.contracts
        total = buy + sell
        return {
            "window_seconds": window_seconds,
            "trades": n,
            "buy_contracts": buy,
            "sell_contracts": sell,
            "delta_contracts": buy - sell,
            "buy_ratio": (buy / total) if total else None,
        }

    def snapshot(self, symbol: str) -> dict:
        st = self.symbols[symbol.upper()]
        age = None if st.last_message_at is None else round(self.clock() - st.last_message_at, 2)
        last_trade = st.trades[-1] if st.trades else None
        candle = dict(st.last_candle) if st.last_candle else None
        if candle:
            candle["timestamp"] = candle["timestamp"].isoformat()
        return {
            "symbol": st.symbol,
            "connected": self.connected,
            "stale": self.is_stale(symbol),
            "seconds_since_update": age,
            "last_price": last_trade.price if last_trade else (candle or {}).get("close"),
            "live_candle": candle,
            "trade_flow_60s": self.trade_flow(symbol, 60),
        }


# ----------------------------------------------------------------- stream --
class MexcFuturesStream:
    def __init__(
        self,
        state: LiveMarketState,
        url: str = "wss://contract.mexc.com/edge",
        interval: str = "1m",
        ping_seconds: float = 15.0,
        reconnect_max_seconds: float = 60.0,
        rest_url: str = "https://contract.mexc.com",
        depth_enabled: bool = True,
        snapshot_limit: int = 1000,
        refresh_seconds: float = 120.0,
    ):
        self.state = state
        self.url = url
        self.interval = interval
        self.ping_seconds = ping_seconds
        self.reconnect_max = reconnect_max_seconds
        self._venue_to_symbol = {to_venue_symbol(s): s for s in state.symbols}
        self._stop = asyncio.Event()
        self.rest_url = rest_url
        self.depth_enabled = depth_enabled
        self.snapshot_limit = snapshot_limit
        self.refresh_seconds = refresh_seconds
        self._last_snapshot_request: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

    def subscription_messages(self) -> list[dict]:
        msgs = []
        for vs in self._venue_to_symbol:
            msgs.append({"method": "sub.kline", "param": {"symbol": vs, "interval": MEXC_INTERVALS[self.interval]}})
            msgs.append({"method": "sub.deal", "param": {"symbol": vs}})
            if self.depth_enabled:
                msgs.append({"method": "sub.depth", "param": {"symbol": vs}})
        return msgs

    def handle_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.debug("non-JSON message ignored")
            return
        channel = msg.get("channel", "")
        if channel == "pong" or channel.startswith("rs.sub"):
            return
        if channel == "rs.error":
            log.warning("MEXC stream error: %s", msg.get("data"))
            self.state.last_error = str(msg.get("data"))
            return

        vs = msg.get("symbol")
        if not vs and isinstance(msg.get("data"), dict):
            vs = msg["data"].get("symbol")
        symbol = self._venue_to_symbol.get(vs or "")
        if symbol is None:
            return

        if channel == "push.kline":
            candle = parse_kline_push(msg)
            if candle:
                st = self.state.touch(symbol)
                prev = st.last_candle
                if prev and candle["timestamp"] > prev["timestamp"]:
                    st.closed_candles.append(prev)  # previous minute has closed
                if prev is None or candle["timestamp"] >= prev["timestamp"]:
                    st.last_candle = candle
        elif channel == "push.deal":
            trades = parse_deal_push(msg, symbol)
            if trades:
                self.state.touch(symbol).trades.extend(trades)
        elif channel == "push.depth" and self.depth_enabled:
            data = msg.get("data")
            if isinstance(data, dict):
                st = self.state.touch(symbol)
                st.book.on_update(data)
                if st.book.needs_snapshot:
                    self._schedule_snapshot(symbol)

    # -- order book snapshots ---------------------------------------------------
    def _schedule_snapshot(self, symbol: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # called outside an event loop (unit tests)
        book = self.state.symbols[symbol].book
        book.snapshot_pending = True
        task = loop.create_task(self._fetch_snapshot(symbol))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _fetch_snapshot(self, symbol: str) -> None:
        book = self.state.symbols[symbol].book
        # MEXC allows 20 depth requests / 2 s; keep well under that.
        wait = 1.0 - (time.monotonic() - self._last_snapshot_request.get(symbol, 0.0))
        if wait > 0:
            await asyncio.sleep(wait)
        await asyncio.sleep(0.3)  # let a few updates buffer so the snapshot overlaps them
        self._last_snapshot_request[symbol] = time.monotonic()
        try:
            async with httpx.AsyncClient(base_url=self.rest_url, timeout=20) as client:
                r = await client.get(f"/api/v1/contract/depth/{to_venue_symbol(symbol)}",
                                     params={"limit": self.snapshot_limit})
                r.raise_for_status()
                payload = r.json()
            data = payload.get("data") or {}
            if not payload.get("success", False) or "version" not in data:
                raise ValueError(f"bad depth snapshot: {str(payload)[:200]}")
            if book.synced:  # periodic self-check: rebuild from the snapshot while staying live
                drift = book.refresh(data.get("bids", []), data.get("asks", []), int(data["version"]))
                log.debug("%s order book refresh: drift %s", symbol, drift)
                return
            ok = book.on_snapshot(data.get("bids", []), data.get("asks", []), int(data["version"]))
            if ok:
                log.info("%s order book synced at version %s (%d bids, %d asks)", symbol,
                         book.book.version, len(book.book.bids), len(book.book.asks))
            else:
                log.info("%s order book snapshot did not line up (%s); retrying", symbol, book.last_resync_reason)
        except Exception as exc:
            book.snapshot_pending = False
            book.last_resync_reason = f"snapshot failed: {type(exc).__name__}: {exc}"
            log.warning("%s depth snapshot failed: %s", symbol, exc)
            await asyncio.sleep(5)  # back off before retrying
        # Still not synced: try again (next update will also trigger this)
        if not book.synced and book.needs_snapshot:
            self._schedule_snapshot(symbol)

    async def _refresher(self) -> None:
        """Every couple of minutes, re-check each synced book against a fresh snapshot."""
        while True:
            await asyncio.sleep(self.refresh_seconds)
            for sym, st in self.state.symbols.items():
                if st.book.synced and not st.book.snapshot_pending:
                    self._schedule_snapshot(sym)
                    await asyncio.sleep(1.0)

    async def _pinger(self, ws) -> None:
        while True:
            await asyncio.sleep(self.ping_seconds)
            await ws.send(json.dumps({"method": "ping"}))

    async def _run_once(self) -> None:
        async with websockets.connect(self.url, ping_interval=None, open_timeout=15, max_size=2**22) as ws:
            for st in self.state.symbols.values():
                st.book.reset("new connection")
            for m in self.subscription_messages():
                await ws.send(json.dumps(m))
            self.state.connected = True
            self.state.connected_since = self.state.clock()
            log.info("Connected to %s (%d symbols)", self.url, len(self._venue_to_symbol))
            pinger = asyncio.create_task(self._pinger(ws))
            refresher = asyncio.create_task(self._refresher()) if self.depth_enabled else None
            try:
                while not self._stop.is_set():
                    # If nothing at all arrives for 3 ping periods the socket is dead.
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.ping_seconds * 3)
                    self.handle_message(raw)
            finally:
                pinger.cancel()
                if refresher:
                    refresher.cancel()

    async def run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors, timeouts, closed sockets
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Stream disconnected: %s", self.state.last_error)
            finally:
                self.state.connected = False
            if self._stop.is_set():
                break
            if time.monotonic() - started > 60:
                delay = 1.0  # connection was healthy for a while: reset backoff
            self.state.reconnects += 1
            sleep_for = min(delay, self.reconnect_max) * (0.8 + 0.4 * random.random())
            log.info("Reconnecting in %.1fs", sleep_for)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass
            delay *= 2

    def stop(self) -> None:
        self._stop.set()
