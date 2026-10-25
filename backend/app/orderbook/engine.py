"""Samples every order book once a second: measurements, rolling
imbalance, large-order tracking, alerts and recording.

The live WebSocket keeps the books current; this engine only reads them.
Nothing is measured or alerted while a book is out of sync or the feed is stale.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from app.data.live_stream import LiveMarketState
from app.orderbook.alerts import AlertEngine
from app.orderflow.flow import OrderFlow
from app.orderbook.features import BANDS, RollingMean, book_metrics, ladder
from app.orderbook.large_orders import LargeOrderTracker
from app.orderbook.recorder import OrderBookRecorder
from app.signals.rows import build_row

log = logging.getLogger(__name__)
IMBALANCE_BAND = 0.0025   # 0.25% either side of mid


def fmt_price(p: float) -> str:
    return f"{p:,.1f}" if p >= 1000 else f"{p:,.2f}"


class OrderBookEngine:
    def __init__(self, state: LiveMarketState, contract_size: Callable[[str], float | None],
                 recorder: OrderBookRecorder | None = None, clock: Callable[[], float] = time.monotonic,
                 on_row: Callable[[str, dict], None] | None = None,
                 on_tick: Callable | None = None):
        self.state = state
        self.contract_size = contract_size
        self.recorder = recorder
        self.clock = clock
        self.on_row = on_row          # signal engine: one row per symbol per second
        self.on_tick = on_tick        # S/R analyzer: (symbol, time, book or None, trades, contract size)
        syms = list(state.symbols)
        self.trackers = {s: LargeOrderTracker() for s in syms}
        self.rolling = {s: RollingMean() for s in syms}
        self.alerts = {s: AlertEngine(s, s.replace("USDT.P", "").replace("USDT", ""), fmt_price) for s in syms}
        self.flow = {s: OrderFlow(s.replace("USDT.P", "").replace("USDT", "")) for s in syms}
        self._last_trade_ts = {s: time.time() for s in syms}
        self.views: dict[str, dict] = {s: {"status": "waiting"} for s in syms}
        self.band_index = BANDS.index(IMBALANCE_BAND)

    def _new_trades(self, symbol: str):
        st = self.state.symbols[symbol]
        last = self._last_trade_ts[symbol]
        out = [t for t in st.trades if t.timestamp.timestamp() > last]
        if out:
            self._last_trade_ts[symbol] = max(t.timestamp.timestamp() for t in out)
        return out

    def step(self, symbol: str) -> dict:
        now, wall = self.clock(), time.time()
        st = self.state.symbols[symbol]
        bs = st.book
        health = {
            "synced": bs.synced,
            "version": bs.book.version if bs.synced else None,
            "resyncs": bs.resyncs,
            "refreshes": bs.refreshes,
            "last_drift": bs.last_drift,
            "last_resync_reason": bs.last_resync_reason,
            "recording": self.recorder is not None,
            "recorded_files": self.recorder.files_written if self.recorder else 0,
        }
        view = {"health": health, "alerts": list(self.alerts[symbol].recent)}
        trades = self._new_trades(symbol)
        cs = self.contract_size(symbol)
        if cs is not None:   # market orders don't depend on the order book being synced
            self.flow[symbol].add(trades, cs)
            view["flow"] = self.flow[symbol].view()
        if not bs.synced or self.state.is_stale(symbol) or cs is None:
            view["status"] = "stale" if bs.synced and self.state.is_stale(symbol) else "syncing"
            # Reset the large-order history: we can't tell what happened while blind.
            self.trackers[symbol] = LargeOrderTracker()
            self.views[symbol] = view
            self._row(symbol, build_row(wall, False, None, trades, cs, [], now))
            self._tick(symbol, wall, None, trades, cs, health)
            return view

        book = bs.book
        m = book_metrics(book, cs)
        roll = self.rolling[symbol]
        band = m["bands"][self.band_index]
        roll.add(now, band["imbalance"])
        imb = {
            "band_pct": IMBALANCE_BAND,
            "now": band["imbalance"],
            "avg_5s": roll.mean(5, now),
            "avg_15s": roll.mean(15, now),
            "avg_60s": roll.mean(60, now),
            "coverage_60s": roll.coverage(60, now),
        }
        tracker = self.trackers[symbol]
        ended = tracker.update(now, book, trades)
        flow = self.state.trade_flow(symbol, 60)
        self.alerts[symbol].evaluate(now, m["mid"], cs, list(tracker.active.values()), ended,
                                     imb["avg_15s"], imb["avg_60s"], imb["coverage_60s"], flow["buy_ratio"],
                                     threshold=tracker.threshold)
        if self.recorder:
            self.recorder.add(symbol, wall, book, m, cs)
        self._row(symbol, build_row(wall, True, m, trades, cs, list(tracker.active.values()), now))
        self._tick(symbol, wall, book, trades, cs, health)

        view.update({
            "status": "live",
            "metrics": m,
            "imbalance": imb,
            "ladder": ladder(book, cs),
            "large_orders": tracker.view(now, m["mid"], cs),
            "alerts": list(self.alerts[symbol].recent),
            "flow": view.get("flow"),
        })
        self.views[symbol] = view
        return view

    def _row(self, symbol: str, row: dict) -> None:
        if self.on_row is None:
            return
        try:
            self.on_row(symbol, row)
        except Exception:
            log.exception("signal engine failed for %s", symbol)

    def _tick(self, symbol: str, t: float, book, trades: list, cs, health: dict | None = None) -> None:
        if self.on_tick is None:
            return
        try:
            self.on_tick(symbol, t, book, trades, cs, health)
        except Exception:
            log.exception("S/R analyzer failed for %s", symbol)

    async def run(self, every: float = 1.0) -> None:
        try:
            while True:
                for s in self.state.symbols:
                    try:
                        self.step(s)
                    except Exception:
                        log.exception("Order-book engine step failed for %s", s)
                if self.recorder:
                    for s in self.recorder.due(time.time()):
                        await asyncio.to_thread(self.recorder.flush, s)
                await asyncio.sleep(every)
        finally:
            if self.recorder:
                self.recorder.flush_all()
