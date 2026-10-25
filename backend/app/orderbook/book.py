"""Local order book for one MEXC perpetual, kept in sync from
a REST snapshot plus incremental WebSocket updates.

Procedure:
  1. Subscribe to `sub.depth` and buffer the `push.depth` messages.
  2. Fetch a snapshot: GET /api/v1/contract/depth/{symbol}  -> data {asks, bids, version}
  3. Once the buffer reaches back to the snapshot's version, build the book from
     the snapshot and apply every buffered update with a higher version, in order.
  4. Apply each later update whose version is higher than the book's.

MEXC bundles several changes into each push, so versions rise in steps of
several (checked on the live feed 2026-10-22: steps of 2-12). Each level in a
push is the level's new absolute size, so applying pushes in order stays
correct; a skipped version number is not a lost update. To catch any silent
error anyway, the book is rebuilt from a fresh snapshot every couple of
minutes (`refresh`) and the difference is measured.
Levels are [price, contracts, order_count]; contracts == 0 removes the level.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _levels(raw) -> list[tuple[float, float]]:
    out = []
    for lv in raw or []:
        try:
            out.append((float(lv[0]), float(lv[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


class OrderBook:
    """Price -> contracts for each side."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.version: int = 0
        # Deepest snapshot prices: beyond them the book may be incomplete.
        self.snapshot_low_bid: float | None = None
        self.snapshot_high_ask: float | None = None

    def load_snapshot(self, bids, asks, version: int) -> None:
        self.bids = {p: q for p, q in _levels(bids) if q > 0}
        self.asks = {p: q for p, q in _levels(asks) if q > 0}
        self.version = int(version)
        self.snapshot_low_bid = min(self.bids) if self.bids else None
        self.snapshot_high_ask = max(self.asks) if self.asks else None

    def apply(self, bids, asks) -> None:
        for side, levels in ((self.bids, bids), (self.asks, asks)):
            for p, q in _levels(levels):
                if q <= 0:
                    side.pop(p, None)
                else:
                    side[p] = q

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        return None if b is None or a is None else (a + b) / 2

    def is_crossed(self) -> bool:
        b, a = self.best_bid, self.best_ask
        return b is not None and a is not None and b >= a

    def top(self, n: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return bids, asks

    def coverage_pct(self) -> float:
        """How far from mid (as a fraction) the book is known to be complete."""
        mid = self.mid
        if mid is None or self.snapshot_low_bid is None or self.snapshot_high_ask is None:
            return 0.0
        return max(0.0, min(mid - self.snapshot_low_bid, self.snapshot_high_ask - mid) / mid)


def top_level_drift(a: OrderBook, b: OrderBook, n: int = 20) -> float:
    """Share of the top-n price levels (both sides) where two books disagree."""
    diffs = total = 0
    for side_a, side_b, rev in ((a.bids, b.bids, True), (a.asks, b.asks, False)):
        prices = sorted(set(side_a) | set(side_b), reverse=rev)[:n]
        for p in prices:
            total += 1
            if abs(side_a.get(p, 0.0) - side_b.get(p, 0.0)) > 1e-9:
                diffs += 1
    return diffs / total if total else 0.0


@dataclass
class BookSync:
    """Version bookkeeping around an OrderBook. Pure logic, no I/O:
    the caller fetches snapshots when `needs_snapshot` is True, and
    periodically for `refresh`."""

    symbol: str
    book: OrderBook = field(default_factory=OrderBook)
    synced: bool = False
    buffer: list = field(default_factory=list)          # updates waiting for a snapshot
    recent: deque = field(default_factory=lambda: deque(maxlen=3000))  # applied updates, for refresh
    snapshot_pending: bool = False
    resyncs: int = 0
    refreshes: int = 0
    last_drift: float | None = None
    last_resync_reason: str | None = None
    last_update: float | None = None
    max_buffer: int = 5000

    @property
    def needs_snapshot(self) -> bool:
        return not self.synced and not self.snapshot_pending and bool(self.buffer)

    def reset(self, reason: str) -> None:
        if self.synced:
            self.resyncs += 1
            log.info("%s order book resync: %s", self.symbol, reason)
        self.last_resync_reason = reason
        self.synced = False
        self.buffer.clear()
        self.recent.clear()
        self.book = OrderBook()

    def on_update(self, data: dict) -> None:
        try:
            v = int(data["version"])
        except (KeyError, TypeError, ValueError):
            return
        upd = (v, data.get("bids", []), data.get("asks", []))
        if not self.synced:
            self.buffer.append(upd)
            if len(self.buffer) > self.max_buffer:
                self.buffer = self.buffer[-self.max_buffer:]
            return
        if v <= self.book.version:
            return  # duplicate or out of date
        self.book.apply(upd[1], upd[2])
        self.book.version = v
        self.recent.append(upd)
        self.last_update = time.monotonic()
        if self.book.is_crossed():
            self.reset("crossed book (best bid >= best ask)")

    def _build(self, bids, asks, version: int, updates) -> OrderBook | None:
        """Snapshot + every later update, or None if the updates don't reach back
        to the snapshot (then changes between them could be missing)."""
        ups = sorted(updates, key=lambda u: u[0])
        if not ups or ups[0][0] > version:
            return None
        book = OrderBook()
        book.load_snapshot(bids, asks, version)
        for v, b, a in ups:
            if v > book.version:
                book.apply(b, a)
                book.version = v
        return book

    def on_snapshot(self, bids, asks, version: int) -> bool:
        """Load a snapshot and replay buffered updates. Returns True if synced."""
        self.snapshot_pending = False
        book = self._build(bids, asks, version, self.buffer)
        if book is None:
            first = min((u[0] for u in self.buffer), default=None)
            self.last_resync_reason = f"snapshot {version} is older than the first buffered update ({first})"
            return False
        if book.is_crossed():
            self.last_resync_reason = "crossed book after snapshot"
            return False
        self.book = book
        self.recent = deque(sorted(self.buffer, key=lambda u: u[0]), maxlen=self.recent.maxlen)
        self.synced = True
        self.buffer.clear()
        self.last_update = time.monotonic()
        return True

    def refresh(self, bids, asks, version: int) -> float | None:
        """Rebuild from a fresh snapshot while staying live. Returns the drift
        (share of top-20 levels that differed) or None if it couldn't be used."""
        self.snapshot_pending = False
        if not self.synced:
            return None
        book = self._build(bids, asks, version, list(self.recent))
        if book is None or book.version != self.book.version or book.is_crossed():
            return None
        drift = top_level_drift(book, self.book)
        self.book = book
        self.refreshes += 1
        self.last_drift = drift
        if drift > 0.05:
            log.warning("%s order book drift %.0f%% corrected by refresh", self.symbol, drift * 100)
        return drift
